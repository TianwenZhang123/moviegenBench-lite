#!/usr/bin/env python3
"""Expand 7.16-best to the complete 52-group/312-shot audited dataset."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from build_716_best import (
    AUTODL,
    GROUP_LABELS,
    GROUP_PRIORITY,
    METRICS,
    SHORT,
    TARGET,
    Bundle,
    composite,
    copy_file,
    ensure_sources,
    load_metrics,
    metric_values,
)


def fvd_value(directory: Path) -> float | None:
    path = directory / "fvd.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("fvd")
    return float(value) if value is not None else None


def main() -> None:
    mapping_path = SHORT / "mapping.tsv"
    with mapping_path.open(encoding="utf-8") as handle:
        mapping_rows = list(csv.DictReader(handle, delimiter="\t"))
    by_id = {row["video_id"]: row for row in mapping_rows}
    by_shot = {
        (Path(row["reference_path"]).parent.name, Path(row["reference_path"]).name): row
        for row in mapping_rows
    }
    if len(mapping_rows) != 252 or len(by_id) != 252 or len(by_shot) != 252:
        raise ValueError("Expected 252 unique a/c/l mappings")

    g1_base = load_metrics(SHORT / "results_baseline1")
    g1_c = load_metrics(SHORT / "results_c-v")
    g1_r = load_metrics(SHORT / "results_r-v")
    main_ids = set(by_id)
    for name, metrics in [("baseline1", g1_base), ("c-v", g1_c), ("r-v", g1_r)]:
        if set(metrics["vclip"]) != main_ids:
            raise ValueError(f"Group 1 {name} IDs do not match main mapping")

    g2_c = load_metrics(AUTODL / "14B/eval/7.15/results_7.15")
    g2_ids = set(g2_c["vclip"])
    if len(g2_ids) != 81 or not g2_ids <= main_ids:
        raise ValueError("Group 2 must contain 81 mapped IDs")

    replacement_list = AUTODL / "new-baseline/o-video/o_replacement_backup/shots"
    selected_shots = {(p.parent.name, p.name) for p in replacement_list.glob("*/*.mp4")}
    if len(selected_shots) != 32 or not selected_shots <= set(by_shot):
        raise ValueError("Group 3 evaluated subset must contain 32 mapped shots")
    g3_ids = {f"{batch}__{Path(shot).stem}" for batch, shot in selected_shots}
    g3 = {
        name: load_metrics(AUTODL / f"new-baseline/eval/shots/{name}")
        for name in ("baseline1", "c-v", "r-v")
    }
    for name, metrics in g3.items():
        if set(metrics["vclip"]) != g3_ids:
            raise ValueError(f"Group 3 {name} IDs do not match evaluated subset")

    bundles_by_id: dict[str, dict[str, Bundle]] = {}
    for video_id, row in by_id.items():
        index = int(row["index"])
        reference = Path(row["reference_path"])
        batch, shot = reference.parent.name, reference.name
        canonical = f"{batch}/{Path(shot).stem}"
        common = {
            "canonical_id": canonical,
            "index": index,
            "eval_id": video_id,
            "reference": reference,
            "baseline": AUTODL / f"baseline1/generated/shots/{batch}/{shot}",
            "baseline_metrics": g1_base,
        }
        bundles = {
            "group1": Bundle(
                group="group1",
                methods={
                    "c-v": AUTODL / f"14B/v/c-v/{index}.mp4",
                    "r-v": AUTODL / f"14B/v/r-v/{index}.mp4",
                },
                method_metrics={"c-v": g1_c, "r-v": g1_r},
                evaluation_roots={
                    "baseline1": SHORT / "results_baseline1",
                    "c-v": SHORT / "results_c-v",
                    "r-v": SHORT / "results_r-v",
                },
                **common,
            )
        }
        if video_id in g2_ids:
            bundles["group2"] = Bundle(
                group="group2",
                methods={"c-v": AUTODL / f"14B/v/7.15all-caption-video/{index}.mp4"},
                method_metrics={"c-v": g2_c},
                evaluation_roots={
                    "baseline1": SHORT / "results_baseline1",
                    "c-v": AUTODL / "14B/eval/7.15/results_7.15",
                },
                **common,
            )
        if (batch, shot) in selected_shots:
            g3_id = f"{batch}__{Path(shot).stem}"
            bundles["group3"] = Bundle(
                group="group3",
                canonical_id=canonical,
                index=index,
                eval_id=g3_id,
                reference=AUTODL / f"new-baseline/o-video/shots/{batch}/{shot}",
                baseline=AUTODL / f"new-baseline/baseline1/shots/{batch}/{shot}",
                methods={
                    "c-v": AUTODL / f"new-baseline/methods/c-v/{batch}/{shot}",
                    "r-v": AUTODL / f"new-baseline/methods/r-v/{batch}/{shot}",
                },
                baseline_metrics=g3["baseline1"],
                method_metrics={"c-v": g3["c-v"], "r-v": g3["r-v"]},
                evaluation_roots={
                    "baseline1": AUTODL / "new-baseline/eval/shots/baseline1",
                    "c-v": AUTODL / "new-baseline/eval/shots/c-v",
                    "r-v": AUTODL / "new-baseline/eval/shots/r-v",
                },
            )
        for bundle in bundles.values():
            ensure_sources(bundle)
        bundles_by_id[video_id] = bundles

    # Pick a winning source bundle where one exists; otherwise retain group 1 in full.
    chosen_main: list[dict[str, object]] = []
    candidate_counts: Counter[str] = Counter()
    for video_id in sorted(main_ids, key=lambda value: int(by_id[value]["index"])):
        bundles = bundles_by_id[video_id]
        candidates = []
        qualified_by_group: dict[str, dict[str, float]] = {}
        for group, bundle in bundles.items():
            baseline_score = composite(bundle.baseline_metrics, bundle.eval_id)
            qualified = {
                method: composite(metrics, bundle.eval_id)
                for method, metrics in bundle.method_metrics.items()
                if composite(metrics, bundle.eval_id) > baseline_score
            }
            qualified_by_group[group] = qualified
            for method in qualified:
                candidate_counts[f"{group}:{method}"] += 1
            if qualified:
                best_score = max(qualified.values())
                candidates.append(
                    (
                        best_score - baseline_score,
                        best_score,
                        GROUP_PRIORITY[group],
                        group,
                        baseline_score,
                    )
                )
        if candidates:
            _, _, _, chosen_group, baseline_score = max(candidates)
            has_winner = True
        else:
            chosen_group = "group1"
            baseline_score = composite(bundles["group1"].baseline_metrics, video_id)
            has_winner = False
        chosen_main.append(
            {
                "video_id": video_id,
                "bundle": bundles[chosen_group],
                "group1": bundles["group1"],
                "qualified": qualified_by_group[chosen_group],
                "baseline_score": baseline_score,
                "has_winner": has_winner,
            }
        )

    # Validate the unevaluated d-series before touching the existing output.
    d_samples = []
    for number in range(1, 11):
        batch = f"d{number:03d}"
        for shot_number in range(1, 7):
            shot = f"shot_{shot_number:02d}.mp4"
            paths = {
                "reference": AUTODL / f"new-baseline/o-video/shots/{batch}/{shot}",
                "baseline": AUTODL / f"new-baseline/baseline1/shots/{batch}/{shot}",
                "c-v": AUTODL / f"new-baseline/methods/c-v/{batch}/{shot}",
                "r-v": AUTODL / f"new-baseline/methods/r-v/{batch}/{shot}",
            }
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"Missing d-series source: {path}")
            d_samples.append((batch, shot, paths))
    if len(d_samples) != 60:
        raise ValueError("Expected 60 d-series shots")

    # All preflight checks passed. Replace only the generated target.
    if TARGET.exists():
        shutil.rmtree(TARGET)
    for directory in [
        TARGET / "o-video/shots",
        TARGET / "baseline1/shots",
        TARGET / "methods/c-v",
        TARGET / "methods/r-v",
        TARGET / "eval/shots/baseline1",
        TARGET / "eval/shots/c-v",
        TARGET / "eval/shots/r-v",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, object]] = []
    manifest_samples: list[dict[str, object]] = []
    derived = {
        dataset: {metric: [] for metric in METRICS}
        for dataset in ("baseline1", "c-v", "r-v")
    }
    bundle_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    def add_derived(
        dataset: str,
        canonical_id: str,
        bundle_group: str,
        source_group: str,
        reference_output: Path,
        generated_output: Path,
        values: dict[str, float] | None,
        evaluation_source: Path | None,
    ) -> None:
        batch, shot_stem = canonical_id.split("/", 1)
        for metric, (_, key) in METRICS.items():
            derived[dataset][metric].append(
                {
                    "video_id": f"{batch}__{shot_stem}",
                    "canonical_id": canonical_id,
                    "bundle_group": bundle_group,
                    "source_group": source_group,
                    "evaluation_status": "evaluated" if values is not None else "not_evaluated",
                    "evaluation_source": str(evaluation_source) if evaluation_source else None,
                    "reference_path": str(reference_output),
                    "generated_path": str(generated_output),
                    key: values[metric] if values is not None else None,
                }
            )

    for entry in chosen_main:
        bundle: Bundle = entry["bundle"]
        group1: Bundle = entry["group1"]
        qualified: dict[str, float] = entry["qualified"]
        baseline_score = float(entry["baseline_score"])
        has_winner = bool(entry["has_winner"])
        bundle_counts[bundle.group] += 1
        batch, shot_stem = bundle.canonical_id.split("/", 1)
        shot = f"{shot_stem}.mp4"
        ref_out = TARGET / f"o-video/shots/{batch}/{shot}"
        base_out = TARGET / f"baseline1/shots/{batch}/{shot}"
        copy_file(bundle.reference, ref_out)
        copy_file(bundle.baseline, base_out)
        base_values = metric_values(bundle.baseline_metrics, bundle.eval_id)
        add_derived(
            "baseline1",
            bundle.canonical_id,
            bundle.group,
            bundle.group,
            ref_out,
            base_out,
            base_values,
            bundle.evaluation_roots["baseline1"],
        )

        # Group 2 has no r-v; use group 1 r-v, which shares the same reference/baseline.
        method_bundles = {
            "c-v": bundle,
            "r-v": group1 if bundle.group == "group2" else bundle,
        }
        sample_methods = []
        for method in ("c-v", "r-v"):
            method_bundle = method_bundles[method]
            source = method_bundle.methods[method]
            method_out = TARGET / f"methods/{method}/{batch}/{shot}"
            copy_file(source, method_out)
            values = metric_values(method_bundle.method_metrics[method], method_bundle.eval_id)
            method_score = composite(method_bundle.method_metrics[method], method_bundle.eval_id)
            is_better = method_score > baseline_score
            if not has_winner:
                status = "group1_default_no_method_better"
            elif bundle.group == "group2" and method == "r-v":
                status = "group1_fallback_method2_absent"
            elif method in qualified:
                status = "evaluated_winner"
            else:
                status = "same_bundle_companion"
            status_counts[status] += 1
            eval_root = method_bundle.evaluation_roots[method]
            add_derived(
                method,
                bundle.canonical_id,
                bundle.group,
                method_bundle.group,
                ref_out,
                method_out,
                values,
                eval_root,
            )
            row = {
                "canonical_id": bundle.canonical_id,
                "index": bundle.index,
                "bundle_group": bundle.group,
                "bundle_group_label": GROUP_LABELS[bundle.group],
                "method": method,
                "method_source_group": method_bundle.group,
                "selection_status": status,
                "is_better_than_baseline": is_better,
                "baseline_score": baseline_score,
                "method_score": method_score,
                "improvement": method_score - baseline_score,
                "baseline_clip": base_values["clip"],
                "method_clip": values["clip"],
                "baseline_vclip": base_values["vclip"],
                "method_vclip": values["vclip"],
                "baseline_lpips": base_values["lpips"],
                "method_lpips": values["lpips"],
                "baseline_flow_epe": base_values["flow"],
                "method_flow_epe": values["flow"],
                "baseline_ssim": base_values["ssim"],
                "method_ssim": values["ssim"],
                "reference_source": str(bundle.reference),
                "baseline_source": str(bundle.baseline),
                "method_source": str(source),
                "baseline_evaluation_source": str(bundle.evaluation_roots["baseline1"]),
                "method_evaluation_source": str(eval_root),
                "reference_output": str(ref_out),
                "baseline_output": str(base_out),
                "method_output": str(method_out),
            }
            csv_rows.append(row)
            sample_methods.append(
                {
                    "method": method,
                    "source_group": method_bundle.group,
                    "source": str(source),
                    "status": status,
                    "is_better_than_baseline": is_better,
                    "score": method_score,
                    "improvement": method_score - baseline_score,
                }
            )
        manifest_samples.append(
            {
                "canonical_id": bundle.canonical_id,
                "index": bundle.index,
                "bundle_group": bundle.group,
                "evaluation_status": "evaluated",
                "reference_source": str(bundle.reference),
                "baseline_source": str(bundle.baseline),
                "baseline_score": baseline_score,
                "methods": sample_methods,
            }
        )

    for batch, shot, paths in d_samples:
        shot_stem = Path(shot).stem
        canonical = f"{batch}/{shot_stem}"
        bundle_counts["group3"] += 1
        ref_out = TARGET / f"o-video/shots/{batch}/{shot}"
        base_out = TARGET / f"baseline1/shots/{batch}/{shot}"
        copy_file(paths["reference"], ref_out)
        copy_file(paths["baseline"], base_out)
        add_derived("baseline1", canonical, "group3", "group3", ref_out, base_out, None, None)
        sample_methods = []
        for method in ("c-v", "r-v"):
            method_out = TARGET / f"methods/{method}/{batch}/{shot}"
            copy_file(paths[method], method_out)
            add_derived(method, canonical, "group3", "group3", ref_out, method_out, None, None)
            status_counts["d_series_direct_uncompared"] += 1
            csv_rows.append(
                {
                    "canonical_id": canonical,
                    "index": "",
                    "bundle_group": "group3",
                    "bundle_group_label": GROUP_LABELS["group3"],
                    "method": method,
                    "method_source_group": "group3",
                    "selection_status": "d_series_direct_uncompared",
                    "is_better_than_baseline": "",
                    "baseline_score": "",
                    "method_score": "",
                    "improvement": "",
                    "baseline_clip": "",
                    "method_clip": "",
                    "baseline_vclip": "",
                    "method_vclip": "",
                    "baseline_lpips": "",
                    "method_lpips": "",
                    "baseline_flow_epe": "",
                    "method_flow_epe": "",
                    "baseline_ssim": "",
                    "method_ssim": "",
                    "reference_source": str(paths["reference"]),
                    "baseline_source": str(paths["baseline"]),
                    "method_source": str(paths[method]),
                    "baseline_evaluation_source": "",
                    "method_evaluation_source": "",
                    "reference_output": str(ref_out),
                    "baseline_output": str(base_out),
                    "method_output": str(method_out),
                }
            )
            sample_methods.append(
                {
                    "method": method,
                    "source_group": "group3",
                    "source": str(paths[method]),
                    "status": "d_series_direct_uncompared",
                    "is_better_than_baseline": None,
                    "score": None,
                    "improvement": None,
                }
            )
        manifest_samples.append(
            {
                "canonical_id": canonical,
                "index": None,
                "bundle_group": "group3",
                "evaluation_status": "not_evaluated_direct_inclusion",
                "reference_source": str(paths["reference"]),
                "baseline_source": str(paths["baseline"]),
                "baseline_score": None,
                "methods": sample_methods,
            }
        )

    if len(manifest_samples) != 312 or len(csv_rows) != 624:
        raise ValueError("Complete output must contain 312 samples and 624 method rows")

    with (TARGET / "selection.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    formula = (
        "score = 100/70 × [30×VCLIP + 10×CLIP + 10×(1−LPIPS) "
        "+ 10×1/(1+Flow-EPE) + 10×SSIM]"
    )
    source_fvd = {
        "group1": {
            "baseline1": fvd_value(SHORT / "results_baseline1"),
            "c-v": fvd_value(SHORT / "results_c-v"),
            "r-v": fvd_value(SHORT / "results_r-v"),
        },
        "group2": {
            "baseline1": fvd_value(SHORT / "results_baseline1"),
            "c-v": fvd_value(AUTODL / "14B/eval/7.15/results_7.15"),
            "r-v": None,
        },
        "group3": {
            name: fvd_value(AUTODL / f"new-baseline/eval/shots/{name}")
            for name in ("baseline1", "c-v", "r-v")
        },
    }
    manifest = {
        "title": "7.16-best complete 52-group short-video dataset",
        "formula": formula,
        "fvd_used_for_selection": False,
        "sample_count": 312,
        "evaluated_sample_count": 252,
        "unevaluated_direct_sample_count": 60,
        "method_file_count": 624,
        "bundle_counts": dict(bundle_counts),
        "status_counts": dict(status_counts),
        "candidate_counts": dict(candidate_counts),
        "samples": manifest_samples,
    }
    (TARGET / "eval/shots/selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for dataset, metrics in derived.items():
        for metric, items in metrics.items():
            filename, key = METRICS[metric]
            evaluated = [item for item in items if item[key] is not None]
            data = {
                "metric": key,
                "dataset": dataset,
                "provenance": "Derived from existing evaluations; no metric was recomputed.",
                "item_count": len(items),
                "evaluated_count": len(evaluated),
                "not_evaluated_count": len(items) - len(evaluated),
                "average_score_over_evaluated_items": sum(item[key] for item in evaluated) / len(evaluated),
                "items": items,
            }
            (TARGET / f"eval/shots/{dataset}/{filename}").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        (TARGET / f"eval/shots/{dataset}/fvd.json").write_text(
            json.dumps(
                {
                    "metric": "fvd",
                    "dataset": dataset,
                    "final_mixed_dataset_fvd": None,
                    "status": "not_recomputed_for_final_mixed_dataset",
                    "reason": "FVD is distribution-level and cannot be merged or subsetted per video.",
                    "source_evaluation_fvd_values": source_fvd,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    lines = [
        "# 7.16-best 完整短视频数据审计报告",
        "",
        "## 最终规模",
        "",
        "- 52 组、312 个短视频：a 系列 120、c 系列 12、d 系列 60、l 系列 120。",
        "- `o-video/shots`、`baseline1/shots`、`methods/c-v`、`methods/r-v` 各 312 个视频。",
        "- a/c/l 的 252 个样本均有评测；d 系列 60 个样本按要求直接采用第3组，明确标记为未评测。",
        f"- 最终来源套件：第1组 {bundle_counts['group1']}，第2组 {bundle_counts['group2']}，第3组 {bundle_counts['group3']}（含 d 系列 60 个）。",
        "",
        "## 填充与覆盖规则",
        "",
        "- a/c/l 首先以第1组完整填充；没有 method 优于 baseline 时保留第1组全套。",
        "- 有更优候选时，按最大综合提升选择来源套件。",
        "- 第3组入选时，reference、baseline1、c-v、r-v 整套采用第3组，避免跨组混配。",
        "- 第2组入选时采用 7.15 c-v；其没有 r-v，因此 r-v 使用共享相同参考/baseline 的第1组版本。",
        "- d001–d010 没有对比评测，reference、baseline1、c-v、r-v 均直接采用第3组。",
        "",
        "## 评分",
        "",
        f"- `{formula}`",
        "- FVD 不参与逐视频筛选；最终混合数据集未重新计算 FVD。各来源已有 FVD 值保存在 `eval/shots/*/fvd.json` 中。",
        "",
        "## 评测完整性",
        "",
        "- `eval/shots/baseline1`、`c-v`、`r-v` 的每个指标 JSON 均包含完整 312 个 ID。",
        "- 每个 JSON 中 252 个 a/c/l 条目带现有评测值；60 个 d 条目使用 `null` 并标记 `not_evaluated`，没有静默遗漏。",
        "- `selection.csv` 每个样本固定两行（c-v、r-v），共 624 行。",
        "",
        "## 历史路径异常",
        "",
        "- 第1组评测保留旧 `/14B/c-v`、`r-v` 路径；当前文件位于 `/14B/v/c-v`、`v/r-v`，曾替换的文件已从评测前备份还原。",
        "- 第2组评测链接指向旧 `/14B/7.15all-caption-video`，当前文件位于 `/14B/v/7.15all-caption-video`。",
        "- 第3组 JSON 指向的 `_matched` 暂存目录当前不存在。",
        "- 映射依据为完整评测 ID、主 mapping、当前来源文件和来源套件目录结构；报告与 CSV 保留完整来源路径。",
        "",
        "## 全部样本来源",
        "",
        "| 样本 | 套件 | 评测状态 | c-v 状态 | r-v 状态 |",
        "|---|---|---|---|---|",
    ]
    for sample in manifest_samples:
        methods = {item["method"]: item["status"] for item in sample["methods"]}
        lines.append(
            f"| {sample['canonical_id']} | {sample['bundle_group']} | "
            f"{sample['evaluation_status']} | {methods['c-v']} | {methods['r-v']} |"
        )
    (TARGET / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "samples": len(manifest_samples),
                "csv_rows": len(csv_rows),
                "bundle_counts": dict(bundle_counts),
                "status_counts": dict(status_counts),
                "target": str(TARGET),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
