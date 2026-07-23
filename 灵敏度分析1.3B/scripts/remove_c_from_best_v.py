#!/usr/bin/env python3
"""Remove c001/c002 from best-v and recompute all derivable summaries."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/root/autodl-tmp/7.16-best/best-v")
REMOVED = {"c001", "c002"}
SHORT_METRICS = {
    "clip": ("clip_frame_similarity.json", "clip_frame_similarity"),
    "vclip": ("vclip_video_similarity.json", "vclip_video_similarity"),
    "lpips": ("lpips.json", "lpips_mean"),
    "flow": ("optical_flow_consistency.json", "optical_flow_endpoint_error_mean"),
    "ssim": ("ssim.json", "ssim_mean"),
}


def keep_id(value: str) -> bool:
    return value.split("/", 1)[0].split("__", 1)[0] not in REMOVED


def rewrite_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    # Remove short-video batch directories and long videos.
    for relative in ("o-video/shots", "baseline1/shots", "methods/best-v"):
        for group in REMOVED:
            path = ROOT / relative / group
            if path.exists():
                shutil.rmtree(path)
    for relative in ("o-video/video", "baseline1/video", "methods/long-best-v"):
        for group in REMOVED:
            (ROOT / relative / f"{group}.mp4").unlink(missing_ok=True)
        manifest_path = ROOT / relative / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["videos"] = [x for x in manifest.get("videos", []) if x.get("batch") not in REMOVED]
            manifest["count"] = len(manifest["videos"])
            rewrite_json(manifest_path, manifest)

    # Filter selection CSV and correct paths after the prior bset-v -> best-v rename.
    csv_path = ROOT / "selection.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if keep_id(row["canonical_id"])]
    if len(rows) != 300:
        raise ValueError(f"Expected 300 remaining selection rows, found {len(rows)}")
    for row in rows:
        batch, stem = row["canonical_id"].split("/", 1)
        row["reference_output"] = str(ROOT / f"o-video/shots/{batch}/{stem}.mp4")
        row["baseline_output"] = str(ROOT / f"baseline1/shots/{batch}/{stem}.mp4")
        row["method_output"] = str(ROOT / f"methods/best-v/{batch}/{stem}.mp4")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    # Rename the inherited short-eval method directory and filter all per-shot items.
    old_method_eval = ROOT / "eval/shots/bset-v"
    method_eval = ROOT / "eval/shots/best-v"
    if old_method_eval.exists() and not method_eval.exists():
        old_method_eval.rename(method_eval)
    for dataset_dir in (ROOT / "eval/shots/baseline1", method_eval):
        for metric, (filename, key) in SHORT_METRICS.items():
            path = dataset_dir / filename
            data = json.loads(path.read_text(encoding="utf-8"))
            items = [item for item in data["items"] if keep_id(item["canonical_id"])]
            if len(items) != 300:
                raise ValueError(f"{path}: expected 300 items, found {len(items)}")
            for item in items:
                item["reference_path"] = item["reference_path"].replace(
                    "/root/autodl-tmp/bset-v/", str(ROOT) + "/"
                )
                item["generated_path"] = item["generated_path"].replace(
                    "/root/autodl-tmp/bset-v/", str(ROOT) + "/"
                ).replace("/methods/bset-v/", "/methods/best-v/")
            data["items"] = items
            data["item_count"] = 300
            data["average_score"] = sum(float(item[key]) for item in items) / 300
            if dataset_dir == method_eval:
                data["dataset"] = "best-v"
            rewrite_json(path, data)

    # Filter long-video per-item metrics and recompute averages. FVD is intentionally removed.
    for dataset_dir in (ROOT / "eval/baseline1", ROOT / "eval/long-best-v"):
        for _, (filename, key) in SHORT_METRICS.items():
            path = dataset_dir / filename
            data = json.loads(path.read_text(encoding="utf-8"))
            items = [item for item in data["items"] if keep_id(item["video_id"])]
            if len(items) != 50:
                raise ValueError(f"{path}: expected 50 items, found {len(items)}")
            data["items"] = items
            data["average_score"] = sum(float(item[key]) for item in items) / 50
            rewrite_json(path, data)
        (dataset_dir / "fvd.json").unlink(missing_ok=True)
        for filename in ("per_video_metrics.csv", "evaluation_summary.md"):
            (dataset_dir / filename).unlink(missing_ok=True)
    for filename in ("long_comparison.csv", "long_comparison_summary.md"):
        (ROOT / "eval" / filename).unlink(missing_ok=True)

    # Recompute short-video audit summary and report from the remaining selection rows.
    winners = Counter(row["selected_method"] for row in rows)
    reasons = Counter(row["selection_reason"] for row in rows)
    sources = Counter(f"{row['selected_method']}:{row['source_group']}" for row in rows)
    deltas = [float(row["selected_difference"]) for row in rows]
    base_scores = [float(row["selected_baseline_score"]) for row in rows]
    method_scores = [float(row["selected_method_score"]) for row in rows]
    raw = {"baseline1": {}, "best-v": {}}
    for metric in SHORT_METRICS:
        field = metric if metric != "flow" else "flow_epe"
        raw["baseline1"][metric] = sum(float(row[f"baseline_{field}"]) for row in rows) / 300
        raw["best-v"][metric] = sum(float(row[f"method_{field}"]) for row in rows) / 300
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_id"].split("/", 1)[0]].append(float(row["selected_difference"]))
    summary = {
        "dataset": "best-v",
        "removed_groups": sorted(REMOVED),
        "sample_count": 300,
        "group_count": 50,
        "winner_counts": dict(winners),
        "selection_reason_counts": dict(reasons),
        "winner_source_counts": dict(sorted(sources.items())),
        "composite": {
            "baseline_mean": sum(base_scores) / 300,
            "method_mean": sum(method_scores) / 300,
            "mean_difference": sum(deltas) / 300,
            "positive_items": sum(x > 0 for x in deltas),
            "zero_items": sum(x == 0 for x in deltas),
            "negative_items": sum(x < 0 for x in deltas),
        },
        "raw_metric_means": raw,
        "fvd": {"value": None, "status": "short_video_mixed_dataset_not_recomputed"},
        "group_mean_differences": {k: sum(v) / len(v) for k, v in sorted(grouped.items())},
    }
    rewrite_json(ROOT / "eval/shots/eval_summary.json", summary)
    c = summary["composite"]
    lines = [
        "# best-v 短视频筛选汇总（已删除 c 组）", "",
        "- 已彻底移除 c001、c002，共 2 组、12 条短视频。",
        "- 当前数据集包含 a、d、l 共 50 组、300 条短视频。",
        "- 每条 reference、baseline1、method 仍严格同源对应。", "",
        "## 汇总", "",
        "| 项目 | 数值 |", "|---|---:|",
        f"| baseline 综合均值 | {c['baseline_mean']:.6f} |",
        f"| best-v 综合均值 | {c['method_mean']:.6f} |",
        f"| 平均分差 | {c['mean_difference']:+.6f} |",
        f"| 正差/负差/零差条目 | {c['positive_items']} / {c['negative_items']} / {c['zero_items']} |",
        f"| c-v / r-v 入选 | {winners['c-v']} / {winners['r-v']} |", "",
        "完整逐条来源、原始指标和哈希见 `selection.csv`。",
    ]
    (ROOT / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
