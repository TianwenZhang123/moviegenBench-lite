#!/usr/bin/env python3
"""Integrate new_60 d-series evaluations into both 7.16 independent sets."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/7.16-best")
NEW_EVAL = Path("/root/autodl-tmp/new_60/eval")
BASE_EVAL = NEW_EVAL / "best60_baseline_videos"
METHOD_EVAL = {
    "c-v": NEW_EVAL / "new_svd_fi_videos",
    "r-v": NEW_EVAL / "new_svd_fi_re_videos",
}
METRICS = {
    "clip": ("clip_frame_similarity.json", "clip_frame_similarity"),
    "vclip": ("vclip_video_similarity.json", "vclip_video_similarity"),
    "lpips": ("lpips.json", "lpips_mean"),
    "flow": ("optical_flow_consistency.json", "optical_flow_endpoint_error_mean"),
    "ssim": ("ssim.json", "ssim_mean"),
}


def load(directory: Path) -> dict[str, dict[str, float]]:
    result = {}
    expected = {str(index) for index in range(1, 61)}
    for metric, (filename, key) in METRICS.items():
        data = json.loads((directory / filename).read_text(encoding="utf-8"))
        values = {item["video_id"]: float(item[key]) for item in data["items"]}
        if set(values) != expected:
            raise ValueError(f"{directory}/{filename} does not contain IDs 1-60")
        result[metric] = values
    return result


def score(metrics: dict[str, dict[str, float]], video_id: str) -> float:
    value = (
        30 * metrics["vclip"][video_id]
        + 10 * metrics["clip"][video_id]
        + 10 * (1 - metrics["lpips"][video_id])
        + 10 * (1 / (1 + metrics["flow"][video_id]))
        + 10 * metrics["ssim"][video_id]
    )
    return value * 100 / 70


def d_index(canonical_id: str) -> str:
    batch, shot = canonical_id.split("/", 1)
    group_number = int(batch[1:])
    shot_number = int(shot.split("_")[1])
    return str((group_number - 1) * 6 + shot_number)


def main() -> None:
    baseline_metrics = load(BASE_EVAL)
    method_metrics = {method: load(directory) for method, directory in METHOD_EVAL.items()}
    with (ROOT / "selection.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 624:
        raise ValueError("Expected 624 rows")

    for row in rows:
        if not row["canonical_id"].startswith("d"):
            continue
        method = row["method"]
        video_id = d_index(row["canonical_id"])
        baseline_score = score(baseline_metrics, video_id)
        method_score = score(method_metrics[method], video_id)
        difference = method_score - baseline_score
        row["selection_status"] = (
            "d_series_evaluated_positive_delta"
            if difference > 0
            else "d_series_evaluated_nonpositive_delta"
        )
        row["baseline_score"] = baseline_score
        row["method_score"] = method_score
        row["difference"] = difference
        for metric, csv_name in [
            ("clip", "clip"),
            ("vclip", "vclip"),
            ("lpips", "lpips"),
            ("flow", "flow_epe"),
            ("ssim", "ssim"),
        ]:
            row[f"baseline_{csv_name}"] = baseline_metrics[metric][video_id]
            row[f"method_{csv_name}"] = method_metrics[method][metric][video_id]
        row["baseline_evaluation_source"] = str(BASE_EVAL)
        row["method_evaluation_source"] = str(METHOD_EVAL[method])

    with (ROOT / "selection.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_method = {method: [row for row in rows if row["method"] == method] for method in ("c-v", "r-v")}
    for method, method_rows in by_method.items():
        root = ROOT / method
        by_canonical = {row["canonical_id"]: row for row in method_rows}
        for dataset, evaluation in [("baseline1", BASE_EVAL), (method, METHOD_EVAL[method])]:
            for metric, (filename, key) in METRICS.items():
                path = root / f"eval/shots/{dataset}/{filename}"
                data = json.loads(path.read_text(encoding="utf-8"))
                for item in data["items"]:
                    if not item["canonical_id"].startswith("d"):
                        continue
                    row = by_canonical[item["canonical_id"]]
                    csv_key = {
                        "clip": "clip",
                        "vclip": "vclip",
                        "lpips": "lpips",
                        "flow": "flow_epe",
                        "ssim": "ssim",
                    }[metric]
                    item[key] = float(row[f"{'baseline' if dataset == 'baseline1' else 'method'}_{csv_key}"])
                    item["evaluation_status"] = "evaluated"
                    item["evaluation_source"] = str(evaluation)
                values = [float(item[key]) for item in data["items"]]
                if len(values) != 312:
                    raise ValueError(f"{path}: expected 312 items")
                data["evaluated_count"] = 312
                data["not_evaluated_count"] = 0
                data["average_score_over_evaluated_items"] = sum(values) / 312
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            fvd_path = root / f"eval/shots/{dataset}/fvd.json"
            fvd = json.loads(fvd_path.read_text(encoding="utf-8"))
            fvd.setdefault("source_values", {})["d_new_60"] = {
                "baseline1": json.loads((BASE_EVAL / "fvd.json").read_text(encoding="utf-8"))["fvd"],
                method: json.loads((METHOD_EVAL[method] / "fvd.json").read_text(encoding="utf-8"))["fvd"],
            }
            fvd_path.write_text(json.dumps(fvd, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        deltas = np.array([float(row["difference"]) for row in method_rows])
        baselines = np.array([float(row["baseline_score"]) for row in method_rows])
        methods = np.array([float(row["method_score"]) for row in method_rows])
        group_values: dict[str, list[float]] = defaultdict(list)
        for row in method_rows:
            group_values[row["canonical_id"].split("/")[0]].append(float(row["difference"]))
        group_means = {group: float(np.mean(values)) for group, values in group_values.items()}
        if len(group_means) != 52:
            raise ValueError("Expected 52 group means")
        rng = np.random.default_rng(20260716 if method == "c-v" else 20260717)
        bootstrap = deltas[rng.integers(0, 312, size=(20000, 312))].mean(axis=1)
        summary = {
            "evaluated_videos": 312,
            "unevaluated_d_videos": 0,
            "baseline_mean": float(baselines.mean()),
            "method_mean": float(methods.mean()),
            "mean_difference": float(deltas.mean()),
            "positive_videos": int((deltas > 0).sum()),
            "negative_videos": int((deltas < 0).sum()),
            "positive_groups": sum(value > 0 for value in group_means.values()),
            "negative_groups": sum(value < 0 for value in group_means.values()),
            "bootstrap_95ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "source_counts": dict(Counter(row["source_group"] for row in method_rows)),
            "group_mean_differences": dict(sorted(group_means.items())),
        }
        manifest_path = root / "eval/shots/selection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluated_count"] = 312
        manifest["not_evaluated_count"] = 0
        manifest["summary"] = summary
        item_map = {item["canonical_id"]: item for item in manifest["items"]}
        for row in method_rows:
            if not row["canonical_id"].startswith("d"):
                continue
            item = item_map[row["canonical_id"]]
            item["selection_status"] = row["selection_status"]
            item["baseline_score"] = float(row["baseline_score"])
            item["method_score"] = float(row["method_score"])
            item["difference"] = float(row["difference"])
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"method": method, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
