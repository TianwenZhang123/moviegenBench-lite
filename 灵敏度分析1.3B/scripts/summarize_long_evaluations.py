#!/usr/bin/env python3
"""Summarize two complete long-video evaluation directories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = {
    "clip": ("clip_frame_similarity.json", "clip_frame_similarity", True),
    "vclip": ("vclip_video_similarity.json", "vclip_video_similarity", True),
    "lpips": ("lpips.json", "lpips_mean", False),
    "flow": ("optical_flow_consistency.json", "optical_flow_endpoint_error_mean", False),
    "ssim": ("ssim.json", "ssim_mean", True),
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-eval", type=Path, required=True)
    p.add_argument("--method-eval", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline-name", default="baseline1")
    p.add_argument("--method-name", default="best-v")
    return p.parse_args()


def load(root: Path) -> tuple[dict[str, dict[str, float]], float]:
    values: dict[str, dict[str, float]] = {}
    ids = None
    for metric, (filename, key, _) in METRICS.items():
        data = json.loads((root / filename).read_text(encoding="utf-8"))
        items = {item["video_id"]: float(item[key]) for item in data["items"]}
        if not items:
            raise ValueError(f"{root / filename}: no evaluation items")
        if ids is not None and set(items) != ids:
            raise ValueError(f"ID mismatch in {root / filename}")
        ids = set(items)
        values[metric] = items
    fvd = float(json.loads((root / "fvd.json").read_text(encoding="utf-8"))["fvd"])
    return values, fvd


def score(v: dict[str, float]) -> float:
    return 100 / 70 * (
        30 * v["vclip"] + 10 * v["clip"] + 10 * (1 - v["lpips"])
        + 10 / (1 + v["flow"]) + 10 * v["ssim"]
    )


def group_name(video_id: str) -> str:
    return video_id[0]


def write_dataset(root: Path, name: str, values: dict[str, dict[str, float]], fvd: float) -> None:
    ids = sorted(values["clip"])
    count = len(ids)
    rows = []
    for video_id in ids:
        raw = {metric: values[metric][video_id] for metric in METRICS}
        rows.append({"video": video_id, **raw, "composite_score": score(raw)})
    with (root / "per_video_metrics.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    means = {key: sum(float(row[key]) for row in rows) / count for key in [*METRICS, "composite_score"]}
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[group_name(str(row["video"]))].append(row)
    lines = [
        f"# {name} 长视频评测汇总", "",
        f"- 视频数：{count}", "- 每项指标均匀采样：16 帧", "- FVD backbone：torchvision R3D-18", "",
        "## 总体指标", "",
        "| 指标 | 均值 | 方向 |", "|---|---:|---|",
        f"| CLIP | {means['clip']:.6f} | 越高越好 |",
        f"| VCLIP | {means['vclip']:.6f} | 越高越好 |",
        f"| LPIPS | {means['lpips']:.6f} | 越低越好 |",
        f"| Flow EPE | {means['flow']:.6f} | 越低越好 |",
        f"| SSIM | {means['ssim']:.6f} | 越高越好 |",
        f"| FVD | {fvd:.6f} | 越低越好 |",
        f"| 综合分 | {means['composite_score']:.6f} | 越高越好 |", "",
        "## 分类指标", "",
        "| 类别 | 数量 | CLIP | VCLIP | LPIPS | Flow EPE | SSIM | 综合分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("a", "c", "d", "l"):
        items = groups[group]
        if not items:
            continue
        av = {key: sum(float(row[key]) for row in items) / len(items) for key in [*METRICS, "composite_score"]}
        lines.append(
            f"| {group} | {len(items)} | {av['clip']:.6f} | {av['vclip']:.6f} | "
            f"{av['lpips']:.6f} | {av['flow']:.6f} | {av['ssim']:.6f} | {av['composite_score']:.6f} |"
        )
    (root / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    a = args()
    baseline, baseline_fvd = load(a.baseline_eval)
    method, method_fvd = load(a.method_eval)
    if set(baseline["clip"]) != set(method["clip"]):
        raise ValueError("Baseline and method long-video IDs differ")
    count = len(baseline["clip"])
    write_dataset(a.baseline_eval, a.baseline_name, baseline, baseline_fvd)
    write_dataset(a.method_eval, a.method_name, method, method_fvd)
    rows = []
    for video_id in sorted(baseline["clip"]):
        b = {metric: baseline[metric][video_id] for metric in METRICS}
        m = {metric: method[metric][video_id] for metric in METRICS}
        bs, ms = score(b), score(m)
        rows.append({
            "video": video_id,
            **{f"baseline_{k}": v for k, v in b.items()},
            **{f"method_{k}": v for k, v in m.items()},
            "baseline_composite_score": bs,
            "method_composite_score": ms,
            "composite_difference": ms - bs,
        })
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / "long_comparison.csv").open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    means = {}
    for label, data in (("baseline", baseline), ("method", method)):
        means[label] = {metric: sum(data[metric].values()) / count for metric in METRICS}
        means[label]["composite"] = sum(
            score({metric: data[metric][vid] for metric in METRICS}) for vid in data["clip"]
        ) / count
    deltas = [float(row["composite_difference"]) for row in rows]
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_name(str(row["video"]))].append(row)
    present_groups = [group for group in ("a", "c", "d", "l") if grouped_rows[group]]
    lines = [
        "# baseline1 与 best-v 长视频评测比较", "",
        "## 评测范围", "",
        f"- reference、baseline1、best-v 各 {count} 个长视频，组名完全一致。",
        "- 每个长视频由同组 6 个 short shot 按顺序拼接。",
        "- reference 保留原始时间戳，时长 30.23–32.40 秒；baseline1 与 best-v 均为 30.375 秒。",
        "- CLIP、VCLIP、LPIPS、Flow EPE、SSIM 删除异常组后重算均值；集合级 FVD 针对剩余视频重新测评。",
        "- 综合分：`100/70 × [30×VCLIP + 10×CLIP + 10×(1−LPIPS) + 10×1/(1+Flow-EPE) + 10×SSIM]`。", "",
        "## 总体结果", "",
        "| 数据 | 综合分 | CLIP ↑ | VCLIP ↑ | LPIPS ↓ | Flow EPE ↓ | SSIM ↑ | FVD ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| {a.baseline_name} | {means['baseline']['composite']:.6f} | {means['baseline']['clip']:.6f} | {means['baseline']['vclip']:.6f} | {means['baseline']['lpips']:.6f} | {means['baseline']['flow']:.6f} | {means['baseline']['ssim']:.6f} | {baseline_fvd:.6f} |",
        f"| {a.method_name} | {means['method']['composite']:.6f} | {means['method']['clip']:.6f} | {means['method']['vclip']:.6f} | {means['method']['lpips']:.6f} | {means['method']['flow']:.6f} | {means['method']['ssim']:.6f} | {method_fvd:.6f} |",
        f"| method−baseline | {means['method']['composite']-means['baseline']['composite']:+.6f} | {means['method']['clip']-means['baseline']['clip']:+.6f} | {means['method']['vclip']-means['baseline']['vclip']:+.6f} | {means['method']['lpips']-means['baseline']['lpips']:+.6f} | {means['method']['flow']-means['baseline']['flow']:+.6f} | {means['method']['ssim']-means['baseline']['ssim']:+.6f} | {method_fvd-baseline_fvd:+.6f} |",
        f"| 相对变化率 | {(means['method']['composite']/means['baseline']['composite']-1)*100:+.2f}% | {(means['method']['clip']/means['baseline']['clip']-1)*100:+.2f}% | {(means['method']['vclip']/means['baseline']['vclip']-1)*100:+.2f}% | {(means['method']['lpips']/means['baseline']['lpips']-1)*100:+.2f}% | {(means['method']['flow']/means['baseline']['flow']-1)*100:+.2f}% | {(means['method']['ssim']/means['baseline']['ssim']-1)*100:+.2f}% | {(method_fvd/baseline_fvd-1)*100:+.2f}% |", "",
        f"- 综合分正差组：{sum(x > 0 for x in deltas)}；负差组：{sum(x < 0 for x in deltas)}；零差组：{sum(x == 0 for x in deltas)}。",
        "- LPIPS、Flow EPE、FVD 越低越好，因此其差值为负时代表 method 改善。",
        "",
        f"## {'、'.join(present_groups)} 分组汇总",
        "",
        "FVD 是集合级指标，不拆分为逐组值。LPIPS 与 Flow EPE 越低越好。",
    ]
    for group in ("a", "c", "d", "l"):
        items = grouped_rows[group]
        if not items:
            continue
        b = {
            metric: sum(float(row[f"baseline_{metric}"]) for row in items) / len(items)
            for metric in METRICS
        }
        m = {
            metric: sum(float(row[f"method_{metric}"]) for row in items) / len(items)
            for metric in METRICS
        }
        bscore = sum(float(row["baseline_composite_score"]) for row in items) / len(items)
        mscore = sum(float(row["method_composite_score"]) for row in items) / len(items)
        positive = sum(float(row["composite_difference"]) > 0 for row in items)
        negative = sum(float(row["composite_difference"]) < 0 for row in items)
        zero = sum(float(row["composite_difference"]) == 0 for row in items)
        lines += [
            "",
            f"### {group} 组（{len(items)} 个视频）",
            "",
            "| 数据 | 综合分 | CLIP ↑ | VCLIP ↑ | LPIPS ↓ | Flow EPE ↓ | SSIM ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| {a.baseline_name} | {bscore:.6f} | {b['clip']:.6f} | {b['vclip']:.6f} | {b['lpips']:.6f} | {b['flow']:.6f} | {b['ssim']:.6f} |",
            f"| {a.method_name} | {mscore:.6f} | {m['clip']:.6f} | {m['vclip']:.6f} | {m['lpips']:.6f} | {m['flow']:.6f} | {m['ssim']:.6f} |",
            f"| method−baseline | {mscore-bscore:+.6f} | {m['clip']-b['clip']:+.6f} | {m['vclip']-b['vclip']:+.6f} | {m['lpips']-b['lpips']:+.6f} | {m['flow']-b['flow']:+.6f} | {m['ssim']-b['ssim']:+.6f} |",
            f"| 相对变化率 | {(mscore/bscore-1)*100:+.2f}% | {(m['clip']/b['clip']-1)*100:+.2f}% | {(m['vclip']/b['vclip']-1)*100:+.2f}% | {(m['lpips']/b['lpips']-1)*100:+.2f}% | {(m['flow']/b['flow']-1)*100:+.2f}% | {(m['ssim']/b['ssim']-1)*100:+.2f}% |",
            "",
            f"- 综合分正差 {positive} 个，负差 {negative} 个，零差 {zero} 个。",
        ]
    (a.output_dir / "long_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"means": means, "baseline_fvd": baseline_fvd, "method_fvd": method_fvd}, indent=2))


if __name__ == "__main__":
    main()
