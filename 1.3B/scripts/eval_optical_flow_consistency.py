#!/usr/bin/env python3
"""使用 OpenCV 光流对原视频与生成视频做时序一致性差异测评。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from eval_video_metrics_common import (
    ensure_dir,
    list_video_pairs,
    resolve_output_root,
    sample_aligned_video_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate optical-flow consistency between reference and generated videos."
    )
    parser.add_argument("--reference-dir", default="/root/video/generated/video")
    parser.add_argument("--generated-dir", default="/root/baseline1/generated/video")
    parser.add_argument(
        "--output-root",
        default="",
        help="可选输出目录；不传时自动写到生成视频对应根目录下的 comparison。",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=256)
    return parser.parse_args()


def rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def compute_flow(frame_prev: np.ndarray, frame_next: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        rgb_to_gray(frame_prev),
        rgb_to_gray(frame_next),
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def flow_endpoint_error(reference_flow: np.ndarray, generated_flow: np.ndarray) -> float:
    difference = reference_flow - generated_flow
    error_map = np.linalg.norm(difference, axis=2)
    return float(np.mean(error_map))


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root, args.generated_dir)
    ensure_dir(output_root)

    pairs = list_video_pairs(args.reference_dir, args.generated_dir)

    items: list[dict[str, object]] = []
    per_video_scores: list[float] = []

    for video_id, reference_path, generated_path in pairs:
        reference_frames, generated_frames = sample_aligned_video_frames(
            reference_path,
            generated_path,
            num_frames=args.num_frames,
            target_size=args.frame_size,
        )

        step_scores: list[float] = []
        for index in range(len(reference_frames) - 1):
            reference_flow = compute_flow(reference_frames[index], reference_frames[index + 1])
            generated_flow = compute_flow(generated_frames[index], generated_frames[index + 1])
            step_scores.append(flow_endpoint_error(reference_flow, generated_flow))

        video_score = float(sum(step_scores) / len(step_scores))
        per_video_scores.append(video_score)
        items.append(
            {
                "video_id": video_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "optical_flow_endpoint_error_mean": video_score,
                "step_scores": step_scores,
                "method": "opencv_farneback_endpoint_error",
            }
        )
        print(f"{video_id}: optical-flow EPE={video_score:.6f}")

    result = {
        "metric": "optical_flow_consistency",
        "reference_dir": str(Path(args.reference_dir).expanduser().resolve()),
        "generated_dir": str(Path(args.generated_dir).expanduser().resolve()),
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "optical_flow_method": "opencv_farneback",
        "distance": "endpoint_error",
        "average_score": float(sum(per_video_scores) / len(per_video_scores)),
        "lower_is_better": True,
        "items": items,
    }
    output_path = output_root / "optical_flow_consistency.json"
    write_json(output_path, result)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
