#!/usr/bin/env python3
"""使用 SSIM 对原视频与生成视频做逐帧结构相似度测评。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from eval_video_metrics_common import (
    ensure_dir,
    list_video_pairs,
    resolve_output_root,
    sample_aligned_video_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SSIM between reference and generated videos.")
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


def compute_frame_ssim(reference_frame: np.ndarray, generated_frame: np.ndarray) -> float:
    reference_float = reference_frame.astype(np.float32) / 255.0
    generated_float = generated_frame.astype(np.float32) / 255.0
    return float(
        structural_similarity(
            reference_float,
            generated_float,
            data_range=1.0,
            channel_axis=2,
        )
    )


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
        frame_scores = [
            compute_frame_ssim(reference_frame, generated_frame)
            for reference_frame, generated_frame in zip(reference_frames, generated_frames)
        ]
        video_score = float(sum(frame_scores) / len(frame_scores))
        per_video_scores.append(video_score)
        items.append(
            {
                "video_id": video_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "ssim_mean": video_score,
                "frame_scores": frame_scores,
            }
        )
        print(f"{video_id}: SSIM={video_score:.6f}")

    result = {
        "metric": "ssim",
        "reference_dir": str(Path(args.reference_dir).expanduser().resolve()),
        "generated_dir": str(Path(args.generated_dir).expanduser().resolve()),
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "average_score": float(sum(per_video_scores) / len(per_video_scores)),
        "higher_is_better": True,
        "items": items,
    }
    output_path = output_root / "ssim.json"
    write_json(output_path, result)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
