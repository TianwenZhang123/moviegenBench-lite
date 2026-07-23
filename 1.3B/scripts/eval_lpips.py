#!/usr/bin/env python3
"""使用 LPIPS 对原视频与生成视频做逐帧感知距离测评。"""

from __future__ import annotations

import argparse
from pathlib import Path

import lpips
import torch

from eval_video_metrics_common import (
    ensure_dir,
    list_video_pairs,
    resolve_output_root,
    sample_aligned_video_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LPIPS between reference and generated videos.")
    parser.add_argument("--reference-dir", default="/root/video/generated/video")
    parser.add_argument("--generated-dir", default="/root/baseline1/generated/video")
    parser.add_argument(
        "--output-root",
        default="",
        help="可选输出目录；不传时自动写到生成视频对应根目录下的 comparison。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument(
        "--net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone。",
    )
    return parser.parse_args()


def frame_to_lpips_tensor(frame, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root, args.generated_dir)
    ensure_dir(output_root)

    pairs = list_video_pairs(args.reference_dir, args.generated_dir)
    metric = lpips.LPIPS(net=args.net).to(args.device)
    metric.eval()

    items: list[dict[str, object]] = []
    per_video_scores: list[float] = []

    for video_id, reference_path, generated_path in pairs:
        reference_frames, generated_frames = sample_aligned_video_frames(
            reference_path,
            generated_path,
            num_frames=args.num_frames,
            target_size=args.frame_size,
        )

        frame_scores: list[float] = []
        with torch.inference_mode():
            for reference_frame, generated_frame in zip(reference_frames, generated_frames):
                reference_tensor = frame_to_lpips_tensor(reference_frame, args.device)
                generated_tensor = frame_to_lpips_tensor(generated_frame, args.device)
                score = metric(reference_tensor, generated_tensor).item()
                frame_scores.append(float(score))

        video_score = float(sum(frame_scores) / len(frame_scores))
        per_video_scores.append(video_score)
        items.append(
            {
                "video_id": video_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "lpips_mean": video_score,
                "frame_scores": frame_scores,
            }
        )
        print(f"{video_id}: LPIPS={video_score:.6f}")

    result = {
        "metric": "lpips",
        "reference_dir": str(Path(args.reference_dir).expanduser().resolve()),
        "generated_dir": str(Path(args.generated_dir).expanduser().resolve()),
        "backbone": args.net,
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "average_score": float(sum(per_video_scores) / len(per_video_scores)),
        "lower_is_better": True,
        "items": items,
    }
    output_path = output_root / "lpips.json"
    write_json(output_path, result)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
