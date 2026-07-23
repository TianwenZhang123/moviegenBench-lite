#!/usr/bin/env python3
"""使用视频特征近似计算两组视频之间的 FVD。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

from eval_video_metrics_common import (
    compute_mean_and_covariance,
    ensure_dir,
    frames_to_tensor,
    frechet_distance,
    list_video_pairs,
    resolve_output_root,
    sample_video_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FVD between reference and generated videos.")
    parser.add_argument("--reference-dir", default="/root/video/generated/video")
    parser.add_argument("--generated-dir", default="/root/baseline1/generated/video")
    parser.add_argument(
        "--output-root",
        default="",
        help="可选输出目录；不传时自动写到生成视频对应根目录下的 comparison。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    return parser.parse_args()


def build_feature_extractor() -> tuple[torch.nn.Module, R3D_18_Weights]:
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights


def encode_video(
    *,
    model: torch.nn.Module,
    weights: R3D_18_Weights,
    video_path: Path,
    num_frames: int,
    device: str,
) -> np.ndarray:
    frame_size = weights.transforms().crop_size[0]
    frames = sample_video_frames(
        video_path,
        num_frames=num_frames,
        target_size=(frame_size, frame_size),
    )
    # torchvision 视频分类预处理期望输入为 [T, C, H, W]。
    video_tensor = frames_to_tensor(frames)
    video_tensor = weights.transforms()(video_tensor).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = model(video_tensor)
    return features.squeeze(0).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root, args.generated_dir)
    ensure_dir(output_root)

    pairs = list_video_pairs(args.reference_dir, args.generated_dir)
    model, weights = build_feature_extractor()
    model = model.to(args.device)

    reference_features: list[np.ndarray] = []
    generated_features: list[np.ndarray] = []
    items: list[dict[str, object]] = []

    for video_id, reference_path, generated_path in pairs:
        reference_feature = encode_video(
            model=model,
            weights=weights,
            video_path=reference_path,
            num_frames=args.num_frames,
            device=args.device,
        )
        generated_feature = encode_video(
            model=model,
            weights=weights,
            video_path=generated_path,
            num_frames=args.num_frames,
            device=args.device,
        )
        reference_features.append(reference_feature)
        generated_features.append(generated_feature)
        items.append(
            {
                "video_id": video_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
            }
        )
        print(f"{video_id}: extracted features")

    reference_array = np.stack(reference_features, axis=0)
    generated_array = np.stack(generated_features, axis=0)
    ref_mean, ref_cov = compute_mean_and_covariance(reference_array)
    gen_mean, gen_cov = compute_mean_and_covariance(generated_array)
    score = frechet_distance(ref_mean, ref_cov, gen_mean, gen_cov)

    result = {
        "metric": "fvd",
        "reference_dir": str(Path(args.reference_dir).expanduser().resolve()),
        "generated_dir": str(Path(args.generated_dir).expanduser().resolve()),
        "feature_backbone": "torchvision_r3d_18",
        "num_frames": args.num_frames,
        "fvd": score,
        "items": items,
    }
    output_path = output_root / "fvd.json"
    write_json(output_path, result)
    print(f"FVD={score:.6f}")
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
