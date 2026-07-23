#!/usr/bin/env python3
"""使用逐帧 CLIP image cosine 平均，对原视频与生成视频做重建相似度测评。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from eval_video_metrics_common import (
    ensure_dir,
    list_video_pairs,
    load_local_clip_model,
    resolve_output_root,
    sample_aligned_video_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate video-to-video similarity with mean per-frame CLIP cosine."
    )
    parser.add_argument("--reference-dir", default="/root/video/generated/video")
    parser.add_argument("--generated-dir", default="/root/baseline1/generated/video")
    parser.add_argument(
        "--output-root",
        default="",
        help="可选输出目录；不传时自动写到生成视频对应根目录下的 comparison。",
    )
    parser.add_argument(
        "--model-name",
        default="/root/autodl-tmp/model/CLIP-ViT-B-16",
        help="本地 CLIP 模型目录。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    return parser.parse_args()


def encode_frames(
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    frames: list,
    device: str,
) -> torch.Tensor:
    inputs = processor(images=frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        if hasattr(image_outputs, "pooler_output"):
            image_features = image_outputs.pooler_output
        else:
            image_features = image_outputs
    return F.normalize(image_features, dim=-1)


def compute_frame_mean_similarity(
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    reference_path: Path,
    generated_path: Path,
    num_frames: int,
    device: str,
) -> tuple[float, list[float]]:
    reference_frames, generated_frames = sample_aligned_video_frames(
        reference_path,
        generated_path,
        num_frames=num_frames,
    )
    reference_features = encode_frames(
        model=model,
        processor=processor,
        frames=reference_frames,
        device=device,
    )
    generated_features = encode_frames(
        model=model,
        processor=processor,
        frames=generated_frames,
        device=device,
    )
    frame_scores = (reference_features * generated_features).sum(dim=-1)
    frame_scores_list = [float(score) for score in frame_scores.detach().cpu()]
    return float(frame_scores.mean().item()), frame_scores_list


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root, args.generated_dir)
    ensure_dir(output_root)

    pairs = list_video_pairs(args.reference_dir, args.generated_dir)
    model = load_local_clip_model(args.model_name, args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name, local_files_only=True)

    items: list[dict[str, object]] = []
    scores: list[float] = []
    for video_id, reference_path, generated_path in pairs:
        score, frame_scores = compute_frame_mean_similarity(
            model=model,
            processor=processor,
            reference_path=reference_path,
            generated_path=generated_path,
            num_frames=args.num_frames,
            device=args.device,
        )
        scores.append(score)
        items.append(
            {
                "video_id": video_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "clip_frame_similarity": score,
                "frame_scores": frame_scores,
            }
        )
        print(f"{video_id}: CLIP frame similarity={score:.6f}")

    result = {
        "metric": "clip_frame_similarity",
        "description": "Mean of per-frame CLIP image-feature cosine similarities after aligned uniform sampling.",
        "reference_dir": str(Path(args.reference_dir).expanduser().resolve()),
        "generated_dir": str(Path(args.generated_dir).expanduser().resolve()),
        "model_name": args.model_name,
        "num_frames": args.num_frames,
        "average_score": float(sum(scores) / len(scores)),
        "items": items,
    }
    output_path = output_root / "clip_frame_similarity.json"
    write_json(output_path, result)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
