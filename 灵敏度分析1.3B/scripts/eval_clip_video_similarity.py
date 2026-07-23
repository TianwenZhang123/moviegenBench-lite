#!/usr/bin/env python3
"""使用 CLIP 对原视频与生成视频做视频级相似度测评。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPProcessor, CLIPModel

from eval_video_metrics_common import (
    cosine_similarity,
    ensure_dir,
    list_video_pairs,
    load_local_clip_model,
    resolve_output_root,
    sample_video_frames,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate video-to-video similarity with CLIP.")
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
    parser.add_argument("--num-frames", type=int, default=8)
    return parser.parse_args()


def encode_video(
    *,
    model: CLIPModel,
    processor: CLIPProcessor,
    video_path: Path,
    num_frames: int,
    device: str,
) -> torch.Tensor:
    frames = sample_video_frames(video_path, num_frames=num_frames)
    inputs = processor(images=frames, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        image_outputs = model.get_image_features(pixel_values=pixel_values)
        if hasattr(image_outputs, "pooler_output"):
            image_features = image_outputs.pooler_output
        else:
            image_features = image_outputs
    return image_features.mean(dim=0)


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
        reference_feature = encode_video(
            model=model,
            processor=processor,
            video_path=reference_path,
            num_frames=args.num_frames,
            device=args.device,
        )
        generated_feature = encode_video(
            model=model,
            processor=processor,
            video_path=generated_path,
            num_frames=args.num_frames,
            device=args.device,
        )
        score = cosine_similarity(reference_feature, generated_feature)
        scores.append(score)
        items.append(
            {
                "video_id": video_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "clip_video_similarity": score,
            }
        )
        print(f"{video_id}: CLIP similarity={score:.6f}")

    result = {
        "metric": "clip_video_similarity",
        "reference_dir": str(Path(args.reference_dir).expanduser().resolve()),
        "generated_dir": str(Path(args.generated_dir).expanduser().resolve()),
        "model_name": args.model_name,
        "num_frames": args.num_frames,
        "average_score": float(sum(scores) / len(scores)),
        "items": items,
    }
    output_path = output_root / "clip_video_similarity.json"
    write_json(output_path, result)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
