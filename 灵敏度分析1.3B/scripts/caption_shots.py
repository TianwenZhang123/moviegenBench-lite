#!/usr/bin/env python3
"""对 shots 目录下的分镜视频批量生成 caption，并输出为分组目录结构。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_video_caption import (
    build_model,
    caption_video,
    normalize_caption,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption shot videos and save as caption_xx.txt under each batch directory."
    )
    parser.add_argument(
        "--shots-root",
        default="/root/baseline1/shots",
        help="shots 根目录，格式为 shots_root/<批次>/shot_01.mp4。",
    )
    parser.add_argument(
        "--output-root",
        default="/root/baseline1/caption",
        help="caption 输出根目录，格式为 output_root/<批次>/caption_01.txt。",
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct",
        help="本地 Qwen2.5-VL 模型目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="推理设备，例如 cuda:0 或 cpu。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="caption 最大生成 token 数。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已存在的 caption 文件。",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the video content in English.",
        help="给视觉模型的提示词。",
    )
    return parser.parse_args()


def discover_shot_groups(shots_root: Path) -> list[tuple[str, list[Path]]]:
    if not shots_root.exists():
        raise FileNotFoundError(f"Shots root not found: {shots_root}")

    groups: list[tuple[str, list[Path]]] = []
    for batch_dir in sorted(path for path in shots_root.iterdir() if path.is_dir()):
        videos = sorted(path for path in batch_dir.glob("shot_*.mp4") if path.is_file())
        if videos:
            groups.append((batch_dir.name, videos))
    return groups


def caption_output_name(shot_path: Path) -> str:
    suffix = shot_path.stem.split("_")[-1]
    return f"caption_{suffix}.txt"


def main() -> None:
    args = parse_args()
    shots_root = Path(args.shots_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    groups = discover_shot_groups(shots_root)
    if not groups:
        raise FileNotFoundError(f"No shot videos found under {shots_root}")

    model, processor = build_model(model_path, args.device)
    summary: dict[str, object] = {
        "model_path": str(model_path),
        "shots_root": str(shots_root),
        "output_root": str(output_root),
        "prompt": args.prompt,
        "batches": [],
    }

    for batch_name, videos in groups:
        batch_output_dir = output_root / batch_name
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        batch_results: list[dict[str, str]] = []

        for video_path in videos:
            output_path = batch_output_dir / caption_output_name(video_path)
            if output_path.exists() and not args.overwrite:
                caption = output_path.read_text(encoding="utf-8").strip()
                print(f"[skip] {batch_name}/{video_path.name} -> {output_path.name}")
            else:
                print(f"[caption] {batch_name}/{video_path.name}")
                caption = normalize_caption(
                    caption_video(
                        model=model,
                        processor=processor,
                        video_path=video_path,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
                write_text(output_path, caption)
                print(f"[done] {batch_name}/{output_path.name}: {caption}")

            batch_results.append(
                {
                    "shot_id": video_path.stem,
                    "shot_file": video_path.name,
                    "shot_path": str(video_path),
                    "caption_file": output_path.name,
                    "caption_path": str(output_path),
                    "caption": caption,
                }
            )

        cast_batches = summary["batches"]
        assert isinstance(cast_batches, list)
        cast_batches.append(
            {
                "batch_id": batch_name,
                "output_dir": str(batch_output_dir),
                "captions": batch_results,
            }
        )

    write_json(output_root / "captions.json", summary)
    print(f"[saved] {output_root / 'captions.json'}")


if __name__ == "__main__":
    main()
