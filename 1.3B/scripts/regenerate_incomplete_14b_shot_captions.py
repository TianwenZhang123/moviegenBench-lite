#!/usr/bin/env python3
"""Regenerate incomplete captions from the flattened generated/shots set."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Set this before importing torch through qwen_video_caption.
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

from qwen_video_caption import (  # noqa: E402
    build_model,
    caption_video,
    normalize_caption,
    write_text,
)


INCOMPLETE_IDS = (
    "3,5,6,20,21,37,40,47,65,67,77,79,98,102,105,110,116,117,118,"
    "122,123,125,127,132,133,136,137,138,140,141,142,146,147,148,149,"
    "153,154,156,158,161,162,164,165,166,168,175,176,177,179,180,183,"
    "184,185,186,187,188,192,195,199,200,203,208,210,213,214,215,220,"
    "223,224,225,226,228,229,235,236,238,239,240,243,247,248,249,251,252"
)
MAX_NEW_TOKENS = 200
NUM_FRAMES = 32
PROMPT = (
    "Respond in English only. Describe all visible content and scene changes in concise English. "
    "Use 70-110 words, cover the beginning, middle, and end, avoid repetition and "
    "speculation, and finish with a complete sentence."
)
TERMINAL_PUNCTUATION = (".", "!", "?", '.”', '!”', '?”')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots-root", default="/root/video/generated/shots")
    parser.add_argument(
        "--output-dir", default="/root/experiments/14B_shot_caption_regenerated"
    )
    parser.add_argument(
        "--model-path", default="/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids", default=INCOMPLETE_IDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_ids(value: str) -> list[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    allowed = {int(item) for item in INCOMPLETE_IDS.split(",")}
    unknown = sorted(set(ids) - allowed)
    if unknown:
        raise ValueError(f"IDs not listed as incomplete: {unknown}")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs were supplied.")
    return ids


def flatten_shots(shots_root: Path) -> list[Path]:
    groups = sorted(path for path in shots_root.iterdir() if path.is_dir())
    shots = [shot for group in groups for shot in sorted(group.glob("shot_*.mp4"))]
    if len(shots) != 252:
        raise RuntimeError(f"Expected 252 shots, found {len(shots)} under {shots_root}")
    return shots


def main() -> None:
    args = parse_args()
    shots_root = Path(args.shots_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    ids = selected_ids(args.ids)
    shots = flatten_shots(shots_root)
    mapping = {item: shots[item - 1] for item in ids}

    print(f"Selected {len(ids)} of {len(shots)} flattened shots")
    print(f"118 -> {shots[117]}")
    print(f"Output: {output_dir}")
    print(f"max_new_tokens={MAX_NEW_TOKENS} (unchanged)")
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = build_model(model_path, args.device)
    results = []
    for item in ids:
        video_path = mapping[item]
        output_path = output_dir / f"{item}.txt"
        if output_path.exists() and not args.overwrite:
            caption = output_path.read_text(encoding="utf-8").strip()
            print(f"[skip] {item} <- {video_path.parent.name}/{video_path.name}")
        else:
            print(f"[caption] {item} <- {video_path.parent.name}/{video_path.name}")
            caption = normalize_caption(
                caption_video(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    prompt=PROMPT,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_frames=NUM_FRAMES,
                )
            )
            write_text(output_path, caption)
            print(f"[done] {item}: {caption}")
        results.append(
            {
                "id": item,
                "shot_path": str(video_path),
                "output_path": str(output_path),
                "word_count": len(caption.split()),
                "character_count": len(caption),
                "complete_ending": caption.endswith(TERMINAL_PUNCTUATION),
                "caption": caption,
            }
        )

    # Always rebuild a complete manifest from all available target captions, even
    # when --ids is used to retry only a subset.
    manifest_results = []
    for target_id in (int(item) for item in INCOMPLETE_IDS.split(",")):
        target_output = output_dir / f"{target_id}.txt"
        if not target_output.is_file():
            continue
        target_caption = target_output.read_text(encoding="utf-8").strip()
        manifest_results.append(
            {
                "id": target_id,
                "shot_path": str(shots[target_id - 1]),
                "output_path": str(target_output),
                "word_count": len(target_caption.split()),
                "character_count": len(target_caption),
                "complete_ending": target_caption.endswith(TERMINAL_PUNCTUATION),
                "caption": target_caption,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mapping_rule": "sort batch directories, sort shot_*.mp4 within each batch, then use 1-based index",
        "shots_root": str(shots_root),
        "model_path": str(model_path),
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_frames": NUM_FRAMES,
        "prompt": PROMPT,
        "caption_count": len(manifest_results),
        "complete_ending_count": sum(row["complete_ending"] for row in manifest_results),
        "captions": manifest_results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[saved] {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
