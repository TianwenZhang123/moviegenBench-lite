#!/usr/bin/env python3
"""Regenerate only the incomplete 14B captions with a concise prompt.

The original captioning script and its 200-token generation limit are kept
unchanged.  Results are written to an isolated experiment directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


INCOMPLETE_IDS = (
    "3,5,6,20,21,37,40,47,65,67,77,79,98,102,105,110,116,117,118,"
    "122,123,125,127,132,133,136,137,138,140,141,142,146,147,148,149,"
    "153,154,156,158,161,162,164,165,166,168,175,176,177,179,180,183,"
    "184,185,186,187,188,192,195,199,200,203,208,210,213,214,215,220,"
    "223,224,225,226,228,229,235,236,238,239,240,243,247,248,249,251,252"
)
MAX_NEW_TOKENS = 200
PROMPT = (
    "Describe the video's visible content in English in 90-120 words. "
    "Cover the main subjects, actions, setting, and important scene changes. "
    "Be concise, avoid repetition and speculation, and end with a complete sentence."
)
TERMINAL_PUNCTUATION = (".", "!", "?", '.”', '!”', '?”')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the 84 incomplete 14B captions without raising max tokens."
    )
    parser.add_argument("--video-dir", default="/root/autodl-tmp/14B/best-v")
    parser.add_argument(
        "--output-dir", default="/root/experiments/14B_caption_regenerated"
    )
    parser.add_argument(
        "--model-path", default="/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument(
        "--ids",
        default=INCOMPLETE_IDS,
        help="Optional subset of the known incomplete IDs, comma-separated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    allowed = set(INCOMPLETE_IDS.split(","))
    unknown = sorted(set(ids) - allowed, key=lambda item: int(item))
    if unknown:
        raise ValueError(f"IDs not listed as incomplete: {', '.join(unknown)}")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs were supplied.")
    return ids


def validate_inputs(video_dir: Path, model_path: Path, ids: list[str]) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    missing = [item for item in ids if not (video_dir / f"{item}.mp4").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing videos: {', '.join(missing)}")


def write_validation(output_dir: Path, ids: list[str]) -> Path:
    results = []
    for item in ids:
        path = output_dir / f"{item}.txt"
        caption = path.read_text(encoding="utf-8").strip()
        results.append(
            {
                "id": item,
                "path": str(path),
                "word_count": len(caption.split()),
                "character_count": len(caption),
                "complete_ending": caption.endswith(TERMINAL_PUNCTUATION),
                "ending": caption[-100:],
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt": PROMPT,
        "caption_count": len(results),
        "complete_ending_count": sum(row["complete_ending"] for row in results),
        "captions": results,
    }
    report_path = output_dir / "validation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path


def main() -> None:
    args = parse_args()
    ids = parse_ids(args.ids)
    if not ids:
        raise ValueError("No IDs selected.")

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    original_script = Path(__file__).with_name("qwen_video_caption.py")
    validate_inputs(video_dir, model_path, ids)

    command = [
        sys.executable,
        str(original_script),
        "--video-dir",
        str(video_dir),
        "--output-dir",
        str(output_dir),
        "--model-path",
        str(model_path),
        "--device",
        args.device,
        "--ids",
        ",".join(ids),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--num-frames",
        str(args.num_frames),
        "--prompt",
        PROMPT,
    ]
    if args.overwrite:
        command.append("--overwrite")

    print(f"Selected {len(ids)} incomplete captions")
    print(f"Output: {output_dir}")
    print(f"max_new_tokens={MAX_NEW_TOKENS} (unchanged)")
    if args.dry_run:
        print("Dry run: generation was not started.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    report_path = write_validation(output_dir, ids)
    print(f"Validation: {report_path}")


if __name__ == "__main__":
    main()
