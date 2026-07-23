#!/usr/bin/env python3
"""Concatenate shots/<batch>/shot_*.mp4 into one normalized long video per batch.

The encoding settings intentionally match concat_14b_long_videos.py so results are
directly comparable with the existing new-baseline long-video evaluations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--crf", type=int, default=10)
    parser.add_argument("--expected-shots", type=int, default=6)
    parser.add_argument(
        "--stream-copy",
        action="store_true",
        help="Preserve source timestamps/FPS like new-baseline reference concatenation.",
    )
    return parser.parse_args()


def concat_group(ffmpeg: str, inputs: list[Path], output: Path, fps: int, crf: int) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as handle:
        for path in inputs:
            escaped = str(path.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        handle.flush()
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", handle.name,
                "-map", "0:v:0", "-an",
                "-vf", f"format=rgb24,format=yuv420p,setpts=N/({fps}*TB)",
                "-r", str(fps), "-c:v", "libx264", "-preset", "medium",
                "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-y", str(output),
            ],
            check=True,
        )


def concat_group_copy(ffmpeg: str, inputs: list[Path], output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as handle:
        for path in inputs:
            escaped = str(path.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
        handle.flush()
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", handle.name,
                "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
                "-y", str(output),
            ],
            check=True,
        )


def main() -> None:
    args = parse_args()
    source = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    batches = sorted(path for path in source.iterdir() if path.is_dir())
    if not batches:
        raise FileNotFoundError(f"No batch directories under {source}")
    plan = []
    for batch in batches:
        shots = sorted(batch.glob("shot_*.mp4"))
        if len(shots) != args.expected_shots:
            raise ValueError(f"{batch}: expected {args.expected_shots} shots, found {len(shots)}")
        expected_names = [f"shot_{number:02d}.mp4" for number in range(1, args.expected_shots + 1)]
        if [path.name for path in shots] != expected_names:
            raise ValueError(f"Noncanonical shot sequence in {batch}")
        plan.append((batch.name, shots))
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.mp4"):
        stale.unlink()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    manifest = {
        "source": str(source), "mode": "stream_copy" if args.stream_copy else "normalized_encode",
        "fps": None if args.stream_copy else args.fps,
        "crf": None if args.stream_copy else args.crf,
        "count": len(plan), "videos": [],
    }
    for batch, shots in plan:
        destination = output / f"{batch}.mp4"
        if args.stream_copy:
            concat_group_copy(ffmpeg, shots, destination)
        else:
            concat_group(ffmpeg, shots, destination, args.fps, args.crf)
        manifest["videos"].append({
            "batch": batch,
            "output_path": str(destination),
            "input_paths": [str(path) for path in shots],
        })
        print(f"[saved] {destination}", flush=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
