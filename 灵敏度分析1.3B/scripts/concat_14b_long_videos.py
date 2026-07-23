#!/usr/bin/env python3
"""Rebuild 14B long videos using an existing long-video manifest as mapping."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate 14B clips with manifest mapping.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--crf", type=int, default=10)
    return parser.parse_args()


def concat_group(
    ffmpeg: str,
    input_paths: list[Path],
    output_path: Path,
    fps: int,
    crf: int,
) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8") as manifest:
        for video in input_paths:
            escaped = str(video.resolve()).replace("'", "'\\''")
            manifest.write(f"file '{escaped}'\n")
        manifest.flush()
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                manifest.name,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"format=rgb24,format=yuv420p,setpts=N/({fps}*TB)",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(output_path),
            ],
            check=True,
        )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    template_path = args.template_manifest.expanduser().resolve()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    result = {
        "source": str(input_dir),
        "mapping_template": str(template_path),
        "fps": args.fps,
        "crf": args.crf,
        "count": len(template["videos"]),
        "videos": [],
    }
    for item in template["videos"]:
        indices = item["input_indices"]
        input_paths = [input_dir / f"{index}.mp4" for index in indices]
        missing = [str(path) for path in input_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing input files: {missing}")
        output_path = output_dir / f"{item['batch']}.mp4"
        concat_group(ffmpeg, input_paths, output_path, args.fps, args.crf)
        result["videos"].append(
            {
                "batch": item["batch"],
                "output_path": str(output_path),
                "input_indices": indices,
                "input_paths": [str(path) for path in input_paths],
            }
        )
        print(f"[saved] {output_path}")

    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
