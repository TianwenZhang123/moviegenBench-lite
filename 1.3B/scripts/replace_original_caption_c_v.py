#!/usr/bin/env python3
"""Replace selected c-v clips with blue original-caption videos and rebuild long-c-v."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


SOURCE = Path("/root/autodl-tmp/blue-v/蓝色_videos_original_caption_videos")
SHORT_DIR = Path("/root/autodl-tmp/14B/v/c-v")
LONG_DIR = Path("/root/autodl-tmp/14B/v/long-c-v")
LONG_MAPPING = LONG_DIR / "manifest.json"
BACKUP = SHORT_DIR / "original_caption_replacement_backup"
MANIFEST = SHORT_DIR / "original_caption_replacement_manifest.tsv"
SUMMARY = SHORT_DIR / "original_caption_replacement_summary.json"
CRF = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_spec(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    spec = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return spec


def rebuild_long(ffmpeg: str, batch: str, indices: list[int]) -> dict[str, object]:
    inputs = [SHORT_DIR / f"{index}.mp4" for index in indices]
    destination = LONG_DIR / f"{batch}.mp4"
    original_long = BACKUP / "long" / destination.name
    width, height, fps, total_frames = video_spec(original_long)
    if len(inputs) != 6 or total_frames % len(inputs):
        raise ValueError(f"Invalid mapping/spec for {batch}")
    frames_per_clip = total_frames // len(inputs)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for video in inputs:
        if not video.is_file():
            raise FileNotFoundError(video)
        command += ["-i", str(video)]

    chains: list[str] = []
    labels: list[str] = []
    for position in range(len(inputs)):
        label = f"v{position}"
        labels.append(f"[{label}]")
        chains.append(
            f"[{position}:v:0]fps={fps:g},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"trim=start_frame=0:end_frame={frames_per_clip},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
    chains.append("".join(labels) + f"concat=n={len(inputs)}:v=1:a=0[outv]")
    command += [
        "-filter_complex", ";".join(chains),
        "-map", "[outv]",
        "-an",
        "-r", f"{fps:g}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", str(destination),
    ]
    subprocess.run(command, check=True)
    actual = video_spec(destination)
    expected = (width, height, fps, total_frames)
    if actual != expected:
        raise RuntimeError(f"Rebuilt {batch} spec {actual} != original spec {expected}")
    return {
        "batch": batch,
        "input_indices": indices,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": total_frames,
    }


def main() -> None:
    sources = sorted(SOURCE.glob("*.mp4"), key=lambda path: int(path.stem))
    if len(sources) != 32 or any(not path.stem.isdigit() for path in sources):
        raise ValueError(f"Expected 32 numerically named source videos, got {len(sources)}")
    source_indices = {int(path.stem) for path in sources}

    long_mapping = json.loads(LONG_MAPPING.read_text(encoding="utf-8"))["videos"]
    affected = [item for item in long_mapping if source_indices.intersection(item["input_indices"])]
    if len(affected) != 24:
        raise ValueError(f"Expected 24 affected long videos, got {len(affected)}")

    (BACKUP / "short").mkdir(parents=True, exist_ok=True)
    (BACKUP / "long").mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = SHORT_DIR / source.name
        backup = BACKUP / "short" / source.name
        if not target.is_file():
            raise FileNotFoundError(target)
        if not backup.exists():
            shutil.copy2(target, backup)
    for item in affected:
        target = LONG_DIR / f"{item['batch']}.mp4"
        backup = BACKUP / "long" / target.name
        if not backup.exists():
            shutil.copy2(target, backup)

    rows = []
    index_to_batch = {
        index: item["batch"] for item in affected for index in item["input_indices"] if index in source_indices
    }
    for source in sources:
        index = int(source.stem)
        target = SHORT_DIR / source.name
        backup = BACKUP / "short" / source.name
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"Hash mismatch after replacing index {index}")
        rows.append(
            {
                "index": index,
                "long_batch": index_to_batch[index],
                "source": str(source),
                "target": str(target),
                "backup": str(backup),
                "old_sha256": sha256(backup),
                "new_sha256": target_hash,
            }
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    rebuilt = []
    for item in affected:
        rebuilt.append(rebuild_long(ffmpeg, item["batch"], item["input_indices"]))
        print(f"[rebuilt] {item['batch']}.mp4")

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(SOURCE),
        "short_target": str(SHORT_DIR),
        "long_target": str(LONG_DIR),
        "replacement_count": len(rows),
        "affected_long_video_count": len(rebuilt),
        "backup": str(BACKUP),
        "manifest": str(MANIFEST),
        "rebuilt_specs": rebuilt,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[done] replaced {len(rows)} {SHORT_DIR.name} clips "
        f"and rebuilt {len(rebuilt)} {LONG_DIR.name} videos"
    )


if __name__ == "__main__":
    main()
