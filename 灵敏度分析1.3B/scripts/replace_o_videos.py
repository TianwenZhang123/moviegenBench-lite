#!/usr/bin/env python3
"""Replace mapped new-baseline/o-video shots from blue-v/o and rebuild long videos."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


SOURCE = Path("/root/autodl-tmp/blue-v/o")
TARGET = Path("/root/autodl-tmp/new-baseline/o-video")
MAPPING = Path("/root/autodl-tmp/14B/eval/short-eval/mapping.tsv")
BACKUP = TARGET / "o_replacement_backup"
MANIFEST = TARGET / "o_replacement_manifest.tsv"
SUMMARY = TARGET / "o_replacement_summary.json"
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


def rebuild_long(ffmpeg: str, group: str) -> dict[str, object]:
    inputs = [TARGET / "shots" / group / f"shot_{number:02d}.mp4" for number in range(1, 7)]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shots for {group}: {missing}")

    destination = TARGET / "video" / f"{group}.mp4"
    original_long = BACKUP / "video" / destination.name
    width, height, fps, total_frames = video_spec(original_long)
    if total_frames % 6:
        raise ValueError(f"Original {group} frame count {total_frames} is not divisible by 6")
    frames_per_shot = total_frames // 6

    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for video in inputs:
        command += ["-i", str(video)]

    chains: list[str] = []
    labels: list[str] = []
    for index in range(6):
        label = f"v{index}"
        labels.append(f"[{label}]")
        chains.append(
            f"[{index}:v:0]fps={fps:g},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"trim=start_frame=0:end_frame={frames_per_shot},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
    chains.append("".join(labels) + "concat=n=6:v=1:a=0[outv]")
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
        raise RuntimeError(f"Rebuilt {group} spec {actual} != original spec {expected}")
    return {"group": group, "width": width, "height": height, "fps": fps, "frames": total_frames}


def main() -> None:
    source_files = sorted(SOURCE.glob("*.mp4"), key=lambda path: int(path.stem))
    if len(source_files) != 32 or any(not path.stem.isdigit() for path in source_files):
        raise ValueError(f"Expected 32 numerically named MP4 files, got {len(source_files)}")

    with MAPPING.open(newline="", encoding="utf-8") as handle:
        index_mapping = {
            int(row["index"]): Path(row["reference_path"]).relative_to("/root/video/generated/shots")
            for row in csv.DictReader(handle, delimiter="\t")
        }

    replacements = []
    for source in source_files:
        index = int(source.stem)
        relative = index_mapping[index]
        target = TARGET / "shots" / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        replacements.append((index, source, relative, target))
    affected_groups = sorted({relative.parts[0] for _, _, relative, _ in replacements})

    (BACKUP / "shots").mkdir(parents=True, exist_ok=True)
    (BACKUP / "video").mkdir(parents=True, exist_ok=True)
    for _, _, relative, target in replacements:
        backup = BACKUP / "shots" / relative
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
    for group in affected_groups:
        source_long = TARGET / "video" / f"{group}.mp4"
        backup_long = BACKUP / "video" / source_long.name
        if not backup_long.exists():
            shutil.copy2(source_long, backup_long)

    rows = []
    for index, source, relative, target in replacements:
        backup = BACKUP / "shots" / relative
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"Hash mismatch after replacing index {index}")
        rows.append(
            {
                "index": index,
                "group": relative.parts[0],
                "shot": relative.name,
                "source": str(source),
                "target": str(target),
                "backup": str(backup),
                "old_sha256": sha256(backup),
                "new_sha256": target_hash,
            }
        )

    rebuilt_specs = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for group in affected_groups:
        rebuilt_specs.append(rebuild_long(ffmpeg, group))
        print(f"[rebuilt] {group}.mp4")

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(SOURCE),
        "target": str(TARGET),
        "replacement_count": len(rows),
        "affected_long_video_count": len(affected_groups),
        "affected_long_videos": [f"{group}.mp4" for group in affected_groups],
        "backup": str(BACKUP),
        "crf": CRF,
        "rebuilt_specs": rebuilt_specs,
        "manifest": str(MANIFEST),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] replaced {len(rows)} shots and rebuilt {len(affected_groups)} long videos")


if __name__ == "__main__":
    main()
