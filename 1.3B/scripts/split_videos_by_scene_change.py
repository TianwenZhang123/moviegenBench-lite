#!/usr/bin/env python3
"""按视频内容变化程度切分镜头，输出为 generated/shots 同款目录结构。"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import av
import imageio_ffmpeg


SHOWINFO_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


@dataclass
class VideoInfo:
    duration_seconds: float
    fps: float | None
    width: int
    height: int


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split videos into shots based on content changes."
    )
    parser.add_argument(
        "--input-dir",
        default="/root/video/generated/video",
        help="单个视频文件，或包含多个 mp4 的目录。",
    )
    parser.add_argument(
        "--output-root",
        default="/root/baseline1/shots",
        help="输出根目录，格式为 output_root/<视频名>/shot_01.mp4。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="ffmpeg scene 检测阈值，越小越容易切分。",
    )
    parser.add_argument(
        "--min-shot-seconds",
        type=float,
        default=2.0,
        help="最短镜头时长，小于该值的切点会被忽略。",
    )
    parser.add_argument(
        "--max-shot-seconds",
        type=float,
        default=8.0,
        help="最长镜头时长；大于该值时会额外均分，设为 0 可关闭。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出。",
    )
    return parser.parse_args()


def discover_videos(input_arg: str) -> list[Path]:
    path = Path(input_arg).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.mp4"))
    raise FileNotFoundError(f"Video path does not exist: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_video_info(video_path: Path) -> VideoInfo:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        else:
            raise RuntimeError(f"Unable to read duration from {video_path}")

        fps = float(stream.average_rate) if stream.average_rate else None
        return VideoInfo(
            duration_seconds=duration_seconds,
            fps=fps,
            width=stream.width,
            height=stream.height,
        )


def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}:\n{' '.join(command)}\n{result.stdout}"
        )
    return result.stdout


def detect_scene_times(
    ffmpeg_path: str,
    video_path: Path,
    threshold: float,
    duration_seconds: float,
) -> list[float]:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter_complex",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    output = run_command(command)

    scene_times: list[float] = []
    for match in SHOWINFO_RE.finditer(output):
        pts_time = float(match.group(1))
        if 0.0 < pts_time < duration_seconds:
            if not scene_times or abs(scene_times[-1] - pts_time) > 1e-3:
                scene_times.append(pts_time)
    return scene_times


def filter_short_segments(
    scene_times: list[float],
    duration_seconds: float,
    min_shot_seconds: float,
) -> list[float]:
    if min_shot_seconds <= 0:
        return scene_times[:]

    filtered: list[float] = []
    last_boundary = 0.0
    for scene_time in scene_times:
        if scene_time - last_boundary >= min_shot_seconds:
            filtered.append(scene_time)
            last_boundary = scene_time

    if filtered and duration_seconds - filtered[-1] < min_shot_seconds:
        filtered.pop()
    return filtered


def enforce_max_shot_length(
    scene_times: list[float],
    duration_seconds: float,
    max_shot_seconds: float,
) -> list[float]:
    if max_shot_seconds <= 0:
        return scene_times[:]

    boundaries = [0.0, *scene_times, duration_seconds]
    expanded: list[float] = []

    for start, end in zip(boundaries, boundaries[1:]):
        expanded.append(start)
        segment_duration = end - start
        if segment_duration <= max_shot_seconds:
            continue

        split_count = int(segment_duration // max_shot_seconds)
        for step in range(1, split_count + 1):
            split_time = start + step * max_shot_seconds
            if split_time < end - 1e-3:
                expanded.append(split_time)

    expanded.append(duration_seconds)
    deduped = sorted({round(value, 3) for value in expanded})
    return [value for value in deduped if 0.0 < value < duration_seconds]


def build_segments(
    scene_times: list[float],
    duration_seconds: float,
) -> list[Segment]:
    boundaries = [0.0, *scene_times, duration_seconds]
    segments: list[Segment] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start > 0.05:
            segments.append(Segment(start=start, end=end))
    return segments


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def render_segment(
    ffmpeg_path: str,
    video_path: Path,
    output_path: Path,
    segment: Segment,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-ss",
        format_seconds(segment.start),
        "-i",
        str(video_path),
        "-t",
        format_seconds(segment.duration),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_command(command)


def remove_stale_shots(batch_output_dir: Path, expected_count: int) -> None:
    for stale_path in sorted(batch_output_dir.glob("shot_*.mp4")):
        match = re.fullmatch(r"shot_(\d+)\.mp4", stale_path.name)
        if match and int(match.group(1)) > expected_count:
            stale_path.unlink()


def split_video(
    video_path: Path,
    output_root: Path,
    ffmpeg_path: str,
    threshold: float,
    min_shot_seconds: float,
    max_shot_seconds: float,
    overwrite: bool,
) -> list[Segment]:
    video_info = get_video_info(video_path)
    scene_times = detect_scene_times(
        ffmpeg_path=ffmpeg_path,
        video_path=video_path,
        threshold=threshold,
        duration_seconds=video_info.duration_seconds,
    )
    scene_times = filter_short_segments(
        scene_times=scene_times,
        duration_seconds=video_info.duration_seconds,
        min_shot_seconds=min_shot_seconds,
    )
    scene_times = enforce_max_shot_length(
        scene_times=scene_times,
        duration_seconds=video_info.duration_seconds,
        max_shot_seconds=max_shot_seconds,
    )
    segments = build_segments(
        scene_times=scene_times,
        duration_seconds=video_info.duration_seconds,
    )

    batch_output_dir = output_root / video_path.stem
    ensure_dir(batch_output_dir)
    for index, segment in enumerate(segments, start=1):
        output_path = batch_output_dir / f"shot_{index:02d}.mp4"
        render_segment(
            ffmpeg_path=ffmpeg_path,
            video_path=video_path,
            output_path=output_path,
            segment=segment,
            overwrite=overwrite,
        )
    if overwrite:
        remove_stale_shots(batch_output_dir, expected_count=len(segments))
    return segments


def main() -> int:
    args = parse_args()
    videos = discover_videos(args.input_dir)
    if not videos:
        print(f"No mp4 videos found in {args.input_dir}", file=sys.stderr)
        return 1

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)

    for video_path in videos:
        segments = split_video(
            video_path=video_path,
            output_root=output_root,
            ffmpeg_path=ffmpeg_path,
            threshold=args.threshold,
            min_shot_seconds=args.min_shot_seconds,
            max_shot_seconds=args.max_shot_seconds,
            overwrite=args.overwrite,
        )
        print(
            f"{video_path.name}: split into {len(segments)} shots -> "
            f"{output_root / video_path.stem}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
