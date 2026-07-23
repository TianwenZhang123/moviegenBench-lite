#!/usr/bin/env python3
"""根据分镜 caption 批量生成 5 秒视频，并输出为 generated 同款目录结构。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# 某些运行环境会注入非法的 OMP_NUM_THREADS，先兜底为 1，避免 libgomp 警告。
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
from tqdm import tqdm

from diffsynth.utils.data import save_video

from generate_videos_from_captions import build_pipeline, compute_num_frames
from generate_wan_storyboard import concatenate_videos, parse_gpu_ids, parse_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read grouped shot captions and render each shot before concatenating."
    )
    parser.add_argument(
        "--caption-root",
        default="/root/baseline1/caption",
        help="caption 根目录，格式为 caption_root/<批次>/caption_01.txt。",
    )
    parser.add_argument(
        "--model-dir",
        default="/root/autodl-tmp/model/Wan2.1-T2V-14B",
        help="Wan2.1-T2V-14B 模型目录。",
    )
    parser.add_argument(
        "--output-root",
        default="/root/baseline1/generated",
        help="输出根目录，保持与 video/generated 一致。",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="用于生成的 GPU 列表，逗号分隔；例如 0,1。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="采样步数，默认 30。",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=6.0,
        help="CFG scale，默认 6.0。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="输出视频帧率，默认 16。",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=5,
        # 中文说明：这里默认改成 5 秒，是因为当前输入是一组分镜 caption，
        # 每个 caption 对应一个 shot，目标是生成与 generated/shots 一致的 5 秒分镜。
        help="单个分镜视频时长，默认 5 秒。",
    )
    parser.add_argument(
        "--resolution",
        default="832x480",
        help="输出分辨率，例如 832x480。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子；每个批次、每个分镜会在此基础上递增。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="已存在的视频是否覆盖。",
    )
    parser.add_argument(
        "--mode",
        choices=("orchestrate", "worker"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--assigned-shots",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu-slot",
        default="0",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def discover_caption_batches(caption_root: str) -> list[dict]:
    root = Path(caption_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Caption root does not exist: {root}")

    batches: list[dict] = []
    for batch_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        caption_files = sorted(path for path in batch_dir.glob("caption_*.txt") if path.is_file())
        if not caption_files:
            continue
        batches.append(
            {
                "batch_id": batch_dir.name,
                "batch_dir": batch_dir,
                "caption_files": caption_files,
            }
        )
    return batches


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_caption(text: str) -> str:
    return " ".join(text.strip().split())


def build_prompt(caption: str) -> str:
    return caption


def build_negative_prompt() -> str:
    return ""


def resolve_output_paths(output_root: Path, batch_id: str) -> dict[str, Path]:
    return {
        "shots_dir": output_root / "shots" / batch_id,
        "logs_dir": output_root / "logs" / batch_id,
        "render_config_path": output_root / "render_configs" / f"{batch_id}.json",
        "final_video": output_root / "video" / f"{batch_id}.mp4",
    }


def write_launch_log(path: Path, command: list[str], gpu_id: str) -> None:
    command_str = " ".join(shlex.quote(part) for part in command)
    lines = [
        f"gpu={gpu_id}",
        f"command={command_str}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def wait_for_one(active_processes: list[dict]) -> dict:
    while True:
        for index, item in enumerate(active_processes):
            return_code = item["process"].poll()
            if return_code is not None:
                finished = active_processes.pop(index)
                finished["return_code"] = return_code
                return finished
        time.sleep(1)


def split_shots(shot_count: int, worker_count: int) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(worker_count)]
    for shot_index in range(shot_count):
        buckets[shot_index % worker_count].append(shot_index)
    return [bucket for bucket in buckets if bucket]


def build_batch_payload(
    *,
    args: argparse.Namespace,
    batch_id: str,
    caption_files: list[Path],
    output_paths: dict[str, Path],
) -> dict:
    shots: list[dict] = []
    for index, caption_file in enumerate(caption_files, start=1):
        caption = sanitize_caption(caption_file.read_text(encoding="utf-8"))
        shots.append(
            {
                "shot": index,
                "caption_file": str(caption_file),
                "caption": caption,
                "prompt": build_prompt(caption),
                "negative_prompt": build_negative_prompt(),
                "seed": args.seed + index,
                "video_path": str(output_paths["shots_dir"] / f"shot_{index:02d}.mp4"),
            }
        )

    return {
        "batch_id": batch_id,
        "model_dir": str(Path(args.model_dir).resolve()),
        "gpus": parse_gpu_ids(args.gpus),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "fps": args.fps,
        "duration_seconds": args.duration_seconds,
        "num_frames": compute_num_frames(args.fps, args.duration_seconds),
        "resolution": args.resolution,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "shots_dir": str(output_paths["shots_dir"]),
        "video_path": str(output_paths["final_video"]),
        "shots": shots,
    }


def load_batch_payload(caption_root: str, batch_id: str) -> dict:
    batches = discover_caption_batches(caption_root)
    for batch in batches:
        if batch["batch_id"] == batch_id:
            return batch
    raise FileNotFoundError(f"Batch not found: {batch_id}")


def worker_render(args: argparse.Namespace) -> None:
    if not args.batch_id.strip():
        raise ValueError("Worker mode requires --batch-id.")

    batch = load_batch_payload(args.caption_root, args.batch_id.strip())
    caption_files = batch["caption_files"]
    batch_id = batch["batch_id"]
    output_root = Path(args.output_root).resolve()
    output_paths = resolve_output_paths(output_root, batch_id)
    ensure_dir(output_paths["shots_dir"])
    ensure_dir(output_paths["logs_dir"])
    ensure_dir(output_paths["render_config_path"].parent)
    ensure_dir(output_paths["final_video"].parent)

    render_payload = build_batch_payload(
        args=args,
        batch_id=batch_id,
        caption_files=caption_files,
        output_paths=output_paths,
    )
    write_json(output_paths["render_config_path"], render_payload)

    assigned_shots = [int(item) for item in args.assigned_shots.split(",") if item.strip()]
    if not assigned_shots:
        raise ValueError("Worker mode requires --assigned-shots.")

    width, height = parse_resolution(args.resolution)
    print(
        f"[worker] batch={batch_id} gpu={args.gpu_slot} "
        f"shots={assigned_shots} steps={args.steps} cfg_scale={args.cfg_scale} "
        f"num_frames={render_payload['num_frames']}"
    )

    pipe = build_pipeline(model_dir=Path(args.model_dir).expanduser().resolve(), device="cuda")

    for shot_index in assigned_shots:
        shot = render_payload["shots"][shot_index]
        output_path = Path(shot["video_path"])
        if output_path.exists() and not args.overwrite:
            print(f"[worker] skip existing shot: {output_path}")
            continue

        shot_seed = args.seed + shot["shot"]
        print(f"[worker] rendering shot={shot['shot']} seed={shot_seed}")
        frames = pipe(
            prompt=shot["prompt"],
            negative_prompt=shot["negative_prompt"],
            seed=shot_seed,
            rand_device="cpu",
            height=height,
            width=width,
            num_frames=render_payload["num_frames"],
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            tiled=True,
            output_type="quantized",
            progress_bar_cmd=tqdm,
        )
        save_video(frames, str(output_path), fps=args.fps)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def orchestrate_batches(args: argparse.Namespace) -> None:
    batches = discover_caption_batches(args.caption_root)
    if not batches:
        raise FileNotFoundError(f"No caption batches found in {args.caption_root}")

    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root / "logs")
    ensure_dir(output_root / "render_configs")
    ensure_dir(output_root / "shots")
    ensure_dir(output_root / "video")

    index_items: list[dict[str, object]] = []
    for batch in batches:
        batch_id = batch["batch_id"]
        output_paths = resolve_output_paths(output_root, batch_id)
        payload = build_batch_payload(
            args=args,
            batch_id=batch_id,
            caption_files=batch["caption_files"],
            output_paths=output_paths,
        )
        write_json(output_paths["render_config_path"], payload)
        index_items.append(
            {
                "batch_id": batch_id,
                "caption_dir": str(batch["batch_dir"]),
                "render_config_path": str(output_paths["render_config_path"]),
                "shots_dir": str(output_paths["shots_dir"]),
                "video_path": str(output_paths["final_video"]),
                "num_shots": len(batch["caption_files"]),
                "duration_seconds": args.duration_seconds,
                "num_frames": payload["num_frames"],
            }
        )
    write_json(output_root / "index.json", {"items": index_items})

    gpu_ids = parse_gpu_ids(args.gpus)
    print(f"[orchestrate] batches={len(batches)} workers={len(gpu_ids)}")

    for batch in batches:
        batch_id = batch["batch_id"]
        output_paths = resolve_output_paths(output_root, batch_id)
        ensure_dir(output_paths["shots_dir"])
        ensure_dir(output_paths["logs_dir"])
        ensure_dir(output_paths["render_config_path"].parent)
        ensure_dir(output_paths["final_video"].parent)

        shot_count = len(batch["caption_files"])
        shot_groups = split_shots(shot_count, min(len(gpu_ids), shot_count))
        active_processes: list[dict] = []
        failed_logs: list[Path] = []

        for slot, shot_indices in enumerate(shot_groups):
            gpu_id = gpu_ids[slot]
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--caption-root",
                args.caption_root,
                "--model-dir",
                args.model_dir,
                "--output-root",
                args.output_root,
                "--gpus",
                args.gpus,
                "--steps",
                str(args.steps),
                "--cfg-scale",
                str(args.cfg_scale),
                "--fps",
                str(args.fps),
                "--duration-seconds",
                str(args.duration_seconds),
                "--resolution",
                args.resolution,
                "--seed",
                str(args.seed),
                "--batch-id",
                batch_id,
                "--assigned-shots",
                ",".join(str(index) for index in shot_indices),
                "--gpu-slot",
                gpu_id,
            ]
            if args.overwrite:
                command.append("--overwrite")

            launch_log_path = output_paths["logs_dir"] / "launch.log"
            worker_log_path = output_paths["logs_dir"] / f"worker_gpu{gpu_id}.log"
            write_launch_log(launch_log_path, command, gpu_id)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["OMP_NUM_THREADS"] = "1"

            with worker_log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(Path(__file__).resolve().parent),
                )

            active_processes.append(
                {
                    "process": process,
                    "gpu_id": gpu_id,
                    "log_path": worker_log_path,
                }
            )

        while active_processes:
            finished = wait_for_one(active_processes)
            if finished["return_code"] != 0:
                failed_logs.append(finished["log_path"])

        if failed_logs:
            failed_str = ", ".join(str(path) for path in failed_logs)
            raise RuntimeError(f"One or more workers failed. Check logs: {failed_str}")

        ordered_shots = [
            output_paths["shots_dir"] / f"shot_{shot_index:02d}.mp4"
            for shot_index in range(1, shot_count + 1)
        ]
        missing = [path for path in ordered_shots if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing rendered shots: {missing}")

        final_video = output_paths["final_video"]
        if final_video.exists() and not args.overwrite:
            print(f"[orchestrate] skip concat, final video exists: {final_video}")
            continue

        print(f"[orchestrate] concatenating batch={batch_id} -> {final_video}")
        concatenate_videos(ordered_shots, final_video, fps=args.fps)


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        worker_render(args)
        return

    orchestrate_batches(args)


if __name__ == "__main__":
    main()
