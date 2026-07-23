#!/usr/bin/env python3
"""根据 caption 直接生成 30 秒单段视频，并保持与 generate_wan 一致的输出结构。"""

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

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video

from generate_wan_storyboard import build_model_configs, parse_gpu_ids, parse_resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read caption txt files and render one 30-second video for each caption."
    )
    parser.add_argument(
        "--caption-dir",
        default="/root/baseline/caption",
        help="单个 caption txt，或者包含多个 txt 的目录。",
    )
    parser.add_argument(
        "--model-dir",
        default="/root/autodl-tmp/model/Wan2.1-T2V-14B",
        help="Wan2.1-T2V-14B 模型目录。",
    )
    parser.add_argument(
        "--output-root",
        default="/root/baseline/gen-video",
        help="所有生成结果的根目录。",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="用于生成的 GPU 列表，逗号分隔；例如 0,1。",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="只处理指定 caption id，支持逗号和连续范围；例如 a005-a010 或 a005,a007。",
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
        default=30,
        help="目标视频时长，默认 30 秒。",
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
        help="基础随机种子；每个视频会在此基础上递增。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="已存在的视频是否覆盖。",
    )
    parser.add_argument(
        "--render-batch",
        default="",
        help="可选的批次编号；仅在输入单个 caption 文件时生效。",
    )
    parser.add_argument(
        "--mode",
        choices=("orchestrate", "worker"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpu-slot",
        default="0",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def split_id_parts(value: str) -> tuple[str, str]:
    prefix = value.rstrip("0123456789")
    digits = value[len(prefix):]
    return prefix, digits


def expand_id_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []

    if "-" not in token:
        return [token]

    start, end = [part.strip() for part in token.split("-", 1)]
    start_prefix, start_digits = split_id_parts(start)
    end_prefix, end_digits = split_id_parts(end)
    if (
        not start_digits
        or not end_digits
        or start_prefix != end_prefix
        or len(start_digits) != len(end_digits)
    ):
        return [token]

    start_num = int(start_digits)
    end_num = int(end_digits)
    step = 1 if end_num >= start_num else -1
    width = len(start_digits)
    return [
        f"{start_prefix}{number:0{width}d}"
        for number in range(start_num, end_num + step, step)
    ]


def parse_ids(ids_arg: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for token in ids_arg.split(","):
        for item in expand_id_token(token):
            if item not in seen:
                ids.append(item)
                seen.add(item)
    return ids


def filter_paths_by_ids(paths: list[Path], ids_arg: str, kind: str) -> list[Path]:
    target_ids = parse_ids(ids_arg)
    if not target_ids:
        return paths

    by_stem = {path.stem: path for path in paths}
    missing = [item for item in target_ids if item not in by_stem]
    if missing:
        raise FileNotFoundError(f"Missing {kind} ids: {', '.join(missing)}")
    return [by_stem[item] for item in target_ids]


def discover_caption_files(caption_arg: str, ids_arg: str = "") -> list[Path]:
    path = Path(caption_arg).expanduser().resolve()
    if path.is_file():
        return filter_paths_by_ids([path], ids_arg, "caption")
    if path.is_dir():
        caption_files = sorted(path.glob("*.txt"))
        return filter_paths_by_ids(caption_files, ids_arg, "caption")
    raise FileNotFoundError(f"Caption path does not exist: {path}")


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
 

def compute_num_frames(fps: int, duration_seconds: int) -> int:
    # 现有 5 秒工作流使用 81 帧，对应 16 * 5 + 1，因此这里沿用同一换算方式。
    return fps * duration_seconds + 1


def resolve_batch_id(caption_file: Path, render_batch: str, total_count: int) -> str:
    if render_batch.strip() and total_count == 1:
        return render_batch.strip()
    return caption_file.stem


def resolve_output_paths(output_root: Path, batch_id: str) -> dict[str, Path]:
    return {
        "logs_dir": output_root / "logs" / batch_id,
        "render_config_path": output_root / "render_configs" / f"{batch_id}.json",
        "final_video": output_root / "video" / f"{batch_id}.mp4",
    }


def build_render_payload(
    *,
    args: argparse.Namespace,
    caption_file: Path,
    batch_id: str,
    seed: int,
    gpu_id: str,
    output_paths: dict[str, Path],
) -> dict:
    caption = sanitize_caption(caption_file.read_text(encoding="utf-8"))
    return {
        "caption_file": str(caption_file),
        "batch_id": batch_id,
        "caption": caption,
        "prompt": build_prompt(caption),
        "negative_prompt": build_negative_prompt(),
        "model_dir": str(Path(args.model_dir).resolve()),
        "gpus": parse_gpu_ids(args.gpus),
        "assigned_gpu": gpu_id,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "fps": args.fps,
        "duration_seconds": args.duration_seconds,
        "num_frames": compute_num_frames(args.fps, args.duration_seconds),
        "resolution": args.resolution,
        "seed": seed,
        "overwrite": args.overwrite,
        "video_path": str(output_paths["final_video"]),
        "logs_dir": str(output_paths["logs_dir"]),
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


def build_pipeline(model_dir: Path, device: str) -> WanVideoPipeline:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model_configs, tokenizer_config = build_model_configs(model_dir)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )


def worker_render(args: argparse.Namespace) -> None:
    caption_files = discover_caption_files(args.caption_dir)
    if len(caption_files) != 1:
        raise ValueError("Worker mode requires exactly one caption file.")

    caption_file = caption_files[0]
    batch_id = resolve_batch_id(caption_file, args.render_batch, total_count=1)
    output_root = Path(args.output_root).resolve()
    output_paths = resolve_output_paths(output_root, batch_id)
    ensure_dir(output_paths["logs_dir"])
    ensure_dir(output_paths["render_config_path"].parent)
    ensure_dir(output_paths["final_video"].parent)

    gpu_id = args.gpu_slot
    seed = args.seed
    render_payload = build_render_payload(
        args=args,
        caption_file=caption_file,
        batch_id=batch_id,
        seed=seed,
        gpu_id=gpu_id,
        output_paths=output_paths,
    )
    write_json(output_paths["render_config_path"], render_payload)

    final_video = output_paths["final_video"]
    if final_video.exists() and not args.overwrite:
        print(f"[worker] skip existing video: {final_video}")
        return

    width, height = parse_resolution(args.resolution)
    print(
        f"[worker] batch={batch_id} gpu={gpu_id} "
        f"steps={args.steps} cfg_scale={args.cfg_scale} num_frames={render_payload['num_frames']}"
    )

    pipe = build_pipeline(model_dir=Path(args.model_dir).expanduser().resolve(), device="cuda")
    frames = pipe(
        prompt=render_payload["prompt"],
        negative_prompt=render_payload["negative_prompt"],
        seed=seed,
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
    save_video(frames, str(final_video), fps=args.fps)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def orchestrate_captions(args: argparse.Namespace) -> None:
    caption_files = discover_caption_files(args.caption_dir, args.ids)
    if not caption_files:
        raise FileNotFoundError(f"No caption txt files found in {args.caption_dir}")

    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root / "logs")
    ensure_dir(output_root / "render_configs")
    ensure_dir(output_root / "video")

    gpu_ids = parse_gpu_ids(args.gpus)
    print(f"[orchestrate] captions={len(caption_files)} workers={len(gpu_ids)}")

    index_items: list[dict[str, str | int]] = []
    for index, caption_file in enumerate(caption_files):
        batch_id = resolve_batch_id(caption_file, args.render_batch, total_count=len(caption_files))
        output_paths = resolve_output_paths(output_root, batch_id)
        index_items.append(
            {
                "batch_id": batch_id,
                "caption_file": str(caption_file),
                "render_config_path": str(output_paths["render_config_path"]),
                "video_path": str(output_paths["final_video"]),
                "duration_seconds": args.duration_seconds,
                "num_frames": compute_num_frames(args.fps, args.duration_seconds),
                "seed": args.seed + index + 1,
            }
        )
    write_json(output_root / "index.json", {"items": index_items})

    available_gpus = gpu_ids.copy()
    active_processes: list[dict] = []
    failed_logs: list[Path] = []

    for index, caption_file in enumerate(caption_files):
        batch_id = resolve_batch_id(caption_file, args.render_batch, total_count=len(caption_files))
        output_paths = resolve_output_paths(output_root, batch_id)
        ensure_dir(output_paths["logs_dir"])
        ensure_dir(output_paths["render_config_path"].parent)
        ensure_dir(output_paths["final_video"].parent)

        while not available_gpus:
            finished = wait_for_one(active_processes)
            available_gpus.append(finished["gpu_id"])
            if finished["return_code"] != 0:
                failed_logs.append(finished["log_path"])

        gpu_id = available_gpus.pop(0)
        seed = args.seed + index + 1
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--caption-dir",
            str(caption_file),
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
            str(seed),
            "--render-batch",
            batch_id,
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
        available_gpus.append(finished["gpu_id"])
        if finished["return_code"] != 0:
            failed_logs.append(finished["log_path"])

    if failed_logs:
        failed_str = ", ".join(str(path) for path in failed_logs)
        raise RuntimeError(f"One or more workers failed. Check logs: {failed_str}")


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        worker_render(args)
        return

    orchestrate_captions(args)


if __name__ == "__main__":
    main()
