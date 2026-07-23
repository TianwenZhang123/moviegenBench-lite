#!/usr/bin/env python3
"""根据 storyboard JSON 调用 Wan 逐分镜生成视频，并自动拼接成整条成片。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# 某些运行环境会注入非法的 OMP_NUM_THREADS，先兜底为 1，避免 libgomp 警告。
if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import imageio
import torch
from tqdm import tqdm

from diffsynth import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read storyboard JSON files and render each shot with Wan before concatenating them."
    )
    parser.add_argument(
        "--storyboard",
        required=True,
        help="单个 storyboard JSON，或者包含多个 JSON 的目录。",
    )
    parser.add_argument(
        "--model-dir",
        default="/root/autodl-tmp/model/Wan2.1-T2V-14B",
        help="Wan2.1-T2V-14B 模型目录。",
    )
    parser.add_argument(
        "--output-root",
        default="/root/video/generated",
        help="所有生成结果的根目录。",
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
        "--num-frames",
        type=int,
        default=81,
        help="每个分镜生成帧数。Wan 常用 81 帧，对应约 5 秒。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子；每个分镜会在此基础上递增。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="已存在的分镜和拼接结果是否覆盖。",
    )
    parser.add_argument(
        "--render-batch",
        default="",
        help="可选的批次编号，例如 001/002；传入后会按旧目录结构输出。",
    )
    parser.add_argument(
        "--mode",
        choices=("orchestrate", "worker"),
        default="orchestrate",
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


def discover_storyboards(storyboard_arg: str) -> list[Path]:
    path = Path(storyboard_arg).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    raise FileNotFoundError(f"Storyboard path does not exist: {path}")


def parse_gpu_ids(gpus_arg: str) -> list[str]:
    gpu_ids = [item.strip() for item in gpus_arg.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided.")
    return gpu_ids


def parse_resolution(resolution: str) -> tuple[int, int]:
    width_str, height_str = resolution.lower().split("x", 1)
    return int(width_str), int(height_str)


def load_storyboard(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compose_prompt(storyboard: dict, shot: dict) -> tuple[str, str]:
    defaults = storyboard.get("global_defaults", {})
    style = defaults.get("style", "").strip()
    negative = defaults.get("negative_prompt", "").strip()
    prompt = shot["prompt"].strip()
    if style:
        prompt = f"{prompt} {style}"
    return prompt, negative


def build_model_configs(model_dir: Path) -> tuple[list[ModelConfig], ModelConfig]:
    # 这里全部走本地路径，避免脚本运行时误触发远程下载。
    dit_shards = sorted(str(path) for path in model_dir.glob("diffusion_pytorch_model*.safetensors"))
    if not dit_shards:
        raise FileNotFoundError(f"No DiT safetensor shards found in {model_dir}")

    text_encoder = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = model_dir / "Wan2.1_VAE.pth"
    tokenizer_dir = model_dir / "google" / "umt5-xxl"

    for required in (text_encoder, vae, tokenizer_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required Wan asset: {required}")

    model_configs = [
        ModelConfig(path=str(text_encoder)),
        ModelConfig(path=str(vae)),
        ModelConfig(path=dit_shards),
    ]
    tokenizer_config = ModelConfig(path=str(tokenizer_dir))
    return model_configs, tokenizer_config


def split_shots(shot_count: int, worker_count: int) -> list[list[int]]:
    # 按轮询方式把分镜打散到多张卡，尽量让两张卡都持续工作。
    buckets: list[list[int]] = [[] for _ in range(worker_count)]
    for shot_index in range(shot_count):
        buckets[shot_index % worker_count].append(shot_index)
    return [bucket for bucket in buckets if bucket]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def concatenate_videos(input_paths: Iterable[Path], output_path: Path, fps: int) -> None:
    # 不依赖系统 ffmpeg，直接用 imageio 逐段读取并顺序写出。
    writer = imageio.get_writer(output_path, fps=fps, quality=8)
    try:
        for input_path in input_paths:
            reader = imageio.get_reader(input_path)
            try:
                for frame in reader:
                    writer.append_data(frame)
            finally:
                reader.close()
    finally:
        writer.close()


def storyboard_output_dir(output_root: Path, storyboard: dict) -> Path:
    return output_root / storyboard["id"]


def resolve_output_paths(output_root: Path, storyboard: dict, render_batch: str) -> dict[str, Path]:
    """统一管理输出路径。

    1. 默认沿用当前脚本的 storyboard-id 目录结构。
    2. 如果传入 render_batch，则兼容 001 的旧目录布局，方便后续批量生成 002、003。
    """

    if render_batch:
        batch = render_batch.strip()
        return {
            "shots_dir": output_root / "shots" / batch,
            "logs_dir": output_root / "logs" / batch,
            "render_config_path": output_root / "render_configs" / f"{batch}.json",
            "final_video": output_root / "video" / f"{batch}.mp4",
        }

    output_dir = storyboard_output_dir(output_root, storyboard)
    return {
        "shots_dir": output_dir / "shots",
        "logs_dir": output_dir / "logs",
        "render_config_path": output_dir / "render_config.json",
        "final_video": output_dir / f"{storyboard['id']}_final.mp4",
    }


def orchestrate_storyboard(args: argparse.Namespace, storyboard_path: Path) -> None:
    storyboard = load_storyboard(storyboard_path)
    output_root = Path(args.output_root).resolve()
    output_paths = resolve_output_paths(output_root, storyboard, args.render_batch)
    shots_dir = output_paths["shots_dir"]
    logs_dir = output_paths["logs_dir"]
    ensure_dir(shots_dir)
    ensure_dir(logs_dir)
    ensure_dir(output_paths["render_config_path"].parent)
    ensure_dir(output_paths["final_video"].parent)

    # 记录一次本次任务配置，方便后续复现。
    write_json(
        output_paths["render_config_path"],
        {
            "storyboard_path": str(storyboard_path),
            "storyboard_id": storyboard["id"],
            "model_dir": str(Path(args.model_dir).resolve()),
            "gpus": parse_gpu_ids(args.gpus),
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "num_frames": args.num_frames,
            "seed": args.seed,
            "overwrite": args.overwrite,
            "render_batch": args.render_batch,
            "shots_dir": str(shots_dir),
            "video_path": str(output_paths["final_video"]),
        },
    )

    shot_count = len(storyboard["shots"])
    gpu_ids = parse_gpu_ids(args.gpus)
    worker_count = min(len(gpu_ids), shot_count)
    shot_groups = split_shots(shot_count, worker_count)

    print(f"[orchestrate] storyboard={storyboard['id']} shots={shot_count} workers={worker_count}")

    processes: list[tuple[subprocess.Popen, Path]] = []
    for slot, shot_indices in enumerate(shot_groups):
        gpu_id = gpu_ids[slot]
        log_path = logs_dir / f"worker_gpu{gpu_id}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--storyboard",
            str(storyboard_path),
            "--model-dir",
            args.model_dir,
            "--output-root",
            args.output_root,
            "--steps",
            str(args.steps),
            "--cfg-scale",
            str(args.cfg_scale),
            "--num-frames",
            str(args.num_frames),
            "--seed",
            str(args.seed),
            "--gpus",
            args.gpus,
            "--render-batch",
            args.render_batch,
            "--assigned-shots",
            ",".join(str(index) for index in shot_indices),
            "--gpu-slot",
            gpu_id,
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        env = os.environ.copy()
        # 通过 CUDA_VISIBLE_DEVICES 把每个 worker 固定到单独一张卡上。
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["OMP_NUM_THREADS"] = "1"

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path(__file__).resolve().parent),
            )
        processes.append((process, log_path))

    failed_logs: list[Path] = []
    for process, log_path in processes:
        return_code = process.wait()
        if return_code != 0:
            failed_logs.append(log_path)

    if failed_logs:
        failed_str = ", ".join(str(path) for path in failed_logs)
        raise RuntimeError(f"One or more workers failed. Check logs: {failed_str}")

    fps = int(storyboard.get("global_defaults", {}).get("fps", 16))
    ordered_shots = [shots_dir / f"shot_{shot['shot']:02d}.mp4" for shot in storyboard["shots"]]
    missing = [path for path in ordered_shots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing rendered shots: {missing}")

    final_video = output_paths["final_video"]
    if final_video.exists() and not args.overwrite:
        print(f"[orchestrate] skip concat, final video exists: {final_video}")
        return

    print(f"[orchestrate] concatenating -> {final_video}")
    concatenate_videos(ordered_shots, final_video, fps=fps)


def worker_render(args: argparse.Namespace) -> None:
    storyboard_path = Path(args.storyboard).expanduser().resolve()
    storyboard = load_storyboard(storyboard_path)
    output_paths = resolve_output_paths(Path(args.output_root).resolve(), storyboard, args.render_batch)
    shots_dir = output_paths["shots_dir"]
    ensure_dir(shots_dir)

    assigned_shots = [int(item) for item in args.assigned_shots.split(",") if item.strip()]
    if not assigned_shots:
        raise ValueError("Worker mode requires --assigned-shots.")

    defaults = storyboard.get("global_defaults", {})
    fps = int(defaults.get("fps", 16))
    width, height = parse_resolution(defaults.get("resolution", "832x480"))
    model_dir = Path(args.model_dir).expanduser().resolve()

    # 开启 TF32 可稍微提升大模型推理吞吐。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_configs, tokenizer_config = build_model_configs(model_dir)

    print(
        f"[worker] storyboard={storyboard['id']} gpu={args.gpu_slot} "
        f"shots={assigned_shots} steps={args.steps} cfg_scale={args.cfg_scale}"
    )

    # 每个 worker 只在自己的 GPU 上加载一次模型，避免每个分镜重复初始化。
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )

    for shot_index in assigned_shots:
        shot = storyboard["shots"][shot_index]
        output_path = shots_dir / f"shot_{shot['shot']:02d}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"[worker] skip existing shot: {output_path}")
            continue

        prompt, negative_prompt = compose_prompt(storyboard, shot)
        seed = args.seed + shot["shot"]
        print(f"[worker] rendering shot={shot['shot']} seed={seed}")

        # 这里直接生成一整个 5s 分镜，符合“逐分镜生成，再统一拼接”的工作流。
        frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            rand_device="cpu",
            height=height,
            width=width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            tiled=True,
            output_type="quantized",
            progress_bar_cmd=tqdm,
        )

        save_video(frames, str(output_path), fps=fps)

        # 每个分镜结束后释放缓存，避免长任务里显存碎片越积越多。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        worker_render(args)
        return

    storyboards = discover_storyboards(args.storyboard)
    for storyboard_path in storyboards:
        orchestrate_storyboard(args, storyboard_path)


if __name__ == "__main__":
    main()
