#!/usr/bin/env python3
"""Generate storyboard videos with DashScope Wan2.7 and save to video/generated."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import imageio
import requests


CREATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
SUCCESS_STATUSES = {"SUCCEEDED", "SUCCESS", "COMPLETED"}
FAIL_STATUSES = {"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}
REQUEST_RETRY_COUNT = 5
REQUEST_RETRY_SLEEP_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render storyboard JSON files with DashScope Wan2.7, then concatenate shots."
    )
    parser.add_argument(
        "--storyboard-dir",
        default="/root/video/storyboards/advertising",
        help="包含 storyboard JSON 的目录。",
    )
    parser.add_argument("--start", type=int, default=6, help="起始编号，默认 6。")
    parser.add_argument("--end", type=int, default=20, help="结束编号，默认 20。")
    parser.add_argument(
        "--output-root",
        default="/root/video/generated",
        help="输出根目录，默认 /root/video/generated。",
    )
    parser.add_argument(
        "--batch-prefix",
        default="a",
        help="输出批次名前缀，默认 a；例如 live_action_film 可传 l 生成 l001。",
    )
    parser.add_argument("--env-file", default="/root/.env", help="保存 API key 的 .env 文件。")
    parser.add_argument("--model", default="wan2.7-t2v", help="DashScope 模型名。")
    parser.add_argument("--resolution", default="720P", help="DashScope resolution 参数。")
    parser.add_argument("--ratio", default="16:9", help="DashScope ratio 参数。")
    parser.add_argument("--duration", type=int, default=5, help="单个 shot 时长，默认 5 秒。")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=6,
        help="同时提交和轮询的最大 5 秒 shot 任务数，默认 6。",
    )
    parser.add_argument("--fps", type=int, default=30, help="拼接后视频帧率，默认 30。")
    parser.add_argument("--poll-interval", type=int, default=10, help="任务轮询间隔秒数。")
    parser.add_argument("--timeout", type=int, default=1800, help="单个任务超时秒数。")
    parser.add_argument(
        "--prompt-extend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启 prompt_extend，默认开启。",
    )
    parser.add_argument(
        "--watermark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启 watermark，默认开启。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 shot 和最终视频。",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key(env_file: Path) -> str:
    load_env_file(env_file)
    for name in ("DASHSCOPE_API_KEY", "Wan2.7_API_KEY", "WAN27_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "Missing API key. Put DASHSCOPE_API_KEY or Wan2.7_API_KEY in the .env file."
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_storyboard(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compose_prompt(storyboard: dict, shot: dict) -> str:
    defaults = storyboard.get("global_defaults", {})
    style = defaults.get("style", "").strip()
    prompt = shot["prompt"].strip()
    if style:
        prompt = f"{prompt} {style}"
    return " ".join(prompt.split())


def storyboard_number(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer storyboard number from {path}")
    return f"{int(match.group(1)):03d}"


def resolve_output_paths(
    output_root: Path,
    storyboard_path: Path,
    batch_prefix: str,
) -> dict[str, Path | str]:
    number = storyboard_number(storyboard_path)
    batch = f"{batch_prefix}{number}"
    return {
        "batch": batch,
        "shots_dir": output_root / "shots" / batch,
        "logs_dir": output_root / "logs" / batch,
        "render_config_path": output_root / "render_configs" / f"{batch}.json",
        "final_video": output_root / "video" / f"{batch}.mp4",
    }


def dashscope_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }


def request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_RETRY_COUNT + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code < 500:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        except requests.RequestException as exc:
            last_error = exc

        if attempt < REQUEST_RETRY_COUNT:
            sleep_seconds = REQUEST_RETRY_SLEEP_SECONDS * attempt
            print(
                f"[warn] {method.upper()} {url} failed on attempt "
                f"{attempt}/{REQUEST_RETRY_COUNT}: {last_error}; retrying in {sleep_seconds}s"
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"{method.upper()} {url} failed after {REQUEST_RETRY_COUNT} attempts: {last_error}")


def create_task(
    *,
    api_key: str,
    model: str,
    prompt: str,
    resolution: str,
    ratio: str,
    duration: int,
    prompt_extend: bool,
    watermark: bool,
) -> str:
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "resolution": resolution,
            "ratio": ratio,
            "prompt_extend": prompt_extend,
            "watermark": watermark,
            "duration": duration,
        },
    }
    response = request_with_retries(
        "post",
        CREATE_URL,
        headers=dashscope_headers(api_key),
        json=payload,
        timeout=60,
    )
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if response.status_code >= 400:
        raise RuntimeError(f"Task creation failed: HTTP {response.status_code} {data}")

    task_id = data.get("output", {}).get("task_id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Task creation response did not include task_id: {data}")
    return task_id


def get_task_data(api_key: str, task_id: str) -> dict:
    response = request_with_retries(
        "get",
        TASK_URL.format(task_id=task_id),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise
    if response.status_code >= 400:
        raise RuntimeError(f"Task polling failed: HTTP {response.status_code} {data}")
    return data


def poll_task(api_key: str, task_id: str, poll_interval: int, timeout: int) -> dict:
    deadline = time.time() + timeout
    last_data: dict = {}
    while time.time() < deadline:
        data = get_task_data(api_key, task_id)
        last_data = data
        status = str(data.get("output", {}).get("task_status", "")).upper()
        if status in SUCCESS_STATUSES:
            return data
        if status in FAIL_STATUSES:
            raise RuntimeError(f"Task {task_id} failed: {data}")

        print(f"[poll] task={task_id} status={status or 'UNKNOWN'}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Task {task_id} timed out after {timeout}s. Last response: {last_data}")


def extract_video_url(task_data: dict) -> str:
    output = task_data.get("output", {})
    for key in ("video_url", "url"):
        value = output.get(key)
        if value:
            return value
    results = output.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                value = item.get("video_url") or item.get("url")
                if value:
                    return value
    raise RuntimeError(f"Task succeeded but no video_url was found: {task_data}")


def download_video(url: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with request_with_retries("get", url, stream=True, timeout=300) as response:
        response.raise_for_status()
        tmp_path = output_path.with_suffix(output_path.suffix + ".part")
        with tmp_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
        tmp_path.replace(output_path)


def concatenate_videos(input_paths: Iterable[Path], output_path: Path, fps: int) -> None:
    ensure_dir(output_path.parent)
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


def render_shot(
    *,
    args: argparse.Namespace,
    api_key: str,
    storyboard: dict,
    shot: dict,
    shot_path: Path,
    log_path: Path,
) -> None:
    if shot_path.exists() and not args.overwrite:
        print(f"[shot] skip existing {shot_path}")
        return

    prompt = compose_prompt(storyboard, shot)
    print(f"[shot] create storyboard={storyboard['id']} shot={shot['shot']} -> {shot_path}")
    task_id = create_task(
        api_key=api_key,
        model=args.model,
        prompt=prompt,
        resolution=args.resolution,
        ratio=args.ratio,
        duration=int(shot.get("duration_seconds", args.duration)),
        prompt_extend=args.prompt_extend,
        watermark=args.watermark,
    )
    task_data = poll_task(api_key, task_id, args.poll_interval, args.timeout)
    video_url = extract_video_url(task_data)
    download_video(video_url, shot_path)
    write_json(
        log_path,
        {
            "task_id": task_id,
            "shot": shot["shot"],
            "prompt": prompt,
            "output_path": str(shot_path),
            "task_response": task_data,
        },
    )


def render_shots_concurrent(
    *,
    args: argparse.Namespace,
    api_key: str,
    storyboard: dict,
    shots_dir: Path,
    logs_dir: Path,
) -> None:
    pending = []
    for shot in storyboard["shots"]:
        shot_path = shots_dir / f"shot_{shot['shot']:02d}.mp4"
        log_path = logs_dir / f"shot_{shot['shot']:02d}.json"
        if shot_path.exists() and not args.overwrite:
            print(f"[shot] skip existing {shot_path}")
            continue
        prompt = compose_prompt(storyboard, shot)
        existing_task_id = ""
        if log_path.exists() and not args.overwrite:
            try:
                existing_log = read_json(log_path)
                existing_task_id = str(existing_log.get("task_id", "")).strip()
            except Exception as exc:
                print(f"[warn] could not read existing log {log_path}: {exc}")
        pending.append(
            {
                "shot": shot,
                "shot_path": shot_path,
                "log_path": log_path,
                "prompt": prompt,
                "task_id": existing_task_id,
            }
        )

    active: list[dict] = []
    max_concurrent = max(1, args.max_concurrent)
    deadline_by_task: dict[str, float] = {}

    def submit_until_full() -> None:
        while pending and len(active) < max_concurrent:
            item = pending.pop(0)
            shot = item["shot"]
            task_id = item.get("task_id", "")
            if task_id:
                print(
                    f"[shot] resume storyboard={storyboard['id']} shot={shot['shot']} "
                    f"task={task_id} -> {item['shot_path']}"
                )
            else:
                print(
                    f"[shot] create storyboard={storyboard['id']} shot={shot['shot']} "
                    f"-> {item['shot_path']}"
                )
                task_id = create_task(
                    api_key=api_key,
                    model=args.model,
                    prompt=item["prompt"],
                    resolution=args.resolution,
                    ratio=args.ratio,
                    duration=int(shot.get("duration_seconds", args.duration)),
                    prompt_extend=args.prompt_extend,
                    watermark=args.watermark,
                )
                write_json(
                    item["log_path"],
                    {
                        "task_id": task_id,
                        "shot": shot["shot"],
                        "prompt": item["prompt"],
                        "output_path": str(item["shot_path"]),
                        "status": "SUBMITTED",
                    },
                )
            item["task_id"] = task_id
            active.append(item)
            deadline_by_task[task_id] = time.time() + args.timeout

    submit_until_full()
    while active:
        for item in list(active):
            task_id = item["task_id"]
            if time.time() > deadline_by_task[task_id]:
                raise TimeoutError(f"Task {task_id} timed out after {args.timeout}s.")

            task_data = get_task_data(api_key, task_id)
            status = str(task_data.get("output", {}).get("task_status", "")).upper()
            shot_number = item["shot"]["shot"]
            if status in SUCCESS_STATUSES:
                video_url = extract_video_url(task_data)
                download_video(video_url, item["shot_path"])
                write_json(
                    item["log_path"],
                    {
                        "task_id": task_id,
                        "shot": shot_number,
                        "prompt": item["prompt"],
                        "output_path": str(item["shot_path"]),
                        "task_response": task_data,
                    },
                )
                active.remove(item)
                print(f"[shot] done storyboard={storyboard['id']} shot={shot_number}")
                submit_until_full()
            elif status in FAIL_STATUSES:
                raise RuntimeError(f"Task {task_id} failed: {task_data}")
            else:
                print(
                    f"[poll] storyboard={storyboard['id']} shot={shot_number} "
                    f"task={task_id} status={status or 'UNKNOWN'}"
                )
        if active:
            time.sleep(args.poll_interval)


def render_storyboard(args: argparse.Namespace, api_key: str, storyboard_path: Path) -> None:
    storyboard = load_storyboard(storyboard_path)
    output_root = Path(args.output_root).expanduser().resolve()
    paths = resolve_output_paths(output_root, storyboard_path, args.batch_prefix.strip())
    shots_dir = Path(paths["shots_dir"])
    logs_dir = Path(paths["logs_dir"])
    final_video = Path(paths["final_video"])
    ensure_dir(shots_dir)
    ensure_dir(logs_dir)
    ensure_dir(final_video.parent)

    write_json(
        Path(paths["render_config_path"]),
        {
            "storyboard_path": str(storyboard_path),
            "storyboard_id": storyboard["id"],
            "backend": "dashscope",
            "model": args.model,
            "resolution": args.resolution,
            "ratio": args.ratio,
            "duration": args.duration,
            "prompt_extend": args.prompt_extend,
            "watermark": args.watermark,
            "overwrite": args.overwrite,
            "render_batch": paths["batch"],
            "shots_dir": str(shots_dir),
            "video_path": str(final_video),
        },
    )

    print(f"[storyboard] {storyboard_path.name} id={storyboard['id']} batch={paths['batch']}")
    render_shots_concurrent(
        args=args,
        api_key=api_key,
        storyboard=storyboard,
        shots_dir=shots_dir,
        logs_dir=logs_dir,
    )

    ordered_shots = [shots_dir / f"shot_{shot['shot']:02d}.mp4" for shot in storyboard["shots"]]
    missing = [str(path) for path in ordered_shots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing rendered shots: {missing}")

    if final_video.exists() and not args.overwrite:
        print(f"[storyboard] skip concat, final exists {final_video}")
        return
    print(f"[storyboard] concatenate -> {final_video}")
    concatenate_videos(ordered_shots, final_video, fps=args.fps)


def main() -> None:
    args = parse_args()
    storyboard_dir = Path(args.storyboard_dir).expanduser().resolve()
    api_key = get_api_key(Path(args.env_file).expanduser().resolve())

    for number in range(args.start, args.end + 1):
        storyboard_path = storyboard_dir / f"{number:03d}.json"
        if not storyboard_path.exists():
            raise FileNotFoundError(f"Storyboard not found: {storyboard_path}")
        render_storyboard(args, api_key, storyboard_path)


if __name__ == "__main__":
    main()
