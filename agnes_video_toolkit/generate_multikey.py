# -*- coding: utf-8 -*-
"""
Agnes Video V2.0  多Key并行批量文生视频
=========================================
- 多个独立 API Key 并行工作，每个 Key 处理不同 prompt 子集（轮询分配）
- 每个 Key 有独立的 progress 文件，互不冲突
- 共享输出目录，按文件名检测已完成任务
- 自动提交 → 轮询 → 下载 → 去音频
- 速率：每个 Key 每 120 秒提交一次

用法：
  python generate_multikey.py -p prompts --api-keys key1,key2,key3 -o ./output
"""

# 修复 Windows cmd GBK 编码
import sys as _sys
if _sys.platform == "win32":
    import io
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import sys
import json
import time
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import requests

# ── 常量 ───────────────────────────────────────────────────
API_BASE    = "https://apihub.agnes-ai.com"
CREATE_URL  = f"{API_BASE}/v1/videos"
QUERY_URL   = f"{API_BASE}/agnesapi"
POLL_INTERVAL  = 10             # 轮询间隔（秒）
MAX_POLL_TIME  = 1800           # 单任务最长等待（秒）
WAIT_BETWEEN   = 120            # 同一 Key 两次提交间隔（秒）
NO_PROXY = {'http': None, 'https': None}  # 绕过本地代理

# ── ffmpeg 检测 ────────────────────────────────────────────
FFMPEG_OK = False
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    FFMPEG_OK = True
except Exception:
    pass


# ╔══════════════════════════════════════════════════════════╗
# ║                     Worker 线程                         ║
# ╚══════════════════════════════════════════════════════════╝

def worker(worker_id: int, api_key: str, prompts: list,
           output_dir: Path, video_kwargs: dict):
    """
    单个 worker 线程：
      prompts: [(name, text), ...]
      worker_id: 用于区分 progress 文件名
    """
    progress_file = output_dir / f"progress_w{worker_id}.json"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ── 加载 / 初始化进度 ──
    if progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
    else:
        progress = {"completed": [], "failed": [], "tasks": {}}

    def save():
        progress_file.write_text(json.dumps(progress, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    def remove_audio(raw: Path, final: Path) -> Path | None:
        """ffmpeg 去音频，不可用时返回 None"""
        if not FFMPEG_OK:
            print(f"  [W{worker_id}] ffmpeg 不可用，跳过")
            return None
        cmd = ["ffmpeg", "-y", "-i", str(raw), "-c:v", "copy", "-an", str(final)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  [W{worker_id}] ffmpeg warn: {proc.stderr[:200]}")
            return None
        print(f"  [W{worker_id}] 去音频完成 -> {final.name}")
        return final

    def retry_submit(payload, max_retries=5):
        """提交任务，429/503 自动等待重试"""
        for attempt in range(max_retries):
            try:
                resp = requests.post(CREATE_URL, headers=headers,
                                     json=payload, timeout=30, proxies=NO_PROXY)
                if resp.status_code == 429:
                    wait = 65  # 超过 1 分钟确保限制重置
                    print(f"  [W{worker_id}] 429 频率限制, 等 {wait}s 后重试 "
                          f"({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 503:
                    wait = 30
                    print(f"  [W{worker_id}] 503 服务繁忙, 等 {wait}s 后重试 "
                          f"({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 30
                    print(f"  [W{worker_id}] 请求失败: {e}, 等 {wait}s 重试 "
                          f"({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"提交失败，已重试 {max_retries} 次")

    def retry_query(video_id, max_retries=3):
        """查询任务，503 自动重试"""
        for attempt in range(max_retries):
            try:
                qresp = requests.get(
                    QUERY_URL, params={"video_id": video_id},
                    headers=headers, timeout=30, proxies=NO_PROXY)
                if qresp.status_code == 503:
                    time.sleep(15)
                    continue
                qresp.raise_for_status()
                return qresp.json()
            except requests.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(15)
                else:
                    raise

    # ── 错峰启动：每个 worker 延迟 worker_id * 5 秒 ──
    stagger = worker_id * 5
    if stagger > 0:
        print(f"  [W{worker_id}] 错峰延迟 {stagger}s...")
        time.sleep(stagger)

    # ── 主循环 ──
    total = len(prompts)
    for i, (name, prompt_text) in enumerate(prompts, 1):
        prompt_text = prompt_text.strip()
        output_path = output_dir / f"{name}.mp4"

        # 已完成检查：progress 中记录 OR 文件已存在
        if name in progress["completed"]:
            print(f"[W{worker_id}][{i}/{total}] SKIP {name} (progress)")
            continue
        if output_path.exists():
            print(f"[W{worker_id}][{i}/{total}] SKIP {name} (file exists)")
            # 补充记录到 progress
            progress["completed"].append(name)
            progress["tasks"][name] = {
                "prompt": prompt_text[:80],
                "output": str(output_path),
                "size": output_path.stat().st_size,
                "timestamp": datetime.now().isoformat(),
            }
            save()
            continue

        print(f"\n{'─' * 50}")
        print(f"[W{worker_id}][{i}/{total}] {name}")
        print(f"  prompt: {prompt_text[:90]}{'...' if len(prompt_text) > 90 else ''}")
        print(f"{'─' * 50}")

        try:
            # ① 提交任务
            payload = {
                "model": "agnes-video-v2.0",
                "prompt": prompt_text,
                "height": video_kwargs.get("height", 480),
                "width": video_kwargs.get("width", 832),
                "num_frames": video_kwargs.get("num_frames", 81),
                "frame_rate": video_kwargs.get("frame_rate", 15),
            }
            if video_kwargs.get("seed") is not None:
                payload["seed"] = video_kwargs["seed"]
            if video_kwargs.get("negative_prompt"):
                payload["negative_prompt"] = video_kwargs["negative_prompt"]

            print(f"  [W{worker_id}] ① 提交...")
            task_data = retry_submit(payload)
            video_id = task_data.get("video_id")
            snip = video_id[:40] if video_id else "N/A"
            print(f"  [W{worker_id}]    video_id={snip}...")

            # ② 轮询等待
            print(f"  [W{worker_id}] ② 等待生成...")
            start_ts = time.time()
            download_url = None
            while time.time() - start_ts < MAX_POLL_TIME:
                result = retry_query(video_id)
                if result is None:
                    time.sleep(POLL_INTERVAL)
                    continue

                st = result.get("status", "?")
                pg = result.get("progress", 0)
                print(f"  [W{worker_id}]   状态={st}, 进度={pg}%")

                if st == "completed":
                    # 实际 API 用 "url" 字段（非文档写的 remixed_from_video_id）
                    download_url = result.get("url") or result.get("remixed_from_video_id")
                    break
                elif st == "failed":
                    raise RuntimeError(
                        f"生成失败: {result.get('error', 'unknown')}"
                    )
                else:
                    time.sleep(POLL_INTERVAL)
            else:
                raise TimeoutError(f"任务超时 ({MAX_POLL_TIME}s)")

            if not download_url:
                raise RuntimeError("响应中无下载链接")

            # ③ 下载
            raw_path = output_dir / f"{name}_raw.mp4"
            print(f"  [W{worker_id}] ③ 下载...")
            dresp = requests.get(download_url, stream=True, timeout=600, proxies=NO_PROXY)
            dresp.raise_for_status()
            total_bytes = int(dresp.headers.get("content-length", 0))
            downloaded = 0
            with open(raw_path, "wb") as fout:
                for chunk in dresp.iter_content(8192):
                    fout.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        print(f"    下载进度: {pct:.0f}%", end="\r")
            print(f"\n  [W{worker_id}] 下载完成 -> {raw_path.name}")

            # ④ 去音频
            print(f"  [W{worker_id}] ④ 去音频...")
            removed = remove_audio(raw_path, output_path)
            if removed is None:
                if output_path.exists():
                    output_path.unlink()
                raw_path.rename(output_path)
            else:
                if raw_path.exists():
                    raw_path.unlink()

            # ⑤ 记录
            progress["completed"].append(name)
            progress["tasks"][name] = {
                "prompt": prompt_text,
                "video_id": video_id,
                "output": str(output_path),
                "size": output_path.stat().st_size,
                "timestamp": datetime.now().isoformat(),
            }
            save()
            print(f"  [W{worker_id}] [OK] {output_path.name}")

        except Exception as e:
            print(f"  [W{worker_id}] [FAIL] {e}")
            progress["failed"].append({
                "name": name,
                "prompt": prompt_text,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            save()

        # ── 速率控制（最后一条不用等）──
        if i < total:
            print(f"  [W{worker_id}] 等待 {WAIT_BETWEEN}s...")
            time.sleep(WAIT_BETWEEN)

    done = len(progress["completed"])
    fail = len(progress["failed"])
    print(f"\n[W{worker_id}] ====== 完成: {done}/{total}, 失败: {fail} ======")


# ╔══════════════════════════════════════════════════════════╗
# ║                      CLI 入口                           ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Agnes Video V2.0 多Key并行批量生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_multikey.py -p prompts --api-keys key1,key2,key3 -o ./output
  python generate_multikey.py -p prompts --api-keys key1,key2 --height 1080 --width 1920
        """,
    )
    parser.add_argument("-p", "--prompts", required=True,
                        help="Prompt 目录或文件路径")
    parser.add_argument("--api-keys", required=True,
                        help="逗号分隔的 API Key: key1,key2,key3")
    parser.add_argument("-o", "--output-dir", default="./output",
                        help="共享输出目录（默认: ./output）")
    # 视频参数（所有 Key 共享）
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81,
                        help="总帧数，8n+1，≤441（默认: 81）")
    parser.add_argument("--frame-rate", type=int, default=15)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--negative-prompt", default="")
    args = parser.parse_args()

    # ── API Keys ──
    api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    if not api_keys:
        print("[ERROR] 未提供 API Key")
        sys.exit(1)
    print(f"共加载 {len(api_keys)} 个 API Key")

    # ── 加载 Prompts ──
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"[ERROR] 路径不存在: {args.prompts}")
        sys.exit(1)

    if prompts_path.is_dir():
        txt_files = sorted(prompts_path.glob("*.txt"),
                           key=lambda p: p.stem)
        all_prompts = []
        for fp in txt_files:
            content = fp.read_text(encoding="utf-8").strip()
            if content:
                lines = [l for l in content.split("\n")
                         if l.strip() and not l.strip().startswith("#")]
                if lines:
                    all_prompts.append((fp.stem, lines[0]))
    else:
        lines = [l.strip() for l in
                 prompts_path.read_text(encoding="utf-8").split("\n")
                 if l.strip() and not l.strip().startswith("#")]
        all_prompts = [(f"prompt_{i:03d}", l) for i, l in enumerate(lines, 1)]

    if not all_prompts:
        print("[ERROR] 无有效 prompt")
        sys.exit(1)

    # ── 轮询分配到 Bucket ──
    num_keys = len(api_keys)
    buckets = [[] for _ in range(num_keys)]
    for idx, item in enumerate(all_prompts):
        buckets[idx % num_keys].append(item)

    print("=" * 55)
    print(f"  Agnes Video V2.0  多Key并行 ({num_keys} keys)")
    print("=" * 55)
    print(f"  总 Prompts      : {len(all_prompts)}")
    for ki, b in enumerate(buckets):
        names = [n for n, _ in b]
        print(f"  Key {ki+1}          : {len(b)} prompts ({names[0]} ... {names[-1]})")
    print(f"  输出目录        : {args.output_dir}")
    print(f"  分辨率          : {args.width}x{args.height}")
    print(f"  帧数x帧率       : {args.num_frames}x{args.frame_rate} "
          f"~{args.num_frames / args.frame_rate:.1f}s")
    if not FFMPEG_OK:
        print(f"  [INFO] ffmpeg 不可用, 保留原始音频")
    print("=" * 55)

    # ── 输出目录 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "seed": args.seed,
        "negative_prompt": args.negative_prompt,
    }

    # ── 启动线程 ──
    threads = []
    for ki, (api_key, bucket) in enumerate(zip(api_keys, buckets)):
        t = threading.Thread(
            target=worker,
            args=(ki, api_key, bucket, output_dir, video_kwargs),
            daemon=True,
            name=f"Worker-{ki}",
        )
        t.start()
        threads.append(t)
        print(f"  [W{ki}] 启动, {len(bucket)} prompts")

    print("=" * 55)
    print("  所有 Worker 已启动. Ctrl+C 中断, 进度不会丢失.")
    print("=" * 55)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[中断] 等待线程安全退出...")
        # daemon threads will exit when main exits

    print("\n" + "=" * 55)
    print("  全部完成!")
    print(f"  输出目录: {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
