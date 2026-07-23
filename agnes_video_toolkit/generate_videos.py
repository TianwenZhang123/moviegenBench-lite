# -*- coding: utf-8 -*-
"""
Agnes Video V2.0 批量文生视频工具
===================================
功能：
  1. 从 prompt 文件读取批量提示词
  2. 异步提交视频生成任务 (POST /v1/videos)
  3. 轮询任务状态直到完成 (GET /agnesapi)
  4. 下载生成的视频
  5. 使用 ffmpeg 去除音频（输出无声视频）
  6. 速率控制：默认每次请求后等待 120 秒
  7. 断点续传：progress.json 记录已完成/失败任务

用法：
  python generate_videos.py -p prompts.txt -k YOUR_API_KEY -o ./output
"""
# 修复 Windows cmd GBK 编码问题
import sys as _sys
if _sys.platform == "win32":
    import io
    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import requests

# ── API 配置 ──────────────────────────────────────────────

# ── ffmpeg 全局检测 ───────────────────────────────────────
_FFMPEG_AVAILABLE = False
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    _FFMPEG_AVAILABLE = True
except (FileNotFoundError, subprocess.CalledProcessError):
    pass
API_BASE = "https://apihub.agnes-ai.com"
CREATE_URL = f"{API_BASE}/v1/videos"
QUERY_URL = f"{API_BASE}/agnesapi"

POLL_INTERVAL = 10       # 轮询间隔（秒）
MAX_POLL_TIME = 1800     # 单个任务最大等待时间（秒，30分钟）


# ╔══════════════════════════════════════════════════════════╗
# ║                    AgnesVideoGenerator                   ║
# ╚══════════════════════════════════════════════════════════╝

class AgnesVideoGenerator:
    """封装 Agnes Video V2.0 API 的批量视频生成器"""

    ffmpeg_available = _FFMPEG_AVAILABLE

    def __init__(self, api_key: str, output_dir: str, wait_seconds: int = 120,
                 keep_audio: bool = False):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.wait_seconds = wait_seconds
        self.keep_audio = keep_audio
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 进度持久化 ──
        self.progress_file = self.output_dir / "progress.json"
        self.progress = self._load_progress()

    # ── 进度管理 ──────────────────────────────────────────

    def _load_progress(self) -> dict:
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"completed": [], "failed": [], "tasks": {}}

    def _save_progress(self):
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.md5(prompt.strip().encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _output_name(prompt_hash: str, name: str = None) -> str:
        """返回用于输出文件名的标识符，优先使用传入的 name"""
        return name if name else prompt_hash

    # ── API 调用 ──────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def submit_task(self, prompt: str, **kwargs) -> dict:
        """提交文生视频任务，返回包含 video_id / task_id 的响应 dict"""
        payload = {
            "model": "agnes-video-v2.0",
            "prompt": prompt.strip(),
            "height": kwargs.get("height", 480),
            "width": kwargs.get("width", 832),
            "num_frames": kwargs.get("num_frames", 81),
            "frame_rate": kwargs.get("frame_rate", 15),
        }
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]
        if kwargs.get("negative_prompt"):
            payload["negative_prompt"] = kwargs["negative_prompt"]

        resp = requests.post(CREATE_URL, headers=self._headers(),
                             json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def query_task(self, video_id: str) -> dict:
        """通过 video_id 查询任务状态"""
        resp = requests.get(QUERY_URL, params={"video_id": video_id},
                            headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def poll_until_complete(self, video_id: str,
                            max_wait: int = MAX_POLL_TIME,
                            poll_interval: int = POLL_INTERVAL) -> dict:
        """轮询直到 completed / failed / 超时"""
        short_id = video_id[:24]
        start = time.time()
        while time.time() - start < max_wait:
            try:
                result = self.query_task(video_id)
                status = result.get("status")
                progress = result.get("progress", 0)
                print(f"  [{short_id}] 状态: {status}, 进度: {progress}%")

                if status == "completed":
                    return result
                if status == "failed":
                    error = result.get("error", "未知错误")
                    raise RuntimeError(f"任务失败: {error}")
                if status in ("queued", "in_progress"):
                    time.sleep(poll_interval)
                else:
                    raise RuntimeError(f"未知状态: {status}")
            except requests.RequestException as e:
                print(f"  查询网络错误: {e}，{poll_interval}s 后重试...")
                time.sleep(poll_interval)

        raise TimeoutError(f"任务超时 ({max_wait}s)")

    # ── 下载 & 后处理 ─────────────────────────────────────

    @staticmethod
    def download_video(url: str, output_path: Path):
        """流式下载视频文件"""
        print(f"  下载: {url[:100]}...")
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  下载进度: {pct:.0f}%", end="\r")
        print(f"  下载完成 → {output_path.name}")

    def remove_audio(self, input_path: Path, output_path: Path) -> Path | None:
        """
        使用 ffmpeg 去除音轨（-c:v copy -an，无损且极快）
        若 ffmpeg 不可用或 keep_audio=True，跳过并返回 None
        """
        if self.keep_audio or not AgnesVideoGenerator.ffmpeg_available:
            print(f"  跳过去音频 (ffmpeg不可用或--keep-audio)")
            return None

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "copy",
            "-an",
            str(output_path),
        ]
        print(f"  去音频: {input_path.name} -> {output_path.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  [WARN] ffmpeg:\n{proc.stderr[:300]}")
            return None
        else:
            print(f"  去音频完成")
            return output_path

    # ── 单任务流程 ────────────────────────────────────────

    def generate_one(self, prompt: str, index: int, total: int,
                      name: str = None, **kwargs) -> str | None:
        """
        处理一条 prompt 的完整流程：提交 → 轮询 → 下载 → 去音频
        name: 输出文件标识名（如 "001_A"），为 None 时用 prompt 的 hash
        返回最终无声视频路径，失败返回 None
        """
        phash = self._prompt_hash(prompt)
        out_name = self._output_name(phash, name)

        # 跳过已完成的（用 out_name 匹配更可靠）
        if phash in self.progress["completed"]:
            prev = self.progress["tasks"].get(phash, {})
            prev_path = prev.get("silent_path", "")
            if prev_path and Path(prev_path).exists():
                print(f"\n[{index}/{total}] [SKIP] 已完成: {out_name} | {prompt[:50]}...")
                return prev_path

        print(f"\n{'─' * 55}")
        label = f"[{out_name}]" if name else ""
        print(f"[{index}/{total}] {label} {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'─' * 55}")

        try:
            # ① 提交
            print("  ① 提交任务...")
            task_data = self.submit_task(prompt, **kwargs)
            video_id = task_data.get("video_id")
            task_id = task_data.get("id", task_data.get("task_id"))
            print(f"     video_id = {video_id}")
            print(f"     task_id  = {task_id}")

            # ② 轮询等待
            print("  ② 等待生成...")
            result = self.poll_until_complete(video_id)

            # ③ 下载
            # 实际API返回的下载链接字段是 "url"，文档写的 "remixed_from_video_id" 是错的
            download_url = result.get("url") or result.get("remixed_from_video_id")
            if not download_url:
                raise RuntimeError(
                    f"响应中没有下载链接（url/remixed_from_video_id）。"
                    f"完整字段: {list(result.keys())}"
                )

            raw_path = self.output_dir / f"{out_name}_raw.mp4"
            self.download_video(download_url, raw_path)

            # ④ 去音频（若 ffmpeg 不可用则直接保留原始视频）
            silent_path = self.output_dir / f"{out_name}.mp4"
            removed = self.remove_audio(raw_path, silent_path)

            if removed is not None:
                # 去音频成功 → 清理原始有声文件
                if raw_path.exists() and silent_path.exists():
                    raw_path.unlink()
            else:
                # ffmpeg 不可用 → 将原始视频重命名作为最终文件
                if raw_path.exists():
                    if silent_path.exists():
                        silent_path.unlink()
                    raw_path.rename(silent_path)

            final_path = silent_path
            # ⑤ 记录进度
            self.progress["completed"].append(phash)
            self.progress["tasks"][phash] = {
                "name": out_name,
                "prompt": prompt.strip(),
                "video_id": video_id,
                "task_id": task_id,
                "silent_path": str(final_path),
                "size": Path(final_path).stat().st_size,
                "timestamp": datetime.now().isoformat(),
            }
            # 同时从 failed 中移除（如果之前失败过）
            self.progress["failed"] = [
                f for f in self.progress["failed"] if f.get("prompt_hash") != phash
            ]
            self._save_progress()

            print(f"  [OK] 成功 -> {final_path.name}")
            return str(final_path)

        except Exception as e:
            print(f"  [FAIL] {e}")
            self.progress["failed"].append({
                "prompt_hash": phash,
                "name": out_name,
                "prompt": prompt.strip(),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            self._save_progress()
            return None

    # ── 批量入口 ──────────────────────────────────────────

    def generate_batch(self, prompts, **kwargs) -> list:
        """
        批量生成，每个任务间等待 wait_seconds 秒
        prompts 支持两种格式：
          - list[str]: 纯 prompt 列表
          - list[tuple]: [(name, prompt), ...] 带名称的 prompt
        """
        total = len(prompts)
        results = []

        for i, item in enumerate(prompts, 1):
            if isinstance(item, (tuple, list)):
                name, prompt = item
            else:
                name, prompt = None, item

            prompt = prompt.strip()
            if not prompt:
                continue

            result = self.generate_one(prompt, i, total, name=name, **kwargs)
            results.append(result)

            # 最后一个任务后不需要等待
            if i < total:
                print(f"\n[WAIT] {self.wait_seconds}s 后提交下一个任务...")
                for remaining in range(self.wait_seconds, 0, -10):
                    print(f"   剩余 {remaining:>3} 秒...", end="\r")
                    time.sleep(10)
                print(" " * 30, end="\r")  # 清除行
                print("   等待完成 [OK]")

        # ── 汇总 ──
        success = [r for r in results if r is not None]
        print(f"\n{'=' * 55}")
        print(f"  批量生成完成: {len(success)}/{total} 成功")
        print(f"  失败任务数: {len(self.progress['failed'])}")
        print(f"  输出目录: {self.output_dir}")
        if self.progress["failed"]:
            print(f"  失败列表:")
            for f in self.progress["failed"]:
                print(f"    - [{f['prompt_hash']}] {f['prompt'][:60]}...")
                print(f"      错误: {f['error'][:100]}")
        return results


# ╔══════════════════════════════════════════════════════════╗
# ║                       CLI 入口                          ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Agnes Video V2.0 批量文生视频（自动去音频）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 (默认: 832×480, 81帧, 15fps, 约5.4秒)
  python generate_videos.py -p prompts.txt -k sk-xxx -o ./output

  # 自定义参数
  python generate_videos.py -p prompts.txt -k sk-xxx \\
      --height 1080 --width 1920 --num-frames 241 --frame-rate 24

  # 增加等待时间到 3 分钟
  python generate_videos.py -p prompts.txt -k sk-xxx -w 180
        """,
    )
    parser.add_argument("-p", "--prompts", required=True,
                        help="Prompt 文件路径（每行一个 prompt，# 开头为注释）")
    parser.add_argument("-k", "--api-key",
                        help="API Key（也可通过环境变量 AGNES_API_KEY 设置）")
    parser.add_argument("-o", "--output-dir", default="./output",
                        help="输出目录（默认: ./output）")
    parser.add_argument("-w", "--wait", type=int, default=120,
                        help="两次请求间隔秒数（默认: 120）")
    # 视频参数
    parser.add_argument("--height", type=int, default=480,
                        help="视频高度（默认: 480）")
    parser.add_argument("--width", type=int, default=832,
                        help="视频宽度（默认: 832）")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="总帧数，需满足 8n+1，≤441（默认: 81）")
    parser.add_argument("--frame-rate", type=int, default=15,
                        help="帧率（默认: 15）")
    parser.add_argument("--seed", type=int,
                        help="随机种子（可选，用于复现）")
    parser.add_argument("--negative-prompt", default="",
                        help="反向提示词（不希望出现的内容）")
    parser.add_argument("--keep-audio", action="store_true",
                        help="保留原始音频（不调用 ffmpeg 去音频）")
    args = parser.parse_args()

    # ── API Key ──
    api_key = args.api_key or os.environ.get("AGNES_API_KEY")
    if not api_key:
        print("[ERROR] 请设置 API Key: --api-key 或 环境变量 AGNES_API_KEY")
        sys.exit(1)

    # ── 读取 Prompts ──
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"[ERROR] Prompt 文件/目录不存在: {args.prompts}")
        sys.exit(1)

    if prompts_path.is_dir():
        # ── 目录模式：每个 .txt 文件一个 prompt，文件名作为标识 ──
        txt_files = sorted(
            prompts_path.glob("*.txt"),
            key=lambda p: p.stem  # 按文件名排序
        )
        if not txt_files:
            print(f"[ERROR] 目录中无 .txt 文件: {args.prompts}")
            sys.exit(1)

        prompts = []
        skipped_empty = []
        for fp in txt_files:
            name = fp.stem  # 例如 "001_A"
            content = fp.read_text(encoding="utf-8").strip()
            if content:
                # 去掉可能的第一行注释
                lines = [l for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]
                if lines:
                    prompts.append((name, lines[0]))
                else:
                    skipped_empty.append(name)
            else:
                skipped_empty.append(name)

        if skipped_empty:
            print(f"[WARN] 跳过了 {len(skipped_empty)} 个空文件: {skipped_empty[:5]}...")

    else:
        # ── 单文件模式：每行一个 prompt ──
        with open(prompts_path, "r", encoding="utf-8") as f:
            raw = f.readlines()

        prompts = []
        for line in raw:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                prompts.append(stripped)

    if not prompts:
        print("[ERROR] 没有有效的 prompt")
        sys.exit(1)

    # ── 打印配置 ──
    print("=" * 55)
    print("  Agnes Video V2.0  批量文生视频")
    print("=" * 55)
    print(f"  Prompts 数量   : {len(prompts)}")
    print(f"  输出目录       : {args.output_dir}")
    print(f"  请求间隔       : {args.wait} 秒")
    print(f"  分辨率         : {args.width}×{args.height}")
    print(f"  帧数×帧率      : {args.num_frames}×{args.frame_rate} "
          f"≈ {args.num_frames / args.frame_rate:.1f}s")
    if args.seed is not None:
        print(f"  随机种子       : {args.seed}")
    if args.negative_prompt:
        print(f"  反向提示词     : {args.negative_prompt[:60]}...")
    if args.keep_audio:
        print(f"  音频处理       : 保留原始音频")
    print("=" * 55)

    # ── ffmpeg 状态 ──
    if not _FFMPEG_AVAILABLE and not args.keep_audio:
        print("[提示] 未检测到 ffmpeg，视频将保留原始音频。")
        print("       安装 ffmpeg 后可用 --keep-audio 跳过此提示。")
        print("       也可批量去音频: ffmpeg -i input.mp4 -c:v copy -an output.mp4")
        print("=" * 55)

    # ── 执行 ──
    generator = AgnesVideoGenerator(
        api_key=api_key,
        output_dir=args.output_dir,
        wait_seconds=args.wait,
        keep_audio=args.keep_audio,
    )
    generator.generate_batch(
        prompts,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )


if __name__ == "__main__":
    main()
