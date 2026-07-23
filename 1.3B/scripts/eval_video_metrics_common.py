#!/usr/bin/env python3
"""视频测评公共工具。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, CLIPConfig, CLIPModel, XCLIPConfig, XCLIPModel


def _materialize_meta_buffers_from_state_dict(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Attach checkpoint values for non-persistent buffers left on meta."""
    for name, buffer in list(model.named_buffers()):
        if not buffer.is_meta or name not in state_dict:
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], state_dict[name])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_root(output_root: str, generated_dir: str) -> Path:
    if output_root.strip():
        return Path(output_root).expanduser().resolve()

    generated_root = Path(generated_dir).expanduser().resolve()
    # 例如 /root/baseline1/generated/video -> /root/baseline1/comparison
    # 例如 /root/baseline/gen-video/video -> /root/baseline/comparison
    if generated_root.parent.name in {"generated", "gen-video"}:
        return generated_root.parent.parent / "comparison"
    return generated_root.parent / "comparison"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_local_xclip_model(model_path: str, device: str) -> XCLIPModel:
    resolved = Path(model_path).expanduser().resolve()
    config = XCLIPConfig.from_pretrained(resolved, local_files_only=True)
    # Memory-map local weights and assign their storage directly to the module.
    # This avoids keeping both a full state_dict copy and a model copy in RAM,
    # which is important in evaluation containers with a small cgroup limit.
    state_dict = torch.load(
        resolved / "pytorch_model.bin",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    # Build on the meta device so random initialization does not temporarily
    # allocate a second full copy of the model before assign=True attaches the
    # memory-mapped checkpoint tensors.
    with torch.device("meta"):
        model = XCLIPModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    _materialize_meta_buffers_from_state_dict(model, state_dict)
    if missing_keys or unexpected_keys:
        print(
            "[warn] XCLIP state_dict loaded with "
            f"{len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys."
        )
    model = model.to(device)
    model.eval()
    return model


def load_local_clip_model(model_path: str, device: str) -> CLIPModel:
    resolved = Path(model_path).expanduser().resolve()
    config = CLIPConfig.from_pretrained(resolved, local_files_only=True)
    state_dict = torch.load(
        resolved / "pytorch_model.bin",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    with torch.device("meta"):
        model = CLIPModel(config)
    model.load_state_dict(state_dict, strict=False, assign=True)
    _materialize_meta_buffers_from_state_dict(model, state_dict)
    model = model.to(device)
    model.eval()
    return model


def list_video_pairs(reference_dir: str, generated_dir: str) -> list[tuple[str, Path, Path]]:
    reference_root = Path(reference_dir).expanduser().resolve()
    generated_root = Path(generated_dir).expanduser().resolve()
    if not reference_root.exists():
        raise FileNotFoundError(f"Reference video dir not found: {reference_root}")
    if not generated_root.exists():
        raise FileNotFoundError(f"Generated video dir not found: {generated_root}")

    generated_map = {path.stem: path for path in generated_root.glob("*.mp4") if path.is_file()}
    pairs: list[tuple[str, Path, Path]] = []
    for reference_path in sorted(path for path in reference_root.glob("*.mp4") if path.is_file()):
        generated_path = generated_map.get(reference_path.stem)
        if generated_path is not None:
            pairs.append((reference_path.stem, reference_path, generated_path))
    if not pairs:
        raise FileNotFoundError(
            f"No matched video pairs found between {reference_root} and {generated_root}"
        )
    return pairs


def decode_video_frames(video_path: Path) -> list[np.ndarray]:
    with av.open(str(video_path)) as container:
        frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return frames


def decode_sampled_video_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    """Uniformly sample frames without retaining the entire decoded video.

    The selected indices are identical to ``sample_frame_indices``.  This is
    important for long-video evaluation in memory-limited containers, where
    holding hundreds of full-resolution frames alongside a model can exceed
    the cgroup limit.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)

    if total_frames <= 0:
        with av.open(str(video_path)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))
    if total_frames <= 0:
        raise ValueError(f"No frames decoded from {video_path}")

    indices = sample_frame_indices(total_frames, num_frames)
    wanted = set(int(index) for index in indices)
    selected: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                selected[index] = frame.to_rgb().to_ndarray()
                if len(selected) == len(wanted):
                    break
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"Failed to decode sampled frame indices {sorted(missing)} from {video_path}")
    return [selected[int(index)] for index in indices]


def resize_frame(frame: np.ndarray, target_size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(target_size, int):
        resize_size = (target_size, target_size)
    else:
        resize_size = target_size

    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=resize_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (tensor.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()


def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return np.linspace(0, total_frames - 1, num=num_frames, dtype=int)


def sample_aligned_video_frames(
    reference_path: Path,
    generated_path: Path,
    num_frames: int,
    target_size: int | tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sampled_reference = decode_sampled_video_frames(reference_path, num_frames)
    sampled_generated = decode_sampled_video_frames(generated_path, num_frames)

    if target_size is None:
        return sampled_reference, sampled_generated

    resized_reference = [resize_frame(frame, target_size) for frame in sampled_reference]
    resized_generated = [resize_frame(frame, target_size) for frame in sampled_generated]
    return resized_reference, resized_generated


def sample_video_frames(
    video_path: Path,
    num_frames: int,
    target_size: int | tuple[int, int] | None = None,
) -> list[np.ndarray]:
    sampled = decode_sampled_video_frames(video_path, num_frames)
    if target_size is None:
        return sampled

    return [resize_frame(frame, target_size) for frame in sampled]


def frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    frame_tensors = [
        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        for frame in frames
    ]
    return torch.stack(frame_tensors, dim=0)


def cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = F.normalize(lhs.reshape(1, -1), dim=-1)
    rhs = F.normalize(rhs.reshape(1, -1), dim=-1)
    return float((lhs * rhs).sum().item())


def compute_mean_and_covariance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    covariance = np.cov(features, rowvar=False)
    return mean, covariance


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    # 这里使用特征值分解计算对称半正定矩阵平方根，避免额外依赖 scipy。
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, a_min=0.0, a_max=None)
    sqrt_diag = np.diag(np.sqrt(clipped))
    return eigenvectors @ sqrt_diag @ eigenvectors.T


def frechet_distance(
    mean1: np.ndarray,
    cov1: np.ndarray,
    mean2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    mean_diff = mean1 - mean2
    sqrt_cov1 = matrix_sqrt_psd(cov1)
    cov_prod_sqrt = matrix_sqrt_psd(sqrt_cov1 @ cov2 @ sqrt_cov1)
    value = mean_diff.dot(mean_diff) + np.trace(cov1 + cov2 - 2.0 * cov_prod_sqrt)
    if math.isnan(value):
        raise ValueError("Computed FVD is NaN.")
    return float(np.real(value))
