#!/usr/bin/env python3
"""Compute Kinetics-400 Inception-I3D Fréchet Video Distance.

This implementation uses 400-dimensional pre-softmax I3D logits, matching the
standard FVD feature space. Videos are uniformly sampled to 16 RGB frames,
resized to 224x224, and normalized from uint8 to [-1, 1].
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.nn import functional as F

from pytorch_i3d import InceptionI3d


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--generated-cache", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def videos_by_id(root: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    result = {
        str(path.relative_to(root).with_suffix("")): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    }
    if not result:
        raise ValueError(f"No videos found in {root}")
    return result


def matched_videos(reference_dir: Path, generated_dir: Path) -> list[tuple[str, Path, Path]]:
    references = videos_by_id(reference_dir)
    generated = videos_by_id(generated_dir)
    if references.keys() != generated.keys():
        missing = sorted(references.keys() - generated.keys())
        extra = sorted(generated.keys() - references.keys())
        raise ValueError(f"Video IDs differ; missing={missing[:10]}, extra={extra[:10]}")
    return [(key, references[key], generated[key]) for key in sorted(references)]


def decode_uniform(path: Path, num_frames: int, resolution: int) -> torch.Tensor:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
    if total_frames <= 0:
        with av.open(str(path)) as container:
            total_frames = sum(1 for _ in container.decode(video=0))
    if total_frames <= 0:
        raise ValueError(f"No decodable frames: {path}")
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    wanted = set(int(index) for index in indices)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frames[index] = frame.to_rgb().to_ndarray()
                if len(frames) == len(wanted):
                    break
    missing = wanted - frames.keys()
    if missing:
        raise ValueError(f"Failed to decode frame indices {sorted(missing)}: {path}")
    # Some concatenated long videos change resolution at shot boundaries, so
    # resize each selected frame before stacking it with the others.
    resized = []
    for index in indices:
        frame = torch.from_numpy(frames[int(index)]).permute(2, 0, 1).float().unsqueeze(0)
        frame = F.interpolate(frame, (resolution, resolution), mode="bilinear", align_corners=False)
        resized.append(frame.squeeze(0))
    tensor = torch.stack(resized)
    tensor = tensor.div_(127.5).sub_(1.0)
    return tensor.permute(1, 0, 2, 3).contiguous()


def load_model(weights_path: Path, device: str) -> InceptionI3d:
    model = InceptionI3d(num_classes=400)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def load_cache(cache: Path | None, ids: list[str]) -> np.ndarray | None:
    if cache is None or not cache.exists():
        return None
    data = np.load(cache, allow_pickle=False)
    cached_ids = data["ids"].astype(str).tolist()
    features = data["features"]
    if cached_ids != ids or features.shape != (len(ids), 400):
        raise ValueError(f"Cache does not match requested videos: {cache}")
    print(f"[cache] loaded {cache} {features.shape}", flush=True)
    return features.astype(np.float64)


def encode_paths(
    paths: list[Path],
    ids: list[str],
    model: InceptionI3d,
    device: str,
    num_frames: int,
    resolution: int,
    batch_size: int,
    cache: Path | None,
) -> np.ndarray:
    cached = load_cache(cache, ids)
    if cached is not None:
        return cached
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            videos = torch.stack([decode_uniform(p, num_frames, resolution) for p in batch_paths])
            batch_features = model(videos.to(device)).cpu().numpy()
            features.append(batch_features)
            print(f"[encode] {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)
    result = np.concatenate(features).astype(np.float64)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ids=np.asarray(ids), features=result.astype(np.float32))
        print(f"[cache] saved {cache}", flush=True)
    return result


def frechet_distance(reference: np.ndarray, generated: np.ndarray) -> float:
    mu_r, mu_g = reference.mean(axis=0), generated.mean(axis=0)
    cov_r = np.cov(reference, rowvar=False, ddof=1)
    cov_g = np.cov(generated, rowvar=False, ddof=1)
    difference = mu_r - mu_g
    covmean = sqrtm(cov_r @ cov_g)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_r.shape[0]) * 1e-6
        covmean = sqrtm((cov_r + offset) @ (cov_g + offset))
    if np.iscomplexobj(covmean):
        max_imaginary = float(np.max(np.abs(covmean.imag)))
        if max_imaginary > 1e-3:
            raise ValueError(f"Large imaginary component in covariance square root: {max_imaginary}")
        covmean = covmean.real
    score = difference @ difference + np.trace(cov_r + cov_g - 2.0 * covmean)
    return max(float(score), 0.0)


def main() -> None:
    args = parse_args()
    if args.num_frames != 16 or args.resolution != 224:
        raise ValueError("This standard configuration requires --num-frames 16 --resolution 224")
    pairs = matched_videos(args.reference_dir, args.generated_dir)
    ids = [item[0] for item in pairs]
    model = load_model(args.weights.expanduser().resolve(), args.device)
    reference_features = encode_paths(
        [item[1] for item in pairs], ids, model, args.device, args.num_frames,
        args.resolution, args.batch_size, args.reference_cache,
    )
    generated_features = encode_paths(
        [item[2] for item in pairs], ids, model, args.device, args.num_frames,
        args.resolution, args.batch_size, args.generated_cache,
    )
    score = frechet_distance(reference_features, generated_features)
    self_score = frechet_distance(reference_features, reference_features)
    result = {
        "metric": "fvd",
        "implementation": "standard_i3d_fvd_pytorch",
        "reference_dir": str(args.reference_dir.expanduser().resolve()),
        "generated_dir": str(args.generated_dir.expanduser().resolve()),
        "feature_backbone": "inception_i3d_kinetics_400",
        "feature_layer": "400d_pre_softmax_logits",
        "weights_path": str(args.weights.expanduser().resolve()),
        "weights_sha256": sha256(args.weights),
        "num_videos": len(pairs),
        "num_frames": args.num_frames,
        "frame_sampling": "uniform_full_video_including_endpoints",
        "input_resolution": [args.resolution, args.resolution],
        "input_range": [-1.0, 1.0],
        "resize": "bilinear_direct_square",
        "covariance_ddof": 1,
        "fvd": score,
        "sanity_check_reference_vs_itself": self_score,
        "status": "recomputed_standard_i3d_fvd_2026-07-20",
        "items": [
            {"video_id": key, "reference_path": str(rp), "generated_path": str(gp)}
            for key, rp, gp in pairs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FVD={score:.9f}", flush=True)
    print(f"self_FVD={self_score:.12f}", flush=True)
    print(f"[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
