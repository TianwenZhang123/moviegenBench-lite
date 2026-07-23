#!/usr/bin/env bash
set -euo pipefail

name=$1
gpu=$2
base=/root/autodl-tmp/new-baseline
ref=$base/o-video/video

case "$name" in
  baseline1) gen=$base/baseline1/video ;;
  long-c-v) gen=$base/methods/long-c-v ;;
  long-r-v) gen=$base/methods/long-r-v ;;
  *) echo "Unknown dataset: $name" >&2; exit 2 ;;
esac
out=$base/eval/$name
mkdir -p "$out"

echo "[start LPIPS] $name GPU $gpu"
CUDA_VISIBLE_DEVICES=$gpu python -u /root/scripts/eval_lpips.py \
  --reference-dir "$ref" --generated-dir "$gen" --output-root "$out" \
  --device cuda:0 --num-frames 16 --frame-size 256
echo "[done LPIPS] $name"

echo "[start FVD] $name GPU $gpu"
CUDA_VISIBLE_DEVICES=$gpu python -u /root/scripts/eval_fvd.py \
  --reference-dir "$ref" --generated-dir "$gen" --output-root "$out" \
  --device cuda:0 --num-frames 16
echo "[done FVD] $name"
