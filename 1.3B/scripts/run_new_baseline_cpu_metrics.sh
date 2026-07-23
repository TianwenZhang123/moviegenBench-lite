#!/usr/bin/env bash
set -euo pipefail

name=$1
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

echo "[start FLOW] $name"
python -u /root/scripts/eval_optical_flow_consistency.py \
  --reference-dir "$ref" --generated-dir "$gen" --output-root "$out" \
  --num-frames 16 --frame-size 256
echo "[done FLOW] $name"

echo "[start SSIM] $name"
python -u /root/scripts/eval_ssim.py \
  --reference-dir "$ref" --generated-dir "$gen" --output-root "$out" \
  --num-frames 16 --frame-size 256
echo "[done SSIM] $name"
