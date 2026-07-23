#!/usr/bin/env python3
"""Apply the mapped blue baseline replacements to new-baseline/baseline1."""

from pathlib import Path

import replace_o_videos as replacement


replacement.SOURCE = Path("/root/autodl-tmp/blue-v/蓝色_baseline_videos")
replacement.TARGET = Path("/root/autodl-tmp/new-baseline/baseline1")
replacement.BACKUP = replacement.TARGET / "baseline1_replacement_backup"
replacement.MANIFEST = replacement.TARGET / "baseline1_replacement_manifest.tsv"
replacement.SUMMARY = replacement.TARGET / "baseline1_replacement_summary.json"


if __name__ == "__main__":
    replacement.main()
