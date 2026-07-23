#!/usr/bin/env python3
"""Apply blue rewritten-caption replacements to r-v and long-r-v."""

from pathlib import Path

import replace_original_caption_c_v as replacement


replacement.SOURCE = Path("/root/autodl-tmp/blue-v/蓝色_videos_rewritten_caption_videos")
replacement.SHORT_DIR = Path("/root/autodl-tmp/14B/v/r-v")
replacement.LONG_DIR = Path("/root/autodl-tmp/14B/v/long-r-v")
replacement.LONG_MAPPING = replacement.LONG_DIR / "manifest.json"
replacement.BACKUP = replacement.SHORT_DIR / "rewritten_replacement_backup"
replacement.MANIFEST = replacement.SHORT_DIR / "rewritten_replacement_manifest.tsv"
replacement.SUMMARY = replacement.SHORT_DIR / "rewritten_replacement_summary.json"


if __name__ == "__main__":
    replacement.main()
