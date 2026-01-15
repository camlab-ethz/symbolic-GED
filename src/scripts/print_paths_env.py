#!/usr/bin/env python3
"""Print pipeline paths as shell-compatible environment assignments.

Usage:
  eval "$(python3 scripts/print_paths_env.py --paths-config configs/paths_48000_fixed.yaml)"
"""

from __future__ import annotations

import argparse

from utils.paths import PipelinePaths, libgen_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Print pipeline paths as bash env assignments")
    parser.add_argument(
        "--paths-config",
        type=str,
        default="configs/paths_48000_fixed.yaml",
        help="YAML file with dataset/checkpoint/output paths",
    )
    args = parser.parse_args()

    root = libgen_root()
    cfg_path = args.paths_config
    # Resolve relative config path from repo root for convenience.
    if not cfg_path.startswith("/"):
        cfg_path = str(root / cfg_path)

    p = PipelinePaths.from_yaml(cfg_path)
    # Keep names consistent with existing sbatch scripts
    print(f'CSV_METADATA="{p.csv_metadata}"')
    print(f'SPLIT_DIR="{p.split_dir}"')
    print(f'TOKENIZED_DIR="{p.tokenized_dir}"')
    print(f'CHECKPOINT_ROOT="{p.checkpoint_root}"')
    print(f'ANALYSIS_ROOT="{p.analysis_root}"')
    print(f'FIGURES_ROOT="{p.figures_root}"')


if __name__ == "__main__":
    main()

