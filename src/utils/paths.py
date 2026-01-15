"""Helpers for loading and resolving pipeline paths.

Goal: stop duplicating dataset/checkpoint/output directories across scripts and sbatch files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def libgen_root() -> Path:
    """Return the absolute path to the src repo root."""
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (root / p))


@dataclass(frozen=True)
class PipelinePaths:
    csv_metadata: str
    split_dir: str
    tokenized_dir: str
    checkpoint_root: str
    analysis_root: str
    figures_root: str

    @classmethod
    def from_yaml(
        cls, paths_config: str, root: Optional[Path] = None
    ) -> "PipelinePaths":
        root = libgen_root() if root is None else root
        with open(paths_config, "r") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        dataset = data.get("dataset", {}) or {}
        training = data.get("training", {}) or {}
        reports = data.get("reports", {}) or {}

        return cls(
            csv_metadata=_resolve(root, dataset["csv_metadata"]),
            split_dir=_resolve(root, dataset["split_dir"]),
            tokenized_dir=_resolve(root, dataset["tokenized_dir"]),
            checkpoint_root=_resolve(root, training["checkpoint_root"]),
            analysis_root=_resolve(
                root, reports.get("analysis_root", "analysis_results")
            ),
            figures_root=_resolve(
                root, reports.get("figures_root", "experiments/reports")
            ),
        )
