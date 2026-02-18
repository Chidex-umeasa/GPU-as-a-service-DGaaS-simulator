"""
YAML-based experiment configuration loader.

Example config_default.yaml
----------------------------
total_gpus: 16
n_jobs: 800
schedulers:
  - fifo
  - sjf
  - fair
  - edf
  - priority
  - rr
  - backfill
seeds: [1, 2, 3]
arrival_rates: [0.2, 0.5, 0.9]
outdir: experiments/out
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment sweep."""

    total_gpus: int = 16
    n_jobs: int = 800
    schedulers: List[str] = field(
        default_factory=lambda: ["fifo", "sjf", "fair", "edf", "priority", "rr", "backfill"]
    )
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    arrival_rates: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.9])
    outdir: str = "experiments/out"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required to load YAML configs. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
