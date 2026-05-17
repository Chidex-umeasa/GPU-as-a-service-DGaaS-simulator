from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUCluster:
    total_gpus: int
    available_gpus: int = None  # set in __post_init__

    def __post_init__(self):
        if self.total_gpus <= 0:
            raise ValueError("total_gpus must be > 0")
        self.available_gpus = self.total_gpus

    def allocate(self, n: int, gpu_type: Optional[str] = None) -> Optional[str]:
        """
        Allocate n GPUs. gpu_type is accepted for API compatibility with
        HeterogeneousCluster but ignored (homogeneous cluster).
        Returns None (no type differentiation in homogeneous cluster).
        """
        if n <= 0:
            raise ValueError("allocate n must be > 0")
        if n > self.available_gpus:
            raise RuntimeError("Not enough GPUs available")
        self.available_gpus -= n
        return None

    def release(self, n: int, gpu_type: Optional[str] = None) -> None:
        """
        Release n GPUs. gpu_type is accepted for API compatibility with
        HeterogeneousCluster but ignored (homogeneous cluster).
        """
        if n <= 0:
            raise ValueError("release n must be > 0")
        self.available_gpus += n
        if self.available_gpus > self.total_gpus:
            raise RuntimeError("Released more GPUs than total")
