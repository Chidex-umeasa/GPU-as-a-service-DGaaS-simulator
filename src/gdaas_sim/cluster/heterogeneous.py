from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GpuTypeSpec:
    """Specification for one GPU type in a heterogeneous cluster."""
    name: str
    count: int
    speed_factor: float = 1.0   # relative compute speed (job finishes in duration/speed_factor)
    cost_per_hour: float = 1.0  # cost per GPU per time unit


class HeterogeneousCluster:
    """
    A GPU cluster with multiple GPU types (e.g., A100, V100, T4).

    Allocation policy:
    - If a job has a gpu_type_preference, try to satisfy it first.
    - Fall back to any type with enough free GPUs.
    - Returns the type string actually allocated.
    """

    def __init__(self, gpu_types: List[GpuTypeSpec]) -> None:
        if not gpu_types:
            raise ValueError("gpu_types must not be empty")
        self.gpu_types: Dict[str, GpuTypeSpec] = {spec.name: spec for spec in gpu_types}
        self._available_by_type: Dict[str, int] = {spec.name: spec.count for spec in gpu_types}
        self.total_gpus: int = sum(spec.count for spec in gpu_types)
        self.available_gpus: int = self.total_gpus  # compat with GPUCluster interface

    # ------------------------------------------------------------------
    # Core allocation interface (matches GPUCluster signature)
    # ------------------------------------------------------------------

    def allocate(self, n: int, gpu_type: Optional[str] = None) -> Optional[str]:
        """
        Allocate n GPUs of the preferred type (or any type if preference not
        satisfied). Returns the type string of the allocated GPUs.
        Raises RuntimeError if allocation fails.
        """
        if n <= 0:
            raise ValueError("allocate n must be > 0")

        # Try preferred type first
        if gpu_type is not None and gpu_type in self._available_by_type:
            if self._available_by_type[gpu_type] >= n:
                self._available_by_type[gpu_type] -= n
                self.available_gpus -= n
                return gpu_type

        # Fall back to any type with enough free GPUs (in insertion order)
        for type_name, avail in self._available_by_type.items():
            if avail >= n:
                self._available_by_type[type_name] -= n
                self.available_gpus -= n
                return type_name

        raise RuntimeError(
            f"Not enough GPUs available: need {n}, "
            f"available_by_type={dict(self._available_by_type)}"
        )

    def release(self, n: int, gpu_type: Optional[str] = None) -> None:
        """
        Release n GPUs of the given type back to the cluster.
        If gpu_type is None, releases to the first type (best-effort fallback).
        """
        if n <= 0:
            raise ValueError("release n must be > 0")

        if gpu_type is None:
            # Fallback: release to the first defined type
            gpu_type = next(iter(self._available_by_type))

        if gpu_type not in self._available_by_type:
            raise ValueError(f"Unknown gpu_type: {gpu_type!r}")

        self._available_by_type[gpu_type] += n
        self.available_gpus += n

        spec = self.gpu_types[gpu_type]
        if self._available_by_type[gpu_type] > spec.count:
            raise RuntimeError(
                f"Released more GPUs than total for type {gpu_type!r}: "
                f"available={self._available_by_type[gpu_type]} > count={spec.count}"
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def available_of_type(self, gpu_type: str) -> int:
        """Return the number of currently available GPUs of the given type."""
        if gpu_type not in self._available_by_type:
            raise ValueError(f"Unknown gpu_type: {gpu_type!r}")
        return self._available_by_type[gpu_type]

    def cost_per_gpu_hour_of(self, gpu_type: str) -> float:
        """Return the cost per GPU per time unit for the given type."""
        if gpu_type not in self.gpu_types:
            raise ValueError(f"Unknown gpu_type: {gpu_type!r}")
        return self.gpu_types[gpu_type].cost_per_hour

    def speed_factor_of(self, gpu_type: str) -> float:
        """Return the relative compute speed factor for the given type."""
        if gpu_type not in self.gpu_types:
            raise ValueError(f"Unknown gpu_type: {gpu_type!r}")
        return self.gpu_types[gpu_type].speed_factor
