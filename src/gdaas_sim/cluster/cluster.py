from dataclasses import dataclass


@dataclass
class GPUCluster:
    total_gpus: int
    available_gpus: int = None  # set in __post_init__

    def __post_init__(self):
        if self.total_gpus <= 0:
            raise ValueError("total_gpus must be > 0")
        self.available_gpus = self.total_gpus

    def allocate(self, n: int) -> None:
        if n <= 0:
            raise ValueError("allocate n must be > 0")
        if n > self.available_gpus:
            raise RuntimeError("Not enough GPUs available")
        self.available_gpus -= n

    def release(self, n: int) -> None:
        if n <= 0:
            raise ValueError("release n must be > 0")
        self.available_gpus += n
        if self.available_gpus > self.total_gpus:
            raise RuntimeError("Released more GPUs than total")
