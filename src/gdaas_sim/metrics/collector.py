from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricsCollector:
    arrivals: int = 0
    starts: int = 0
    finishes: int = 0

    # time accounting for utilization
    last_time: float = 0.0
    busy_gpu_time: float = 0.0  # integral of (used GPUs) over time

    # job-level stats
    wait_times: List[float] = field(default_factory=list)
    turnaround_times: List[float] = field(default_factory=list)

    # internal
    _in_service: Dict[str, int] = field(default_factory=dict)  # job_id -> gpus

    def _update_util(self, now: float, cluster) -> None:
        dt = now - self.last_time
        if dt < 0:
            raise ValueError("Time went backwards")

        used = cluster.total_gpus - cluster.available_gpus
        self.busy_gpu_time += used * dt
        self.last_time = now

    def on_job_arrival(self, job, now: float) -> None:
        self.arrivals += 1

    def on_job_start(self, job, now: float) -> None:
        self.starts += 1

    def on_job_finish(self, job, now: float) -> None:
        self.finishes += 1
        if job.start_time is not None:
            self.wait_times.append(job.start_time - job.arrival_time)
        if job.finish_time is not None:
            self.turnaround_times.append(job.finish_time - job.arrival_time)

    def finalize(self, now: float, cluster) -> None:
        self._update_util(now, cluster)

    def snapshot_utilization(self, now: float, cluster) -> float:
        """
        Utilization over [0, now].
        """
        if now <= 0:
            return 0.0
        # ensure util time is updated
        self._update_util(now, cluster)
        return self.busy_gpu_time / (cluster.total_gpus * now)
