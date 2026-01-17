from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricsCollector:
    arrivals: int = 0
    starts: int = 0
    finishes: int = 0
    sla_wait_violations: int = 0
    sla_deadline_violations: int = 0


    # time accounting for utilization
    last_time: float = 0.0
    busy_gpu_time: float = 0.0  # integral of (used GPUs) over time

    # job-level stats
    wait_times: List[float] = field(default_factory=list)
    turnaround_times: List[float] = field(default_factory=list)

    # per-tenant stats (for fairness)
    tenant_gpu_time: Dict[str, float] = field(default_factory=dict)

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

        if job.start_time is not None:
            wait = job.start_time - job.arrival_time
            self.wait_times.append(wait)

            if job.max_wait is not None and wait > job.max_wait:
                self.sla_wait_violations += 1

        if job.finish_time is not None:
            tat = job.finish_time - job.arrival_time
            self.turnaround_times.append(tat)

            if job.deadline is not None and job.finish_time > job.deadline:
                self.sla_deadline_violations += 1

        # per-tenant GPU time (approx = duration * gpus_required)
        if job.start_time is not None and job.finish_time is not None:
            runtime = job.finish_time - job.start_time
            gpu_time = runtime * job.gpus_required
            self.tenant_gpu_time[job.tenant_id] = (
                self.tenant_gpu_time.get(job.tenant_id, 0.0) + gpu_time
            )

    def finalize(self, now: float, cluster) -> None:
        self._update_util(now, cluster)

    def snapshot_utilization(self, now: float, cluster) -> float:
        """
        Utilization over [0, now].
        """
        if now <= 0:
            return 0.0
        self._update_util(now, cluster)
        return self.busy_gpu_time / (cluster.total_gpus * now)

    def jain_fairness(self) -> Optional[float]:
        """
        Jain's fairness index over per-tenant GPU time.
        1.0 = perfectly fair, 1/n = extremely unfair.
        """
        if not self.tenant_gpu_time:
            return None
        vals = list(self.tenant_gpu_time.values())
        s = sum(vals)
        sq = sum(v * v for v in vals)
        if sq == 0:
            return 1.0
        n = len(vals)
        return (s * s) / (n * sq)
