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
    busy_gpu_time: float = 0.0   # integral of (busy GPUs) over time
    _current_busy_gpus: int = 0  # self-tracked; updated on start/finish

    # job-level stats
    wait_times: List[float] = field(default_factory=list)
    turnaround_times: List[float] = field(default_factory=list)

    # per-tenant stats (for fairness)
    tenant_gpu_time: Dict[str, float] = field(default_factory=dict)

    def _record_interval(self, now: float) -> None:
        """Integrate GPU usage over [last_time, now] using the current busy count."""
        dt = now - self.last_time
        if dt < 0:
            raise ValueError("Time went backwards")
        self.busy_gpu_time += self._current_busy_gpus * dt
        self.last_time = now

    # Keep _update_util as a compatibility alias (used by snapshot_utilization)
    def _update_util(self, now: float, cluster) -> None:
        self._record_interval(now)

    def on_job_arrival(self, job, now: float) -> None:
        self.arrivals += 1

    def on_job_start(self, job, now: float) -> None:
        self.starts += 1
        # Record the interval with the PREVIOUS busy count, then add this job
        self._record_interval(now)
        self._current_busy_gpus += job.gpus_required

    def on_job_finish(self, job, now: float) -> None:
        self.finishes += 1
        # Record the interval while this job is still counted as busy, then remove it
        self._record_interval(now)
        self._current_busy_gpus -= job.gpus_required

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
        self._record_interval(now)

    def snapshot_utilization(self, now: float, cluster) -> float:
        """Return GPU utilization over [0, now] as a fraction in [0, 1]."""
        if now <= 0:
            return 0.0
        self._record_interval(now)
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
