from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class TenantFairScheduler(SchedulerBase):
    """
    Simple fair-share scheduler:
      - Maintains a waiting dict of jobs (job_id -> Job) for O(1) removal.
      - Tracks per-tenant running GPU usage.
      - At each decision, chooses a runnable job whose tenant currently
        has the lowest GPU usage.
    """

    def __init__(self):
        self._waiting: Dict[str, Job] = {}  # job_id -> Job; O(1) deletion
        self._running_gpu = defaultdict(int)  # tenant_id -> GPUs in use

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._waiting[job.job_id] = job

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._waiting:
            return None

        chosen: Optional[Job] = None
        chosen_share: Optional[int] = None

        for job in self._waiting.values():
            if job.gpus_required > available_gpus:
                continue

            share = self._running_gpu[job.tenant_id]
            if chosen is None or share < chosen_share:
                chosen = job
                chosen_share = share

        if chosen is None:
            return None

        del self._waiting[chosen.job_id]  # O(1) dict removal
        return chosen

    def on_job_start(self, job: Job, now: float) -> None:
        self._running_gpu[job.tenant_id] += job.gpus_required

    def on_job_finish(self, job: Job, now: float) -> None:
        self._running_gpu[job.tenant_id] -= job.gpus_required
        if self._running_gpu[job.tenant_id] <= 0:
            del self._running_gpu[job.tenant_id]
