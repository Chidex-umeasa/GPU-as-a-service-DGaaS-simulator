from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class TenantFairScheduler(SchedulerBase):
    """
    Simple fair-share scheduler:
      - Maintains a waiting list of jobs.
      - Tracks per-tenant running GPU usage.
      - At each decision, chooses a runnable job whose tenant currently
        has the lowest GPU usage.
    """

    def __init__(self):
        self._waiting: List[Job] = []
        self._running_gpu = defaultdict(int)  # tenant_id -> GPUs in use

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._waiting.append(job)

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._waiting:
            return None

        chosen = None
        chosen_share = None

        for job in self._waiting:
            if job.gpus_required > available_gpus:
                continue

            share = self._running_gpu.get(job.tenant_id, 0)
            if chosen is None or share < chosen_share:
                chosen = job
                chosen_share = share

        if chosen is None:
            return None

        # remove chosen from waiting list
        self._waiting.remove(chosen)
        return chosen

    def on_job_start(self, job: Job, now: float) -> None:
        self._running_gpu[job.tenant_id] += job.gpus_required

    def on_job_finish(self, job: Job, now: float) -> None:
        self._running_gpu[job.tenant_id] -= job.gpus_required
        if self._running_gpu[job.tenant_id] <= 0:
            del self._running_gpu[job.tenant_id]
