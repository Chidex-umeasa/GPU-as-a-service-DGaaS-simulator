from __future__ import annotations

from collections import deque
from typing import Optional

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class FIFOScheduler(SchedulerBase):
    def __init__(self):
        self.q = deque()

    def on_job_arrival(self, job: Job, now: float) -> None:
        self.q.append(job)

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self.q:
            return None

        # FIFO but only dispatch if runnable with available GPUs
        if self.q[0].gpus_required <= available_gpus:
            return self.q.popleft()

        return None
