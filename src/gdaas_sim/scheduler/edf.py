from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class EDFScheduler(SchedulerBase):
    """
    Earliest-Deadline-First (non-preemptive).
    Jobs without deadlines are treated as lowest priority (deadline = +inf).
    """

    def __init__(self):
        self._heap: List[Tuple[float, int, Job]] = []
        self._seq = 0

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._seq += 1
        dl = float(job.deadline) if job.deadline is not None else float("inf")
        heapq.heappush(self._heap, (dl, self._seq, job))

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._heap:
            return None

        tmp: List[Tuple[float, int, Job]] = []
        chosen: Optional[Job] = None

        while self._heap:
            dl, seq, job = heapq.heappop(self._heap)
            if job.gpus_required <= available_gpus:
                chosen = job
                break
            tmp.append((dl, seq, job))

        for item in tmp:
            heapq.heappush(self._heap, item)

        return chosen
