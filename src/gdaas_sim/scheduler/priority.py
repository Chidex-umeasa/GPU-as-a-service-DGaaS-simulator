from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class PriorityScheduler(SchedulerBase):
    """
    Non-preemptive priority scheduler.

    Higher ``job.priority`` integer → scheduled before lower-priority jobs.
    Within the same priority level, jobs are dispatched in arrival order (FIFO).

    Heap key: ``(-priority, seq)`` — negating priority converts the min-heap
    into an effective max-heap on the priority dimension.
    """

    def __init__(self):
        self._heap: List[Tuple[int, int, Job]] = []
        self._seq: int = 0

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (-job.priority, self._seq, job))

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._heap:
            return None

        tmp: List[Tuple[int, int, Job]] = []
        chosen: Optional[Job] = None

        while self._heap:
            neg_pri, seq, job = heapq.heappop(self._heap)
            if job.gpus_required <= available_gpus:
                chosen = job
                break
            tmp.append((neg_pri, seq, job))  # preserve original tuple

        # Restore skipped entries with their original keys
        for item in tmp:
            heapq.heappush(self._heap, item)

        return chosen
