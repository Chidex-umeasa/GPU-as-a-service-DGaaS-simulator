from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class SJFScheduler(SchedulerBase):
    """
    Non-preemptive shortest-job-first using duration as the key.
    """

    def __init__(self):
        self._heap: List[Tuple[float, int, Job]] = []
        self._seq = 0

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (job.duration, self._seq, job))

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._heap:
            return None

        # We must find the shortest job that fits available GPUs.
        # We'll temporarily pop until we find a runnable job.
        tmp = []
        chosen = None

        while self._heap:
            _, _, job = heapq.heappop(self._heap)
            if job.gpus_required <= available_gpus:
                chosen = job
                break
            tmp.append(job)

        # Push back the ones we skipped
        for j in tmp:
            self._seq += 1
            heapq.heappush(self._heap, (j.duration, self._seq, j))

        return chosen
