from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class RoundRobinScheduler(SchedulerBase):
    """
    Per-tenant round-robin scheduler.

    Each dispatch cycle gives the next tenant in rotation one job slot.
    Tenants are ordered by their first job arrival; new tenants are appended
    to the end of the rotation.  A tenant is skipped if its head-of-queue job
    does not fit in the currently available GPUs.

    This enforces equal turn-taking across tenants without the starvation risk
    of strict fair-share: every tenant gets at most one job dispatched before
    the pointer advances to the next tenant.
    """

    def __init__(self):
        self._queues: Dict[str, Deque[Job]] = {}  # tenant_id -> FIFO queue
        self._order: List[str] = []               # tenant rotation order
        self._turn: int = 0                        # index into self._order

    def on_job_arrival(self, job: Job, now: float) -> None:
        if job.tenant_id not in self._queues:
            self._queues[job.tenant_id] = deque()
            self._order.append(job.tenant_id)
        self._queues[job.tenant_id].append(job)

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._order:
            return None

        n = len(self._order)
        # Try each tenant once starting from the current turn pointer
        for offset in range(n):
            idx = (self._turn + offset) % n
            tenant = self._order[idx]
            q = self._queues[tenant]
            if q and q[0].gpus_required <= available_gpus:
                job = q.popleft()
                # Advance turn so the NEXT call starts from the tenant after this one
                self._turn = (idx + 1) % n
                return job

        return None  # no tenant has a runnable head-of-queue job
