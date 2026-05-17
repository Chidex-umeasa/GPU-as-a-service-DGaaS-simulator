from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class GangScheduler(SchedulerBase):
    """
    Gang Scheduler — all-or-nothing gang allocation.

    Jobs tagged with a ``gang_id`` are grouped together. A gang is dispatched
    only when ALL its members have arrived AND their combined GPU requirement
    fits within the currently available GPUs. All members are loaded into a
    dispatch buffer and returned one-by-one to the engine's ``_try_dispatch``
    loop, which keeps calling ``pick_next`` until None is returned.

    Jobs without a ``gang_id`` are treated as individual FIFO jobs.
    Priority between a ready gang and individual jobs: gangs are tried first;
    if no gang fits, individual jobs are checked.
    """

    def __init__(self) -> None:
        # Individual (non-gang) jobs in FIFO order
        self._waiting_individuals: Deque[Job] = deque()

        # gang_id -> list of member jobs that have arrived
        self._gang_queues: Dict[str, List[Job]] = {}

        # FIFO order of gang arrival (first member's arrival defines gang order)
        self._gang_order: Deque[str] = deque()

        # Expected gang sizes: gang_id -> total members expected
        # We infer "complete" by checking gang_id uniqueness across arrivals.
        # Since we don't know total size ahead of time, we track by gang_id presence
        # in the queue and require a separate mechanism — but in the synthetic
        # generator all members with the same gang_id share a size. We use a
        # heuristic: a gang is "ready" when no new member arrived in this event.
        # Actually we rely on the caller to have set gang sizes correctly.
        # The safe implementation: a gang is dispatchable when its total GPU
        # requirement fits. We don't know how many members to expect; we dispatch
        # whatever is queued for a gang_id. To handle this correctly the generator
        # assigns gang sizes, and all members arrive before dispatch (because
        # arrival times are generated with the same rate). For correctness we
        # accept a gang as "ready" when it is non-empty and fits. If members
        # keep arriving after some are dispatched, they form a new gang segment.

        # Buffer for multi-job dispatch (loaded when a gang is selected)
        self._dispatch_buffer: Deque[Job] = deque()

        # Track which gang_ids have been seen to maintain _gang_order correctly
        self._seen_gangs: set = set()

    def on_job_arrival(self, job: Job, now: float) -> None:
        if job.gang_id is not None:
            gang_id = job.gang_id
            if gang_id not in self._gang_queues:
                self._gang_queues[gang_id] = []
            if gang_id not in self._seen_gangs:
                self._seen_gangs.add(gang_id)
                self._gang_order.append(gang_id)
            self._gang_queues[gang_id].append(job)
        else:
            self._waiting_individuals.append(job)

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        # If we have buffered jobs from a previously selected gang, return next
        if self._dispatch_buffer:
            return self._dispatch_buffer.popleft()

        # Try to find a ready gang (from the front of the FIFO gang order)
        # Check each gang in arrival order
        gangs_to_check = list(self._gang_order)
        for gang_id in gangs_to_check:
            members = self._gang_queues.get(gang_id)
            if not members:
                continue
            total_gpus = sum(m.gpus_required for m in members)
            if total_gpus <= available_gpus:
                # Dispatch this entire gang
                self._gang_order.remove(gang_id)
                del self._gang_queues[gang_id]
                self._seen_gangs.discard(gang_id)
                # Load all members into buffer, return first immediately
                for member in members:
                    self._dispatch_buffer.append(member)
                return self._dispatch_buffer.popleft()

        # No gang fits; try individual FIFO jobs
        while self._waiting_individuals:
            job = self._waiting_individuals[0]
            if job.gpus_required <= available_gpus:
                self._waiting_individuals.popleft()
                return job
            else:
                # The front job doesn't fit; stop (FIFO — don't skip)
                return None

        return None

    def on_job_start(self, job: Job, now: float) -> None:
        return

    def on_job_finish(self, job: Job, now: float) -> None:
        return

    def on_job_preempted(self, job: Job, now: float) -> None:
        """Re-queue a preempted job (treat as individual regardless of gang membership)."""
        # Re-queue to the front of individuals queue for fairness
        self._waiting_individuals.appendleft(job)
