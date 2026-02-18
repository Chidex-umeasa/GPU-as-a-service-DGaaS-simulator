from __future__ import annotations

import heapq
from collections import deque
from typing import Deque, List, Optional, Tuple

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class EASYBackfillScheduler(SchedulerBase):
    """
    EASY (Extensible Argonne Scheduling sYstem) Backfill Scheduler.

    This is the algorithm used by SLURM, the dominant scheduler in HPC clusters
    (runs >60% of the Top500 supercomputers).

    Core idea
    ---------
    Standard FIFO starves the cluster when the head-of-queue job is too large
    to run with currently available GPUs.  EASY Backfill fixes this by:

    1. Giving the head-of-queue job (the "pivot") a **reservation**: computing
       the earliest future time at which enough GPUs will be free for it.
    2. Allowing smaller jobs to **backfill** around the pivot — i.e., run now —
       provided they will finish *before* the pivot's reservation time.  This
       guarantees the pivot starts no later than its reserved time.

    Algorithm (per pick_next call)
    --------------------------------
    a) If the pivot fits current available GPUs → dispatch it immediately.
    b) Otherwise, compute shadow_time = earliest time pivot can start
       (walk running-job finish times in order, accumulate freed GPUs).
    c) Scan queue[1:] for a "backfill candidate":
           gpus_required <= available_gpus
           AND now + duration <= shadow_time
       Dispatch the first eligible candidate found.
    d) If no candidate qualifies → return None (stall).

    Complexity
    ----------
    pick_next: O(n log n) worst case (shadow_time sort) + O(n) queue scan.
    on_job_finish: O(n) list comprehension + O(n log n) heapify (bounded queue).
    For production use with millions of jobs, replace the list+heapify with a
    sorted container (e.g. SortedList from sortedcontainers).
    """

    def __init__(self):
        self._queue: Deque[Job] = deque()
        # min-heap of (expected_finish_time, job_id, gpus_required)
        self._running: List[Tuple[float, str, int]] = []

    # ------------------------------------------------------------------
    # SchedulerBase hooks
    # ------------------------------------------------------------------

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._queue.append(job)

    def on_job_start(self, job: Job, now: float) -> None:
        """Record a newly started job so we can compute future GPU availability."""
        expected_finish = now + job.duration
        heapq.heappush(self._running, (expected_finish, job.job_id, job.gpus_required))

    def on_job_finish(self, job: Job, now: float) -> None:
        """Remove a completed job from the running set."""
        self._running = [e for e in self._running if e[1] != job.job_id]
        heapq.heapify(self._running)

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._queue:
            return None

        pivot = self._queue[0]

        # Case A: pivot fits right now → dispatch it
        if pivot.gpus_required <= available_gpus:
            self._queue.popleft()
            return pivot

        # Case B: pivot cannot run → compute its shadow (reservation) time
        shadow_time = self._compute_shadow_time(
            available_gpus=available_gpus,
            pivot_gpus=pivot.gpus_required,
        )

        # Scan the rest of the queue for a valid backfill candidate
        for i in range(1, len(self._queue)):
            candidate = self._queue[i]
            fits_now = candidate.gpus_required <= available_gpus
            finishes_before_shadow = (now + candidate.duration) <= shadow_time
            if fits_now and finishes_before_shadow:
                del self._queue[i]  # O(n) deque deletion — acceptable at this scale
                return candidate

        return None  # nothing can run without violating the pivot's reservation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_shadow_time(self, available_gpus: int, pivot_gpus: int) -> float:
        """
        Return the earliest future time at which ``pivot_gpus`` GPUs will be free.

        Walk running jobs in finish-time order, accumulating the GPUs they release.
        Stop as soon as the cumulative freed GPUs meet or exceed ``pivot_gpus``.
        """
        free = available_gpus
        for finish_time, _job_id, gpus in sorted(self._running):
            free += gpus
            if free >= pivot_gpus:
                return finish_time
        # Fallback: all running jobs finish but still not enough GPUs (shouldn't
        # happen in a correct simulation, but return a large sentinel just in case).
        return float("inf")
