from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.priority import PriorityScheduler


class PreemptivePriorityScheduler(PriorityScheduler):
    """
    Preemptive priority scheduler.

    Extends PriorityScheduler with the ability to preempt a low-priority running
    job when a higher-priority job arrives and there are not enough free GPUs.

    The engine calls ``on_preempt_request`` whenever ``pick_next`` returns None
    (i.e., the highest-priority waiting job cannot fit in available GPUs).
    If the scheduler identifies a victim, the engine will preempt it and then
    retry dispatching.
    """

    def __init__(self) -> None:
        super().__init__()
        # job_id -> (priority, Job) for all currently running jobs
        self._running: Dict[str, Tuple[int, Job]] = {}

    # ------------------------------------------------------------------
    # Overridden hooks
    # ------------------------------------------------------------------

    def on_job_start(self, job: Job, now: float) -> None:
        super().on_job_start(job, now)
        self._running[job.job_id] = (job.priority, job)

    def on_job_finish(self, job: Job, now: float) -> None:
        super().on_job_finish(job, now)
        self._running.pop(job.job_id, None)

    def on_job_preempted(self, job: Job, now: float) -> None:
        """Re-queue the preempted job and remove it from running tracking."""
        self._running.pop(job.job_id, None)
        # Re-enqueue as a normal arrival so it competes again by priority
        self.on_job_arrival(job, now)

    # ------------------------------------------------------------------
    # Preemption logic
    # ------------------------------------------------------------------

    def on_preempt_request(
        self,
        now: float,
        available_gpus: int,
        running_jobs: List[Job],
    ) -> Optional[Job]:
        """
        Called by the engine when pick_next returned None (no job fits).

        Check whether a running low-priority job can be preempted to make room
        for the highest-priority waiting job.

        Returns the victim job to preempt, or None if no beneficial preemption
        is possible.
        """
        # Peek at the highest-priority waiting job without popping
        if not self._heap:
            return None

        neg_pri, _seq, waiting_job = self._heap[0]
        waiting_priority = -neg_pri

        if not self._running:
            return None

        # Find the running job with the lowest priority
        victim: Optional[Job] = None
        victim_priority: Optional[int] = None

        for job_id, (pri, job) in self._running.items():
            if victim is None or pri < victim_priority:
                victim = job
                victim_priority = pri

        if victim is None:
            return None

        # Only preempt if:
        # 1. The victim has strictly lower priority than the waiting job
        # 2. Releasing the victim's GPUs would give enough GPUs for the waiting job
        if (
            victim_priority < waiting_priority
            and victim.gpus_required + available_gpus >= waiting_job.gpus_required
        ):
            return victim

        return None
