from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from gdaas_sim.sim.engine import Job


class SchedulerBase(ABC):
    @abstractmethod
    def on_job_arrival(self, job: Job, now: float) -> None:
        ...

    @abstractmethod
    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        ...

    # Optional hooks — no-op defaults
    def on_job_start(self, job: Job, now: float) -> None:
        return

    def on_job_finish(self, job: Job, now: float) -> None:
        return

    def on_job_blocked(self, job: Job, now: float) -> None:
        return

    def on_job_preempted(self, job: Job, now: float) -> None:
        """Called when a job is preempted. Default: no-op."""
        return

    # Note: on_preempt_request is NOT defined here by default.
    # Only schedulers that support preemption implement it.
    # The engine uses hasattr() to detect its presence.
    # Signature when implemented:
    #   def on_preempt_request(self, now: float, available_gpus: int,
    #                          running_jobs: List[Job]) -> Optional[Job]: ...
