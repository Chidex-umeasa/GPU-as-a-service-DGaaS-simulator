from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from gdaas_sim.sim.engine import Job


class SchedulerBase(ABC):
    @abstractmethod
    def on_job_arrival(self, job: Job, now: float) -> None:
        ...

    @abstractmethod
    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        ...

    # Optional hooks
    def on_job_start(self, job: Job, now: float) -> None:
        return

    def on_job_finish(self, job: Job, now: float) -> None:
        return

    def on_job_blocked(self, job: Job, now: float) -> None:
        return
