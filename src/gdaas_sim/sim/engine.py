from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import heapq


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    job_id: str = field(compare=False)


@dataclass
class Job:
    job_id: str
    arrival_time: float
    duration: float
    gpus_required: int = 1
    tenant_id: str = "tenant_0"  # NEW: multi-tenant support

    # SLA fields
    deadline: Optional[float] = None      # absolute simulation time deadline
    max_wait: Optional[float] = None      # max allowed queue wait

    # Scheduling priority: higher integer = higher priority (used by PriorityScheduler)
    priority: int = 0

    # Filled by simulator
    start_time: Optional[float] = None
    finish_time: Optional[float] = None


class SimEngine:
    """
    Discrete-event simulation engine for scheduling jobs on a GPU cluster.

    Event types:
      - "ARRIVE": job enters queue
      - "FINISH": job completes and frees GPUs
    """

    def __init__(self):
        self.time: float = 0.0
        self._seq: int = 0
        self._pq: List[Event] = []  # priority queue of events
        self.jobs: Dict[str, Job] = {}

    def schedule_event(self, time: float, kind: str, job_id: str) -> None:
        self._seq += 1
        heapq.heappush(self._pq, Event(time=time, seq=self._seq, kind=kind, job_id=job_id))

    def run(
        self,
        jobs: List[Job],
        cluster,
        scheduler,
        metrics,
        end_time: Optional[float] = None,
    ) -> Dict[str, Job]:
        """
        Run the simulation until the event queue is empty or until end_time.
        """
        # Register jobs + schedule arrivals
        for j in jobs:
            if j.job_id in self.jobs:
                raise ValueError(f"Duplicate job_id: {j.job_id}")
            self.jobs[j.job_id] = j
            self.schedule_event(j.arrival_time, "ARRIVE", j.job_id)

        # Main loop
        while self._pq:
            ev = heapq.heappop(self._pq)

            if end_time is not None and ev.time > end_time:
                break

            self.time = ev.time

            if ev.kind == "ARRIVE":
                job = self.jobs[ev.job_id]
                scheduler.on_job_arrival(job, now=self.time)
                metrics.on_job_arrival(job, now=self.time)

            elif ev.kind == "FINISH":
                job = self.jobs[ev.job_id]
                cluster.release(job.gpus_required)
                job.finish_time = self.time
                scheduler.on_job_finish(job, now=self.time)
                metrics.on_job_finish(job, now=self.time)

            else:
                raise ValueError(f"Unknown event kind: {ev.kind}")

            # After each event, attempt to schedule as many jobs as possible
            self._try_dispatch(cluster, scheduler, metrics)

        # Finalize metrics with end-of-sim time
        metrics.finalize(now=self.time, cluster=cluster)
        return self.jobs

    def _try_dispatch(self, cluster, scheduler, metrics) -> None:
        """
        Keep starting jobs while the scheduler has runnable jobs and GPUs are available.
        """
        while True:
            job = scheduler.pick_next(now=self.time, available_gpus=cluster.available_gpus)
            if job is None:
                return

            if job.gpus_required > cluster.available_gpus:
                # Scheduler gave a non-runnable job; stop to avoid infinite loops.
                scheduler.on_job_blocked(job, now=self.time)
                return

            # Start the job
            cluster.allocate(job.gpus_required)
            job.start_time = self.time
            job.finish_time = None
            scheduler.on_job_start(job, now=self.time)   # NEW: notify scheduler
            metrics.on_job_start(job, now=self.time)

            # Schedule finish
            self.schedule_event(self.time + job.duration, "FINISH", job.job_id)
