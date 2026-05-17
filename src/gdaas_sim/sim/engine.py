from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
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
    tenant_id: str = "tenant_0"

    # SLA fields
    deadline: Optional[float] = None
    max_wait: Optional[float] = None

    # Scheduling priority: higher integer = higher priority
    priority: int = 0

    # Filled by simulator
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    # v0.3.0 additions (all with defaults for backwards compatibility)
    gpu_type_preference: Optional[str] = None   # preferred GPU type
    allocated_gpu_type: Optional[str] = None    # set by engine after allocation
    spot: bool = False                          # spot/preemptible-anytime job
    preemptible: bool = False                   # can be preempted (includes spot)
    gang_id: Optional[str] = None              # gang scheduling group ID
    first_start_time: Optional[float] = None   # set on first dispatch, never reset
    remaining_duration: Optional[float] = None # set if preempted mid-run
    preemption_count: int = 0                  # how many times preempted
    cost_per_gpu_hour: float = 1.0             # cost rate (updated by engine from cluster)


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
        self._pq: List[Event] = []
        self.jobs: Dict[str, Job] = {}
        self._running_jobs: Dict[str, Job] = {}
        self._finish_event_seqs: Dict[str, int] = {}
        self._cancelled: Set[int] = set()

    def schedule_event(self, time: float, kind: str, job_id: str) -> int:
        """Schedule an event and return its sequence number."""
        self._seq += 1
        seq = self._seq
        heapq.heappush(self._pq, Event(time=time, seq=seq, kind=kind, job_id=job_id))
        return seq

    def preempt_job(self, job_id: str, cluster, scheduler, metrics) -> None:
        """Preempt a running job: cancel its FINISH event, release GPUs, re-queue it."""
        job = self._running_jobs.get(job_id)
        if job is None:
            return  # already finished or not running

        # Cancel the FINISH event
        finish_seq = self._finish_event_seqs.get(job_id)
        if finish_seq is not None:
            self._cancelled.add(finish_seq)

        # Compute remaining duration
        elapsed = self.time - (job.start_time if job.start_time is not None else self.time)
        job.remaining_duration = max(0.0, job.duration - elapsed)

        # Increment preemption count
        job.preemption_count += 1

        # Release GPUs (with gpu_type for heterogeneous clusters)
        cluster.release(job.gpus_required, job.allocated_gpu_type)

        # Notify metrics
        metrics.on_job_preempted(job, now=self.time)

        # Remove from tracking dicts
        self._running_jobs.pop(job_id, None)
        self._finish_event_seqs.pop(job_id, None)

        # Reset job scheduling state
        job.start_time = None
        job.finish_time = None
        job.allocated_gpu_type = None

        # Re-queue the job via scheduler
        scheduler.on_job_preempted(job, now=self.time)

    def run(
        self,
        jobs,
        cluster,
        scheduler,
        metrics,
        end_time: Optional[float] = None,
    ) -> Dict[str, Job]:
        """Run the simulation until the event queue is empty or until end_time."""
        for j in jobs:
            if j.job_id in self.jobs:
                raise ValueError(f"Duplicate job_id: {j.job_id}")
            self.jobs[j.job_id] = j
            self.schedule_event(j.arrival_time, "ARRIVE", j.job_id)

        while self._pq:
            ev = heapq.heappop(self._pq)

            # Skip cancelled events
            if ev.seq in self._cancelled:
                self._cancelled.discard(ev.seq)
                continue

            if end_time is not None and ev.time > end_time:
                break

            self.time = ev.time

            if ev.kind == "ARRIVE":
                job = self.jobs[ev.job_id]
                scheduler.on_job_arrival(job, now=self.time)
                metrics.on_job_arrival(job, now=self.time)

            elif ev.kind == "FINISH":
                job = self.jobs[ev.job_id]
                # Use pop with None to handle edge case where job was already preempted
                self._running_jobs.pop(ev.job_id, None)
                self._finish_event_seqs.pop(ev.job_id, None)
                cluster.release(job.gpus_required, job.allocated_gpu_type)
                job.finish_time = self.time
                scheduler.on_job_finish(job, now=self.time)
                metrics.on_job_finish(job, now=self.time)

            else:
                raise ValueError(f"Unknown event kind: {ev.kind}")

            self._try_dispatch(cluster, scheduler, metrics)

        metrics.finalize(now=self.time, cluster=cluster)
        return self.jobs

    def _try_dispatch(self, cluster, scheduler, metrics) -> None:
        """Keep starting jobs while the scheduler has runnable jobs and GPUs are available."""
        while True:
            job = scheduler.pick_next(now=self.time, available_gpus=cluster.available_gpus)

            if job is None:
                # Check if scheduler supports preemption requests
                if hasattr(scheduler, 'on_preempt_request'):
                    running_list = list(self._running_jobs.values())
                    victim = scheduler.on_preempt_request(
                        now=self.time,
                        available_gpus=cluster.available_gpus,
                        running_jobs=running_list,
                    )
                    if victim is not None:
                        self.preempt_job(victim.job_id, cluster, scheduler, metrics)
                        continue
                return

            if job.gpus_required > cluster.available_gpus:
                scheduler.on_job_blocked(job, now=self.time)
                return

            # Allocate GPUs (gpu_type-aware for heterogeneous clusters)
            allocated_type = cluster.allocate(
                job.gpus_required,
                getattr(job, 'gpu_type_preference', None),
            )
            job.allocated_gpu_type = allocated_type

            # Update cost rate from cluster if available
            if hasattr(cluster, 'cost_per_gpu_hour_of') and allocated_type is not None:
                job.cost_per_gpu_hour = cluster.cost_per_gpu_hour_of(allocated_type)

            job.start_time = self.time
            job.finish_time = None

            # Set first_start_time only on first dispatch (never reset)
            if job.first_start_time is None:
                job.first_start_time = self.time

            # Use remaining_duration if set (job was preempted before)
            effective_dur = (
                job.remaining_duration
                if job.remaining_duration is not None
                else job.duration
            )
            job.remaining_duration = None  # clear after use

            scheduler.on_job_start(job, now=self.time)
            metrics.on_job_start(job, now=self.time)

            finish_seq = self.schedule_event(self.time + effective_dur, "FINISH", job.job_id)
            self._running_jobs[job.job_id] = job
            self._finish_event_seqs[job.job_id] = finish_seq
