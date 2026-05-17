from __future__ import annotations

from typing import Dict, Optional

from gdaas_sim.sim.engine import Job
from gdaas_sim.scheduler.base import SchedulerBase


class WeightedFairShareScheduler(SchedulerBase):
    """
    Weighted Fair Share Scheduler.

    Each tenant has a weight (default 1.0). The scheduler tracks the
    normalized GPU-time debt for each tenant:

        debt[tenant] = cumulative_gpu_time[tenant] / weight[tenant]

    At each scheduling decision, the scheduler picks a runnable job whose
    tenant has the lowest debt, achieving weighted fair sharing across tenants.
    """

    def __init__(self, tenant_weights: Optional[Dict[str, float]] = None) -> None:
        self._weights: Dict[str, float] = dict(tenant_weights) if tenant_weights else {}
        self._waiting: Dict[str, Job] = {}           # job_id -> Job
        self._gpu_time_debt: Dict[str, float] = {}   # tenant_id -> normalized GPU time
        self._running_gpu: Dict[str, int] = {}        # tenant_id -> currently running GPUs

    def _get_weight(self, tenant_id: str) -> float:
        return self._weights.get(tenant_id, 1.0)

    def on_job_arrival(self, job: Job, now: float) -> None:
        self._waiting[job.job_id] = job

    def pick_next(self, now: float, available_gpus: int) -> Optional[Job]:
        if not self._waiting:
            return None

        chosen: Optional[Job] = None
        chosen_debt: Optional[float] = None

        for job in self._waiting.values():
            if job.gpus_required > available_gpus:
                continue
            debt = self._gpu_time_debt.get(job.tenant_id, 0.0)
            if chosen is None or debt < chosen_debt:
                chosen = job
                chosen_debt = debt

        if chosen is None:
            return None

        del self._waiting[chosen.job_id]
        return chosen

    def on_job_start(self, job: Job, now: float) -> None:
        tenant = job.tenant_id
        self._running_gpu[tenant] = self._running_gpu.get(tenant, 0) + job.gpus_required

    def on_job_finish(self, job: Job, now: float) -> None:
        tenant = job.tenant_id
        if job.start_time is not None:
            runtime = now - job.start_time
            weight = self._get_weight(tenant)
            self._gpu_time_debt[tenant] = (
                self._gpu_time_debt.get(tenant, 0.0) + (runtime * job.gpus_required) / weight
            )
        self._running_gpu[tenant] = max(0, self._running_gpu.get(tenant, 0) - job.gpus_required)

    def on_job_preempted(self, job: Job, now: float) -> None:
        """Re-queue preempted job and record partial GPU-time debt."""
        tenant = job.tenant_id
        if job.start_time is not None:
            elapsed = now - job.start_time
            weight = self._get_weight(tenant)
            self._gpu_time_debt[tenant] = (
                self._gpu_time_debt.get(tenant, 0.0) + (elapsed * job.gpus_required) / weight
            )
        self._running_gpu[tenant] = max(0, self._running_gpu.get(tenant, 0) - job.gpus_required)
        # Re-queue the job
        self._waiting[job.job_id] = job
