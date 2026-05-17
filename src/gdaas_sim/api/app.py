"""
GDaaS Simulator REST API
========================
Run with: uvicorn gdaas_sim.api.app:app --reload

Requires:
    pip install "gdaas-sim[api]"
    or: pip install fastapi uvicorn pydantic
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as _e:
    raise ImportError(
        "FastAPI and Pydantic are required to run the GDaaS Simulator API. "
        "Install them with: pip install 'gdaas-sim[api]' "
        "or: pip install fastapi uvicorn pydantic"
    ) from _e

from gdaas_sim import __version__
from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.metrics.collector import MetricsCollector
from gdaas_sim.scheduler.backfill import EASYBackfillScheduler
from gdaas_sim.scheduler.edf import EDFScheduler
from gdaas_sim.scheduler.fair_share import TenantFairScheduler
from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.scheduler.gang import GangScheduler
from gdaas_sim.scheduler.preemptive_priority import PreemptivePriorityScheduler
from gdaas_sim.scheduler.priority import PriorityScheduler
from gdaas_sim.scheduler.round_robin import RoundRobinScheduler
from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.scheduler.weighted_fair import WeightedFairShareScheduler
from gdaas_sim.sim.engine import SimEngine
from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GDaaS Simulator API",
    description="GPU-as-a-Service Discrete-Event Simulator REST API",
    version=__version__,
)

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler registry
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULER_DESCRIPTIONS: Dict[str, str] = {
    "fifo":               "First In, First Out — simple queue order",
    "sjf":                "Shortest Job First — minimizes average wait time",
    "fair":               "Tenant Fair Share — equal GPU allocation across tenants",
    "edf":                "Earliest Deadline First — minimizes deadline violations",
    "priority":           "Priority-based — higher priority jobs run first",
    "rr":                 "Round Robin — rotates dispatch across tenants",
    "backfill":           "EASY Backfill — SLURM-style backfill for high utilization",
    "preemptive_priority":"Preemptive Priority — preempts low-priority jobs for urgent ones",
    "weighted_fair":      "Weighted Fair Share — fair sharing with tenant-specific weights",
    "gang":               "Gang Scheduler — all-or-nothing gang (MPI) job allocation",
}


def _build_scheduler(name: str, tenant_weights: Optional[Dict[str, float]] = None):
    """Instantiate the named scheduler."""
    if name == "fifo":
        return FIFOScheduler()
    elif name == "sjf":
        return SJFScheduler()
    elif name == "fair":
        return TenantFairScheduler()
    elif name == "edf":
        return EDFScheduler()
    elif name == "priority":
        return PriorityScheduler()
    elif name == "rr":
        return RoundRobinScheduler()
    elif name == "backfill":
        return EASYBackfillScheduler()
    elif name == "preemptive_priority":
        return PreemptivePriorityScheduler()
    elif name == "weighted_fair":
        return WeightedFairShareScheduler(tenant_weights=tenant_weights)
    elif name == "gang":
        return GangScheduler()
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scheduler '{name}'. "
                   f"Valid choices: {list(SCHEDULER_DESCRIPTIONS.keys())}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    scheduler: str = Field("fifo", description="Scheduler algorithm to use")
    total_gpus: int = Field(16, gt=0, description="Total GPUs in the cluster")
    n_jobs: int = Field(300, gt=0, description="Number of jobs in the workload")
    arrival_rate: float = Field(0.5, gt=0.0, description="Average job arrival rate (jobs/time unit)")
    n_tenants: int = Field(3, gt=0, description="Number of tenants")
    seed: int = Field(42, description="Random seed for reproducibility")
    spot_frac: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of spot/preemptible jobs")
    gang_frac: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of jobs in gangs")
    gpu_type_names: Optional[List[str]] = Field(None, description="GPU type names (heterogeneous)")
    tenant_weights: Optional[Dict[str, float]] = Field(
        None, description="Per-tenant weights for weighted_fair scheduler"
    )


class SimulateResponse(BaseModel):
    scheduler: str
    utilization: float
    avg_wait: float
    p95_wait: float
    jobs_finished: int
    sla_wait_violations: int
    sla_deadline_violations: int
    jain_fairness: Optional[float]
    drf_fairness: Optional[float]
    total_cost: float
    tenant_cost: Dict[str, float]
    preemptions: int
    sim_end_time: float


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    """Health check — returns version info."""
    return {
        "status": "ok",
        "name": "GDaaS Simulator API",
        "version": __version__,
    }


@app.get("/schedulers", summary="List available schedulers")
def list_schedulers():
    """Return all available scheduler algorithms with descriptions."""
    return {
        "schedulers": [
            {"name": name, "description": desc}
            for name, desc in SCHEDULER_DESCRIPTIONS.items()
        ]
    }


@app.post("/simulate", response_model=SimulateResponse, summary="Run a simulation")
def simulate(req: SimulateRequest) -> SimulateResponse:
    """
    Run a discrete-event simulation with the specified scheduler and workload.
    Returns key performance metrics.
    """
    # Build workload
    cfg = WorkloadConfig(
        n_jobs=req.n_jobs,
        arrival_rate=req.arrival_rate,
        n_tenants=req.n_tenants,
        seed=req.seed,
        spot_frac=req.spot_frac,
        gang_frac=req.gang_frac,
        gpu_type_names=req.gpu_type_names,
    )
    jobs = generate_synthetic(cfg)

    # Build cluster and components
    cluster = GPUCluster(total_gpus=req.total_gpus)
    metrics = MetricsCollector()
    engine = SimEngine()
    scheduler = _build_scheduler(req.scheduler, req.tenant_weights)

    # Run simulation
    engine.run(jobs=jobs, cluster=cluster, scheduler=scheduler, metrics=metrics)

    sim_end = engine.time
    util = (
        metrics.busy_gpu_time / (req.total_gpus * sim_end) if sim_end > 0 else 0.0
    )

    wait_sorted = sorted(metrics.wait_times) if metrics.wait_times else []
    avg_wait = sum(wait_sorted) / len(wait_sorted) if wait_sorted else 0.0
    p95_idx = int(0.95 * (len(wait_sorted) - 1)) if len(wait_sorted) > 1 else 0
    p95_wait = wait_sorted[p95_idx] if wait_sorted else 0.0

    total_cost = sum(metrics.tenant_cost.values())

    return SimulateResponse(
        scheduler=req.scheduler,
        utilization=round(util, 6),
        avg_wait=round(avg_wait, 4),
        p95_wait=round(p95_wait, 4),
        jobs_finished=metrics.finishes,
        sla_wait_violations=metrics.sla_wait_violations,
        sla_deadline_violations=metrics.sla_deadline_violations,
        jain_fairness=metrics.jain_fairness(),
        drf_fairness=metrics.drf_fairness(),
        total_cost=round(total_cost, 4),
        tenant_cost={k: round(v, 4) for k, v in metrics.tenant_cost.items()},
        preemptions=metrics.preemptions,
        sim_end_time=round(sim_end, 4),
    )
