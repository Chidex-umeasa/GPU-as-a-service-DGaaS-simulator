"""End-to-end integration tests: all schedulers must run cleanly to completion."""
from __future__ import annotations

import pytest

from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.metrics.collector import MetricsCollector
from gdaas_sim.scheduler.backfill import EASYBackfillScheduler
from gdaas_sim.scheduler.edf import EDFScheduler
from gdaas_sim.scheduler.fair_share import TenantFairScheduler
from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.scheduler.priority import PriorityScheduler
from gdaas_sim.scheduler.round_robin import RoundRobinScheduler
from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.sim.engine import SimEngine
from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic

ALL_SCHEDULERS = [
    ("fifo", FIFOScheduler),
    ("sjf", SJFScheduler),
    ("fair", TenantFairScheduler),
    ("edf", EDFScheduler),
    ("priority", PriorityScheduler),
    ("rr", RoundRobinScheduler),
    ("backfill", EASYBackfillScheduler),
]


def _run_sim(scheduler_cls, total_gpus=8, n_jobs=100, seed=42):
    cfg = WorkloadConfig(n_jobs=n_jobs, arrival_rate=0.5, n_tenants=3, seed=seed)
    jobs = generate_synthetic(cfg)
    cluster = GPUCluster(total_gpus=total_gpus)
    metrics = MetricsCollector()
    engine = SimEngine()
    scheduler = scheduler_cls()
    finished = engine.run(jobs=jobs, cluster=cluster, scheduler=scheduler, metrics=metrics)
    return engine, metrics, finished


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_all_jobs_complete(name, scheduler_cls):
    _, metrics, _ = _run_sim(scheduler_cls)
    assert metrics.arrivals == 100, f"[{name}] arrivals mismatch"
    assert metrics.finishes == 100, f"[{name}] not all jobs finished"


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_wait_times_count_equals_finishes(name, scheduler_cls):
    """Regression: len(wait_times) must equal finishes (catches double-append bug)."""
    _, metrics, _ = _run_sim(scheduler_cls)
    assert len(metrics.wait_times) == metrics.finishes, (
        f"[{name}] wait_times has {len(metrics.wait_times)} entries "
        f"but {metrics.finishes} jobs finished — possible double-append bug"
    )


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_turnaround_count_equals_finishes(name, scheduler_cls):
    """Regression: len(turnaround_times) must equal finishes."""
    _, metrics, _ = _run_sim(scheduler_cls)
    assert len(metrics.turnaround_times) == metrics.finishes, (
        f"[{name}] turnaround_times has {len(metrics.turnaround_times)} entries "
        f"but {metrics.finishes} jobs finished"
    )


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_all_wait_times_non_negative(name, scheduler_cls):
    _, metrics, _ = _run_sim(scheduler_cls)
    negatives = [w for w in metrics.wait_times if w < 0]
    assert not negatives, f"[{name}] negative wait times: {negatives[:5]}"


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_utilization_in_valid_range(name, scheduler_cls):
    engine, metrics, _ = _run_sim(scheduler_cls, total_gpus=8)
    sim_end = engine.time
    if sim_end > 0:
        util = metrics.busy_gpu_time / (8 * sim_end)
        assert 0.0 <= util <= 1.0, f"[{name}] utilization {util:.3f} out of [0,1]"


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_finish_time_after_start_time(name, scheduler_cls):
    _, _, finished = _run_sim(scheduler_cls)
    for job in finished.values():
        if job.start_time is not None and job.finish_time is not None:
            assert job.finish_time >= job.start_time, (
                f"[{name}] job {job.job_id}: finish {job.finish_time} < start {job.start_time}"
            )


@pytest.mark.parametrize("name,scheduler_cls", ALL_SCHEDULERS)
def test_start_time_after_arrival_time(name, scheduler_cls):
    _, _, finished = _run_sim(scheduler_cls)
    for job in finished.values():
        if job.start_time is not None:
            assert job.start_time >= job.arrival_time - 1e-9, (
                f"[{name}] job {job.job_id}: started before arrival"
            )


def test_backfill_improves_utilization_vs_fifo_under_mixed_load():
    """
    With a mix of large and small jobs, backfill should achieve utilization
    at least as good as FIFO (it can only be equal or better).
    """
    cfg = WorkloadConfig(
        n_jobs=200,
        arrival_rate=0.8,
        gpu_req_choices=(1, 1, 4, 4, 8),  # large jobs that can block FIFO
        seed=42,
    )

    def measure_util(scheduler_cls):
        jobs = generate_synthetic(cfg)
        cluster = GPUCluster(total_gpus=8)
        metrics = MetricsCollector()
        engine = SimEngine()
        engine.run(jobs=jobs, cluster=cluster,
                   scheduler=scheduler_cls(), metrics=metrics)
        sim_end = engine.time
        return metrics.busy_gpu_time / (8 * sim_end) if sim_end > 0 else 0.0

    fifo_util = measure_util(FIFOScheduler)
    backfill_util = measure_util(EASYBackfillScheduler)
    assert backfill_util >= fifo_util - 0.01, (
        f"Backfill utilization {backfill_util:.3f} unexpectedly much worse "
        f"than FIFO {fifo_util:.3f}"
    )
