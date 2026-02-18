"""Tests for SimEngine discrete-event loop."""
from __future__ import annotations

import pytest

from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.metrics.collector import MetricsCollector
from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.sim.engine import Job, SimEngine


def _run(jobs, total_gpus=8):
    engine = SimEngine()
    cluster = GPUCluster(total_gpus=total_gpus)
    scheduler = FIFOScheduler()
    metrics = MetricsCollector()
    finished = engine.run(jobs=jobs, cluster=cluster, scheduler=scheduler, metrics=metrics)
    return engine, cluster, metrics, finished


def test_all_jobs_finish():
    jobs = [
        Job("j0", 0.0, 5.0, gpus_required=1),
        Job("j1", 1.0, 3.0, gpus_required=2),
        Job("j2", 2.0, 7.0, gpus_required=1),
    ]
    _, _, metrics, _ = _run(jobs)
    assert metrics.finishes == 3
    assert metrics.arrivals == 3


def test_finish_time_after_start_time():
    jobs = [Job("j0", 0.0, 5.0)]
    _, _, _, finished = _run(jobs)
    j = finished["j0"]
    assert j.finish_time is not None
    assert j.start_time is not None
    assert j.finish_time >= j.start_time


def test_start_time_at_or_after_arrival():
    jobs = [Job("j0", 3.5, 2.0)]
    _, _, _, finished = _run(jobs)
    j = finished["j0"]
    assert j.start_time >= j.arrival_time


def test_gpu_constraint_respected():
    """A job requiring more GPUs than available must wait for a running job to finish."""
    jobs = [
        Job("j_big", 0.0, 10.0, gpus_required=6),   # uses 6/8 GPUs
        Job("j_also_big", 0.0, 3.0, gpus_required=4),  # needs 4, only 2 left → waits
    ]
    _, _, _, finished = _run(jobs, total_gpus=8)
    # j_also_big must start after j_big finishes (at t=10)
    assert finished["j_also_big"].start_time >= 10.0


def test_duplicate_job_id_raises():
    jobs = [Job("j0", 0.0, 1.0), Job("j0", 1.0, 1.0)]
    with pytest.raises(ValueError, match="Duplicate job_id"):
        _run(jobs)


def test_engine_time_advances():
    jobs = [Job("j0", 0.0, 5.0), Job("j1", 0.0, 3.0)]
    engine, _, _, _ = _run(jobs)
    assert engine.time > 0.0


def test_end_time_stops_simulation():
    """With end_time set, jobs arriving after it should not be processed."""
    jobs = [
        Job("j_early", 0.0, 1.0),
        Job("j_late", 100.0, 1.0),
    ]
    engine = SimEngine()
    cluster = GPUCluster(total_gpus=4)
    scheduler = FIFOScheduler()
    metrics = MetricsCollector()
    engine.run(jobs=jobs, cluster=cluster, scheduler=scheduler,
               metrics=metrics, end_time=10.0)
    assert metrics.finishes == 1  # only j_early should complete
