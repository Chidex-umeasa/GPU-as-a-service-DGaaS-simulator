"""Tests for MetricsCollector — including regression tests for the duplicate-append bug."""
from __future__ import annotations

import pytest

from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.metrics.collector import MetricsCollector
from gdaas_sim.sim.engine import Job


def _finished_job(
    job_id: str,
    arrival: float,
    start: float,
    finish: float,
    gpus: int = 1,
    tenant: str = "t0",
    deadline: float | None = None,
    max_wait: float | None = None,
) -> Job:
    j = Job(
        job_id,
        arrival_time=arrival,
        duration=finish - start,
        gpus_required=gpus,
        tenant_id=tenant,
        deadline=deadline,
        max_wait=max_wait,
    )
    j.start_time = start
    j.finish_time = finish
    return j


# -------------------------------------------------------------------------
# BUG-FIX REGRESSION: each job must appear exactly ONCE in the lists
# -------------------------------------------------------------------------

def test_wait_times_no_duplicate():
    """REGRESSION: on_job_finish must append wait_time exactly once per job."""
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=3.0, finish=8.0)
    mc.on_job_finish(job, now=8.0)
    assert len(mc.wait_times) == 1, (
        f"Expected 1 entry in wait_times, got {len(mc.wait_times)} — "
        "duplicate-append bug may not be fixed"
    )
    assert mc.wait_times[0] == pytest.approx(3.0)


def test_turnaround_no_duplicate():
    """REGRESSION: on_job_finish must append turnaround exactly once per job."""
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=3.0, finish=8.0)
    mc.on_job_finish(job, now=8.0)
    assert len(mc.turnaround_times) == 1, (
        f"Expected 1 entry in turnaround_times, got {len(mc.turnaround_times)} — "
        "duplicate-append bug may not be fixed"
    )
    assert mc.turnaround_times[0] == pytest.approx(8.0)


def test_multiple_jobs_correct_count():
    """n jobs → exactly n entries in wait_times and turnaround_times."""
    mc = MetricsCollector()
    for i in range(5):
        job = _finished_job(f"j{i}", arrival=float(i), start=float(i) + 1.0, finish=float(i) + 4.0)
        mc.on_job_finish(job, now=float(i) + 4.0)
    assert len(mc.wait_times) == 5
    assert len(mc.turnaround_times) == 5


# -------------------------------------------------------------------------
# SLA violation tracking
# -------------------------------------------------------------------------

def test_sla_wait_violation_counted():
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=6.0, finish=8.0, max_wait=5.0)
    mc.on_job_finish(job, now=8.0)
    assert mc.sla_wait_violations == 1


def test_sla_wait_no_violation_when_within_limit():
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=3.0, finish=8.0, max_wait=5.0)
    mc.on_job_finish(job, now=8.0)
    assert mc.sla_wait_violations == 0


def test_sla_deadline_violation_counted():
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=1.0, finish=9.0, deadline=8.0)
    mc.on_job_finish(job, now=9.0)
    assert mc.sla_deadline_violations == 1


def test_sla_deadline_no_violation_when_on_time():
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=1.0, finish=7.0, deadline=8.0)
    mc.on_job_finish(job, now=7.0)
    assert mc.sla_deadline_violations == 0


# -------------------------------------------------------------------------
# Jain's Fairness Index
# -------------------------------------------------------------------------

def test_jain_fairness_equal_tenants():
    mc = MetricsCollector()
    mc.tenant_gpu_time = {"t0": 100.0, "t1": 100.0, "t2": 100.0}
    assert mc.jain_fairness() == pytest.approx(1.0)


def test_jain_fairness_single_tenant():
    mc = MetricsCollector()
    mc.tenant_gpu_time = {"t0": 50.0}
    assert mc.jain_fairness() == pytest.approx(1.0)


def test_jain_fairness_returns_none_when_empty():
    mc = MetricsCollector()
    assert mc.jain_fairness() is None


def test_jain_fairness_unequal_tenants_less_than_one():
    mc = MetricsCollector()
    mc.tenant_gpu_time = {"t0": 90.0, "t1": 10.0}
    jain = mc.jain_fairness()
    assert jain is not None
    assert 0.0 < jain < 1.0


# -------------------------------------------------------------------------
# Utilization
# -------------------------------------------------------------------------

def test_utilization_correct_integration():
    """
    Simulate a job using 2 of 4 GPUs for 10 time units.
    Expected utilization = (2 * 10) / (4 * 10) = 0.5
    """
    from gdaas_sim.sim.engine import Job as _Job
    mc = MetricsCollector()
    cluster = GPUCluster(total_gpus=4)

    j = _Job("j0", arrival_time=0.0, duration=10.0, gpus_required=2)
    j.start_time = 0.0
    j.finish_time = 10.0

    cluster.allocate(2)
    mc.on_job_start(j, now=0.0)

    cluster.release(2)
    mc.on_job_finish(j, now=10.0)

    mc.finalize(now=10.0, cluster=cluster)
    util = mc.busy_gpu_time / (4 * 10)
    assert util == pytest.approx(0.5)


def test_utilization_zero_at_start():
    mc = MetricsCollector()
    cluster = GPUCluster(total_gpus=4)
    assert mc.snapshot_utilization(now=0.0, cluster=cluster) == pytest.approx(0.0)


# -------------------------------------------------------------------------
# Per-tenant GPU time
# -------------------------------------------------------------------------

def test_tenant_gpu_time_accumulated():
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=0.0, finish=5.0, gpus=2, tenant="t0")
    mc.on_job_finish(job, now=5.0)
    # runtime=5, gpus=2 → gpu_time=10
    assert mc.tenant_gpu_time.get("t0", 0.0) == pytest.approx(10.0)


def test_arrivals_starts_finishes_counted():
    mc = MetricsCollector()
    job = _finished_job("j0", arrival=0.0, start=1.0, finish=6.0)
    mc.on_job_arrival(job, now=0.0)
    mc.on_job_start(job, now=1.0)
    mc.on_job_finish(job, now=6.0)
    assert mc.arrivals == 1
    assert mc.starts == 1
    assert mc.finishes == 1
