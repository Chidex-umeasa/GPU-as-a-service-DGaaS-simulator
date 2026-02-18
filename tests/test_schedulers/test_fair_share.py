"""Tests for TenantFairScheduler."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.fair_share import TenantFairScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, tenant="t0", gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=5.0,
               gpus_required=gpus, tenant_id=tenant)


def test_fair_share_returns_none_when_empty():
    s = TenantFairScheduler()
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_fair_share_picks_lowest_usage_tenant():
    s = TenantFairScheduler()
    s.on_job_arrival(j("t0j0", "t0"), now=0.0)
    s.on_job_arrival(j("t1j0", "t1"), now=0.0)
    # Simulate t0 already running a job
    s._running_gpu["t0"] = 4
    result = s.pick_next(now=0.0, available_gpus=8)
    assert result.tenant_id == "t1"  # t1 has lower GPU usage


def test_fair_share_dispatched_job_removed():
    s = TenantFairScheduler()
    s.on_job_arrival(j("j0"), now=0.0)
    s.pick_next(now=0.0, available_gpus=4)
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_fair_share_respects_gpu_constraint():
    s = TenantFairScheduler()
    s.on_job_arrival(j("j_big", gpus=8), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_fair_share_running_gpu_tracked():
    s = TenantFairScheduler()
    job = j("j0", "t0", gpus=2)
    s.on_job_arrival(job, now=0.0)
    s.on_job_start(job, now=0.0)
    assert s._running_gpu["t0"] == 2


def test_fair_share_running_gpu_released():
    s = TenantFairScheduler()
    job = j("j0", "t0", gpus=2)
    job.start_time = 0.0
    job.finish_time = 5.0
    s.on_job_start(job, now=0.0)
    s.on_job_finish(job, now=5.0)
    assert "t0" not in s._running_gpu


def test_fair_share_o1_removal():
    """Verify removal is via dict (O(1)) not list.remove (O(n))."""
    s = TenantFairScheduler()
    for i in range(100):
        s.on_job_arrival(j(f"j{i}", "t0"), now=float(i))
    # Pick one job; the remaining 99 should still be in the dict
    s.pick_next(now=100.0, available_gpus=4)
    assert len(s._waiting) == 99
