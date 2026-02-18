"""Tests for RoundRobinScheduler."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.round_robin import RoundRobinScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, tenant="t0", gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=5.0,
               gpus_required=gpus, tenant_id=tenant)


def test_rr_alternates_between_tenants():
    s = RoundRobinScheduler()
    s.on_job_arrival(j("t0j0", "t0"), now=0.0)
    s.on_job_arrival(j("t0j1", "t0"), now=0.0)
    s.on_job_arrival(j("t1j0", "t1"), now=0.0)
    dispatched = [s.pick_next(now=0.0, available_gpus=4) for _ in range(3)]
    tenant_sequence = [d.tenant_id for d in dispatched]
    # Should alternate t0 → t1 → t0
    assert tenant_sequence == ["t0", "t1", "t0"]


def test_rr_single_tenant():
    s = RoundRobinScheduler()
    s.on_job_arrival(j("j0", "t0"), now=0.0)
    s.on_job_arrival(j("j1", "t0"), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "j0"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "j1"


def test_rr_skips_tenant_with_no_runnable_job():
    s = RoundRobinScheduler()
    s.on_job_arrival(j("t0j0", "t0", gpus=8), now=0.0)  # needs 8, won't fit
    s.on_job_arrival(j("t1j0", "t1", gpus=1), now=0.0)
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result is not None
    assert result.tenant_id == "t1"


def test_rr_returns_none_when_empty():
    s = RoundRobinScheduler()
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_rr_returns_none_when_all_oversized():
    s = RoundRobinScheduler()
    s.on_job_arrival(j("j0", gpus=8), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_rr_new_tenant_appended_to_rotation():
    s = RoundRobinScheduler()
    s.on_job_arrival(j("t0j0", "t0"), now=0.0)
    s.pick_next(now=0.0, available_gpus=4)  # dispatches t0j0
    # Now t2 arrives (new tenant)
    s.on_job_arrival(j("t2j0", "t2"), now=1.0)
    result = s.pick_next(now=1.0, available_gpus=4)
    assert result is not None
    assert result.tenant_id == "t2"
