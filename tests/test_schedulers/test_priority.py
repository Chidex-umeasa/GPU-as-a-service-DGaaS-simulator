"""Tests for PriorityScheduler."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.priority import PriorityScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, priority=0, gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=5.0,
               gpus_required=gpus, priority=priority)


def test_priority_highest_first():
    s = PriorityScheduler()
    s.on_job_arrival(j("low", priority=0), now=0.0)
    s.on_job_arrival(j("high", priority=3), now=0.0)
    s.on_job_arrival(j("mid", priority=1), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "high"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "mid"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "low"


def test_priority_fifo_tiebreak():
    """Within the same priority level, earlier-arriving jobs go first."""
    s = PriorityScheduler()
    s.on_job_arrival(j("first", priority=2, arrival=0.0), now=0.0)
    s.on_job_arrival(j("second", priority=2, arrival=1.0), now=1.0)
    assert s.pick_next(now=1.0, available_gpus=4).job_id == "first"
    assert s.pick_next(now=1.0, available_gpus=4).job_id == "second"


def test_priority_gpu_constraint_respected():
    s = PriorityScheduler()
    s.on_job_arrival(j("big", priority=5, gpus=8), now=0.0)
    s.on_job_arrival(j("small", priority=1, gpus=1), now=0.0)
    # big has highest priority but needs 8 GPUs; only 4 available → skip
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result.job_id == "small"


def test_priority_skipped_job_survives():
    s = PriorityScheduler()
    s.on_job_arrival(j("big", priority=5, gpus=8), now=0.0)
    s.pick_next(now=0.0, available_gpus=4)   # nothing dispatched
    # Now with 8 GPUs, big should be dispatched
    result = s.pick_next(now=1.0, available_gpus=8)
    assert result is not None
    assert result.job_id == "big"


def test_priority_empty_returns_none():
    s = PriorityScheduler()
    assert s.pick_next(now=0.0, available_gpus=4) is None
