"""Tests for EDFScheduler."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.edf import EDFScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, deadline=None, gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=5.0,
               gpus_required=gpus, deadline=deadline)


def test_edf_earliest_deadline_first():
    s = EDFScheduler()
    s.on_job_arrival(j("late", deadline=100.0), now=0.0)
    s.on_job_arrival(j("early", deadline=20.0), now=0.0)
    s.on_job_arrival(j("middle", deadline=50.0), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "early"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "middle"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "late"


def test_edf_no_deadline_treated_as_lowest_priority():
    s = EDFScheduler()
    s.on_job_arrival(j("no_dl", deadline=None), now=0.0)
    s.on_job_arrival(j("has_dl", deadline=50.0), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "has_dl"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "no_dl"


def test_edf_gpu_constraint_respected():
    s = EDFScheduler()
    s.on_job_arrival(j("big", deadline=10.0, gpus=8), now=0.0)
    s.on_job_arrival(j("small", deadline=50.0, gpus=1), now=0.0)
    # big has earliest deadline but needs 8 GPUs; only 4 available
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result.job_id == "small"


def test_edf_empty_returns_none():
    s = EDFScheduler()
    assert s.pick_next(now=0.0, available_gpus=4) is None
