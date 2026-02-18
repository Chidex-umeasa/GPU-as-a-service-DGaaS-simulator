"""Tests for FIFOScheduler."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=5.0, gpus_required=gpus)


def test_fifo_returns_none_when_empty():
    s = FIFOScheduler()
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_fifo_arrival_order():
    s = FIFOScheduler()
    s.on_job_arrival(j("j0"), now=0.0)
    s.on_job_arrival(j("j1"), now=1.0)
    s.on_job_arrival(j("j2"), now=2.0)
    assert s.pick_next(now=2.0, available_gpus=4).job_id == "j0"
    assert s.pick_next(now=2.0, available_gpus=4).job_id == "j1"
    assert s.pick_next(now=2.0, available_gpus=4).job_id == "j2"


def test_fifo_blocks_on_head_gpu_constraint():
    """FIFO must return None if the head job needs more GPUs than available."""
    s = FIFOScheduler()
    s.on_job_arrival(j("j_big", gpus=8), now=0.0)
    s.on_job_arrival(j("j_small", gpus=1), now=1.0)
    # Only 4 GPUs available — j_big can't run, so FIFO returns None
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result is None


def test_fifo_dispatches_when_gpus_sufficient():
    s = FIFOScheduler()
    s.on_job_arrival(j("j0", gpus=4), now=0.0)
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result.job_id == "j0"
