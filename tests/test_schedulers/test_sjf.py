"""Tests for SJFScheduler — including regression test for the seq-corruption bug."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, duration=5.0, gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=duration, gpus_required=gpus)


def test_sjf_picks_shortest_first():
    s = SJFScheduler()
    s.on_job_arrival(j("long", 10.0), now=0.0)
    s.on_job_arrival(j("short", 2.0), now=0.0)
    s.on_job_arrival(j("medium", 5.0), now=0.0)
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "short"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "medium"
    assert s.pick_next(now=0.0, available_gpus=4).job_id == "long"


def test_sjf_skips_oversized_picks_next_shortest():
    """When the shortest job needs more GPUs than available, skip and find next shortest."""
    s = SJFScheduler()
    s.on_job_arrival(j("tiny", 1.0, gpus=4), now=0.0)   # shortest but needs 4 GPUs
    s.on_job_arrival(j("small", 3.0, gpus=1), now=0.0)
    # Only 2 GPUs available — tiny is skipped, small should be dispatched
    result = s.pick_next(now=0.0, available_gpus=2)
    assert result is not None
    assert result.job_id == "small"


def test_sjf_skipped_job_remains_in_heap():
    """After skip-and-dispatch, the skipped job must still be dispatchable later."""
    s = SJFScheduler()
    s.on_job_arrival(j("big", 1.0, gpus=4), now=0.0)
    s.on_job_arrival(j("small", 5.0, gpus=1), now=0.0)
    s.pick_next(now=0.0, available_gpus=2)   # small dispatched, big skipped
    result = s.pick_next(now=5.0, available_gpus=4)
    assert result is not None
    assert result.job_id == "big"


def test_sjf_seq_not_mutated_on_reinsert():
    """
    REGRESSION: _seq must not increase when re-inserting skipped jobs.
    If it does, the tie-breaking order becomes non-deterministic.
    """
    s = SJFScheduler()
    s.on_job_arrival(j("j_big", 1.0, gpus=4), now=0.0)
    s.on_job_arrival(j("j_small", 5.0, gpus=1), now=0.0)
    seq_before = s._seq
    s.pick_next(now=0.0, available_gpus=1)  # j_big skipped, j_small dispatched
    assert s._seq == seq_before, (
        f"_seq should not change on skip+reinsert: was {seq_before}, "
        f"now {s._seq}. This is the heap-seq-corruption bug."
    )


def test_sjf_returns_none_when_all_oversized():
    s = SJFScheduler()
    s.on_job_arrival(j("j0", gpus=8), now=0.0)
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result is None


def test_sjf_empty_returns_none():
    s = SJFScheduler()
    assert s.pick_next(now=0.0, available_gpus=8) is None
