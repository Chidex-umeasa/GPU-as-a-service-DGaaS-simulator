"""Tests for EASYBackfillScheduler — the SLURM-style backfill algorithm."""
from __future__ import annotations

import pytest

from gdaas_sim.scheduler.backfill import EASYBackfillScheduler
from gdaas_sim.sim.engine import Job


def j(job_id, duration=5.0, gpus=1, arrival=0.0):
    return Job(job_id, arrival_time=arrival, duration=duration, gpus_required=gpus)


def _start(s: EASYBackfillScheduler, job: Job, now: float = 0.0):
    """Helper: simulate a job starting (sets start_time and notifies scheduler)."""
    job.start_time = now
    s.on_job_start(job, now=now)


def _finish(s: EASYBackfillScheduler, job: Job, finish_time: float):
    """Helper: simulate a job finishing."""
    job.finish_time = finish_time
    s.on_job_finish(job, now=finish_time)


# -----------------------------------------------------------------------
# Basic dispatch
# -----------------------------------------------------------------------

def test_backfill_dispatches_pivot_when_fits():
    s = EASYBackfillScheduler()
    job = j("j0", gpus=2)
    s.on_job_arrival(job, now=0.0)
    result = s.pick_next(now=0.0, available_gpus=4)
    assert result is not None
    assert result.job_id == "j0"


def test_backfill_returns_none_when_empty():
    s = EASYBackfillScheduler()
    assert s.pick_next(now=0.0, available_gpus=4) is None


def test_backfill_pivot_blocked_no_candidates():
    """Pivot needs 4 GPUs; only 1 available; no other jobs → None."""
    s = EASYBackfillScheduler()
    s.on_job_arrival(j("pivot", gpus=4), now=0.0)
    result = s.pick_next(now=0.0, available_gpus=1)
    assert result is None


# -----------------------------------------------------------------------
# Core backfill logic
# -----------------------------------------------------------------------

def test_backfill_eligible_candidate_dispatched():
    """
    Setup:
      - running_job: uses 3 GPUs, finishes at t=10
      - available: 1 GPU  (total=4, 3 in use)
      - pivot: needs 4 GPUs → shadow_time = t=10
      - candidate: needs 1 GPU, duration=3 → finishes at t=3 ≤ t=10 ✓
    Expected: candidate is backfilled.
    """
    s = EASYBackfillScheduler()
    running = j("running", duration=10.0, gpus=3)
    _start(s, running, now=0.0)

    pivot = j("pivot", duration=5.0, gpus=4)
    candidate = j("candidate", duration=3.0, gpus=1)
    s.on_job_arrival(pivot, now=0.0)
    s.on_job_arrival(candidate, now=0.0)

    result = s.pick_next(now=0.0, available_gpus=1)
    assert result is not None
    assert result.job_id == "candidate"


def test_backfill_candidate_blocked_when_exceeds_shadow():
    """
    Setup:
      - running_job: uses 3 GPUs, finishes at t=5
      - available: 1 GPU
      - pivot: needs 4 GPUs → shadow_time = t=5
      - candidate: needs 1 GPU, duration=10 → finishes at t=10 > t=5 ✗
    Expected: None (candidate would violate pivot's reservation).
    """
    s = EASYBackfillScheduler()
    running = j("running", duration=5.0, gpus=3)
    _start(s, running, now=0.0)

    pivot = j("pivot", duration=2.0, gpus=4)
    too_long = j("too_long", duration=10.0, gpus=1)
    s.on_job_arrival(pivot, now=0.0)
    s.on_job_arrival(too_long, now=0.0)

    result = s.pick_next(now=0.0, available_gpus=1)
    assert result is None


def test_backfill_pivot_dispatched_after_running_job_finishes():
    """After the running job releases GPUs, the pivot should run."""
    s = EASYBackfillScheduler()
    running = j("running", duration=5.0, gpus=3)
    _start(s, running, now=0.0)

    pivot = j("pivot", duration=4.0, gpus=4)
    s.on_job_arrival(pivot, now=0.0)

    # At t=0: pivot blocked (only 1 GPU free, needs 4)
    assert s.pick_next(now=0.0, available_gpus=1) is None

    # Simulate running job finishing at t=5, freeing 3 GPUs → 4 total free
    _finish(s, running, finish_time=5.0)

    result = s.pick_next(now=5.0, available_gpus=4)
    assert result is not None
    assert result.job_id == "pivot"


def test_backfill_pivot_stays_at_head_after_candidate_backfilled():
    """Backfilling a candidate must not remove the pivot from the queue."""
    s = EASYBackfillScheduler()
    running = j("running", duration=10.0, gpus=3)
    _start(s, running, now=0.0)

    pivot = j("pivot", duration=5.0, gpus=4)
    candidate = j("candidate", duration=2.0, gpus=1)
    s.on_job_arrival(pivot, now=0.0)
    s.on_job_arrival(candidate, now=0.0)

    backfilled = s.pick_next(now=0.0, available_gpus=1)
    assert backfilled.job_id == "candidate"

    # pivot must still be the head of the queue
    assert s._queue[0].job_id == "pivot"


# -----------------------------------------------------------------------
# Shadow time computation
# -----------------------------------------------------------------------

def test_shadow_time_with_multiple_running_jobs():
    """
    3 running jobs each holding 2 GPUs; pivot needs 6; 0 GPUs currently free.
    Jobs finish at t=3, t=7, t=10 (each releasing 2 GPUs).

    Accumulation:
      After t=3:  free = 0 + 2 = 2  → not enough (need 6)
      After t=7:  free = 2 + 2 = 4  → not enough (need 6)
      After t=10: free = 4 + 2 = 6  → shadow_time = 10
    """
    s = EASYBackfillScheduler()
    r1 = j("r1", duration=3.0, gpus=2)
    r2 = j("r2", duration=7.0, gpus=2)
    r3 = j("r3", duration=10.0, gpus=2)
    _start(s, r1, now=0.0)
    _start(s, r2, now=0.0)
    _start(s, r3, now=0.0)

    shadow = s._compute_shadow_time(available_gpus=0, pivot_gpus=6)
    assert shadow == pytest.approx(10.0)


def test_shadow_time_met_at_first_release():
    """
    Pivot needs 4 GPUs; 2 currently free.
    r1 finishes at t=5 releasing 2 GPUs → free = 4 ≥ 4 → shadow = 5.
    r2 at t=10 should not be reached.
    """
    s = EASYBackfillScheduler()
    r1 = j("r1", duration=5.0, gpus=2)
    r2 = j("r2", duration=10.0, gpus=2)
    _start(s, r1, now=0.0)
    _start(s, r2, now=0.0)

    shadow = s._compute_shadow_time(available_gpus=2, pivot_gpus=4)
    assert shadow == pytest.approx(5.0)
