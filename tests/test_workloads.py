"""Tests for the synthetic workload generator."""
from __future__ import annotations

import pytest

from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic


def test_correct_number_of_jobs():
    cfg = WorkloadConfig(n_jobs=100, seed=1)
    jobs = generate_synthetic(cfg)
    assert len(jobs) == 100


def test_job_ids_unique():
    cfg = WorkloadConfig(n_jobs=200, seed=2)
    jobs = generate_synthetic(cfg)
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids))


def test_arrivals_non_decreasing():
    cfg = WorkloadConfig(n_jobs=200, seed=3)
    jobs = generate_synthetic(cfg)
    for i in range(1, len(jobs)):
        assert jobs[i].arrival_time >= jobs[i - 1].arrival_time


def test_durations_positive():
    cfg = WorkloadConfig(n_jobs=100, seed=4)
    jobs = generate_synthetic(cfg)
    assert all(j.duration > 0 for j in jobs)


def test_gpu_requirements_valid():
    cfg = WorkloadConfig(n_jobs=200, seed=5, gpu_req_choices=(1, 2, 4))
    jobs = generate_synthetic(cfg)
    valid = set(cfg.gpu_req_choices)
    assert all(j.gpus_required in valid for j in jobs)


def test_tenant_ids_within_range():
    cfg = WorkloadConfig(n_jobs=300, seed=6, n_tenants=5)
    jobs = generate_synthetic(cfg)
    expected = {f"tenant_{i}" for i in range(5)}
    actual = {j.tenant_id for j in jobs}
    assert actual.issubset(expected)


def test_reproducible_with_same_seed():
    cfg = WorkloadConfig(n_jobs=50, seed=99)
    jobs_a = generate_synthetic(cfg)
    jobs_b = generate_synthetic(cfg)
    for a, b in zip(jobs_a, jobs_b):
        assert a.arrival_time == pytest.approx(b.arrival_time)
        assert a.duration == pytest.approx(b.duration)
        assert a.gpus_required == b.gpus_required
        assert a.tenant_id == b.tenant_id


def test_different_seeds_produce_different_jobs():
    cfg_a = WorkloadConfig(n_jobs=50, seed=1)
    cfg_b = WorkloadConfig(n_jobs=50, seed=2)
    jobs_a = generate_synthetic(cfg_a)
    jobs_b = generate_synthetic(cfg_b)
    # Very unlikely to be identical
    arrivals_a = [j.arrival_time for j in jobs_a]
    arrivals_b = [j.arrival_time for j in jobs_b]
    assert arrivals_a != arrivals_b


def test_interactive_jobs_have_sla_fields():
    cfg = WorkloadConfig(n_jobs=500, seed=7, interactive_frac=0.5)
    jobs = generate_synthetic(cfg)
    interactive = [j for j in jobs if j.max_wait is not None]
    assert len(interactive) > 0
    for j in interactive:
        assert j.deadline is not None
        assert j.max_wait > 0


def test_priority_field_within_range():
    cfg = WorkloadConfig(n_jobs=200, seed=8, priority_levels=4)
    jobs = generate_synthetic(cfg)
    assert all(0 <= j.priority < 4 for j in jobs)
