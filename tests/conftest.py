"""Shared pytest fixtures for GDaaS Simulator tests."""
from __future__ import annotations

import pytest

from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.sim.engine import Job


def make_job(
    job_id: str,
    duration: float = 5.0,
    gpus: int = 1,
    arrival: float = 0.0,
    tenant: str = "t0",
    priority: int = 0,
    deadline: float | None = None,
    max_wait: float | None = None,
) -> Job:
    """Factory helper used across all test modules."""
    return Job(
        job_id=job_id,
        arrival_time=arrival,
        duration=duration,
        gpus_required=gpus,
        tenant_id=tenant,
        priority=priority,
        deadline=deadline,
        max_wait=max_wait,
    )


@pytest.fixture
def four_gpu_cluster() -> GPUCluster:
    return GPUCluster(total_gpus=4)


@pytest.fixture
def eight_gpu_cluster() -> GPUCluster:
    return GPUCluster(total_gpus=8)


@pytest.fixture
def simple_jobs() -> list[Job]:
    """Three jobs with known properties for deterministic tests."""
    return [
        make_job("j0", duration=5.0, gpus=1, arrival=0.0, tenant="t0", priority=2),
        make_job("j1", duration=2.0, gpus=2, arrival=1.0, tenant="t1", priority=1),
        make_job("j2", duration=10.0, gpus=1, arrival=2.0, tenant="t0", priority=0),
    ]
