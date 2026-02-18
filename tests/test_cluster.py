"""Tests for GPUCluster resource management."""
from __future__ import annotations

import pytest

from gdaas_sim.cluster.cluster import GPUCluster


def test_initial_available_equals_total():
    c = GPUCluster(total_gpus=8)
    assert c.available_gpus == 8


def test_allocate_reduces_available():
    c = GPUCluster(total_gpus=8)
    c.allocate(3)
    assert c.available_gpus == 5


def test_release_increases_available():
    c = GPUCluster(total_gpus=8)
    c.allocate(3)
    c.release(3)
    assert c.available_gpus == 8


def test_over_allocation_raises():
    c = GPUCluster(total_gpus=4)
    with pytest.raises(RuntimeError):
        c.allocate(5)


def test_over_release_raises():
    c = GPUCluster(total_gpus=4)
    with pytest.raises(RuntimeError):
        c.release(1)  # nothing allocated


def test_allocate_zero_raises():
    c = GPUCluster(total_gpus=4)
    with pytest.raises(ValueError):
        c.allocate(0)


def test_release_zero_raises():
    c = GPUCluster(total_gpus=4)
    with pytest.raises(ValueError):
        c.release(0)


def test_zero_total_gpus_raises():
    with pytest.raises(ValueError):
        GPUCluster(total_gpus=0)


def test_multiple_allocate_release_cycles():
    c = GPUCluster(total_gpus=16)
    c.allocate(4)
    c.allocate(8)
    assert c.available_gpus == 4
    c.release(8)
    assert c.available_gpus == 12
    c.release(4)
    assert c.available_gpus == 16
