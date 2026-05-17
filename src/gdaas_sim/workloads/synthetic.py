from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from gdaas_sim.sim.engine import Job


@dataclass
class WorkloadConfig:
    n_jobs: int = 800
    arrival_rate: float = 0.5
    duration_mean: float = 10.0
    duration_sigma: float = 0.8
    gpu_req_choices: tuple = (1, 1, 1, 2, 2, 4)

    n_tenants: int = 3
    interactive_frac: float = 0.25  # 25% interactive, 75% batch

    # SLA knobs (in "time units" of the simulator)
    interactive_max_wait: float = 5.0
    interactive_deadline_slack: float = 20.0  # deadline = arrival + slack
    batch_deadline_slack: Optional[float] = None  # None -> no deadline

    # Priority levels drawn uniformly from [0, priority_levels)
    priority_levels: int = 4

    seed: int = 7

    # v0.3.0 additions (all with defaults for backwards compatibility)
    spot_frac: float = 0.0           # fraction of jobs that are spot (preemptible)
    gang_frac: float = 0.0           # fraction of jobs that are part of gangs
    gang_size_range: tuple = (2, 4)  # min/max gang size (inclusive)
    spot_cost_factor: float = 0.4    # spot jobs cost this fraction of regular cost
    base_cost_per_gpu_hour: float = 1.0  # base cost per GPU per time unit
    gpu_type_names: Optional[List[str]] = None   # if set, assign GPU type preferences
    gpu_type_probs: Optional[List[float]] = None # probabilities for each GPU type


def generate_synthetic(cfg: WorkloadConfig) -> List[Job]:
    rng = np.random.default_rng(cfg.seed)

    inter_arrivals = rng.exponential(1.0 / cfg.arrival_rate, size=cfg.n_jobs)
    arrivals = np.cumsum(inter_arrivals)

    mu = np.log(max(cfg.duration_mean, 1e-6))
    durations = rng.lognormal(mean=mu, sigma=cfg.duration_sigma, size=cfg.n_jobs)

    gpu_reqs = rng.choice(cfg.gpu_req_choices, size=cfg.n_jobs)

    tenant_ids = [f"tenant_{i}" for i in range(cfg.n_tenants)]
    tenants = rng.choice(tenant_ids, size=cfg.n_jobs)

    is_interactive = rng.random(cfg.n_jobs) < cfg.interactive_frac
    priorities = rng.integers(0, max(cfg.priority_levels, 1), size=cfg.n_jobs)

    # Spot job assignment
    is_spot = rng.random(cfg.n_jobs) < cfg.spot_frac

    # Gang assignment: assign gang_frac of jobs to gangs
    gang_ids: List[Optional[str]] = [None] * cfg.n_jobs
    if cfg.gang_frac > 0.0:
        gang_size_min, gang_size_max = cfg.gang_size_range
        gang_size_min = max(2, gang_size_min)
        gang_size_max = max(gang_size_min, gang_size_max)

        # Decide which jobs are gang members (using a contiguous grouping approach)
        n_gang_jobs = int(cfg.n_jobs * cfg.gang_frac)
        # Shuffle indices to pick which jobs become gang members
        gang_candidate_indices = list(range(cfg.n_jobs))
        rng.shuffle(gang_candidate_indices)
        gang_candidate_indices = sorted(gang_candidate_indices[:n_gang_jobs])

        gang_counter = 0
        i = 0
        while i < len(gang_candidate_indices):
            size = int(rng.integers(gang_size_min, gang_size_max + 1))
            group = gang_candidate_indices[i:i + size]
            if len(group) < 2:
                break  # don't create single-member gangs
            gang_name = f"gang_{gang_counter:04d}"
            for idx in group:
                gang_ids[idx] = gang_name
            gang_counter += 1
            i += size

    # GPU type preference assignment
    gpu_type_preferences: List[Optional[str]] = [None] * cfg.n_jobs
    if cfg.gpu_type_names:
        type_names = cfg.gpu_type_names
        probs = cfg.gpu_type_probs
        if probs is not None:
            total = sum(probs)
            probs = [p / total for p in probs]
        chosen_types = rng.choice(type_names, size=cfg.n_jobs, p=probs)
        gpu_type_preferences = list(chosen_types)

    jobs: List[Job] = []
    for i in range(cfg.n_jobs):
        arr = float(arrivals[i])
        dur = float(durations[i])

        if bool(is_interactive[i]):
            max_wait = cfg.interactive_max_wait
            deadline = arr + cfg.interactive_deadline_slack
        else:
            max_wait = None
            deadline = (arr + cfg.batch_deadline_slack) if cfg.batch_deadline_slack else None

        spot = bool(is_spot[i])
        preemptible = spot  # spot jobs are always preemptible
        cost_per_gpu_hour = (
            cfg.base_cost_per_gpu_hour * cfg.spot_cost_factor
            if spot
            else cfg.base_cost_per_gpu_hour
        )

        jobs.append(
            Job(
                job_id=f"job_{i:05d}",
                arrival_time=arr,
                duration=dur,
                gpus_required=int(gpu_reqs[i]),
                tenant_id=str(tenants[i]),
                max_wait=max_wait,
                deadline=deadline,
                priority=int(priorities[i]),
                spot=spot,
                preemptible=preemptible,
                gang_id=gang_ids[i],
                cost_per_gpu_hour=cost_per_gpu_hour,
                gpu_type_preference=gpu_type_preferences[i],
            )
        )

    return jobs
