from __future__ import annotations

import csv
import os
from typing import List, Optional

import numpy as np

from gdaas_sim.sim.engine import Job


def load_trace_csv(path: str, tenant_count: int = 3, seed: int = 42) -> List[Job]:
    """
    Load jobs from a CSV trace file.

    Required columns: job_id, arrival_time, duration, gpus_required
    Optional columns: tenant_id, priority, deadline, max_wait, spot, gang_id

    Missing optional columns receive sensible defaults:
    - tenant_id: round-robin assignment across tenant_0..tenant_{tenant_count-1}
    - priority: 0
    - deadline: None
    - max_wait: None
    - spot: False
    - gang_id: None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace file not found: {path}")

    rng = np.random.default_rng(seed)
    tenant_ids = [f"tenant_{i}" for i in range(tenant_count)]

    jobs: List[Job] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"job_id", "arrival_time", "duration", "gpus_required"}
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty or has no header")
        available = set(reader.fieldnames)
        missing_required = required - available
        if missing_required:
            raise ValueError(f"CSV missing required columns: {missing_required}")

        for row_idx, row in enumerate(reader):
            job_id = row["job_id"].strip()
            arrival_time = float(row["arrival_time"])
            duration = float(row["duration"])
            gpus_required = int(row["gpus_required"])

            # Optional fields with defaults
            if "tenant_id" in row and row["tenant_id"].strip():
                tenant_id = row["tenant_id"].strip()
            else:
                tenant_id = tenant_ids[row_idx % tenant_count]

            priority = int(row["priority"]) if "priority" in row and row["priority"].strip() else 0

            deadline: Optional[float] = None
            if "deadline" in row and row["deadline"].strip() and row["deadline"].strip().lower() not in ("none", "null", ""):
                deadline = float(row["deadline"])

            max_wait: Optional[float] = None
            if "max_wait" in row and row["max_wait"].strip() and row["max_wait"].strip().lower() not in ("none", "null", ""):
                max_wait = float(row["max_wait"])

            spot = False
            if "spot" in row and row["spot"].strip():
                spot_val = row["spot"].strip().lower()
                spot = spot_val in ("1", "true", "yes")

            gang_id: Optional[str] = None
            if "gang_id" in row and row["gang_id"].strip() and row["gang_id"].strip().lower() not in ("none", "null", ""):
                gang_id = row["gang_id"].strip()

            jobs.append(Job(
                job_id=job_id,
                arrival_time=arrival_time,
                duration=duration,
                gpus_required=gpus_required,
                tenant_id=tenant_id,
                priority=priority,
                deadline=deadline,
                max_wait=max_wait,
                spot=spot,
                preemptible=spot,
                gang_id=gang_id,
            ))

    return jobs


def generate_alibaba_trace(n_jobs: int = 500, seed: int = 42) -> List[Job]:
    """
    Generate a synthetic trace approximating Alibaba cluster workload statistics.

    Characteristics:
    - Heavy-tailed job durations (Pareto distribution, shape ~1.5)
    - Bursty arrivals (Poisson with time-varying rates simulating daily patterns)
    - Mixed GPU requirements: mostly 1-GPU, occasional 4/8 GPU jobs
    - Multiple tenant types: production (high priority), research (medium), dev (low)
    - 15% interactive jobs with tight SLA requirements
    - Some gang jobs (10% of batch workload)
    """
    rng = np.random.default_rng(seed)

    # Tenant types with weights and priority levels
    tenant_types = [
        ("production", 3, 0.5),   # (name_suffix, base_priority, fraction)
        ("research",   2, 0.35),
        ("dev",        1, 0.15),
    ]

    # Build tenant list
    tenants = []
    fractions = []
    for name, pri, frac in tenant_types:
        tenants.append((name, pri))
        fractions.append(frac)

    # Bursty arrivals: Poisson with sinusoidal rate variation (simulate daily cycle)
    # Base rate: 1 job per time unit (scaled)
    base_rate = 1.0
    time = 0.0
    arrival_times: List[float] = []
    while len(arrival_times) < n_jobs:
        # Sinusoidal rate variation: peak at t=50, trough at t=25 (arbitrary units)
        rate = base_rate * (0.5 + 0.5 * np.sin(time * 0.1))
        rate = max(rate, 0.05)  # avoid zero rate
        gap = rng.exponential(1.0 / rate)
        time += gap
        arrival_times.append(time)

    # Heavy-tailed durations: Pareto(a=1.5, x_m=1.0) — many short, few very long
    # Pareto: x_m / U^(1/a) where U ~ Uniform(0,1)
    u = rng.uniform(0, 1, size=n_jobs)
    durations = 1.0 / (u ** (1.0 / 1.5))
    # Clip to reasonable range [0.1, 500]
    durations = np.clip(durations, 0.1, 500.0)

    # GPU requirements: heavily skewed toward 1 GPU
    gpu_weights = [0.60, 0.20, 0.10, 0.07, 0.03]  # 1, 2, 4, 8, 16 GPUs
    gpu_choices = [1, 2, 4, 8, 16]
    gpu_reqs = rng.choice(gpu_choices, size=n_jobs, p=gpu_weights)

    # Tenant assignment
    tenant_fracs = np.array(fractions)
    tenant_choices = rng.choice(len(tenants), size=n_jobs, p=tenant_fracs)

    # Interactive fraction: 15%
    is_interactive = rng.random(n_jobs) < 0.15

    # Gang assignment: 10% of batch jobs form gangs
    gang_ids: List[Optional[str]] = [None] * n_jobs
    gang_counter = 0
    batch_indices = [i for i in range(n_jobs) if not is_interactive[i]]
    n_gang_jobs = int(len(batch_indices) * 0.10)
    gang_candidates = list(rng.choice(batch_indices, size=n_gang_jobs, replace=False))
    gang_candidates.sort()

    i = 0
    while i < len(gang_candidates):
        size = int(rng.integers(2, 5))
        group = gang_candidates[i:i + size]
        if len(group) >= 2:
            gname = f"gang_{gang_counter:04d}"
            for idx in group:
                gang_ids[idx] = gname
            gang_counter += 1
        i += size

    jobs: List[Job] = []
    for idx in range(n_jobs):
        arr = float(arrival_times[idx])
        dur = float(durations[idx])
        tenant_name, base_priority = tenants[tenant_choices[idx]]
        tenant_id = f"tenant_{tenant_name}"

        if bool(is_interactive[idx]):
            max_wait = 3.0    # tight SLA for interactive jobs
            deadline = arr + 15.0
            priority = base_priority + 1
        else:
            max_wait = None
            deadline = None
            priority = base_priority

        jobs.append(Job(
            job_id=f"ali_{idx:06d}",
            arrival_time=arr,
            duration=dur,
            gpus_required=int(gpu_reqs[idx]),
            tenant_id=tenant_id,
            priority=priority,
            deadline=deadline,
            max_wait=max_wait,
            gang_id=gang_ids[idx],
        ))

    return jobs
