from __future__ import annotations

from dataclasses import dataclass
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
            )
        )

    return jobs
