from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from gdaas_sim.sim.engine import Job


@dataclass
class WorkloadConfig:
    n_jobs: int = 500
    arrival_rate: float = 0.5  # jobs per time unit (Poisson process)
    duration_mean: float = 10.0
    duration_sigma: float = 0.8  # lognormal sigma
    gpu_req_choices: tuple = (1, 1, 1, 2, 2, 4)  # skew toward 1-2 GPUs
    seed: int = 7


def generate_synthetic(cfg: WorkloadConfig) -> List[Job]:
    """
    Generate a synthetic workload:
      - Inter-arrivals are exponential with rate cfg.arrival_rate
      - Durations follow a lognormal distribution (heavy-tailed-ish)
      - GPU requirements sampled from cfg.gpu_req_choices
    """
    rng = np.random.default_rng(cfg.seed)

    inter_arrivals = rng.exponential(1.0 / cfg.arrival_rate, size=cfg.n_jobs)
    arrivals = np.cumsum(inter_arrivals)

    # Lognormal: mean of log-space is adjusted so median ~ duration_mean-ish.
    mu = np.log(max(cfg.duration_mean, 1e-6))
    durations = rng.lognormal(mean=mu, sigma=cfg.duration_sigma, size=cfg.n_jobs)

    gpu_reqs = rng.choice(cfg.gpu_req_choices, size=cfg.n_jobs)

    jobs: List[Job] = []
    for i in range(cfg.n_jobs):
        jobs.append(
            Job(
                job_id=f"job_{i:05d}",
                arrival_time=float(arrivals[i]),
                duration=float(durations[i]),
                gpus_required=int(gpu_reqs[i]),
            )
        )
    return jobs
