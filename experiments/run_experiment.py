from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from gdaas_sim.sim.engine import SimEngine
from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic
from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.scheduler.fair_share import TenantFairScheduler
from gdaas_sim.metrics.collector import MetricsCollector


def percentile(xs, p: float):
    if not xs:
        return None
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def run_one(scheduler_name: str, total_gpus: int, seed: int, arrival_rate: float):
    cfg = WorkloadConfig(n_jobs=800, arrival_rate=arrival_rate, seed=seed)
    jobs = generate_synthetic(cfg)

    cluster = GPUCluster(total_gpus=total_gpus)
    metrics = MetricsCollector()
    engine = SimEngine()

    if scheduler_name == "fifo":
        scheduler = FIFOScheduler()
    elif scheduler_name == "sjf":
        scheduler = SJFScheduler()
    elif scheduler_name == "fair":
        scheduler = TenantFairScheduler()
    else:
        raise ValueError("scheduler must be fifo, sjf, or fair")

    engine.run(jobs=jobs, cluster=cluster, scheduler=scheduler, metrics=metrics)

    sim_end = engine.time
    util = metrics.busy_gpu_time / (total_gpus * sim_end) if sim_end > 0 else 0.0
    jain = metrics.jain_fairness()

    return {
        "scheduler": scheduler_name,
        "seed": seed,
        "arrival_rate": arrival_rate,
        "total_gpus": total_gpus,
        "sim_end_time": sim_end,
        "jobs_arrived": metrics.arrivals,
        "jobs_finished": metrics.finishes,
        "utilization": util,
        "avg_wait": (sum(metrics.wait_times) / len(metrics.wait_times)) if metrics.wait_times else None,
        "p95_wait": percentile(metrics.wait_times, 95),
        "avg_turnaround": (sum(metrics.turnaround_times) / len(metrics.turnaround_times))
        if metrics.turnaround_times
        else None,
        "jain_gpu_time": jain,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments/out")
    ap.add_argument("--total_gpus", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--arrival_rates", type=float, nargs="+", default=[0.2, 0.5, 0.9])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for sch in ["fifo", "sjf", "fair"]:
        for rate in args.arrival_rates:
            for seed in args.seeds:
                rows.append(run_one(sch, args.total_gpus, seed, rate))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Plot: avg wait vs load, p95 wait vs load, utilization vs load, fairness vs load
    for metric in ["avg_wait", "p95_wait", "utilization", "jain_gpu_time"]:
        plt.figure()
        for sch in ["fifo", "sjf", "fair"]:
            sub = df[df["scheduler"] == sch].groupby("arrival_rate")[metric].mean().reset_index()
            plt.plot(sub["arrival_rate"], sub[metric], marker="o", label=sch)
        plt.xlabel("arrival_rate (jobs/time unit)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs load")
        plt.legend()
        fig_path = os.path.join(args.outdir, f"{metric}.png")
        plt.savefig(fig_path, dpi=160, bbox_inches="tight")
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
