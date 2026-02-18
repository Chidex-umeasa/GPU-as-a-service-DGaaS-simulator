"""
GDaaS Simulator — Experiment Runner
====================================
Runs a sweep of scheduling algorithms across multiple arrival rates and seeds,
outputs a CSV of results, and generates publication-quality plots.

Usage
-----
    python experiments/run_experiment.py
    python experiments/run_experiment.py --total_gpus 32 --seeds 1 2 3 4 5
    python experiments/run_experiment.py --config experiments/config_default.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rich imports for professional CLI output
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from gdaas_sim.sim.engine import SimEngine
from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic
from gdaas_sim.metrics.collector import MetricsCollector
from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.scheduler.fair_share import TenantFairScheduler
from gdaas_sim.scheduler.edf import EDFScheduler
from gdaas_sim.scheduler.priority import PriorityScheduler
from gdaas_sim.scheduler.round_robin import RoundRobinScheduler
from gdaas_sim.scheduler.backfill import EASYBackfillScheduler

console = Console() if RICH_AVAILABLE else None

SCHEDULER_MAP = {
    "fifo":     FIFOScheduler,
    "sjf":      SJFScheduler,
    "fair":     TenantFairScheduler,
    "edf":      EDFScheduler,
    "priority": PriorityScheduler,
    "rr":       RoundRobinScheduler,
    "backfill": EASYBackfillScheduler,
}

SCHEDULER_LABELS = {
    "fifo":     "FIFO",
    "sjf":      "SJF",
    "fair":     "Fair Share",
    "edf":      "EDF",
    "priority": "Priority",
    "rr":       "Round Robin",
    "backfill": "EASY Backfill",
}

# Distinct, print-safe colour palette
COLORS = {
    "fifo":     "#2196F3",
    "sjf":      "#F44336",
    "fair":     "#4CAF50",
    "edf":      "#FF9800",
    "priority": "#9C27B0",
    "rr":       "#00BCD4",
    "backfill": "#E91E63",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def _fmt(val: Any, decimals: int = 3) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(int(val)) if isinstance(val, (int, float)) else str(val)


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------

def run_one(
    scheduler_name: str,
    total_gpus: int,
    seed: int,
    arrival_rate: float,
    n_jobs: int = 800,
) -> dict:
    cfg = WorkloadConfig(n_jobs=n_jobs, arrival_rate=arrival_rate, seed=seed)
    jobs = generate_synthetic(cfg)

    cls = SCHEDULER_MAP.get(scheduler_name)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Valid options: {list(SCHEDULER_MAP.keys())}"
        )

    cluster = GPUCluster(total_gpus=total_gpus)
    metrics = MetricsCollector()
    engine = SimEngine()
    engine.run(jobs=jobs, cluster=cluster, scheduler=cls(), metrics=metrics)

    sim_end = engine.time
    util = metrics.busy_gpu_time / (total_gpus * sim_end) if sim_end > 0 else 0.0

    return {
        "scheduler":               scheduler_name,
        "scheduler_label":         SCHEDULER_LABELS[scheduler_name],
        "seed":                    seed,
        "arrival_rate":            arrival_rate,
        "total_gpus":              total_gpus,
        "n_jobs":                  n_jobs,
        "sim_end_time":            sim_end,
        "jobs_arrived":            metrics.arrivals,
        "jobs_finished":           metrics.finishes,
        "utilization":             util,
        "avg_wait":                (sum(metrics.wait_times) / len(metrics.wait_times))
                                   if metrics.wait_times else None,
        "p95_wait":                percentile(metrics.wait_times, 95),
        "p99_wait":                percentile(metrics.wait_times, 99),
        "avg_turnaround":          (sum(metrics.turnaround_times) / len(metrics.turnaround_times))
                                   if metrics.turnaround_times else None,
        "sla_wait_violations":     metrics.sla_wait_violations,
        "sla_deadline_violations": metrics.sla_deadline_violations,
        "jain_gpu_time":           metrics.jain_fairness(),
    }


# ---------------------------------------------------------------------------
# Rich summary table
# ---------------------------------------------------------------------------

def print_banner() -> None:
    if not RICH_AVAILABLE or console is None:
        print("=" * 60)
        print("  GDaaS Simulator — Experiment Runner  v0.2.0")
        print("  GPU-as-a-Service Discrete-Event Simulator")
        print("=" * 60)
        return
    console.print(
        Panel.fit(
            "[bold cyan]GDaaS Simulator[/bold cyan]  [dim]v0.2.0[/dim]\n"
            "[dim]GPU-as-a-Service Discrete-Event Simulator[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def print_summary_table(df: pd.DataFrame) -> None:
    if not RICH_AVAILABLE or console is None:
        print(df.groupby("scheduler")[
            ["utilization", "avg_wait", "p95_wait", "jain_gpu_time"]
        ].mean().to_string())
        return

    table = Table(
        title="[bold]Results Summary[/bold] (averaged over seeds & arrival rates)",
        box=box.ROUNDED,
        highlight=True,
        show_lines=True,
    )
    table.add_column("Scheduler",      style="bold cyan", no_wrap=True)
    table.add_column("Utilization",    style="green",  justify="right")
    table.add_column("Avg Wait",       style="yellow", justify="right")
    table.add_column("P95 Wait",       style="yellow", justify="right")
    table.add_column("SLA Wait Viol.", style="red",    justify="right")
    table.add_column("SLA DL Viol.",   style="red",    justify="right")
    table.add_column("Jain Fairness",  style="blue",   justify="right")

    summary = df.groupby("scheduler").mean(numeric_only=True).reset_index()
    order = list(SCHEDULER_MAP.keys())
    summary["_ord"] = summary["scheduler"].map({k: i for i, k in enumerate(order)})
    summary = summary.sort_values("_ord").drop(columns=["_ord"])

    for _, row in summary.iterrows():
        table.add_row(
            SCHEDULER_LABELS.get(row["scheduler"], row["scheduler"]),
            _fmt(row.get("utilization"), 4),
            _fmt(row.get("avg_wait"), 2),
            _fmt(row.get("p95_wait"), 2),
            _fmt(row.get("sla_wait_violations")),
            _fmt(row.get("sla_deadline_violations")),
            _fmt(row.get("jain_gpu_time"), 4),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_METRICS = [
    ("avg_wait",               "Average Wait Time (time units)"),
    ("p95_wait",               "P95 Wait Time (time units)"),
    ("utilization",            "GPU Utilization (fraction)"),
    ("jain_gpu_time",          "Jain's Fairness Index"),
    ("sla_wait_violations",    "SLA Wait Violations (count)"),
    ("sla_deadline_violations","SLA Deadline Violations (count)"),
]


def make_plots(df: pd.DataFrame, schedulers: list[str], outdir: str) -> None:
    for metric, ylabel in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sch in schedulers:
            sub = (
                df[df["scheduler"] == sch]
                .groupby("arrival_rate")[metric]
                .mean()
                .reset_index()
            )
            if sub.empty:
                continue
            ax.plot(
                sub["arrival_rate"],
                sub[metric],
                marker="o",
                linewidth=2,
                markersize=7,
                label=SCHEDULER_LABELS.get(sch, sch),
                color=COLORS.get(sch),
            )
        ax.set_xlabel("Arrival Rate (jobs / time unit)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} vs Load", fontsize=14, fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(outdir, f"{metric}.png")
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if RICH_AVAILABLE and console:
            console.print(f"  [green]Wrote[/green] {fig_path}")
        else:
            print(f"Wrote {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GDaaS Simulator — scheduling algorithm comparison"
    )
    ap.add_argument("--outdir",        default="experiments/out")
    ap.add_argument("--total_gpus",    type=int,   default=16)
    ap.add_argument("--n_jobs",        type=int,   default=800)
    ap.add_argument("--seeds",         type=int,   nargs="+", default=[1, 2, 3])
    ap.add_argument("--arrival_rates", type=float, nargs="+", default=[0.2, 0.5, 0.9])
    ap.add_argument(
        "--schedulers", nargs="+",
        default=list(SCHEDULER_MAP.keys()),
        choices=list(SCHEDULER_MAP.keys()),
    )
    ap.add_argument("--config", default=None,
                    help="Path to YAML experiment config (overrides CLI flags)")
    args = ap.parse_args()

    if args.config:
        try:
            from gdaas_sim.config import ExperimentConfig
            cfg = ExperimentConfig.from_yaml(args.config)
            args.total_gpus    = cfg.total_gpus
            args.n_jobs        = cfg.n_jobs
            args.seeds         = cfg.seeds
            args.arrival_rates = cfg.arrival_rates
            args.schedulers    = cfg.schedulers
            args.outdir        = cfg.outdir
        except Exception as e:
            print(f"Warning: could not load config ({e}). Using CLI args.")

    os.makedirs(args.outdir, exist_ok=True)
    print_banner()

    total_runs = len(args.schedulers) * len(args.arrival_rates) * len(args.seeds)
    rows: list[dict] = []

    if RICH_AVAILABLE and console:
        console.print(
            f"\n[dim]Cluster:[/dim] [bold]{args.total_gpus}[/bold] GPUs  "
            f"[dim]Jobs/run:[/dim] [bold]{args.n_jobs}[/bold]  "
            f"[dim]Total runs:[/dim] [bold]{total_runs}[/bold]\n"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running experiments...", total=total_runs)
            for sch in args.schedulers:
                for rate in args.arrival_rates:
                    for seed in args.seeds:
                        progress.update(
                            task,
                            description=(
                                f"[cyan]{SCHEDULER_LABELS[sch]:14s}[/cyan]"
                                f" rate=[yellow]{rate}[/yellow]"
                                f" seed=[dim]{seed}[/dim]"
                            ),
                        )
                        rows.append(run_one(sch, args.total_gpus, seed, rate, args.n_jobs))
                        progress.advance(task)
    else:
        for i, (sch, rate, seed) in enumerate(
            (s, r, sd)
            for s in args.schedulers
            for r in args.arrival_rates
            for sd in args.seeds
        ):
            print(f"[{i+1}/{total_runs}] {sch} rate={rate} seed={seed}")
            rows.append(run_one(sch, args.total_gpus, seed, rate, args.n_jobs))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "results.csv")
    df.to_csv(csv_path, index=False)

    if RICH_AVAILABLE and console:
        console.print(f"\n[green]OK[/green] Wrote [bold]{csv_path}[/bold]\n")
        print_summary_table(df)
        console.print("\n[bold]Generating plots...[/bold]")
    else:
        print(f"\nWrote {csv_path}")

    make_plots(df, args.schedulers, args.outdir)

    if RICH_AVAILABLE and console:
        console.print(
            f"\n[bold green]Done![/bold green] "
            f"Results in [cyan]{args.outdir}/[/cyan]"
        )
    else:
        print(f"\nDone! Results in {args.outdir}/")


if __name__ == "__main__":
    main()
