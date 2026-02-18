# GDaaS Simulator

> **GPU-as-a-Service Discrete-Event Simulator** — a research-grade framework for studying GPU cluster scheduling, multi-tenant fairness, SLA enforcement, and utilization tradeoffs.

Built in Python 3.10+ with **no simulation framework dependencies** — just a heap-based event loop, pluggable schedulers, and clean metrics collection.

---

## Architecture

```
                          +------------------+
         Job arrivals --> |   SimEngine      |  <-- discrete-event heap
                          |  (engine.py)     |
                          +--------+---------+
                                   |
               +-------------------+-------------------+
               |                   |                   |
       +-------v------+   +--------v-----+   +---------v------+
       |  Scheduler   |   |  GPUCluster  |   | MetricsCollect |
       |  (pluggable) |   | (allocate /  |   | (wait, util,   |
       |  7 algorithms|   |  release)    |   |  fairness, SLA)|
       +--------------+   +--------------+   +----------------+
               |
    +----------+----------+----------+-----------+-----------+----------+
    |          |          |          |           |           |          |
  FIFO       SJF      FairShare    EDF       Priority  RoundRobin  EASYBackfill
```

**Event loop:** arrivals and completions are pushed onto a `heapq`. At each tick the engine pops the earliest event, calls the scheduler's `pick_next()`, allocates GPUs, and schedules the finish event. No external framework (SimPy, etc.) is used.

---

## Scheduling Algorithms

| Algorithm | Key Property | Complexity |
|-----------|-------------|------------|
| **FIFO** | Arrival order; blocks on head-of-queue | O(1) |
| **SJF** | Shortest expected duration first, non-preemptive | O(log n) heap |
| **Fair Share** | Lowest cumulative GPU-time tenant served first | O(n) scan, O(1) remove |
| **EDF** | Earliest absolute deadline first | O(log n) heap |
| **Priority** | User-assigned integer priority, FIFO tiebreak | O(log n) heap |
| **Round Robin** | Per-tenant turn-taking, FIFO within tenant | O(n) worst case |
| **EASY Backfill** | SLURM-style pivot reservation + backfill window | O(n log n) |

### EASY Backfill

The algorithm used by **SLURM** — the dominant scheduler on Top500 supercomputers. It works in two phases:

1. **Pivot reservation** — the head-of-queue job (`queue[0]`) gets a guaranteed start time called the *shadow time*, computed by summing freed GPUs from running jobs until the pivot's requirement is met.
2. **Backfill** — remaining queued jobs can run *now* if they (a) fit the currently available GPUs and (b) finish before the shadow time, so they never delay the pivot.

This allows small jobs to fill GPU gaps without starving large jobs, improving utilization significantly under mixed workloads.

---

## Quick Start

### Install

```bash
git clone <repo-url>
cd gdaas-sim

# Core + CLI
pip install -e ".[dev]"

# Core + CLI + dashboard
pip install -e ".[dev,dashboard]"
```

### Run the experiment CLI

```bash
# Default sweep: 7 schedulers x 3 arrival rates x 3 seeds = 63 runs
python experiments/run_experiment.py

# Custom sweep
python experiments/run_experiment.py --total_gpus 32 --seeds 1 2 3 4 5 --arrival_rates 0.2 0.5 0.9

# Using a YAML config file
python experiments/run_experiment.py --config experiments/config_default.yaml
```

Outputs a `results.csv` and six publication-quality PNG plots to `experiments/out/`.

### Launch the interactive dashboard

```bash
streamlit run dashboard/app.py
```

Adjust cluster size, arrival rate, and schedulers in the sidebar — the dashboard re-runs the simulation live and shows:
- Metric cards + bar charts (utilization, wait time, SLA violations)
- Plotly Gantt chart of the first 100 jobs, colored by tenant
- Jain fairness index + per-tenant GPU time breakdown

### Run the test suite

```bash
pytest tests/ -v
# 132 passed
```

### Use the Python API

```python
from gdaas_sim import (
    SimEngine, GPUCluster, MetricsCollector,
    WorkloadConfig, generate_synthetic,
    EASYBackfillScheduler,
)

cfg = WorkloadConfig(n_jobs=500, arrival_rate=0.5, seed=42)
jobs = generate_synthetic(cfg)

cluster = GPUCluster(total_gpus=16)
metrics = MetricsCollector()
engine  = SimEngine()

engine.run(jobs=jobs, cluster=cluster, scheduler=EASYBackfillScheduler(), metrics=metrics)

sim_end  = engine.time
util     = metrics.busy_gpu_time / (16 * sim_end)
avg_wait = sum(metrics.wait_times) / len(metrics.wait_times)

print(f"Utilization : {util:.1%}")
print(f"Avg wait    : {avg_wait:.1f} time units")
print(f"Jain index  : {metrics.jain_fairness():.4f}")
```

---

## Metrics

| Metric | Formula / Description |
|--------|-----------------------|
| **GPU Utilization** | `busy_gpu_time / (total_gpus * sim_duration)` |
| **Avg / P95 / P99 Wait** | Queue wait time distribution across all jobs |
| **Avg Turnaround** | `finish_time - arrival_time` averaged over all jobs |
| **SLA Wait Violations** | Count of jobs where `wait_time > job.max_wait` |
| **SLA Deadline Violations** | Count of jobs where `finish_time > job.deadline` |
| **Jain's Fairness Index** | `(sum xi)^2 / (n * sum xi^2)`; 1.0 = perfect fairness |

Utilization is computed as a continuous-time integral: `MetricsCollector` tracks the currently-busy GPU count and integrates it incrementally at every job start/finish event, giving an exact result regardless of job overlap patterns.

---

## Project Structure

```
gdaas-sim/
├── src/gdaas_sim/
│   ├── __init__.py              # Public API, v0.2.0
│   ├── sim/
│   │   └── engine.py            # SimEngine, Job dataclass, Event loop
│   ├── cluster/
│   │   └── cluster.py           # GPUCluster (allocate / release)
│   ├── scheduler/
│   │   ├── base.py              # BaseScheduler ABC
│   │   ├── fifo.py              # FIFOScheduler
│   │   ├── sjf.py               # SJFScheduler
│   │   ├── fair_share.py        # TenantFairScheduler
│   │   ├── edf.py               # EDFScheduler
│   │   ├── priority.py          # PriorityScheduler
│   │   ├── round_robin.py       # RoundRobinScheduler
│   │   └── backfill.py          # EASYBackfillScheduler (SLURM-style)
│   ├── metrics/
│   │   └── collector.py         # MetricsCollector (utilization, wait, fairness)
│   ├── workloads/
│   │   └── synthetic.py         # WorkloadConfig, generate_synthetic()
│   └── config.py                # ExperimentConfig (YAML loader)
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_engine.py
│   ├── test_cluster.py
│   ├── test_metrics.py          # Regression tests for duplicate-append bug
│   ├── test_workloads.py
│   ├── test_integration.py      # End-to-end: all 7 schedulers
│   └── test_schedulers/
│       ├── test_fifo.py
│       ├── test_sjf.py          # Regression: seq not mutated on re-insert
│       ├── test_fair_share.py
│       ├── test_edf.py
│       ├── test_priority.py
│       ├── test_round_robin.py
│       └── test_backfill.py     # Shadow time + backfill eligibility
│
├── experiments/
│   ├── run_experiment.py        # Rich CLI sweep runner
│   ├── config_default.yaml      # Default experiment config
│   └── out/                     # Generated CSVs + PNGs
│
├── dashboard/
│   └── app.py                   # Streamlit + Plotly interactive dashboard
│
└── pyproject.toml               # v0.2.0, optional [dashboard] extras
```

---

## Bug Fixes (v0.2.0)

Three correctness bugs were found and fixed in the original codebase:

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `metrics/collector.py` | `wait_times` and `turnaround_times` appended **twice** per job, doubling all statistics | Removed duplicate unconditional appends; values recorded once inside SLA-check block |
| 2 | `scheduler/sjf.py` | `pick_next()` discarded the original `seq` key on re-insert, inflating `_seq` and corrupting priority order | Preserve full `(dur, seq, job)` tuple; never call `self._seq += 1` during restore |
| 3 | `scheduler/fair_share.py` | `self._waiting.remove(chosen)` performed O(n) linear scan on every dispatch | Changed `_waiting` from `List[Job]` to `Dict[str, Job]`; deletion is now O(1) |
| 4 | `metrics/collector.py` | GPU utilization always reported **0.0** — integration only ran at `finalize()` when all GPUs already idle | Added `_current_busy_gpus` counter; `_record_interval()` integrates incrementally at every start/finish event |

---

## Roadmap

- [x] Discrete-event engine (pure Python, no SimPy)
- [x] 7 scheduling algorithms including EASY Backfill (SLURM-style)
- [x] SLA tracking (max-wait + deadline), Jain fairness index
- [x] 132-test pytest suite with bug-fix regression tests
- [x] Rich CLI with progress bar and results table
- [x] Streamlit dashboard with Plotly + Gantt chart
- [x] YAML configuration
- [ ] Preemption support
- [ ] Heterogeneous GPU types + cost modelling
- [ ] Fractional GPU / MIG partitioning
