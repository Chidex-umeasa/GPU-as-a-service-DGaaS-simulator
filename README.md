# GDaaS Simulator (GPU-as-a-Service)

A research-oriented **discrete-event simulator** for studying **GPU-as-a-Service (GDaaS)** systems: job arrivals, queueing, scheduling policies, GPU utilization, and latency/throughput tradeoffs — without needing real GPUs.

This project is part of a broader **AI + Education + Systems** research portfolio, with a systems focus on:
- GPU resource provisioning and scheduling
- multi-tenant fairness (planned)
- SLA / deadline behavior (planned)
- cost-aware allocation (planned)
- fractional GPU / MIG-style partitioning (planned)

---

## Project Goals

### What this simulator helps you evaluate
- **Scheduling policies** (FIFO, SJF, and more)
- **Cluster utilization** under different loads
- **Queueing delay** (avg + tail latency)
- **Turnaround time**
- (Planned) **Fairness** across tenants (Jain index)
- (Planned) **SLA violations** (max wait, deadlines)
- (Planned) **Cost** and efficiency metrics

---

## Repo Structure
gdaas-sim/
pyproject.toml
README.md
src/gdaas_sim/
init.py
sim/engine.py # discrete-event simulation engine
cluster/cluster.py # GPU cluster capacity model
workloads/synthetic.py # synthetic job generation
scheduler/
base.py # scheduler interface
fifo.py # FIFO baseline
sjf.py # SJF baseline
metrics/collector.py # utilization + wait/turnaround metrics
experiments/run_experiment.py # experiment runner (CSV + plots)


---
## Requirements
- Python **3.10+**
- Dependencies are defined in `pyproject.toml`:
  - numpy
  - pandas
  - matplotlib
---
## Install (Editable)
From the repo root:
```bash
pip install -e .

Run Experiment
Run the baseline experiment sweep:
python experiments/run_experiment.py

Outputs will be created in:
experiments/out/
  results.csv
  avg_wait.png
  p95_wait.png
  utilization.png

You can change GPU count, seeds, and arrival rates:
python experiments/run_experiment.py --total_gpus 32 --seeds 1 2 3 4 --arrival_rates 0.2 0.5 0.9 1.2

How It Works (High-Level)
The simulator is event-driven:
ARRIVE: job enters the scheduler queue
FINISH: job completes; GPUs are released
After each event, the engine asks the scheduler for runnable jobs and dispatches them while resources allow.
Schedulers are pluggable, so you can add new scheduling algorithms without changing the simulation engine.

Current Features (MVP)
✅ Discrete-event simulation engine (arrivals + completions)
✅ GPU cluster capacity model
✅ Synthetic workload generator (Poisson arrivals, lognormal durations)
✅ Baseline schedulers:
-FIFO
-SJF (non-preemptive)
✅ Metrics:
-utilization
-average wait time
-P95 wait time
-average turnaround time
✅ Experiment runner that outputs CSV + plots

Roadmap (Research Extensions)
Phase 1B
-Multi-tenant workloads (tenant_id)
-Jain’s fairness index
-Weighted quotas / fair sharing (GDaaS-style)
Phase 2
-SLA constraints (deadline/max-wait)
-Preemption
-Cost-aware scheduling (heterogeneous GPU types)
Phase 3
-Fractional GPUs / MIG partitioning
-RL scheduler baseline (optional)

Reproducibility
-Workload generation uses deterministic seeds
-Experiments run multiple seeds for stability
-Output CSV is suitable for further analysis and paper-style plots