"""
GDaaS Simulator — Interactive Streamlit Dashboard
===================================================
Launch with:
    streamlit run dashboard/app.py

Requires:
    pip install -e ".[dashboard]"
    (installs streamlit and plotly in addition to core dependencies)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GDaaS Simulator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error(
        "plotly is required for the dashboard. "
        "Install it with: `pip install plotly`"
    )
    st.stop()

from gdaas_sim.cluster.cluster import GPUCluster
from gdaas_sim.metrics.collector import MetricsCollector
from gdaas_sim.scheduler.backfill import EASYBackfillScheduler
from gdaas_sim.scheduler.edf import EDFScheduler
from gdaas_sim.scheduler.fair_share import TenantFairScheduler
from gdaas_sim.scheduler.fifo import FIFOScheduler
from gdaas_sim.scheduler.priority import PriorityScheduler
from gdaas_sim.scheduler.round_robin import RoundRobinScheduler
from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.sim.engine import SimEngine
from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SCHEDULER_MAP = {
    "FIFO":          FIFOScheduler,
    "SJF":           SJFScheduler,
    "Fair Share":    TenantFairScheduler,
    "EDF":           EDFScheduler,
    "Priority":      PriorityScheduler,
    "Round Robin":   RoundRobinScheduler,
    "EASY Backfill": EASYBackfillScheduler,
}

COLORS = {
    "FIFO":          "#2196F3",
    "SJF":           "#F44336",
    "Fair Share":    "#4CAF50",
    "EDF":           "#FF9800",
    "Priority":      "#9C27B0",
    "Round Robin":   "#00BCD4",
    "EASY Backfill": "#E91E63",
}


# ──────────────────────────────────────────────────────────────────────────────
# Simulation runner (cached so re-renders don't re-run the sim)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(
    scheduler_name: str,
    total_gpus: int,
    n_jobs: int,
    arrival_rate: float,
    n_tenants: int,
    seed: int,
) -> dict:
    cfg = WorkloadConfig(
        n_jobs=n_jobs,
        arrival_rate=arrival_rate,
        n_tenants=n_tenants,
        seed=seed,
    )
    jobs = generate_synthetic(cfg)
    cluster = GPUCluster(total_gpus=total_gpus)
    metrics = MetricsCollector()
    engine = SimEngine()
    scheduler = SCHEDULER_MAP[scheduler_name]()
    finished = engine.run(jobs=jobs, cluster=cluster,
                          scheduler=scheduler, metrics=metrics)

    sim_end = engine.time
    util = metrics.busy_gpu_time / (total_gpus * sim_end) if sim_end > 0 else 0.0

    wait_sorted = sorted(metrics.wait_times) if metrics.wait_times else []
    p95_idx = int(0.95 * (len(wait_sorted) - 1)) if wait_sorted else 0

    job_records = []
    for j in finished.values():
        if j.start_time is not None and j.finish_time is not None:
            job_records.append({
                "job_id":    j.job_id,
                "tenant_id": j.tenant_id,
                "start":     j.start_time,
                "finish":    j.finish_time,
                "wait":      round(j.start_time - j.arrival_time, 3),
                "duration":  round(j.duration, 3),
                "gpus":      j.gpus_required,
                "priority":  j.priority,
            })

    return {
        "scheduler":      scheduler_name,
        "utilization":    util,
        "avg_wait":       sum(metrics.wait_times) / len(metrics.wait_times)
                          if metrics.wait_times else 0.0,
        "p95_wait":       wait_sorted[p95_idx] if wait_sorted else 0.0,
        "jobs_finished":  metrics.finishes,
        "sla_wait_viol":  metrics.sla_wait_violations,
        "sla_dl_viol":    metrics.sla_deadline_violations,
        "jain":           metrics.jain_fairness(),
        "tenant_gpu_time": dict(metrics.tenant_gpu_time),
        "job_df":         pd.DataFrame(job_records),
        "sim_end":        sim_end,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🖥️ GDaaS Simulator")
    st.caption("GPU-as-a-Service Discrete-Event Simulator")
    st.divider()

    st.subheader("Cluster")
    total_gpus = st.slider("Total GPUs", min_value=4, max_value=64, value=16, step=4)

    st.subheader("Workload")
    n_jobs       = st.slider("Number of Jobs", 50, 1000, 300, step=50)
    arrival_rate = st.slider("Arrival Rate (jobs/time unit)", 0.1, 2.0, 0.5, step=0.1)
    n_tenants    = st.slider("Number of Tenants", 1, 8, 3)
    seed         = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

    st.subheader("Schedulers to Compare")
    selected = []
    defaults = {"FIFO", "SJF", "EASY Backfill"}
    for name in SCHEDULER_MAP:
        if st.checkbox(name, value=(name in defaults)):
            selected.append(name)

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    st.caption(
        "**EASY Backfill** is the algorithm used by SLURM, "
        "the scheduler powering most of the Top500 supercomputers."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────
st.title("GDaaS Simulator — GPU Scheduling Research Dashboard")
st.markdown(
    "Compare **7 GPU scheduling algorithms** across utilization, wait time, "
    "SLA compliance, and multi-tenant fairness metrics using a "
    "discrete-event simulation engine built from scratch."
)

if not run_btn:
    st.info(
        "Configure your experiment in the sidebar, then click **▶ Run Simulation**.",
        icon="ℹ️",
    )
    st.stop()

if not selected:
    st.warning("Select at least one scheduler in the sidebar.", icon="⚠️")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Run simulations
# ──────────────────────────────────────────────────────────────────────────────
results = []
prog = st.progress(0, text="Starting simulations...")
for i, name in enumerate(selected):
    prog.progress((i + 1) / len(selected), text=f"Running {name}...")
    r = run_simulation(name, total_gpus, n_jobs, arrival_rate, n_tenants, int(seed))
    results.append(r)
prog.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_results, tab_gantt, tab_fairness, tab_data = st.tabs(
    ["📊 Results", "📅 Job Timeline (Gantt)", "⚖️ Fairness", "🗂️ Raw Data"]
)

# ── Tab 1: Results ────────────────────────────────────────────────────────────
with tab_results:
    # Metric cards
    st.subheader("Key Metrics by Scheduler")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            st.markdown(f"**{r['scheduler']}**")
            st.metric("GPU Utilization", f"{r['utilization']:.1%}")
            st.metric("Avg Wait",        f"{r['avg_wait']:.2f}")
            st.metric("P95 Wait",        f"{r['p95_wait']:.2f}")
            st.metric("Jobs Finished",   r["jobs_finished"])

    st.divider()

    df_results = pd.DataFrame([
        {
            "Scheduler":       r["scheduler"],
            "Utilization":     r["utilization"],
            "Avg Wait":        r["avg_wait"],
            "P95 Wait":        r["p95_wait"],
            "SLA Wait Viol.":  r["sla_wait_viol"],
            "SLA DL Viol.":    r["sla_dl_viol"],
        }
        for r in results
    ])

    color_map = {r["scheduler"]: COLORS.get(r["scheduler"], "#888888") for r in results}

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            df_results, x="Scheduler", y="Utilization",
            title="GPU Utilization", color="Scheduler",
            color_discrete_map=color_map,
            text_auto=".1%",
            range_y=[0, 1.05],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.bar(
            df_results, x="Scheduler", y="Avg Wait",
            title="Average Wait Time", color="Scheduler",
            color_discrete_map=color_map,
            text_auto=".2f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig = px.bar(
            df_results, x="Scheduler", y="P95 Wait",
            title="P95 Wait Time (tail latency)", color="Scheduler",
            color_discrete_map=color_map,
            text_auto=".2f",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        sla_df = df_results.melt(
            id_vars="Scheduler",
            value_vars=["SLA Wait Viol.", "SLA DL Viol."],
            var_name="Violation Type",
            value_name="Count",
        )
        fig = px.bar(
            sla_df, x="Scheduler", y="Count",
            color="Violation Type", barmode="group",
            title="SLA Violations",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Gantt Chart ────────────────────────────────────────────────────────
with tab_gantt:
    if len(results) > 1:
        gantt_scheduler = st.selectbox(
            "Select scheduler for Gantt chart", [r["scheduler"] for r in results]
        )
        gantt_r = next(r for r in results if r["scheduler"] == gantt_scheduler)
    else:
        gantt_r = results[0]

    max_jobs = st.slider("Max jobs to display", 20, 200, 80, step=10)
    df_gantt = gantt_r["job_df"].head(max_jobs).copy()

    if df_gantt.empty:
        st.info("No completed jobs to display.")
    else:
        # Convert float timestamps to datetime for Plotly timeline
        origin = pd.Timestamp("2024-01-01")
        df_gantt["Start"] = origin + pd.to_timedelta(df_gantt["start"], unit="s")
        df_gantt["Finish"] = origin + pd.to_timedelta(df_gantt["finish"], unit="s")

        fig = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="Finish",
            y="tenant_id",
            color="tenant_id",
            hover_data={
                "job_id": True,
                "wait": True,
                "duration": True,
                "gpus": True,
                "priority": True,
                "Start": False,
                "Finish": False,
            },
            title=f"Job Execution Timeline — {gantt_r['scheduler']} "
                  f"(first {len(df_gantt)} jobs)",
            labels={"tenant_id": "Tenant"},
        )
        fig.update_yaxes(categoryorder="category ascending")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each bar = one job.  Bars are coloured by tenant.  "
            "Hover for wait time, duration, GPU count, and priority."
        )


# ── Tab 3: Fairness ───────────────────────────────────────────────────────────
with tab_fairness:
    st.subheader("Jain's Fairness Index")
    st.markdown(
        "Jain's index = 1.0 means **perfectly equal** GPU allocation across tenants.  "
        "Lower values indicate some tenants received disproportionately more resources."
    )

    jain_df = pd.DataFrame([
        {"Scheduler": r["scheduler"], "Jain Index": r["jain"] or 0.0}
        for r in results
    ])
    fig = px.bar(
        jain_df, x="Scheduler", y="Jain Index",
        color="Scheduler",
        color_discrete_map=color_map,
        text_auto=".4f",
        range_y=[0, 1.05],
        title="Jain's Fairness Index (higher = fairer)",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Perfect fairness")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Tenant GPU Time")
    cols = st.columns(min(len(results), 4))
    for col, r in zip(cols, results):
        with col:
            gpu_time = r["tenant_gpu_time"]
            if gpu_time:
                fig = px.pie(
                    names=list(gpu_time.keys()),
                    values=list(gpu_time.values()),
                    title=r["scheduler"],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Raw Data ───────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Simulation Summary")
    summary_rows = [
        {
            "Scheduler":          r["scheduler"],
            "Utilization":        f"{r['utilization']:.4f}",
            "Avg Wait":           f"{r['avg_wait']:.3f}",
            "P95 Wait":           f"{r['p95_wait']:.3f}",
            "Jobs Finished":      r["jobs_finished"],
            "SLA Wait Viol.":     r["sla_wait_viol"],
            "SLA DL Viol.":       r["sla_dl_viol"],
            "Jain Fairness":      f"{r['jain']:.4f}" if r["jain"] else "-",
            "Sim End Time":       f"{r['sim_end']:.2f}",
        }
        for r in results
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.subheader("Per-Job Records (first selected scheduler)")
    r0 = results[0]
    st.dataframe(r0["job_df"], use_container_width=True)

    csv = r0["job_df"].to_csv(index=False).encode()
    st.download_button(
        label="Download job records as CSV",
        data=csv,
        file_name=f"jobs_{r0['scheduler']}.csv",
        mime="text/csv",
    )
