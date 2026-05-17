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
from gdaas_sim.scheduler.gang import GangScheduler
from gdaas_sim.scheduler.preemptive_priority import PreemptivePriorityScheduler
from gdaas_sim.scheduler.priority import PriorityScheduler
from gdaas_sim.scheduler.round_robin import RoundRobinScheduler
from gdaas_sim.scheduler.sjf import SJFScheduler
from gdaas_sim.scheduler.weighted_fair import WeightedFairShareScheduler
from gdaas_sim.sim.engine import SimEngine
from gdaas_sim.workloads.synthetic import WorkloadConfig, generate_synthetic

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SCHEDULER_MAP = {
    "FIFO":                   FIFOScheduler,
    "SJF":                    SJFScheduler,
    "Fair Share":             TenantFairScheduler,
    "EDF":                    EDFScheduler,
    "Priority":               PriorityScheduler,
    "Round Robin":            RoundRobinScheduler,
    "EASY Backfill":          EASYBackfillScheduler,
    "Preemptive Priority":    PreemptivePriorityScheduler,
    "Weighted Fair Share":    WeightedFairShareScheduler,
    "Gang":                   GangScheduler,
}

COLORS = {
    "FIFO":                "#2196F3",
    "SJF":                 "#F44336",
    "Fair Share":          "#4CAF50",
    "EDF":                 "#FF9800",
    "Priority":            "#9C27B0",
    "Round Robin":         "#00BCD4",
    "EASY Backfill":       "#E91E63",
    "Preemptive Priority": "#795548",
    "Weighted Fair Share": "#607D8B",
    "Gang":                "#FF5722",
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
    spot_frac: float = 0.0,
    gang_frac: float = 0.0,
) -> dict:
    cfg = WorkloadConfig(
        n_jobs=n_jobs,
        arrival_rate=arrival_rate,
        n_tenants=n_tenants,
        seed=seed,
        spot_frac=spot_frac,
        gang_frac=gang_frac,
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
    p95_idx = int(0.95 * (len(wait_sorted) - 1)) if len(wait_sorted) > 1 else 0

    job_records = []
    for j in finished.values():
        if j.start_time is not None and j.finish_time is not None:
            job_records.append({
                "job_id":    j.job_id,
                "tenant_id": j.tenant_id,
                "start":     j.start_time,
                "finish":    j.finish_time,
                "wait":      round((j.first_start_time or j.start_time) - j.arrival_time, 3),
                "duration":  round(j.duration, 3),
                "gpus":      j.gpus_required,
                "priority":  j.priority,
                "spot":      j.spot,
                "preemptions": j.preemption_count,
            })

    total_cost = sum(metrics.tenant_cost.values())

    # Build utilization history as fraction over time
    util_history = []
    for t, busy in metrics.utilization_snapshots:
        util_history.append({"time": t, "busy_gpus": busy,
                              "utilization": busy / total_gpus if total_gpus > 0 else 0.0})

    return {
        "scheduler":        scheduler_name,
        "utilization":      util,
        "avg_wait":         sum(metrics.wait_times) / len(metrics.wait_times)
                            if metrics.wait_times else 0.0,
        "p95_wait":         wait_sorted[p95_idx] if wait_sorted else 0.0,
        "jobs_finished":    metrics.finishes,
        "sla_wait_viol":    metrics.sla_wait_violations,
        "sla_dl_viol":      metrics.sla_deadline_violations,
        "jain":             metrics.jain_fairness(),
        "drf":              metrics.drf_fairness(),
        "tenant_gpu_time":  dict(metrics.tenant_gpu_time),
        "tenant_cost":      dict(metrics.tenant_cost),
        "total_cost":       total_cost,
        "preemptions":      metrics.preemptions,
        "job_df":           pd.DataFrame(job_records),
        "util_history":     pd.DataFrame(util_history),
        "sim_end":          sim_end,
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

    st.subheader("Advanced Workload")
    spot_frac = st.slider("Spot Job Fraction", 0.0, 0.5, 0.0, step=0.05,
                          help="Fraction of jobs that are spot/preemptible (lower cost)")
    gang_frac = st.slider("Gang Job Fraction", 0.0, 0.3, 0.0, step=0.05,
                          help="Fraction of jobs that belong to gangs (all-or-nothing allocation)")

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
    "Compare **10 GPU scheduling algorithms** across utilization, wait time, "
    "SLA compliance, multi-tenant fairness, and cost metrics using a "
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
    r = run_simulation(
        name, total_gpus, n_jobs, arrival_rate, n_tenants, int(seed),
        spot_frac=spot_frac, gang_frac=gang_frac,
    )
    results.append(r)
prog.empty()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_results, tab_gantt, tab_fairness, tab_cost, tab_data = st.tabs(
    ["📊 Results", "📅 Job Timeline", "⚖️ Fairness", "💰 Cost & Efficiency", "🗂️ Raw Data"]
)

color_map = {r["scheduler"]: COLORS.get(r["scheduler"], "#888888") for r in results}

# ── Tab 1: Results ────────────────────────────────────────────────────────────
with tab_results:
    st.subheader("Key Metrics by Scheduler")
    cols = st.columns(len(results))
    for col, r in zip(cols, results):
        with col:
            st.markdown(f"**{r['scheduler']}**")
            st.metric("GPU Utilization", f"{r['utilization']:.1%}")
            st.metric("Avg Wait",        f"{r['avg_wait']:.2f}")
            st.metric("P95 Wait",        f"{r['p95_wait']:.2f}")
            st.metric("Jobs Finished",   r["jobs_finished"])
            st.metric("Preemptions",     r["preemptions"])

    st.divider()

    df_results = pd.DataFrame([
        {
            "Scheduler":       r["scheduler"],
            "Utilization":     r["utilization"],
            "Avg Wait":        r["avg_wait"],
            "P95 Wait":        r["p95_wait"],
            "SLA Wait Viol.":  r["sla_wait_viol"],
            "SLA DL Viol.":    r["sla_dl_viol"],
            "Preemptions":     r["preemptions"],
        }
        for r in results
    ])

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


# ── Tab 2: Job Timeline (Gantt) ───────────────────────────────────────────────
with tab_gantt:
    if len(results) > 1:
        gantt_scheduler = st.selectbox(
            "Select scheduler for timeline", [r["scheduler"] for r in results]
        )
        gantt_r = next(r for r in results if r["scheduler"] == gantt_scheduler)
    else:
        gantt_r = results[0]

    color_by = st.radio("Color by", ["Tenant", "GPU Count"], horizontal=True)
    max_jobs = st.slider("Max jobs to display", 20, 200, 80, step=10)
    df_gantt = gantt_r["job_df"].head(max_jobs).copy()

    if df_gantt.empty:
        st.info("No completed jobs to display.")
    else:
        origin = pd.Timestamp("2024-01-01")
        df_gantt["Start"] = origin + pd.to_timedelta(df_gantt["start"], unit="s")
        df_gantt["Finish"] = origin + pd.to_timedelta(df_gantt["finish"], unit="s")
        df_gantt["gpus_str"] = df_gantt["gpus"].astype(str) + " GPU(s)"

        color_col = "tenant_id" if color_by == "Tenant" else "gpus_str"
        color_label = "Tenant" if color_by == "Tenant" else "GPU Count"

        fig = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="Finish",
            y="tenant_id",
            color=color_col,
            hover_data={
                "job_id": True,
                "wait": True,
                "duration": True,
                "gpus": True,
                "priority": True,
                "spot": True,
                "preemptions": True,
                "Start": False,
                "Finish": False,
            },
            title=f"Job Execution Timeline — {gantt_r['scheduler']} "
                  f"(first {len(df_gantt)} jobs)",
            labels={"tenant_id": "Tenant", color_col: color_label},
        )
        fig.update_yaxes(categoryorder="category ascending")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each bar = one job. Hover for wait time, duration, GPU count, priority, and spot status."
        )

    # Utilization over time
    st.subheader("GPU Utilization Over Time")
    util_df = gantt_r["util_history"]
    if not util_df.empty:
        fig_util = px.line(
            util_df, x="time", y="utilization",
            title=f"GPU Utilization Over Time — {gantt_r['scheduler']}",
            labels={"time": "Simulation Time", "utilization": "Utilization Fraction"},
        )
        fig_util.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_util, use_container_width=True)
    else:
        st.info("No utilization snapshots available.")


# ── Tab 3: Fairness ───────────────────────────────────────────────────────────
with tab_fairness:
    st.subheader("Fairness Indices")
    st.markdown(
        "**Jain's index**: 1.0 = perfectly equal GPU allocation across tenants.  \n"
        "**DRF index**: 1.0 = perfectly proportional GPU allocation. "
        "Higher = fairer for both metrics."
    )

    fairness_df = pd.DataFrame([
        {
            "Scheduler":  r["scheduler"],
            "Jain Index": r["jain"] or 0.0,
            "DRF Index":  r["drf"] or 0.0,
        }
        for r in results
    ])

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fig = px.bar(
            fairness_df, x="Scheduler", y="Jain Index",
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

    with col_f2:
        fig = px.bar(
            fairness_df, x="Scheduler", y="DRF Index",
            color="Scheduler",
            color_discrete_map=color_map,
            text_auto=".4f",
            range_y=[0, 1.05],
            title="DRF Fairness Index (higher = fairer)",
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


# ── Tab 4: Cost & Efficiency ──────────────────────────────────────────────────
with tab_cost:
    st.subheader("Cost & Efficiency Analysis")

    cost_df = pd.DataFrame([
        {
            "Scheduler":        r["scheduler"],
            "Total Cost":       r["total_cost"],
            "Jobs Finished":    r["jobs_finished"],
            "Preemptions":      r["preemptions"],
            "Cost Efficiency":  (r["jobs_finished"] / r["total_cost"]
                                 if r["total_cost"] > 0 else 0.0),
        }
        for r in results
    ])

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig = px.bar(
            cost_df, x="Scheduler", y="Total Cost",
            color="Scheduler",
            color_discrete_map=color_map,
            text_auto=".2f",
            title="Total Compute Cost (GPU-hours × rate)",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        fig = px.bar(
            cost_df, x="Scheduler", y="Cost Efficiency",
            color="Scheduler",
            color_discrete_map=color_map,
            text_auto=".3f",
            title="Cost Efficiency (jobs finished per unit cost)",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Per-tenant cost breakdown (grouped bar)
    st.subheader("Per-Tenant Cost Breakdown")
    tenant_cost_rows = []
    for r in results:
        for tenant, cost in r["tenant_cost"].items():
            tenant_cost_rows.append({
                "Scheduler": r["scheduler"],
                "Tenant":    tenant,
                "Cost":      cost,
            })

    if tenant_cost_rows:
        tc_df = pd.DataFrame(tenant_cost_rows)
        fig = px.bar(
            tc_df, x="Tenant", y="Cost",
            color="Scheduler",
            color_discrete_map=color_map,
            barmode="group",
            title="Per-Tenant Compute Cost by Scheduler",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Preemption counts
    st.subheader("Preemption Counts")
    if cost_df["Preemptions"].sum() == 0:
        st.info("No preemptions occurred. Enable Preemptive Priority scheduler and set spot_frac > 0 to see preemptions.")
    else:
        fig = px.bar(
            cost_df, x="Scheduler", y="Preemptions",
            color="Scheduler",
            color_discrete_map=color_map,
            text_auto=True,
            title="Number of Job Preemptions by Scheduler",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Raw Data ───────────────────────────────────────────────────────────
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
            "DRF Fairness":       f"{r['drf']:.4f}" if r["drf"] else "-",
            "Total Cost":         f"{r['total_cost']:.2f}",
            "Preemptions":        r["preemptions"],
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
