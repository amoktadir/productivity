# ============================================================
# Streamlit App: Work Study + Automatic Line Balancing
# Based on Moktadir et al. (2017) productivity improvement logic
# Author-ready template for industrial engineering analysis
# ============================================================

import io
import math
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Work Study & Line Balancing",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e3e6f0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #4e73df;
    }
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #2c3e50; font-weight: 700; letter-spacing: -0.5px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def compute_work_study(
    df: pd.DataFrame,
    standard_rating: float = 100,
    relaxation_allowance: float = 15,
    contingency_allowance: float = 3,
    working_minutes_per_day: float = 480,
    rounding: int = 2,
) -> pd.DataFrame:
    """Calculates selected, basic, standard time and capacity."""
    out = df.copy()

    required_cols = ["operation"]
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    observed_cols = [c for c in out.columns if str(c).lower().startswith("obs")]
    if "selected_time" not in out.columns:
        if observed_cols:
            out["selected_time"] = out[observed_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            out["selected_time"] = np.nan

    out["selected_time"] = safe_numeric(out["selected_time"])
    out["rating"] = safe_numeric(out.get("rating", pd.Series([standard_rating] * len(out))), standard_rating)
    out["manpower"] = safe_numeric(out.get("manpower", pd.Series([1] * len(out))), 1)
    out["machine_or_manual"] = out.get("machine_or_manual", "Manual")

    allowance_factor = 1 + (relaxation_allowance + contingency_allowance) / 100

    out["basic_time"] = out["selected_time"] * (out["rating"] / standard_rating)
    out["standard_time"] = out["basic_time"] * allowance_factor

    available_centimin = working_minutes_per_day * 100
    out["capacity_day_100"] = (available_centimin * out["manpower"]) / out["standard_time"].replace(0, np.nan)
    out["capacity_day_85"] = out["capacity_day_100"] * 0.85

    if rounding is not None:
        out["selected_time"] = out["selected_time"].round(rounding)
        out["basic_time"] = out["basic_time"].round(rounding)
        out["standard_time"] = out["standard_time"].round(rounding)
        out["capacity_day_100"] = out["capacity_day_100"].round(0)
        out["capacity_day_85"] = out["capacity_day_85"].round(0)

    return out


def production_summary(
    df: pd.DataFrame,
    actual_output_per_day: float,
    working_minutes_per_day: float = 480,
) -> Dict[str, float]:
    total_standard_time = float(pd.to_numeric(df["standard_time"], errors="coerce").fillna(0).sum())
    total_manpower = float(pd.to_numeric(df.get("manpower", pd.Series([1]*len(df))), errors="coerce").fillna(1).sum())

    available_centimin_total = total_manpower * working_minutes_per_day * 100
    standard_output_100 = available_centimin_total / total_standard_time if total_standard_time > 0 else np.nan
    productivity_per_worker = actual_output_per_day / total_manpower if total_manpower > 0 else np.nan
    
    # Pure decimal calculation (no explicit scaling factor)
    efficiency = actual_output_per_day / standard_output_100 if standard_output_100 and standard_output_100 > 0 else np.nan
    work_content_minutes = total_standard_time / 100

    return {
        "total_standard_time_centimin": total_standard_time,
        "work_content_minutes_per_piece": work_content_minutes,
        "total_manpower": total_manpower,
        "standard_output_100": standard_output_100,
        "actual_output_per_day": actual_output_per_day,
        "productivity_piece_worker_day": productivity_per_worker,
        "line_efficiency": efficiency,
    }


def compare_existing_proposed(existing_summary: Dict[str, float], proposed_summary: Dict[str, float]) -> Dict[str, float]:
    existing_wc = existing_summary["work_content_minutes_per_piece"]
    proposed_wc = proposed_summary["work_content_minutes_per_piece"]
    existing_std_out = existing_summary["standard_output_100"]
    proposed_std_out = proposed_summary["standard_output_100"]

    # Pure decimal calculations (no explicit scaling factor)
    work_content_reduction = ((existing_wc - proposed_wc) / existing_wc) if existing_wc > 0 else np.nan
    productivity_improvement = ((proposed_std_out - existing_std_out) / existing_std_out) if existing_std_out > 0 else np.nan
    efficiency_improvement = (
        (proposed_summary["line_efficiency"] - existing_summary["line_efficiency"]) / existing_summary["line_efficiency"]
        if existing_summary["line_efficiency"] > 0 else np.nan
    )

    return {
        "work_content_reduction": work_content_reduction,
        "productivity_improvement": productivity_improvement,
        "efficiency_improvement": efficiency_improvement,
    }


def sequential_line_balance(
    df: pd.DataFrame,
    cycle_time: float,
    time_col: str = "standard_time",
    order_col: str = "sl_no",
) -> pd.DataFrame:
    work = df.copy()
    if order_col in work.columns:
        work = work.sort_values(order_col)
    else:
        work = work.reset_index(drop=True)
        work[order_col] = np.arange(1, len(work) + 1)

    stations = []
    current_station = 1
    current_time = 0.0
    current_ops = []

    for _, row in work.iterrows():
        t = float(row[time_col])
        if t > cycle_time:
            if current_ops:
                for op_idx in current_ops:
                    stations[op_idx] = current_station
                current_station += 1
                current_ops = []
                current_time = 0.0

            stations.append(current_station)
            current_station += 1
            continue

        if current_time + t <= cycle_time or not current_ops:
            stations.append(current_station)
            current_ops.append(len(stations) - 1)
            current_time += t
        else:
            current_station += 1
            current_time = t
            current_ops = [len(stations)]
            stations.append(current_station)

    work["auto_station"] = stations
    station_time = work.groupby("auto_station")[time_col].transform("sum")
    work["station_time"] = station_time
    work["balance_loss_centimin"] = cycle_time - work["station_time"]
    
    # Pure decimal utilization (no scaling factor)
    work["station_utilization"] = (work["station_time"] / cycle_time)
    work["bottleneck_flag"] = work[time_col] > cycle_time

    return work


def station_summary(balanced_df: pd.DataFrame, cycle_time: float) -> pd.DataFrame:
    s = (
        balanced_df.groupby("auto_station")
        .agg(
            operations=("operation", lambda x: " | ".join(x.astype(str))),
            station_time=("standard_time", "sum"),
            n_operations=("operation", "count"),
            has_bottleneck=("bottleneck_flag", "max"),
        )
        .reset_index()
    )
    s["idle_time"] = cycle_time - s["station_time"]
    # Pure decimal utilization
    s["utilization"] = (s["station_time"] / cycle_time)
    return s


def make_download_excel(existing_df, proposed_df, balanced_df, station_df, summaries):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        existing_df.to_excel(writer, index=False, sheet_name="Existing_Line")
        proposed_df.to_excel(writer, index=False, sheet_name="Proposed_Line")
        balanced_df.to_excel(writer, index=False, sheet_name="Auto_Balanced_Line")
        station_df.to_excel(writer, index=False, sheet_name="Station_Summary")
        pd.DataFrame([summaries]).to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)
    return output


# ------------------------------------------------------------
# Demo Data Injection
# ------------------------------------------------------------
def demo_existing_data():
    data = [
        [1, "Cutting all leather", 566, 560, 490, 545, 550, 80, "M/c", 1],
        [2, "Cutting all PVC, lining, reinforcement", 273, 262, 279, 269, 275, 80, "M/c", 3],
        [3, "Inspection and numbering", 108, 118, 120, 111, 115, 80, "Manual", 2],
        [4, "Splitting leather", 33, 37, 34, 29, 35, 80, "M/c", 1],
        [5, "Skiving leather, rabus and PVC", 240, 250, 235, 246, 238, 85, "M/c", 3],
        [6, "Sewing back part face to face", 40, 37, 32, 42, 35, 80, "M/c", 1],
        [7, "Tape attaching and hammering", 105, 110, 100, 120, 118, 85, "Manual", 2],
        [8, "Double stitching to back part", 22, 23, 27, 19, 25, 80, "M/c", 1],
        [9, "Gluing on back leather and folding", 230, 221, 225, 235, 218, 80, "Manual", 2],
        [10, "Gluing on back trim part and folding", 208, 230, 220, 227, 223, 80, "Manual", 2]
    ]
    return pd.DataFrame(data, columns=["sl_no", "operation", "obs1", "obs2", "obs3", "obs4", "obs5", "rating", "machine_or_manual", "manpower"])


def default_proposed_from_existing(existing_df):
    proposed = existing_df.copy()
    reduction_rules = {
        "Inspection and numbering": 0.00,
        "Gluing on back leather and folding": 0.80,
        "Gluing on back trim part and folding": 0.85,
    }
    for op, factor in reduction_rules.items():
        mask = proposed["operation"].str.lower() == op.lower()
        obs_cols = [c for c in proposed.columns if c.startswith("obs")]
        proposed.loc[mask, obs_cols] = proposed.loc[mask, obs_cols] * factor

    proposed["proposal_note"] = ""
    proposed.loc[proposed["operation"].str.contains("Gluing", case=False, na=False), "proposal_note"] = "Use spray adhesive"
    return proposed


def template_dataframe():
    return pd.DataFrame({
        "sl_no": [1, 2],
        "operation": ["Operation A", "Operation B"],
        "obs1": [120, 90], "obs2": [118, 92], "obs3": [122, 91],
        "obs4": [121, 89], "obs5": [119, 90],
        "rating": [80, 85], "machine_or_manual": ["Manual", "M/c"], "manpower": [1, 1],
    })


# ------------------------------------------------------------
# Sidebar setup
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043064.png", width=60)
    st.title("Line Configuration")
    
    with st.expander("⏱️ Time Allowances", expanded=True):
        standard_rating = st.number_input("Standard Rating", min_value=1.0, value=100.0, step=1.0)
        relaxation_allowance = st.number_input("Relaxation Allowance (%)", min_value=0.0, value=15.0, step=0.5)
        contingency_allowance = st.number_input("Contingency Allowance (%)", min_value=0.0, value=3.0, step=0.5)
        working_minutes_per_day = st.number_input("Working Mins/Day", min_value=1.0, value=480.0, step=15.0)
        
    with st.expander("📊 Output Metrics", expanded=True):
        actual_output_existing = st.number_input("Actual Existing Output/Day", min_value=0.0, value=240.0, step=1.0)
        actual_output_proposed = st.number_input("Actual Proposed Output/Day", min_value=0.0, value=240.0, step=1.0)

    st.markdown("---")
    balance_mode = st.selectbox("Line Balancing Target", ["Takt Time (Demand Based)", "Manual Cycle Time"])
    if balance_mode == "Takt Time (Demand Based)":
        demand_per_day = st.number_input("Required Demand/Day", min_value=1.0, value=240.0, step=1.0)
        cycle_time = (working_minutes_per_day * 100) / demand_per_day
    else:
        cycle_time = st.number_input("Cycle Time (centiminutes)", min_value=1.0, value=500.0, step=10.0)

    st.success(f"**Target Cycle Time:**\n{cycle_time:.2f} cMin ({cycle_time/100:.2f} mins)")


# ------------------------------------------------------------
# Main Dashboard UI
# ------------------------------------------------------------
st.title("Industrial Work Study & Productivity Dashboard")
st.caption("Advanced interactive toolkit for time study, line efficiency tracking, and automated bottleneck stratification.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Data Input & Study", 
    "📈 Productivity Diagnostics", 
    "⚖️ Auto Line Balancing", 
    "💾 Export Options"
])

with tab1:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_existing = st.file_uploader("Upload Existing Line (CSV/Excel)", type=["csv", "xlsx"])
    with col_up2:
        uploaded_proposed = st.file_uploader("Upload Proposed Line (CSV/Excel)", type=["csv", "xlsx"])

    existing_raw = pd.read_csv(uploaded_existing) if uploaded_existing else demo_existing_data()
    proposed_raw = pd.read_csv(uploaded_proposed) if uploaded_proposed else default_proposed_from_existing(existing_raw)

    st.markdown("#### 1. Edit Source Data")
    existing_raw = st.data_editor(existing_raw, use_container_width=True, num_rows="dynamic")
    
    # Background calculations
    existing_calc = compute_work_study(existing_raw, standard_rating, relaxation_allowance, contingency_allowance, working_minutes_per_day)
    proposed_calc = compute_work_study(proposed_raw, standard_rating, relaxation_allowance, contingency_allowance, working_minutes_per_day)

    st.markdown("#### 2. Bottleneck Stratification")
    if PLOTLY_AVAILABLE:
        bottleneck_df = existing_calc.sort_values("standard_time", ascending=True).tail(15)
        fig = px.bar(
            bottleneck_df, x="standard_time", y="operation", 
            color="machine_or_manual", orientation="h",
            title="Operation Time Distribution (Top 15 Bottlenecks)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto='.0f'
        )
        fig.update_layout(template="plotly_white", yaxis_title="", xaxis_title="Standard Time (cMin)", height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    existing_summary = production_summary(existing_calc, actual_output_existing, working_minutes_per_day)
    proposed_summary = production_summary(proposed_calc, actual_output_proposed, working_minutes_per_day)
    comparison = compare_existing_proposed(existing_summary, proposed_summary)

    st.markdown("### Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Work Content Reduction", f"{comparison['work_content_reduction']:.2%}")
    m2.metric("Productivity Improvement", f"{comparison['productivity_improvement']:.2%}")
    m3.metric("Existing Efficiency", f"{existing_summary['line_efficiency']:.2%}")
    m4.metric("Proposed Efficiency", f"{proposed_summary['line_efficiency']:.2%}", delta=f"{comparison['efficiency_improvement']:.2%}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        # Standard Output Comparison
        if PLOTLY_AVAILABLE:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=["Existing", "Proposed"], y=[existing_summary['standard_output_100'], proposed_summary['standard_output_100']], 
                                  name="Output (pcs/day)", marker_color="#4e73df", texttemplate="%{y:.0f}", textposition="outside"))
            fig1.update_layout(title="Standard Output Capacity", template="plotly_white", yaxis_title="Pieces per day")
            st.plotly_chart(fig1, use_container_width=True)

    with c2:
        # Work Content Comparison
        if PLOTLY_AVAILABLE:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=["Existing", "Proposed"], y=[existing_summary['work_content_minutes_per_piece'], proposed_summary['work_content_minutes_per_piece']], 
                                  name="Work Content", marker_color="#1cc88a", texttemplate="%{y:.2f}", textposition="outside"))
            fig2.update_layout(title="Total Work Content", template="plotly_white", yaxis_title="Minutes per piece")
            st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col_radio, _ = st.columns([1, 2])
    with col_radio:
        line_for_balance = st.radio("Active Balancing Source:", ["Proposed Line", "Existing Line"], horizontal=True)
    
    balance_source = proposed_calc if line_for_balance == "Proposed Line" else existing_calc
    balanced_df = sequential_line_balance(balance_source, cycle_time=cycle_time)
    station_df = station_summary(balanced_df, cycle_time=cycle_time)

    st.markdown("### Station Configuration Results")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Required Stations", f"{station_df['auto_station'].nunique()}")
    s2.metric("Peak Station Load", f"{station_df['station_time'].max():.0f} cMin")
    s3.metric("Average Station Utilization", f"{station_df['utilization'].mean():.2%}")
    s4.metric("System Balancing Efficiency", f"{station_df['station_time'].sum()/(len(station_df)*cycle_time):.2%}")

    if PLOTLY_AVAILABLE:
        fig_bal = px.bar(
            station_df, x="auto_station", y="station_time",
            title="Station Workload vs Target Cycle Time",
            text="station_time", color="utilization",
            color_continuous_scale="Viridis"
        )
        fig_bal.add_hline(y=cycle_time, line_dash="dash", line_color="red", annotation_text="Cycle Time Barrier")
        fig_bal.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_bal.update_layout(template="plotly_white", xaxis_title="Station ID", yaxis_title="Allocated Time (cMin)")
        st.plotly_chart(fig_bal, use_container_width=True)

    st.markdown("#### Station Utilization Breakdown")
    # Using st.column_config for a professional, interactive UI table
    st.dataframe(
        station_df[['auto_station', 'operations', 'station_time', 'utilization', 'has_bottleneck']], 
        use_container_width=True,
        column_config={
            "auto_station": st.column_config.NumberColumn("Station ID"),
            "operations": "Assigned Operations",
            "station_time": st.column_config.NumberColumn("Time (cMin)", format="%.1f"),
            "utilization": st.column_config.ProgressColumn(
                "Utilization %",
                help="Percentage of cycle time used",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "has_bottleneck": st.column_config.CheckboxColumn("Bottleneck Warning")
        }
    )

with tab4:
    st.markdown("### Data Export & Archiving")
    st.info("Download the mathematically synchronized data arrays for your final publication or secondary processing pipelines.")
    
    balanced_export = sequential_line_balance(proposed_calc, cycle_time=cycle_time)
    station_export = station_summary(balanced_export, cycle_time=cycle_time)

    all_summary = {
        **{f"existing_{k}": v for k, v in existing_summary.items()},
        **{f"proposed_{k}": v for k, v in proposed_summary.items()},
        **comparison,
        "cycle_time_centimin": cycle_time,
    }

    excel_file = make_download_excel(existing_calc, proposed_calc, balanced_export, station_export, all_summary)

    c1, c2, c3 = st.columns(3)
    c1.download_button("📥 Download Master Excel Report", data=excel_file, file_name="WorkStudy_Optimized_Report.xlsx")
    c2.download_button("📄 Download Proposed Line (CSV)", data=proposed_calc.to_csv(index=False).encode("utf-8"), file_name="Proposed_Calculated.csv")
    c3.download_button("📊 Download Station Array (CSV)", data=balanced_export.to_csv(index=False).encode("utf-8"), file_name="Auto_Balanced_Line.csv")
