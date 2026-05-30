
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
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Work Study & Line Balancing App",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #fbfbfd;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #ececf1;
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }
    .block-container {padding-top: 1.5rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
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
    rounding: int = 0,
) -> pd.DataFrame:
    """
    Calculates selected time, basic time, standard time and capacity.
    Time unit follows the paper: centiminutes by default.
    480 minutes/day = 48000 centiminutes/day.
    """

    out = df.copy()

    required_cols = ["operation"]
    for c in required_cols:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    # Auto-detect observed time columns
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

    # Available time in centiminutes because input time is centiminutes
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
    efficiency = actual_output_per_day / standard_output_100 * 100 if standard_output_100 and standard_output_100 > 0 else np.nan
    work_content_minutes = total_standard_time / 100

    return {
        "total_standard_time_centimin": total_standard_time,
        "work_content_minutes_per_piece": work_content_minutes,
        "total_manpower": total_manpower,
        "standard_output_100": standard_output_100,
        "actual_output_per_day": actual_output_per_day,
        "productivity_piece_worker_day": productivity_per_worker,
        "line_efficiency_percent": efficiency,
    }


def compare_existing_proposed(existing_summary: Dict[str, float], proposed_summary: Dict[str, float]) -> Dict[str, float]:
    existing_wc = existing_summary["work_content_minutes_per_piece"]
    proposed_wc = proposed_summary["work_content_minutes_per_piece"]
    existing_std_out = existing_summary["standard_output_100"]
    proposed_std_out = proposed_summary["standard_output_100"]

    work_content_reduction = ((existing_wc - proposed_wc) / existing_wc) * 100 if existing_wc > 0 else np.nan
    productivity_improvement = ((proposed_std_out - existing_std_out) / existing_std_out) * 100 if existing_std_out > 0 else np.nan
    efficiency_improvement = (
        (proposed_summary["line_efficiency_percent"] - existing_summary["line_efficiency_percent"])
        / existing_summary["line_efficiency_percent"] * 100
        if existing_summary["line_efficiency_percent"] > 0 else np.nan
    )

    return {
        "work_content_reduction_percent": work_content_reduction,
        "productivity_improvement_percent": productivity_improvement,
        "efficiency_improvement_percent": efficiency_improvement,
    }


def sequential_line_balance(
    df: pd.DataFrame,
    cycle_time: float,
    time_col: str = "standard_time",
    order_col: str = "sl_no",
) -> pd.DataFrame:
    """
    Simple automatic line balancing for sequential assembly operations.
    It groups consecutive operations into workstations without exceeding cycle time
    where possible. If a single task exceeds cycle time, it becomes one station
    and is flagged as bottleneck.
    """

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
        op_name = row["operation"]

        # If task alone exceeds cycle time, assign to a separate station
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

        # Fit into current station
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
    work["station_utilization_percent"] = (work["station_time"] / cycle_time) * 100
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
    s["utilization_percent"] = (s["station_time"] / cycle_time) * 100
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


def plot_bar(df, x, y, title, horizontal=False):
    if PLOTLY_AVAILABLE:
        if horizontal:
            fig = px.bar(df, x=y, y=x, orientation="h", title=title, text=y)
            fig.update_layout(height=max(450, 18 * len(df)), yaxis={"categoryorder": "total ascending"})
        else:
            fig = px.bar(df, x=x, y=y, title=title, text=y)
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.25 if horizontal else 5)))
        if horizontal:
            ax.barh(df[x], df[y])
            ax.set_xlabel(y)
        else:
            ax.bar(df[x], df[y])
            ax.set_ylabel(y)
            plt.xticks(rotation=45, ha="right")
        ax.set_title(title)
        st.pyplot(fig)


# ------------------------------------------------------------
# Demo data, shortened from the paper structure
# Users can replace this by uploading complete CSV/Excel data.
# ------------------------------------------------------------
def demo_existing_data():
    data = [
        [1, "Cutting all leather", 566, 560, 490, 545, 550, 80, "M/c", 1],
        [2, "Cutting all PVC, lining, reinforcement", 273, 262, 279, 269, 275, 80, "M/c", 3],
        [3, "Inspection and numbering", 108, 118, 120, 111, 115, 80, "Manual", 2],
        [4, "Splitting leather", 33, 37, 34, 29, 35, 80, "M/c", 1],
        [5, "Skiving leather, rabus and PVC", 240, 250, 235, 246, 238, 85, "M/c", 3],
        [6, "Sewing back part face to face", 40, 37, 32, 42, 35, 80, "M/c", 1],
        [7, "Double way tape attaching to back joining and hammering", 105, 110, 100, 120, 118, 85, "Manual", 2],
        [8, "Double stitching to back part", 22, 23, 27, 19, 25, 80, "M/c", 1],
        [9, "Gluing on back leather and foam and top folding", 230, 221, 225, 235, 218, 80, "Manual", 2],
        [10, "Gluing on back trim part and folding", 208, 230, 220, 227, 223, 80, "Manual", 2],
        [15, "Gluing on front leather and folding", 259, 275, 250, 265, 269, 85, "Manual", 4],
        [21, "Front and inner trimming sewing 2.5 mm", 89, 93, 97, 85, 95, 80, "M/c", 1],
        [23, "Gluing on back inner trim and folding", 167, 156, 163, 160, 159, 80, "Manual", 1],
        [28, "Gluing on gusset and EVA and top folding", 211, 201, 209, 200, 204, 75, "Manual", 2],
        [38, "Piping stitching with front and back parts", 320, 322, 329, 334, 337, 75, "M/c", 1],
        [47, "Gusset stitching with front and back part", 425, 437, 439, 420, 423, 80, "M/c", 1],
        [57, "Thread burning, cleaning and polishing", 389, 361, 379, 357, 385, 80, "Manual", 3],
        [60, "Price tag attaching and packaging", 233, 227, 220, 229, 240, 80, "Manual", 1],
    ]
    return pd.DataFrame(
        data,
        columns=["sl_no", "operation", "obs1", "obs2", "obs3", "obs4", "obs5", "rating", "machine_or_manual", "manpower"],
    )


def default_proposed_from_existing(existing_df):
    proposed = existing_df.copy()

    # Example improvement rules based on Table 3 logic:
    # remove inspection, reduce adhesive/gluing tasks, reduce CNC-applicable sewing tasks, reduce staffing.
    reduction_rules = {
        "Inspection and numbering": 0.00,
        "Gluing on back leather and foam and top folding": 0.80,
        "Gluing on back trim part and folding": 0.85,
        "Gluing on front leather and folding": 0.80,
        "Front and inner trimming sewing 2.5 mm": 0.50,
        "Gluing on back inner trim and folding": 0.75,
        "Gluing on gusset and EVA and top folding": 0.83,
    }

    for op, factor in reduction_rules.items():
        mask = proposed["operation"].str.lower() == op.lower()
        obs_cols = [c for c in proposed.columns if c.startswith("obs")]
        proposed.loc[mask, obs_cols] = proposed.loc[mask, obs_cols] * factor

    proposed["proposal_note"] = ""
    proposed.loc[proposed["operation"].str.contains("Gluing", case=False, na=False), "proposal_note"] = (
        "Suggested spraying/water-based adhesive method to reduce application time."
    )
    proposed.loc[proposed["operation"].str.contains("Inspection", case=False, na=False), "proposal_note"] = (
        "Avoid or transfer inspection/numbering from the selected line where feasible."
    )
    proposed.loc[proposed["operation"].str.contains("sewing 2.5", case=False, na=False), "proposal_note"] = (
        "CNC or fixture-assisted stitching for curved/small components."
    )
    return proposed


def load_uploaded(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def template_dataframe():
    return pd.DataFrame(
        {
            "sl_no": [1, 2, 3],
            "operation": ["Operation A", "Operation B", "Operation C"],
            "obs1": [120, 90, 150],
            "obs2": [118, 92, 148],
            "obs3": [122, 91, 152],
            "obs4": [121, 89, 151],
            "obs5": [119, 90, 149],
            "rating": [80, 85, 100],
            "machine_or_manual": ["Manual", "M/c", "Manual"],
            "manpower": [1, 1, 2],
        }
    )


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

standard_rating = st.sidebar.number_input("Standard rating", min_value=1.0, value=100.0, step=1.0)
relaxation_allowance = st.sidebar.number_input("Relaxation allowance (%)", min_value=0.0, value=15.0, step=0.5)
contingency_allowance = st.sidebar.number_input("Contingency allowance (%)", min_value=0.0, value=3.0, step=0.5)
working_minutes_per_day = st.sidebar.number_input("Working time per day (minutes)", min_value=1.0, value=480.0, step=15.0)
actual_output_existing = st.sidebar.number_input("Actual existing output/day", min_value=0.0, value=240.0, step=1.0)
actual_output_proposed = st.sidebar.number_input("Actual proposed output/day", min_value=0.0, value=240.0, step=1.0)

st.sidebar.markdown("---")
balance_mode = st.sidebar.selectbox("Line balancing target", ["Takt/cycle time from demand", "Manual cycle time"])
if balance_mode == "Takt/cycle time from demand":
    demand_per_day = st.sidebar.number_input("Required demand/day", min_value=1.0, value=240.0, step=1.0)
    cycle_time = (working_minutes_per_day * 100) / demand_per_day
else:
    cycle_time = st.sidebar.number_input("Cycle time (centiminutes)", min_value=1.0, value=500.0, step=10.0)

st.sidebar.info(f"Current cycle time: **{cycle_time:.2f} centiminutes** ({cycle_time/100:.2f} min)")


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("⚙️ Work Study, Productivity Analysis & Automatic Line Balancing")
st.caption(
    "Upload operation-time data, calculate basic/standard time, identify bottlenecks, and automatically group operations into balanced stations."
)

with st.expander("Required input format"):
    st.write(
        """
        Upload CSV or Excel with at least:
        **operation**, **rating**, **manpower**, and either **selected_time** or observed-time columns named **obs1, obs2, obs3...**.
        Optional columns: **sl_no**, **machine_or_manual**, **proposal_note**.
        Time unit should be **centiminutes** to match the paper-style calculation.
        """
    )
    st.dataframe(template_dataframe(), use_container_width=True)
    st.download_button(
        "Download input template CSV",
        data=template_dataframe().to_csv(index=False).encode("utf-8"),
        file_name="work_study_input_template.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# Data input
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["1) Data & Time Study", "2) Productivity Comparison", "3) Automatic Line Balancing", "4) Downloads"]
)

with tab1:
    st.subheader("Data input")

    uploaded_existing = st.file_uploader("Upload existing-line CSV/Excel", type=["csv", "xlsx"], key="existing")
    uploaded_proposed = st.file_uploader("Upload proposed-line CSV/Excel (optional)", type=["csv", "xlsx"], key="proposed")

    if uploaded_existing:
        existing_raw = load_uploaded(uploaded_existing)
    else:
        existing_raw = demo_existing_data()
        st.info("Using built-in demo data. Upload your full 60-operation table for the real analysis.")

    if uploaded_proposed:
        proposed_raw = load_uploaded(uploaded_proposed)
    else:
        proposed_raw = default_proposed_from_existing(existing_raw)

    st.markdown("#### Existing line data")
    existing_raw = st.data_editor(existing_raw, use_container_width=True, num_rows="dynamic", key="edit_existing")

    st.markdown("#### Proposed line data")
    proposed_raw = st.data_editor(proposed_raw, use_container_width=True, num_rows="dynamic", key="edit_proposed")

    existing_calc = compute_work_study(
        existing_raw,
        standard_rating=standard_rating,
        relaxation_allowance=relaxation_allowance,
        contingency_allowance=contingency_allowance,
        working_minutes_per_day=working_minutes_per_day,
        rounding=0,
    )
    proposed_calc = compute_work_study(
        proposed_raw,
        standard_rating=standard_rating,
        relaxation_allowance=relaxation_allowance,
        contingency_allowance=contingency_allowance,
        working_minutes_per_day=working_minutes_per_day,
        rounding=0,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Existing line calculated table")
        st.dataframe(existing_calc, use_container_width=True)
    with c2:
        st.markdown("#### Proposed line calculated table")
        st.dataframe(proposed_calc, use_container_width=True)

    st.markdown("#### Bottleneck view")
    bottleneck_n = st.slider("Show top N bottlenecks", 5, min(30, len(existing_calc)), 10)
    bottleneck_df = existing_calc.sort_values("standard_time", ascending=False).head(bottleneck_n)
    plot_bar(bottleneck_df, "operation", "standard_time", "Top bottleneck operations by standard time", horizontal=True)

with tab2:
    st.subheader("Productivity and work-content comparison")

    existing_summary = production_summary(existing_calc, actual_output_existing, working_minutes_per_day)
    proposed_summary = production_summary(proposed_calc, actual_output_proposed, working_minutes_per_day)
    comparison = compare_existing_proposed(existing_summary, proposed_summary)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Existing work content", f"{existing_summary['work_content_minutes_per_piece']:.2f} min/piece")
    m2.metric("Proposed work content", f"{proposed_summary['work_content_minutes_per_piece']:.2f} min/piece")
    m3.metric("Work-content reduction", f"{comparison['work_content_reduction_percent']:.2f}%")
    m4.metric("Productivity improvement", f"{comparison['productivity_improvement_percent']:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Existing standard output", f"{existing_summary['standard_output_100']:.0f} pcs/day")
    m6.metric("Proposed standard output", f"{proposed_summary['standard_output_100']:.0f} pcs/day")
    m7.metric("Existing line efficiency", f"{existing_summary['line_efficiency_percent']:.2f}%")
    m8.metric("Proposed line efficiency", f"{proposed_summary['line_efficiency_percent']:.2f}%")

    summary_df = pd.DataFrame(
        [
            {"Scenario": "Existing", **existing_summary},
            {"Scenario": "Proposed", **proposed_summary},
        ]
    )

    st.markdown("#### Summary table")
    st.dataframe(summary_df, use_container_width=True)

    chart_df = pd.DataFrame(
        {
            "Metric": ["Work content (min/piece)", "Standard output (pcs/day)", "Line efficiency (%)"],
            "Existing": [
                existing_summary["work_content_minutes_per_piece"],
                existing_summary["standard_output_100"],
                existing_summary["line_efficiency_percent"],
            ],
            "Proposed": [
                proposed_summary["work_content_minutes_per_piece"],
                proposed_summary["standard_output_100"],
                proposed_summary["line_efficiency_percent"],
            ],
        }
    ).melt(id_vars="Metric", var_name="Scenario", value_name="Value")

    if PLOTLY_AVAILABLE:
        fig = px.bar(chart_df, x="Metric", y="Value", color="Scenario", barmode="group", text="Value")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(title="Existing vs Proposed Performance", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(chart_df, use_container_width=True)

with tab3:
    st.subheader("Automatic line balancing")

    line_for_balance = st.radio("Use which line for balancing?", ["Proposed line", "Existing line"], horizontal=True)
    balance_source = proposed_calc if line_for_balance == "Proposed line" else existing_calc

    balanced_df = sequential_line_balance(balance_source, cycle_time=cycle_time)
    station_df = station_summary(balanced_df, cycle_time=cycle_time)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("No. of auto stations", f"{station_df['auto_station'].nunique()}")
    s2.metric("Max station time", f"{station_df['station_time'].max():.0f} centimin")
    s3.metric("Avg station utilization", f"{station_df['utilization_percent'].mean():.2f}%")
    s4.metric("Balancing efficiency", f"{station_df['station_time'].sum()/(len(station_df)*cycle_time)*100:.2f}%")

    st.markdown("#### Auto-balanced operation assignment")
    st.dataframe(balanced_df, use_container_width=True)

    st.markdown("#### Station summary")
    st.dataframe(station_df, use_container_width=True)

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            station_df,
            x="auto_station",
            y="station_time",
            text="station_time",
            title="Station load after automatic line balancing",
        )
        fig.add_hline(y=cycle_time, line_dash="dash", annotation_text="Cycle time")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(xaxis_title="Station", yaxis_title="Station time (centiminutes)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        plot_bar(station_df, "auto_station", "station_time", "Station load after automatic line balancing")

    overload = station_df[station_df["station_time"] > cycle_time]
    if not overload.empty:
        st.warning(
            "Some stations exceed cycle time. Consider adding manpower, splitting tasks, method improvement, or increasing cycle time."
        )
        st.dataframe(overload, use_container_width=True)

with tab4:
    st.subheader("Export results")

    balanced_df = sequential_line_balance(proposed_calc, cycle_time=cycle_time)
    station_df = station_summary(balanced_df, cycle_time=cycle_time)

    existing_summary = production_summary(existing_calc, actual_output_existing, working_minutes_per_day)
    proposed_summary = production_summary(proposed_calc, actual_output_proposed, working_minutes_per_day)
    comparison = compare_existing_proposed(existing_summary, proposed_summary)

    all_summary = {
        **{f"existing_{k}": v for k, v in existing_summary.items()},
        **{f"proposed_{k}": v for k, v in proposed_summary.items()},
        **comparison,
        "cycle_time_centimin": cycle_time,
    }

    excel_file = make_download_excel(existing_calc, proposed_calc, balanced_df, station_df, all_summary)

    st.download_button(
        "Download full Excel report",
        data=excel_file,
        file_name="work_study_line_balancing_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        "Download calculated proposed line CSV",
        data=proposed_calc.to_csv(index=False).encode("utf-8"),
        file_name="proposed_line_calculated.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download auto-balanced line CSV",
        data=balanced_df.to_csv(index=False).encode("utf-8"),
        file_name="auto_balanced_line.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Formula basis: selected time → basic time using rating; basic time → standard time using relaxation and contingency allowances; "
    "capacity/day uses available daily time and manpower; productivity improvement compares proposed and existing standard outputs."
)
