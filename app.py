import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# markov_integration/ lives next to this file -- make it importable
# regardless of streamlit's CWD (the same class of bug hit earlier this
# session with scripts run by path instead of by module).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from markov_integration.trajectory_service import TrajectoryPredictionService

DB_PATH = Path(__file__).resolve().parent / "anpr_demo_fixed.db"

st.set_page_config(
    page_title="ANPR Trajectory Dashboard",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python repair_sqlite_import.py "
            f"--csv handoff_package_v2/anpr_hits_final.csv --db anpr_demo_fixed.db"
        )

    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_resource
def get_prediction_service():
    # Point the service at the SAME db this dashboard reads, regardless of
    # its own default (which is also this file, but explicit beats implicit
    # if the two ever diverge on a different machine).
    os.environ.setdefault("ANPR_DB_PATH", str(DB_PATH))
    return TrajectoryPredictionService()


@st.cache_data
def get_plates():
    conn = get_connection()

    query = """
        SELECT DISTINCT plate
        FROM events
        ORDER BY plate
        LIMIT 1000
    """

    return pd.read_sql_query(query, conn)["plate"].tolist()


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


@st.cache_data
def load_violations(plate: str = None, violation_type: str = None, limit: int = 1000):
    conn = get_connection()
    if not _table_exists(conn, "violations"):
        return pd.DataFrame()

    query = "SELECT * FROM violations WHERE 1=1"
    params = []
    if plate:
        query += " AND plate = ?"
        params.append(plate)
    if violation_type:
        query += " AND violation_type = ?"
        params.append(violation_type)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    return pd.read_sql_query(query, conn, params=params)


@st.cache_data
def load_repeat_offenders(min_count: int = 2):
    conn = get_connection()
    if not _table_exists(conn, "violations"):
        return pd.DataFrame()
    query = """
        SELECT plate, COUNT(*) as violation_count,
               GROUP_CONCAT(DISTINCT violation_type) as violation_types
        FROM violations
        WHERE plate IS NOT NULL AND plate != ''
        GROUP BY plate
        HAVING violation_count >= ?
        ORDER BY violation_count DESC
    """
    return pd.read_sql_query(query, conn, params=(min_count,))


@st.cache_data
def get_prediction_plates():
    conn = get_connection()
    if not _table_exists(conn, "camera_predictions"):
        return []
    query = "SELECT DISTINCT plate FROM camera_predictions ORDER BY plate LIMIT 1000"
    return pd.read_sql_query(query, conn)["plate"].tolist()


@st.cache_data
def load_predictions_for_plate(plate: str):
    conn = get_connection()
    query = """
        SELECT timestamp, current_camera, prev_camera, speed_mps, model_source,
               predicted_camera, predicted_prob, rank, actual_next_camera
        FROM camera_predictions
        WHERE plate = ?
        ORDER BY timestamp ASC, model_source ASC, rank ASC
    """
    return pd.read_sql_query(query, conn, params=(plate,))


@st.cache_data
def load_prediction_accuracy():
    conn = get_connection()
    if not _table_exists(conn, "camera_predictions"):
        return pd.DataFrame()
    query = """
        SELECT model_source,
               AVG(top1_hit) as top1_accuracy,
               AVG(topk_hit) as topk_accuracy,
               COUNT(*) as num_events
        FROM (
            SELECT plate, timestamp, model_source,
                   MAX(CASE WHEN rank = 1 AND predicted_camera = actual_next_camera
                            THEN 1 ELSE 0 END) as top1_hit,
                   MAX(CASE WHEN predicted_camera = actual_next_camera
                            THEN 1 ELSE 0 END) as topk_hit
            FROM camera_predictions
            WHERE actual_next_camera IS NOT NULL
            GROUP BY plate, timestamp, model_source
        ) t
        GROUP BY model_source
        ORDER BY topk_accuracy DESC
    """
    return pd.read_sql_query(query, conn)


@st.cache_data
def load_events(plate: str):
    conn = get_connection()

    query = """
        SELECT
            sim_time,
            camera_id,
            camera_lon,
            camera_lat,
            plate,
            vehicle_lon,
            vehicle_lat,
            speed_mps,
            camera_name,
            is_hub
        FROM events
        WHERE plate = ?
        ORDER BY sim_time ASC
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(plate,),
    )


# ---------------------------------------------------------------------------
# City-wide macro analytics (Component 3: heatmap, avg speed, route density,
# OD patterns, congestion bottlenecks, flow trend) -- all aggregated across
# EVERY camera node, unlike the per-plate queries above. speed_mps has a
# small number of corrupted sentinel values in the source data (2 rows at
# -1.07e9, an int32 NaN-cast artifact) -- every speed aggregate below filters
# to the physically plausible range [0, 50) m/s so those don't poison a mean.
# ---------------------------------------------------------------------------
_SPEED_FILTER = "speed_mps >= 0 AND speed_mps < 50"


@st.cache_data
def load_camera_summary():
    conn = get_connection()
    query = f"""
        SELECT camera_id, camera_name, camera_lat, camera_lon, is_hub,
               COUNT(*) as detections,
               AVG(CASE WHEN {_SPEED_FILTER} THEN speed_mps END) as avg_speed
        FROM events
        GROUP BY camera_id
        ORDER BY detections DESC
    """
    return pd.read_sql_query(query, conn)


@st.cache_data
def compute_bottleneck_scores():
    """Congestion proxy: cameras seeing MORE traffic than average while
    vehicles pass through SLOWER than average are where the city is
    actually backing up. Neither density nor speed alone tells you that --
    a camera can be busy and free-flowing (a highway), or quiet and slow
    (a residential dead-end isn't a bottleneck). Combining both as z-scores
    surfaces the ones that are busy AND slow."""
    df = load_camera_summary().dropna(subset=["avg_speed"]).copy()
    if len(df) < 2:
        df["bottleneck_score"] = 0.0
        return df
    density_z = (df["detections"] - df["detections"].mean()) / df["detections"].std(ddof=0)
    speed_z = (df["avg_speed"] - df["avg_speed"].mean()) / df["avg_speed"].std(ddof=0)
    df["bottleneck_score"] = density_z - speed_z  # high density, low speed -> high score
    return df.sort_values("bottleneck_score", ascending=False)


@st.cache_data
def load_route_edges(max_gap_seconds: float = 900.0, min_count: int = 3):
    """Camera-to-camera route density: for every plate's chronological
    camera sequence, count each consecutive (A -> B) hop (skipping
    same-camera repeats and gaps wider than max_gap_seconds -- those are two
    unrelated visits, not one continuous route). This is the same edge
    definition src.gnn_codes' graph_builder.build_graph uses for the
    trajectory-prediction graph, computed independently here in pandas so
    this dashboard doesn't need that module's torch_geometric dependency."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT plate, sim_time, camera_id, camera_lat, camera_lon FROM events "
        "ORDER BY plate, sim_time", conn)
    if df.empty:
        return pd.DataFrame(columns=["from_id", "to_id", "count", "from_lat", "from_lon", "to_lat", "to_lon"])

    edges = {}
    coords = {}
    for _plate, group in df.groupby("plate", sort=False):
        rows = group.itertuples(index=False)
        prev = next(rows, None)
        if prev is not None:
            coords[prev.camera_id] = (prev.camera_lat, prev.camera_lon)
        for row in rows:
            coords[row.camera_id] = (row.camera_lat, row.camera_lon)
            if row.camera_id != prev.camera_id and (row.sim_time - prev.sim_time) <= max_gap_seconds:
                key = (prev.camera_id, row.camera_id)
                edges[key] = edges.get(key, 0) + 1
            prev = row

    rows_out = [
        {"from_id": a, "to_id": b, "count": c,
         "from_lat": coords[a][0], "from_lon": coords[a][1],
         "to_lat": coords[b][0], "to_lon": coords[b][1]}
        for (a, b), c in edges.items() if c >= min_count
    ]
    return pd.DataFrame(rows_out).sort_values("count", ascending=False)


@st.cache_data
def load_od_matrix(max_gap_seconds: float = 900.0, top_n: int = 15):
    """Origin-destination pattern: for each plate's LATEST continuous
    segment (same gap rule as load_route_edges -- a segment starting fresh
    after a long absence is a new trip, not a continuation), origin = first
    camera, destination = last camera. Excludes single-camera segments
    (no destination to speak of)."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT plate, sim_time, camera_id FROM events ORDER BY plate, sim_time", conn)
    if df.empty:
        return pd.DataFrame(columns=["origin", "destination", "trips"])

    od_pairs = []
    for _plate, group in df.groupby("plate", sort=False):
        times = group["sim_time"].to_numpy()
        cams = group["camera_id"].to_numpy()
        seg_start = 0
        for i in range(1, len(times) + 1):
            if i == len(times) or (times[i] - times[i - 1]) > max_gap_seconds:
                origin, destination = cams[seg_start], cams[i - 1]
                if origin != destination:
                    od_pairs.append((origin, destination))
                seg_start = i
    if not od_pairs:
        return pd.DataFrame(columns=["origin", "destination", "trips"])
    od_df = pd.DataFrame(od_pairs, columns=["origin", "destination"])
    counts = od_df.value_counts().reset_index(name="trips")
    return counts.sort_values("trips", ascending=False).head(top_n)


@st.cache_data
def load_flow_trend(bucket_seconds: float = 1800.0):
    """City-wide detection volume over time, bucketed -- the closest
    single-camera-network proxy for a traffic-flow trend line. Buckets by
    raw sim_time (simulation seconds), NOT calendar time (see the
    sim_time-corruption note elsewhere in this file)."""
    conn = get_connection()
    query = f"""
        SELECT CAST(sim_time / {bucket_seconds} AS INT) * {bucket_seconds} as bucket_start,
               COUNT(*) as detections,
               AVG(CASE WHEN {_SPEED_FILTER} THEN speed_mps END) as avg_speed
        FROM events
        GROUP BY bucket_start
        ORDER BY bucket_start
    """
    return pd.read_sql_query(query, conn)


def render_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    title: str,
):
    map_df = df.dropna(subset=[lat_col, lon_col]).copy()

    if map_df.empty:
        st.info(f"No coordinates found for {title.lower()}.")
        return

    fig = px.scatter_map(
        map_df,
        lat=lat_col,
        lon=lon_col,
        color="camera_id",
        hover_data=[
            "sim_time",
            "camera_name",
            "speed_mps",
            "is_hub",
        ],
        zoom=11,
        center={
            "lat": map_df[lat_col].mean(),
            "lon": map_df[lon_col].mean(),
        },
        height=520,
    )

    fig.update_traces(marker={"size": 11})
    st.plotly_chart(fig, use_container_width=True)


def render_trajectory_tab(plates):
    with st.sidebar:
        st.header("Vehicle search")

        plate = st.selectbox(
            "Select a plate",
            plates,
        )

        map_mode = st.radio(
            "Map view",
            ["Camera detections", "Vehicle ground-truth path"],
        )

    df = load_events(plate)
    # sim_time is a raw SUMO simulation-seconds float, NOT a calendar
    # timestamp (fixed this session -- the original import_data.py ran
    # pd.to_datetime() on it, silently corrupting every row to a bogus
    # 1970-01-01-plus-N-seconds string; see repair_sqlite_import.py /
    # markov_integration's own docstrings for the full story). Keep it as
    # plain numeric seconds throughout this dashboard.

    first_seen = df["sim_time"].min()
    last_seen = df["sim_time"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total events", len(df))
    col2.metric("Cameras visited", df["camera_id"].nunique())
    col3.metric("First seen", f"t={first_seen:.0f}s" if pd.notna(first_seen) else "Unknown")
    col4.metric("Last seen", f"t={last_seen:.0f}s" if pd.notna(last_seen) else "Unknown")

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader(map_mode)

        if map_mode == "Camera detections":
            render_map(
                df,
                lat_col="camera_lat",
                lon_col="camera_lon",
                title="camera detections",
            )
        else:
            render_map(
                df,
                lat_col="vehicle_lat",
                lon_col="vehicle_lon",
                title="vehicle ground-truth path",
            )

    with right:
        st.subheader("Speed over time")

        speed_df = df.dropna(subset=["sim_time", "speed_mps"])

        if speed_df.empty:
            st.info("No speed data exists for this vehicle.")
        else:
            speed_chart = px.line(
                speed_df,
                x="sim_time",
                y="speed_mps",
                markers=True,
                labels={
                    "sim_time": "Simulation time (s)",
                    "speed_mps": "Speed (m/s)",
                },
            )
            st.plotly_chart(speed_chart, use_container_width=True)

        st.subheader("Camera visits")

        camera_counts = (
            df["camera_id"]
            .value_counts()
            .rename_axis("camera_id")
            .reset_index(name="detections")
        )

        camera_chart = px.bar(
            camera_counts,
            x="camera_id",
            y="detections",
            color="camera_id",
            labels={
                "camera_id": "Camera",
                "detections": "Detections",
            },
        )

        camera_chart.update_layout(showlegend=False)

        st.plotly_chart(camera_chart, use_container_width=True)

    st.subheader("Ordered event data for the ML model")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "See the \"Next-camera prediction\" tab for this plate's live and "
        "offline-evaluated next-camera predictions, and the \"Violations\" "
        "tab for any flagged rule violations."
    )


# ---------------------------------------------------------------------------
# Component 4: Alert System -- blacklist matches + route anomalies, raised by
# BOTH branches (src.violations.violation_detector's live pipeline, and
# gnn_codes/.../replay_demo.py's aggregate-branch replay) into one shared
# `alerts` table. This dashboard section is read-only + acknowledge; adding
# blacklist entries is scripts/manage_blacklist.py (kept as a CLI so it's
# usable from either branch's environment, not just this dashboard's venv).
# ---------------------------------------------------------------------------
def _get_write_connection():
    """A second, non-cached connection for writes (acknowledging alerts) --
    get_connection() is @st.cache_resource'd read-only-in-spirit; writes
    through it would be invisible to a fresh cached read in another session.
    Cheap: SQLite connections are lightweight, and this is only used for the
    rare "acknowledge" click, not the polling read path."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(ttl=4)
def load_active_alerts(limit: int = 200):
    conn = get_connection()
    if not _table_exists(conn, "alerts"):
        return pd.DataFrame()
    query = """
        SELECT id, timestamp, alert_type, plate, camera_id, detail, severity
        FROM alerts WHERE acknowledged = 0
        ORDER BY timestamp DESC LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=(limit,))


@st.cache_data(ttl=10)
def load_alert_history(limit: int = 500):
    conn = get_connection()
    if not _table_exists(conn, "alerts"):
        return pd.DataFrame()
    return pd.read_sql_query(
        "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))


def render_alerts_tab():
    conn = get_connection()
    if not _table_exists(conn, "alerts"):
        st.info(
            "No `alerts` table yet -- it's created the first time the live "
            "pipeline or replay_demo.py runs with the Alert System wired in "
            "(see src/violations/violation_detector.py and "
            "gnn_codes/chennai-demo/trajectory_project/replay_demo.py)."
        )
        return

    @st.fragment(run_every=5)
    def live_alerts_panel():
        active = load_active_alerts(limit=200)
        st.caption("Auto-refreshes every 5 seconds.")
        if active.empty:
            st.success("No active alerts.")
            return

        high = active[active["severity"] == "high"]
        medium = active[active["severity"] != "high"]
        col1, col2 = st.columns(2)
        col1.metric("High severity", len(high))
        col2.metric("Medium severity", len(medium))

        for row in active.itertuples():
            icon = "🚨" if row.severity == "high" else "⚠️"
            label = "Blacklisted vehicle" if row.alert_type == "blacklist_match" else "Route anomaly"
            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"{icon} **{label}** — plate `{row.plate}` at `{row.camera_id}`")
                    st.caption(row.detail)
                with right:
                    if st.button("Acknowledge", key=f"ack_{row.id}"):
                        write_conn = _get_write_connection()
                        write_conn.execute(
                            "UPDATE alerts SET acknowledged=1, acknowledged_by=?, acknowledged_at=? WHERE id=?",
                            ("dashboard-operator", pd.Timestamp.now().timestamp(), row.id))
                        write_conn.commit()
                        write_conn.close()
                        load_active_alerts.clear()
                        st.rerun()

    st.subheader("Active alerts")
    live_alerts_panel()

    st.divider()
    st.subheader("Blacklist")
    st.caption("Managed via `python scripts/manage_blacklist.py add/remove/list` from either branch's environment.")
    if _table_exists(conn, "blacklist"):
        bl = pd.read_sql_query("SELECT plate, reason, added_by, active FROM blacklist ORDER BY active DESC", conn)
        st.dataframe(bl, use_container_width=True, hide_index=True)
    else:
        st.info("No `blacklist` table yet.")

    st.subheader("Alert history")
    history = load_alert_history()
    if history.empty:
        st.info("No alerts logged yet.")
    else:
        type_counts = history["alert_type"].value_counts().rename_axis("alert_type").reset_index(name="count")
        hist_chart = px.bar(
            type_counts, x="alert_type", y="count", color="alert_type",
            labels={"alert_type": "Alert type", "count": "Count"},
        )
        hist_chart.update_layout(showlegend=False)
        st.plotly_chart(hist_chart, use_container_width=True, key="alert_history_chart")
        st.dataframe(history, use_container_width=True, hide_index=True)


def render_violations_tab():
    conn = get_connection()
    if not _table_exists(conn, "violations"):
        st.info(
            "No `violations` table yet -- it's created the first time "
            "smart_traffic_ppo's ViolationDetector runs (see "
            "src/violations/violation_detector.py)."
        )
        return

    all_violations = load_violations(limit=5000)
    if all_violations.empty:
        st.info("No violations logged yet.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total violations", len(all_violations))
    col2.metric("Vehicles flagged", all_violations["vehicle_id"].nunique())
    col3.metric(
        "Provisional (low-confidence)",
        int(all_violations["provisional"].sum()) if "provisional" in all_violations else 0,
    )

    st.subheader("Violations by type")
    type_counts = (
        all_violations["violation_type"]
        .value_counts()
        .rename_axis("violation_type")
        .reset_index(name="count")
    )
    type_chart = px.bar(
        type_counts, x="violation_type", y="count", color="violation_type",
        labels={"violation_type": "Violation type", "count": "Count"},
    )
    type_chart.update_layout(showlegend=False)
    st.plotly_chart(type_chart, use_container_width=True)

    st.subheader("Repeat offenders")
    repeat_offenders = load_repeat_offenders(min_count=2)
    if repeat_offenders.empty:
        st.info("No plate has >= 2 recorded violations yet.")
    else:
        st.dataframe(repeat_offenders, use_container_width=True, hide_index=True)

    st.subheader("All violations")
    violation_types = ["(all)"] + sorted(all_violations["violation_type"].unique().tolist())
    selected_type = st.selectbox("Filter by violation type", violation_types)
    filtered = load_violations(
        violation_type=None if selected_type == "(all)" else selected_type, limit=1000)
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_live_prediction_section():
    """NEW: live per-plate next-camera prediction, powered by
    markov_integration's TrajectoryPredictionService -- unlike the
    pre-computed `camera_predictions` table (only populated by the offline
    GNN+Markov replay demo), this queries the database directly for ANY
    plate present in `events` right now, with proper trajectory
    segmentation (a time-gap starts a new segment; only the latest segment
    is used) and consecutive-duplicate-camera collapsing before predicting.
    """
    st.subheader("Live prediction (any plate, queried now)")
    st.caption(
        "Markov-only (1st-order, falling back from 2nd-order when the exact "
        "2-camera state hasn't been seen before). Queries the database directly "
        "for this plate's most recent trajectory segment -- works for any plate "
        "in `events`, not just ones from the offline replay demo below."
    )
    service = get_prediction_service()
    plate = st.text_input("Enter a plate to predict for", key="live_predict_plate")
    if not plate:
        return

    result = service.predict_for_plate(plate.strip())
    if result["status"] == "plate_not_found":
        st.warning(f"No events found for plate {plate!r}.")
        return
    if result["status"] == "invalid_sim_time":
        st.error("This plate's events have non-numeric sim_time values (data issue).")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw events", result["raw_event_count"])
    col2.metric("Active segment events", result["active_segment_event_count"])
    col3.metric("Unique cameras (segment)", result["unique_camera_count"])
    st.caption(f"Active history used for prediction: {' -> '.join(result['active_history'])}")

    markov_result = result["markov_result"]
    if not markov_result or not markov_result["input_valid"]:
        st.info(markov_result["explanation"] if markov_result else "No prediction available.")
        return

    st.markdown(f"*{markov_result['explanation']}*")
    preds_df = pd.DataFrame(markov_result["top_predictions"])
    if preds_df.empty:
        st.info("No candidate next cameras.")
        return
    chart = px.bar(
        preds_df, x="camera_id", y="probability", color="camera_name",
        labels={"camera_id": "Candidate next camera", "probability": "Probability"},
    )
    st.plotly_chart(chart, use_container_width=True, key="live_prediction_chart")
    st.dataframe(preds_df, use_container_width=True, hide_index=True)


def render_prediction_tab():
    conn = get_connection()

    render_live_prediction_section()
    st.divider()

    st.subheader("Offline replay accuracy (GNN + Markov ensemble vs. ground truth)")
    st.caption(
        "Only populated by the offline replay demo "
        "(gnn_codes/chennai-demo/trajectory_project/replay_demo.py), which "
        "knows the real next camera from the synthetic dataset's ground "
        "truth -- includes the GNN model, which the live section above does "
        "not (Markov-only there, see its own caption)."
    )
    if not _table_exists(conn, "camera_predictions"):
        st.info("No `camera_predictions` table yet -- run replay_demo.py.")
        return

    accuracy_df = load_prediction_accuracy()
    if accuracy_df.empty:
        st.info("No ground-truth-backed predictions logged yet (run replay_demo.py).")
    else:
        display_df = accuracy_df.copy()
        display_df["top1_accuracy"] = (display_df["top1_accuracy"] * 100).round(1)
        display_df["topk_accuracy"] = (display_df["topk_accuracy"] * 100).round(1)
        display_df = display_df.rename(columns={
            "model_source": "Model", "top1_accuracy": "Top-1 accuracy (%)",
            "topk_accuracy": "Top-k accuracy (%)", "num_events": "Events",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        acc_chart = px.bar(
            accuracy_df, x="model_source", y=["top1_accuracy", "topk_accuracy"],
            barmode="group",
            labels={"model_source": "Model", "value": "Accuracy", "variable": "Metric"},
        )
        st.plotly_chart(acc_chart, use_container_width=True)

    st.subheader("Per-plate prediction lookup (offline replay results)")
    prediction_plates = get_prediction_plates()
    if not prediction_plates:
        st.info("No predictions logged for any plate yet.")
        return

    plate = st.selectbox("Select a plate", prediction_plates, key="prediction_plate")
    df = load_predictions_for_plate(plate)
    if df.empty:
        st.info("No predictions for this plate.")
        return

    events = df[["timestamp", "current_camera", "prev_camera", "actual_next_camera"]].drop_duplicates()
    event_options = [
        f"t={row.timestamp:.0f}  {row.current_camera} -> ? "
        f"(actual: {row.actual_next_camera or 'unknown'})"
        for row in events.itertuples()
    ]
    selected = st.selectbox("Select a prediction event", event_options)
    selected_row = events.iloc[event_options.index(selected)]

    event_df = df[
        (df["timestamp"] == selected_row["timestamp"])
        & (df["current_camera"] == selected_row["current_camera"])
    ]

    for model_source in ["gnn", "markov_1st", "markov_2nd", "ensemble"]:
        model_df = event_df[event_df["model_source"] == model_source].sort_values("rank")
        if model_df.empty:
            continue
        hit = selected_row["actual_next_camera"] in model_df["predicted_camera"].values
        st.markdown(f"**{model_source}** {'✅ actual camera in top-k' if hit else ''}")
        chart = px.bar(
            model_df, x="predicted_camera", y="predicted_prob",
            labels={"predicted_camera": "Candidate next camera", "predicted_prob": "Probability"},
        )
        if selected_row["actual_next_camera"]:
            chart.add_vline(x=selected_row["actual_next_camera"], line_dash="dash", line_color="green")
        st.plotly_chart(chart, use_container_width=True, key=f"pred_chart_{model_source}")


def render_city_analytics_tab():
    """City-wide macro traffic analytics -- aggregated across every camera
    node, in contrast to every other tab's per-plate view. Answers "how is
    the city moving right now", not "where has this vehicle been"."""
    summary = load_camera_summary()
    if summary.empty:
        st.info("No camera data available.")
        return

    total_detections = int(summary["detections"].sum())
    city_avg_speed = float((summary["avg_speed"] * summary["detections"]).sum() / summary["detections"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Total detections", f"{total_detections:,}")
    col2.metric("Camera nodes", len(summary))
    col3.metric("City-wide avg speed", f"{city_avg_speed:.2f} m/s")

    # -- Traffic density heatmap ------------------------------------------
    st.subheader("Traffic density heatmap")
    st.caption("Detection volume by camera location, city-wide.")
    heat_fig = px.density_map(
        summary, lat="camera_lat", lon="camera_lon", z="detections",
        radius=35, zoom=10.5,
        center={"lat": summary["camera_lat"].mean(), "lon": summary["camera_lon"].mean()},
        hover_name="camera_name", height=480,
        color_continuous_scale="Inferno",
    )
    st.plotly_chart(heat_fig, use_container_width=True, key="city_heatmap")

    left, right = st.columns(2)

    with left:
        # -- Average speed by camera node ----------------------------------
        st.subheader("Average speed by camera node")
        speed_df = summary.dropna(subset=["avg_speed"]).sort_values("avg_speed")
        speed_chart = px.bar(
            speed_df, x="avg_speed", y="camera_name", orientation="h",
            color="is_hub",
            labels={"avg_speed": "Avg speed (m/s)", "camera_name": "Camera", "is_hub": "Hub"},
            height=560,
        )
        st.plotly_chart(speed_chart, use_container_width=True, key="speed_by_camera")

    with right:
        # -- Congestion bottleneck detection ------------------------------
        st.subheader("Congestion bottlenecks")
        st.caption(
            "High detection volume + low average speed, as z-scores "
            "(`density_z - speed_z`) -- busy AND slow, not just busy or just slow."
        )
        bottlenecks = compute_bottleneck_scores()
        top_bottlenecks = bottlenecks.head(10)
        bn_chart = px.bar(
            top_bottlenecks.sort_values("bottleneck_score"),
            x="bottleneck_score", y="camera_name", orientation="h",
            color="bottleneck_score", color_continuous_scale="OrRd",
            labels={"bottleneck_score": "Bottleneck score", "camera_name": "Camera"},
            height=560,
        )
        st.plotly_chart(bn_chart, use_container_width=True, key="bottleneck_chart")

    # -- Route density map --------------------------------------------------
    st.subheader("Route density")
    st.caption(
        "Camera-to-camera hops aggregated across every vehicle's route "
        "(consecutive sightings within a 15-minute gap). Thicker = more "
        "vehicles travelled that hop."
    )
    edges = load_route_edges()
    if edges.empty:
        st.info("Not enough route data to compute edges yet.")
    else:
        top_edges = edges.head(40)
        max_count = top_edges["count"].max()
        route_fig = go.Figure()
        for row in top_edges.itertuples():
            width = 1.5 + 6.5 * (row.count / max_count)
            route_fig.add_trace(go.Scattermap(
                lat=[row.from_lat, row.to_lat], lon=[row.from_lon, row.to_lon],
                mode="lines",
                line=dict(width=width, color="rgba(47,77,140,0.65)"),
                hoverinfo="text",
                text=f"{row.from_id} -> {row.to_id}: {row.count} vehicles",
                showlegend=False,
            ))
        route_fig.add_trace(go.Scattermap(
            lat=summary["camera_lat"], lon=summary["camera_lon"],
            mode="markers",
            marker=dict(size=9, color="#2f4d8c"),
            text=summary["camera_name"], hoverinfo="text",
            showlegend=False,
        ))
        route_fig.update_layout(
            map=dict(center=dict(lat=summary["camera_lat"].mean(), lon=summary["camera_lon"].mean()), zoom=10.5),
            height=560, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(route_fig, use_container_width=True, key="route_density_map")

    # -- Origin-destination patterns -----------------------------------
    st.subheader("Origin-destination patterns")
    st.caption("Top trip-generating camera pairs (first camera -> last camera of each vehicle's latest continuous route).")
    od = load_od_matrix()
    if od.empty:
        st.info("Not enough multi-camera trips to compute OD patterns yet.")
    else:
        od_left, od_right = st.columns([1.3, 1])
        with od_left:
            cameras = pd.unique(od[["origin", "destination"]].values.ravel())
            idx = {c: i for i, c in enumerate(cameras)}
            sankey_fig = go.Figure(go.Sankey(
                node=dict(label=list(cameras), pad=14, thickness=14,
                          color="#2f4d8c"),
                link=dict(
                    source=[idx[o] for o in od["origin"]],
                    target=[idx[d] for d in od["destination"]],
                    value=od["trips"],
                ),
            ))
            sankey_fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(sankey_fig, use_container_width=True, key="od_sankey")
        with od_right:
            st.dataframe(
                od.rename(columns={"origin": "Origin", "destination": "Destination", "trips": "Trips"}),
                use_container_width=True, hide_index=True, height=420,
            )

    # -- Traffic flow trend --------------------------------------------------
    st.subheader("Traffic flow trend")
    st.caption("City-wide detection volume over time (bucketed simulation time, not calendar time).")
    trend = load_flow_trend()
    if trend.empty:
        st.info("No trend data available.")
    else:
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=trend["bucket_start"], y=trend["detections"], mode="lines",
            fill="tozeroy", name="Detections", line=dict(color="#2f4d8c"),
        ))
        trend_fig.update_layout(
            xaxis_title="Simulation time (s)", yaxis_title="Detections per bucket",
            height=340, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(trend_fig, use_container_width=True, key="flow_trend_chart")


def main():
    st.title("🚗 ANPR Trajectory Dashboard")
    st.caption("Local SQLite dashboard for trajectory-model data.")

    try:
        plates = get_plates()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Could not open the SQLite database: {error}")
        return

    if not plates:
        st.warning("No vehicle records exist in the database.")
        return

    tab_trajectory, tab_analytics, tab_alerts, tab_violations, tab_prediction = st.tabs(
        ["Trajectory", "City Analytics", "Alerts", "Violations", "Next-camera prediction"]
    )
    with tab_trajectory:
        render_trajectory_tab(plates)
    with tab_analytics:
        render_city_analytics_tab()
    with tab_alerts:
        render_alerts_tab()
    with tab_violations:
        render_violations_tab()
    with tab_prediction:
        render_prediction_tab()


if __name__ == "__main__":
    main()
