import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from markov_integration.config import DB_PATH, MODEL_DIR
from compare_service import compare, gnn_load_error

GROUND_TRUTH_ROUTES_JSON = MODEL_DIR / "ground_truth_routes.json"
MARKOV_EVAL_JSON = MODEL_DIR / "evaluation_results.json"
TORCH_GNN_METRICS_JSON = MODEL_DIR / "trajectory_project" / "metrics.json"
NUMPY_GNN_METRICS_JSON = MODEL_DIR / "gnn_artifacts" / "metrics.json"
CAMERA_NAMES_JSON = MODEL_DIR / "camera_names.json"

st.set_page_config(
    page_title="Smart Traffic Intelligence — Dual Route Prediction",
    page_icon="🚗",
    layout="wide",
)

COLOR_OBSERVED = "#2563eb"    # blue
COLOR_PROB = "#f59e0b"        # amber
COLOR_GNN = "#8b5cf6"         # purple
COLOR_TRUTH = "#16a34a"       # green
COLOR_WRONG = "#dc2626"       # red


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

@st.cache_resource
def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError("Database not found. Run: python import_data.py")
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data
def get_plates(min_observations: int = 2):
    conn = get_connection()
    query = """
        SELECT plate, COUNT(*) AS n
        FROM events
        GROUP BY plate
        HAVING COUNT(*) >= ?
        ORDER BY plate
        LIMIT 2000
    """
    return pd.read_sql_query(query, conn, params=(min_observations,))["plate"].tolist()


@st.cache_data
def load_events(plate: str):
    conn = get_connection()
    query = """
        SELECT sim_time, camera_id, camera_lon, camera_lat, plate,
               vehicle_lon, vehicle_lat, speed_mps, camera_name
        FROM events WHERE plate = ? ORDER BY sim_time ASC
    """
    return pd.read_sql_query(query, conn, params=(plate,))


@st.cache_data
def load_camera_meta():
    with open(CAMERA_NAMES_JSON, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_markov_metrics():
    if not MARKOV_EVAL_JSON.exists():
        return None
    with open(MARKOV_EVAL_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("metrics")


@st.cache_data
def load_gnn_metrics():
    """Prefers the trained PyTorch Geometric GraphSAGE model's metrics
    (trajectory_project/metrics.json); falls back to the numpy GNN's."""
    path = TORCH_GNN_METRICS_JSON if TORCH_GNN_METRICS_JSON.exists() else NUMPY_GNN_METRICS_JSON
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data["_source"] = "torch" if path == TORCH_GNN_METRICS_JSON else "numpy"
    return data


@st.cache_data
def load_sumo_route(plate: str):
    if not GROUND_TRUTH_ROUTES_JSON.exists():
        return None
    with open(GROUND_TRUTH_ROUTES_JSON, encoding="utf-8") as fh:
        routes = json.load(fh)
    return routes.get(plate)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _coords(cam_ids, cam_meta):
    lats, lons, names = [], [], []
    for c in cam_ids:
        meta = cam_meta.get(c)
        if not meta:
            continue
        lats.append(meta["lat"])
        lons.append(meta["lon"])
        names.append(f"{c} ({meta.get('name', c)})")
    return lats, lons, names


def render_route_map(cam_meta, observed, prob_path, gnn_path, truth_path=None):
    fig = go.Figure()

    def add_trace(cam_ids, color, label, dash=None, width=4):
        if not cam_ids:
            return
        lats, lons, names = _coords(cam_ids, cam_meta)
        if not lats:
            return
        fig.add_trace(
            go.Scattermapbox(
                lat=lats, lon=lons, mode="lines+markers+text",
                text=[c for c in cam_ids if c in cam_meta],
                textposition="top center",
                hovertext=names, hoverinfo="text",
                marker=dict(size=13, color=color),
                line=dict(width=width, color=color),
                name=label,
            )
        )

    add_trace(observed, COLOR_OBSERVED, "Observed prefix")
    add_trace([observed[-1]] + prob_path if observed and prob_path else prob_path, COLOR_PROB, "Part 1: Probabilistic prediction")
    add_trace([observed[-1]] + gnn_path if observed and gnn_path else gnn_path, COLOR_GNN, "Part 2: GNN prediction")
    if truth_path:
        add_trace([observed[-1]] + truth_path if observed else truth_path, COLOR_TRUTH, "Ground-truth future")

    all_cams = observed + prob_path + gnn_path + (truth_path or [])
    lats, lons, _ = _coords(all_cams, cam_meta)
    center = {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)} if lats else {"lat": 13.0, "lon": 80.2}

    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=11, center=center),
        margin=dict(l=0, r=0, t=0, b=0), height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_k_bar(top_k, color, title):
    if not top_k:
        st.info("No prediction available.")
        return
    df = pd.DataFrame(top_k)
    df["probability_pct"] = df["probability"] * 100
    fig = px.bar(
        df, x="camera_id", y="probability_pct", text="probability_pct",
        labels={"camera_id": "Candidate camera", "probability_pct": "Probability (%)"},
        title=title,
    )
    fig.update_traces(marker_color=color, texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_model_card(name, color, result, reveal, ground_truth_future):
    st.markdown(f"### {name}")
    if result is None:
        st.warning("This model is unavailable.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Next camera", result["next_camera"] or "—")
    c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

    render_top_k_bar(result["top_k"], color, "Top-3 candidate probabilities")

    path_str = " → ".join(result["future_path"]) if result["future_path"] else "—"
    st.markdown(f"**Predicted path:** `{path_str}`")
    st.caption(result["explanation"])

    if reveal and ground_truth_future:
        verdict = "✅ Correct" if result["next_correct"] else "❌ Incorrect"
        st.markdown(
            f"**Next-camera verdict:** {verdict}  \n"
            f"**Path match rate:** {result['path_match_rate'] * 100:.0f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("🚗 Smart Traffic Intelligence — Dual Route Prediction")
    st.caption(
        "Part 1 learns historical transition probabilities from observed ANPR sequences. "
        "Part 2 uses the same observations plus graph structure to learn spatially constrained "
        "next-camera probabilities. Both models are tested on vehicles excluded from their "
        "training data. The actual future route is held back until evaluation."
    )

    try:
        plates = get_plates(min_observations=2)
    except FileNotFoundError as error:
        st.error(str(error))
        return

    if not plates:
        st.warning("No vehicles with at least 2 observations exist in the database.")
        return

    with st.sidebar:
        st.header("Vehicle & prediction controls")
        plate = st.selectbox("Select a plate", plates)
        prefix_length = st.slider("Prefix length (observed cameras used)", 1, 6, 2)
        rollout_steps = st.slider("Rollout steps (future cameras to predict)", 1, 3, 2)
        reveal = st.checkbox("🔓 Reveal actual future route", value=False)

        gnn_err = gnn_load_error()
        if gnn_err:
            st.warning(
                "Part 2 GNN artefacts not found — train them first:\n\n"
                "`python -m gnn_integration.train_gnn`"
            )

    result = compare(plate, prefix_length=prefix_length, rollout_steps=rollout_steps)

    if result["status"] != "ok":
        st.error(f"No usable trajectory found for plate {plate}.")
        return

    st.subheader("Observed vehicle history")
    st.markdown(f"**Observed prefix:** `{' → '.join(result['observed_prefix'])}`")

    cam_meta = load_camera_meta()
    truth_for_map = result["ground_truth_future"] if reveal else None
    render_route_map(
        cam_meta,
        result["observed_prefix"],
        (result["probabilistic"] or {}).get("future_path", []),
        (result["gnn"] or {}).get("future_path", []),
        truth_for_map,
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        render_model_card(
            "Part 1: Probabilistic Route Model", COLOR_PROB,
            result["probabilistic"], reveal, result["ground_truth_future"],
        )
    with right:
        gnn_title = (
            "Part 2: GNN Route Model (GraphSAGE)"
            if result["gnn"] and result["gnn"].get("model_order_used") == "gnn_sage"
            else "Part 2: Graph Neural Network Route Model"
        )
        render_model_card(
            gnn_title, COLOR_GNN,
            result["gnn"], reveal, result["ground_truth_future"],
        )

    st.divider()
    st.subheader("Ground-truth reveal & evaluation")
    if not reveal:
        st.info("Enable **Reveal actual future route** in the sidebar to compare against ground truth.")
    else:
        gt = result["ground_truth_future"]
        st.markdown(f"**Actual future (held-out CSV observations):** `{' → '.join(gt) if gt else '—'}`")
        if not gt:
            st.caption(
                "No further camera observations exist for this plate beyond the chosen prefix "
                "(end of trajectory) — nothing to score against."
            )

        sumo_route = load_sumo_route(plate)
        st.markdown("**SUMO simulator ground-truth road route** (informational — road edges, not cameras):")
        st.code(" → ".join(sumo_route) if sumo_route else "No SUMO route recorded for this plate.")
        st.caption(
            "Road network geometry and vehicle routes were generated in SUMO. ANPR-style "
            "observations are generated at camera locations. This road-edge route is shown for "
            "context only — it is not compared directly against predicted camera paths without "
            "a camera-to-edge mapping."
        )

    st.divider()
    st.subheader("Aggregate held-out performance")
    m1, m2 = st.columns(2)

    markov_metrics = load_markov_metrics()
    with m1:
        st.markdown("**Part 1 — Probabilistic (2nd-order Markov)**")
        if markov_metrics:
            mm = markov_metrics.get("2nd_order", {})
            st.write(
                pd.DataFrame(
                    {
                        "Metric": ["Top-1 accuracy", "Top-3 accuracy", "Mean P(actual)"],
                        "Value": [
                            f"{mm.get('top1_accuracy', 0) * 100:.1f}%",
                            f"{mm.get('top3_accuracy', 0) * 100:.1f}%",
                            f"{mm.get('mean_prob_actual', 0):.3f}",
                        ],
                    }
                ).set_index("Metric")
            )
        else:
            st.info("No evaluation_results.json found.")

    gnn_metrics = load_gnn_metrics()
    with m2:
        is_torch = gnn_metrics and gnn_metrics.get("_source") == "torch"
        st.markdown(
            "**Part 2 — GNN (GraphSAGE, PyTorch Geometric)**"
            if is_torch else "**Part 2 — GNN (2-hop message passing + MLP decoder)**"
        )
        if gnn_metrics and gnn_metrics.get("top1_accuracy") is not None:
            if is_torch:
                st.write(
                    pd.DataFrame(
                        {
                            "Metric": ["Top-1 accuracy", "Top-3 accuracy", "Frequency baseline (top-1)"],
                            "Value": [
                                f"{gnn_metrics['top1_accuracy'] * 100:.1f}%",
                                f"{gnn_metrics['top3_accuracy'] * 100:.1f}%",
                                f"{gnn_metrics['baseline_top1_accuracy'] * 100:.1f}%",
                            ],
                        }
                    ).set_index("Metric")
                )
                st.caption(
                    f"graph: {gnn_metrics.get('graph_nodes')} nodes / {gnn_metrics.get('graph_edges')} edges · "
                    f"{gnn_metrics.get('n_train_vehicles')} train / {gnn_metrics.get('n_test_vehicles')} test vehicles"
                )
            else:
                st.write(
                    pd.DataFrame(
                        {
                            "Metric": ["Top-1 accuracy", "Top-3 accuracy", "MRR"],
                            "Value": [
                                f"{gnn_metrics['top1_accuracy'] * 100:.1f}%",
                                f"{gnn_metrics['top3_accuracy'] * 100:.1f}%",
                                f"{gnn_metrics['mrr']:.3f}",
                            ],
                        }
                    ).set_index("Metric")
                )
                st.caption(f"n_train={gnn_metrics.get('n_train')}, n_test={gnn_metrics.get('n_test')}")
        else:
            st.info("GNN not trained yet — run `python -m gnn_integration.train_gnn`.")

    st.caption(
        "Both models are evaluated on held-out vehicles. The future camera sequence is hidden "
        "at prediction time and revealed only for scoring."
    )

    with st.expander("Raw event explorer"):
        df = load_events(plate)
        df["sim_time"] = pd.to_datetime(df["sim_time"], errors="coerce")
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
