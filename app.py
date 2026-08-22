import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path("anpr_demo.db")

st.set_page_config(
    page_title="ANPR Trajectory Dashboard",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run: python import_data.py"
        )

    return sqlite3.connect(DB_PATH, check_same_thread=False)


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


def main():
    st.title("🚗 ANPR Trajectory Dashboard")
    st.caption("Local SQLite dashboard for trajectory-model data.")

    try:
        plates = get_plates()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python import_data.py")
        return
    except Exception as error:
        st.error(f"Could not open the SQLite database: {error}")
        return

    if not plates:
        st.warning("No vehicle records exist in the database.")
        return

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
    df["sim_time"] = pd.to_datetime(df["sim_time"], errors="coerce")

    first_seen = df["sim_time"].min()
    last_seen = df["sim_time"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total events", len(df))
    col2.metric("Cameras visited", df["camera_id"].nunique())
    col3.metric(
        "First seen",
        first_seen.strftime("%d %b, %H:%M")
        if pd.notna(first_seen)
        else "Unknown",
    )
    col4.metric(
        "Last seen",
        last_seen.strftime("%d %b, %H:%M")
        if pd.notna(last_seen)
        else "Unknown",
    )

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
                    "sim_time": "Simulation time",
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
        "Pass this time-ordered DataFrame to your teammate's ML "
        "preprocessing and trajectory-prediction function."
    )


if __name__ == "__main__":
    main()