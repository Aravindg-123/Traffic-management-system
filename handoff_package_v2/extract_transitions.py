import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

CSV_PATH = "anpr_hits_final.csv"
START_TIME = datetime(2026, 1, 15, 8, 0, 0)

# 1. Load
df = pd.read_csv(CSV_PATH)

# Handle missing values: drop rows missing essential fields
essential_cols = ["sim_time", "camera_id", "plate"]
before = len(df)
df = df.dropna(subset=essential_cols)
after = len(df)
if before != after:
    print(f"Dropped {before - after} rows with missing essential values (sim_time/camera_id/plate)")

# 2. Convert sim_time to datetime
df["sim_time"] = pd.to_numeric(df["sim_time"], errors="coerce")
df = df.dropna(subset=["sim_time"])
df["datetime"] = df["sim_time"].apply(lambda s: START_TIME + timedelta(seconds=float(s)))

# 3. Sort by plate, then sim_time
df = df.sort_values(["plate", "sim_time"]).reset_index(drop=True)

# is_hub lookup per camera_id (assume consistent per camera)
hub_lookup = df.drop_duplicates("camera_id").set_index("camera_id")["is_hub"].to_dict()

# 4 & 5. Extract ordered camera sequences per plate, skipping consecutive repeats, and transitions
all_transitions = []

for plate, group in df.groupby("plate", sort=False):
    group = group.sort_values("sim_time")
    records = group[["camera_id", "sim_time"]].to_records(index=False)

    prev_cam = None
    prev_time = None
    for cam, t in records:
        if prev_cam is None:
            prev_cam, prev_time = cam, t
            continue
        if cam == prev_cam:
            # consecutive same-camera repeat: skip (do not update prev, per "skip repeats")
            continue
        time_delta = float(t) - float(prev_time)
        all_transitions.append({
            "from": prev_cam,
            "to": cam,
            "plate": plate,
            "time_delta": time_delta
        })
        prev_cam, prev_time = cam, t

print(f"Total transitions extracted: {len(all_transitions)}")

trans_df = pd.DataFrame(all_transitions)

# Unique (from, to) pairs
unique_pairs = trans_df[["from", "to"]].drop_duplicates()
print(f"Unique (from, to) pairs: {len(unique_pairs)}")

# Top 10 most frequent transitions
top10 = (
    trans_df.groupby(["from", "to"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
)
print("\nTop 10 most frequent transitions:")
for _, row in top10.iterrows():
    print(f"  {row['from']} -> {row['to']}: {row['count']}")

# Hub vs non-hub transition breakdown
def is_hub_transition(row):
    from_hub = bool(hub_lookup.get(row["from"], False))
    to_hub = bool(hub_lookup.get(row["to"], False))
    return from_hub or to_hub

trans_df["involves_hub"] = trans_df.apply(is_hub_transition, axis=1)
hub_count = int(trans_df["involves_hub"].sum())
non_hub_count = int((~trans_df["involves_hub"]).sum())
print(f"\nHub vs non-hub transition breakdown:")
print(f"  Transitions involving at least one hub camera: {hub_count}")
print(f"  Transitions with no hub camera involved: {non_hub_count}")

# 7. Save all_transitions.json
json_records = [
    {
        "from": t["from"],
        "to": t["to"],
        "plate": t["plate"],
        "time_delta": int(round(t["time_delta"]))
    }
    for t in all_transitions
]

with open("all_transitions.json", "w") as f:
    json.dump(json_records, f, indent=2)

# Verify JSON validity
with open("all_transitions.json", "r") as f:
    loaded = json.load(f)
assert len(loaded) == len(json_records), "JSON record count mismatch!"
print(f"\nJSON validity verified: all_transitions.json contains {len(loaded)} records")

# 8. Create transition_counts.csv (30x30 matrix)
all_cameras = sorted(df["camera_id"].unique())
print(f"\nNumber of unique cameras: {len(all_cameras)}")

count_matrix = pd.DataFrame(0, index=all_cameras, columns=all_cameras)
pair_counts = trans_df.groupby(["from", "to"]).size()
for (f, t), c in pair_counts.items():
    count_matrix.loc[f, t] = c

count_matrix.to_csv("transition_counts.csv")
print("Saved transition_counts.csv")

print("\nDone.")
