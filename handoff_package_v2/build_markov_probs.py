import pandas as pd
import numpy as np

# 1. Load transition_counts.csv
counts = pd.read_csv("transition_counts.csv", index_col=0)
cameras = list(counts.index)
n = len(cameras)

# 2. Normalize each row to probabilities; zero-sum rows -> uniform 1/n
row_sums = counts.sum(axis=1)
probs = counts.div(row_sums, axis=0)

zero_sum_rows = row_sums[row_sums == 0].index.tolist()
if zero_sum_rows:
    probs.loc[zero_sum_rows, :] = 1.0 / n

print(f"Dead-end cameras (zero outgoing transitions, smoothed to uniform 1/{n}): {zero_sum_rows}")

# 3. Save transition_probs_1st.csv, 4 decimal places
probs_rounded = probs.round(4)
probs_rounded.to_csv("transition_probs_1st.csv")
print("Saved transition_probs_1st.csv")

# Get is_hub info per camera from raw hits data
hits = pd.read_csv("anpr_hits_final.csv")
hub_lookup = hits.drop_duplicates("camera_id").set_index("camera_id")["is_hub"].to_dict()
hub_cameras = [c for c in cameras if bool(hub_lookup.get(c, False))]
non_hub_cameras = [c for c in cameras if not bool(hub_lookup.get(c, False))]

print(f"\nHub cameras ({len(hub_cameras)}): {hub_cameras}")

# 4a. Top hub cameras CAM_01, CAM_03, CAM_14 - top 3 next cameras
print("\n--- 4a. Top 3 next cameras for hub cameras CAM_01, CAM_03, CAM_14 ---")
for cam in ["CAM_01", "CAM_03", "CAM_14"]:
    if cam not in probs.index:
        print(f"{cam}: not found in matrix")
        continue
    row = probs.loc[cam].sort_values(ascending=False).head(3)
    print(f"{cam} (is_hub={hub_lookup.get(cam)}):")
    for to_cam, p in row.items():
        print(f"    -> {to_cam}: {p:.4f}")

# 4b. 3 random non-hub cameras
print("\n--- 4b. Top 3 next cameras for 3 random non-hub cameras ---")
rng = np.random.default_rng(42)
sample_non_hub = rng.choice(non_hub_cameras, size=min(3, len(non_hub_cameras)), replace=False)
for cam in sample_non_hub:
    row = probs.loc[cam].sort_values(ascending=False).head(3)
    print(f"{cam} (is_hub=False):")
    for to_cam, p in row.items():
        print(f"    -> {to_cam}: {p:.4f}")

# 4c. Cameras with exactly 1 outgoing transition with probability > 50% (nearly deterministic)
print("\n--- 4c. Nearly deterministic cameras (exactly 1 outgoing prob > 50%) ---")
det_count = 0
det_cameras = []
for cam in cameras:
    row = probs.loc[cam]
    above_50 = row[row > 0.5]
    if len(above_50) == 1:
        det_count += 1
        det_cameras.append(cam)
print(f"Count: {det_count}")
print(f"Cameras: {det_cameras}")

# 4d. Cameras with 5+ outgoing transitions each with probability > 5% (high uncertainty)
print("\n--- 4d. High-uncertainty cameras (5+ outgoing transitions each with prob > 5%) ---")
unc_count = 0
unc_cameras = []
for cam in cameras:
    row = probs.loc[cam]
    above_5 = row[row > 0.05]
    if len(above_5) >= 5:
        unc_count += 1
        unc_cameras.append(cam)
print(f"Count: {unc_count}")
print(f"Cameras: {unc_cameras}")

# 4e. Entropy of each hub camera's distribution
print("\n--- 4e. Entropy of each hub camera's distribution ---")
def entropy(p_row):
    p = p_row.values
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

for cam in hub_cameras:
    e = entropy(probs.loc[cam])
    print(f"{cam}: entropy = {e:.4f} bits")

print("\nDone.")
