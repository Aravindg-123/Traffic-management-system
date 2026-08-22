import json
import random
from collections import defaultdict

import pandas as pd
import numpy as np

RANDOM_SEED = 42
CAMERAS_COUNT = 30

# 1. Load all_transitions.json
with open("all_transitions.json", "r") as f:
    transitions = json.load(f)

# Hub lookup for breakdown (from raw hits data)
hits = pd.read_csv("anpr_hits_final.csv")
hub_lookup = hits.drop_duplicates("camera_id").set_index("camera_id")["is_hub"].to_dict()
all_cameras = sorted(hits["camera_id"].unique())

# 2. Reconstruct full trajectories per plate (same as Task 3)
plate_transitions = defaultdict(list)
for t in transitions:
    plate_transitions[t["plate"]].append(t)

trajectories = {}
for plate, trans_list in plate_transitions.items():
    seq = [trans_list[0]["from"]]
    for t in trans_list:
        seq.append(t["to"])
    trajectories[plate] = seq

# 3. Extract ALL 2-step sequences with full context (prev2 may be null for the
# first hop of a trajectory)
records = []
for plate, seq in trajectories.items():
    for i in range(len(seq) - 1):
        prev1 = seq[i]
        actual = seq[i + 1]
        prev2 = seq[i - 1] if i - 1 >= 0 else None
        records.append({"plate": plate, "prev2": prev2, "prev1": prev1, "actual": actual})

print(f"Total records (1st-order transitions, some with prev2 context): {len(records)}")
with_prev2 = sum(1 for r in records if r["prev2"] is not None)
print(f"Records with prev2 context available: {with_prev2}")

# 4. Split: 80% train, 20% test (random shuffle, then split by index)
random.seed(RANDOM_SEED)
shuffled = records[:]
random.shuffle(shuffled)
split_idx = int(0.8 * len(shuffled))
train_records = shuffled[:split_idx]
test_records = shuffled[split_idx:]
print(f"Train: {len(train_records)}, Test: {len(test_records)}")

# 5. Build 1st-order probability matrix from TRAIN transitions only
counts_1st = pd.DataFrame(0, index=all_cameras, columns=all_cameras)
for r in train_records:
    counts_1st.loc[r["prev1"], r["actual"]] += 1

row_sums_1st = counts_1st.sum(axis=1)
probs_1st = counts_1st.div(row_sums_1st, axis=0)
zero_rows_1st = row_sums_1st[row_sums_1st == 0].index.tolist()
if zero_rows_1st:
    probs_1st.loc[zero_rows_1st, :] = 1.0 / CAMERAS_COUNT

# Build 2nd-order probability matrix from TRAIN records with prev2 context only
counts_2nd = defaultdict(lambda: defaultdict(int))
for r in train_records:
    if r["prev2"] is None:
        continue
    state = f"{r['prev2']}_{r['prev1']}"
    counts_2nd[state][r["actual"]] += 1

probs_2nd = {}
for state, next_counts in counts_2nd.items():
    total = sum(next_counts.values())
    probs_2nd[state] = {cam: c / total for cam, c in next_counts.items()}

print(f"Train-derived 1st-order matrix: {probs_1st.shape}")
print(f"Train-derived 2nd-order states: {len(probs_2nd)}")


def predict_1st(current_camera, top_k=3):
    row = probs_1st.loc[current_camera].sort_values(ascending=False).head(top_k)
    return list(zip(row.index, row.values))


def predict_2nd(prev_camera, current_camera, top_k=3):
    state = f"{prev_camera}_{current_camera}"
    if state not in probs_2nd:
        return predict_1st(current_camera, top_k)
    next_probs = probs_2nd[state]
    top = sorted(next_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return top


def prob_for(preds, camera):
    for cam, p in preds:
        if cam == camera:
            return p
    return 0.0


# 6. Evaluate on TEST set
eval_rows = []
for r in test_records:
    plate, prev2, prev1, actual = r["plate"], r["prev2"], r["prev1"], r["actual"]

    preds_1st = predict_1st(prev1, top_k=3)
    top1_1st = preds_1st[0][0] if preds_1st else None
    p_actual_1st = prob_for(preds_1st, actual)
    in_top3_1st = actual in [c for c, _ in preds_1st]
    correct_1st = (top1_1st == actual)

    if prev2 is not None:
        preds_2nd = predict_2nd(prev2, prev1, top_k=3)
    else:
        preds_2nd = preds_1st
    top1_2nd = preds_2nd[0][0] if preds_2nd else None
    p_actual_2nd = prob_for(preds_2nd, actual)
    in_top3_2nd = actual in [c for c, _ in preds_2nd]
    correct_2nd = (top1_2nd == actual)

    eval_rows.append({
        "plate": plate,
        "prev2": prev2,
        "prev1": prev1,
        "actual": actual,
        "pred1_1st": top1_1st,
        "pred1_2nd": top1_2nd,
        "p_actual_1st": p_actual_1st,
        "p_actual_2nd": p_actual_2nd,
        "correct_1st": correct_1st,
        "correct_2nd": correct_2nd,
        "in_top3_1st": in_top3_1st,
        "in_top3_2nd": in_top3_2nd,
        "prev1_is_hub": bool(hub_lookup.get(prev1, False)),
    })

eval_df = pd.DataFrame(eval_rows)

# 7. Compute metrics
def compute_metrics(df, correct_col, top3_col, p_col):
    top1_acc = df[correct_col].mean()
    top3_acc = df[top3_col].mean()
    mean_p = df[p_col].mean()
    return top1_acc, top3_acc, mean_p

top1_acc_1st, top3_acc_1st, mean_p_1st = compute_metrics(eval_df, "correct_1st", "in_top3_1st", "p_actual_1st")
top1_acc_2nd, top3_acc_2nd, mean_p_2nd = compute_metrics(eval_df, "correct_2nd", "in_top3_2nd", "p_actual_2nd")

# Hub vs non-hub breakdown
hub_df = eval_df[eval_df["prev1_is_hub"]]
non_hub_df = eval_df[~eval_df["prev1_is_hub"]]

hub_acc_1st = hub_df["correct_1st"].mean() if len(hub_df) else float("nan")
non_hub_acc_1st = non_hub_df["correct_1st"].mean() if len(non_hub_df) else float("nan")
hub_acc_2nd = hub_df["correct_2nd"].mean() if len(hub_df) else float("nan")
non_hub_acc_2nd = non_hub_df["correct_2nd"].mean() if len(non_hub_df) else float("nan")

# 8. Print results table
print("\n" + "=" * 62)
print(f"{'Model':<14}{'Top-1 Acc':<14}{'Top-3 Acc':<14}{'Mean P(actual)':<18}")
print("-" * 62)
print(f"{'1st-order':<14}{top1_acc_1st*100:>7.1f}%     {top3_acc_1st*100:>7.1f}%     {mean_p_1st:>10.4f}")
print(f"{'2nd-order':<14}{top1_acc_2nd*100:>7.1f}%     {top3_acc_2nd*100:>7.1f}%     {mean_p_2nd:>10.4f}")
print("=" * 62)

print(f"\nHub vs Non-hub breakdown (accuracy when prev1 camera is hub vs non-hub):")
print(f"  1st-order: hub={hub_acc_1st*100:.1f}% (n={len(hub_df)}), non-hub={non_hub_acc_1st*100:.1f}% (n={len(non_hub_df)})")
print(f"  2nd-order: hub={hub_acc_2nd*100:.1f}% (n={len(hub_df)}), non-hub={non_hub_acc_2nd*100:.1f}% (n={len(non_hub_df)})")

# 9. Save evaluation_results.json
results = {
    "n_train": len(train_records),
    "n_test": len(test_records),
    "random_seed": RANDOM_SEED,
    "metrics": {
        "1st_order": {
            "top1_accuracy": top1_acc_1st,
            "top3_accuracy": top3_acc_1st,
            "mean_prob_actual": mean_p_1st,
            "hub_accuracy": hub_acc_1st,
            "non_hub_accuracy": non_hub_acc_1st,
        },
        "2nd_order": {
            "top1_accuracy": top1_acc_2nd,
            "top3_accuracy": top3_acc_2nd,
            "mean_prob_actual": mean_p_2nd,
            "hub_accuracy": hub_acc_2nd,
            "non_hub_accuracy": non_hub_acc_2nd,
        },
    },
    "per_transition_results": eval_rows,
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))

with open("evaluation_results.json") as f:
    json.load(f)
print("\nSaved and verified evaluation_results.json")

# 10. Save markov_evaluation.csv
csv_cols = ["plate", "prev2", "prev1", "actual", "pred1_1st", "pred1_2nd",
            "p_actual_1st", "p_actual_2nd", "correct_1st", "correct_2nd"]
eval_df[csv_cols].to_csv("markov_evaluation.csv", index=False)
print("Saved markov_evaluation.csv")

print("\nDone.")
