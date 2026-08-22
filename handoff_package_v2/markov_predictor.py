import json
import random
from collections import defaultdict

import pandas as pd


def predict_next_camera(current_camera, top_k=3):
    """
    Given current camera, return top-k most likely next cameras with probabilities.
    Returns: list of tuples [(camera_id, probability), ...]
    """
    probs = pd.read_csv('handoff_package_v2/transition_probs_1st.csv', index_col=0)
    row = probs.loc[current_camera]
    top = row.sort_values(ascending=False).head(top_k)
    return list(zip(top.index, top.values))


def predict_next_camera_2nd(prev_camera, current_camera, top_k=3):
    """
    Given last 2 cameras, return top-k most likely next cameras.
    Returns: list of tuples [(camera_id, probability), ...]
    If (prev, curr) state not found, fall back to 1st-order prediction for current_camera.
    """
    with open('handoff_package_v2/transition_probs_2nd.json', 'r') as f:
        probs_2nd = json.load(f)

    state = f"{prev_camera}_{current_camera}"
    if state not in probs_2nd:
        return predict_next_camera(current_camera, top_k)

    next_probs = probs_2nd[state]
    top = sorted(next_probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(cam, prob) for cam, prob in top]


def _reconstruct_trajectories():
    with open('handoff_package_v2/all_transitions.json', 'r') as f:
        transitions = json.load(f)

    plate_transitions = defaultdict(list)
    for t in transitions:
        plate_transitions[t["plate"]].append(t)

    trajectories = {}
    for plate, trans_list in plate_transitions.items():
        seq = [trans_list[0]["from"]]
        for t in trans_list:
            seq.append(t["to"])
        trajectories[plate] = seq
    return trajectories


def _prob_for(preds, camera):
    for cam, p in preds:
        if cam == camera:
            return p
    return 0.0


if __name__ == "__main__":
    # Test on 3 hub cameras and 3 non-hub cameras
    test_cameras = ["CAM_01", "CAM_03", "CAM_14", "CAM_05", "CAM_12", "CAM_25"]
    for cam in test_cameras:
        preds = predict_next_camera(cam, top_k=3)
        print(f"{cam}: {preds}")

    # Task 3, step 8: 10-example 1st-order vs 2nd-order comparison
    print("\n--- 1st-order vs 2nd-order comparison (10 random 2-step examples) ---")
    trajectories = _reconstruct_trajectories()

    triples = []
    for plate, seq in trajectories.items():
        for i in range(len(seq) - 2):
            triples.append((seq[i], seq[i + 1], seq[i + 2], plate))

    random.seed(42)
    sample = random.sample(triples, min(10, len(triples)))

    more_confident_2nd = 0
    print(f"{'prev2':8} {'prev1':8} {'actual':8} {'1st pred (top1)':22} {'2nd pred (top1)':22} {'more confident'}")
    for prev2, prev1, actual, plate in sample:
        preds_1st = predict_next_camera(prev1, top_k=3)
        preds_2nd = predict_next_camera_2nd(prev2, prev1, top_k=3)

        p1_correct = _prob_for(preds_1st, actual)
        p2_correct = _prob_for(preds_2nd, actual)

        winner = "2nd-order" if p2_correct > p1_correct else ("1st-order" if p1_correct > p2_correct else "tie")
        if p2_correct > p1_correct:
            more_confident_2nd += 1

        top1_1st = preds_1st[0] if preds_1st else None
        top1_2nd = preds_2nd[0] if preds_2nd else None
        print(f"{prev2:8} {prev1:8} {actual:8} "
              f"{str(top1_1st):22} {str(top1_2nd):22} "
              f"1st_p(actual)={p1_correct:.4f} 2nd_p(actual)={p2_correct:.4f} -> {winner}")

    print(f"\n2nd-order was more confident on the correct next camera in "
          f"{more_confident_2nd}/{len(sample)} examples")
