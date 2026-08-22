import json
import random
from collections import defaultdict, OrderedDict

CAMERAS_COUNT = 30

# 1. Load all_transitions.json
with open("all_transitions.json", "r") as f:
    transitions = json.load(f)

print(f"Loaded {len(transitions)} transitions")

# 2. Reconstruct full trajectories per plate.
# The transitions list was generated in chronological order per plate (grouped by
# plate, sorted by sim_time, in original extraction order), so grouping while
# preserving list order reconstructs each plate's trajectory correctly - no
# absolute timestamp is stored in the json, only time_delta between hops, so
# we infer order from the existing sequence rather than re-sorting by time_delta.
plate_transitions = defaultdict(list)
for t in transitions:
    plate_transitions[t["plate"]].append(t)

trajectories = {}
for plate, trans_list in plate_transitions.items():
    seq = [trans_list[0]["from"]]
    for t in trans_list:
        seq.append(t["to"])
    trajectories[plate] = seq

print(f"Reconstructed trajectories for {len(trajectories)} plates")

# 3. Extract 2-step sequences: (cam_{t-2}, cam_{t-1}) -> cam_t
counts_2nd = defaultdict(lambda: defaultdict(int))
total_2step = 0
for plate, seq in trajectories.items():
    for i in range(len(seq) - 2):
        prev2, prev1, nxt = seq[i], seq[i + 1], seq[i + 2]
        state = f"{prev2}_{prev1}"
        counts_2nd[state][nxt] += 1
        total_2step += 1

print(f"Total 2-step sequences extracted: {total_2step}")
print(f"Unique 2-step states: {len(counts_2nd)}")

# 4. Save transition_counts_2nd.json
counts_2nd_plain = {state: dict(next_counts) for state, next_counts in counts_2nd.items()}
with open("transition_counts_2nd.json", "w") as f:
    json.dump(counts_2nd_plain, f, indent=2)
print("Saved transition_counts_2nd.json")

# 5. Normalize to probabilities -> transition_probs_2nd.json
probs_2nd = {}
for state, next_counts in counts_2nd_plain.items():
    total = sum(next_counts.values())
    probs_2nd[state] = {cam: round(c / total, 4) for cam, c in next_counts.items()}

with open("transition_probs_2nd.json", "w") as f:
    json.dump(probs_2nd, f, indent=2)
print("Saved transition_probs_2nd.json")

# Verify JSON validity
with open("transition_counts_2nd.json") as f:
    json.load(f)
with open("transition_probs_2nd.json") as f:
    json.load(f)
print("JSON validity verified for both files")

# 6. Statistics
print("\n--- Statistics ---")

# a. Number of unique (prev_2) states
num_states = len(counts_2nd_plain)
print(f"a. Unique (prev_2) states: {num_states}")

# b. Average number of possible next cameras per state
avg_next = sum(len(v) for v in counts_2nd_plain.values()) / num_states
print(f"b. Average number of possible next cameras per state: {avg_next:.4f}")

# c. Sparsity
possible_combos = num_states * CAMERAS_COUNT
nonzero_combos = sum(len(v) for v in counts_2nd_plain.values())
sparsity_pct = 100.0 * (1 - nonzero_combos / possible_combos)
print(f"c. Possible (state, next) combinations: {possible_combos}")
print(f"   Nonzero combinations: {nonzero_combos}")
print(f"   Sparsity: {sparsity_pct:.2f}% are zero")

# d. States with only 1 outgoing transition (deterministic given 2-camera history)
deterministic_states = [s for s, v in counts_2nd_plain.items() if len(v) == 1]
print(f"d. States with exactly 1 outgoing next camera: {len(deterministic_states)} "
      f"({100 * len(deterministic_states) / num_states:.2f}% of states)")

# e. Coverage comparison
print(f"e. 1st-order unique (from,to) pairs: 595")
print(f"   2nd-order unique (prev_2) states: {num_states}")
print(f"   2nd-order captures {num_states} distinct 2-camera-history contexts, "
      f"vs 595 single-camera contexts for 1st-order.")

print("\nDone.")
