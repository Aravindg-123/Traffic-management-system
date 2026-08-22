# Markov Chain vs GNN for ANPR Trajectory Prediction — Analysis Report

## 1. Executive Summary

The Markov model is a purely statistical baseline built by counting how often vehicles moved from camera X to camera Y (1st-order) or from camera-pair (X,Y) to camera Z (2nd-order), then normalizing each row into a probability distribution — no training, no gradient descent, no GPU, just a frequency table built once from the transition data. On a held-out 20% test split (698 records, matrices trained only on the other 80%), the 1st-order model achieved **24.8% Top-1 / 46.8% Top-3 accuracy**, and the 2nd-order model achieved **25.4% Top-1 / 46.6% Top-3 accuracy** with noticeably better calibration (mean probability assigned to the true next camera: 0.1474 vs 0.1222). Given a 30-camera network where many nodes are high-entropy junctions, this is a meaningful floor — it establishes what "no learning at all" gets you. **Recommendation: include the Markov model in the final system, not as a competitor to the GNN, but as a mandatory baseline, fallback, and interpretability layer** — its cost is near-zero and it materially de-risks the GNN's deployment.

## 2. Model Comparison Table

| Aspect | Markov 1st-order | Markov 2nd-order | GNN (typical) |
|--------|-----------------|-----------------|---------------|
| Training required | No | No | Yes (epochs, GPU) |
| Top-1 Accuracy | 24.8% | 25.4% | [FILL IN] |
| Top-3 Accuracy | 46.8% | 46.6% | [FILL IN] |
| Inference speed | Instant (table lookup) | Instant (table lookup) | GPU-dependent |
| Interpretability | High (explicit probabilities) | Medium (conditional probs) | Low (black-box embeddings) |
| Handles sparse data | Yes (smoothing) | No (zero probs for unseen) | Yes (learned embeddings) |
| Captures long-range patterns | No | Limited (2 hops) | Yes (graph propagation) |
| Memory footprint | Tiny (30×30 matrix) | Small (39 states) | Larger (weights + embeddings) |
| Calibration | Good (explicit probs) | Better (0.147 vs 0.122) | Unknown |

## 3. When Markov Is Better Than GNN

1. **Baseline validation** — if the GNN can't beat ~25% Top-1 / ~47% Top-3, something is wrong with its training, features, or data pipeline. The Markov numbers are the sanity floor any learned model must clear to justify its added complexity.
2. **Fallback for out-of-distribution inputs** — a GNN can produce a confident-looking but meaningless output for an unseen graph pattern or malformed input; the Markov table always returns a well-defined probability distribution (falling back to uniform smoothing for truly unseen states), so it degrades gracefully instead of failing silently.
3. **Debugging and interpretability** — every prediction is traceable to an exact frequency: "CAM_11→CAM_02→CAM_22 was seen 100% of the time in training" is auditable in a way that a GNN's embedding-space decision is not, which matters for stakeholder trust and incident review.
4. **Resource-constrained deployment** — the entire 1st-order model is a 30×30 CSV; the 2nd-order model is a small JSON. Both can run on edge hardware, in a spreadsheet, or inside a lightweight API with no GPU and no inference framework.
5. **Cold-start scenarios** — if the GNN needs retraining (new cameras added, schema changes) but the transition table can be rebuilt from raw hits in seconds, the Markov model can serve predictions immediately while the GNN is retrained/redeployed.

## 4. When GNN Is Better Than Markov

1. **Long-range dependencies** — trajectories longer than 2–3 hops. The Markov model's context window is fixed and shrinks its usable data fast (only 511 of 3,490 transitions had 2-camera history at all); a GNN can propagate information across arbitrarily long paths in the road graph.
2. **Sparse 2nd-order states** — with only 39 (prev2,prev1) states observed and 90.85% of the (state, next-camera) combinations at zero count, the 2nd-order Markov table cannot generalize to unseen camera pairs. A GNN's learned embeddings let it interpolate between similar-but-unseen contexts instead of falling back to a coarser model.
3. **Multi-modal / conditional route patterns** — a GNN can learn that route choice depends on interacting factors (which the Markov model cannot condition on) rather than only on the immediately preceding cameras.
4. **Feature-rich inputs** — the Markov model uses only camera identity. A GNN can ingest time-of-day, speed, vehicle type, and road-network topology as node/edge features, which the pure counting approach structurally cannot incorporate without exploding the state space (e.g., a time-of-day-aware Markov model would need separate matrices per time bucket, discussed in Section 7).
5. **Non-linear/contextual patterns** — e.g., "vehicles from CAM_01 to CAM_11 usually go to CAM_10, except during rush hour when they go to CAM_28." The Markov model averages over all such context and reports one static probability; a GNN with the right input features can learn the conditional split.

## 5. Recommendation

### Should Markov be included in final system?
**Answer: Yes, conditionally** — not as a replacement for the GNN, but as a required companion component.

### If Yes, how?
- **As a baseline metric** — report Markov Top-1/Top-3 accuracy alongside every GNN evaluation run, so any accuracy claim is judged against the ~25%/~47% floor rather than in isolation.
- **As a fallback** — if GNN confidence falls below a threshold (or the input pattern is unseen/malformed), fall back to the Markov prediction rather than trusting a low-confidence or undefined GNN output.
- **As an ensemble input** — blend GNN and Markov probabilities (e.g., weighted average, or use Markov as a prior the GNN's output is combined with), which the 2nd-order model's superior calibration (0.1474 vs 0.1222 mean P(actual)) suggests could meaningfully sharpen a blended distribution's calibration even if it doesn't move Top-1 accuracy much on its own.
- **For interpretability/debugging** — surface the raw Markov probabilities in the ops/debugging dashboard next to GNN outputs, so an analyst can immediately see "is this GNN prediction consistent with historical frequency, or is it doing something the data alone wouldn't suggest?"

### If No, why (not applicable here, but noted for completeness)
A "no" recommendation would only make sense if the GNN's accuracy advantage were overwhelming (e.g., >2x Markov's Top-1) *and* the team had no need for interpretability, fallback safety, or a baseline — none of which holds for an ANPR system used in an operational/investigative context, where explainability and graceful degradation carry real weight beyond raw accuracy.

## 6. Appendix: Key Statistics

### Top 10 most predictable transitions (highest Markov confidence)
Only 2 transitions exceed the 60% confidence bar; the rest of the top 10 are the next-highest observed probabilities in the network.

| From → To | Probability | Count |
|---|---|---|
| CAM_02 → CAM_22 | 0.6296 | 102 |
| CAM_17 → CAM_30 | 0.6296 | 34 |
| CAM_10 → CAM_30 | 0.5853 | 175 |
| CAM_11 → CAM_02 | 0.4627 | 93 |
| CAM_21 → CAM_11 | 0.3684 | 21 |
| CAM_06 → CAM_11 | 0.3548 | 22 |
| CAM_11 → CAM_27 | 0.3184 | 64 |
| CAM_09 → CAM_11 | 0.2909 | 16 |
| CAM_01 → CAM_11 | 0.2701 | 74 |
| CAM_08 → CAM_11 | 0.2632 | 15 |

### Top 10 least predictable transitions (Markov confidence <15%)
559 of the 595 observed (from,to) pairs fall below 15% confidence; the 10 lowest (all single-occurrence, tied near the smoothing floor) are:

| From → To | Probability | Count |
|---|---|---|
| CAM_10 → CAM_09 | 0.0033 | 1 |
| CAM_10 → CAM_02 | 0.0033 | 1 |
| CAM_10 → CAM_21 | 0.0033 | 1 |
| CAM_10 → CAM_05 | 0.0033 | 1 |
| CAM_10 → CAM_03 | 0.0033 | 1 |
| CAM_10 → CAM_18 | 0.0033 | 1 |
| CAM_01 → CAM_02 | 0.0036 | 1 |
| CAM_01 → CAM_13 | 0.0036 | 1 |
| CAM_01 → CAM_18 | 0.0036 | 1 |
| CAM_01 → CAM_19 | 0.0036 | 1 |

### Hub camera entropy values (from Task 2)
| Hub Camera | Entropy (bits) |
|---|---|
| CAM_01 | 3.6124 |
| CAM_30 | 3.8703 |
| CAM_03 | 3.8825 |
| CAM_20 | 4.0967 |
| CAM_14 | 4.1938 |
| CAM_22 | 4.2037 |

All 6 hub cameras sit in the 3.6–4.2 bit range (max possible = log2(30) ≈ 4.91 bits), confirming hubs are high-uncertainty junctions with many plausible next cameras rather than predictable pass-throughs — consistent with hub-camera Top-1 accuracy (20.0%) trailing non-hub accuracy (~28%) in the Task 4 evaluation.

### Data sparsity stats
- 1st-order: 595 unique observed (from,to) pairs out of 900 possible (30×30) — 66.1% coverage / 33.9% zero.
- 2nd-order: 39 unique (prev2,prev1) states observed; of the 1,170 possible (state, next-camera) combinations (39 states × 30 cameras), only 107 are nonzero — **90.85% sparsity**.
- Only 511 of 3,490 total transitions (14.6%) had a 2-camera history available at all, because median trajectory length per plate is short (median 2 hits/plate), limiting how much data the 2nd-order model can ever learn from without longer observed trajectories.

## 7. Next Steps (Optional Future Work)

- **3rd-order Markov model** — if longer trajectories become available (more hits per plate, longer observation windows), extend to a 3-camera history; expect diminishing returns without more data given the sparsity already seen at 2nd-order.
- **Time-weighted transitions** — build separate transition matrices for rush hour vs off-peak (or finer time buckets), directly addressing the "non-linear pattern" limitation noted in Section 4 without requiring a full GNN.
- **Hybrid Markov + GNN ensemble** — formalize the blending approach from Section 5 (e.g., a learned mixing weight, or using Markov probabilities as an additional GNN input feature) and evaluate whether it beats either model alone on Top-1/Top-3 accuracy and calibration.
