"""
gnn_integration.gnn_model
==========================
A compact, dependency-free ("no torch/PyG available") graph neural network
for next-camera prediction, implemented directly in numpy.

Architecture (Simplified-GCN style)
------------------------------------
1. Fixed 2-hop mean "message passing" operator ``S`` (built once from the
   train-only directed camera graph, see graph_builder.build_smoothing_operator).
2. Trainable node projection:
       CX  = concat(X, S @ X)                    (N x 2F)
       Emb = ReLU(CX @ W1 + b1)                   (N x H1)
3. Trainable decoder MLP scores each *legal* outgoing neighbour of the
   current camera:
       feat_k = concat(Emb[curr], Emb[prev]_or_zeros, Emb[k], speed, dt)
       logit_k = Dense2(ReLU(Dense1(feat_k)))
4. A directed legal-transition mask (train-graph out-neighbours of `curr`)
   is applied before softmax, so the model can never predict an
   observed-impossible transition.

Training is full-batch gradient descent (the graph has 30 nodes and a few
thousand examples, so this converges in well under a second).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class SimpleGNN:
    def __init__(
        self,
        camera_list: List[str],
        X: np.ndarray,
        S: np.ndarray,
        hidden1: int = 16,
        decoder_hidden: int = 32,
        seed: int = 42,
    ):
        self.camera_list = camera_list
        self.camera_index = {c: i for i, c in enumerate(camera_list)}
        self.X = X
        self.S = S
        self.n_cameras, self.n_feat = X.shape
        self.h1 = hidden1
        self.hd = decoder_hidden

        rng = np.random.default_rng(seed)
        in1 = 2 * self.n_feat
        self.W1 = rng.normal(0, np.sqrt(2.0 / in1), size=(in1, hidden1))
        self.b1 = np.zeros(hidden1)

        dec_in = 3 * hidden1 + 2  # curr, prev, candidate embeddings + speed + dt
        self.Wd1 = rng.normal(0, np.sqrt(2.0 / dec_in), size=(dec_in, decoder_hidden))
        self.bd1 = np.zeros(decoder_hidden)
        self.Wd2 = rng.normal(0, np.sqrt(2.0 / decoder_hidden), size=(decoder_hidden, 1))
        self.bd2 = np.zeros(1)

        self._CX = np.concatenate([self.X, self.S @ self.X], axis=1)  # (N, 2F), fixed

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (Pre1, Emb) so callers/backprop can reuse the pre-activation."""
        pre1 = self._CX @ self.W1 + self.b1
        emb = _relu(pre1)
        return pre1, emb

    def _decoder_forward(
        self, feat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """feat: (n_cand, dec_in) -> logits: (n_cand,)."""
        pre2 = feat @ self.Wd1 + self.bd1
        a1 = _relu(pre2)
        logits = (a1 @ self.Wd2 + self.bd2).reshape(-1)
        return pre2, a1, logits

    def score_candidates(
        self,
        curr: str,
        prev: Optional[str],
        candidates: List[str],
        speed_norm: float,
        dt_norm: float,
    ) -> np.ndarray:
        """Returns softmax probabilities over `candidates` (already legal-masked
        by the caller choosing which cameras to pass in)."""
        _, emb = self.embeddings()
        feat = self._build_features(emb, curr, prev, candidates, speed_norm, dt_norm)
        _, _, logits = self._decoder_forward(feat)
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def _build_features(
        self,
        emb: np.ndarray,
        curr: str,
        prev: Optional[str],
        candidates: List[str],
        speed_norm: float,
        dt_norm: float,
    ) -> np.ndarray:
        curr_vec = emb[self.camera_index[curr]]
        prev_vec = (
            emb[self.camera_index[prev]] if prev is not None and prev in self.camera_index
            else np.zeros(self.h1)
        )
        rows = []
        for cand in candidates:
            cand_vec = emb[self.camera_index[cand]]
            rows.append(
                np.concatenate(
                    [curr_vec, prev_vec, cand_vec, [speed_norm], [dt_norm]]
                )
            )
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------
    # Training (full-batch manual backprop)
    # ------------------------------------------------------------------

    def _prepare_batch(self, examples, train_graph, speed_lookup):
        """Pads every example's candidate set to the batch's max width so
        the whole epoch can run as a handful of matmuls instead of a
        python-level loop (this graph only has 30 nodes / ~3k examples,
        but ragged per-example loops were still the training bottleneck)."""
        rows = []
        for ex in examples:
            curr, prev, actual = ex["prev1"], ex["prev2"], ex["actual"]
            candidates = sorted(train_graph.get(curr, {}).keys())
            if actual not in candidates or curr not in self.camera_index:
                continue
            rows.append(
                (
                    self.camera_index[curr],
                    self.camera_index[prev] if prev is not None and prev in self.camera_index else -1,
                    [self.camera_index[c] for c in candidates],
                    candidates.index(actual),
                    speed_lookup.get(curr, 0.5),
                )
            )
        if not rows:
            raise ValueError("No trainable examples: check train_graph / camera_index alignment.")

        M = len(rows)
        max_c = max(len(r[2]) for r in rows)

        curr_idx = np.array([r[0] for r in rows], dtype=int)
        prev_idx = np.array([r[1] for r in rows], dtype=int)
        actual_pos = np.array([r[3] for r in rows], dtype=int)
        speed_arr = np.array([r[4] for r in rows], dtype=float)

        cand_idx = np.zeros((M, max_c), dtype=int)
        cand_mask = np.zeros((M, max_c), dtype=bool)
        for i, r in enumerate(rows):
            c = r[2]
            cand_idx[i, : len(c)] = c
            cand_mask[i, : len(c)] = True

        return {
            "M": M, "max_c": max_c,
            "curr_idx": curr_idx, "prev_idx": prev_idx,
            "cand_idx": cand_idx, "cand_mask": cand_mask,
            "actual_pos": actual_pos, "speed_arr": speed_arr,
        }

    def train(
        self,
        examples: List[Dict],
        train_graph: Dict[str, Dict[str, int]],
        epochs: int = 400,
        lr: float = 0.05,
        l2: float = 1e-4,
        speed_lookup: Optional[Dict[str, float]] = None,
        verbose_every: int = 50,
    ) -> List[float]:
        """`examples`: list of {prev2, prev1, actual} dicts (train split only).
        `train_graph`: prev1 -> {actual: count}, defines the legal candidate
        set (and therefore also the masked softmax) for every example.
        """
        speed_lookup = speed_lookup or {}
        batch = self._prepare_batch(examples, train_graph, speed_lookup)
        M, max_c = batch["M"], batch["max_c"]
        curr_idx, prev_idx = batch["curr_idx"], batch["prev_idx"]
        cand_idx, cand_mask = batch["cand_idx"], batch["cand_mask"]
        actual_pos, speed_arr = batch["actual_pos"], batch["speed_arr"]
        prev_valid = (prev_idx >= 0)
        prev_idx_safe = np.where(prev_valid, prev_idx, 0)
        dt_arr = np.full(M, 0.5)
        H1 = self.h1

        losses = []
        for epoch in range(epochs):
            pre1, emb = self.embeddings()  # (N, H1)

            curr_vec = emb[curr_idx]                          # (M, H1)
            prev_vec = emb[prev_idx_safe] * prev_valid[:, None]  # (M, H1)
            cand_vec = emb[cand_idx]                           # (M, max_c, H1)

            curr_b = np.broadcast_to(curr_vec[:, None, :], (M, max_c, H1))
            prev_b = np.broadcast_to(prev_vec[:, None, :], (M, max_c, H1))
            speed_b = np.broadcast_to(speed_arr[:, None, None], (M, max_c, 1))
            dt_b = np.broadcast_to(dt_arr[:, None, None], (M, max_c, 1))

            feat = np.concatenate([curr_b, prev_b, cand_vec, speed_b, dt_b], axis=2)
            feat2d = feat.reshape(M * max_c, -1)

            pre2 = feat2d @ self.Wd1 + self.bd1
            a1 = _relu(pre2)
            logits2d = (a1 @ self.Wd2 + self.bd2).reshape(M, max_c)
            logits2d = np.where(cand_mask, logits2d, -1e9)

            shifted = logits2d - logits2d.max(axis=1, keepdims=True)
            exp = np.exp(shifted) * cand_mask
            probs = exp / exp.sum(axis=1, keepdims=True)

            loss = -np.log(probs[np.arange(M), actual_pos] + 1e-12).mean()
            losses.append(loss)
            if verbose_every and epoch % verbose_every == 0:
                print(f"  epoch {epoch:4d}  loss={loss:.4f}")

            dlogits2d = probs.copy()
            dlogits2d[np.arange(M), actual_pos] -= 1.0
            dlogits2d = dlogits2d * cand_mask / M

            dlogits = dlogits2d.reshape(M * max_c, 1)
            dWd2 = a1.T @ dlogits
            dbd2 = dlogits.sum(axis=0)

            dA1 = dlogits @ self.Wd2.T
            dPre2 = dA1 * (pre2 > 0)

            dWd1 = feat2d.T @ dPre2
            dbd1 = dPre2.sum(axis=0)

            dFeat2d = dPre2 @ self.Wd1.T
            dFeat = dFeat2d.reshape(M, max_c, -1)

            dCurr = dFeat[:, :, :H1].sum(axis=1)                # (M, H1)
            dPrev = dFeat[:, :, H1 : 2 * H1].sum(axis=1) * prev_valid[:, None]
            dCand = dFeat[:, :, 2 * H1 : 3 * H1]                # (M, max_c, H1)

            dEmb = np.zeros_like(emb)
            np.add.at(dEmb, curr_idx, dCurr)
            np.add.at(dEmb, prev_idx_safe, dPrev)
            np.add.at(dEmb, cand_idx.reshape(-1), dCand.reshape(-1, H1))

            dPre1 = dEmb * (pre1 > 0)
            dW1 = self._CX.T @ dPre1 + l2 * self.W1
            db1 = dPre1.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.Wd1 -= lr * (dWd1 + l2 * self.Wd1)
            self.bd1 -= lr * dbd1
            self.Wd2 -= lr * (dWd2 + l2 * self.Wd2)
            self.bd2 -= lr * dbd2

        return losses

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, path) -> None:
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            Wd1=self.Wd1, bd1=self.bd1,
            Wd2=self.Wd2, bd2=self.bd2,
        )

    def load_weights(self, path) -> None:
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.Wd1, self.bd1 = data["Wd1"], data["bd1"]
        self.Wd2, self.bd2 = data["Wd2"], data["bd2"]
