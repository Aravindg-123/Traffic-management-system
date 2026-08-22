import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import math
from collections import defaultdict
import numpy as np

class CameraGNN(nn.Module):
    def __init__(self, in_dim=3, hidden=32, out_dim=16):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class NextNodePredictor(nn.Module):
    """Edge features (log_w1, log_w2, avg_time_norm, avg_speed_norm) get a
    dedicated path so they can't be drowned out by GNN embeddings."""
    def __init__(self, emb_dim=16):
        super().__init__()
        self.edge_path = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.embed_path = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, cur_emb, neighbor_embs, speed, edge_feats):
        K = neighbor_embs.size(0)
        cur_rep = cur_emb.unsqueeze(0).expand(K, -1)
        speed_rep = torch.full((K, 1), speed)
        embed_feat = torch.cat([cur_rep, neighbor_embs, speed_rep], dim=1)

        edge_score = self.edge_path(edge_feats).squeeze(-1)
        embed_score = self.embed_path(embed_feat).squeeze(-1)

        scores = edge_score + 0.1 * embed_score
        return F.log_softmax(scores, dim=0)


def normalize_node_features(G, nodes):
    """3 node features: normalized lon, normalized lat, is_hub (already 0/1)."""
    lons = np.array([G.nodes[n]["lon"] for n in nodes])
    lats = np.array([G.nodes[n]["lat"] for n in nodes])
    hubs = np.array([G.nodes[n]["is_hub"] for n in nodes])
    lon_norm = (lons - lons.mean()) / (lons.std() + 1e-8)
    lat_norm = (lats - lats.mean()) / (lats.std() + 1e-8)
    feats = np.stack([lon_norm, lat_norm, hubs], axis=1)
    return torch.tensor(feats, dtype=torch.float)


def _compute_global_stats(G):
    times = [d["avg_travel_time"] for _, _, d in G.edges(data=True)]
    speeds = [d["avg_speed"] for _, _, d in G.edges(data=True)]
    return (np.mean(times), np.std(times)), (np.mean(speeds), np.std(speeds))


def build_order2_counts(trajs_dict):
    counts = defaultdict(lambda: defaultdict(int))
    for plate, path in trajs_dict.items():
        for i in range(1, len(path) - 1):
            prev_cam = path[i - 1][1]
            cur_cam = path[i][1]
            next_cam = path[i + 1][1]
            counts[(prev_cam, cur_cam)][next_cam] += 1
    return counts


def _edge_feats_for(G, cur_cam, neighbors, time_stats, speed_stats, order2_counts, prev_cam):
    feats = []
    order2_for_pair = order2_counts.get((prev_cam, cur_cam), {}) if prev_cam else {}
    for n in neighbors:
        w1 = G[cur_cam][n]["weight"]
        w2 = order2_for_pair.get(n, 0)
        t = G[cur_cam][n]["avg_travel_time"]
        s = G[cur_cam][n]["avg_speed"]
        log_w1 = math.log1p(w1)
        log_w2 = math.log1p(w2)
        t_norm = (t - time_stats[0]) / (time_stats[1] + 1e-8)
        s_norm = (s - speed_stats[0]) / (speed_stats[1] + 1e-8)
        feats.append([log_w1, log_w2, t_norm, s_norm])
    return torch.tensor(feats, dtype=torch.float)


def build_training_samples(train_trajs, G, node_idx):
    time_stats, speed_stats = _compute_global_stats(G)
    order2_counts = build_order2_counts(train_trajs)
    samples = []
    for plate, path in train_trajs.items():
        for i in range(len(path) - 1):
            cur_cam = path[i][1]
            speed = path[i][2]
            next_cam = path[i + 1][1]
            prev_cam = path[i - 1][1] if i > 0 else None
            neighbors = list(G.successors(cur_cam))
            if next_cam not in neighbors or len(neighbors) < 1:
                continue
            edge_feats = _edge_feats_for(G, cur_cam, neighbors, time_stats, speed_stats, order2_counts, prev_cam)
            samples.append((cur_cam, next_cam, speed, neighbors, edge_feats))
    return samples, order2_counts, time_stats, speed_stats


def train_model(gnn, head, data, samples, node_idx, epochs=600, lr=0.01, batch_size=32):
    import random
    opt = torch.optim.Adam(list(gnn.parameters()) + list(head.parameters()), lr=lr)
    for epoch in range(epochs):
        gnn.train(); head.train()
        random.shuffle(samples)
        epoch_loss = 0.0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            opt.zero_grad()
            embeddings = gnn(data.x, data.edge_index)
            batch_loss = 0.0
            for cur_cam, next_cam, speed, neighbors, edge_feats in batch:
                cur_idx = node_idx[cur_cam]
                neigh_idx = [node_idx[n] for n in neighbors]
                target = neighbors.index(next_cam)
                cur_emb = embeddings[cur_idx]
                neigh_embs = embeddings[neigh_idx]
                log_probs = head(cur_emb, neigh_embs, speed / 30.0, edge_feats)
                batch_loss = batch_loss - log_probs[target]
            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            opt.step()
            epoch_loss += batch_loss.item() * len(batch)
        epoch_loss /= len(samples)
        if epoch % 30 == 0:
            print(f"Epoch {epoch}: loss={epoch_loss:.4f}")
    return gnn, head


def predict_next(gnn, head, data, node_idx, G, cur_cam, speed, time_stats, speed_stats, order2_counts, prev_cam=None):
    gnn.eval(); head.eval()
    with torch.no_grad():
        embeddings = gnn(data.x, data.edge_index)
        neighbors = list(G.successors(cur_cam))
        if not neighbors:
            return None, {}
        cur_idx = node_idx[cur_cam]
        neigh_idx = [node_idx[n] for n in neighbors]
        cur_emb = embeddings[cur_idx]
        neigh_embs = embeddings[neigh_idx]
        edge_feats = _edge_feats_for(G, cur_cam, neighbors, time_stats, speed_stats, order2_counts, prev_cam)
        log_probs = head(cur_emb, neigh_embs, speed / 30.0, edge_feats)
        probs = torch.exp(log_probs)
        ranked = sorted(zip(neighbors, probs.tolist()), key=lambda x: -x[1])
        return ranked[0][0], dict(ranked)