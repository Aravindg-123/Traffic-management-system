import networkx as nx
from collections import defaultdict

def build_graph(hits_df, cam_loc, max_travel_time=600):
    G = nx.DiGraph()

    for cam_id, loc in cam_loc.items():
        G.add_node(cam_id, lon=loc["lon"], lat=loc["lat"],
                   is_hub=1.0 if loc.get("is_hub", False) else 0.0)

    edge_times = defaultdict(list)
    edge_speeds = defaultdict(list)
    edge_counts = defaultdict(int)

    for plate, group in hits_df.groupby("plate"):
        g = group.sort_values("sim_time").reset_index(drop=True)
        for i in range(len(g) - 1):
            u, v = g.loc[i, "camera_id"], g.loc[i + 1, "camera_id"]
            if u == v:
                continue
            dt = g.loc[i + 1, "sim_time"] - g.loc[i, "sim_time"]
            if dt > max_travel_time:
                continue
            edge_counts[(u, v)] += 1
            edge_times[(u, v)].append(dt)
            edge_speeds[(u, v)].append(g.loc[i, "speed_mps"])

    for (u, v), cnt in edge_counts.items():
        avg_time = sum(edge_times[(u, v)]) / len(edge_times[(u, v)])
        avg_speed = sum(edge_speeds[(u, v)]) / len(edge_speeds[(u, v)])
        G.add_edge(u, v, weight=cnt, avg_travel_time=avg_time, avg_speed=avg_speed)

    return G

def graph_to_pyg(G):
    import torch
    from torch_geometric.data import Data
    from predictor import normalize_node_features

    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    x = normalize_node_features(G, nodes)

    edge_index, edge_attr = [], []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_idx[u], node_idx[v]])
        edge_attr.append([data["weight"], data["avg_travel_time"], data["avg_speed"]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_idx, nodes