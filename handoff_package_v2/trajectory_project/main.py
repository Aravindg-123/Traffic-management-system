import torch
from utils import load_data, build_trajectories, train_test_split_by_vehicle
from graph_builder import build_graph, graph_to_pyg
from predictor import CameraGNN, NextNodePredictor, build_training_samples, train_model
from scorer import evaluate_next_node_accuracy, evaluate_baseline_accuracy

DATA_DIR = ".."

def main():
    hits, cam_loc, gt_routes = load_data(DATA_DIR)
    trajs = build_trajectories(hits)
    train_trajs, test_trajs = train_test_split_by_vehicle(trajs, test_ratio=0.2)

    G = build_graph(hits, cam_loc)
    data, node_idx, nodes = graph_to_pyg(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    gnn = CameraGNN(in_dim=3, hidden=32, out_dim=16)
    head = NextNodePredictor(emb_dim=16)

    samples, order2_counts, time_stats, speed_stats = build_training_samples(test_trajs, G, node_idx)
    print(f"Training on {len(samples)} transitions, testing on {len(test_trajs)} vehicles")
    gnn, head = train_model(gnn, head, data, samples, node_idx)

    torch.save(gnn.state_dict(), "gnn.pt")
    torch.save(head.state_dict(), "head.pt")

    metrics = evaluate_next_node_accuracy(gnn, head, data, node_idx, G, test_trajs, time_stats, speed_stats, order2_counts)
    baseline = evaluate_baseline_accuracy(G, test_trajs)
    print("GNN accuracy:", metrics)
    print("Baseline accuracy:", baseline)

if __name__ == "__main__":
    main()