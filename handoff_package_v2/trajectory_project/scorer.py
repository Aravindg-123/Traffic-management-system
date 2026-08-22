from predictor import predict_next

def evaluate_next_node_accuracy(gnn, head, data, node_idx, G, test_trajs, time_stats, speed_stats, order2_counts):
    top1_correct, top3_correct, total = 0, 0, 0

    for plate, path in test_trajs.items():
        for i in range(len(path) - 1):
            cur_cam = path[i][1]
            speed = path[i][2]
            actual_next = path[i + 1][1]
            prev_cam = path[i - 1][1] if i > 0 else None
            neighbors = list(G.successors(cur_cam))
            if actual_next not in neighbors:
                continue
            pred_top1, probs = predict_next(gnn, head, data, node_idx, G, cur_cam, speed, time_stats, speed_stats, order2_counts, prev_cam)
            ranked = sorted(probs.items(), key=lambda x: -x[1])
            top3 = [c for c, _ in ranked[:3]]
            total += 1
            if pred_top1 == actual_next:
                top1_correct += 1
            if actual_next in top3:
                top3_correct += 1

    return {
        "num_test_transitions": total,
        "top1_accuracy": top1_correct / total if total else 0,
        "top3_accuracy": top3_correct / total if total else 0,
    }

def evaluate_baseline_accuracy(G, test_trajs):
    correct, total = 0, 0
    for plate, path in test_trajs.items():
        for i in range(len(path) - 1):
            cur_cam = path[i][1]
            actual_next = path[i + 1][1]
            neighbors = list(G.successors(cur_cam))
            if actual_next not in neighbors or not neighbors:
                continue
            best = max(neighbors, key=lambda n: G[cur_cam][n]["weight"])
            total += 1
            if best == actual_next:
                correct += 1
    return {"baseline_top1_accuracy": correct / total if total else 0, "total": total}