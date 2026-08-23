import pandas as pd
import json

def load_data(data_dir):
    hits = pd.read_csv(f"{data_dir}/anpr_hits_final.csv")
    with open(f"{data_dir}/camera_locations.json") as f:
        cam_loc = json.load(f)
    with open(f"{data_dir}/camera_names.json") as f:
        cam_names = json.load(f)
    with open(f"{data_dir}/ground_truth_routes.json") as f:
        gt_routes = json.load(f)

    # merge is_hub into cam_loc so graph_builder has everything in one dict
    for cam_id, info in cam_loc.items():
        info["is_hub"] = cam_names.get(cam_id, {}).get("is_hub", False)

    return hits, cam_loc, gt_routes

def build_trajectories(hits_df):
    trajs = {}
    for plate, group in hits_df.groupby("plate"):
        g = group.sort_values("sim_time")
        trajs[plate] = list(zip(g["sim_time"], g["camera_id"], g["speed_mps"]))
    return trajs

def train_test_split_by_vehicle(trajs, test_ratio=0.2, seed=42):
    import random
    plates = list(trajs.keys())
    random.Random(seed).shuffle(plates)
    n_test = int(len(plates) * test_ratio)
    test_plates = set(plates[:n_test])
    train = {p: t for p, t in trajs.items() if p not in test_plates}
    test = {p: t for p, t in trajs.items() if p in test_plates}
    return train, test