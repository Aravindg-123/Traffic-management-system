import folium

def plot_predicted_vs_actual(cam_loc, predicted_path, gt_path, out_html="route_comparison.html"):
    center = list(cam_loc.values())[0]
    m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=12)

    for cam_id, loc in cam_loc.items():
        folium.CircleMarker(
            [loc["lat"], loc["lon"]], radius=4, color="gray",
            fill=True, popup=cam_id
        ).add_to(m)

    def draw_path(path, color, label):
        coords = [(cam_loc[c]["lat"], cam_loc[c]["lon"]) for c in path if c in cam_loc]
        if coords:
            folium.PolyLine(coords, color=color, weight=4, opacity=0.8, tooltip=label).add_to(m)

    draw_path(gt_path, "blue", "Ground Truth")
    draw_path(predicted_path, "red", "Predicted")

    m.save(out_html)
    return out_html