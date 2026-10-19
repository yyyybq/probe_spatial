#!/usr/bin/env python3
"""Re-export the V1 paper assets using exact atomic metric associations.

Reuses the existing visualization package and preserves its earlier outputs.
No experiment, canonical promotion, or raw-result modification is performed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import build_visualization_package as base


OUT = base.ROOT / "results/analysis/q00_1096_paper_revision_20260917T124943Z"
base.OUT = OUT
base.FIG = OUT / "figures"
base.TAB = OUT / "tables"
base.DATA = OUT / "data"
base.INV = OUT / "inventory"


def exact_metric_landscape(analysis):
    data = analysis[analysis.metric_value.notna()].copy()
    data["capability"] = [base.capability_for(p, m) for p, m in zip(data.probe, data.metric_name)]
    data = data[data.capability.notna()].copy()
    data["artifact_metric_source"] = data.selected_result_path
    data["source_universe_id"] = data.universe_id
    atomic = pd.read_csv(base.CANONICAL_ALL)
    depth = atomic[atomic.metric_name.eq("mean_loss_depth")].set_index("experiment_id")
    cameras = atomic[atomic.metric_name.eq("Auc_30")].copy()
    attached, evidence = [], []
    parents = analysis[analysis.probe.eq("streaming_depth") & analysis.metric_value.notna()]
    for parent in parents.itertuples(index=False):
        candidates = cameras[(cameras.model == parent.model) &
                             (cameras.layer == parent.layer) & (cameras.prefix == parent.prefix)]
        matches = []
        for camera in candidates.itertuples(index=False):
            depth_id = camera.experiment_id.replace("_camera_", "_depth_", 1)
            if depth_id not in depth.index:
                continue
            sibling = depth.loc[depth_id]
            if isinstance(sibling, pd.DataFrame):
                raise ValueError(f"Ambiguous atomic depth identity: {depth_id}")
            if (sibling.raw_artifact_path == camera.raw_artifact_path and
                    np.isclose(float(sibling.metric_value), parent.metric_value,
                               rtol=1e-9, atol=1e-10)):
                matches.append(camera)
        if len(matches) > 1:
            raise ValueError(f"Ambiguous atomic camera association: {parent.universe_id}")
        evidence.append({"universe_id": parent.universe_id, "model": parent.model,
                         "campaign": parent.campaign, "layer": parent.layer,
                         "prefix": parent.prefix, "parent_depth_value": parent.metric_value,
                         "candidate_count": len(candidates), "exact_match_count": len(matches),
                         "camera_experiment_id": matches[0].experiment_id if matches else "",
                         "camera_value": matches[0].metric_value if matches else np.nan,
                         "atomic_artifact": matches[0].raw_artifact_path if matches else ""})
        if not matches:
            continue
        camera = matches[0]
        rec = parent._asdict()
        rec.update({"probe": "streaming_depth/camera", "metric_name": "Auc_30",
                    "metric_value": float(camera.metric_value),
                    "metric_direction": "higher_is_better", "capability": "A1 Camera",
                    "artifact_metric_source": str(base.CANONICAL_ALL.relative_to(base.ROOT)),
                    "source_universe_id": parent.universe_id,
                    "atomic_experiment_id": camera.experiment_id,
                    "atomic_raw_artifact_path": camera.raw_artifact_path})
        attached.append(rec)
    data = pd.concat([data, pd.DataFrame(attached)], ignore_index=True, sort=False)
    identity = ["model", "campaign", "protocol", "model_variant", "layer", "prefix", "capability"]
    if data.duplicated(identity).any():
        raise ValueError("Multiple numeric records per complete configuration identity")
    data["oriented_value"] = np.where(data.metric_direction.eq("lower_is_better"),
                                      -data.metric_value, data.metric_value)
    data.to_csv(base.DATA / "metric_landscape.csv", index=False)
    pd.DataFrame(evidence).to_csv(base.DATA / "camera_atomic_associations.csv", index=False)
    return data


def geometry_dynamics(rep):
    geom = ["A1 Depth", "A1 Camera", "A2 View"]
    dynamics = ["C1 Dynamics", "C2 Integration", "C3 Counterfactual"]
    rows = []
    for model in base.model_order(rep.model.unique().tolist()):
        z = rep[rep.model.eq(model)]
        g = z[z.capability.isin(geom)]
        d = z[z.capability.isin(dynamics)]
        ng, nd = g.capability.nunique(), d.capability.nunique()
        rows.append({"model": model, "model_display": base.MODEL_META[model][1],
                     "model_family": base.MODEL_META[model][0],
                     "geometry_score": g.relative_score.mean() if ng == 3 else np.nan,
                     "dynamics_score": d.relative_score.mean() if nd == 3 else np.nan,
                     "geometry_components": ng, "dynamics_components": nd,
                     "quality": base.worst_quality(z.quality),
                     "axis_status": "plotted" if ng == nd == 3 else "A2 score unavailable"})
    out = pd.DataFrame(rows)
    plotted = out[out.axis_status.eq("plotted")].copy()
    for axis in ["geometry", "dynamics"]:
        out[axis + "_rank"] = out[axis + "_score"].rank(ascending=False, method="min")
    rho = base.spearman_no_scipy(plotted.geometry_score, plotted.dynamics_score)
    out["spearman_rho"] = rho
    out.to_csv(base.DATA / "fig02_geometry_vs_dynamics_data.csv", index=False)
    fig, (ax, notes) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3.9, 1.1]})
    for r in plotted.itertuples(index=False):
        ax.scatter(r.geometry_score, r.dynamics_score, s=220,
                   color=base.FAMILY_COLORS[r.model_family],
                   marker=base.FAMILY_MARKERS[r.model_family], edgecolor="#222", zorder=3)
        offset = {"qwen2_5_vl": (-120, 12), "vjepa2": (10, -20)}.get(r.model, (8, 9))
        ax.annotate(r.model_display, (r.geometry_score, r.dynamics_score),
                    xytext=offset, textcoords="offset points", fontsize=15)
    ax.set(xlim=(-.05, 1.12), ylim=(-.05, 1.10),
           xlabel="Direct geometry (A1 depth, A1 camera, A2)",
           ylabel="Dynamics understanding (C1–C3 probes)")
    ax.grid(color="#E8E8E8")
    notes.axis("off")
    notes.text(0, .94, "Fixed task composition\n\nGeometry: 3 metrics\nDynamics: 3 metrics\n\n"
               "LingBot-video\nA2 score unavailable;\nno 2D coordinate.\n\n"
               "Scores describe the\nselected diagnostic\nprobes.",
               va="top", fontsize=15, linespacing=1.5)
    title = "Geometry and dynamics understanding have different model orderings"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .95])
    base.save_figure(fig, "fig02_geometry_vs_dynamics")
    return out, {"rho": rho, "title": title, "plotted": plotted}


original_save_figure = base.save_figure


def paper_save_figure(fig, stem, directory=None):
    if stem == "fig04a_prefix_delta_heatmap":
        fig.axes[0].set_title("Longer history has task-dependent effects", loc="left",
                              fontweight="bold", pad=18)
    return original_save_figure(fig, stem, directory)


base.save_figure = paper_save_figure
base.metric_landscape = exact_metric_landscape
base.fig02 = geometry_dynamics

if __name__ == "__main__":
    base.main()
