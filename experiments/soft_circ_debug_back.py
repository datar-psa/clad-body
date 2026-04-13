"""Debug outlier intersection points at the back of the underbust cross-section.

For the p97 body, finds all edge crossings at underbust_z where:
  - the point is in the back half (py > 0)
  - the radius is anomalously large relative to the true body surface

For each such crossing, prints:
  - va, vb positions (3D)
  - edge length and dz
  - top bone for each vertex
  - whether either vertex is above/below the slice plane

Also renders a figure showing:
  left:  underbust cross-section with outlier points highlighted (orange stars)
  right: edge-length histogram of outlier vs normal back edges

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_debug_back.py
"""

import json, os, random, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

CLAD_BODY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLAD_BODY_ROOT)
sys.path.insert(0, os.path.join(CLAD_BODY_ROOT, "experiments"))

from clad_body.load.anny import load_anny_from_params
from clad_body.measure.anny import BREAST_BONES
from soft_circ_experiment import build_torso_edge_indices, bust_z_differentiable, _apply_coordinate_transform
from soft_circ_improve import soft_circ, _make_model_pack, _forward

DATA = "/home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/questionnaire/data_10k_42/train.json"
OUT  = os.path.join(CLAD_BODY_ROOT, "experiments", "soft_circ_debug_back.png")

N_BINS = 72;  SIGMA_Z = 0.005;  TAU = 0.050
CLIP_WINDOW = 3;  CLIP_THRESH = 0.012
BUST_A, BUST_B = 0.9936, -0.9925
UB_A,   UB_B   = 0.9758,  0.4157


def find_p97(entries, model, torso_edges, breast_idx, pose):
    records = []
    for e in entries:
        m = e["measurements"]
        ref_b, ref_u = m.get("bust_cm"), m.get("underbust_cm")
        if ref_b is None or ref_u is None:
            continue
        try:
            body  = load_anny_from_params(e["params"], requires_grad=False)
            verts = _forward(model, body, pose)
            bz    = bust_z_differentiable(model, verts)
            uz    = verts[0, breast_idx, 2].min()
            sb = soft_circ(verts, torso_edges, bz, N_BINS, SIGMA_Z, TAU,
                           clip_window=CLIP_WINDOW, clip_thresh=CLIP_THRESH).item()*100
            su = soft_circ(verts, torso_edges, uz, N_BINS, SIGMA_Z, TAU,
                           clip_window=CLIP_WINDOW, clip_thresh=CLIP_THRESH).item()*100
            score = max(abs(BUST_A*sb+BUST_B-ref_b), abs(UB_A*su+UB_B-ref_u))
            records.append((score, e["params"], verts, bz.item(), uz.item()))
        except Exception:
            pass
    records.sort(key=lambda x: x[0])
    return records[int(len(records)*0.97)]


if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)

    print("Setting up model...")
    model, torso_edges, breast_idx, pose = _make_model_pack(entries[0]["params"])

    # Bone name lookup
    bone_names = []
    if hasattr(model, "bone_names"):
        bone_names = list(model.bone_names)
    elif hasattr(model, "_bone_names"):
        bone_names = list(model._bone_names)

    vbw = model.vertex_bone_weights.detach().cpu().numpy()  # (V, K)
    vbi = model.vertex_bone_indices.detach().cpu().numpy()  # (V, K)

    def top_bone(vi):
        row_w = vbw[vi]
        best  = int(np.argmax(row_w))
        bi    = int(vbi[vi, best])
        bname = bone_names[bi] if bi < len(bone_names) else str(bi)
        return bname, float(row_w[best])

    print("Finding p97 body...")
    _, params, verts, bz_val, uz_val = find_p97(entries, model, torso_edges, breast_idx, pose)
    print(f"  bust_z={bz_val*100:.1f} cm  underbust_z={uz_val*100:.1f} cm")

    # ── Full edge intersection at underbust_z ──────────────────────────────────
    v    = verts[0]
    edges= torso_edges
    va   = v[edges[:, 0]];  vb = v[edges[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz   = zb - za
    z_t  = torch.tensor(uz_val, dtype=verts.dtype)
    t    = (z_t - za) / (dz + 1e-10)

    w_cross = torch.sigmoid(t / TAU) * torch.sigmoid((1.0 - t) / TAU)
    w_dz    = torch.sigmoid(torch.abs(dz) / SIGMA_Z - 1.0)
    w       = (w_cross * w_dz).detach().numpy()

    t_c = t.clamp(0, 1)
    px  = (va[:, 0] + t_c * (vb[:, 0] - va[:, 0])).detach().numpy()
    py  = (va[:, 1] + t_c * (vb[:, 1] - va[:, 1])).detach().numpy()
    r   = np.sqrt(px**2 + py**2)

    va_np = va.detach().numpy()
    vb_np = vb.detach().numpy()

    # Edge lengths
    edge_len = np.linalg.norm(va_np - vb_np, axis=1)
    dz_np    = np.abs(dz.detach().numpy())

    # Active crossings
    active = w > 0.05

    # Build convex hull of high-weight back region for reference radius
    back = active & (py > 0.01)
    hard = w > 0.3

    # Reference: hull of ALL high-weight points
    pts_hard = np.column_stack([px[hard], py[hard]])
    hull_ref_r = None
    if len(pts_hard) >= 3:
        try:
            hull = ConvexHull(pts_hard)
            from matplotlib.path import Path
            hp = pts_hard[hull.vertices]
            hull_path = Path(np.vstack([hp, hp[0]]))
        except Exception:
            hull_path = None
    else:
        hull_path = None

    # Outlier = back point outside the convex hull of all high-weight pts
    if hull_path is not None:
        pts_back = np.column_stack([px[back], py[back]])
        back_indices = np.where(back)[0]
        inside = hull_path.contains_points(pts_back)
        outlier_idx = back_indices[~inside & (w[back] > 0.05)]
    else:
        outlier_idx = np.array([], dtype=int)

    normal_back_idx = np.where(back & (w > 0.05))[0]
    normal_back_idx = normal_back_idx[np.isin(normal_back_idx, outlier_idx, invert=True)]

    print(f"\n  Active back crossings: {back.sum()}")
    print(f"  Outlier back crossings (outside hull): {len(outlier_idx)}")

    # ── Print outlier edge details ─────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'#':>3}  {'va xyz (cm)':>28}  {'vb xyz (cm)':>28}  "
          f"{'len cm':>6}  {'dz cm':>6}  {'w':>5}  {'px,py':>14}  r cm  top_bone_a")
    print(f"{'─'*80}")

    outlier_sorted = sorted(outlier_idx, key=lambda i: -r[i])
    for k, i in enumerate(outlier_sorted[:20]):
        vi_a = int(edges[i, 0]);  vi_b = int(edges[i, 1])
        bna, bwa = top_bone(vi_a)
        bnb, bwb = top_bone(vi_b)
        print(f"{k:>3}  "
              f"({va_np[i,0]*100:5.1f},{va_np[i,1]*100:5.1f},{va_np[i,2]*100:6.1f})  "
              f"({vb_np[i,0]*100:5.1f},{vb_np[i,1]*100:5.1f},{vb_np[i,2]*100:6.1f})  "
              f"{edge_len[i]*100:6.2f}  {dz_np[i]*100:6.2f}  "
              f"{w[i]:5.3f}  "
              f"({px[i]*100:5.1f},{py[i]*100:5.1f})  "
              f"{r[i]*100:5.1f}  "
              f"{bna}({bwa:.2f}) | {bnb}({bwb:.2f})")

    # ── Statistics: edge length distribution ──────────────────────────────────
    print(f"\nOutlier edge lengths (cm): "
          f"mean={edge_len[outlier_idx].mean()*100:.1f}  "
          f"max={edge_len[outlier_idx].max()*100:.1f}  "
          f"min={edge_len[outlier_idx].min()*100:.1f}")
    if len(normal_back_idx):
        print(f"Normal back edge lengths (cm): "
              f"mean={edge_len[normal_back_idx].mean()*100:.1f}  "
              f"max={edge_len[normal_back_idx].max()*100:.1f}  "
              f"min={edge_len[normal_back_idx].min()*100:.1f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Back outlier edge debug — p97 body (underbust z={uz_val*100:.1f} cm)",
                 fontsize=12, fontweight="bold")

    # Panel 1: cross-section with outlier points + their source edges
    ax = axes[0]
    ax.scatter(px[active]*100, py[active]*100,
               c=w[active], cmap="Greys_r", s=5, alpha=0.4, vmin=0, vmax=1, zorder=1)
    if len(pts_hard) >= 3:
        try:
            hull2 = ConvexHull(pts_hard)
            hp2 = pts_hard[hull2.vertices]
            hp2 = np.vstack([hp2, hp2[0]])
            ax.plot(hp2[:,0]*100, hp2[:,1]*100, "r-", lw=2, label="ConvexHull", zorder=3)
        except Exception:
            pass
    # Outlier points
    if len(outlier_idx):
        ax.scatter(px[outlier_idx]*100, py[outlier_idx]*100,
                   c="darkorange", s=60, marker="*", zorder=5, label="Outlier pts")
        # Draw source edges for top outliers
        for i in outlier_sorted[:10]:
            ax.plot([va_np[i,0]*100, vb_np[i,0]*100],
                    [va_np[i,1]*100, vb_np[i,1]*100],
                    "orange", lw=1.2, alpha=0.6, zorder=4)
            ax.plot(va_np[i,0]*100, va_np[i,1]*100, "y^", ms=5, zorder=6)
            ax.plot(vb_np[i,0]*100, vb_np[i,1]*100, "ys", ms=5, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)");  ax.set_ylabel("y (cm)")
    ax.set_title("Cross-section: orange=outlier pts, lines=source edges\n"
                 "triangle=va, square=vb")
    ax.legend(fontsize=8)
    ax.annotate("BACK", xy=(0, 3), ha="center", fontsize=9, color="grey")
    ax.annotate("FRONT", xy=(0, -23), ha="center", fontsize=9, color="grey")

    # Panel 2: source edge endpoints projected to XZ (front view)
    ax2 = axes[1]
    # Plot body vertices as silhouette
    verts_np = verts[0].detach().numpy()
    ax2.scatter(verts_np[:,0]*100, verts_np[:,2]*100,
                s=0.2, c="lightgrey", alpha=0.2, zorder=1)
    ax2.axhline(uz_val*100, color="steelblue", lw=1.5, ls="--",
                alpha=0.7, label=f"z={uz_val*100:.1f} cm")
    for i in outlier_sorted[:10]:
        ax2.plot([va_np[i,0]*100, vb_np[i,0]*100],
                 [va_np[i,2]*100, vb_np[i,2]*100],
                 "orange", lw=2, alpha=0.8, zorder=4)
        ax2.plot(va_np[i,0]*100, va_np[i,2]*100, "y^", ms=7, zorder=5)
        ax2.plot(vb_np[i,0]*100, vb_np[i,2]*100, "ys", ms=7, zorder=5)
    ax2.set_xlabel("x (cm)");  ax2.set_ylabel("z (cm)")
    ax2.set_title("Front view (XZ): outlier source edges\n(orange=edge, △=va, □=vb)")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")

    # Panel 3: edge length distribution: outliers vs normal back
    ax3 = axes[2]
    bins = np.linspace(0, 15, 40)
    if len(normal_back_idx):
        ax3.hist(edge_len[normal_back_idx]*100, bins=bins, alpha=0.6,
                 color="steelblue", label=f"Normal back ({len(normal_back_idx)})")
    if len(outlier_idx):
        ax3.hist(edge_len[outlier_idx]*100, bins=bins, alpha=0.8,
                 color="darkorange", label=f"Outlier back ({len(outlier_idx)})")
    ax3.set_xlabel("Edge length (cm)");  ax3.set_ylabel("Count")
    ax3.set_title("Edge length: outlier vs normal back crossings")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT}")
