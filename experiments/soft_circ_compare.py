"""Before/after spike-clip comparison on the worst-case body.

2×2 grid:
  col 0 — before clip (outlier spikes visible)
  col 1 — after clip  (clean polygon)
  row 0 — bust cross-section
  row 1 — underbust cross-section

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_compare.py
    # → saves experiments/soft_circ_compare.png
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
from clad_body.measure.anny import load_phenotype_params
from soft_circ_experiment import build_torso_edge_indices, bust_z_differentiable, _apply_coordinate_transform
from soft_circ_improve import soft_circ, _make_model_pack, _forward

DATA = "/home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/questionnaire/data_10k_42/train.json"
OUT  = os.path.join(CLAD_BODY_ROOT, "experiments", "soft_circ_compare.png")

# Algorithm params
N_BINS      = 72
SIGMA_Z     = 0.005
TAU         = 0.050
CLIP_WINDOW = 3
CLIP_THRESH = 0.012   # metres

# Calibrated corrections (clipped)
BUST_A, BUST_B = 0.9936, -0.9925
UB_A,   UB_B   = 0.9758,  0.4157

# Calibrated corrections (no clip, tau=0.05)
BUST_A0, BUST_B0 = 0.9400, 3.4012
UB_A0,   UB_B0   = 0.9636, 0.8008


def _raw_debug(verts, edge_indices, z, clip_window=0, clip_thresh=0.0):
    """Return (raw_cm, px, py, w, r_bin, bin_centers) for plotting."""
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=verts.dtype)
    v  = verts[0]
    va = v[edge_indices[:, 0]];  vb = v[edge_indices[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz = zb - za
    t  = (z - za) / (dz + 1e-10)
    w  = torch.sigmoid(t / TAU) * torch.sigmoid((1.0 - t) / TAU)
    w  = w * torch.sigmoid(torch.abs(dz) / SIGMA_Z - 1.0)
    t_c = t.clamp(0, 1)
    px = va[:, 0] + t_c * (vb[:, 0] - va[:, 0])
    py = va[:, 1] + t_c * (vb[:, 1] - va[:, 1])
    r  = torch.sqrt(px**2 + py**2 + 1e-10)
    theta = torch.atan2(py, px)

    bin_centers = torch.linspace(-np.pi, np.pi*(1-2.0/N_BINS), N_BINS, dtype=verts.dtype)
    sig_th = (2*np.pi/N_BINS) * 0.6
    ang_diff = theta.unsqueeze(-1) - bin_centers.unsqueeze(0)
    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
    ang_aff  = torch.exp(-ang_diff**2 / (2*sig_th**2))
    comb_w   = w.unsqueeze(-1) * ang_aff
    masked_w = comb_w * (w.unsqueeze(-1) > 0.01).float()
    log_w    = torch.log(masked_w + 1e-30) + r.unsqueeze(-1) / TAU
    log_max  = log_w.max(dim=0, keepdim=True).values
    exp_w    = torch.exp(log_w - log_max) * (masked_w > 1e-20).float()
    r_bin    = (r.unsqueeze(-1) * exp_w).sum(0) / (exp_w.sum(0) + 1e-10)

    if clip_window > 0 and clip_thresh > 0.0:
        r_np = r_bin.detach().numpy().copy()
        n = N_BINS
        for i in range(n):
            idxs = [(i - clip_window + j) % n for j in range(2*clip_window+1) if j != clip_window]
            nbr_med = float(np.median(r_np[idxs]))
            if r_np[i] > nbr_med + clip_thresh:
                r_np[i] = nbr_med
        r_bin = torch.tensor(r_np, dtype=verts.dtype)

    pts  = torch.stack([r_bin*torch.cos(bin_centers), r_bin*torch.sin(bin_centers)], dim=-1)
    circ = torch.sum(torch.linalg.norm(torch.roll(pts,-1,0)-pts, dim=-1)).item()*100
    return circ, px.detach().numpy(), py.detach().numpy(), w.detach().numpy(), r_bin.detach().numpy(), bin_centers.numpy()


def draw_cross_section(ax, px, py, w, r_bin, bin_centers, ref_cm, raw_cm, label, color):
    """Draw intersection scatter + soft polygon + convex hull on ax."""
    # Scatter: all active crossing points
    active = w > 0.05
    ax.scatter(px[active]*100, py[active]*100,
               c=w[active], cmap="Greys_r", s=5, alpha=0.4, vmin=0, vmax=1, zorder=1)

    # Convex hull of high-weight points (reference)
    hard = w > 0.3
    pts_hard = np.column_stack([px[hard], py[hard]])
    if len(pts_hard) >= 3:
        try:
            hull = ConvexHull(pts_hard)
            hp   = pts_hard[hull.vertices]
            hp   = np.vstack([hp, hp[0]])
            hull_circ = np.sum(np.linalg.norm(np.diff(hp, axis=0), axis=1))*100
            ax.plot(hp[:,0]*100, hp[:,1]*100, "r-", lw=2, label=f"ConvexHull {hull_circ:.1f} cm", zorder=3)
        except Exception:
            hull_circ = 0.0

    # Soft polygon
    bx = r_bin * np.cos(bin_centers)
    by = r_bin * np.sin(bin_centers)
    poly = np.column_stack([bx, by])
    poly = np.vstack([poly, poly[0]])
    ax.plot(poly[:,0]*100, poly[:,1]*100, color=color, lw=2.5,
            label=f"Soft polygon {raw_cm:.1f} cm", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)", fontsize=9)
    ax.set_ylabel("y (cm)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, alpha=0.3)

    err = raw_cm - ref_cm   # approximate (no linear correction in plot for clarity)
    return err


def find_worst_body(entries, model, torso_edges, breast_idx, pose):
    best_score, best = 0.0, {}
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
            err_b = abs(BUST_A*sb + BUST_B - ref_b)
            err_u = abs(UB_A*su + UB_B - ref_u)
            score = max(err_b, err_u)
            if score > best_score:
                best_score = score
                best = {"verts": verts, "bz": bz.item(), "uz": uz.item(),
                        "ref_b": ref_b, "ref_u": ref_u,
                        "sb": sb, "su": su, "err_b": err_b, "err_u": err_u}
        except Exception:
            pass
    return best


if __name__ == "__main__":
    print("Loading dataset (100 bodies)...")
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)

    print("Setting up model...")
    model, torso_edges, breast_idx, pose = _make_model_pack(entries[0]["params"])

    print("Finding worst-case body (clipped algorithm)...")
    w = find_worst_body(entries, model, torso_edges, breast_idx, pose)
    print(f"  bust err={w['err_b']:.2f} cm  ub err={w['err_u']:.2f} cm")

    verts = w["verts"]
    bz    = torch.tensor(w["bz"])
    uz    = torch.tensor(w["uz"])

    # ── 2×2 figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Spike-clip improvement — worst-case cross-section\n"
        f"(tau={TAU}, n_bins={N_BINS}, clip window={CLIP_WINDOW} bins, thresh={CLIP_THRESH*1000:.0f} mm)",
        fontsize=12, fontweight="bold"
    )

    for row, (z_val, ref_cm, mname) in enumerate([
        (bz, w["ref_b"], "BUST"),
        (uz, w["ref_u"], "UNDERBUST"),
    ]):
        for col, (clip_w, clip_t, label, color) in enumerate([
            (0,           0.0,         "Before clip",  "tab:orange"),
            (CLIP_WINDOW, CLIP_THRESH, "After clip",   "tab:blue"),
        ]):
            raw_cm, px, py, weights, r_bin, bc = _raw_debug(
                verts, torso_edges, z_val, clip_w, clip_t)

            ax = axes[row, col]
            draw_cross_section(ax, px, py, weights, r_bin, bc, ref_cm, raw_cm, label, color)

            ax.set_title(
                f"{mname} — {label}\n"
                f"raw={raw_cm:.1f} cm  ref={ref_cm:.1f} cm  Δ={raw_cm-ref_cm:+.1f} cm",
                fontsize=10
            )

    for row in range(2):
        # Share y-axis limits between before/after for same measurement
        ylims = [axes[row, c].get_ylim() for c in range(2)]
        xlims = [axes[row, c].get_xlim() for c in range(2)]
        ymin = min(y[0] for y in ylims);  ymax = max(y[1] for y in ylims)
        xmin = min(x[0] for x in xlims);  xmax = max(x[1] for x in xlims)
        for c in range(2):
            axes[row, c].set_ylim(ymin, ymax)
            axes[row, c].set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")
