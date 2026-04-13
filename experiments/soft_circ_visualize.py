"""Visualise a worst-case soft-circumference cross-section.

Finds the body with the largest corrected error (>p95) on bust or underbust,
then plots:
  • Raw intersection points (grey dots, sized by crossing weight)
  • Soft-polygon from angular binning (blue)
  • Hard convex-hull of the same points (red dashed) — reference method
  • Angular bin mass bar chart (bottom) — where coverage is thin

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_visualize.py
    # → saves experiments/soft_circ_worst_case.png
"""

import json, os, random, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull

CLAD_BODY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLAD_BODY_ROOT)
sys.path.insert(0, os.path.join(CLAD_BODY_ROOT, "experiments"))

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure import measure
from clad_body.measure.anny import BREAST_BONES
from soft_circ_experiment import (
    build_torso_edge_indices, bust_z_differentiable, _apply_coordinate_transform,
)
from soft_circ_improve import soft_circ, _make_model_pack, _forward

DATA = "/home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/questionnaire/data_10k_42/train.json"
OUT  = os.path.join(CLAD_BODY_ROOT, "experiments", "soft_circ_worst_case.png")

TAU         = 0.050
SIGMA_Z     = 0.005
N_BINS      = 72
CLIP_WINDOW = 3       # rolling-median spike clip: half-window in bins
CLIP_THRESH = 0.012   # absolute threshold in metres above neighbor median

# Calibrated on 100-body dataset with clip=3×12mm, tau=0.050
BUST_A, BUST_B = 0.9936, -0.9925
UB_A,   UB_B   = 0.9758,  0.4157


# ── Instrumented soft_circ — returns all intermediate tensors ─────────────────

def soft_circ_debug(verts, edge_indices, z, n_bins=72, sigma_z=0.005, tau=0.050,
                    clip_window=CLIP_WINDOW, clip_thresh=CLIP_THRESH):
    """Same as soft_circ but also returns debug arrays for visualisation."""
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=verts.dtype, device=verts.device)

    v   = verts[0]
    va  = v[edge_indices[:, 0]]
    vb  = v[edge_indices[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz  = zb - za
    t   = (z - za) / (dz + 1e-10)

    w_cross  = torch.sigmoid(t / tau) * torch.sigmoid((1.0 - t) / tau)
    w_dz     = torch.sigmoid(torch.abs(dz) / sigma_z - 1.0)
    w        = w_cross * w_dz

    t_c = t.clamp(0, 1)
    px  = va[:, 0] + t_c * (vb[:, 0] - va[:, 0])
    py  = va[:, 1] + t_c * (vb[:, 1] - va[:, 1])
    r   = torch.sqrt(px**2 + py**2 + 1e-10)
    theta = torch.atan2(py, px)

    bin_centers = torch.linspace(-np.pi, np.pi*(1-2.0/n_bins), n_bins,
                                 dtype=verts.dtype)
    bin_width   = 2*np.pi / n_bins
    sig_th      = bin_width * 0.6

    ang_diff = theta.unsqueeze(-1) - bin_centers.unsqueeze(0)
    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
    ang_aff  = torch.exp(-ang_diff**2 / (2*sig_th**2))

    comb_w   = w.unsqueeze(-1) * ang_aff
    masked_w = comb_w * (w.unsqueeze(-1) > 0.01).float()
    log_w    = torch.log(masked_w + 1e-30) + r.unsqueeze(-1) / tau
    log_max  = log_w.max(dim=0, keepdim=True).values
    exp_w    = torch.exp(log_w - log_max) * (masked_w > 1e-20).float()
    r_bin    = (r.unsqueeze(-1) * exp_w).sum(0) / (exp_w.sum(0) + 1e-10)
    bin_mass = comb_w.sum(0)   # total angular+crossing weight per bin

    # Spike clip: one-sided rolling-median, downward only
    if clip_window > 0 and clip_thresh > 0.0:
        r_np = r_bin.detach().cpu().numpy().copy()
        ww, n = clip_window, n_bins
        for i in range(n):
            idxs = [(i - ww + j) % n for j in range(2 * ww + 1) if j != ww]
            nbr_med = float(np.median(r_np[idxs]))
            if r_np[i] > nbr_med + clip_thresh:
                r_np[i] = nbr_med
        r_bin = torch.tensor(r_np, dtype=verts.dtype)

    pts  = torch.stack([r_bin*torch.cos(bin_centers),
                        r_bin*torch.sin(bin_centers)], dim=-1)
    circ = torch.sum(torch.linalg.norm(torch.roll(pts,-1,0)-pts, dim=-1))

    return (
        circ.item()*100,
        px.detach().numpy(),   py.detach().numpy(),
        w.detach().numpy(),
        r_bin.detach().numpy(),
        bin_centers.numpy(),
        bin_mass.detach().numpy(),
    )


# ── Find worst-case body ───────────────────────────────────────────────────────

def find_worst(entries, model, torso_edges, breast_idx, pose):
    """Find the body with the largest combined (max of bust, underbust) corrected error."""
    worst = {"err_bust": 0, "err_ub": 0, "bust_raw": 0, "ub_raw": 0,
             "ref_bust": 0, "ref_ub": 0, "bust_z": 0, "ub_z": 0, "verts": None}
    best_score = 0.0

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
                worst = {
                    "err_bust": err_b, "err_ub": err_u,
                    "bust_raw": sb, "ub_raw": su,
                    "ref_bust": ref_b, "ref_ub": ref_u,
                    "bust_z": bz.item(), "ub_z": uz.item(),
                    "verts": verts,
                }
        except Exception:
            pass
    return worst


# ── Plot a single cross-section ────────────────────────────────────────────────

def plot_cross_section(ax_main, ax_mass, ax_r,
                       px, py, w,
                       r_bin, bin_centers, bin_mass,
                       ref_cm, raw_cm, corr_cm, z_m, title):
    """Main cross-section panel + angular mass + radius comparison."""

    # ── Mask: only edges with meaningful crossing weight ──
    active = w > 0.05
    px_a, py_a, w_a = px[active], py[active], w[active]

    # ── Convex hull of high-weight points (hard reference) ──
    hard = w > 0.3
    pts_hard = np.column_stack([px[hard], py[hard]])
    if len(pts_hard) >= 3:
        try:
            hull = ConvexHull(pts_hard)
            hull_pts = pts_hard[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax_main.plot(hull_pts[:,0]*100, hull_pts[:,1]*100,
                         "r--", lw=1.5, label="ConvexHull (hard w>0.3)", zorder=3)
            hull_circ = np.sum(np.linalg.norm(np.diff(hull_pts, axis=0), axis=1))*100
        except Exception:
            hull_circ = 0
    else:
        hull_circ = 0

    # ── Scatter: all active intersection points ──
    sc = ax_main.scatter(px_a*100, py_a*100, c=w_a, cmap="viridis",
                         s=6, alpha=0.6, vmin=0, vmax=1, zorder=2)
    plt.colorbar(sc, ax=ax_main, label="crossing weight w", shrink=0.8)

    # ── Soft polygon ──
    bx = r_bin * np.cos(bin_centers)
    by = r_bin * np.sin(bin_centers)
    poly_pts = np.column_stack([bx, by])
    poly_pts = np.vstack([poly_pts, poly_pts[0]])
    ax_main.plot(poly_pts[:,0]*100, poly_pts[:,1]*100,
                 "b-", lw=2, label="Soft polygon (72 bins)", zorder=4)

    ax_main.set_aspect("equal")
    ax_main.set_xlabel("x (cm)")
    ax_main.set_ylabel("y (cm)")
    ax_main.set_title(
        f"{title}  |  Z={z_m*100:.1f} cm\n"
        f"ref={ref_cm:.1f}  raw={raw_cm:.1f}  corr={corr_cm:.1f}  "
        f"err={corr_cm-ref_cm:+.1f} cm"
    )
    ax_main.legend(fontsize=8)
    ax_main.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_main.axvline(0, color="k", lw=0.5, alpha=0.3)

    # ── Angular mass bar chart ──
    deg = np.degrees(bin_centers)
    ax_mass.bar(deg, bin_mass, width=360/N_BINS*0.8, color="steelblue", alpha=0.7)
    ax_mass.axhline(0.5, color="r", lw=1, ls="--", label="w=0.5 threshold")
    ax_mass.set_xlabel("Bin angle (°)")
    ax_mass.set_ylabel("Bin mass")
    ax_mass.set_title("Angular coverage (bin mass)")
    ax_mass.legend(fontsize=8)
    ax_mass.set_xlim(-185, 185)

    # ── Radius per bin: soft vs reference hull ──
    ax_r.plot(deg, r_bin*100, "b-o", ms=3, label="Soft r_bin (cm)")
    ax_r.set_xlabel("Bin angle (°)")
    ax_r.set_ylabel("Radius (cm)")
    ax_r.set_title("Radius per angular bin")
    ax_r.legend(fontsize=8)
    ax_r.set_xlim(-185, 185)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)

    print("Setting up model...")
    model, torso_edges, breast_idx, pose = _make_model_pack(entries[0]["params"])
    print(f"  Torso edges: {len(torso_edges)}")

    print("Finding worst-case body...")
    worst = find_worst(entries, model, torso_edges, breast_idx, pose)
    print(f"  Worst bust error:      {worst['err_bust']:.2f} cm  "
          f"(raw={worst['bust_raw']:.1f} corr={BUST_A*worst['bust_raw']+BUST_B:.1f} ref={worst['ref_bust']:.1f})")
    print(f"  Worst underbust error: {worst['err_ub']:.2f} cm  "
          f"(raw={worst['ub_raw']:.1f} corr={UB_A*worst['ub_raw']+UB_B:.1f} ref={worst['ref_ub']:.1f})")

    verts = worst["verts"]
    bz    = torch.tensor(worst["bust_z"])
    uz    = torch.tensor(worst["ub_z"])

    # Debug pass for both heights
    (b_circ, b_px, b_py, b_w, b_rbin, b_bc, b_mass) = soft_circ_debug(
        verts, torso_edges, bz, N_BINS, SIGMA_Z, TAU)
    (u_circ, u_px, u_py, u_w, u_rbin, u_bc, u_mass) = soft_circ_debug(
        verts, torso_edges, uz, N_BINS, SIGMA_Z, TAU)

    print(f"\n  Bust   raw={b_circ:.1f}  corr={BUST_A*b_circ+BUST_B:.1f}  ref={worst['ref_bust']:.1f}")
    print(f"  Underbust raw={u_circ:.1f}  corr={UB_A*u_circ+UB_B:.1f}  ref={worst['ref_ub']:.1f}")

    # ── Figure: 2 measurements × 3 panels ──────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Worst-case cross-section visualisation  (tau=0.050, n_bins=72)",
                 fontsize=13, fontweight="bold")

    plot_cross_section(
        axes[0,0], axes[0,1], axes[0,2],
        b_px, b_py, b_w, b_rbin, b_bc, b_mass,
        ref_cm=worst["ref_bust"],
        raw_cm=b_circ,
        corr_cm=BUST_A*b_circ + BUST_B,
        z_m=worst["bust_z"],
        title="BUST",
    )
    plot_cross_section(
        axes[1,0], axes[1,1], axes[1,2],
        u_px, u_py, u_w, u_rbin, u_bc, u_mass,
        ref_cm=worst["ref_ub"],
        raw_cm=u_circ,
        corr_cm=UB_A*u_circ + UB_B,
        z_m=worst["ub_z"],
        title="UNDERBUST",
    )

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT}")
