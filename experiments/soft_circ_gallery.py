"""Gallery: 6 bodies × 2 rows (bust/underbust) cross-section overlay.

Shows soft polygon (blue) vs convex hull (red) after spike clip, on 6 bodies
sampled to cover the full error range (p0, p20, p40, p60, p80, p100 of UB error).

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_gallery.py
    # → saves experiments/soft_circ_gallery.png
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
from soft_circ_experiment import bust_z_differentiable
from soft_circ_improve import soft_circ, _make_model_pack, _forward

DATA = "/home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/questionnaire/data_10k_42/train.json"
OUT  = os.path.join(CLAD_BODY_ROOT, "experiments", "soft_circ_gallery.png")

N_BINS      = 72
SIGMA_Z     = 0.005
TAU         = 0.050
CLIP_WINDOW = 0
CLIP_THRESH = 0.0
RECENTER    = True

BUST_A, BUST_B = 0.9700, 2.16
UB_A,   UB_B   = 0.9872, 1.14


def cross_section(verts, edge_indices, z):
    """Return (px, py, w, r_bin, bin_centers, raw_cm) with recenter."""
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

    if RECENTER:
        w_sum = w.sum() + 1e-10
        cx = (w * px).sum() / w_sum
        cy = (w * py).sum() / w_sum
        cx, cy = cx.detach(), cy.detach()
    else:
        cx = cy = torch.tensor(0.0, dtype=verts.dtype)

    dx, dy = px - cx, py - cy
    r  = torch.sqrt(dx**2 + dy**2 + 1e-10)
    theta = torch.atan2(dy, dx)

    bin_centers = torch.linspace(-np.pi, np.pi*(1-2/N_BINS), N_BINS, dtype=verts.dtype)
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

    if CLIP_WINDOW > 0 and CLIP_THRESH > 0.0:
        r_np = r_bin.detach().numpy().copy()
        n = N_BINS
        for i in range(n):
            idxs = [(i - CLIP_WINDOW + j) % n for j in range(2*CLIP_WINDOW+1) if j != CLIP_WINDOW]
            nbr_med = float(np.median(r_np[idxs]))
            if r_np[i] > nbr_med + CLIP_THRESH:
                r_np[i] = nbr_med
        r_bin = torch.tensor(r_np, dtype=verts.dtype)

    # Convert back to absolute coordinates for the polygon
    pts  = torch.stack([cx + r_bin*torch.cos(bin_centers),
                        cy + r_bin*torch.sin(bin_centers)], dim=-1)
    raw_cm = torch.sum(torch.linalg.norm(torch.roll(pts,-1,0)-pts, dim=-1)).item()*100
    return (px.detach().numpy(), py.detach().numpy(), w.detach().numpy(),
            r_bin.detach().numpy(), bin_centers.numpy(), raw_cm,
            float(cx), float(cy))


def draw(ax, px, py, w, r_bin, bc, ref_cm, raw_cm, corr_cm, title, cx=0.0, cy=0.0):
    active = w > 0.05
    ax.scatter(px[active]*100, py[active]*100,
               c=w[active], cmap="Greys_r", s=4, alpha=0.35, vmin=0, vmax=1, zorder=1)

    hard = w > 0.3
    pts_h = np.column_stack([px[hard], py[hard]])
    if len(pts_h) >= 3:
        try:
            hull = ConvexHull(pts_h)
            hp = np.vstack([pts_h[hull.vertices], pts_h[hull.vertices[0]]])
            ax.plot(hp[:,0]*100, hp[:,1]*100, "r-", lw=1.8, label="Hull (ref)", zorder=3)
        except Exception:
            pass

    bx = cx + r_bin * np.cos(bc);  by = cy + r_bin * np.sin(bc)
    poly = np.vstack([np.column_stack([bx, by]), [bx[0], by[0]]])
    err  = corr_cm - ref_cm
    ax.plot(poly[:,0]*100, poly[:,1]*100, "b-", lw=2,
            label=f"Soft  corr={corr_cm:.1f}", zorder=4)

    ax.set_aspect("equal")
    ax.set_title(f"{title}\nref={ref_cm:.1f}  corr={corr_cm:.1f}  err={err:+.1f} cm",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="k", lw=0.3, alpha=0.3)
    ax.axvline(0, color="k", lw=0.3, alpha=0.3)


if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)

    print("Setting up model...")
    model, torso_edges, breast_idx, pose = _make_model_pack(entries[0]["params"])

    print("Evaluating all 100 bodies...")
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

            bpx, bpy, bw, br, bbc, b_raw, bcx, bcy = cross_section(verts, torso_edges, bz)
            upx, upy, uw, ur, ubc, u_raw, ucx, ucy = cross_section(verts, torso_edges, uz)

            b_corr = BUST_A*b_raw + BUST_B
            u_corr = UB_A*u_raw + UB_B

            records.append({
                "ref_b": ref_b, "ref_u": ref_u,
                "b_raw": b_raw, "u_raw": u_raw,
                "b_corr": b_corr, "u_corr": u_corr,
                "b_err": abs(b_corr - ref_b), "u_err": abs(u_corr - ref_u),
                "score": max(abs(b_corr-ref_b), abs(u_corr-ref_u)),
                "verts": verts, "bz": bz.item(), "uz": uz.item(),
                "bpx": bpx, "bpy": bpy, "bw": bw, "br": br, "bbc": bbc,
                "bcx": bcx, "bcy": bcy,
                "upx": upx, "upy": upy, "uw": uw, "ur": ur, "ubc": ubc,
                "ucx": ucx, "ucy": ucy,
            })
        except Exception:
            pass

    records.sort(key=lambda r: r["score"])
    n = len(records)
    # Pick 6 bodies spanning p5 → p95 of combined error
    picks_idx = [int(n * q) for q in [0.05, 0.25, 0.45, 0.65, 0.82, 0.97]]
    picks_idx = [min(p, n-1) for p in picks_idx]
    picks = [records[i] for i in picks_idx]
    labels = ["p5 (good)", "p25", "p45", "p65", "p82", "p97 (worst)"]

    print(f"  Selected {len(picks)} bodies. Score range: "
          f"{records[0]['score']:.2f}–{records[-1]['score']:.2f} cm")

    # ── 2 rows (bust, underbust) × 6 cols ────────────────────────────────────
    fig, axes = plt.subplots(2, 6, figsize=(22, 8))
    rc_label = "recenter" if RECENTER else f"clip {CLIP_WINDOW}×{CLIP_THRESH*1000:.0f}mm"
    fig.suptitle(
        f"Gallery: 6 bodies across error range  "
        f"(tau={TAU}, {rc_label})\n"
        f"Blue = soft polygon (corrected)  |  Red = convex hull (reference)",
        fontsize=11, fontweight="bold"
    )

    for col, (rec, lbl) in enumerate(zip(picks, labels)):
        draw(axes[0, col],
             rec["bpx"], rec["bpy"], rec["bw"], rec["br"], rec["bbc"],
             rec["ref_b"], rec["b_raw"], rec["b_corr"],
             f"BUST — {lbl}", rec.get("bcx", 0), rec.get("bcy", 0))
        draw(axes[1, col],
             rec["upx"], rec["upy"], rec["uw"], rec["ur"], rec["ubc"],
             rec["ref_u"], rec["u_raw"], rec["u_corr"],
             f"UB — {lbl}", rec.get("ucx", 0), rec.get("ucy", 0))

    # Uniform axis limits per row
    for row in range(2):
        xlims = [axes[row, c].get_xlim() for c in range(6)]
        ylims = [axes[row, c].get_ylim() for c in range(6)]
        xmin = min(x[0] for x in xlims);  xmax = max(x[1] for x in xlims)
        ymin = min(y[0] for y in ylims);  ymax = max(y[1] for y in ylims)
        for c in range(6):
            axes[row, c].set_xlim(xmin, xmax)
            axes[row, c].set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")
