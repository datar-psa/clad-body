"""Visualise soft-polygon cross-section on a body close to the dataset mean.

Picks the body whose (bust_cm, underbust_cm) is nearest to the dataset mean,
then draws bust + underbust cross-sections (after spike clip).

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_avg.py
    # → saves experiments/soft_circ_avg.png
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
OUT  = os.path.join(CLAD_BODY_ROOT, "experiments", "soft_circ_avg.png")

N_BINS = 72;  SIGMA_Z = 0.005;  TAU = 0.050
CLIP_WINDOW = 3;  CLIP_THRESH = 0.012
BUST_A, BUST_B = 0.9936, -0.9925
UB_A,   UB_B   = 0.9758,  0.4157


def cross_section(verts, edges, z):
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=verts.dtype)
    v  = verts[0]
    va = v[edges[:, 0]];  vb = v[edges[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz = zb - za
    t  = (z - za) / (dz + 1e-10)
    w  = torch.sigmoid(t / TAU) * torch.sigmoid((1 - t) / TAU)
    w  = w * torch.sigmoid(torch.abs(dz) / SIGMA_Z - 1.0)
    t_c = t.clamp(0, 1)
    px = va[:, 0] + t_c * (vb[:, 0] - va[:, 0])
    py = va[:, 1] + t_c * (vb[:, 1] - va[:, 1])
    r  = torch.sqrt(px**2 + py**2 + 1e-10)
    theta = torch.atan2(py, px)

    bc = torch.linspace(-np.pi, np.pi*(1-2/N_BINS), N_BINS, dtype=verts.dtype)
    sig_th = (2*np.pi/N_BINS) * 0.6
    ad = torch.atan2(torch.sin(theta.unsqueeze(-1) - bc.unsqueeze(0)),
                     torch.cos(theta.unsqueeze(-1) - bc.unsqueeze(0)))
    cw = w.unsqueeze(-1) * torch.exp(-ad**2 / (2*sig_th**2))
    mw = cw * (w.unsqueeze(-1) > 0.01).float()
    lw = torch.log(mw + 1e-30) + r.unsqueeze(-1) / TAU
    ew = torch.exp(lw - lw.max(0, keepdim=True).values) * (mw > 1e-20).float()
    rb = (r.unsqueeze(-1) * ew).sum(0) / (ew.sum(0) + 1e-10)

    r_np = rb.detach().numpy().copy()
    for i in range(N_BINS):
        idxs = [(i-CLIP_WINDOW+j) % N_BINS for j in range(2*CLIP_WINDOW+1) if j != CLIP_WINDOW]
        nm = float(np.median(r_np[idxs]))
        if r_np[i] > nm + CLIP_THRESH:
            r_np[i] = nm
    rb = torch.tensor(r_np, dtype=verts.dtype)

    pts = torch.stack([rb*torch.cos(bc), rb*torch.sin(bc)], dim=-1)
    raw_cm = torch.sum(torch.linalg.norm(torch.roll(pts,-1,0)-pts,dim=-1)).item()*100
    return (px.numpy(), py.numpy(), w.numpy(), r_np, bc.numpy(), raw_cm)


def draw_panel(ax, px, py, w, r_bin, bc, ref_cm, corr_cm, title):
    active = w > 0.05
    sc = ax.scatter(px[active]*100, py[active]*100, c=w[active],
                    cmap="Blues", s=6, alpha=0.5, vmin=0, vmax=1, zorder=1)

    hard = w > 0.3
    ph = np.column_stack([px[hard], py[hard]])
    if len(ph) >= 3:
        try:
            hull = ConvexHull(ph)
            hp = np.vstack([ph[hull.vertices], ph[hull.vertices[0]]])
            h_circ = np.sum(np.linalg.norm(np.diff(hp, axis=0), axis=1))*100
            ax.plot(hp[:,0]*100, hp[:,1]*100, "r-", lw=2.5,
                    label=f"Convex hull  {h_circ:.1f} cm", zorder=3)
        except Exception:
            pass

    bx = r_bin * np.cos(bc);  by = r_bin * np.sin(bc)
    poly = np.vstack([np.column_stack([bx, by]), [bx[0], by[0]]])
    ax.plot(poly[:,0]*100, poly[:,1]*100, "b-", lw=2.5,
            label=f"Soft polygon  {corr_cm:.1f} cm", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)", fontsize=10)
    ax.set_ylabel("y (cm)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        f"{title}\nref = {ref_cm:.1f} cm  |  corrected = {corr_cm:.1f} cm  |  err = {corr_cm-ref_cm:+.1f} cm",
        fontsize=10
    )
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    return sc


if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)

    # Dataset mean bust + underbust
    valid = [(e, e["measurements"]["bust_cm"], e["measurements"]["underbust_cm"])
             for e in entries
             if e["measurements"].get("bust_cm") and e["measurements"].get("underbust_cm")]
    mean_b = np.mean([v[1] for v in valid])
    mean_u = np.mean([v[2] for v in valid])
    print(f"  Dataset mean: bust={mean_b:.1f} cm  underbust={mean_u:.1f} cm")

    # Pick body closest to mean (normalised distance)
    std_b = np.std([v[1] for v in valid])
    std_u = np.std([v[2] for v in valid])
    best_d, best_e = 1e9, None
    for e, b, u in valid:
        d = ((b - mean_b)/std_b)**2 + ((u - mean_u)/std_u)**2
        if d < best_d:
            best_d, best_e = d, e
    print(f"  Chosen body: bust={best_e['measurements']['bust_cm']:.1f}  "
          f"underbust={best_e['measurements']['underbust_cm']:.1f}")

    print("Setting up model...")
    model, torso_edges, breast_idx, pose = _make_model_pack(best_e["params"])

    body  = load_anny_from_params(best_e["params"], requires_grad=False)
    verts = _forward(model, body, pose)
    bz    = bust_z_differentiable(model, verts)
    uz    = verts[0, breast_idx, 2].min()

    bpx, bpy, bw, br, bbc, b_raw = cross_section(verts, torso_edges, bz)
    upx, upy, uw, ur, ubc, u_raw = cross_section(verts, torso_edges, uz)

    b_corr = BUST_A*b_raw + BUST_B
    u_corr = UB_A*u_raw + UB_B
    ref_b  = best_e["measurements"]["bust_cm"]
    ref_u  = best_e["measurements"]["underbust_cm"]

    print(f"  Bust:      raw={b_raw:.1f}  corr={b_corr:.1f}  ref={ref_b:.1f}  err={b_corr-ref_b:+.1f}")
    print(f"  Underbust: raw={u_raw:.1f}  corr={u_corr:.1f}  ref={ref_u:.1f}  err={u_corr-ref_u:+.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Average body cross-section  (bust ref={ref_b:.0f} cm, ub ref={ref_u:.0f} cm)\n"
        f"tau={TAU}, n_bins={N_BINS}, spike clip {CLIP_WINDOW}×{CLIP_THRESH*1000:.0f} mm",
        fontsize=12, fontweight="bold"
    )

    draw_panel(ax1, bpx, bpy, bw, br, bbc, ref_b, b_corr, "BUST")
    draw_panel(ax2, upx, upy, uw, ur, ubc, ref_u, u_corr, "UNDERBUST")

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")
