"""4-view body render for the p97 worst-case body.

Shows:
  • Body silhouette from front / back / left / right
  • Underbust cross-section plane (grey horizontal band)
  • Intersection points at uz coloured by crossing weight
  • Soft polygon ring at uz (blue) + convex hull (red)
  • Spike vertices highlighted in orange (bins where soft > hull locally)

Helps identify which back vertices cause the underbust polygon to diverge.

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_4view.py
    # → saves experiments/soft_circ_4view.png
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

from clad_body.load.anny import load_anny_from_params
from soft_circ_experiment import bust_z_differentiable
from soft_circ_improve import soft_circ, _make_model_pack, _forward

DATA = "/home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/questionnaire/data_10k_42/train.json"
OUT  = os.path.join(CLAD_BODY_ROOT, "experiments", "soft_circ_4view.png")

N_BINS = 72;  SIGMA_Z = 0.005;  TAU = 0.050
CLIP_WINDOW = 3;  CLIP_THRESH = 0.012
BUST_A, BUST_B = 0.9936, -0.9925
UB_A,   UB_B   = 0.9758,  0.4157


def cross_section_full(verts, edges, z):
    """Returns intersection geometry + r_bin (with clip) for underbust."""
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
    # also keep 3-D position of va and vb for 4-view
    pz_pt = (za + t_c * (zb - za)).detach().numpy()

    r     = torch.sqrt(px**2 + py**2 + 1e-10)
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

    rb_np = rb.detach().numpy().copy()
    for i in range(N_BINS):
        idxs = [(i-CLIP_WINDOW+j) % N_BINS for j in range(2*CLIP_WINDOW+1) if j != CLIP_WINDOW]
        nm = float(np.median(rb_np[idxs]))
        if rb_np[i] > nm + CLIP_THRESH:
            rb_np[i] = nm
    rb_clipped = torch.tensor(rb_np, dtype=verts.dtype)

    pts  = torch.stack([rb_clipped*torch.cos(bc), rb_clipped*torch.sin(bc)], dim=-1)
    raw  = torch.sum(torch.linalg.norm(torch.roll(pts,-1,0)-pts,dim=-1)).item()*100

    return dict(
        px=px.detach().numpy(), py=py.detach().numpy(), pz=pz_pt,
        w=w.detach().numpy(),
        r=r.detach().numpy(), theta=theta.detach().numpy(),
        r_bin=rb_np, bc=bc.numpy(),
        va_xyz=va.detach().numpy(), vb_xyz=vb.detach().numpy(),
        raw_cm=raw,
    )


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
            records.append((score, e, verts, bz.item(), uz.item(), ref_b, ref_u, sb, su))
        except Exception:
            pass
    records.sort(key=lambda x: x[0])
    return records[int(len(records)*0.97)]   # p97


def draw_body_view(ax, verts_np, faces, proj_x, proj_z, xlabel, title, z_slice,
                   pts_x, pts_z, pts_w, spike_mask):
    """Draw projected body silhouette + intersection ring at z_slice."""
    # body silhouette via scatter of projected vertices (light grey)
    ax.scatter(verts_np[:, proj_x]*100, verts_np[:, proj_z]*100,
               s=0.3, c="lightgrey", alpha=0.3, zorder=1)

    # underbust plane band
    ax.axhline(z_slice*100, color="steelblue", lw=1.2, ls="--", alpha=0.6, zorder=2)

    # intersection points (coloured by crossing weight)
    active = pts_w > 0.05
    ax.scatter(pts_x[active]*100, pts_z[active]*100,
               c=pts_w[active], cmap="YlOrRd", s=20, alpha=0.8,
               vmin=0, vmax=1, zorder=4)

    # spike intersection points (orange, larger)
    if spike_mask.any():
        ax.scatter(pts_x[spike_mask]*100, pts_z[spike_mask]*100,
                   c="darkorange", s=60, marker="*", zorder=5, label="back-spike edges")

    ax.set_xlabel(xlabel + " (cm)", fontsize=8)
    ax.set_ylabel("z (cm)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


if __name__ == "__main__":
    print("Loading dataset...")
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)

    print("Setting up model...")
    model, torso_edges, breast_idx, pose = _make_model_pack(entries[0]["params"])

    print("Evaluating 100 bodies to find p97...")
    row = find_p97(entries, model, torso_edges, breast_idx, pose)
    score, entry, verts, bz_val, uz_val, ref_b, ref_u, sb, su = row
    b_corr = BUST_A*sb + BUST_B;  u_corr = UB_A*su + UB_B
    print(f"  p97 body: bust ref={ref_b:.1f} corr={b_corr:.1f} err={b_corr-ref_b:+.1f}")
    print(f"            ub   ref={ref_u:.1f} corr={u_corr:.1f} err={u_corr-ref_u:+.1f}")

    verts_np = verts[0].detach().numpy()   # (N_verts, 3)  z-up, metres

    # Full cross-section geometry at underbust_z
    ub = cross_section_full(verts, torso_edges, uz_val)

    # Identify "back-spike" edges: intersection points in the back half (py > 0)
    # whose angular bin has r_bin significantly above the convex hull of high-w points
    active_mask = ub["w"] > 0.05
    hard_mask   = ub["w"] > 0.3
    pts_hard    = np.column_stack([ub["px"][hard_mask], ub["py"][hard_mask]])
    back_mask   = (ub["py"] > 0.02) & active_mask   # back half

    # Build convex hull of hard points to get reference radius per angle
    hull_r_at_theta = None
    if len(pts_hard) >= 3:
        try:
            hull = ConvexHull(pts_hard)
            # For each intersection point, estimate hull radius at its angle
            hull_pts = pts_hard[hull.vertices]
            # Compare r of each point to the interpolated hull radius in its direction
            r_pts  = ub["r"]
            th_pts = ub["theta"]
            # Approximate: is the intersection point outside the convex hull?
            from matplotlib.path import Path
            hull_path = Path(np.vstack([hull_pts, hull_pts[0]]))
            inside_hull = hull_path.contains_points(
                np.column_stack([ub["px"], ub["py"]]))
            spike_mask = back_mask & ~inside_hull & (ub["w"] > 0.1)
        except Exception:
            spike_mask = np.zeros(len(ub["w"]), dtype=bool)
    else:
        spike_mask = np.zeros(len(ub["w"]), dtype=bool)

    print(f"  Back-spike edge count: {spike_mask.sum()}")
    if spike_mask.sum() > 0:
        sp_px = ub["px"][spike_mask]*100
        sp_py = ub["py"][spike_mask]*100
        sp_r  = ub["r"][spike_mask]*100
        for i in range(min(10, spike_mask.sum())):
            print(f"    [{i}] x={sp_px[i]:.1f} y={sp_py[i]:.1f} r={sp_r[i]:.1f} cm")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"p97 worst-case body — underbust back-spike analysis\n"
        f"ub ref={ref_u:.1f} cm  corr={u_corr:.1f} cm  err={u_corr-ref_u:+.1f} cm  |  "
        f"Stars = intersection points outside convex hull on back half",
        fontsize=11, fontweight="bold"
    )

    # Row 1: 4 body views  (front / back / left side / right side)
    # coord: x=lateral, y=ant-post (neg=front), z=vertical
    view_specs = [
        # (proj_x_col, proj_z_col, flip_x, xlabel, title)
        (0,  2, False, "x (lateral)",   "Front  (looking -y→+y)"),
        (0,  2, True,  "-x (lateral)",  "Back   (looking +y→-y)"),
        (1,  2, False, "y (ant-post)",  "Right side  (looking +x→-x)"),
        (1,  2, True,  "-y (ant-post)", "Left side   (looking -x→+x)"),
    ]

    axes_top = [fig.add_subplot(2, 4, i+1) for i in range(4)]

    for ax, (cx, cz, flip, xlabel, title) in zip(axes_top, view_specs):
        sign = -1 if flip else 1
        draw_body_view(
            ax, verts_np, None,
            proj_x=cx, proj_z=cz,
            xlabel=("−" if flip else "") + xlabel.split("(")[0].strip(),
            title=title, z_slice=uz_val,
            pts_x=sign*ub["px"], pts_z=ub["pz"],
            pts_w=ub["w"], spike_mask=spike_mask,
        )

    # Row 2: cross-section at underbust_z  (full detail)
    ax_cs   = fig.add_subplot(2, 4, (5, 7))   # span cols 1-3
    ax_rbkg = fig.add_subplot(2, 4, 8)

    # -- Cross-section scatter + soft polygon + hull --
    active = ub["w"] > 0.05
    sc = ax_cs.scatter(ub["px"][active]*100, ub["py"][active]*100,
                       c=ub["w"][active], cmap="YlOrRd", s=8, alpha=0.6,
                       vmin=0, vmax=1, zorder=2)
    plt.colorbar(sc, ax=ax_cs, label="crossing weight", shrink=0.7)

    # Convex hull
    if len(pts_hard) >= 3:
        try:
            hull2 = ConvexHull(pts_hard)
            hp = pts_hard[hull2.vertices]
            hp = np.vstack([hp, hp[0]])
            ax_cs.plot(hp[:,0]*100, hp[:,1]*100, "r-", lw=2.5, label="Convex hull (ref)", zorder=3)
        except Exception:
            pass

    # Soft polygon
    bx = ub["r_bin"] * np.cos(ub["bc"])
    by = ub["r_bin"] * np.sin(ub["bc"])
    poly = np.vstack([np.column_stack([bx, by]), [bx[0], by[0]]])
    ax_cs.plot(poly[:,0]*100, poly[:,1]*100, "b-", lw=2.5,
               label=f"Soft polygon {ub['raw_cm']:.1f} cm", zorder=4)

    # Spike points
    if spike_mask.any():
        ax_cs.scatter(ub["px"][spike_mask]*100, ub["py"][spike_mask]*100,
                      c="darkorange", s=80, marker="*", zorder=5,
                      label=f"Back spike pts ({spike_mask.sum()})")

    ax_cs.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax_cs.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax_cs.set_aspect("equal")
    ax_cs.set_xlabel("x (cm)", fontsize=9);  ax_cs.set_ylabel("y (cm)", fontsize=9)
    ax_cs.set_title(f"Underbust cross-section at z={uz_val*100:.1f} cm", fontsize=10)
    ax_cs.legend(fontsize=8)
    ax_cs.annotate("BACK", xy=(0, 3), ha="center", fontsize=9, color="grey")
    ax_cs.annotate("FRONT", xy=(0, -28), ha="center", fontsize=9, color="grey")

    # -- r_bin vs angle --
    deg = np.degrees(ub["bc"])
    ax_rbkg.plot(deg, ub["r_bin"]*100, "b-o", ms=3, label="r_bin (clipped)")
    # also plot unclipped r_bin for comparison
    rb_noclip = []
    v  = verts[0]
    va = v[torso_edges[:, 0]];  vb_t = v[torso_edges[:, 1]]
    za_t = va[:, 2];  zb_t = vb_t[:, 2]
    uz_t = torch.tensor(uz_val, dtype=verts.dtype)
    dz_t = zb_t - za_t
    t_t  = (uz_t - za_t) / (dz_t + 1e-10)
    w_t  = torch.sigmoid(t_t/TAU) * torch.sigmoid((1-t_t)/TAU)
    w_t  = w_t * torch.sigmoid(torch.abs(dz_t)/SIGMA_Z - 1.0)
    t_c_t = t_t.clamp(0,1)
    px_t = va[:,0] + t_c_t*(vb_t[:,0]-va[:,0])
    py_t = va[:,1] + t_c_t*(vb_t[:,1]-va[:,1])
    r_t  = torch.sqrt(px_t**2 + py_t**2 + 1e-10)
    th_t = torch.atan2(py_t, px_t)
    bc_t = torch.tensor(ub["bc"], dtype=verts.dtype)
    sig_th = (2*np.pi/N_BINS)*0.6
    ad_t = torch.atan2(torch.sin(th_t.unsqueeze(-1)-bc_t.unsqueeze(0)),
                       torch.cos(th_t.unsqueeze(-1)-bc_t.unsqueeze(0)))
    cw_t = w_t.unsqueeze(-1)*torch.exp(-ad_t**2/(2*sig_th**2))
    mw_t = cw_t*(w_t.unsqueeze(-1)>0.01).float()
    lw_t = torch.log(mw_t+1e-30)+r_t.unsqueeze(-1)/TAU
    ew_t = torch.exp(lw_t-lw_t.max(0,keepdim=True).values)*(mw_t>1e-20).float()
    rb_nc= ((r_t.unsqueeze(-1)*ew_t).sum(0)/(ew_t.sum(0)+1e-10)).detach().numpy()
    ax_rbkg.plot(deg, rb_nc*100, "r--", lw=1.2, alpha=0.7, label="r_bin (no clip)")
    ax_rbkg.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax_rbkg.set_xlabel("Bin angle (°)", fontsize=8)
    ax_rbkg.set_ylabel("Radius (cm)", fontsize=8)
    ax_rbkg.set_title("r_bin: clipped vs raw", fontsize=9)
    ax_rbkg.legend(fontsize=8)
    ax_rbkg.set_xlim(-185, 185)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")
