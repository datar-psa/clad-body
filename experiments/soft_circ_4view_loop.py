"""4-view body render with soft circumference loop overlaid.

Uses the same render_4view pipeline (pyrender GPU + matplotlib overlay)
and projects the soft-polygon ring at bust and underbust heights as
coloured 3D loops on top of the body mesh.

The soft polygon points (r_bin * cos/sin, z) are the actual per-bin
radius estimates — the loop that the circumference integrates around.

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_4view_loop.py [body_name]
    body_name: average (default), slim, curvy, plus_size

    # → saves experiments/soft_circ_4view_loop_<body>.png
"""

import json, os, random, sys
import numpy as np
import torch
import trimesh as _trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLAD_BODY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLAD_BODY_ROOT)
sys.path.insert(0, os.path.join(CLAD_BODY_ROOT, "experiments"))

from clad_body.load.anny import load_anny_from_params
from clad_body.measure.anny import load_phenotype_params
from clad_body.measure.anny import BREAST_BONES
from clad_body.measure._render import _render_views_pyrender, _project_3d_to_2d
from soft_circ_experiment import (
    build_torso_edge_indices, bust_z_differentiable, _apply_coordinate_transform,
    TESTDATA_DIR, ALL_SUBJECTS,
)
from soft_circ_improve import _make_model_pack, _forward, soft_circ
from clad_body.load.anny import build_anny_apose

DATA = "/home/arkadius/Projects/datar/clad/vton-exp/hmr/body-tuning/questionnaire/data_10k_42/train.json"

N_BINS      = 72
SIGMA_Z     = 0.005
TAU         = 0.050
CLIP_WINDOW = 0
CLIP_THRESH = 0.0
BUST_A, BUST_B = 0.9700, 2.16
UB_A,   UB_B   = 0.9872, 1.14
RECENTER    = True

LOOP_COLOR_BUST = "deepskyblue"
LOOP_COLOR_UB   = "limegreen"
LOOP_LW         = 3.0


def compute_soft_ring(verts, edges, z_val):
    """Compute the 3D soft-polygon ring at height z_val.

    Returns (N_BINS+1, 3) array of 3D points (closed loop).
    """
    if not isinstance(z_val, torch.Tensor):
        z_val = torch.tensor(z_val, dtype=verts.dtype)

    v  = verts[0]
    va = v[edges[:, 0]];  vb = v[edges[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz = zb - za
    t  = (z_val - za) / (dz + 1e-10)
    w  = torch.sigmoid(t / TAU) * torch.sigmoid((1 - t) / TAU)
    w  = w * torch.sigmoid(torch.abs(dz) / SIGMA_Z - 1.0)
    t_c = t.clamp(0, 1)
    px = va[:, 0] + t_c * (vb[:, 0] - va[:, 0])
    py = va[:, 1] + t_c * (vb[:, 1] - va[:, 1])

    # Recenter polar origin to weighted crossing centroid
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

    bc = torch.linspace(-np.pi, np.pi*(1-2/N_BINS), N_BINS, dtype=verts.dtype)
    sig_th = (2*np.pi/N_BINS) * 0.6
    ad = torch.atan2(torch.sin(theta.unsqueeze(-1) - bc.unsqueeze(0)),
                     torch.cos(theta.unsqueeze(-1) - bc.unsqueeze(0)))
    cw = w.unsqueeze(-1) * torch.exp(-ad**2 / (2*sig_th**2))
    mw = cw * (w.unsqueeze(-1) > 0.01).float()
    lw = torch.log(mw + 1e-30) + r.unsqueeze(-1) / TAU
    ew = torch.exp(lw - lw.max(0, keepdim=True).values) * (mw > 1e-20).float()
    rb = (r.unsqueeze(-1) * ew).sum(0) / (ew.sum(0) + 1e-10)

    if CLIP_WINDOW > 0 and CLIP_THRESH > 0.0:
        rb_np = rb.detach().numpy().copy()
        for i in range(N_BINS):
            idxs = [(i-CLIP_WINDOW+j) % N_BINS for j in range(2*CLIP_WINDOW+1) if j != CLIP_WINDOW]
            nm = float(np.median(rb_np[idxs]))
            if rb_np[i] > nm + CLIP_THRESH:
                rb_np[i] = nm
    else:
        rb_np = rb.detach().numpy().copy()

    bc_np = bc.numpy()
    cx_np, cy_np = float(cx), float(cy)
    z_np  = float(z_val.item() if hasattr(z_val, "item") else z_val)
    ring_xy = np.column_stack([cx_np + rb_np * np.cos(bc_np),
                               cy_np + rb_np * np.sin(bc_np)])
    ring_3d  = np.column_stack([ring_xy, np.full(N_BINS, z_np)])

    # Close the loop
    ring_3d = np.vstack([ring_3d, ring_3d[:1]])
    return ring_3d


def load_body(name):
    """Load one of the testdata bodies by name."""
    params_path = None
    for subj in ALL_SUBJECTS:
        if name.lower() in subj.lower():
            params_path = os.path.join(TESTDATA_DIR, subj, "anny_params.json")
            break
    if params_path is None:
        # default: first subject
        params_path = os.path.join(TESTDATA_DIR, ALL_SUBJECTS[0], "anny_params.json")
        name = ALL_SUBJECTS[0]
    print(f"  Loading testdata body: {params_path}")
    return load_phenotype_params(params_path), name


def load_p97_body(model, torso_edges, breast_idx, pose):
    """Find the p97 worst-case body from the 100-body dataset."""
    with open(DATA) as f:
        all_e = json.load(f)
    entries = random.Random(42).sample(all_e, 100)
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
            records.append((score, e["params"], verts, bz, uz))
        except Exception:
            pass
    records.sort(key=lambda x: x[0])
    _, params, verts, bz, uz = records[int(len(records)*0.97)]
    return params, verts, bz, uz


if __name__ == "__main__":
    body_name = sys.argv[1] if len(sys.argv) > 1 else "average"
    use_p97   = body_name == "p97"

    print("Setting up model...")
    if use_p97:
        with open(DATA) as f:
            all_e = json.load(f)
        seed_params = random.Random(42).sample(all_e, 1)[0]["params"]
        model, torso_edges, breast_idx, pose = _make_model_pack(seed_params)
    else:
        # Use testdata body
        params, body_name = load_body(body_name)
        body_obj = load_anny_from_params(params, requires_grad=False)
        model, torso_edges, breast_idx, pose = _make_model_pack(params)

    if use_p97:
        print("Finding p97 body from 100-body dataset...")
        params, verts, bz, uz = load_p97_body(model, torso_edges, breast_idx, pose)
        body_obj = load_anny_from_params(params, requires_grad=False)
        verts_final = verts
    else:
        verts_final = _forward(model, body_obj, pose)
        bz  = bust_z_differentiable(model, verts_final)
        uz  = verts_final[0, breast_idx, 2].min()

    print(f"  bust_z={bz.item()*100:.1f} cm   underbust_z={uz.item()*100:.1f} cm")

    # Build trimesh from forward-pass vertices (Z-up) + model faces
    verts_np = verts_final[0].detach().numpy()
    faces_np = (model.faces.detach().cpu().numpy().astype(np.int32)
                if hasattr(model.faces, "detach") else np.asarray(model.faces, np.int32))
    mesh = _trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

    # Compute soft rings
    print("Computing soft rings...")
    bust_ring = compute_soft_ring(verts_final, torso_edges, bz)
    ub_ring   = compute_soft_ring(verts_final, torso_edges, uz)
    print(f"  Bust ring: {len(bust_ring)} pts  z={bz.item()*100:.1f} cm")
    print(f"  UB ring:   {len(ub_ring)} pts  z={uz.item()*100:.1f} cm")

    # Render 4 views
    views = [
        ("Front",    10, -90),
        ("Side (R)", 10,   0),
        ("Back",     10,  90),
        ("3/4 View", 20, -60),
    ]
    center = (verts_np.max(0) + verts_np.min(0)) / 2
    height = verts_np[:, 2].max() - verts_np[:, 2].min()
    print("Rendering mesh (pyrender)...")
    rendered, xmag, ymag, vw, vh = _render_views_pyrender(mesh, views, center, height)

    # Compose figure
    fig, axes = plt.subplots(1, 4, figsize=(22, 9))
    b_corr = BUST_A * soft_circ(verts_final, torso_edges, bz, N_BINS, SIGMA_Z, TAU,
                                 clip_window=CLIP_WINDOW, clip_thresh=CLIP_THRESH,
                                 recenter=RECENTER).item()*100 + BUST_B
    u_corr = UB_A * soft_circ(verts_final, torso_edges, uz, N_BINS, SIGMA_Z, TAU,
                                clip_window=CLIP_WINDOW, clip_thresh=CLIP_THRESH,
                                recenter=RECENTER).item()*100 + UB_B

    for i, ((img, cam_pose), (view_title, _, _)) in enumerate(zip(rendered, views)):
        ax = axes[i]
        ax.imshow(img)

        # Project and draw bust ring (blue)
        px, py = _project_3d_to_2d(bust_ring, cam_pose, xmag, ymag, vw, vh)
        ax.plot(px, py, c=LOOP_COLOR_BUST, lw=LOOP_LW, label=f"Soft bust {b_corr:.1f} cm")

        # Project and draw underbust ring (green)
        px, py = _project_3d_to_2d(ub_ring, cam_pose, xmag, ymag, vw, vh)
        ax.plot(px, py, c=LOOP_COLOR_UB, lw=LOOP_LW, label=f"Soft UB {u_corr:.1f} cm")

        ax.set_title(view_title, fontsize=11)
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=9, loc="lower right",
                      facecolor="white", framealpha=0.8)

    plt.suptitle(
        f"Soft circumference loops — {body_name}\n"
        f"Bust {b_corr:.1f} cm  |  Underbust {u_corr:.1f} cm  "
        f"(tau={TAU}, n_bins={N_BINS}, clip {CLIP_WINDOW}×{CLIP_THRESH*1000:.0f}mm)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    out = os.path.join(CLAD_BODY_ROOT, "experiments",
                       f"soft_circ_4view_loop_{body_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
