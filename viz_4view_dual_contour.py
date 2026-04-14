#!/usr/bin/env python3
"""Render 4-view body render with BOTH ref-Z and soft-Z stomach contours
overlaid, so the reader can see exactly where each algorithm places the
stomach circumference on the body.

Picks worst-case bodies from the 100-body validation and produces:
    viz_stomach_output/4VIEW_<tag>_idx<N>_*.png
"""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure import measure
from clad_body.measure.anny import (
    build_arm_mask,
    build_torso_mesh,
    measure_grad,
)
from clad_body.measure._slicer import torso_circumference_at_z
from clad_body.measure._render import (
    _project_3d_to_2d,
    _render_views_pyrender,
    extract_measurement_contours,
)
from clad_body.measure._soft_circ import (
    _build_torso_vertex_mask,
    _to_zup,
    hip_z,
    waist_z,
    STOMACH_ANTERIOR_TAU,
    STOMACH_Z_GATE_TAU,
    STOMACH_Z_UPPER_FRAC,
)

DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "body-tuning", "questionnaire", "data_10k_42", "test.json",
))

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "viz_stomach_output", "VALIDATION_100bodies_raw.json",
)


def compute_soft_stomach_z(body):
    """Recompute the soft-argmin stomach Z for a body (in metres, Z-up)."""
    model = body.model
    pose = build_anny_apose(model, torch.device("cpu"))
    with torch.no_grad():
        output = model(
            pose_parameters=pose,
            phenotype_kwargs=body.phenotype_kwargs,
            local_changes_kwargs=body.local_changes_kwargs or {},
            pose_parameterization="root_relative_world",
            return_bone_ends=True,
        )
    model._last_bone_heads = output["bone_heads"]
    model._last_bone_tails = output["bone_tails"]
    verts = output["vertices"]

    verts_zup = _to_zup(verts)
    torso_vmask = _build_torso_vertex_mask(model)
    hz = hip_z(model, verts_zup).item()
    wz = waist_z(model, verts_zup).item()
    z_upper = hz + STOMACH_Z_UPPER_FRAC * (wz - hz)

    v = verts_zup[0]
    y = v[:, 1]
    z = v[:, 2]
    BUFFER = 0.02
    hard_mask = ((z > hz - BUFFER) & (z < z_upper + BUFFER)).float()
    z_gate = torch.sigmoid((z - hz) / STOMACH_Z_GATE_TAU) \
           * torch.sigmoid((z_upper - z) / STOMACH_Z_GATE_TAU)
    gate = hard_mask * z_gate * torso_vmask

    shifted = -y / STOMACH_ANTERIOR_TAU
    gs = torch.where(gate > 1e-6, shifted,
                     torch.full_like(shifted, -float("inf")))
    max_s = gs.max()
    if not torch.isfinite(max_s):
        max_s = torch.zeros_like(max_s)
    unnorm = torch.exp(shifted - max_s) * gate
    weights = unnorm / (unnorm.sum() + 1e-12)
    soft_sz = (weights * z).sum().item()
    return soft_sz


def render_4view_dual(body, ref_meas, soft_stomach_z, soft_stomach_cm, out_path,
                     title):
    """Dual-stomach-contour 4-view render.

    Uses the standard Anny contours from measure() for every body dimension
    except stomach, which is shown with BOTH the reference contour (at
    measure()'s argmax Z) and the soft-argmin contour (at measure_grad's Z).
    """
    mesh = ref_meas["_mesh_tri"]

    # Torso-only mesh for stomach contours (arms excluded).
    model = body.model
    verts_np = np.array(mesh.vertices)
    arm_mask = build_arm_mask(model)
    # build_torso_mesh expects faces to come from mesh itself
    torso_mesh = build_torso_mesh(mesh, arm_mask)

    # Reference contours — standard set from measure().
    contours = extract_measurement_contours(mesh, ref_meas,
                                            torso_mesh=torso_mesh)

    # Soft stomach contour (at measure_grad's Z).
    _, soft_stomach_pts_3d = torso_circumference_at_z(
        torso_mesh, soft_stomach_z, max_x_extent=0.60,
        return_contour=True, combine_fragments=True,
    )

    # Views (same as the standard 4-view)
    views = [
        ("Front", 10, -90),
        ("Side (R)", 10, 0),
        ("Back", 10, 90),
        ("3/4 View", 20, -60),
    ]

    center = (verts_np.max(axis=0) + verts_np.min(axis=0)) / 2
    height = verts_np[:, 2].max() - verts_np[:, 2].min()
    rendered, xmag, ymag, vw, vh = _render_views_pyrender(
        mesh, views, center, height)

    fig, axes = plt.subplots(1, 4, figsize=(22, 9))

    ref_color = "#c62828"
    soft_color = "#d02090"
    other_color = "#666666"  # muted for non-stomach contours

    for i, ((img, cam_pose), (vname, _, _)) in enumerate(zip(rendered, views)):
        ax = axes[i]
        ax.imshow(img)

        for name, pts_list in contours.items():
            if name == "stomach":
                continue
            for pts in pts_list:
                loop = np.vstack([pts, pts[:1]])
                px, py = _project_3d_to_2d(loop, cam_pose, xmag, ymag, vw, vh)
                ax.plot(px, py, c=other_color, linewidth=1.2, alpha=0.55)

        # Reference stomach contour (solid red)
        if "stomach" in contours:
            for pts in contours["stomach"]:
                loop = np.vstack([pts, pts[:1]])
                px, py = _project_3d_to_2d(loop, cam_pose, xmag, ymag, vw, vh)
                ax.plot(px, py, c=ref_color, linewidth=3.0,
                        label="ref (measure)" if i == 0 else None)

        # Soft stomach contour (dashed magenta)
        if soft_stomach_pts_3d is not None:
            loop = np.vstack([soft_stomach_pts_3d, soft_stomach_pts_3d[:1]])
            px, py = _project_3d_to_2d(loop, cam_pose, xmag, ymag, vw, vh)
            ax.plot(px, py, c=soft_color, linewidth=3.0, linestyle="--",
                    label="soft (measure_grad)" if i == 0 else None)

        ax.set_title(vname)
        ax.axis("off")

    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.85)

    ref_circ = ref_meas.get("stomach_cm", 0.0)
    ref_z = ref_meas.get("_stomach_z", 0.0)
    plt.suptitle(
        f"{title}\n"
        f"REFERENCE (solid red):   stomach = {ref_circ:.2f} cm  at Z = {ref_z:.4f} m   |   "
        f"SOFT (dashed magenta): stomach = {soft_stomach_cm:.2f} cm  at Z = {soft_stomach_z:.4f} m\n"
        f"ΔZ = {(soft_stomach_z - ref_z) * 100:+.2f} cm    "
        f"Δcirc = {soft_stomach_cm - ref_circ:+.2f} cm",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    with open(DATA_PATH) as f:
        dataset = json.load(f)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Pick top worst cases (by absolute error)
    worst = sorted(results, key=lambda r: -abs(r["err"]))[:3]

    out_dir = os.path.join(os.path.dirname(__file__), "viz_stomach_output")

    for rank, r in enumerate(worst, 1):
        idx = r["idx"]
        entry = dataset[idx]
        print(f"\n[{rank}] idx={idx}  err={r['err']:+.2f} cm  "
              f"{r['gender']} {r.get('shape', '?')}")

        body = load_anny_from_params(entry["params"])

        # Full measurements — we need the mesh and all Z anchors for the
        # non-stomach contours in the 4-view.
        ref_meas = measure(body)

        # Soft stomach Z and circumference
        soft_z = compute_soft_stomach_z(body)
        soft_circ = measure_grad(body, only=["stomach_cm"])["stomach_cm"].item()

        shape = r.get("shape", "x")
        title = (f"idx={idx}  {r['gender']} {shape}  belly={entry['measurements']['labels'].get('belly','?')}  "
                 f"height={entry['measurements']['height_cm']:.0f}cm  mass={entry['measurements']['mass_kg']:.0f}kg  "
                 f"Δ = {r['err']:+.2f} cm")

        out_path = os.path.join(
            out_dir,
            f"4VIEW_BAD_idx{idx}_{r['gender']}_{shape}_{r['err']:+.1f}cm.png"
        )
        render_4view_dual(body, ref_meas, soft_z, soft_circ, out_path, title)


if __name__ == "__main__":
    main()
