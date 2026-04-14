#!/usr/bin/env python3
"""Cross-section contour comparison for a mix of good and bad cases.

Picks ~3 good (|err| < 0.5cm) and ~3 bad (top of worst list) bodies from the
100-body validation and plots, for each:

- Left:  sagittal body outline with both ref-Z and soft-Z horizontal lines
- Mid:   cross-section contour at REFERENCE Z (measure() choice)
- Right: cross-section contour at SOFT Z (measure_grad choice)

Same scale across all cases so shape differences are directly comparable.
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure.anny import (
    build_arm_mask,
    build_torso_mesh,
    measure_grad,
)
from clad_body.measure._slicer import MeshSlicer
from clad_body.measure._soft_circ import (
    _to_zup,
    hip_z,
    waist_z,
)

DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "body-tuning", "questionnaire", "data_10k_42", "test.json",
))

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "viz_stomach_output", "VALIDATION_100bodies_raw.json",
)

N_GOOD = 3
N_BAD = 3


def load_body_state(params):
    body = load_anny_from_params(params)
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
    vnp = verts_zup[0].detach().cpu().numpy()
    hz = hip_z(model, verts_zup).item()
    wz = waist_z(model, verts_zup).item()

    m = measure_grad(body, only=["stomach_cm"])
    soft_circ = m["stomach_cm"].item()

    mesh_tri = trimesh.Trimesh(vertices=vnp,
                                faces=model.faces.detach().cpu().numpy(),
                                process=False)
    arm_mask = build_arm_mask(model)
    torso_mesh = build_torso_mesh(mesh_tri, arm_mask)
    slicer = MeshSlicer(torso_mesh)

    return body, vnp, hz, wz, soft_circ, slicer


def contour_at(slicer, z):
    """Return the convex-hull polygon at height z.

    The raw contour from MeshSlicer can come back as multiple disjoint
    segments or in mesh-traversal (not angular) order, which makes
    fill/plot look jagged.  circumference_at_z already uses the convex
    hull — we match that for plotting so the picture matches the
    measurement.
    """
    from scipy.spatial import ConvexHull
    try:
        contours = slicer.contours_at_z(z)
        if not contours:
            return None
        # Union all segments at this z — the true cross-section may be
        # returned in multiple pieces by the mesh slicer.
        all_pts = np.vstack([seg[0] for seg in contours])
        if len(all_pts) < 3:
            return None
        hull = ConvexHull(all_pts)
        ordered = all_pts[hull.vertices]
        return np.vstack([ordered, ordered[:1]])
    except Exception:
        return None


def plot_case(ax_body, ax_ref, ax_soft, data, xylim_body, xylim_contour, is_good):
    r = data["result"]
    hz, wz = data["hz"], data["wz"]
    ref_sz = data["ref_sz"]
    ref_circ = data["ref_circ"]
    soft_sz = data["soft_sz"]
    soft_circ = data["soft_circ"]
    err = r["err"]
    sag = data["sag"]
    cont_ref = data["cont_ref"]
    cont_soft = data["cont_soft"]

    tag = "GOOD" if is_good else "BAD "
    color_title = "#1b7a1b" if is_good else "#c62828"
    title_prefix = f"[{tag}]  idx={r['idx']}  {r['gender']} {r.get('shape','?')}"
    summary = (f"{title_prefix}  |  err = {err:+.2f} cm  |  "
               f"ref Z={ref_sz:.4f} ({ref_circ:.1f}cm)  vs  "
               f"soft Z={soft_sz:.4f} ({soft_circ:.1f}cm)  ΔZ={(soft_sz - ref_sz)*100:+.2f}cm")

    # Left — sagittal body outline + Z lines
    ax_body.scatter(sag[:, 1] * 100, sag[:, 2], s=1.2, alpha=0.45, c="dimgray")
    ax_body.axhline(hz, c="#1b7a1b", lw=1, linestyle="--", alpha=0.6,
                    label=f"hip_z = {hz:.3f}")
    ax_body.axhline(wz, c="#1b7ab3", lw=1, linestyle="--", alpha=0.6,
                    label=f"waist_z = {wz:.3f}")
    ax_body.axhline(ref_sz, c="#c62828", lw=2, alpha=0.9,
                    label=f"ref Z = {ref_sz:.4f}")
    ax_body.axhline(soft_sz, c="#d02090", lw=2, linestyle=":",
                    label=f"soft Z = {soft_sz:.4f}")
    ax_body.set_ylim(hz - 0.05, wz + 0.05)
    ax_body.set_xlim(*xylim_body)
    ax_body.invert_xaxis()
    ax_body.set_xlabel("Y (cm)  ← anterior", fontsize=9)
    ax_body.set_ylabel("Z (m)", fontsize=9)
    ax_body.set_title(summary, fontsize=10, fontweight="bold", color=color_title)
    ax_body.legend(loc="upper right", fontsize=7)
    ax_body.grid(True, alpha=0.3)

    # Middle — contour at ref Z
    if cont_ref is not None:
        ax_ref.plot(cont_ref[:, 0] * 100, cont_ref[:, 1] * 100, "-", c="#c62828", lw=2)
        ax_ref.fill(cont_ref[:, 0] * 100, cont_ref[:, 1] * 100,
                    c="#c62828", alpha=0.18)
    ax_ref.set_xlim(*xylim_contour)
    ax_ref.set_ylim(*xylim_contour)
    ax_ref.set_aspect("equal")
    ax_ref.set_xlabel("X (cm)", fontsize=9)
    ax_ref.set_ylabel("Y (cm)", fontsize=9)
    ax_ref.invert_yaxis()
    ax_ref.set_title(f"contour @ ref Z={ref_sz:.4f}\ncirc = {ref_circ:.2f} cm",
                     fontsize=9, color="#c62828")
    ax_ref.grid(True, alpha=0.3)

    # Right — contour at soft Z  (overlaid with ref for direct comparison)
    if cont_ref is not None:
        ax_soft.plot(cont_ref[:, 0] * 100, cont_ref[:, 1] * 100, "-", c="#c62828",
                     lw=1.2, alpha=0.5, label=f"ref ({ref_circ:.1f}cm)")
    if cont_soft is not None:
        ax_soft.plot(cont_soft[:, 0] * 100, cont_soft[:, 1] * 100, "-", c="#d02090",
                     lw=2, label=f"soft ({soft_circ:.1f}cm)")
        ax_soft.fill(cont_soft[:, 0] * 100, cont_soft[:, 1] * 100,
                     c="#d02090", alpha=0.18)
    ax_soft.set_xlim(*xylim_contour)
    ax_soft.set_ylim(*xylim_contour)
    ax_soft.set_aspect("equal")
    ax_soft.set_xlabel("X (cm)", fontsize=9)
    ax_soft.set_ylabel("Y (cm)", fontsize=9)
    ax_soft.invert_yaxis()
    ax_soft.set_title(f"contour @ soft Z={soft_sz:.4f}\ncirc = {soft_circ:.2f} cm   Δ={err:+.2f}cm",
                      fontsize=9, color="#d02090")
    ax_soft.legend(loc="lower right", fontsize=7)
    ax_soft.grid(True, alpha=0.3)


def main():
    with open(DATA_PATH) as f:
        dataset = json.load(f)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    abs_sorted = sorted(results, key=lambda r: abs(r["err"]))
    good = abs_sorted[:N_GOOD]
    bad = list(reversed(abs_sorted))[:N_BAD]

    picked = [("good", r) for r in good] + [("bad", r) for r in bad]

    cases_data = []
    for tag, r in picked:
        idx = r["idx"]
        entry = dataset[idx]
        print(f"  loading [{tag}] idx={idx}  err={r['err']:+.2f}  "
              f"{r['gender']} {r.get('shape', '?')}")

        body, vnp, hz, wz, soft_circ, slicer = load_body_state(entry["params"])

        ref_sz = entry["measurements"]["_stomach_z"]
        ref_circ = entry["measurements"]["stomach_cm"]
        soft_sz_v = r["soft"]  # from validation — but we already have circ, recompute Z
        # The validation JSON didn't store soft_sz, so recompute it quickly:
        from clad_body.measure._soft_circ import (
            _build_torso_vertex_mask, STOMACH_Z_GATE_TAU, STOMACH_ANTERIOR_TAU,
            STOMACH_Z_UPPER_FRAC,
        )
        model = body.model
        torso_vmask = _build_torso_vertex_mask(model).numpy().astype(bool)
        v_t = torch.from_numpy(vnp)
        y = v_t[:, 1]
        z = v_t[:, 2]
        z_upper = hz + STOMACH_Z_UPPER_FRAC * (wz - hz)
        BUFFER = 0.02
        hard_mask = ((z > hz - BUFFER) & (z < z_upper + BUFFER)).float()
        z_gate = torch.sigmoid((z - hz) / STOMACH_Z_GATE_TAU) \
               * torch.sigmoid((z_upper - z) / STOMACH_Z_GATE_TAU)
        gate = hard_mask * z_gate * torch.from_numpy(torso_vmask.astype(np.float32))
        shifted = -y / STOMACH_ANTERIOR_TAU
        gs = torch.where(gate > 1e-6, shifted, torch.full_like(shifted, -float("inf")))
        max_s = gs.max()
        if not torch.isfinite(max_s):
            max_s = torch.zeros_like(max_s)
        unnorm = torch.exp(shifted - max_s) * gate
        weights = unnorm / (unnorm.sum() + 1e-12)
        soft_sz = (weights * z).sum().item()

        sag = vnp[np.abs(vnp[:, 0]) < 0.02]
        cont_ref = contour_at(slicer, ref_sz)
        cont_soft = contour_at(slicer, soft_sz)

        cases_data.append({
            "tag": tag,
            "result": r,
            "hz": hz, "wz": wz,
            "ref_sz": ref_sz, "ref_circ": ref_circ,
            "soft_sz": soft_sz, "soft_circ": soft_circ,
            "sag": sag,
            "cont_ref": cont_ref,
            "cont_soft": cont_soft,
        })

    # Compute shared axis limits
    all_sag_y = np.concatenate([c["sag"][:, 1] * 100 for c in cases_data])
    xylim_body = (all_sag_y.min() - 2, all_sag_y.max() + 2)

    all_cont_x = []
    all_cont_y = []
    for c in cases_data:
        for cont in (c["cont_ref"], c["cont_soft"]):
            if cont is not None:
                all_cont_x.extend(cont[:, 0] * 100)
                all_cont_y.extend(cont[:, 1] * 100)
    all_cont = np.array(all_cont_x + all_cont_y)
    clim = max(abs(all_cont.min()), abs(all_cont.max())) + 2
    xylim_contour = (-clim, clim)

    # One big figure per case (much easier to read than a 6-row stack).
    out_paths = []
    for data in cases_data:
        is_good = data["tag"] == "good"
        r = data["result"]
        fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                                 gridspec_kw={"width_ratios": [1.2, 1, 1]})
        plot_case(axes[0], axes[1], axes[2],
                  data, xylim_body, xylim_contour, is_good)
        plt.tight_layout()
        tag = "GOOD" if is_good else "BAD"
        out_path = os.path.join(os.path.dirname(__file__),
                                "viz_stomach_output",
                                f"CONTOUR_{tag}_idx{r['idx']}_{r['gender']}_{r.get('shape','x')}_{r['err']:+.1f}cm.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
        print(f"saved: {out_path}")
    print(f"\n{len(out_paths)} images total.")


if __name__ == "__main__":
    main()
