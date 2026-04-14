#!/usr/bin/env python3
"""Visualize the top-N worst cases from the 100-body validation sweep.

Loads the per-body raw results, picks the worst offenders, and runs the same
deep-dive visualization as viz_worst_case.py for each — so we can see whether
the residual error is still Z-choice, contour shape, or something else.
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure import measure
from clad_body.measure.anny import measure_grad
from clad_body.measure._soft_circ import (
    _build_torso_edges,
    _build_torso_vertex_mask,
    _to_zup,
    hip_z,
    waist_z,
    soft_circumference,
    STOMACH_Z_GATE_TAU,
    STOMACH_ANTERIOR_TAU,
    STOMACH_Z_UPPER_FRAC,
)
from clad_body.measure._slicer import MeshSlicer
from clad_body.measure.anny import build_torso_mesh, build_arm_mask
import trimesh

DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "body-tuning", "questionnaire", "data_10k_42", "test.json",
))

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "viz_stomach_output", "VALIDATION_100bodies_raw.json",
)

N_WORST = 6  # how many worst-case bodies to render


def compute_everything(params, ref):
    """Forward pass + all debug info we need to plot."""
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
    torso_vmask = _build_torso_vertex_mask(model)
    edges = _build_torso_edges(model, model.faces)

    wz = waist_z(model, verts_zup).item()
    hz = hip_z(model, verts_zup).item()
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
    with torch.no_grad():
        gs = torch.where(gate > 1e-6, shifted, torch.full_like(shifted, -float("inf")))
        max_s = gs.max()
        if not torch.isfinite(max_s):
            max_s = torch.zeros_like(max_s)
    unnorm = torch.exp(shifted - max_s) * gate
    weights = unnorm / (unnorm.sum() + 1e-12)
    soft_sz = (weights * z).sum().item()

    m_grad = measure_grad(body, only=["stomach_cm"])
    grad_circ = m_grad["stomach_cm"].item()

    # Contour comparison via MeshSlicer
    vnp = v.detach().cpu().numpy()
    mesh_tri = trimesh.Trimesh(vertices=vnp, faces=model.faces.detach().cpu().numpy(),
                                process=False)
    arm_mask = build_arm_mask(model)
    torso_mesh = build_torso_mesh(mesh_tri, arm_mask)
    slicer = MeshSlicer(torso_mesh)

    def contour_at(z):
        try:
            contours = slicer.contours_at_z(z)
            return contours[0][0] if contours else None
        except Exception:
            return None

    ref_sz = ref["_stomach_z"]
    cont_ref = contour_at(ref_sz)
    cont_soft = contour_at(soft_sz)

    return dict(
        hz=hz, wz=wz, z_upper=z_upper,
        soft_sz=soft_sz,
        grad_circ=grad_circ,
        weights=weights.detach().cpu().numpy(),
        v=vnp,
        torso_vmask=torso_vmask.detach().cpu().numpy().astype(bool),
        cont_ref=cont_ref,
        cont_soft=cont_soft,
        # Hard-argmin within the gated strict range
    )


def profile_nondiff(verts_zup_np, torso_mask, z_lo, z_hi, step=0.002):
    """Non-diff min-Y per Z-band on torso-only verts."""
    tverts = verts_zup_np[torso_mask]
    zs = np.arange(z_lo, z_hi, step)
    profile = np.full(len(zs), np.nan)
    for i, zc in enumerate(zs):
        mask = np.abs(tverts[:, 2] - zc) < 0.002
        if mask.sum() < 3:
            continue
        profile[i] = tverts[mask, 1].min()
    return zs, profile


def plot_worst(idx, body_ref, dataset_entry, debug, err_cm, rank, out_path):
    name = f"idx={idx} {body_ref['labels']['gender']} {body_ref['labels'].get('body_shape', '?')}"
    hz, wz, z_upper = debug["hz"], debug["wz"], debug["z_upper"]
    ref_sz = body_ref["_stomach_z"]
    ref_circ = body_ref["stomach_cm"]
    soft_sz = debug["soft_sz"]
    grad_circ = debug["grad_circ"]
    weights = debug["weights"]
    v = debug["v"]
    torso_mask = debug["torso_vmask"]

    # Dense band profile (reference)
    z_pad = (wz - hz) * 0.2
    zs_dense, dense_prof = profile_nondiff(v, torso_mask, hz - z_pad, wz + z_pad)

    # Hard argmin in [hz, waist_z] (matches reference algorithm)
    strict = torso_mask & (v[:, 2] >= hz) & (v[:, 2] <= wz)
    hard_idx = np.argmin(np.where(strict, v[:, 1], np.inf))
    hard_v = v[hard_idx]

    # Sagittal body outline
    sag_mask = np.abs(v[:, 0]) < 0.02
    sag = v[sag_mask]

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

    fig.suptitle(
        f"RANK #{rank} worst case  |  {name}, belly={body_ref['labels'].get('belly','?')}, "
        f"height={body_ref['height_cm']:.0f}, mass={body_ref['mass_kg']:.0f}  |  "
        f"ref Z = {ref_sz:.4f} m,  soft Z = {soft_sz:.4f} m,  ΔZ = {(soft_sz - ref_sz)*100:+.2f} cm  |  "
        f"ref circ = {ref_circ:.2f} cm,  soft circ = {grad_circ:.2f} cm,  Δ = {err_cm:+.2f} cm",
        fontsize=12, fontweight="bold",
    )

    # ── Top-left: sagittal view with weighted vertex cluster ──
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sag[:, 1] * 100, sag[:, 2], s=1.2, alpha=0.35, c="gray")
    hi_w = weights > 1e-5
    if hi_w.any():
        wp = v[hi_w]
        ww = weights[hi_w]
        sc = ax.scatter(wp[:, 1] * 100, wp[:, 2],
                        s=25 + 250 * ww / (ww.max() + 1e-10),
                        c=ww, cmap="magma", alpha=0.9,
                        edgecolor="black", linewidth=0.25)
        plt.colorbar(sc, ax=ax, label="softmax weight", fraction=0.04, pad=0.02)
    ax.scatter(hard_v[1] * 100, hard_v[2], s=260, marker="*", c="red",
               edgecolor="black", linewidth=1.2, zorder=10,
               label=f"hard argmin  z={hard_v[2]:.4f}")
    ax.axhline(hz, c="#1b7a1b", lw=1.2, linestyle="--", alpha=0.7, label=f"hip_z={hz:.4f}")
    ax.axhline(wz, c="#1b7ab3", lw=1.2, linestyle="--", alpha=0.7, label=f"waist_z={wz:.4f}")
    ax.axhline(ref_sz, c="#c62828", lw=2, alpha=0.9, label=f"ref={ref_sz:.4f} → {ref_circ:.1f}cm")
    ax.axhline(soft_sz, c="#d02090", lw=2, linestyle=":",
               label=f"soft={soft_sz:.4f} → {grad_circ:.1f}cm")
    ax.set_ylim(hz - z_pad, wz + z_pad)
    ax.set_xlabel("Y (cm)  ← anterior", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("Sagittal body outline + weighted vertices", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # ── Top-middle: anterior-Y profile ──
    ax = fig.add_subplot(gs[0, 1])
    valid = ~np.isnan(dense_prof)
    ax.plot(dense_prof[valid] * 100, zs_dense[valid], "-", c="black", lw=1.3, alpha=0.8,
            label="non-diff min-Y per band  (TORSO)")
    if hi_w.any():
        ax.scatter(v[hi_w, 1] * 100, v[hi_w, 2],
                   s=15 + 200 * weights[hi_w] / (weights[hi_w].max() + 1e-10),
                   c=weights[hi_w], cmap="magma", alpha=0.8,
                   edgecolor="black", linewidth=0.2)
    ax.scatter(hard_v[1] * 100, hard_v[2], s=260, marker="*", c="red",
               edgecolor="black", linewidth=1.2, zorder=10)
    ax.axhline(ref_sz, c="#c62828", lw=2, label=f"ref Z = {ref_sz:.4f}")
    ax.axhline(soft_sz, c="#d02090", lw=2, linestyle=":", label=f"soft Z = {soft_sz:.4f}")
    ax.set_ylim(hz - z_pad, wz + z_pad)
    ax.set_xlabel("Anterior Y (cm)", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("Non-diff profile vs soft-argmin cluster", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Top-right: cross-section contours at the two Z values ──
    ax = fig.add_subplot(gs[0, 2])
    if debug["cont_ref"] is not None:
        c = debug["cont_ref"]
        ax.plot(c[:, 0] * 100, c[:, 1] * 100, "-", c="#c62828", lw=2,
                label=f"ref  Z={ref_sz:.4f}  ({ref_circ:.2f} cm)")
        ax.fill(c[:, 0] * 100, c[:, 1] * 100, c="#c62828", alpha=0.15)
    if debug["cont_soft"] is not None:
        c = debug["cont_soft"]
        ax.plot(c[:, 0] * 100, c[:, 1] * 100, ":", c="#d02090", lw=2,
                label=f"soft Z={soft_sz:.4f}  ({grad_circ:.2f} cm)")
        ax.fill(c[:, 0] * 100, c[:, 1] * 100, c="#d02090", alpha=0.15)
    ax.set_xlabel("X (cm)", fontsize=10)
    ax.set_ylabel("Y (cm)", fontsize=10)
    ax.set_title("Torso cross-section contours", fontsize=11)
    ax.legend(loc="upper center", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # ── Bottom-left: zoom near the stomach Z ──
    ax = fig.add_subplot(gs[1, 0])
    zoom_lo = min(soft_sz, ref_sz) - 0.02
    zoom_hi = max(soft_sz, ref_sz) + 0.02
    zmask = (v[:, 2] >= zoom_lo) & (v[:, 2] <= zoom_hi) & torso_mask
    ax.scatter(v[zmask, 1] * 100, v[zmask, 2], s=15,
               c=weights[zmask], cmap="magma", alpha=0.9,
               edgecolor="black", linewidth=0.25)
    ax.scatter(hard_v[1] * 100, hard_v[2], s=200, marker="*", c="red",
               edgecolor="black", linewidth=1.2, zorder=10)
    ax.axhline(ref_sz, c="#c62828", lw=2)
    ax.axhline(soft_sz, c="#d02090", lw=2, linestyle=":")
    ax.set_ylim(zoom_lo, zoom_hi)
    ax.set_xlabel("Y (cm)", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("Zoomed view near stomach Z", fontsize=11)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # ── Bottom-middle: Y histogram ──
    ax = fig.add_subplot(gs[1, 1])
    strict_y = v[strict, 1] * 100
    ax.hist(strict_y, bins=60, color="#888", alpha=0.7, edgecolor="black", lw=0.3)
    ax.axvline(hard_v[1] * 100, c="red", lw=2, label=f"hard min Y = {hard_v[1]*100:.2f} cm")
    w_y = (weights * v[:, 1]).sum() * 100
    ax.axvline(w_y, c="#d02090", lw=2, linestyle=":",
               label=f"soft weighted Y = {w_y:.2f} cm")
    ax.set_xlabel("Y (cm)", fontsize=10)
    ax.set_ylabel(f"# torso verts in [hz, wz]", fontsize=10)
    ax.set_title(f"Y distribution  ({strict.sum()} verts)", fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: top-weighted vertices ──
    ax = fig.add_subplot(gs[1, 2])
    top_n = 15
    top_idx = np.argsort(-weights)[:top_n]
    ax.barh(np.arange(top_n), weights[top_idx], color="purple", alpha=0.7)
    for i, ti in enumerate(top_idx):
        ax.text(weights[ti] + 0.005, i,
                f"z={v[ti,2]:.4f}  y={v[ti,1]*100:+.2f}",
                va="center", fontsize=8)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels([f"#{i+1}" for i in range(top_n)], fontsize=7)
    ax.set_xlim(0, weights[top_idx].max() * 1.8)
    ax.set_xlabel("softmax weight", fontsize=10)
    ax.set_title(f"Top {top_n} weighted vertices", fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    with open(DATA_PATH) as f:
        dataset = json.load(f)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    worst = sorted(results, key=lambda r: -abs(r["err"]))[:N_WORST]

    out_dir = os.path.join(os.path.dirname(__file__), "viz_stomach_output")

    for rank, r in enumerate(worst, 1):
        idx = r["idx"]
        entry = dataset[idx]
        print(f"[{rank}] Rendering idx={idx}  err={r['err']:+.2f} cm  "
              f"gender={r['gender']} shape={r.get('shape','?')}")
        try:
            debug = compute_everything(entry["params"], entry["measurements"])
            out_path = os.path.join(
                out_dir,
                f"WORST_{rank:02d}_idx{idx}_{r['gender']}_{r.get('shape','x')}_{r['err']:+.1f}cm.png",
            )
            plot_worst(idx, entry["measurements"], entry, debug, r["err"], rank, out_path)
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
