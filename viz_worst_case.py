#!/usr/bin/env python3
"""Deep-dive visualization of the worst-case stomach measurement — female_average.

Shows every torso vertex in the belly Z-range with its softmax weight, the
single most-anterior (hard-argmin) vertex, both soft-argmin and non-diff
argmin Z levels, and the circumference contours at each. This makes it
obvious why the algorithm picks what it does.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure import measure
from clad_body.measure.anny import load_phenotype_params, measure_grad
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
from clad_body.load.anny import AnnyBody
import trimesh

TESTDATA = os.path.join(
    os.path.dirname(__file__), "clad_body", "measure", "testdata", "anny"
)
NAME = "female_average"  # the worst-case body


def main():
    params = load_phenotype_params(os.path.join(TESTDATA, NAME, "anny_params.json"))
    body = load_anny_from_params(params)

    # Ground truth + soft
    m_ref = measure(body, only=["stomach_cm"])
    ref_sz = m_ref["_stomach_z"]
    ref_circ = m_ref["stomach_cm"]

    m_grad = measure_grad(body, only=["stomach_cm"])
    grad_circ = m_grad["stomach_cm"].item()

    # Forward pass — grab verts + weights
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
    wz = waist_z(model, verts_zup).item()
    hz = hip_z(model, verts_zup).item()

    v = verts_zup[0].detach().cpu().numpy()
    torso_mask_np = torso_vmask.detach().cpu().numpy().astype(bool)

    # Compute soft-argmin weights
    z_upper = hz + STOMACH_Z_UPPER_FRAC * (wz - hz)
    v_t = verts_zup[0]
    y_t = v_t[:, 1]
    z_t = v_t[:, 2]
    BUFFER = 0.02
    hard_mask = ((z_t > hz - BUFFER) & (z_t < z_upper + BUFFER)).float()
    z_gate = torch.sigmoid((z_t - hz) / STOMACH_Z_GATE_TAU) \
           * torch.sigmoid((z_upper - z_t) / STOMACH_Z_GATE_TAU)
    gate = hard_mask * z_gate * torso_vmask
    shifted = -y_t / STOMACH_ANTERIOR_TAU
    gated_shifted = torch.where(gate > 1e-6, shifted, torch.full_like(shifted, -float("inf")))
    max_shift = gated_shifted.max()
    if not torch.isfinite(max_shift):
        max_shift = torch.zeros_like(max_shift)
    unnorm = torch.exp(shifted - max_shift) * gate
    weights = (unnorm / (unnorm.sum() + 1e-12)).detach().cpu().numpy()
    soft_sz = float((weights * v[:, 2]).sum())

    # Torso vertices in the belly Z-range (broadened for context)
    z_pad = (wz - hz) * 0.25
    in_range = torso_mask_np & (v[:, 2] > hz - z_pad) & (v[:, 2] < wz + z_pad)
    vb = v[in_range]
    wb = weights[in_range]

    # Hard argmin: single most anterior vertex in the strict [hz, z_upper] range
    strict_range = torso_mask_np & (v[:, 2] >= hz) & (v[:, 2] <= z_upper)
    hard_idx_all = np.argmin(np.where(strict_range, v[:, 1], np.inf))
    hard_v = v[hard_idx_all]

    # Non-diff vertex-band argmin (replicates measure_stomach Phase 1 exactly
    # but on the TORSO-masked vertex set so we can compare apples-to-apples).
    # The real measure() pipeline uses torso_mesh.vertices which is a slightly
    # different subset, but for interpretation this captures the idea.
    torso_verts = v[torso_mask_np]
    zs_scan = np.arange(hz, wz, 0.002)
    best_z_band = None
    best_y_band = float("inf")
    scan_profile = []
    for zc in zs_scan:
        mask = np.abs(torso_verts[:, 2] - zc) < 0.002
        if mask.sum() < 3:
            scan_profile.append((zc, np.nan))
            continue
        ymin = torso_verts[mask, 1].min()
        scan_profile.append((zc, ymin))
        if ymin < best_y_band:
            best_y_band = ymin
            best_z_band = zc
    scan_profile = np.array(scan_profile)

    # Contours at ref_sz and soft_sz using MeshSlicer on the torso mesh
    mesh_tri = trimesh.Trimesh(vertices=v, faces=model.faces.detach().cpu().numpy(),
                                process=False)
    arm_mask = build_arm_mask(model)
    torso_mesh = build_torso_mesh(mesh_tri, arm_mask)
    slicer = MeshSlicer(torso_mesh)

    def contour_at(z):
        try:
            _, pts = slicer.contours_at_z(z)[0][:2]
            return pts
        except Exception:
            return None

    cont_ref = None
    cont_soft = None
    try:
        contours_ref = slicer.contours_at_z(ref_sz)
        cont_ref = contours_ref[0][0] if contours_ref else None
    except Exception:
        pass
    try:
        contours_soft = slicer.contours_at_z(soft_sz)
        cont_soft = contours_soft[0][0] if contours_soft else None
    except Exception:
        pass

    # ─── PLOT ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], width_ratios=[1, 1.2, 1])

    fig.suptitle(
        f"WORST CASE: {NAME}  |  "
        f"ref Z = {ref_sz:.4f} m, soft Z = {soft_sz:.4f} m, ΔZ = {(soft_sz - ref_sz) * 100:+.2f} cm  |  "
        f"ref circ = {ref_circ:.2f} cm, soft circ = {grad_circ:.2f} cm, Δ = {grad_circ - ref_circ:+.2f} cm",
        fontsize=14, fontweight="bold",
    )

    # ── TOP LEFT: all torso vertices in Y-vs-Z, coloured by softmax weight ──
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(vb[:, 1] * 100, vb[:, 2], s=6, c="lightgray", alpha=0.5,
               label=f"torso verts in z-range ({in_range.sum()})")
    high_w = wb > 1e-5
    sc = ax.scatter(vb[high_w, 1] * 100, vb[high_w, 2],
                    s=30 + 300 * wb[high_w] / (wb[high_w].max() + 1e-10),
                    c=wb[high_w], cmap="magma", alpha=0.95,
                    edgecolor="black", linewidth=0.3,
                    label=f"verts with softmax weight ({high_w.sum()})")
    ax.scatter(hard_v[1] * 100, hard_v[2], s=240, marker="*", c="red",
               edgecolor="black", linewidth=1.2, zorder=10,
               label=f"hard argmin vertex (y={hard_v[1]*100:.2f}, z={hard_v[2]:.4f})")
    plt.colorbar(sc, ax=ax, label="softmax weight")

    ax.axhline(hz, c="#1b7a1b", lw=1.2, linestyle="--", alpha=0.7, label=f"hip_z = {hz:.4f}")
    ax.axhline(wz, c="#1b7ab3", lw=1.2, linestyle="--", alpha=0.7, label=f"waist_z = {wz:.4f}")
    ax.axhline(z_upper, c="purple", lw=1.4, linestyle="-.", alpha=0.8,
               label=f"z_upper (frac={STOMACH_Z_UPPER_FRAC}) = {z_upper:.4f}")
    ax.axhline(ref_sz, c="#c62828", lw=2.0, alpha=0.9, label=f"ref Z = {ref_sz:.4f}")
    ax.axhline(soft_sz, c="#d02090", lw=2.0, linestyle=":", label=f"soft Z = {soft_sz:.4f}")

    ax.set_xlabel("Y (cm)  ←  anterior  ·  posterior  →", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.set_title("Torso vertices in belly Z-range\n(coloured by soft-argmin weight)", fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── TOP MIDDLE: non-diff band-scan profile + soft selection overlay ──
    ax = fig.add_subplot(gs[0, 1])
    valid = ~np.isnan(scan_profile[:, 1])
    ax.plot(scan_profile[valid, 1] * 100, scan_profile[valid, 0],
            "-", c="black", lw=1.3, alpha=0.75,
            label="non-diff profile (min-Y in ±2mm band, TORSO verts)")

    # Overlay: for each gated torso vertex, show its Y-vs-Z weighted by softmax
    if high_w.any():
        ax.scatter(vb[high_w, 1] * 100, vb[high_w, 2],
                   s=20 + 200 * wb[high_w] / (wb[high_w].max() + 1e-10),
                   c=wb[high_w], cmap="magma", alpha=0.9,
                   edgecolor="black", linewidth=0.2,
                   label="soft-argmin weights")

    # Hard argmin marker
    ax.scatter(hard_v[1] * 100, hard_v[2], s=240, marker="*", c="red",
               edgecolor="black", linewidth=1.2, zorder=10,
               label=f"hard argmin: y={hard_v[1]*100:.2f}, z={hard_v[2]:.4f}")

    ax.axhline(ref_sz, c="#c62828", lw=2.0, alpha=0.9,
               label=f"ref argmin Z = {ref_sz:.4f}  (→ {ref_circ:.2f} cm)")
    ax.axhline(soft_sz, c="#d02090", lw=2.0, linestyle=":",
               label=f"soft argmin Z = {soft_sz:.4f}  (→ {grad_circ:.2f} cm)")
    ax.axhline(best_z_band, c="orange", lw=1.5, linestyle="--", alpha=0.8,
               label=f"replicated band argmin (torso) Z = {best_z_band:.4f}")

    ax.set_xlabel("Y (cm)  ←  anterior", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.set_title("Non-diff band scan vs soft argmin\nOrange = torso-only band argmin for reference", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── TOP RIGHT: cross-section contours at ref_sz and soft_sz ──
    ax = fig.add_subplot(gs[0, 2])
    if cont_ref is not None:
        ax.plot(cont_ref[:, 0] * 100, cont_ref[:, 1] * 100, "-", c="#c62828", lw=2,
                label=f"ref contour @ Z={ref_sz:.4f}  (circ={ref_circ:.2f} cm)")
        ax.fill(cont_ref[:, 0] * 100, cont_ref[:, 1] * 100, c="#c62828", alpha=0.15)
    if cont_soft is not None:
        ax.plot(cont_soft[:, 0] * 100, cont_soft[:, 1] * 100, ":", c="#d02090", lw=2,
                label=f"soft contour @ Z={soft_sz:.4f}  (circ={grad_circ:.2f} cm)")
        ax.fill(cont_soft[:, 0] * 100, cont_soft[:, 1] * 100, c="#d02090", alpha=0.15)
    ax.set_xlabel("X (cm)  lateral", fontsize=11)
    ax.set_ylabel("Y (cm)  anterior ↑", fontsize=11)
    ax.set_title("Torso cross-section contours\nat ref and soft Z", fontsize=11)
    ax.legend(loc="upper center", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # so anterior is up

    # ── BOTTOM ROW: triple zoom ──
    # Zoom 1: the weighted vertex cluster
    ax = fig.add_subplot(gs[1, 0])
    zoom_lo = max(hz, min(soft_sz, ref_sz) - 0.02)
    zoom_hi = min(wz, max(soft_sz, ref_sz) + 0.02)
    zoom_mask = (vb[:, 2] >= zoom_lo) & (vb[:, 2] <= zoom_hi)
    ax.scatter(vb[zoom_mask, 1] * 100, vb[zoom_mask, 2],
               s=20, c=wb[zoom_mask] if zoom_mask.any() else None,
               cmap="magma", alpha=0.9, edgecolor="black", linewidth=0.3)
    ax.scatter(hard_v[1] * 100, hard_v[2], s=200, marker="*", c="red",
               edgecolor="black", linewidth=1.2, zorder=10)
    ax.axhline(ref_sz, c="#c62828", lw=2.0)
    ax.axhline(soft_sz, c="#d02090", lw=2.0, linestyle=":")
    ax.axhline(best_z_band, c="orange", lw=1.5, linestyle="--")
    ax.set_ylim(zoom_lo, zoom_hi)
    ax.set_xlabel("Y (cm)", fontsize=10)
    ax.set_ylabel("Z (m)", fontsize=10)
    ax.set_title("Zoomed view near stomach Z", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Zoom 2: histogram of vertex Y values in the strict [hz, wz] range
    ax = fig.add_subplot(gs[1, 1])
    strict_y = v[strict_range, 1] * 100
    ax.hist(strict_y, bins=60, color="#888", alpha=0.7, edgecolor="black", lw=0.3)
    ax.axvline(hard_v[1] * 100, c="red", lw=2, label=f"hard min Y = {hard_v[1]*100:.2f} cm")
    # Mark the weighted mean Y (what soft argmin effectively sees)
    weighted_y = (weights * v[:, 1]).sum() * 100
    ax.axvline(weighted_y, c="#d02090", lw=2, linestyle=":",
               label=f"soft weighted Y = {weighted_y:.2f} cm")
    ax.set_xlabel("Y (cm)  ←  anterior", fontsize=10)
    ax.set_ylabel("# torso verts in [hip_z, waist_z]", fontsize=10)
    ax.set_title(f"Distribution of Y values\n({strict_range.sum()} torso verts in strict range)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Zoom 3: top-weighted vertices sorted by weight
    ax = fig.add_subplot(gs[1, 2])
    top_n = 20
    top_idx = np.argsort(-weights)[:top_n]
    top_w = weights[top_idx]
    top_z = v[top_idx, 2]
    top_y = v[top_idx, 1] * 100
    bars = ax.barh(np.arange(top_n), top_w, color="purple", alpha=0.7)
    for i, (y, z) in enumerate(zip(top_y, top_z)):
        ax.text(top_w[i] + 0.01, i, f"z={z:.4f}  y={y:.2f}", va="center", fontsize=8)
    ax.axvline(top_w[0] * 0.1, c="gray", lw=0.5, linestyle=":")
    ax.set_xlim(0, top_w.max() * 1.8)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels([f"#{i+1}" for i in range(top_n)], fontsize=7)
    ax.set_xlabel("softmax weight", fontsize=10)
    ax.set_title(f"Top {top_n} weighted vertices\n(soft-argmin picks their weighted Z mean)",
                 fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(os.path.dirname(__file__), "viz_stomach_output",
                            f"WORST_CASE_{NAME}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")

    # Also print key diagnostics
    print(f"\n{NAME} diagnostics:")
    print(f"  hip_z   = {hz:.4f} m")
    print(f"  waist_z = {wz:.4f} m")
    print(f"  ref stomach Z   = {ref_sz:.4f} m  → ref circ   = {ref_circ:.2f} cm")
    print(f"  soft stomach Z  = {soft_sz:.4f} m  → soft circ  = {grad_circ:.2f} cm")
    print(f"  band argmin Z (torso-only replication) = {best_z_band:.4f} m")
    print(f"  hard argmin torso vertex: y={hard_v[1]*100:.3f} cm, z={hard_v[2]:.4f} m")
    print(f"  top 5 weighted verts:")
    for i in range(5):
        idx = top_idx[i]
        print(f"    #{i+1}  w={weights[idx]:.4f}  y={v[idx,1]*100:+.3f} cm  z={v[idx,2]:.4f} m")


if __name__ == "__main__":
    main()
