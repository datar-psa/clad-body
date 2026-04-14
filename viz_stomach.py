#!/usr/bin/env python3
"""Visualize the belly profile + differentiable vs non-differentiable stomach Z.

Produces one figure per testdata body showing:
- Left: sagittal-plane vertex scatter (body outline from the side) zoomed to
  the torso region, with hip_z / waist_z / soft-Z / ref-Z horizontal lines.
- Right: anterior-Y profile Y(z) between hip_z and waist_z, with both the
  non-diff argmax-anterior Z and the differentiable soft-argmin Z marked.
  Soft-argmin weights are rendered as a sideways bar chart so the reader
  can see exactly which samples dominate the Z selection.

Also produces a summary grid (all 6 subjects) for at-a-glance comparison.
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

TESTDATA = os.path.join(
    os.path.dirname(__file__), "clad_body", "measure", "testdata", "anny"
)

SUBJECTS = [
    "male_average",
    "female_average",
    "male_plus_size",
    "female_curvy",
    "female_slim",
    "female_plus_size",
]


def profile_nondiff(verts_zup_np, z_lo, z_hi, step=0.002):
    """Non-differentiable anterior-Y profile: min-Y across vertex bands."""
    zs = np.arange(z_lo, z_hi, step)
    profile = np.full(len(zs), np.nan)
    for i, z in enumerate(zs):
        mask = np.abs(verts_zup_np[:, 2] - z) < 0.002
        if mask.sum() < 3:
            continue
        profile[i] = verts_zup_np[mask, 1].min()
    return zs, profile


def sagittal_slice_points(verts_np, x_band=0.02):
    mask = np.abs(verts_np[:, 0]) < x_band
    return verts_np[mask]


def compute_soft_selection(model, verts):
    """Compute the vertex-based soft-argmin over torso vertices.

    Returns the soft-selected stomach Z plus the per-vertex weights so we
    can visualise which vertices dominate the selection.
    """
    verts_zup = _to_zup(verts)
    torso_vmask = _build_torso_vertex_mask(model).to(verts_zup.device)
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
    gated_shifted = torch.where(gate > 1e-6, shifted, torch.full_like(shifted, -float("inf")))
    max_shift = gated_shifted.max()
    if not torch.isfinite(max_shift):
        max_shift = torch.zeros_like(max_shift)
    unnorm = torch.exp(shifted - max_shift) * gate
    weights = unnorm / (unnorm.sum() + 1e-12)
    soft_z = (weights * z).sum().item()

    return (weights.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            soft_z, hz, wz, z_upper)


def run_subject(name):
    params = load_phenotype_params(os.path.join(TESTDATA, name, "anny_params.json"))
    body = load_anny_from_params(params)

    m_ref = measure(body, only=["stomach_cm"])
    ref_sz = m_ref.get("_stomach_z", 0.0)
    ref_circ = m_ref["stomach_cm"]

    m_grad = measure_grad(body, only=["stomach_cm"])
    grad_circ = m_grad["stomach_cm"].item()

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

    weights, verts_zup_np, soft_sz, hz, wz, z_upper = compute_soft_selection(model, verts)

    z_pad = (wz - hz) * 0.3
    zs_dense, dense_profile = profile_nondiff(verts_zup_np, hz - z_pad, wz + z_pad)

    sag_pts = sagittal_slice_points(verts_zup_np)

    return dict(
        name=name,
        hz=hz, wz=wz, z_upper=z_upper,
        ref_sz=ref_sz, ref_circ=ref_circ,
        soft_sz=soft_sz, grad_circ=grad_circ,
        weights=weights, verts_zup=verts_zup_np,
        zs_dense=zs_dense, dense_profile=dense_profile,
        sag_pts=sag_pts,
    )


def plot_subject(data, out_path):
    name = data["name"]
    hz, wz = data["hz"], data["wz"]
    ref_sz, ref_circ = data["ref_sz"], data["ref_circ"]
    soft_sz, grad_circ = data["soft_sz"], data["grad_circ"]
    weights = data["weights"]
    verts_zup = data["verts_zup"]
    zs_dense = data["zs_dense"]
    dense_profile = data["dense_profile"]
    sag_pts = data["sag_pts"]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.4])
    ax_body = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        f"{name}    |    "
        f"ref Z = {ref_sz:.3f} m,  soft Z = {soft_sz:.3f} m,  ΔZ = {(soft_sz - ref_sz) * 100:+.1f} cm    |    "
        f"ref circ = {ref_circ:.1f} cm,  soft circ = {grad_circ:.1f} cm,  Δ = {grad_circ - ref_circ:+.1f} cm",
        fontsize=14, fontweight="bold",
    )

    z_pad = (wz - hz) * 0.35

    # ── Left: sagittal body outline with per-vertex weights coloured in ──
    ax_body.scatter(sag_pts[:, 1] * 100, sag_pts[:, 2], s=1.2, alpha=0.35, c="dimgray")

    # Highlight the vertices carrying non-negligible softmax weight
    weighted = weights > 1e-4
    if weighted.any():
        wp = verts_zup[weighted]
        ww = weights[weighted]
        sc = ax_body.scatter(wp[:, 1] * 100, wp[:, 2], s=30 + 250 * ww / ww.max(),
                             c=ww, cmap="magma", alpha=0.9, edgecolor="black", linewidth=0.3,
                             label=f"weighted torso verts ({weighted.sum()})")
        plt.colorbar(sc, ax=ax_body, label="softmax weight", fraction=0.035, pad=0.02)

    ax_body.axhline(hz, c="#1b7a1b", lw=1.3, linestyle="--", alpha=0.8, label=f"hip_z = {hz:.3f}")
    ax_body.axhline(wz, c="#1b7ab3", lw=1.3, linestyle="--", alpha=0.8, label=f"waist_z = {wz:.3f}")
    ax_body.axhline(ref_sz, c="#c62828", lw=2.2, alpha=0.85,
                    label=f"ref stomach_z = {ref_sz:.3f}  (circ {ref_circ:.1f} cm)")
    ax_body.axhline(soft_sz, c="#d02090", lw=2.2, linestyle=":",
                    label=f"soft stomach_z = {soft_sz:.3f}  (circ {grad_circ:.1f} cm)")
    ax_body.set_ylim(hz - z_pad, wz + z_pad)
    ax_body.set_xlabel("Y (cm)  ←  anterior  ·  posterior  →", fontsize=11)
    ax_body.set_ylabel("Z height (m)", fontsize=11)
    ax_body.set_title("Sagittal body outline  |  vertices weighted by soft-argmin", fontsize=12)
    ax_body.legend(loc="upper right", fontsize=9)
    ax_body.grid(True, alpha=0.25)
    ax_body.invert_xaxis()

    # ── Right: anterior-Y profile (non-diff reference for validation) ──
    ax_prof.plot(dense_profile * 100, zs_dense, "-", c="black", alpha=0.7, lw=1.3,
                 label="non-diff profile  (min-Y per Z-band)")

    ax_prof.axhline(hz, c="#1b7a1b", lw=1.3, linestyle="--", alpha=0.6)
    ax_prof.axhline(wz, c="#1b7ab3", lw=1.3, linestyle="--", alpha=0.6)
    ax_prof.axhline(ref_sz, c="#c62828", lw=2.2, alpha=0.85,
                    label=f"ref argmin  Z = {ref_sz:.3f}")
    ax_prof.axhline(soft_sz, c="#d02090", lw=2.2, linestyle=":",
                    label=f"soft argmin  Z = {soft_sz:.3f}")

    if not np.all(np.isnan(dense_profile)):
        argmin_idx = np.nanargmin(dense_profile)
        y_min = dense_profile[argmin_idx]
        ax_prof.axvline(y_min * 100, c="#c62828", lw=1.0, linestyle=":", alpha=0.45)

    ax_prof.set_ylim(hz - z_pad, wz + z_pad)
    ax_prof.set_xlabel("Anterior Y (cm)    more negative → more protruding belly", fontsize=11)
    ax_prof.set_ylabel("Z height (m)", fontsize=11)
    ax_prof.set_title("Anterior-Y profile (non-diff vertex band scan)", fontsize=12)
    ax_prof.legend(loc="lower right", fontsize=9)
    ax_prof.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_summary(all_data, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    for ax, data in zip(axes.flat, all_data):
        name = data["name"]
        hz, wz = data["hz"], data["wz"]
        ref_sz, ref_circ = data["ref_sz"], data["ref_circ"]
        soft_sz, grad_circ = data["soft_sz"], data["grad_circ"]

        ax.plot(data["dense_profile"] * 100, data["zs_dense"], "-", c="black",
                alpha=0.7, lw=1.2, label="non-diff min-Y")
        # Scatter: torso vertices in the sagittal band, coloured by softmax weight
        weighted = data["weights"] > 1e-4
        if weighted.any():
            wp = data["verts_zup"][weighted]
            ww = data["weights"][weighted]
            ax.scatter(wp[:, 1] * 100, wp[:, 2], s=15 + 120 * ww / ww.max(),
                       c=ww, cmap="magma", alpha=0.8, edgecolor="black", linewidth=0.2)
        ax.axhline(hz, c="#1b7a1b", lw=1, linestyle="--", alpha=0.5)
        ax.axhline(wz, c="#1b7ab3", lw=1, linestyle="--", alpha=0.5)
        ax.axhline(ref_sz, c="#c62828", lw=2, alpha=0.9)
        ax.axhline(soft_sz, c="#d02090", lw=2, linestyle=":")

        z_pad = (wz - hz) * 0.35
        ax.set_ylim(hz - z_pad, wz + z_pad)
        ax.set_title(
            f"{name}\n"
            f"ref {ref_circ:.1f}cm @ Z={ref_sz:.3f}  |  "
            f"soft {grad_circ:.1f}cm @ Z={soft_sz:.3f}  "
            f"(Δ={grad_circ - ref_circ:+.1f} cm)",
            fontsize=10,
        )
        ax.set_xlabel("Anterior Y (cm)", fontsize=9)
        ax.set_ylabel("Z (m)", fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Stomach measurement: non-diff vs differentiable — all subjects",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "viz_stomach_output")
    os.makedirs(out_dir, exist_ok=True)

    all_data = []
    for name in SUBJECTS:
        print(f"Visualizing {name}...")
        data = run_subject(name)
        all_data.append(data)
        out_path = os.path.join(out_dir, f"{name}_stomach_profile.png")
        plot_subject(data, out_path)

    plot_summary(all_data, os.path.join(out_dir, "ALL_SUBJECTS_summary.png"))
    print(f"\nAll visualizations written to {out_dir}/")

    print("\nSummary:")
    print(f"{'subject':25s}  {'ref_Z':>7s}  {'soft_Z':>7s}  {'ΔZ(cm)':>7s}  "
          f"{'ref':>6s}  {'soft':>6s}  {'Δ(cm)':>7s}")
    for d in all_data:
        dz_cm = (d["soft_sz"] - d["ref_sz"]) * 100
        dc = d["grad_circ"] - d["ref_circ"]
        print(f"{d['name']:25s}  {d['ref_sz']:7.3f}  {d['soft_sz']:7.3f}  "
              f"{dz_cm:+7.1f}  {d['ref_circ']:6.1f}  {d['grad_circ']:6.1f}  {dc:+7.1f}")


if __name__ == "__main__":
    main()
