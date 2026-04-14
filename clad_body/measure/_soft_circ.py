"""Differentiable soft circumference for bust, underbust, hip, and stomach.

Edge-plane intersection with sigmoid gates, angular binning with r-biased
softmax, recentered polar coordinates, and convex hull perimeter.

The polar origin is the weighted centroid of crossing points (not the mesh
XY origin). This eliminates empty back bins caused by Anny's D-shaped
cross-section where the XY origin sits near the posterior surface.

The convex hull of the resulting polygon gives the tape-measure circumference
(bridges concavities like the breast cleavage or gluteal cleft). Hull vertex
selection is non-differentiable but gradients flow through the selected
vertex positions.

Calibration (100-body dataset, random seed 42):
    bust:      A = 0.9997, B = 0.12   (MAE 0.06 cm, max 0.18 cm)
    underbust: A = 0.9830, B = 1.86   (MAE 0.39 cm, max 1.61 cm)
    hip:       A = 1.0039, B = 0.47   (MAE 0.46 cm, max 1.39 cm)

Stomach is not linearly calibrated (A=1, B=0 would be identity) because
the residual 0.93 cm MAE is Z-choice noise, not a systematic bias — see
hmr/findings/soft_stomach.md.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import ConvexHull

from .anny import (
    ARM_HAND_BONES,
    BASE_MESH_HIP_VERTICES,
    BREAST_BONES,
    remap_vertex_indices,
    setup_extended_anthro,
)

# ── Hyperparameters (validated on 100-body sweep) ────────────────────────────

N_BINS = 72
SIGMA_Z = 0.005       # metres — soft gate width for edge dz
TAU = 0.050           # metres — sigmoid gate width + softmax temperature

# ── Calibration coefficients (recenter + convex hull, 100-body fit) ──────────

BUST_A, BUST_B = 0.9997, 0.12
UB_A, UB_B = 0.9830, 1.86
HIP_A, HIP_B = 1.0039, 0.47

# Stomach — see measure_stomach_soft for algorithm
STOMACH_Z_GATE_TAU = 0.005       # metres — soft Z-band gate edge width
STOMACH_ANTERIOR_TAU = 0.001     # metres — soft-argmin over vertex Y values
STOMACH_Z_BUFFER = 0.02          # metres — hard-mask buffer on each side of [hip_z, waist_z]


def _build_torso_edges(model, faces):
    """Build edge index tensor for torso-only mesh (arm vertices excluded).

    Cached on ``model._soft_circ_torso_edges``.

    Returns (E, 2) long tensor.
    """
    cached = getattr(model, "_soft_circ_torso_edges", None)
    if cached is not None:
        return cached

    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    dominant_bone = vbi[np.arange(vbw.shape[0]), np.argmax(vbw, axis=1)]
    arm_mask = np.isin(dominant_bone, list(ARM_HAND_BONES))

    faces_np = faces.detach().cpu().numpy() if hasattr(faces, "detach") else np.asarray(faces)
    face_has_arm = arm_mask[faces_np].any(axis=1)
    torso_faces = faces_np[~face_has_arm]

    edges_set = set()
    for f in torso_faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges_set.add((min(a, b), max(a, b)))

    result = torch.tensor(list(edges_set), dtype=torch.long)
    model._soft_circ_torso_edges = result
    return result


def _build_torso_vertex_mask(model):
    """Torso-only vertex mask (arms/hands excluded), as float tensor.

    Cached on ``model._soft_circ_torso_vertex_mask``.

    Returns (V,) float tensor (1.0 for torso, 0.0 for arm/hand).
    """
    cached = getattr(model, "_soft_circ_torso_vertex_mask", None)
    if cached is not None:
        return cached

    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    dominant_bone = vbi[np.arange(vbw.shape[0]), np.argmax(vbw, axis=1)]
    arm_mask = np.isin(dominant_bone, list(ARM_HAND_BONES))
    torso_mask = (~arm_mask).astype(np.float32)
    result = torch.from_numpy(torso_mask)
    model._soft_circ_torso_vertex_mask = result
    return result


def _build_breast_idx(model):
    """Vertex indices with breast bone weight > 0.3.

    Cached on ``model._soft_circ_breast_idx``.

    Returns (K,) long tensor.
    """
    cached = getattr(model, "_soft_circ_breast_idx", None)
    if cached is not None:
        return cached

    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    bw = np.zeros(vbw.shape[0])
    for bi in BREAST_BONES:
        bw += np.where(vbi == bi, vbw, 0).sum(axis=1)
    result = torch.tensor(np.where(bw > 0.3)[0], dtype=torch.long)
    model._soft_circ_breast_idx = result
    return result


def _to_zup(verts):
    """Convert Anny Y-up verts (1, V, 3) to Z-up, floor-aligned.

    Returns (1, V, 3) tensor with autograd history preserved.
    """
    v = verts[0]

    with torch.no_grad():
        extents = v.max(0).values - v.min(0).values
        height_axis = int(extents.argmax().item())

    if height_axis == 1:  # Y-up → Z-up
        v = v[:, [0, 2, 1]]
        v = v.clone()
        v[:, 2] = -v[:, 2]

    v = v - v[:, 2].min() * torch.tensor([0, 0, 1], dtype=v.dtype, device=v.device)
    return v.unsqueeze(0)


def bust_z(model, verts_zup):
    """Bust height from breast bone tails (differentiable).

    Args:
        model: Anny model with ``_last_bone_heads``/``_last_bone_tails`` set.
        verts_zup: (1, V, 3) Z-up vertices.

    Returns scalar torch tensor (metres).
    """
    tails = getattr(model, "_last_bone_tails", None)
    heads = getattr(model, "_last_bone_heads", None)
    if tails is None or heads is None:
        return None

    tails_t = tails[0]
    labels = model.bone_labels
    try:
        bl = labels.index("breast.L")
        br = labels.index("breast.R")
    except ValueError:
        return None

    with torch.no_grad():
        all_pts = torch.cat([heads[0], tails_t], dim=0)
        extents = all_pts.max(0).values - all_pts.min(0).values
        height_axis = int(extents.argmax().item())

    all_pts_diff = torch.cat([heads[0], tails_t], dim=0)
    raw_min = all_pts_diff[:, height_axis].min()
    raw_range = all_pts_diff[:, height_axis].max() - raw_min

    frac_l = (tails_t[bl, height_axis] - raw_min) / (raw_range + 1e-10)
    frac_r = (tails_t[br, height_axis] - raw_min) / (raw_range + 1e-10)

    mesh_height = verts_zup[0, :, 2].max()
    return (frac_l + frac_r) / 2 * mesh_height


def underbust_z(model, verts_zup):
    """Underbust height from breast vertex min-z (differentiable).

    Returns scalar torch tensor (metres).
    """
    breast_idx = _build_breast_idx(model)
    return verts_zup[0, breast_idx, 2].min()


def _build_hip_idx(model):
    """Vertex indices at hip level (remapped from BASE_MESH_HIP_VERTICES).

    Cached on ``model._soft_circ_hip_idx``.

    Returns (K,) long tensor.
    """
    cached = getattr(model, "_soft_circ_hip_idx", None)
    if cached is not None:
        return cached
    idx = remap_vertex_indices(model, BASE_MESH_HIP_VERTICES)
    result = torch.tensor(idx, dtype=torch.long)
    model._soft_circ_hip_idx = result
    return result


def hip_z(model, verts_zup):
    """Hip height from hip vertex mean Z (differentiable).

    Uses ``BASE_MESH_HIP_VERTICES`` — 28 vertices at the level of greatest
    buttock prominence.  The mean Z of these vertices gives a reliable
    anatomical anchor at ~52% height.

    Returns scalar torch tensor (metres).
    """
    hip_idx = _build_hip_idx(model)
    return verts_zup[0, hip_idx, 2].mean()


def soft_circumference(verts_zup, edge_indices, z):
    """Differentiable circumference at height z.

    Uses recentered polar coordinates + convex hull perimeter.

    Args:
        verts_zup: (1, V, 3) Z-up vertices (metres).
        edge_indices: (E, 2) long tensor — torso-only edges.
        z: scalar torch tensor — cutting plane height.

    Returns:
        circ: scalar tensor — circumference in metres (differentiable).
    """
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=verts_zup.dtype, device=verts_zup.device)

    v = verts_zup[0]
    va = v[edge_indices[:, 0]]
    vb = v[edge_indices[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz = zb - za
    t = (z - za) / (dz + 1e-10)

    # Soft crossing weight
    w = torch.sigmoid(t / TAU) * torch.sigmoid((1.0 - t) / TAU)
    w = w * torch.sigmoid(torch.abs(dz) / SIGMA_Z - 1.0)

    # Intersection points
    t_c = t.clamp(0, 1)
    px = va[:, 0] + t_c * (vb[:, 0] - va[:, 0])
    py = va[:, 1] + t_c * (vb[:, 1] - va[:, 1])

    # Recenter polar origin to weighted crossing centroid
    w_sum = w.sum() + 1e-10
    cx = ((w * px).sum() / w_sum).detach()
    cy = ((w * py).sum() / w_sum).detach()

    dx, dy = px - cx, py - cy
    r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-10)
    theta = torch.atan2(dy, dx)

    # Angular binning
    bin_centers = torch.linspace(
        -np.pi, np.pi * (1 - 2.0 / N_BINS), N_BINS,
        device=verts_zup.device, dtype=verts_zup.dtype,
    )
    sig_th = (2 * np.pi / N_BINS) * 0.6

    ang_diff = theta.unsqueeze(-1) - bin_centers.unsqueeze(0)
    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
    ang_aff = torch.exp(-ang_diff ** 2 / (2 * sig_th ** 2))

    # Soft-max radius per bin (convex hull proxy, biased toward outer surface)
    comb_w = w.unsqueeze(-1) * ang_aff
    masked_w = comb_w * (w.unsqueeze(-1) > 0.01).float()
    log_w = torch.log(masked_w + 1e-30) + r.unsqueeze(-1) / TAU
    log_max = log_w.max(dim=0, keepdim=True).values
    exp_w = torch.exp(log_w - log_max) * (masked_w > 1e-20).float()
    r_bin = (r.unsqueeze(-1) * exp_w).sum(0) / (exp_w.sum(0) + 1e-10)

    # Polygon in absolute coordinates
    pts = torch.stack([
        cx + r_bin * torch.cos(bin_centers),
        cy + r_bin * torch.sin(bin_centers),
    ], dim=-1)

    # Convex hull perimeter (tape-measure: bridges concavities)
    try:
        hull_idx = ConvexHull(pts.detach().cpu().numpy()).vertices
        hp = pts[hull_idx]
        circ = torch.sum(torch.linalg.norm(torch.roll(hp, -1, 0) - hp, dim=-1))
    except Exception:
        circ = torch.sum(torch.linalg.norm(torch.roll(pts, -1, 0) - pts, dim=-1))

    return circ




def measure_bust_underbust(model, verts):
    """Compute differentiable bust and underbust circumference.

    Args:
        model: Anny model (with ``_last_bone_heads``/``_last_bone_tails`` set).
        verts: (1, V, 3) raw Anny vertices (Y-up or Z-up — auto-detected).

    Returns:
        dict with ``bust_cm`` and ``underbust_cm`` as 0-dim torch tensors.
    """
    verts_zup = _to_zup(verts)
    edges = _build_torso_edges(model, model.faces)

    bz = bust_z(model, verts_zup)
    uz = underbust_z(model, verts_zup)

    result = {}

    if bz is not None:
        raw_bust = soft_circumference(verts_zup, edges, bz) * 100
        result["bust_cm"] = BUST_A * raw_bust + BUST_B

    raw_ub = soft_circumference(verts_zup, edges, uz) * 100
    result["underbust_cm"] = UB_A * raw_ub + UB_B

    return result


def measure_hip(model, verts):
    """Compute differentiable hip circumference.

    Same algorithm as bust: soft circumference at the anatomical hip height
    (mean Z of ``BASE_MESH_HIP_VERTICES``).  The convex hull bridges the
    gluteal cleft, matching the tape-measure convention.

    At hip level (~52% height) arms are far above, so the torso-only edge
    set naturally includes all hip geometry without interference.

    Args:
        model: Anny model.
        verts: (1, V, 3) raw Anny vertices (Y-up or Z-up — auto-detected).

    Returns:
        dict with ``hip_cm`` as 0-dim torch tensor.
    """
    verts_zup = _to_zup(verts)
    edges = _build_torso_edges(model, model.faces)

    hz = hip_z(model, verts_zup)
    raw_hip = soft_circumference(verts_zup, edges, hz) * 100
    return {"hip_cm": HIP_A * raw_hip + HIP_B}


def waist_z(model, verts_zup):
    """Waist height from waist vertex loop mean Z (differentiable).

    Uses Anny's built-in waist vertex indices (46 vertices at the natural
    anatomical narrowing).

    Returns scalar torch tensor (metres).
    """
    anthro = setup_extended_anthro(model)
    return verts_zup[0, anthro.waist_vertex_indices, 2].mean()


def measure_stomach_soft(model, verts):
    """Compute differentiable stomach circumference via anterior-vertex argmin.

    Stomach = circumference at the height of maximum anterior protrusion
    between hip and waist.  Mirrors the non-differentiable
    :func:`~clad_body.measure._circumferences.measure_stomach` which scans
    vertex bands between ``hip_anchor_z`` and ``waist_z`` to find the Z
    with the lowest (most anterior) Y, then measures circumference there.

    **Algorithm** (single soft_circumference call):

    1. Build a hard Z-mask (with 2 cm buffer) plus a soft sigmoid Z-gate
       over torso vertices in ``[hip_z, waist_z]``.  The hard mask is
       needed because :math:`\\exp(-y/\\tau)` at small τ can overwhelm a
       soft sigmoid tail (e.g. a breast vertex at y = −24 cm, τ = 1 mm
       gives factor :math:`e^{40}` — far larger than any sigmoid can
       damp).  Arm/hand vertices are zeroed via the torso mask.
    2. Soft-argmin over Y of gated torso vertices: weight each vertex by
       ``exp((−y − max_gated(−y)) / τ) × gate``, then normalise.  The
       max-shift is computed over *gated* vertices only — otherwise a
       very-anterior out-of-region vertex (foot toe) would become the
       pivot and underflow the in-region weights to zero in float32.
    3. One ``soft_circumference`` call at the resolved Z.

    This is faithful to the vertex-band scan in
    :func:`measure_stomach` — both pick the single most-anterior torso
    vertex in the belly Z-range and measure circumference at its Z.
    Validated on 100 random bodies: MAE 0.93 cm, max 3.61 cm.

    Args:
        model: Anny model.
        verts: (1, V, 3) raw Anny vertices (Y-up or Z-up — auto-detected).

    Returns:
        dict with ``stomach_cm`` as 0-dim torch tensor.
    """
    verts_zup = _to_zup(verts)
    edges = _build_torso_edges(model, model.faces)
    torso_vmask = _build_torso_vertex_mask(model).to(verts_zup.device)

    hz = hip_z(model, verts_zup)
    wz = waist_z(model, verts_zup)

    v = verts_zup[0]  # (V, 3)
    y = v[:, 1]
    z = v[:, 2]

    # Hard Z-mask with STOMACH_Z_BUFFER on each side gates out bust / foot /
    # anywhere clearly outside the belly range.  Even a tiny soft-gate
    # (e.g. 1e-15) can be overwhelmed by exp(-y/τ) when τ is small and a
    # far-away vertex is very anterior (e.g. breasts at y = −24 cm).
    # Masking multiplies as a hard 0, so those vertices cannot contribute.
    with torch.no_grad():
        hard_mask = ((z > hz - STOMACH_Z_BUFFER) & (z < wz + STOMACH_Z_BUFFER)).float()

    # Soft Z-band gate provides smooth differentiability near the hip_z /
    # waist_z boundaries (so gradients flow when those anchors move).
    z_gate = torch.sigmoid((z - hz) / STOMACH_Z_GATE_TAU) \
           * torch.sigmoid((wz - z) / STOMACH_Z_GATE_TAU)
    gate = hard_mask * z_gate * torso_vmask  # (V,)

    # Soft-argmin over Y (most anterior = most negative Y), gated
    # multiplicatively so the gate acts as a hard mask.  Two subtleties:
    #
    # 1. A pure softmax with log(gate) penalty is NOT enough: log(1e-30)
    #    ≈ -69 can be overwhelmed by e.g. a foot vertex at y = -30 cm with
    #    tau = 1 mm (-y/tau = 300 ≫ 69), pulling the soft Z to the floor.
    #    So the gate must multiply post-exp, in linear space.
    # 2. The numerical-stability max shift must be taken over *gated*
    #    vertices only.  Otherwise, if a foot vertex is the global
    #    most-anterior, the shift leaves every in-region vertex at
    #    -y/τ ≈ -80, which underflows to 0 in float32 — the gated
    #    sum then becomes 0 and stomach_z collapses to 0.
    shifted = -y / STOMACH_ANTERIOR_TAU
    # Max shift over gated vertices only (detached — shift is for
    # numerical stability, not a differentiable operation).
    with torch.no_grad():
        gated_shifted = torch.where(
            gate > 1e-6, shifted,
            torch.full_like(shifted, -float("inf")),
        )
        max_shift = gated_shifted.max()
        if not torch.isfinite(max_shift):
            max_shift = torch.zeros_like(max_shift)
    unnorm = torch.exp(shifted - max_shift) * gate
    weights = unnorm / (unnorm.sum() + 1e-12)  # (V,)
    stomach_z = (weights * z).sum()

    # Single soft circumference at the resolved belly Z.
    stomach_m = soft_circumference(verts_zup, edges, stomach_z)
    return {"stomach_cm": stomach_m * 100}
