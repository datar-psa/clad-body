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
    LEFT_LEG_BONES,
    RIGHT_LEG_BONES,
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

# Neck — tilted soft circumference perpendicular to the neck bone axis, with
# the plane origin shifted down the neck axis from the ``neck02`` bone head
# by ``NECK_BELOW_ADAMS_APPLE_COEF × body_height``.
#
# ISO 8559-1 §5.3.2 specifies neck girth "at a point just below the bulge at
# the thyroid cartilage (Adam's apple), measured perpendicular to the
# longitudinal axis of the neck" — explicitly below the bulge, not at it.
# Hodgdon & Beckett (1984, NHRC Report 84-11, which calibrated the Navy BF
# coefficients used in :mod:`clad_body.measure.anny`) measured "just below
# the larynx (i.e., Adam's apple) with the tape sloping slightly downward
# to the front."  Both protocols point to the C6 level, ~1.5-2 cm inferior
# to the Adam's apple bulge.  Anny's ``neck02`` bone head sits AT the Adam's
# apple, so we shift downward along the neck axis.
#
# The coefficient is height-proportional rather than absolute because the
# offset is structurally tied to body scale: foot verts extend below the
# foot bone head by a distance that grows with total body size (longer
# feet → bigger gap).  Fitting on 100 random bodies from data_10k_42 gives
# a remarkably tight ratio: offset / body_height = 0.01014 ± 0.00048 (CV
# 4.7 %).  Using the fitted coefficient reaches MAE 0.19 cm / MAX 0.56 cm
# against the :func:`measure_neck` reference — 2.4× better than a fixed
# absolute offset (0.46 / 1.45 cm on the same sample).
NECK_A, NECK_B = 1.0, 0.0
NECK_BELOW_ADAMS_APPLE_COEF = 0.0102  # fraction of body height

# Thigh — soft circumference on per-leg edges at 43 % of body height, which
# is where the ISO plane-sweep reference ALWAYS lands (the sweep is hard-
# capped at 0.43 × height in :func:`~clad_body.measure._circumferences.measure_thigh`,
# so the reference effectively reports "fullest thigh at the sweep cap"
# regardless of bone geometry — using the same fraction here matches it to
# near-identity).
#
# Calibration on 100 random bodies from data_10k_42/test.json:
#     A = 0.999, B = -0.24   (MAE 0.07 cm, P95 0.17 cm, max 0.25 cm)
THIGH_Z_FRAC = 0.43              # fraction of mesh height
THIGH_A, THIGH_B = 0.999, -0.24

# Knee — perpendicular-plane soft circumference at the kneecap centre.
# The plane origin is the per-leg ``upperleg02.tail`` bone position (= knee
# joint articulation = ISO 8559-1 §3.1.17 "centre of the kneecap" on Anny
# A-pose meshes; verified empirically via anterior-protrusion peak).  The
# plane normal is the femur–tibia bisector at that joint, so the slice
# follows the actual leg axis (5–8° off vertical on Anny) instead of being
# horizontal — same convention as the numpy reference
# :func:`~clad_body.measure._circumferences.measure_knee` in joint-anchored
# mode.
#
# Calibration on 100 random bodies from data_10k_42/test.json:
#     A = 0.9716, B = +0.4648
#     MAE 0.24 cm, P95 0.51 cm, max 0.69 cm
KNEE_A, KNEE_B = 0.9716, 0.4648

# Calf — perpendicular-plane soft circumference at the gastrocnemius peak.
# Z is resolved by per-leg soft-argmax over a discrete Z grid, where each
# bin's "score" is a leg-axis-relative spread proxy aggregated via Gaussian
# Z-binning of the leg vertices.  This mirrors what numpy ``measure_calf``
# does — it finds the Z of maximum *circumference* by sweeping
# :func:`_two_leg_avg_circumference`; we approximate that by aggregating
# (x − leg_cx)² + (y − leg_cy)² over a soft Z-binning of leg vertices, then
# soft-argmax over Z bins.
#
# An earlier per-vertex score (``y/τ``, posterior protrusion) was rejected
# because the y peak and the circumference peak don't always coincide on
# slim bodies — y peaked at z≈0.32 while circumference peaked at z≈0.36 on
# female_slim, giving a 1.9 cm under-read.  The per-Z aggregate proxy
# tracks circumference more faithfully (within 0.3 cm on testdata).
#
# The slice is perpendicular to the tibia axis (``ankle − knee``) at the
# resolved Z — same convention as the numpy reference, where Anny's ~8°
# tibia tilt makes a world-frame horizontal slice diverge from what a
# tape would measure on a body standing straight.
#
# Hyperparameters:
#   CALF_N_ZBINS    — number of candidate Zs in the search range
#   CALF_Z_BIN_SIGMA_FRAC — Gaussian Z-binning bandwidth as a fraction of
#                            the bin spacing (smoothing across bins)
#   CALF_SOFTMAX_TAU — temperature on the per-bin spread for Z soft-argmax
#   CALF_Z_BUFFER   — hard-mask buffer on each side of [ankle+6cm,
#                     knee-4cm] (same role as STOMACH_Z_BUFFER)
#
# Calibration on 100 random bodies from data_10k_42/test.json:
#     A = 1.0246, B = -1.0424
#     MAE 0.10 cm, P95 0.23 cm, max 1.72 cm
# Uncalibrated identity (A=1, B=0) is already tight: MAE 0.16 cm, bias
# +0.14 cm — the linear trim mainly compensates for slope-related drift
# on extreme body types, not a systematic offset.
CALF_N_ZBINS = 64
CALF_Z_BIN_SIGMA_FRAC = 1.5      # 1.5 bin widths Gaussian smoothing
CALF_SOFTMAX_TAU = 0.0005        # metres² — temperature on per-bin spread
CALF_Z_BUFFER = 0.005            # metres — hard band guard on each side
CALF_A, CALF_B = 1.0246, -1.0424

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


def _build_neck_edges(model, faces):
    """Build edge index tensor for the neck submesh.

    A face is kept only if all three vertices have summed skinning weight
    > 0.3 on ``neck01`` or ``neck02``. This excludes head/chin vertices
    (which would cross a tilted cutting plane at chin level) and
    clavicle/shoulder vertices (which would cross when the plane tilts
    forward). Topology-only — depends on skinning weights, not phenotype,
    so the result is cached on the model instance.

    Cached on ``model._soft_circ_neck_edges``.

    Returns (E, 2) long tensor.
    """
    cached = getattr(model, "_soft_circ_neck_edges", None)
    if cached is not None:
        return cached

    labels = list(model.bone_labels)
    try:
        neck_bone_idx = [labels.index("neck01"), labels.index("neck02")]
    except ValueError as e:
        raise RuntimeError(f"Anny model missing neck bone: {e}")

    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    bw = np.zeros(vbw.shape[0])
    for bi in neck_bone_idx:
        bw += np.where(vbi == bi, vbw, 0).sum(axis=1)
    neck_mask = bw > 0.3

    faces_np = faces.detach().cpu().numpy() if hasattr(faces, "detach") else np.asarray(faces)
    face_all_neck = neck_mask[faces_np].all(axis=1)
    neck_faces = faces_np[face_all_neck]

    edges_set = set()
    for f in neck_faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges_set.add((min(a, b), max(a, b)))

    result = torch.tensor(list(edges_set), dtype=torch.long)
    model._soft_circ_neck_edges = result
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


def soft_circumference_plane(verts, edge_indices, origin, normal):
    """Differentiable circumference along an arbitrary cutting plane.

    Generalization of :func:`soft_circumference` — accepts an arbitrary plane
    ``(origin, normal)`` instead of a horizontal Z level. Used for the neck
    (ISO §5.3.2 requires perpendicular-to-neck-axis slicing; the neck tilts
    ~15-20° forward from vertical, so a horizontal slice overestimates by
    5-6 %).

    Works in whichever frame ``verts`` is expressed in — the caller is
    responsible for supplying ``origin`` and ``normal`` in the same frame.
    Bone positions from ``model._last_bone_heads/_tails`` are in the same
    frame as ``output["vertices"]``, so passing them directly is safe.

    Same machinery as the horizontal version: sigmoid gate on edge-plane
    crossings, angular binning with r-biased softmax, recentered polar
    coordinates, convex hull perimeter. The only difference is the
    signed-distance computation (``dot(p - origin, normal)`` instead of
    ``z - z_plane``) and a Gram-Schmidt 2D frame on the plane.

    Args:
        verts: (1, V, 3) tensor — vertex positions in any frame.
        edge_indices: (E, 2) long tensor — edges to intersect with the plane.
        origin: (3,) tensor or tuple — any point on the plane.
        normal: (3,) tensor or tuple — plane normal (normalized internally).

    Returns:
        Scalar torch tensor — circumference in the same length units as
        ``verts`` (typically metres).
    """
    if not isinstance(origin, torch.Tensor):
        origin = torch.as_tensor(origin, dtype=verts.dtype, device=verts.device)
    if not isinstance(normal, torch.Tensor):
        normal = torch.as_tensor(normal, dtype=verts.dtype, device=verts.device)
    normal = normal / (torch.linalg.norm(normal) + 1e-10)

    v = verts[0]
    va = v[edge_indices[:, 0]]
    vb = v[edge_indices[:, 1]]

    # Signed distance from the plane for each endpoint
    sa = ((va - origin) * normal).sum(dim=-1)
    sb = ((vb - origin) * normal).sum(dim=-1)
    ds = sb - sa
    t = -sa / (ds + 1e-10)

    # Same soft gates as the horizontal variant — TAU and SIGMA_Z now act
    # along the plane normal rather than world Z, but the numeric scale
    # (5 mm / 5 cm) is geometrically equivalent.
    w = torch.sigmoid(t / TAU) * torch.sigmoid((1.0 - t) / TAU)
    w = w * torch.sigmoid(torch.abs(ds) / SIGMA_Z - 1.0)

    # 3D intersection points along each crossing edge
    t_c = t.clamp(0, 1)
    p = va + t_c.unsqueeze(-1) * (vb - va)

    # Plane-local 2D frame via Gram-Schmidt. Pick the world axis that's
    # least parallel to ``normal`` as the reference — guarantees a
    # well-conditioned ``u`` even for tilted planes. Under no_grad so the
    # discrete argmin can't introduce gradient noise; ``u`` and ``v`` still
    # flow gradients into r, theta via the forward arithmetic.
    with torch.no_grad():
        ax = int(torch.argmin(torch.abs(normal)).item())
        ref = torch.zeros(3, dtype=v.dtype, device=v.device)
        ref[ax] = 1.0
    u = ref - (ref * normal).sum() * normal
    u = u / (torch.linalg.norm(u) + 1e-10)
    vv = torch.linalg.cross(normal, u)

    # Project intersection points into the plane-local (u, v) frame
    p_rel = p - origin
    px = (p_rel * u).sum(dim=-1)
    py = (p_rel * vv).sum(dim=-1)

    # Recenter polar origin to weighted crossing centroid (detached — same
    # rationale as in the horizontal version).
    w_sum = w.sum() + 1e-10
    cx = ((w * px).sum() / w_sum).detach()
    cy = ((w * py).sum() / w_sum).detach()

    dx, dy = px - cx, py - cy
    r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-10)
    theta = torch.atan2(dy, dx)

    # Angular binning — identical to the horizontal variant
    bin_centers = torch.linspace(
        -np.pi, np.pi * (1 - 2.0 / N_BINS), N_BINS,
        device=v.device, dtype=v.dtype,
    )
    sig_th = (2 * np.pi / N_BINS) * 0.6

    ang_diff = theta.unsqueeze(-1) - bin_centers.unsqueeze(0)
    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
    ang_aff = torch.exp(-ang_diff ** 2 / (2 * sig_th ** 2))

    comb_w = w.unsqueeze(-1) * ang_aff
    masked_w = comb_w * (w.unsqueeze(-1) > 0.01).float()
    log_w = torch.log(masked_w + 1e-30) + r.unsqueeze(-1) / TAU
    log_max = log_w.max(dim=0, keepdim=True).values
    exp_w = torch.exp(log_w - log_max) * (masked_w > 1e-20).float()
    r_bin = (r.unsqueeze(-1) * exp_w).sum(0) / (exp_w.sum(0) + 1e-10)

    # Polygon in plane-local coordinates, convex hull perimeter
    pts = torch.stack([
        cx + r_bin * torch.cos(bin_centers),
        cy + r_bin * torch.sin(bin_centers),
    ], dim=-1)

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


def measure_neck_soft(model, verts):
    """Compute differentiable neck circumference perpendicular to neck axis.

    ISO 8559-1 §5.3.2: girth of the neck at the thyroid cartilage (Adam's
    apple), measured perpendicular to the neck's longitudinal axis.

    The neck tilts ~15-20° forward from vertical, so a horizontal slice would
    overestimate by 5-6 %. This function uses :func:`soft_circumference_plane`
    with:

    - ``origin = neck02`` bone head (= ``neck01`` tail ≈ Adam's apple level,
      ~86 % of body height).
    - ``normal = head - neck01`` head (the full neck-chain direction, from
      the base of the neck up to the base of the skull).

    Both anchors come from ``model._last_bone_heads`` which is populated by
    the :func:`measure_grad` forward pass when ``return_bone_ends=True``.
    Gradients flow through the plane anchor and the plane normal via LBS,
    and through the vertex positions via blendshapes + LBS.

    Args:
        model: Anny model with ``_last_bone_heads`` populated by a forward
            pass that used ``return_bone_ends=True``.
        verts: (1, V, 3) raw Anny vertices (same frame as the stored bone
            positions — no Y-up/Z-up conversion applied here).

    Returns:
        dict with ``neck_cm`` as 0-dim torch tensor (cm).
    """
    heads = getattr(model, "_last_bone_heads", None)
    if heads is None:
        raise RuntimeError(
            "measure_neck_soft requires model._last_bone_heads — run a "
            "forward pass with return_bone_ends=True first"
        )

    labels = list(model.bone_labels)
    try:
        neck01_i = labels.index("neck01")
        neck02_i = labels.index("neck02")
        head_i = labels.index("head")
    except ValueError as e:
        raise RuntimeError(f"Anny model missing neck/head bone: {e}")

    neck_base = heads[0, neck01_i]   # base of neck (C7-ish)
    neck_mid = heads[0, neck02_i]    # Adam's apple level (= neck01 tail)
    head_pos = heads[0, head_i]      # base of skull
    axis = head_pos - neck_base
    axis_unit = axis / (torch.linalg.norm(axis) + 1e-10)

    # Shift the plane origin down the neck axis by a fraction of body height
    # to land at the ISO §5.3.2 "just below the Adam's apple" level.  See the
    # module docstring at ``NECK_BELOW_ADAMS_APPLE_COEF`` for the calibration
    # and rationale.  ``body_height`` uses the same verts max/min pattern as
    # ``thigh_z``, so the gradient flows through the crown/sole vertices
    # consistent with the rest of this module.
    body_height = verts[0, :, 2].max() - verts[0, :, 2].min()
    origin = neck_mid - (NECK_BELOW_ADAMS_APPLE_COEF * body_height) * axis_unit

    edges = _build_neck_edges(model, model.faces).to(verts.device)
    raw_neck_cm = soft_circumference_plane(verts, edges, origin, axis) * 100
    return {"neck_cm": NECK_A * raw_neck_cm + NECK_B}


def _build_leg_edges(model, side):
    """Build edge index tensor for a single-leg submesh.

    A face is kept only if all three vertices are dominantly skinned to a bone
    of the requested leg (``'L'`` or ``'R'``). This isolates one thigh so the
    angular binning inside :func:`soft_circumference` doesn't mix crossings
    from the other leg.

    Cached on ``model._soft_circ_leg_edges_<side>``.

    Returns (E, 2) long tensor.
    """
    attr = f"_soft_circ_leg_edges_{side}"
    cached = getattr(model, attr, None)
    if cached is not None:
        return cached

    leg_bones = LEFT_LEG_BONES if side == "L" else RIGHT_LEG_BONES

    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    dominant_bone = vbi[np.arange(vbw.shape[0]), np.argmax(vbw, axis=1)]
    leg_vert_mask = np.isin(dominant_bone, list(leg_bones))

    faces = model.faces
    faces_np = faces.detach().cpu().numpy() if hasattr(faces, "detach") else np.asarray(faces)
    face_all_leg = leg_vert_mask[faces_np].all(axis=1)
    leg_faces = faces_np[face_all_leg]

    edges_set = set()
    for f in leg_faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges_set.add((min(a, b), max(a, b)))

    result = torch.tensor(list(edges_set), dtype=torch.long)
    setattr(model, attr, result)
    return result


def thigh_z(verts_zup):
    """Thigh cutting-plane height — :data:`THIGH_Z_FRAC` × mesh height.

    The ISO plane-sweep reference hard-caps its thigh scan at 0.43 × height
    (see :func:`~clad_body.measure._circumferences.measure_thigh`), so this
    matches that exact Z. ``verts_zup[0, :, 2].max()`` is differentiable,
    so gradients flow from any phenotype or local_change that affects
    overall body height.

    Returns scalar torch tensor (metres).
    """
    mesh_height = verts_zup[0, :, 2].max()
    return THIGH_Z_FRAC * mesh_height


def measure_thigh_soft(model, verts):
    """Compute differentiable thigh circumference.

    Same soft-circumference machinery as bust/hip — edge-plane intersection
    with sigmoid gates, angular binning with r-biased softmax, recentered
    polar coordinates, convex hull perimeter — but applied per-leg to single-
    leg edge sets so the two thighs don't share an angular bin.

    The cutting plane sits at ``THIGH_Z_FRAC × mesh_height`` (43 % of body
    height) — exactly where the ISO plane-sweep reference is hard-capped,
    giving near-identity agreement (MAE 0.07 cm, max 0.25 cm across 100
    random bodies). Left and right circumferences are averaged.

    Args:
        model: Anny model (with ``_last_bone_heads`` / ``_last_bone_tails`` set).
        verts: (1, V, 3) raw Anny vertices (Y-up or Z-up — auto-detected).

    Returns:
        dict with ``thigh_cm`` as 0-dim torch tensor.
    """
    verts_zup = _to_zup(verts)

    edges_L = _build_leg_edges(model, "L").to(verts_zup.device)
    edges_R = _build_leg_edges(model, "R").to(verts_zup.device)

    z = thigh_z(verts_zup)

    circ_L = soft_circumference(verts_zup, edges_L, z)
    circ_R = soft_circumference(verts_zup, edges_R, z)

    raw_thigh_cm = (circ_L + circ_R) / 2 * 100
    return {"thigh_cm": THIGH_A * raw_thigh_cm + THIGH_B}


def _knee_axis_and_origin(model, side):
    """Per-leg knee plane: origin at ``upperleg02.tail`` (= patella centre on
    Anny A-pose meshes per ISO §3.1.17), normal along the femur–tibia
    bisector.

    Both ``femur = upperleg02.tail − upperleg02.head`` and
    ``tibia = lowerleg02.tail − lowerleg01.head`` point downward (head→tail),
    so the bisector is the natural "tape direction" through the knee hinge.
    Anny legs sit 5–8° off vertical; using the bisector instead of a
    horizontal plane removes the ~1 % overestimate that horizontal slicing
    introduces, mirroring how :func:`measure_neck_soft` and
    :func:`measure_upperarm` handle off-vertical limbs.

    Returns ``(origin, axis)`` torch tensors in the model's native frame
    (matches ``model._last_bone_heads`` / ``output["vertices"]``), or
    ``(None, None)`` if bone ends are missing.
    """
    heads = getattr(model, "_last_bone_heads", None)
    tails = getattr(model, "_last_bone_tails", None)
    if heads is None or tails is None:
        return None, None

    labels = list(model.bone_labels)
    try:
        ul2 = labels.index(f"upperleg02.{side}")
        ll1 = labels.index(f"lowerleg01.{side}")
        ll2 = labels.index(f"lowerleg02.{side}")
    except ValueError:
        return None, None

    knee = tails[0, ul2]                      # = heads[0, ll1] (same point)
    femur = tails[0, ul2] - heads[0, ul2]     # perineum → knee
    tibia = tails[0, ll2] - heads[0, ll1]     # knee → ankle

    f_n = torch.linalg.norm(femur) + 1e-10
    t_n = torch.linalg.norm(tibia) + 1e-10
    axis = femur / f_n + tibia / t_n          # bisector (downward)
    axis = axis / (torch.linalg.norm(axis) + 1e-10)
    return knee, axis


def measure_knee_soft(model, verts):
    """Compute differentiable knee circumference at the kneecap centre.

    Per-leg perpendicular slice through the knee joint, normal aligned with
    the femur–tibia bisector at that side (see :func:`_knee_axis_and_origin`).
    Origin sits at ``upperleg02.tail`` which on Anny A-pose meshes coincides
    with the patella's anterior prominence — the ISO §3.1.17 "Centre point
    of kneecap" landmark — within ~0.5 cm.  Mirrors what the numpy reference
    :func:`~clad_body.measure._circumferences.measure_knee` does in
    joint-anchored mode, just with sigmoid-gated edge crossings instead of
    ``trimesh.section`` so gradients flow through bone position (LBS) and
    vertex position (blendshapes + LBS).

    Left and right circumferences are averaged.  Same soft-circumference
    machinery (sigmoid gate + angular binning + r-biased softmax + convex
    hull perimeter), just applied via :func:`soft_circumference_plane`
    instead of the horizontal :func:`soft_circumference`.

    Args:
        model: Anny model (with ``_last_bone_heads`` / ``_last_bone_tails`` set).
        verts: (1, V, 3) raw Anny vertices in the model's native frame.

    Returns:
        dict with ``knee_cm`` as 0-dim torch tensor.

    Raises:
        RuntimeError: if ``model._last_bone_heads`` is missing — knee Z needs
            bone positions, so a forward pass with ``return_bone_ends=True``
            must run first (``measure_grad`` does this automatically).
    """
    edges_L = _build_leg_edges(model, "L").to(verts.device)
    edges_R = _build_leg_edges(model, "R").to(verts.device)

    origin_L, axis_L = _knee_axis_and_origin(model, "L")
    origin_R, axis_R = _knee_axis_and_origin(model, "R")
    if origin_L is None or origin_R is None:
        raise RuntimeError(
            "measure_knee_soft requires model._last_bone_heads / _tails — "
            "run a forward pass with return_bone_ends=True first."
        )

    circ_L = soft_circumference_plane(verts, edges_L, origin_L, axis_L)
    circ_R = soft_circumference_plane(verts, edges_R, origin_R, axis_R)

    raw_knee_cm = (circ_L + circ_R) / 2 * 100
    return {"knee_cm": KNEE_A * raw_knee_cm + KNEE_B}


def _build_leg_vertex_mask(model, side):
    """Single-leg vertex mask (dominantly skinned to that side's leg bones).

    Cached on ``model._soft_circ_leg_vmask_<side>``.

    Used by :func:`measure_calf_soft` to gate the radius-score Z resolver
    so only that side's lower-leg vertices contribute to the soft-argmax.

    Returns (V,) float tensor.
    """
    attr = f"_soft_circ_leg_vmask_{side}"
    cached = getattr(model, attr, None)
    if cached is not None:
        return cached

    leg_bones = LEFT_LEG_BONES if side == "L" else RIGHT_LEG_BONES
    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    dominant_bone = vbi[np.arange(vbw.shape[0]), np.argmax(vbw, axis=1)]
    mask = np.isin(dominant_bone, list(leg_bones)).astype(np.float32)
    result = torch.from_numpy(mask)
    setattr(model, attr, result)
    return result


def _calf_axis_origin_and_z_range(model, side, verts_zup):
    """Per-leg tibia axis, knee point (in mesh frame), and Z search bounds.

    Returns ``(knee_z_t, ankle_z_t, axis, knee_pt)`` where:
      - ``knee_z_t`` / ``ankle_z_t`` are scalar torch tensors (metres) in the
        ``verts_zup`` frame.
      - ``axis`` is a (3,) torch tensor (downward, unit-norm), in the raw
        verts frame (NOT zup) — used by ``soft_circumference_plane`` which
        operates in the model's native frame.
      - ``knee_pt`` is the (3,) knee point in the raw verts frame, used as
        the seed for ``origin = knee + t*tibia`` once the soft-resolved Z
        is known.

    Uses the same height-fraction projection as :func:`knee_z` to map bone
    Z into the ``verts_zup`` frame.
    """
    heads = getattr(model, "_last_bone_heads", None)
    tails = getattr(model, "_last_bone_tails", None)
    if heads is None or tails is None:
        return None

    labels = list(model.bone_labels)
    try:
        ul2 = labels.index(f"upperleg02.{side}")
        ll2 = labels.index(f"lowerleg02.{side}")
        ll1 = labels.index(f"lowerleg01.{side}")
    except ValueError:
        return None

    knee_pt = tails[0, ul2]                 # raw frame
    ankle_pt = tails[0, ll2]
    tibia = ankle_pt - heads[0, ll1]        # knee → ankle (down)
    t_n = torch.linalg.norm(tibia) + 1e-10
    axis = tibia / t_n

    # Project knee Z and ankle Z into the verts_zup frame using the same
    # fractional-height map :func:`bust_z` / :func:`knee_z` use.
    with torch.no_grad():
        all_pts = torch.cat([heads[0], tails[0]], dim=0)
        extents = all_pts.max(0).values - all_pts.min(0).values
        height_axis = int(extents.argmax().item())
    all_pts_diff = torch.cat([heads[0], tails[0]], dim=0)
    raw_min = all_pts_diff[:, height_axis].min()
    raw_range = all_pts_diff[:, height_axis].max() - raw_min
    mesh_height = verts_zup[0, :, 2].max()

    frac_knee = (knee_pt[height_axis] - raw_min) / (raw_range + 1e-10)
    frac_ankle = (ankle_pt[height_axis] - raw_min) / (raw_range + 1e-10)
    knee_z_t = frac_knee * mesh_height
    ankle_z_t = frac_ankle * mesh_height

    return knee_z_t, ankle_z_t, axis, knee_pt


def measure_calf_soft(model, verts):
    """Compute differentiable calf circumference (ISO 8559-1 §5.3.24).

    Mirrors numpy :func:`~clad_body.measure._circumferences.measure_calf`
    in joint-anchored perpendicular-slice mode.  Per-leg algorithm:

    1. Compute the search range in the ``verts_zup`` frame from the bone
       positions: ``z_min = ankle_z + 6 cm``, ``z_max = knee_z − 4 cm``
       (same offsets as the numpy reference).
    2. **Soft-argmax over Z.**  Pick a radius score for each lower-leg
       vertex: ``r = √((x − leg_centre_x)² + y²)`` measured in the
       verts_zup frame.  Wider lower leg = larger r.  Apply a hard Z-band
       mask (with ``CALF_Z_BUFFER`` guard) plus a per-leg vertex mask, and
       soft-argmax via ``softmax(r / τ)`` over the gated vertices.  The
       resulting ``z_calf`` tracks the gastrocnemius peak smoothly under
       phenotype changes.
    3. **One** :func:`soft_circumference_plane` call at the resolved Z,
       sliced perpendicular to the tibia axis.

    Mirrors the :func:`measure_stomach_soft` "soft-argmax then one
    soft_circumference" pattern, plus the per-leg + perpendicular-plane
    machinery from :func:`measure_knee_soft`.

    Left and right circumferences are averaged.

    Args:
        model: Anny model (with ``_last_bone_heads`` / ``_last_bone_tails``
            populated by :func:`measure_grad`).
        verts: (1, V, 3) raw Anny vertices in the model's native frame.

    Returns:
        dict with ``calf_cm`` as 0-dim torch tensor.

    Raises:
        RuntimeError: if ``model._last_bone_heads`` is missing — calf needs
            knee/ankle bones, so a forward pass with
            ``return_bone_ends=True`` must run first.
    """
    verts_zup = _to_zup(verts)
    v_zup = verts_zup[0]                    # (V, 3)

    circs = []
    for side in ("L", "R"):
        bones = _calf_axis_origin_and_z_range(model, side, verts_zup)
        if bones is None:
            raise RuntimeError(
                "measure_calf_soft requires model._last_bone_heads / _tails — "
                "run a forward pass with return_bone_ends=True first."
            )
        knee_z_t, ankle_z_t, axis, knee_pt = bones

        # Search range with 6 cm / 4 cm joint offsets (matches numpy)
        z_min = ankle_z_t + 0.06
        z_max = knee_z_t - 0.04

        # Per-leg vertex mask (lower-leg vertices for THIS side)
        leg_vmask = _build_leg_vertex_mask(model, side).to(verts_zup.device)

        # Hard Z-band mask with CALF_Z_BUFFER guard on each side
        with torch.no_grad():
            band = (
                (v_zup[:, 2] > z_min - CALF_Z_BUFFER)
                & (v_zup[:, 2] < z_max + CALF_Z_BUFFER)
            ).float()
        gate = leg_vmask * band             # (V,)

        # Per-leg axis centre (XY centroid of gated vertices).  Tracks
        # the leg position smoothly so spread = squared distance from
        # the centroid is translation-invariant.  Kept differentiable
        # through the centroid formula: when phenotype moves the leg
        # vertices, the centroid follows, so ``calf_cm`` gradient
        # captures the full response (a detached centroid would miss
        # ~12 % of the FD gradient on weight/muscle perturbations).
        w_sum = gate.sum() + 1e-10
        leg_cx = (v_zup[:, 0] * gate).sum() / w_sum
        leg_cy = (v_zup[:, 1] * gate).sum() / w_sum

        # Per-Z spread proxy → soft-argmax over Z bins.  The numpy
        # reference uses :func:`_two_leg_avg_circumference` to pick the
        # Z of maximum girth; we approximate the same shape function by
        # aggregating per-vertex squared distance from the leg axis into
        # Gaussian Z bins, then taking softmax over the per-bin sum.
        # This tracks the *circumference* peak rather than the
        # individual most-posterior vertex (which can be misaligned on
        # slim bodies where the Y peak and the girth peak diverge).
        #
        # The Z grid follows ``z_min`` / ``z_max`` differentiably (through
        # the knee/ankle bone Zs) so phenotype changes that move the
        # search range carry gradient through to ``calf_z``.  Detaching
        # the grid would break this path and gives autograd a
        # systematically-too-small gradient vs central FD (~12 % off).
        t = torch.linspace(
            0.0, 1.0, CALF_N_ZBINS,
            device=v_zup.device, dtype=v_zup.dtype,
        )
        z_grid = z_min + (z_max - z_min) * t
        sigma_z = (z_max - z_min) / (CALF_N_ZBINS - 1) * CALF_Z_BIN_SIGMA_FRAC
        # Gaussian assignment of each vertex to each Z bin: (V, n_zbins)
        z_diff = v_zup[:, 2:3] - z_grid.unsqueeze(0)
        z_assign = torch.exp(-(z_diff ** 2) / (2 * sigma_z ** 2 + 1e-12))
        z_assign = z_assign * gate.unsqueeze(1)             # (V, n_zbins)

        # Per-vertex spread proxy: squared distance from the leg axis.
        # Differentiable through verts.
        spread_v = (v_zup[:, 0] - leg_cx) ** 2 + (v_zup[:, 1] - leg_cy) ** 2

        # Per-bin spread = vertex-weighted mean of spread_v over the bin.
        spread_bin = (
            (spread_v.unsqueeze(1) * z_assign).sum(0)
            / (z_assign.sum(0) + 1e-10)
        )                                                   # (n_zbins,)

        # Soft-argmax over Z bins.  Spread is in m² (~1e-3 for a calf),
        # CALF_SOFTMAX_TAU sets how peaked the softmax is — small τ
        # picks the single bin with biggest spread, large τ averages.
        score_z = spread_bin / CALF_SOFTMAX_TAU
        weights_z = torch.softmax(score_z, dim=0)
        calf_z = (weights_z * z_grid).sum()

        # Resolve the slice origin in the raw (native) frame: we have the
        # axis there and the knee bone position there, but the soft-
        # resolved Z is in verts_zup.  Convert: knee_pt[height_axis] in raw
        # corresponds to knee_z_t in verts_zup; from soft calf_z we want
        # the matching raw-frame point on the tibia centre line.  The same
        # height-axis projection used in :func:`_calf_axis_origin_and_z_range`
        # is invertible: bone_z / mesh_height is the fractional height, and
        # we can solve ``knee_pt + s * tibia_dir`` so that its projected
        # fractional height matches the desired one.  Practically,
        # parameterise the tibia in the raw frame and solve for the same
        # fractional-height target as the soft Z.
        with torch.no_grad():
            heads = model._last_bone_heads
            tails = model._last_bone_tails
            all_pts = torch.cat([heads[0], tails[0]], dim=0)
            extents = all_pts.max(0).values - all_pts.min(0).values
            height_axis = int(extents.argmax().item())
            raw_min = all_pts[:, height_axis].min()
            raw_range = all_pts[:, height_axis].max() - raw_min
        mesh_height = verts_zup[0, :, 2].max()

        # Target fractional height in [0, 1] from the soft Z
        frac_target = calf_z / (mesh_height + 1e-10)
        # Knee fractional height in raw frame
        frac_knee = (knee_pt[height_axis] - raw_min) / (raw_range + 1e-10)
        # Tibia direction along height_axis in raw frame
        labels = list(model.bone_labels)
        ll1 = labels.index(f"lowerleg01.{side}")
        ll2 = labels.index(f"lowerleg02.{side}")
        tibia_raw = tails[0, ll2] - heads[0, ll1]
        tibia_dz = tibia_raw[height_axis]
        # Step along tibia so that the resulting Z fractional matches
        # frac_target.  Avoid division blow-up if tibia is degenerate.
        s = (frac_target - frac_knee) * raw_range / (tibia_dz + 1e-10)
        origin = knee_pt + s * tibia_raw

        edges = _build_leg_edges(model, side).to(verts.device)
        circ = soft_circumference_plane(verts, edges, origin, axis)
        circs.append(circ)

    raw_calf_cm = (circs[0] + circs[1]) / 2 * 100
    return {"calf_cm": CALF_A * raw_calf_cm + CALF_B}


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
