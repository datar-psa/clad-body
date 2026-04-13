"""Differentiable soft circumference for bust and underbust.

Edge-plane intersection with sigmoid gates, angular binning with r-biased
softmax, recentered polar coordinates, and convex hull perimeter.

The polar origin is the weighted centroid of crossing points (not the mesh
XY origin). This eliminates empty back bins caused by Anny's D-shaped
cross-section where the XY origin sits near the posterior surface.

The convex hull of the resulting polygon gives the tape-measure circumference
(bridges concavities like the breast cleavage). Hull vertex selection is
non-differentiable but gradients flow through the selected vertex positions.

Calibration (100-body dataset, random seed 42):
    bust:      A = 0.9997, B = 0.12   (MAE 0.06 cm, max 0.18 cm)
    underbust: A = 0.9830, B = 1.86   (MAE 0.39 cm, max 1.61 cm)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import ConvexHull

from .anny import ARM_HAND_BONES, BREAST_BONES

# ── Hyperparameters (validated on 100-body sweep) ────────────────────────────

N_BINS = 72
SIGMA_Z = 0.005       # metres — soft gate width for edge dz
TAU = 0.050           # metres — sigmoid gate width + softmax temperature

# ── Calibration coefficients (recenter + convex hull, 100-body fit) ──────────

BUST_A, BUST_B = 0.9997, 0.12
UB_A, UB_B = 0.9830, 1.86


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
