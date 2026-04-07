#!/usr/bin/env python3
"""
Measure Anny body mesh — extract anthropometric measurements.

Uses mesh-plane sweep (ISO 8559-1 compliant) for bust, waist, hip, thigh, and
upper arm circumferences with convex hull (tape-measure simulation).

Usage:
    # Measure from phenotype params JSON
    python -m clad_body.measure.anny path/to/anny_params.json

    # Measure a sample body
    python -m clad_body.measure.anny m_average
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

try:
    import anny
    import torch
except ImportError as _e:
    raise ImportError(
        "clad_body.measure.anny requires the [anny] extra: pip install 'clad-body[anny]'"
    ) from _e

from clad_body.measure._slicer import MAX_TORSO_X_EXTENT, torso_circumference_at_z
from clad_body.measure._circumferences import (
    _front_y_at_z,
    measure_calf,
    measure_knee,
    measure_neck,
    measure_stomach,
    measure_thigh,
    measure_upperarm,
    measure_wrist,
    torso_sweep_bust_hips,
)
from clad_body.measure._lengths import (
    c7_surface_point,
    extract_linear_measurement_polylines,
    measure_back_neck_to_waist,
    measure_crotch_length,
    measure_inseam,
    measure_shirt_length,
    measure_shoulder_width,
    measure_sleeve_length,
)
from clad_body.measure._render import extract_measurement_contours, render_4view
from clad_body.measure._lengths import extract_joints_from_names
from clad_body.measure._render import (
    find_target_json,
    load_target_measurements,
    print_comparison,
)

# Canonical joint name mapping for Anny linear measurements.
# Maps standard names to Anny bone name candidates.
# NOTE: Shoulder bones are INSIDE the body. upperarm01 HEAD gives the correct
# lateral (X) position for the shoulder/arm junction. The actual acromion
# (bony shoulder tip) is found by find_acromion() (max Z above bone tail).
ANNY_JOINT_MAP = {
    # neck01.head (~83.7% height) is the T1 joint — too low for C7.
    # The midpoint of neck01 lands at ~84.6%, matching the anatomical
    # C7 spinous process (84–85% of stature).
    "c7":        [("neck01", "midpoint")],
    "neck_base": ["neck01"],       # base of neck (~84% height)
    "neck_mid":  ["neck02"],       # neck01 tail = neck02 head (~86%, Adam's apple)
    "head":      ["head"],         # top of neck / base of skull
    "side_neck": ["shoulder01.L", "shoulder01.R"],  # clavicle tail (~82%)
    # Shoulder bones are inside the body — use the bone TAIL (outer end of
    # upperarm01) for the correct lateral position; acromion is then found
    # via find_acromion() (max Z above the tail).
    "l_shoulder": [("upperarm01.L", "tail")],
    "r_shoulder": [("upperarm01.R", "tail")],
    "l_elbow":   ["lowerarm01.L"],
    "r_elbow":   ["lowerarm01.R"],
    "l_wrist":   ["wrist.L"],
    "r_wrist":   ["wrist.R"],
}

# Arm/hand bone indices (shoulder01 through all fingers, both sides).
# Excludes clavicle (48, 74) which is at the torso boundary.
ARM_HAND_BONES = set(range(49, 74)) | set(range(75, 100))


# ── Body fat & density estimation ──
# Navy formula (Hodgdon & Beckett, 1984) for normal range,
# Weltman equations (1987/1988) for obese bodies.
# See findings/feature_impact_and_density.md for full research.

def estimate_body_fat_pct(height_cm, waist_cm, hip_cm, neck_cm,
                          mass_kg, gender):
    """Estimate body fat % from circumference measurements.

    Uses the Hodgdon & Beckett (1984) density equations (cm inputs) with the
    Siri transform to get BF%. Switches to Weltman equations (1987/1988) for
    obese bodies where the Navy formula underestimates fat.

    IMPORTANT: The Hodgdon density formulas use cm. The commonly-cited direct
    BF% formulas (86.010 × log10(waist-neck)...) use INCHES — using those
    with cm inputs gives wildly wrong results. We use the density form:
        D = f(log10(circumferences_cm), log10(height_cm))
        BF% = 495/D - 450  (Siri equation)

    Navy formula validated against DXA: SEE ±3.5% in normal range.
    Weltman validated against hydrostatic weighing: SEE ±2.9% in obese range.

    Args:
        height_cm: body height
        waist_cm: waist circumference (Anny vertex loop or narrowest point)
        hip_cm: hip circumference
        neck_cm: neck circumference (minimum in neck region)
        mass_kg: body mass (Anny volume × 980)
        gender: "male" or "female"

    Returns:
        Estimated body fat percentage (clipped to 3-60% range).
    """
    if gender == "male":
        diff = waist_cm - neck_cm
        if diff <= 0:
            return 3.0
        # Hodgdon & Beckett density formula (cm inputs)
        density = (1.0324
                   - 0.19077 * math.log10(diff)
                   + 0.15456 * math.log10(height_cm))
        navy_bf = 495.0 / density - 450.0
        # Switch to Weltman for obese males (Navy underestimates above ~28% BF)
        if navy_bf > 28:
            bf = 0.31457 * waist_cm - 0.10969 * mass_kg + 10.8336
        else:
            bf = navy_bf
    else:
        diff = waist_cm + hip_cm - neck_cm
        if diff <= 0:
            return 3.0
        # Hodgdon & Beckett density formula (cm inputs)
        density = (1.29579
                   - 0.35004 * math.log10(diff)
                   + 0.22100 * math.log10(height_cm))
        navy_bf = 495.0 / density - 450.0
        # Switch to Weltman for obese females (Navy underestimates above ~38% BF)
        if navy_bf > 38:
            bf = (0.11077 * waist_cm - 0.17666 * height_cm
                  + 0.14354 * mass_kg + 51.03301)
        else:
            bf = navy_bf

    return max(3.0, min(60.0, bf))


def body_density_from_bf(bf_pct):
    """Tissue-only body density (kg/m³) from body fat percentage.

    Uses the Siri two-component model (1961):
        density = 1 / (BF/d_fat + (1-BF)/d_ffm)

    Returns density in kg/m³ (e.g. 1039 for 15% BF).

    Note on conventions: this is *tissue-only* density (the value hydrostatic
    weighing reports after subtracting residual lung volume), not whole-body
    density including lung air. Whole-body density averages ~985 kg/m³ (just
    below water, why humans barely float); tissue-only density is ~1030–1080
    depending on body composition. Siri's constants 900 (fat) and 1100 (FFM)
    come from cadaver dissection and are tissue-only. The Anny default
    980 kg/m³ used elsewhere sits between these two conventions — see
    questionnaire/findings/feature_impact_and_density.md for the rationale
    and the open question about which volume convention Anny's mesh follows.
    """
    bf = bf_pct / 100.0
    bf = max(0.03, min(0.60, bf))
    d_fat = 900.0    # kg/m³ — Siri (tissue-only, from cadaver studies)
    d_ffm = 1100.0   # kg/m³ — Siri (tissue-only, from cadaver studies)
    return 1.0 / (bf / d_fat + (1.0 - bf) / d_ffm)


# Median tissue-only density per gender from our sampling distribution
# (200-sample validation, stable across seeds). These match published
# hydrostatic-weighing values for adults with normal body composition
# (PubMed 12400035: ~1060 male, ~1042 female for low-BF Japanese cohort).
#
# IMPORTANT: these are *tissue-only* densities (lung air subtracted), while
# Anny's 980 kg/m³ default is closer to *whole-body* density (lung air
# included). The two conventions differ by ~50 kg/m³. The density correction
# below assumes Anny's mesh volume is consistent with the tissue-only
# convention — which is unverified. Empirically the correction works (lean
# bodies get more mass, soft bodies get less), but the absolute scale rests
# on the 980 calibration being correct for the "average" subject.
_MEDIAN_DENSITY = {"male": 1059, "female": 1031}


def density_corrected_mass(mass_kg, estimated_density, gender):
    """Correct Anny mass for body composition differences across builds.

    Anny uses fixed 980 kg/m³ which gives correct mass for average bodies.
    This function adjusts mass relative to the population median density so
    that athletic bodies (denser) get slightly more mass and soft bodies
    (less dense) get slightly less — without shifting the average.

    Typical corrections: -2 to +1 kg (centered around zero for average build).

    Args:
        mass_kg: Anny mass (volume × 980)
        estimated_density: from body_density_from_bf() (kg/m³)
        gender: "male" or "female"

    Returns:
        Corrected mass in kg.
    """
    ref = _MEDIAN_DENSITY.get(gender, 1045)
    return mass_kg * (estimated_density / ref)


def _infer_gender(model, verts):
    """Infer gender string from Anny model's last phenotype kwargs.

    Falls back to mesh heuristic (hip-to-shoulder ratio) if not available.
    """
    # Check if model has stored phenotype kwargs from last forward pass
    last_pheno = getattr(model, '_last_phenotype_kwargs', None)
    if last_pheno and 'gender' in last_pheno:
        g = last_pheno['gender']
        if hasattr(g, 'item'):
            g = g.item()
        return "female" if g > 0.5 else "male"

    # Fallback: use mesh shape heuristic
    v = verts[0].detach().cpu().numpy() if hasattr(verts, 'detach') else verts
    height = v[:, 2].max()
    # Hip width at ~50% height vs shoulder width at ~80% height
    hip_z = height * 0.50
    shoulder_z = height * 0.80
    hip_mask = np.abs(v[:, 2] - hip_z) < 0.02
    shoulder_mask = np.abs(v[:, 2] - shoulder_z) < 0.02
    hip_xs = v[hip_mask, 0]
    shoulder_xs = v[shoulder_mask, 0]
    hip_width = float(hip_xs.max() - hip_xs.min()) if hip_mask.any() else 0
    shoulder_width = float(shoulder_xs.max() - shoulder_xs.min()) if shoulder_mask.any() else 0
    if shoulder_width > 0:
        return "female" if hip_width / shoulder_width > 0.95 else "male"
    return "male"


def find_acromion(verts, shoulder_joint, side="left"):
    """Find acromion for Anny — highest surface point near bone tail.

    Anny shoulder joints use bone tails (end of upperarm01), which are
    already near the lateral shoulder edge. The acromion is the highest
    surface point (max Z) within ±5cm X of the bone, above the bone Z.

    Args:
        verts: (V, 3) vertex array (Z-up, metres, floor-aligned)
        shoulder_joint: (3,) bone tail position (near shoulder surface)
        side: "left" or "right"
    """
    x = shoulder_joint[0]
    z = shoulder_joint[2]
    mask = (
        (np.abs(verts[:, 0] - x) < 0.05) &
        (verts[:, 2] >= z) &
        (verts[:, 2] < z + 0.15)
    )
    candidates = verts[mask]
    if len(candidates) == 0:
        return shoulder_joint.copy()
    return candidates[np.argmax(candidates[:, 2])].copy()

# Breast bone indices (breast.L=45, breast.R=46 in Anny skeleton).
BREAST_BONES = {45, 46}


def build_arm_mask(model):
    """Build boolean mask of arm/hand vertices from skinning weights.

    Returns (V,) numpy bool array — True for arm/hand vertices.
    """
    vbw = model.vertex_bone_weights.detach().cpu()
    vbi = model.vertex_bone_indices.detach().cpu()
    dominant_k = vbw.argmax(dim=1)
    dominant_bone = vbi[torch.arange(len(vbi)), dominant_k].numpy()

    arm_mask = np.isin(dominant_bone, list(ARM_HAND_BONES))
    return arm_mask


def build_torso_mesh(mesh_tri, arm_mask):
    """Create torso-only trimesh by excluding faces with any arm/hand vertex."""
    faces = np.array(mesh_tri.faces)
    face_has_arm = arm_mask[faces].any(axis=1)
    torso_faces = faces[~face_has_arm]
    return trimesh.Trimesh(
        vertices=np.array(mesh_tri.vertices),
        faces=torso_faces,
        process=False,
    )


def breast_floor_z(model, mesh_verts, weight_threshold=0.3):
    """Find the Z of the inframammary fold from breast bone skinning weights.

    Returns the minimum Z of vertices with breast bone weight > threshold,
    which marks the bottom of the breast tissue (the inframammary fold).
    Uses the Z-up, floor-aligned mesh_verts (from _anny_to_trimesh).

    Returns Z in metres, or None if no breast vertices found.
    """
    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    breast_weight = np.zeros(vbw.shape[0])
    for bone_idx in BREAST_BONES:
        mask = (vbi == bone_idx)
        breast_weight += np.where(mask, vbw, 0).sum(axis=1)
    sig = breast_weight > weight_threshold
    if not sig.any():
        return None
    return float(mesh_verts[sig, 2].min())


# Vertex loops (base mesh indices). Found via find_vertex_loops.py + curate_loops.py.
# WARNING: Hip and bust loops are NOT accurate circumference tracers — they don't
# form clean anatomical rings. They ARE used for Z-height anchoring only (the mean Z
# of each loop gives a reliable anatomical landmark at ~75% for bust, ~52% for hips).
# Do NOT use these for circumference measurement — use plane sweep instead.
# Thigh and upperarm loops are used by tuning_anny.py for differentiable optimization.
BASE_MESH_HIP_VERTICES = [4296, 4295, 4291, 4292, 4336, 4339, 4331, 4359, 10983, 10958, 10965, 10962, 10922, 10921, 10925, 10926, 10923, 10924, 10860, 10867, 10853, 10854, 4218, 4217, 4233, 4225, 4294, 4293]
BASE_MESH_BUST_VERTICES = [1445, 1438, 1855, 1888, 1762, 1798, 8470, 8434, 8560, 8527, 8126, 8133, 8315, 8313, 8365, 8241, 8247, 8253, 1575, 1569, 1563, 1693, 1641, 1643]
BASE_MESH_THIGH_VERTICES = [6745, 6744, 6743, 6742, 6741, 6740, 6739, 6738, 6737, 6736, 6755, 6754, 6753, 6752, 6751, 6750, 6749, 6748, 6747, 6746]
BASE_MESH_UPPERARM_VERTICES = [3788, 1710, 1706, 1705, 1701, 1702, 1703, 1704, 1694, 1700, 1695, 1696, 3763, 1697, 1698, 1699, 1709, 1708, 1707, 3850]


def remap_vertex_indices(model, base_mesh_indices):
    """Map base mesh vertex indices to reduced mesh indices.

    Anny's default model uses remove_unattached_vertices=True which reduces
    from 19,158 to 13,718 vertices. This remaps base mesh indices to the
    reduced mesh, same as anny.Anthropometry.__init__ does for waist vertices.
    """
    base_to_reduced = model.base_mesh_vertex_indices.detach().cpu().numpy().tolist()
    return [base_to_reduced.index(i) for i in base_mesh_indices]


def setup_extended_anthro(model):
    """Create Anthropometry instance with all custom vertex loops attached.

    Cached on ``model._extended_anthro`` — the result depends only on model
    topology (vertex indices, bone hierarchy) which is constant across forward
    passes with different phenotype params.
    """
    cached = getattr(model, "_extended_anthro", None)
    if cached is not None:
        return cached
    anthro = anny.Anthropometry(model)
    anthro.hip_vertex_indices = remap_vertex_indices(model, BASE_MESH_HIP_VERTICES)
    anthro.bust_vertex_indices = remap_vertex_indices(model, BASE_MESH_BUST_VERTICES)
    anthro.thigh_vertex_indices = remap_vertex_indices(model, BASE_MESH_THIGH_VERTICES)
    anthro.upperarm_vertex_indices = remap_vertex_indices(model, BASE_MESH_UPPERARM_VERTICES)
    model._extended_anthro = anthro
    return anthro


def compute_loop_circumference(verts, vertex_indices):
    """Compute circumference of an ordered vertex loop (differentiable).

    Same algorithm as anny.Anthropometry.waist_circumference():
    vertices at given indices form a closed loop, sum of edge lengths.

    Used by tuning_anny.py for gradient-based optimization (waist, thigh).
    For measurement reporting, use plane sweep instead (measure_thigh/measure_upperarm).

    Args:
        verts: (1, V, 3) tensor — full mesh vertices
        vertex_indices: list of int — ordered vertex indices forming closed loop
    Returns:
        scalar tensor — circumference in meters
    """
    loop_verts = verts[:, vertex_indices]
    loop_rolled = torch.roll(loop_verts, shifts=1, dims=1)
    return torch.sum(torch.linalg.norm(loop_rolled - loop_verts, dim=-1), dim=-1)


def _extract_anny_joints(model):
    """Extract canonical joint positions from Anny model after forward pass.

    Uses bone_heads from the last forward pass (stashed on model by
    load_anny_from_params / load_anny_from_verts). Falls back to template_bone_heads
    if no forward pass output is available.

    For shoulder joints (l_shoulder, r_shoulder), uses bone TAILS (end of
    upperarm01) instead of heads. The tail of upperarm01 sits at the
    outer end of the upper arm bone, closer to the actual shoulder tip.

    Returns:
        dict mapping canonical joint names to (3,) numpy arrays (Z-up, metres),
        or empty dict if bone data unavailable.
    """
    bone_heads = getattr(model, "_last_bone_heads", None)
    bone_tails = getattr(model, "_last_bone_tails", None)
    if bone_heads is not None:
        # Posed bone heads: (1, n_joints, 3)
        heads_np = bone_heads[0].detach().cpu().numpy()
    elif hasattr(model, "template_bone_heads"):
        heads_np = model.template_bone_heads.detach().cpu().numpy()
    else:
        return {}

    # Also get tails if available
    tails_np = None
    if bone_tails is not None:
        tails_np = bone_tails[0].detach().cpu().numpy()
    elif hasattr(model, "template_bone_tails"):
        tails_np = model.template_bone_tails.detach().cpu().numpy()

    # Apply same coordinate transform as _anny_to_trimesh: Y-up → Z-up, feet at Z=0
    extents = heads_np.max(axis=0) - heads_np.min(axis=0)
    height_axis = int(np.argmax(extents))
    if height_axis == 1:  # Y-up → Z-up
        heads_np = heads_np[:, [0, 2, 1]].copy()
        heads_np[:, 2] = -heads_np[:, 2]
        if tails_np is not None:
            tails_np = tails_np[:, [0, 2, 1]].copy()
            tails_np[:, 2] = -tails_np[:, 2]
    # Feet at Z=0 (always — matches _anny_to_trimesh)
    z_min = heads_np[:, 2].min()
    heads_np = heads_np.copy()
    heads_np[:, 2] -= z_min
    if tails_np is not None:
        tails_np = tails_np.copy()
        tails_np[:, 2] -= z_min

    bone_labels = list(model.bone_labels) if hasattr(model, "bone_labels") else []
    if not bone_labels:
        return {}

    return extract_joints_from_names(bone_labels, heads_np, ANNY_JOINT_MAP, tails=tails_np)


def _anny_to_trimesh(verts_tensor, model):
    """Convert Anny model output to Z-up trimesh with feet at Z=0."""
    verts = verts_tensor[0].detach().cpu().numpy()
    faces = model.faces.detach().cpu().numpy() if hasattr(model.faces, 'detach') else np.array(model.faces)

    # Detect orientation: height on axis with max extent
    extents = verts.max(axis=0) - verts.min(axis=0)
    height_axis = int(np.argmax(extents))
    if height_axis == 1:  # Y-up → Z-up
        verts = verts[:, [0, 2, 1]]
        verts[:, 2] = -verts[:, 2]

    # Feet at Z=0
    verts[:, 2] -= verts[:, 2].min()

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _breast_prominence_z(model, mesh_verts):
    """Get bust prominence Z from breast bone tail positions.

    The breast.L / breast.R bone tails point to the nipple/bust prominence —
    a better anatomical anchor than the vertex loop (which sits ~2-3cm higher).
    Converts bone tail height to a fraction of skeleton height, then maps to
    the trimesh Z coordinate (Z-up, feet at Z=0).
    """
    tails = getattr(model, '_last_bone_tails', None)
    heads = getattr(model, '_last_bone_heads', None)
    if tails is None or heads is None:
        return None
    tails_np = tails[0].detach().cpu().numpy()
    heads_np = heads[0].detach().cpu().numpy()
    labels = model.bone_labels
    try:
        bl = labels.index("breast.L")
        br = labels.index("breast.R")
    except ValueError:
        return None

    # Find height axis in raw Anny space (axis with largest bone extent)
    all_pts = np.vstack([heads_np, tails_np])
    height_axis = int(np.argmax(all_pts.max(0) - all_pts.min(0)))
    raw_min = all_pts[:, height_axis].min()
    raw_range = all_pts[:, height_axis].max() - raw_min

    # Breast tail height as fraction of skeleton height
    frac_l = (tails_np[bl, height_axis] - raw_min) / raw_range
    frac_r = (tails_np[br, height_axis] - raw_min) / raw_range

    # Map to trimesh Z (feet at 0, head at mesh_height)
    mesh_height = mesh_verts[:, 2].max()
    return float((frac_l + frac_r) / 2 * mesh_height)


def load_phenotype_params(params_path: str) -> dict:
    """Load phenotype params from JSON file."""
    with open(params_path) as f:
        return json.load(f)


def _measure_anny(body, *, groups, render_path=None, title="", device=None):
    """Internal: measure an AnnyBody with selective computation groups.

    Called by ``clad_body.measure.measure()``. Do not call directly.

    Uses ``body.mesh`` (cached trimesh) and ``body._model`` (Anny rigged
    model from :func:`load_anny_from_params`) to avoid redundant model
    creation and forward passes.  Falls back to creating a model from
    ``phenotype_params`` if ``body._model`` is not set.

    Args:
        device: ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    """
    from clad_body.measure import (
        GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_E, GROUP_F, GROUP_G, GROUP_H,
    )
    if body.phenotype_params is None:
        raise ValueError(
            "AnnyBody.phenotype_params is required for measurement. "
            "Use load_anny_from_params() to create the body."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Use body.mesh (cached trimesh from AnnyBody)
    mesh_tri = body.mesh
    mesh_verts = body.vertices
    height = mesh_verts[:, 2].max()

    model = body.model

    # (1, V, 3) torch tensor for anthro methods (waist circ, volume, mass).
    # Distances are coordinate-system invariant so Z-up works fine.
    verts = torch.from_numpy(body.vertices).unsqueeze(0).float().to(device)

    anthro = setup_extended_anthro(model)
    arm_mask = build_arm_mask(model)
    torso_mesh = build_torso_mesh(mesh_tri, arm_mask)

    measurements = {"height_cm": float(height * 100)}

    # ── Group A: Core torso ──────────────────────────────────────────────
    if GROUP_A in groups:
        waist_cm = anthro.waist_circumference(verts).item() * 100
        waist_z = float(mesh_verts[anthro.waist_vertex_indices, 2].mean())
        bust_anchor_z = _breast_prominence_z(model, mesh_verts)
        hip_anchor_z = float(mesh_verts[anthro.hip_vertex_indices, 2].mean())

        bust_cm, bust_z, hip_cm, hip_z = torso_sweep_bust_hips(
            mesh_tri, torso_mesh, waist_z, height,
            bust_anchor_z=bust_anchor_z, hip_anchor_z=hip_anchor_z)

        measurements["hip_cm"] = hip_cm
        measurements["_hip_z"] = hip_z
        measurements["_hip_pct"] = (hip_z / height * 100) if hip_z > 0 else 0
        measurements["bust_cm"] = bust_cm
        measurements["_bust_z"] = bust_z
        measurements["_bust_pct"] = (bust_z / height * 100) if bust_z > 0 else 0
        measurements["waist_cm"] = waist_cm
        measurements["_waist_z"] = waist_z
        measurements["_waist_pct"] = waist_z / height * 100

        stomach_cm, stomach_z, stomach_pct, belly_front_y = measure_stomach(
            torso_mesh, waist_z, hip_anchor_z, height)
        measurements["stomach_cm"] = stomach_cm
        measurements["_stomach_z"] = stomach_z
        measurements["_stomach_pct"] = stomach_pct
        measurements["_belly_front_y"] = belly_front_y

        fold_z = breast_floor_z(model, mesh_verts)
        if fold_z is not None and fold_z > 0:
            underbust_z = fold_z
            underbust_cm = torso_circumference_at_z(
                torso_mesh, underbust_z, max_x_extent=MAX_TORSO_X_EXTENT,
                combine_fragments=True) * 100
        else:
            underbust_z, underbust_cm = 0.0, 0.0
        measurements["underbust_cm"] = underbust_cm
        measurements["_underbust_z"] = underbust_z
        measurements["_underbust_pct"] = (underbust_z / height * 100) if underbust_z > 0 else 0

        belly_depth_cm = 0.0
        if belly_front_y is not None and underbust_z > 0:
            ub_front_y = _front_y_at_z(torso_mesh, underbust_z)
            if ub_front_y is not None:
                belly_depth_cm = (belly_front_y - ub_front_y) * 100
        measurements["belly_depth_cm"] = belly_depth_cm

    # Joints (needed by groups C, D, E, F).
    # _extract_anny_joints produces Z-up, feet-at-0 joints but does NOT
    # XY-center them.  body.vertices IS XY-centered (reposition_apose).
    # Apply the stored XY offset from load_anny_from_params.
    joints = _extract_anny_joints(model)
    if joints:
        xy_offset = getattr(body, '_xy_offset', None)
        if xy_offset is not None and np.any(np.abs(xy_offset) > 1e-6):
            for name in joints:
                joints[name] = joints[name].copy()
                joints[name][:2] += xy_offset

    # ── Group B: Limb sweeps ─────────────────────────────────────────────
    if GROUP_B in groups:
        thigh_cm, thigh_z, thigh_pct = measure_thigh(mesh_tri, height)
        measurements["thigh_cm"] = thigh_cm
        measurements["_thigh_z"] = thigh_z
        measurements["_thigh_pct"] = thigh_pct

        knee_cm, knee_z, knee_pct = measure_knee(mesh_tri, height)
        measurements["knee_cm"] = knee_cm
        measurements["_knee_z"] = knee_z
        measurements["_knee_pct"] = knee_pct

        calf_cm, calf_z, calf_pct = measure_calf(mesh_tri, height)
        measurements["calf_cm"] = calf_cm
        measurements["_calf_z"] = calf_z
        measurements["_calf_pct"] = calf_pct

        upperarm_cm, upperarm_z, upperarm_pct = measure_upperarm(mesh_tri, height)
        measurements["upperarm_cm"] = upperarm_cm
        measurements["_upperarm_z"] = upperarm_z
        measurements["_upperarm_pct"] = upperarm_pct

        wrist_cm, wrist_z, wrist_pct = measure_wrist(mesh_tri, height, joints=joints)
        measurements["wrist_cm"] = wrist_cm
        measurements["_wrist_z"] = wrist_z
        measurements["_wrist_pct"] = wrist_pct

    # ── Group D: Perpendicular (neck) ────────────────────────────────────
    if GROUP_D in groups:
        neck_cm, neck_z, neck_pct, neck_pts = measure_neck(
            mesh_tri, height, joints=joints)
        measurements["neck_cm"] = neck_cm
        measurements["_neck_z"] = neck_z
        measurements["_neck_pct"] = neck_pct
        if neck_pts is not None:
            measurements["_neck_contour_pts"] = neck_pts

    # ── Group E: Mesh geometry (inseam, crotch) ──────────────────────────
    if GROUP_E in groups:
        inseam_cm, inseam_z, inseam_pct = measure_inseam(mesh_tri, height)
        measurements["inseam_cm"] = inseam_cm
        measurements["_inseam_z"] = inseam_z
        measurements["_inseam_pct"] = inseam_pct

        crotch_len, front_rise, back_rise, crotch_f_pts, crotch_b_pts = \
            measure_crotch_length(
                mesh_tri, height,
                measurements.get("_waist_z", 0), inseam_z)
        measurements["crotch_length_cm"] = crotch_len
        measurements["front_rise_cm"] = front_rise
        measurements["back_rise_cm"] = back_rise
        if crotch_f_pts is not None:
            measurements["_crotch_front_pts"] = crotch_f_pts
        if crotch_b_pts is not None:
            measurements["_crotch_back_pts"] = crotch_b_pts

    # ── Group C: Joint linear (shoulder, sleeve) ─────────────────────────
    if GROUP_C in groups:
        c7 = joints.get("c7")
        if c7 is not None:
            measurements["_c7_surface_pt"] = c7_surface_point(
                np.array(mesh_tri.vertices), c7)
        sw_cm, sw_arc = measure_shoulder_width(
            joints, mesh=mesh_tri, acromion_fn=find_acromion)
        measurements["shoulder_width_cm"] = sw_cm
        if sw_arc is not None:
            measurements["_shoulder_arc_pts"] = sw_arc
        measurements["sleeve_length_cm"] = measure_sleeve_length(
            joints, mesh=mesh_tri, acromion_fn=find_acromion)

    # ── Group F: Surface trace (shirt length) ────────────────────────────
    if GROUP_F in groups:
        shirt_cm, shirt_pts = measure_shirt_length(
            joints, mesh_tri, measurements.get("_inseam_z", 0),
            measurements=measurements)
        measurements["shirt_length_cm"] = shirt_cm
        if shirt_pts is not None:
            measurements["_shirt_length_pts"] = shirt_pts

    # ── Group H: Back neck to waist (ISO 5.4.5) ──────────────────────────
    if GROUP_H in groups:
        bnw_cm, bnw_pts = measure_back_neck_to_waist(
            joints, mesh_tri, measurements.get("_waist_z", 0),
            c7_surface=measurements.get("_c7_surface_pt"))
        measurements["back_neck_to_waist_cm"] = bnw_cm
        if bnw_pts is not None:
            measurements["_back_neck_to_waist_pts"] = bnw_pts

    # ── Group G: Body composition ────────────────────────────────────────
    if GROUP_G in groups:
        measurements["volume_m3"] = anthro.volume(verts).item()
        anny_mass = anthro.mass(verts).item()
        measurements["_anny_mass_kg"] = anny_mass  # V×980 (internal)
        measurements["mass_kg"] = anny_mass  # default; overridden by V×ρ below
        measurements["bmi"] = anthro.bmi(verts).item()

        neck = measurements.get("neck_cm", 0)
        if neck > 0:
            gender_str = _infer_gender(model, verts)
            bf_pct = estimate_body_fat_pct(
                height_cm=measurements["height_cm"],
                waist_cm=measurements.get("waist_cm", 0),
                hip_cm=measurements.get("hip_cm", 0),
                neck_cm=neck,
                mass_kg=anny_mass,
                gender=gender_str,
            )
            measurements["body_fat_pct"] = bf_pct
            dens = body_density_from_bf(bf_pct)
            measurements["estimated_density"] = dens
            measurements["density_corrected_mass_kg"] = density_corrected_mass(
                anny_mass, dens, gender_str)
            measurements["mass_kg"] = measurements["volume_m3"] * dens  # V×ρ

    # ── Visualization ────────────────────────────────────────────────────
    # Polylines + contours when we have enough data
    has_linear = GROUP_C in groups or GROUP_E in groups or GROUP_H in groups
    if has_linear:
        measurements["_linear_polylines"] = extract_linear_measurement_polylines(
            mesh_tri, measurements, joints)
    measurements["mesh"] = mesh_tri
    measurements["contours"] = extract_measurement_contours(
        mesh_tri, measurements, torso_mesh=torso_mesh)

    measurements["_mesh_tri"] = mesh_tri
    measurements["_torso_mesh"] = torso_mesh
    measurements["_anthro"] = anthro

    if render_path:
        render_4view(mesh_tri, measurements, render_path,
                     title=title, model_label="Anny", torso_mesh=torso_mesh)

    return measurements


def main():
    parser = argparse.ArgumentParser(
        description="Measure Anny body mesh from phenotype parameters"
    )
    parser.add_argument(
        "input",
        help="Path to anny_params.json or directory containing one"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate 4-view render with measurement contours"
    )
    parser.add_argument(
        "--target", "-t", default=None,
        help="Path to target_measurements.json. Auto-detected if next to params."
    )

    args = parser.parse_args()

    # Resolve params: JSON file or directory containing anny_params.json
    input_path = Path(args.input)

    if input_path.suffix == ".json" and input_path.exists():
        params_dict = load_phenotype_params(input_path)
        source_name = input_path.parent.name
        body_dir = str(input_path.parent)
    elif input_path.is_dir():
        params_json = input_path / "anny_params.json"
        if not params_json.exists():
            print(f"Error: No anny_params.json found in {input_path}")
            sys.exit(1)
        params_dict = load_phenotype_params(str(params_json))
        source_name = input_path.name
        body_dir = str(input_path)
    else:
        print(f"Error: Input must be a params JSON file or directory: {args.input}")
        sys.exit(1)

    print(f"Measuring body: {source_name}")
    print(f"\nPhenotype parameters:")
    for key in ['gender', 'age', 'muscle', 'weight', 'height', 'proportions',
                'cupsize', 'firmness', 'african', 'asian', 'caucasian']:
        if key in params_dict:
            print(f"  {key:>14s}: {params_dict[key]:.4f}")

    # Resolve render path (before measuring, so we can pass it in)
    render_path = None
    if args.plot:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results", source_name)
        os.makedirs(out_dir, exist_ok=True)
        render_path = os.path.join(out_dir, "4view_anny_base_measurements.png")

    # Measure (+ optional 4-view render)
    print(f"\nGenerating mesh and measuring (plane sweep, 2mm steps)...")
    from clad_body.load.anny import load_anny_from_params
    from clad_body.measure import measure
    body = load_anny_from_params(params_dict)
    measurements = measure(body, render_path=render_path, title=source_name)

    # Load target measurements (explicit or auto-detected)
    target_path = args.target
    if target_path is None and body_dir:
        target_path = find_target_json(body_dir)

    if target_path:
        target = load_target_measurements(target_path)
        print(f"\nTarget: {target_path}")
        print_comparison(measurements, target)
    else:
        print(f"\n=== Body Measurements ===")
        print(f"Height:        {measurements['height_cm']:>6.1f} cm")
        print(f"Bust:          {measurements['bust_cm']:>6.1f} cm  (at {measurements['_bust_pct']:.0f}% height)")
        print(f"Waist:         {measurements['waist_cm']:>6.1f} cm  (at {measurements['_waist_pct']:.0f}% height)")
        print(f"Hips:          {measurements['hip_cm']:>6.1f} cm  (at {measurements['_hip_pct']:.0f}% height)")
        if measurements.get("stomach_cm", 0) > 0:
            print(f"Stomach:       {measurements['stomach_cm']:>6.1f} cm  (at {measurements['_stomach_pct']:.0f}% height)")
        if measurements.get("thigh_cm", 0) > 0:
            print(f"Thigh:         {measurements['thigh_cm']:>6.1f} cm  (at {measurements['_thigh_pct']:.0f}% height)")
        if measurements.get("upperarm_cm", 0) > 0:
            print(f"Upper arm:     {measurements['upperarm_cm']:>6.1f} cm  (at {measurements['_upperarm_pct']:.0f}% height)")
        if measurements.get("neck_cm", 0) > 0:
            print(f"Neck:          {measurements['neck_cm']:>6.1f} cm  (at {measurements['_neck_pct']:.0f}% height)")
        if measurements.get("shoulder_width_cm", 0) > 0:
            print(f"Shoulder W:    {measurements['shoulder_width_cm']:>6.1f} cm")
        if measurements.get("sleeve_length_cm", 0) > 0:
            print(f"Sleeve len:    {measurements['sleeve_length_cm']:>6.1f} cm")
        if measurements.get("inseam_cm", 0) > 0:
            print(f"Inseam:        {measurements['inseam_cm']:>6.1f} cm  (crotch at {measurements['_inseam_pct']:.0f}% height)")
        if measurements.get("crotch_length_cm", 0) > 0:
            print(f"Crotch len:    {measurements['crotch_length_cm']:>6.1f} cm  (front {measurements['front_rise_cm']:.1f} + back {measurements['back_rise_cm']:.1f})")
        print(f"Mass:          {measurements['mass_kg']:>6.1f} kg")
        print(f"BMI:           {measurements['bmi']:>6.2f}")
        print(f"Volume:        {measurements['volume_m3']:>6.4f} m³")
        if "body_fat_pct" in measurements:
            print(f"Body fat:      {measurements['body_fat_pct']:>6.1f} %")
            print(f"Density:       {measurements['estimated_density']:>6.0f} kg/m³")
        if body_dir:
            print(f"\nTip: Create {os.path.join(body_dir, 'target_measurements.json')} to compare.")

    return measurements


if __name__ == "__main__":
    main()
