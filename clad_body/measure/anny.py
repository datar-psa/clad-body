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
    measure_inseam_from_perineum_vertices,
    measure_shirt_length,
    measure_shoulder_width,
    measure_sleeve_length,
    measure_sleeve_length_from_joints,
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
    # Shoulder bones are inside the body — use the bone TAIL (outer end of
    # upperarm01) for the correct lateral position; acromion is then found
    # via find_acromion() (max Z above the tail).
    "l_shoulder": [("upperarm01.L", "tail")],
    "r_shoulder": [("upperarm01.R", "tail")],
    # Ball joint = upperarm01 HEAD = the actual shoulder articulation point.
    # Used by measure_sleeve_length_from_joints. Distinct from "l_shoulder"
    # (which is the bone TAIL = mid-bicep) because the legacy find_acromion
    # path is calibrated against the tail-anchored bone position.
    "l_shoulder_ball": [("upperarm01.L", "head")],
    "r_shoulder_ball": [("upperarm01.R", "head")],
    "l_elbow":   ["lowerarm01.L"],
    "r_elbow":   ["lowerarm01.R"],
    "l_wrist":   ["wrist.L"],
    "r_wrist":   ["wrist.R"],
    # upperleg01 TAIL sits at the perineum / inner-thigh merge level — the
    # anatomical crotch. The HEAD is the femoral head (ball joint inside the
    # pelvis, ~8cm above the perineum). Tail matches the mesh-sweep crotch
    # within ~1–2cm across body types and is differentiable through LBS.
    "l_hip":     [("upperleg01.L", "tail")],
    "r_hip":     [("upperleg01.R", "tail")],
    # Knee = upperleg02 TAIL = lowerleg01 HEAD (same point in space, the knee
    # joint articulation). Used by measure_calf to bound the lower-leg sweep
    # below the patella, so a deflated calf doesn't let the kneecap region
    # (wider than the calf belly on tuned bodies) win the max-girth search.
    "l_knee":    [("upperleg02.L", "tail")],
    "r_knee":    [("upperleg02.R", "tail")],
    # Ankle = lowerleg02 TAIL = foot HEAD = ankle joint articulation.
    "l_ankle":   [("lowerleg02.L", "tail")],
    "r_ankle":   [("lowerleg02.R", "tail")],
}

# Arm/hand bone indices (shoulder01 through all fingers, both sides).
# Excludes clavicle (48, 74) which is at the torso boundary.
ARM_HAND_BONES = set(range(49, 74)) | set(range(75, 100))

# Leg/foot bone indices (upperleg01 through toes, excluding pelvis which is a
# torso-boundary bone). Used by soft-thigh to partition vertices into single-leg
# subsets so angular binning around one thigh doesn't mix in the other leg.
LEFT_LEG_BONES = set(range(2, 21))    # upperleg01.L (2) through toe5-3.L (20)
RIGHT_LEG_BONES = set(range(22, 41))  # upperleg01.R (22) through toe5-3.R (40)


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


# ─────────────────────────────────────────────────────────────────────────
# ISO 8559-1 sleeve length: slow reference (calibration only)
#
# This is the SLOW REFERENCE function used to calibrate the differentiable
# runtime function `measure_sleeve_length_from_joints` in _lengths.py.
#
# Architecture (mirrors `measure_inseam` / `measure_inseam_from_perineum_vertices`):
#   - measure_sleeve_length_iso_reference (this file): non-differentiable,
#     poses the body in rest pose with natural ~42° elbow flex, detects
#     acromion / olecranon / wrist styloid via skinning weights and bone
#     geometry, slices the body with two planes (upper-arm + forearm),
#     walks Dijkstra shortest paths along the resulting contours.
#     Used once per body during calibration.
#   - measure_sleeve_length_from_joints (_lengths.py): fast differentiable,
#     bone chain + linear correction calibrated against this reference.
#     Used in the gradient hot loop.
#
# Per ISO 8559-1 §3.1.1: shoulder point = "most lateral point of the
# lateral edge of the spine (acromial process) of the scapula, projected
# vertically to the surface of the skin."
# Per ISO §3.1.10: elbow point = "most prominent point of the olecranon
# of ulna" (the bony bump on the back of a flexed elbow).
# Per ISO §5.4.14/5.4.15: upper arm length and lower arm length are
# measured along the body surface with the elbow bent.
# ─────────────────────────────────────────────────────────────────────────


def _vertices_skinned_to(model, bone_indices, threshold=0.1):
    """Boolean (V,) mask: True if vertex has weight > threshold on any of the
    listed bone indices (taking the sum across the up-to-8 bone slots)."""
    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    weight_per_vertex = np.zeros(vbw.shape[0])
    for bi in bone_indices:
        slot_match = (vbi == bi)
        weight_per_vertex += np.where(slot_match, vbw, 0).sum(axis=1)
    return weight_per_vertex > threshold


def _detect_acromion_iso(model, verts, ball_pos, upperarm01_tail, side):
    """ISO acromion: surface vertex in a tube around the perpendicular ray
    from the ball joint, perpendicular to the upperarm01 bone, picking the
    HIGHEST Z within the tube.

    The ray geometry mirrors how a tape measurer projects the bony
    acromial process onto the skin: starting at the ball joint and going
    outward perpendicular to the humerus, the first point on the skin
    along that direction (constrained to be high) is the acromion.
    """
    labels = list(model.bone_labels)
    bone_idx = [labels.index(f"shoulder01.{side}"), labels.index(f"upperarm01.{side}")]
    skin_mask = _vertices_skinned_to(model, bone_idx, threshold=0.1)

    u = upperarm01_tail - ball_pos
    u = u / np.linalg.norm(u)

    # Lateral perpendicular direction (outward, perpendicular to bone)
    side_sign = +1 if side == "L" else -1
    world_lat = np.array([side_sign, 0.0, 0.0], dtype=verts.dtype)
    lateral = world_lat - (world_lat @ u) * u
    lateral = lateral / np.linalg.norm(lateral)

    candidates = verts[skin_mask]
    delta = candidates - ball_pos
    lateral_proj = delta @ lateral
    perp = delta - lateral_proj[:, None] * lateral
    perp_dist = np.linalg.norm(perp, axis=1)

    near_tube = (perp_dist < 0.015) & (lateral_proj > 0)
    if not near_tube.any():
        raise ValueError(f"No skinned vertices near acromion ray on {side} side")
    tube_verts = candidates[near_tube]
    return tube_verts[np.argmax(tube_verts[:, 2])]


def _detect_olecranon_iso(model, verts, ball_pos, elbow_pos, wrist_pos, side):
    """ISO olecranon: surface vertex within 5 cm of the elbow joint, furthest
    along the geometric back-of-elbow direction.

    The "back of bent elbow" direction is the bisector of the OUTSIDE of
    the bend: -unit(unit(ball-elbow) + unit(wrist-elbow)). This points
    posteriorly when the elbow is flexed and lands on the olecranon
    protrusion regardless of how much the elbow is bent.
    """
    labels = list(model.bone_labels)
    bone_idx = [labels.index(f"upperarm02.{side}"), labels.index(f"lowerarm01.{side}")]
    skin_mask = _vertices_skinned_to(model, bone_idx, threshold=0.1)
    near_mask = np.linalg.norm(verts - elbow_pos, axis=1) < 0.05
    mask = skin_mask & near_mask
    if not mask.any():
        raise ValueError(f"No vertices near elbow joint on {side} side")
    u = ball_pos - elbow_pos; u = u / np.linalg.norm(u)
    v = wrist_pos - elbow_pos; v = v / np.linalg.norm(v)
    back_dir = -(u + v)
    back_dir = back_dir / np.linalg.norm(back_dir)
    candidates = verts[mask]
    proj = (candidates - elbow_pos) @ back_dir
    return candidates[np.argmax(proj)]


def _detect_wrist_styloid_iso(model, verts, wrist_pos, side):
    """ISO wrist landmark (ulnar styloid approximation): most lateral surface
    vertex within a thin Z-slab around the wrist joint, restricted to
    lowerarm02-skinned verts (excludes hand verts distal to the wrist crease).
    """
    labels = list(model.bone_labels)
    bone_idx = [labels.index(f"lowerarm02.{side}")]
    skin_mask = _vertices_skinned_to(model, bone_idx, threshold=0.1)
    z_slab = np.abs(verts[:, 2] - wrist_pos[2]) < 0.008
    mask = skin_mask & z_slab
    if not mask.any():
        raise ValueError(f"No vertices in Z-slab around wrist joint on {side} side")
    candidates = verts[mask]
    if side == "L":
        return candidates[np.argmax(candidates[:, 0])]
    return candidates[np.argmin(candidates[:, 0])]


def _slice_and_walk(verts, faces, p_start, p_end, p_plane):
    """Plane-mesh slice between p_start and p_end, with the plane containing
    the third point p_plane (typically a bone joint to fix the orientation).
    Walks the shortest path along the contour from p_start to p_end.

    Returns (length_cm, polyline_vertices) or raises if no contour exists.
    """
    import trimesh
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    normal = np.cross(p_end - p_start, p_plane - p_start)
    normal = normal / np.linalg.norm(normal)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    lines = trimesh.intersections.mesh_plane(
        mesh, plane_normal=normal, plane_origin=p_start,
    )
    if len(lines) == 0:
        raise ValueError("Plane missed mesh")

    # Chain segments and run Dijkstra
    segments = lines.reshape(-1, 3)
    n_seg = len(lines)
    eps = 1e-5
    rounded = np.round(segments / eps).astype(np.int64)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    n_unique = inverse.max() + 1
    unique_pts = np.zeros((n_unique, 3))
    for i in range(len(segments)):
        unique_pts[inverse[i]] = segments[i]

    rows, cols, data = [], [], []
    for s in range(n_seg):
        a = int(inverse[2 * s])
        b = int(inverse[2 * s + 1])
        if a == b:
            continue
        w = float(np.linalg.norm(unique_pts[a] - unique_pts[b]))
        rows.append(a); cols.append(b); data.append(w)
        rows.append(b); cols.append(a); data.append(w)
    graph = csr_matrix((data, (rows, cols)), shape=(n_unique, n_unique))

    i_start = int(np.argmin(np.linalg.norm(unique_pts - p_start, axis=1)))
    i_end = int(np.argmin(np.linalg.norm(unique_pts - p_end, axis=1)))

    dist, pred = dijkstra(graph, indices=i_start, return_predecessors=True)
    if pred[i_end] == -9999:
        raise ValueError("No contour path between landmarks")
    path = [i_end]
    cur = i_end
    while cur != i_start:
        cur = int(pred[cur])
        path.append(cur)
    path.reverse()
    return float(dist[i_end]) * 100, unique_pts[path]


def measure_sleeve_length_iso_reference(body, side="L"):
    """Slow ISO 8559-1 sleeve length reference for calibration.

    Re-poses the body in REST POSE (lowerarm01 rotation 0° → natural ~42°
    elbow flex, the convention for "elbow bent" in ISO §5.4.14/5.4.15),
    detects acromion / olecranon / wrist styloid landmarks, slices the
    body with two planes (upper-arm bone + acromion + olecranon, forearm
    bone + olecranon + wrist styloid), and walks the shortest path along
    each contour.

    NOT differentiable. Used once per body during calibration of
    `measure_sleeve_length_from_joints` (the fast differentiable runtime).
    Costs ~1 second per body.

    Args:
        body:  AnnyBody from load_anny_from_params (the model is used; the
               body's vertices/joints are NOT used because we re-pose).
        side:  "L" or "R" — which arm to measure. Default left.

    Returns:
        dict with 'sleeve_length_cm', 'acromion', 'olecranon',
        'wrist_styloid', and 'path_vertices' (the polyline of the surface
        walk for visualization).
    """
    import torch

    model = body.model
    pheno = getattr(model, "_last_phenotype_kwargs", None)
    if pheno is None:
        raise ValueError("body.model has no _last_phenotype_kwargs; "
                         "use load_anny_from_params first")

    # Build a rest pose: lowerarm01 rotation = 0° (natural Anny rest position)
    n = model.bone_count
    # Anny models don't expose a single .parameters() iterator; pull device
    # from any tensor attribute we can find.
    device = torch.device("cpu")
    if hasattr(model, "vertex_bone_weights"):
        device = model.vertex_bone_weights.device
    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose = pose.unsqueeze(0).unsqueeze(0).expand(1, n, 4, 4).clone()
    # No rotation applied — rest pose is natural ~42° elbow flex

    # Re-run forward pass with rest pose (slow path: ~0.5-1s)
    local_changes = {}
    if hasattr(body, "phenotype_params") and body.phenotype_params:
        local_changes = body.phenotype_params.get("_local_changes", {}) or {}
    local_kwargs = {l: torch.tensor([v], dtype=torch.float32, device=device)
                    for l, v in local_changes.items()}
    with torch.no_grad():
        out = model(
            pose_parameters=pose,
            phenotype_kwargs=pheno,
            local_changes_kwargs=local_kwargs,
            pose_parameterization="root_relative_world",
            return_bone_ends=True,
        )

    verts_yup = out["vertices"][0].cpu().numpy()
    heads = out["bone_heads"][0].cpu().numpy()
    tails = out["bone_tails"][0].cpu().numpy()
    faces = (model.faces.detach().cpu().numpy()
             if hasattr(model.faces, "detach") else np.array(model.faces)).astype(np.int64)

    # Anny outputs are Z-up natively (verified by checking extents). Just
    # shift to floor and XY-centre to match the canonical clad-body frame.
    z_min = float(verts_yup[:, 2].min())
    verts = verts_yup.copy()
    verts[:, 2] -= z_min
    cxy = (verts[:, :2].max(0) + verts[:, :2].min(0)) / 2
    verts[:, 0] -= cxy[0]; verts[:, 1] -= cxy[1]
    heads = heads.copy(); heads[:, 2] -= z_min
    heads[:, 0] -= cxy[0]; heads[:, 1] -= cxy[1]
    tails = tails.copy(); tails[:, 2] -= z_min
    tails[:, 0] -= cxy[0]; tails[:, 1] -= cxy[1]

    labels = list(model.bone_labels)
    ball_pos = heads[labels.index(f"upperarm01.{side}")]
    upperarm01_tail = tails[labels.index(f"upperarm01.{side}")]
    elbow_pos = heads[labels.index(f"lowerarm01.{side}")]
    wrist_pos = heads[labels.index(f"wrist.{side}")]

    # Landmark detection
    acr = _detect_acromion_iso(model, verts, ball_pos, upperarm01_tail, side)
    olc = _detect_olecranon_iso(model, verts, ball_pos, elbow_pos, wrist_pos, side)
    wrs = _detect_wrist_styloid_iso(model, verts, wrist_pos, side)

    # Two-plane surface walk
    len_upper, walk_upper = _slice_and_walk(verts, faces, acr, olc, ball_pos)
    len_forearm, walk_forearm = _slice_and_walk(verts, faces, olc, wrs, elbow_pos)
    walked = np.vstack([walk_upper, walk_forearm[1:]])  # avoid duplicate olecranon
    total_cm = len_upper + len_forearm

    return {
        "sleeve_length_cm": total_cm,
        "acromion": acr,
        "olecranon": olc,
        "wrist_styloid": wrs,
        "path_vertices": walked,
    }


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
# Do NOT use these for circumference measurement — use plane sweep (reporting)
# or soft circumference (``measure_grad``) instead.
# Upperarm loop is reasonable as a differentiable proxy (< 1 cm vs plane sweep).
BASE_MESH_HIP_VERTICES = [4296, 4295, 4291, 4292, 4336, 4339, 4331, 4359, 10983, 10958, 10965, 10962, 10922, 10921, 10925, 10926, 10923, 10924, 10860, 10867, 10853, 10854, 4218, 4217, 4233, 4225, 4294, 4293]
BASE_MESH_UPPERARM_VERTICES = [3788, 1710, 1706, 1705, 1701, 1702, 1703, 1704, 1694, 1700, 1695, 1696, 3763, 1697, 1698, 1699, 1709, 1708, 1707, 3850]

# Shoulder width landmark seeds (base-mesh indices). The R/L acromion seeds
# are NOT used directly — they only anchor the k-ring patch over which the
# soft acromion does its Z-argmax (the actual argmax vertex drifts off the
# seed on broad-shoulder males). C7 IS used directly (stable on 100/100
# random bodies). See findings/shoulder_width_diff.md.
BASE_MESH_R_ACROMION_SEED = 1557
BASE_MESH_C7_SURFACE_SEED = 858
BASE_MESH_L_ACROMION_SEED = 8235

# Soft-acromion hyperparameters. K=2 covers 100 % of observed argmax landings
# on a 100-body sample. X_BAND mirrors the numpy ±5 cm hard band as a soft
# Gaussian. TAU is the Z softmax temperature — sharp enough to behave like
# a hard argmax at the optimum, soft enough to leak gradient to neighbours.
ACROMION_RING_K = 2
ACROMION_X_BAND = 0.018          # 18 mm Gaussian half-width
ACROMION_SOFTMAX_TAU = 0.010     # 10 mm Z softmax temperature


def remap_vertex_indices(model, base_mesh_indices):
    """Map base mesh vertex indices to reduced mesh indices.

    Anny's default model uses remove_unattached_vertices=True which reduces
    from 19,158 to 13,718 vertices. This remaps base mesh indices to the
    reduced mesh, same as anny.Anthropometry.__init__ does for waist vertices.
    """
    base_to_reduced = model.base_mesh_vertex_indices.detach().cpu().numpy().tolist()
    return [base_to_reduced.index(i) for i in base_mesh_indices]


def compute_k_ring(model, seed_reduced_idx, k):
    """Vertex indices within graph distance ≤ k from seed (reduced mesh).

    BFS over the face-adjacency graph. Topology-only — safe to cache on the
    model. Returns a sorted list of int indices, including the seed.
    """
    faces = model.faces.detach().cpu().numpy() if hasattr(model.faces, "detach") else np.asarray(model.faces)
    n_verts = int(faces.max()) + 1
    neigh = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        neigh[a].update((b, c))
        neigh[b].update((a, c))
        neigh[c].update((a, b))
    frontier = {int(seed_reduced_idx)}
    visited = {int(seed_reduced_idx)}
    for _ in range(k):
        nxt = set()
        for v in frontier:
            nxt |= neigh[v]
        nxt -= visited
        visited |= nxt
        frontier = nxt
    return sorted(visited)


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
    anthro.upperarm_vertex_indices = remap_vertex_indices(model, BASE_MESH_UPPERARM_VERTICES)

    # Shoulder-width landmarks: seeds + per-side k-ring patches for soft-argmax.
    r_seed, c7_seed, l_seed = remap_vertex_indices(
        model,
        [BASE_MESH_R_ACROMION_SEED,
         BASE_MESH_C7_SURFACE_SEED,
         BASE_MESH_L_ACROMION_SEED],
    )
    anthro.shoulder_c7_index = c7_seed  # C7 is stable — used directly
    anthro.shoulder_r_acromion_ring = compute_k_ring(model, r_seed, ACROMION_RING_K)
    anthro.shoulder_l_acromion_ring = compute_k_ring(model, l_seed, ACROMION_RING_K)
    # Bone indices for the per-body lateral X anchor (midpoint of upperarm01).
    bone_labels = list(model.bone_labels)
    anthro.shoulder_r_upperarm_bone = bone_labels.index("upperarm01.R")
    anthro.shoulder_l_upperarm_bone = bone_labels.index("upperarm01.L")

    model._extended_anthro = anthro
    return anthro


def compute_soft_acromion(verts, ring_indices, anchor_x,
                          x_band=ACROMION_X_BAND, tau=ACROMION_SOFTMAX_TAU):
    """Differentiable acromion landmark — soft equivalent of ``find_acromion``.

    Soft analogue of "highest Z within a lateral band around the shoulder":

        log_w = Z / tau − (X − anchor_x)² / (2 · x_band²)

    Returns the softmax-weighted average position over ``ring_indices``.
    The Gaussian X term excludes medial vertices on the trapezius hump
    (which outrank the bony shoulder tip on Z alone) without a hard mask.

    Args:
        verts: (1, V, 3) tensor — axis 0 = lateral X, axis 2 = Z.
        ring_indices: reduced-mesh vertex indices for the search patch.
        anchor_x: (1, 1) tensor — lateral anchor (typically the upperarm01
            bone-midpoint X, so the band drifts via LBS with body shape).
    """
    candidates = verts[:, ring_indices]
    z = candidates[:, :, 2]
    x = candidates[:, :, 0]
    log_w = z / tau - ((x - anchor_x) ** 2) / (2.0 * x_band ** 2)
    weights = torch.softmax(log_w, dim=-1)
    return torch.sum(weights.unsqueeze(-1) * candidates, dim=1)


def compute_shoulder_arc_length(verts, anthro, model, n_samples=30,
                                x_band=ACROMION_X_BAND,
                                tau=ACROMION_SOFTMAX_TAU):
    """ISO 8559-1 §5.4.2 shoulder arc length, fully differentiable.

    Soft acromions on each side + cached C7 vertex + quadratic curve through
    (R, C7, L), sampled at ``n_samples`` points; segment norms summed.
    Per-side X anchor = midpoint of ``upperarm01.{head, tail}`` (the lateral
    shoulder edge — both endpoints move via LBS with shoulder breadth).

    ``model._last_bone_heads`` / ``_last_bone_tails`` must be populated by
    the immediately preceding forward pass (``return_bone_ends=True``).
    """
    heads = model._last_bone_heads[0]
    tails = model._last_bone_tails[0]
    r_anchor = 0.5 * (heads[anthro.shoulder_r_upperarm_bone, 0:1]
                      + tails[anthro.shoulder_r_upperarm_bone, 0:1]).unsqueeze(0)
    l_anchor = 0.5 * (heads[anthro.shoulder_l_upperarm_bone, 0:1]
                      + tails[anthro.shoulder_l_upperarm_bone, 0:1]).unsqueeze(0)

    P_r = compute_soft_acromion(verts, anthro.shoulder_r_acromion_ring,
                                r_anchor, x_band=x_band, tau=tau)
    P_l = compute_soft_acromion(verts, anthro.shoulder_l_acromion_ring,
                                l_anchor, x_band=x_band, tau=tau)
    P_c7 = verts[:, anthro.shoulder_c7_index]

    # Quadratic through (R at t=0, C7 at t=0.5, L at t=1) — closed form.
    a = 2.0 * P_r + 2.0 * P_l - 4.0 * P_c7
    b = 4.0 * P_c7 - 3.0 * P_r - P_l
    c = P_r
    t = torch.linspace(0.0, 1.0, n_samples, dtype=verts.dtype, device=verts.device)
    t2 = t.unsqueeze(-1)
    curve = a.unsqueeze(1) * (t2 ** 2) + b.unsqueeze(1) * t2 + c.unsqueeze(1)
    segs = curve[:, 1:] - curve[:, :-1]
    return torch.sum(torch.linalg.norm(segs, dim=-1), dim=-1).squeeze(0)


def compute_loop_circumference(verts, vertex_indices):
    """Compute circumference of an ordered vertex loop (differentiable).

    Same algorithm as anny.Anthropometry.waist_circumference():
    vertices at given indices form a closed loop, sum of edge lengths.

    Used by measure_grad for waist (46-vertex anatomical loop) and upperarm
    (20 vertices, < 1 cm error). For reporting, use plane sweep
    (``measure()``); for differentiable thigh/bust/hip/underbust use the
    soft circumference path in ``_soft_circ.py``.

    Args:
        verts: (1, V, 3) tensor — full mesh vertices
        vertex_indices: list of int — ordered vertex indices forming closed loop
    Returns:
        scalar tensor — circumference in meters
    """
    loop_verts = verts[:, vertex_indices]
    loop_rolled = torch.roll(loop_verts, shifts=1, dims=1)
    return torch.sum(torch.linalg.norm(loop_rolled - loop_verts, dim=-1), dim=-1)


def _extract_anny_joints(model, as_tensor=False):
    """Extract canonical joint positions from Anny model after forward pass.

    Uses bone_heads from the last forward pass (stashed on model by
    load_anny_from_params / load_anny_from_verts). Falls back to template_bone_heads
    if no forward pass output is available (numpy path only).

    For shoulder joints (l_shoulder, r_shoulder), uses bone TAILS (end of
    upperarm01) instead of heads. The tail of upperarm01 sits at the
    outer end of the upper arm bone, closer to the actual shoulder tip.

    Args:
        model: Anny rigged model with ``_last_bone_heads`` / ``_last_bone_tails``
            stashed from the most recent forward pass.
        as_tensor: If ``True``, return (3,) torch tensors that preserve autograd
            history from the forward pass that produced the bone data.  The
            Y-up → Z-up coordinate swap and floor-alignment are performed
            differentiably so that gradients flow through Z positions.
            Requires ``model._last_bone_heads`` to be present (no template
            fallback — templates have no live gradient).
            Default ``False`` preserves the existing numpy behaviour.

    Returns:
        dict mapping canonical joint names to (3,) arrays/tensors (Z-up, metres),
        or empty dict if bone data unavailable.
    """
    if as_tensor:
        # ── Torch path: preserves autograd history ──────────────────────────
        bone_heads = getattr(model, "_last_bone_heads", None)
        bone_tails = getattr(model, "_last_bone_tails", None)
        if bone_heads is None:
            raise ValueError(
                "model._last_bone_heads is missing — the model must be run with "
                "return_bone_ends=True before calling measure_grad(). "
                "Use load_anny_from_params() which does this automatically."
            )

        heads = bone_heads[0]  # (n_joints, 3) — grad flows from here
        tails = bone_tails[0] if bone_tails is not None else None

        # Detect height axis under no_grad — result is a Python int, not tracked
        with torch.no_grad():
            extents = heads.max(dim=0).values - heads.min(dim=0).values
            height_axis = int(extents.argmax().item())

        if height_axis == 1:  # Y-up → Z-up: [x, y, z] → [x, z, -y]
            heads = torch.stack([heads[:, 0], heads[:, 2], -heads[:, 1]], dim=-1)
            if tails is not None:
                tails = torch.stack([tails[:, 0], tails[:, 2], -tails[:, 1]], dim=-1)

        # Floor at Z=0 (differentiable — z_min carries gradient through Z positions)
        z_min = heads[:, 2].min()
        heads = torch.stack([heads[:, 0], heads[:, 1], heads[:, 2] - z_min], dim=-1)
        if tails is not None:
            tails = torch.stack([tails[:, 0], tails[:, 1], tails[:, 2] - z_min], dim=-1)

        bone_labels = list(model.bone_labels) if hasattr(model, "bone_labels") else []
        if not bone_labels:
            return {}

        # Same extraction logic as extract_joints_from_names but with torch tensors
        name_to_idx = {name: i for i, name in enumerate(bone_labels)}
        result = {}
        for canon_name, candidates in ANNY_JOINT_MAP.items():
            for cand in candidates:
                if isinstance(cand, str):
                    bone, mode = cand, "head"
                else:
                    bone, mode = cand
                if bone not in name_to_idx:
                    continue
                i = name_to_idx[bone]
                if mode == "head":
                    result[canon_name] = heads[i]
                elif mode == "tail" and tails is not None:
                    result[canon_name] = tails[i]
                elif mode == "midpoint" and tails is not None:
                    result[canon_name] = 0.5 * (heads[i] + tails[i])
                else:
                    continue
                break
        return result

    # ── Numpy path (existing behaviour) ─────────────────────────────────────
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

        knee_cm, knee_z, knee_pct = measure_knee(mesh_tri, height, joints=joints)
        measurements["knee_cm"] = knee_cm
        measurements["_knee_z"] = knee_z
        measurements["_knee_pct"] = knee_pct

        calf_cm, calf_z, calf_pct = measure_calf(mesh_tri, height, joints=joints)
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
        # Reporting path: ISO 8559-1 mesh sweep (accurate, non-differentiable).
        # For gradient-based optimisation use measure_grad() which calls
        # measure_inseam_from_perineum_vertices() — a differentiable vertex-pair
        # proxy that tracks this sweep within ~0.2 cm.
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
        # ISO 8559-1 sleeve length: slow surface-walk reference (~1 s per body).
        # For gradient-based optimisation use measure_grad() which calls
        # measure_sleeve_length_from_joints() — a differentiable approximation
        # calibrated against this reference.
        sleeve_ref = measure_sleeve_length_iso_reference(body)
        measurements["sleeve_length_cm"] = sleeve_ref["sleeve_length_cm"]

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


# ── Differentiable measurements ───────────────────────────────────────────────

#: Keys supported by :func:`measure_grad`.  All have fully differentiable
#: implementations that compose existing building blocks.  Callers can inspect
#: this set to check support before requesting a key.
SUPPORTED_KEYS = frozenset({
    "height_cm",
    "bust_cm",
    "underbust_cm",
    "waist_cm",
    "stomach_cm",
    "hip_cm",
    "thigh_cm",
    "knee_cm",
    "calf_cm",
    "upperarm_cm",
    "shoulder_width_cm",
    "inseam_cm",
    "sleeve_length_cm",
    "neck_cm",
    "mass_kg",
})


def measure_grad(body, *, pose=None, only=None):
    """Differentiable Anny measurements for autograd-based mesh optimisation.

    Runs a fresh forward pass with the body's phenotype kwargs and returns
    measurements as torch tensors with autograd history preserved.  Gradients
    flow from the returned tensors back to any tensor in
    ``body.phenotype_kwargs`` / ``body.local_changes_kwargs`` that has
    ``requires_grad=True``.

    Companion to :func:`clad_body.measure.measure`.  Same input (an
    :class:`~clad_body.load.anny.AnnyBody`) and same key naming — you can swap
    between the two APIs with no translation.  Only a subset of keys have
    differentiable implementations; requesting an unsupported key raises
    :exc:`ValueError` rather than silently falling back to numpy (which would
    break gradient flow without warning).

    Example — optimisation loop::

        from clad_body.load.anny import load_anny_from_params
        from clad_body.measure import measure_grad
        import torch

        body = load_anny_from_params(initial_params, requires_grad=True)
        optimizer = torch.optim.Adam(list(body.phenotype_kwargs.values()), lr=0.01)

        for step in range(500):
            optimizer.zero_grad()
            m = measure_grad(body, only=["waist_cm", "inseam_cm"])
            loss = (m["waist_cm"] - 78.0) ** 2 + (m["inseam_cm"] - 82.0) ** 2
            loss.backward()
            optimizer.step()

    Args:
        body: :class:`~clad_body.load.anny.AnnyBody` from
            :func:`~clad_body.load.anny.load_anny_from_params`.  Its
            ``phenotype_kwargs`` and ``local_changes_kwargs`` are used as the
            input to a fresh forward pass on ``body.model``.  Pass
            ``requires_grad=True`` to ``load_anny_from_params`` to enable
            gradients on all tensors, or call
            ``body.phenotype_kwargs[label].requires_grad_(True)`` per-tensor.
        pose: optional ``(1, n_joints, 4, 4)`` pose tensor.  Defaults to Anny
            A-pose via :func:`~clad_body.load.anny.build_anny_apose`.
        only: list of measurement keys to compute.  ``None`` means all
            supported keys.

    Returns:
        dict mapping measurement key → 0-dim torch tensor (cm) with autograd
        history preserved.

    Raises:
        ValueError: if ``only`` contains a key not in :data:`SUPPORTED_KEYS`,
            or if ``body.phenotype_kwargs`` is missing (body wasn't created
            via :func:`load_anny_from_params`).

    Supported keys (Anny):
        ``height_cm``, ``bust_cm``, ``underbust_cm``, ``waist_cm``,
        ``stomach_cm``, ``hip_cm``, ``thigh_cm``, ``knee_cm``,
        ``calf_cm``, ``upperarm_cm``, ``shoulder_width_cm``,
        ``inseam_cm``, ``sleeve_length_cm``, ``neck_cm``, ``mass_kg``.

    Note on shoulder_width_cm:
        Soft-argmax acromions in a Gaussian X-band anchored at the
        upperarm01 bone midpoint, plus a closed-form quadratic curve
        through (R, C7, L). RMS 1.4 cm and R²=0.96 vs the numpy ISO
        reference on 100 random bodies, and SMOOTHER than the reference
        on parameter sweeps (no argmax jumps). See
        findings/shoulder_width_diff.md.

    Note on mass_kg:
        Computed as ``volume(verts) × _MEDIAN_DENSITY[gender]`` where
        ``_MEDIAN_DENSITY = {"male": 1059, "female": 1031}`` (kg/m³,
        population-median tissue density from hydrostatic weighing).
        This is intentionally simpler than the value returned by
        :func:`measure`, which uses a body-fat-corrected density derived
        from neck/waist/hip/height (non-differentiable).  The trade-off:
        ``measure_grad`` mass cannot reflect body composition variance
        within a gender, but it stays differentiable end-to-end and
        tracks real human mass at the population median (within ~2 kg
        for normal-composition adults).  ``measure`` remains the more
        accurate static reporting value.

    Note:
        MHR is not yet supported.  API and supported keys may change between
        minor versions while this is under active development.
    """
    from clad_body.load.anny import AnnyBody, build_anny_apose

    if not isinstance(body, AnnyBody):
        raise TypeError(
            f"measure_grad expects an AnnyBody, got {type(body).__name__}"
        )

    if body.phenotype_kwargs is None:
        raise ValueError(
            "body.phenotype_kwargs is missing — create the body via "
            "load_anny_from_params() so the torch tensors needed for "
            "measure_grad() are populated."
        )

    # Validate keys before paying for the forward pass
    requested = frozenset(SUPPORTED_KEYS) if only is None else frozenset(only)
    unsupported = requested - SUPPORTED_KEYS
    if unsupported:
        raise ValueError(
            f"Key(s) {sorted(unsupported)!r} are not differentiable. "
            f"Supported keys: {sorted(SUPPORTED_KEYS)}"
        )

    model = body.model
    device = next(iter(body.phenotype_kwargs.values())).device

    if pose is None:
        pose = build_anny_apose(model, device)

    # Forward pass — keep the computation graph (no torch.no_grad)
    output = model(
        pose_parameters=pose,
        phenotype_kwargs=body.phenotype_kwargs,
        local_changes_kwargs=body.local_changes_kwargs or {},
        pose_parameterization="root_relative_world",
        return_bone_ends=True,
    )
    model._last_bone_heads = output["bone_heads"]
    model._last_bone_tails = output["bone_tails"]

    return _measure_grad_from_verts(model, output["vertices"], requested=requested)


def _measure_grad_from_verts(model, verts, *, requested):
    """Compute differentiable measurements from a verts tensor.

    Internal — assumes ``model._last_bone_heads`` and ``_last_bone_tails`` are
    already populated by the caller (typically :func:`measure_grad` itself),
    and that ``requested`` is a validated frozenset of keys.
    """
    result = {}

    # Vertex loop indices (cached on model after first call — topology-only)
    anthro = setup_extended_anthro(model)

    # Detect height axis once under no_grad — returns a Python int, not tracked
    with torch.no_grad():
        extents = verts[0].max(dim=0).values - verts[0].min(dim=0).values
        height_axis = int(extents.argmax().item())

    # ── height_cm ────────────────────────────────────────────────────────────
    if "height_cm" in requested:
        col = verts[:, :, height_axis]
        result["height_cm"] = (col.max() - col.min()) * 100

    # ── bust_cm / underbust_cm (soft circumference) ─────────────────────────
    if "bust_cm" in requested or "underbust_cm" in requested:
        from ._soft_circ import measure_bust_underbust
        bu = measure_bust_underbust(model, verts)
        if "bust_cm" in requested and "bust_cm" in bu:
            result["bust_cm"] = bu["bust_cm"]
        if "underbust_cm" in requested:
            result["underbust_cm"] = bu["underbust_cm"]

    # ── hip_cm (soft circumference) ──────────────────────────────────────
    if "hip_cm" in requested:
        from ._soft_circ import measure_hip
        hp = measure_hip(model, verts)
        result["hip_cm"] = hp["hip_cm"]

    # ── waist_cm ─────────────────────────────────────────────────────────────
    if "waist_cm" in requested:
        result["waist_cm"] = (
            compute_loop_circumference(verts, anthro.waist_vertex_indices).squeeze(0) * 100
        )

    # ── stomach_cm (soft max over Z range between hip and waist) ────────────
    if "stomach_cm" in requested:
        from ._soft_circ import measure_stomach_soft
        st = measure_stomach_soft(model, verts)
        result["stomach_cm"] = st["stomach_cm"]

    # ── thigh_cm (soft circumference, per-leg edges) ────────────────────────
    if "thigh_cm" in requested:
        from ._soft_circ import measure_thigh_soft
        th = measure_thigh_soft(model, verts)
        result["thigh_cm"] = th["thigh_cm"]

    # ── knee_cm (soft circumference, per-leg edges, bone-anchored Z) ────────
    if "knee_cm" in requested:
        from ._soft_circ import measure_knee_soft
        kn = measure_knee_soft(model, verts)
        result["knee_cm"] = kn["knee_cm"]

    # ── calf_cm (soft-argmax over Z + perpendicular plane at that Z) ────────
    if "calf_cm" in requested:
        from ._soft_circ import measure_calf_soft
        cf = measure_calf_soft(model, verts)
        result["calf_cm"] = cf["calf_cm"]

    # ── upperarm_cm (also required as input for sleeve_length) ───────────────
    upperarm_loop_cm = None
    if "upperarm_cm" in requested or "sleeve_length_cm" in requested:
        upperarm_loop_cm = (
            compute_loop_circumference(verts, anthro.upperarm_vertex_indices).squeeze(0) * 100
        )
        if "upperarm_cm" in requested:
            result["upperarm_cm"] = upperarm_loop_cm

    # ── shoulder_width_cm — soft-argmax acromions + quadratic arc ────────────
    if "shoulder_width_cm" in requested:
        result["shoulder_width_cm"] = (
            compute_shoulder_arc_length(verts, anthro, model) * 100
        )

    # ── inseam_cm — perineum vertex pair (no joints needed) ──────────────────
    if "inseam_cm" in requested:
        result["inseam_cm"] = measure_inseam_from_perineum_vertices(verts, height_axis)

    # ── sleeve_length_cm (needs differentiable joint tensors) ────────────────
    if "sleeve_length_cm" in requested:
        joints_torch = _extract_anny_joints(model, as_tensor=True)
        result["sleeve_length_cm"] = measure_sleeve_length_from_joints(
            joints_torch, upperarm_loop_cm
        )

    # ── neck_cm (tilted soft circumference perpendicular to neck axis) ───────
    if "neck_cm" in requested:
        from ._soft_circ import measure_neck_soft
        nk = measure_neck_soft(model, verts)
        result["neck_cm"] = nk["neck_cm"]

    # ── mass_kg ───────────────────────────────────────────────────────────────
    # Plain V × ρ_median(gender) — fully differentiable. Differs from measure()
    # which uses BF-corrected density (depends on neck/waist/hip → non-diff).
    # See docstring "Note on mass_kg" for the rationale.
    if "mass_kg" in requested:
        gender_str = _infer_gender(model, verts)
        density = _MEDIAN_DENSITY.get(gender_str, 1045.0)
        volume_m3 = anthro.volume(verts).squeeze(0)
        result["mass_kg"] = volume_m3 * density

    return result


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
