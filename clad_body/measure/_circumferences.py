"""Circumference measurements — torso sweep, limb, neck, and related helpers.

All circumference measurements follow ISO 8559-1 conventions with convex hull
(tape-measure simulation).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from ._slicer import (
    CONTOUR_COLORS,
    MAX_TORSO_X_EXTENT,
    REGIONS,
    MeshSlicer,
    _find_contour_centroids_at_z,
    _perpendicular_limb_contour,
    torso_circumference_at_z,
)


def torso_sweep_bust_hips(full_mesh, torso_mesh, waist_z, height,
                          bust_anchor_z=None, hip_anchor_z=None):
    """Plane sweep for bust and hip circumferences.

    Uses torso-only mesh for bust (arms overlap at bust level in A-pose)
    and full mesh for hips (arms are far above hip level, no interference).
    For hips, uses a larger x_extent threshold (0.60m) since plus-size bodies
    can have hip width approaching/exceeding the default 0.40m filter.

    When bust_anchor_z / hip_anchor_z are provided (from vertex loop positions),
    the search is narrowed to ±3cm around the anatomical landmark. This prevents
    the belly from pulling measurements to the wrong height on large-bellied bodies.

    Returns (bust_cm, bust_z, hip_cm, hip_z).
    """
    if bust_anchor_z is not None:
        # Narrow search around anatomical landmark (vertex loop Z)
        bust_lo = max(bust_anchor_z - 0.03, height * 0.68)
        bust_hi = min(bust_anchor_z + 0.03, height * 0.80)
    else:
        bust_lo = max(waist_z, height * 0.68)
        bust_hi = min(waist_z + 0.30, height * 0.80)

    if hip_anchor_z is not None:
        hip_lo = max(hip_anchor_z - 0.03, height * 0.40)
        hip_hi = min(hip_anchor_z + 0.03, height * 0.56)
    else:
        hip_lo = max(waist_z - 0.25, height * 0.40)
        hip_hi = min(waist_z, height * 0.56)

    bust_cm, bust_z = 0.0, 0.0
    hip_cm, hip_z = 0.0, 0.0

    # Hips: use FULL mesh with relaxed x_extent filter.
    # At hip level (~50% height), arms are far above (elbows ~60%, shoulders ~75%).
    hip_zs = np.arange(hip_lo, hip_hi, 0.002)
    if len(hip_zs) > 0:
        hip_slicer = MeshSlicer(full_mesh)
        hip_circs = np.array([
            hip_slicer.circumference_at_z(z, max_x_extent=0.60) for z in hip_zs
        ])
        hip_valid = hip_circs > 0.30
        if hip_valid.any():
            idx = np.argmax(np.where(hip_valid, hip_circs, -1))
            hip_cm = hip_circs[idx] * 100
            hip_z = hip_zs[idx]

    # Bust: use torso-only mesh (arms overlap at bust level in A-pose).
    # ISO 8559-1 §5.3.4: measure at bust prominence level, NOT max circumference.
    bust_slicer = MeshSlicer(torso_mesh)
    if bust_anchor_z is not None:
        # Anatomical anchor available — measure directly at bust prominence Z.
        circ = bust_slicer.circumference_at_z(bust_anchor_z,
                                              combine_fragments=True)
        if circ > 0.30:
            bust_cm = circ * 100
            bust_z = bust_anchor_z
    else:
        # No anchor — fall back to sweep for max in the anatomical region.
        bust_zs = np.arange(bust_lo, bust_hi, 0.002)
        if len(bust_zs) > 0:
            bust_circs = np.array([
                bust_slicer.circumference_at_z(z, combine_fragments=True)
                for z in bust_zs
            ])
            bust_valid = bust_circs > 0.30
            if bust_valid.any():
                idx = np.argmax(np.where(bust_valid, bust_circs, -1))
                bust_cm = bust_circs[idx] * 100
                bust_z = bust_zs[idx]

    return bust_cm, bust_z, hip_cm, hip_z


def _front_y_at_z(torso_mesh, z):
    """Most anterior Y (metres) from mesh cross-section contour at height *z*."""
    _, pts = torso_circumference_at_z(
        torso_mesh, z, max_x_extent=0.60, return_contour=True,
        combine_fragments=True)
    if pts is not None and len(pts) > 0:
        return float(pts[:, 1].min())  # min Y = most anterior
    return None


def measure_stomach(torso_mesh, waist_z, hip_anchor_z, height):
    """Measure stomach circumference: max torso circ between waist and hips.

    Scans the torso-only mesh from hip_anchor_z up to waist_z and returns the
    maximum circumference found.  On bodies without belly prominence this equals
    approximately the waist circumference.  On bodies with a belly the maximum
    sits below the waist at the point of greatest anterior protrusion.

    Args:
        torso_mesh: torso-only trimesh (arms excluded)
        waist_z: Z height of the natural waist (metres)
        hip_anchor_z: Z height of the hip anchor landmark (metres)
        height: total body height (metres)

    Returns:
        (stomach_cm, stomach_z, stomach_pct, belly_front_y) —
        circumference in cm, Z height in metres, percentage of body height,
        and the most anterior Y coordinate (metres) at belly level.
        belly_front_y can be compared against front-Y at another level
        (e.g. underbust) to determine belly prominence.
        Returns (0, 0, 0, None) if no valid contour found.
    """
    # 3DLOOK: "around the maximum anterior protrusion of the abdomen."
    # Strategy: fast vertex band scan to find where the front protrudes most,
    # then one contour slice for the circumference measurement.
    lo = hip_anchor_z
    hi = waist_z
    if lo >= hi:
        return 0.0, 0.0, 0.0, None

    zs = np.arange(lo, hi, 0.002)  # 2mm steps (vertex bands are cheap)
    if len(zs) == 0:
        return 0.0, 0.0, 0.0, None

    # Phase 1: find Z of maximum anterior protrusion via vertex band min-Y
    # Vertex bands are fast (numpy mask) — good enough for finding the Z,
    # even if individual min-Y values are slightly noisy.
    mesh_verts = np.array(torso_mesh.vertices)
    best_z = None
    best_front_y = float("inf")
    for z in zs:
        mask = np.abs(mesh_verts[:, 2] - z) < 0.002
        if mask.sum() < 3:
            continue
        front_y = mesh_verts[mask, 1].min()  # most anterior (-Y)
        if front_y < best_front_y:
            best_front_y = front_y
            best_z = z

    if best_z is None:
        return 0.0, 0.0, 0.0, None

    # Phase 2: single contour slice for precise circumference + front Y
    slicer = MeshSlicer(torso_mesh)
    circ = slicer.circumference_at_z(best_z, combine_fragments=True)
    if circ < 0.30:
        return 0.0, 0.0, 0.0, None

    # Get precise belly front Y from contour (not vertex band)
    belly_front_y = _front_y_at_z(torso_mesh, best_z)
    if belly_front_y is None:
        belly_front_y = best_front_y  # fallback to vertex band value

    stomach_cm = circ * 100
    stomach_z = float(best_z)
    stomach_pct = stomach_z / height * 100 if height > 0 else 0
    return stomach_cm, stomach_z, stomach_pct, belly_front_y


def body_signature(mesh, step=0.002, low_pct=0.30, high_pct=0.85):
    """Compute body signature: circumference vs height.

    Returns:
        zs: array of Z heights (meters)
        circs: array of circumferences (meters)
    """
    height = mesh.vertices[:, 2].max()
    zs = np.arange(height * low_pct, height * high_pct, step)
    slicer = MeshSlicer(mesh)
    circs = np.array([slicer.circumference_at_z(z) for z in zs])
    return zs, circs


def find_measurement(zs, circs, height, region_name):
    """Find a measurement (max or min circumference) in a height region.

    Returns:
        z: height in meters
        circ_cm: circumference in cm
        pct: height percentage
    """
    region = REGIONS[region_name]
    pcts = zs / height * 100
    mask = (pcts >= region["low_pct"] * 100) & (pcts <= region["high_pct"] * 100)

    if not mask.any():
        return 0, 0, 0

    if region["mode"] == "max":
        idx = np.argmax(np.where(mask, circs, -1))
    else:  # min
        idx = np.argmin(np.where(mask, circs, 999))

    return zs[idx], circs[idx] * 100, pcts[idx]


# ── Limb measurements (plane sweep) ──

def measure_limb_at_z(mesh, z):
    """Find limb contours at a given Z height.

    Returns separate left/right contours by looking for contours that are
    NOT the torso (small X-extent, offset from center).

    Returns:
        list of (circumference_m, x_center) tuples for limb contours,
        sorted by circumference descending.
    """
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []

    try:
        path2d, _ = section.to_2D()
    except Exception:
        return []

    limbs = []
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        x_extent = pts[:, 0].max() - pts[:, 0].min()
        x_center = (pts[:, 0].max() + pts[:, 0].min()) / 2

        if len(pts) < 3:
            continue

        try:
            hull = ConvexHull(pts)
            circ = hull.area
        except Exception:
            closed = np.vstack([pts, pts[:1]])
            circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()

        limbs.append((circ, x_center, x_extent))

    limbs.sort(key=lambda c: -c[0])
    return limbs


def measure_thigh(mesh, height, step=0.002):
    """Measure thigh circumference via plane sweep.

    Sweeps from 30-43% height looking for Z levels where there are exactly
    2 separate leg contours of similar size. Returns max average circumference.
    Upper bound at 43% reaches the upper thigh / gluteal fold region
    (ISO 8559-1 5.3.20: fullest part of upper thigh). The "2 separate contours"
    requirement naturally rejects crotch-merged slices above ~45%.

    Returns:
        (circ_cm, z, pct) or (0, 0, 0) if not found.
    """
    slicer = MeshSlicer(mesh)
    best_circ = 0
    best_z = 0

    for pct in np.arange(0.30, 0.43, step / height):
        z = height * pct
        contours = slicer.limb_contours_at_z(z)
        if len(contours) < 2:
            continue

        # Two largest contours should be similar size (left/right legs)
        c1, xc1, _ = contours[0]
        c2, xc2, _ = contours[1]

        # Check they're on opposite sides and similar size
        if xc1 * xc2 >= 0:  # same side
            continue
        if min(c1, c2) < 0.5 * max(c1, c2):  # too different
            continue

        avg_circ = (c1 + c2) / 2
        if avg_circ > best_circ:
            best_circ = avg_circ
            best_z = z

    if best_circ == 0:
        return 0, 0, 0
    return best_circ * 100, best_z, best_z / height * 100


def measure_upperarm(mesh, height, step=0.002):
    """Measure upper arm circumference perpendicular to the arm axis.

    Two-phase approach:
    1. Fast horizontal sweep (72-80% height) to find the Z of maximum arm
       circumference (identifies arm contours by x_extent < 20cm, offset > 15cm).
    2. At best Z, estimate arm axis from horizontal centroids and re-measure
       with a plane perpendicular to the axis (ISO 8559-1 5.3.16).

    In A-pose (~45 degree arms), a horizontal cut overestimates circumference
    by ~8-10%. Perpendicular slicing gives the true cross-sectional circumference.

    Returns:
        (circ_cm, z, pct) or (0, 0, 0) if not found.
    """
    # Phase 1: fast horizontal sweep to find best Z
    slicer = MeshSlicer(mesh)
    best_circ_horiz = 0
    best_z = 0

    for pct in np.arange(0.72, 0.80, step / height):
        z = height * pct
        contours = slicer.limb_contours_at_z(z)

        arm_circs = []
        for circ, x_center, x_extent in contours:
            if x_extent > 0.20:
                continue
            if abs(x_center) < 0.15:
                continue
            arm_circs.append(circ)

        if len(arm_circs) >= 2:
            arm_circs.sort(reverse=True)
            avg_circ = (arm_circs[0] + arm_circs[1]) / 2
            if avg_circ > best_circ_horiz:
                best_circ_horiz = avg_circ
                best_z = z

    if best_z == 0:
        return 0, 0, 0

    # Phase 2: perpendicular re-measurement at best Z
    def arm_filter(pts):
        x_extent = pts[:, 0].max() - pts[:, 0].min()
        x_center = (pts[:, 0].max() + pts[:, 0].min()) / 2
        return x_extent <= 0.20 and abs(x_center) >= 0.15

    c_mid = _find_contour_centroids_at_z(mesh, best_z, arm_filter)
    c_lo = _find_contour_centroids_at_z(mesh, best_z - 0.10, arm_filter)

    if c_mid and c_lo:
        perp_circs = []
        for cm in c_mid:
            x_sign = np.sign(cm[0])
            same_lo = [c for c in c_lo if np.sign(c[0]) == x_sign]
            if not same_lo:
                continue
            cl = min(same_lo, key=lambda c: np.linalg.norm(c - cm))
            axis = cm - cl
            norm = np.linalg.norm(axis)
            if norm < 0.001:
                continue
            axis = axis / norm
            pts = _perpendicular_limb_contour(mesh, cm, axis, max_dist=0.10)
            if pts is not None and len(pts) >= 3:
                closed = np.vstack([pts, pts[:1]])
                circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
                perp_circs.append(circ)

        if len(perp_circs) >= 2:
            perp_circs.sort(reverse=True)
            best_circ = (perp_circs[0] + perp_circs[1]) / 2
            return best_circ * 100, best_z, best_z / height * 100

    # Fallback: return horizontal measurement
    return best_circ_horiz * 100, best_z, best_z / height * 100


def measure_neck(mesh, height, joints=None, step=0.002):
    """Measure neck circumference — perpendicular to neck axis (ISO 8559-1 §5.3.2).

    ISO definition: "Girth of the neck at a point just below the bulge at
    the thyroid cartilage (Adam's apple), measured perpendicular to the
    longitudinal axis of the neck."

    When ``joints`` are provided (with ``neck_base``, ``neck_mid``, and
    ``head``), slices perpendicular to the neck bone axis at the ``neck_mid``
    position (neck01 tail ≈ Adam's apple level, ~86% height).  The neck
    tilts forward ~15-20° from vertical, so perpendicular slicing avoids
    the ~5-6% overestimate of horizontal planes.

    Without joints, falls back to horizontal plane sweep.

    Args:
        mesh: full body trimesh (Z-up, metres, floor-aligned)
        height: total body height in metres
        joints: dict with ``neck_base`` (3,), ``neck_mid`` (3,), and
            ``head`` (3,) positions, or None for horizontal fallback.
        step: sweep step size in metres (default 2mm)

    Returns:
        (circ_cm, z, pct, contour_pts_or_None)
    """
    if joints and "neck_mid" in joints and "head" in joints and "neck_base" in joints:
        return _measure_neck_perpendicular(mesh, height, joints, step)
    circ_cm, z, pct = _measure_neck_horizontal(mesh, height, step)
    return circ_cm, z, pct, None


def _measure_neck_perpendicular(mesh, height, joints, step=0.002):
    """Neck circumference via perpendicular slice at Adam's apple level.

    Uses neck_mid (neck01 tail = neck02 head) as the anatomical anchor for
    the Adam's apple.  Slices perpendicular to the neck bone axis
    (neck_base → head) at that point.
    """
    neck_base = joints["neck_base"]
    head = joints["head"]
    neck_mid = joints["neck_mid"]
    axis = head - neck_base
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        circ_cm, z, pct = _measure_neck_horizontal(mesh, height, step)
        return circ_cm, z, pct, None
    axis_unit = axis / axis_len

    # Single perpendicular slice at the Adam's apple level (neck_mid)
    pts = _perpendicular_limb_contour(mesh, neck_mid, axis_unit, max_dist=0.08)
    if pts is not None and len(pts) >= 3:
        closed = np.vstack([pts, pts[:1]])
        circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
        if circ >= 0.20:
            return circ * 100, neck_mid[2], neck_mid[2] / height * 100, pts

    # Fallback: sweep a small range around neck_mid (±2cm along axis)
    best_circ = float("inf")
    best_z = 0
    best_pts = None
    margin = 0.02 / axis_len  # ±2cm in axis-parameter space
    t_mid = np.dot(neck_mid - neck_base, axis_unit) / axis_len
    for t in np.arange(max(0.05, t_mid - margin),
                       min(0.95, t_mid + margin),
                       step / axis_len):
        point = neck_base + t * axis
        pts = _perpendicular_limb_contour(mesh, point, axis_unit, max_dist=0.08)
        if pts is None or len(pts) < 3:
            continue
        closed = np.vstack([pts, pts[:1]])
        circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
        if circ < 0.20:
            continue
        if circ < best_circ:
            best_circ = circ
            best_z = point[2]
            best_pts = pts

    if best_circ == float("inf"):
        circ_cm, z, pct = _measure_neck_horizontal(mesh, height, step)
        return circ_cm, z, pct, None
    return best_circ * 100, best_z, best_z / height * 100, best_pts


def _measure_neck_horizontal(mesh, height, step=0.002):
    """Neck circumference via horizontal plane sweep (fallback)."""
    slicer = MeshSlicer(mesh)
    best_circ = float("inf")
    best_z = 0

    for pct in np.arange(0.82, 0.88, step / height):
        z = height * pct
        contours = slicer.limb_contours_at_z(z)

        # Find neck contour: small x_extent, centered near midline.
        # Min circ 0.20m (20cm) rejects mesh fragment artifacts that can
        # appear at the head-neck boundary as tiny disconnected contours.
        for circ, x_center, x_extent in contours:
            if x_extent > 0.25:
                continue
            if abs(x_center) > 0.10:  # neck is near the midline
                continue
            if circ < 0.20:  # reject tiny fragments (real neck > 25cm)
                continue
            if circ < best_circ:
                best_circ = circ
                best_z = z

    if best_circ == float("inf"):
        return 0, 0, 0
    return best_circ * 100, best_z, best_z / height * 100


def measure_knee(mesh, height, step=0.002):
    """Measure knee circumference at mid-patella level (ISO 8559-1 §5.3.22).

    Sweeps from 24-31% height looking for Z levels where there are exactly
    2 separate leg contours of similar size. Returns the circumference at
    the height closest to typical mid-patella level (~28% height).

    Returns:
        (circ_cm, z, pct) or (0, 0, 0) if not found.
    """
    slicer = MeshSlicer(mesh)
    target_pct = 0.275
    best_circ = 0
    best_z = 0
    best_dist = float("inf")

    for pct in np.arange(0.24, 0.31, step / height):
        z = height * pct
        contours = slicer.limb_contours_at_z(z)
        if len(contours) < 2:
            continue

        c1, xc1, _ = contours[0]
        c2, xc2, _ = contours[1]

        if xc1 * xc2 >= 0:  # same side
            continue
        if min(c1, c2) < 0.5 * max(c1, c2):  # too different
            continue

        dist = abs(pct - target_pct)
        if dist < best_dist:
            best_dist = dist
            best_circ = (c1 + c2) / 2
            best_z = z

    if best_circ == 0:
        return 0, 0, 0
    return best_circ * 100, best_z, best_z / height * 100


def measure_calf(mesh, height, joints=None, step=0.002):
    """Measure calf circumference — maximum lower leg girth (ISO 8559-1 §5.3.24).

    Horizontal plane sweep over the lower leg using the same convex-hull
    tape-measure simulation as the rest of the limb sweeps. Finds the
    global max within an anatomical search range, with a sanity check
    that the max is a real local peak rather than a boundary clip.

    Joint-anchored mode (knee + ankle joints provided): sweeps
    ``z ∈ [ankle_z + 6 cm, knee_z − 4 cm]``, where the 6 cm ankle offset
    clears the malleolus and the 4 cm knee offset clears the kneecap.
    The natural calf-belly maximum sits 9–11 cm below the knee on
    untuned bodies, comfortably interior to this range.

    Peak vs. boundary: if the max lands within one step of the upper
    bound (``z_max − step``), the lower leg is monotonically widening
    toward the knee — there is no real calf belly. This happens on
    tuned bodies where the optimizer has deflated the calf as a side
    effect of inflating the thighs/hips. Reporting the boundary value
    would put the measurement on the upper lower-leg (essentially the
    popliteal region), which is anatomically misleading. Instead, we
    fall back to ``knee_z − 0.30 × (knee_z − ankle_z)`` — the typical
    gastrocnemius-peak position (~30 % down from the knee) — and report
    the girth there. The value is no longer a true ISO max, but it's
    where a tape would land on a normal anatomy and the visualization is
    sensible. The reported ``calf_cm`` will be smaller than the boundary
    clip, honestly reflecting the deflated geometry.

    Without joints, falls back to a fixed 16–26 % height range with no
    peak detection — the legacy behavior, vulnerable to the boundary
    case but adequate for un-tuned bodies.

    Returns:
        (circ_cm, z, pct) or (0, 0, 0) if not found.
    """
    z_min, z_max, fallback_z = _calf_search_range(joints, height)
    if z_max <= z_min:
        return 0, 0, 0

    slicer = MeshSlicer(mesh)
    best_circ = 0
    best_z = 0

    for z in np.arange(z_min, z_max, step):
        avg_circ = _two_leg_avg_circumference(slicer, z)
        if avg_circ > best_circ:
            best_circ = avg_circ
            best_z = z

    if best_circ == 0:
        return 0, 0, 0

    # Boundary detection: max at the upper bound means no true calf-belly
    # peak exists. Use the anatomical fallback position when available.
    at_upper_bound = best_z >= z_max - step * 1.5
    if fallback_z is not None and at_upper_bound:
        fb_circ = _two_leg_avg_circumference(slicer, fallback_z)
        if fb_circ > 0:
            best_z = fallback_z
            best_circ = fb_circ

    return best_circ * 100, best_z, best_z / height * 100


def _two_leg_avg_circumference(slicer, z):
    """Average girth of the two leg cross-sections at height z.

    Returns 0 unless there are exactly two contours on opposite sides
    (one per leg) of similar size — same validity gate used by the
    knee/calf horizontal sweeps.
    """
    contours = slicer.limb_contours_at_z(z)
    if len(contours) < 2:
        return 0.0
    c1, xc1, _ = contours[0]
    c2, xc2, _ = contours[1]
    if xc1 * xc2 >= 0:  # same side
        return 0.0
    if min(c1, c2) < 0.5 * max(c1, c2):  # too different
        return 0.0
    return (c1 + c2) / 2


def _calf_search_range(joints, height):
    """Sweep bounds + anatomical fallback for measure_calf.

    Returns (z_min, z_max, fallback_z) in metres. The fallback is None
    when joints are unavailable; in that case the legacy fixed 16–26 %
    range is used and no peak detection happens.

    Joint-anchored bounds: 6 cm above ankle (clears malleolus), 4 cm
    below knee (clears kneecap). Anatomical fallback: 30 % of lower-leg
    span below the knee — the typical gastrocnemius-peak location.
    """
    if joints is not None:
        l_knee = joints.get("l_knee")
        r_knee = joints.get("r_knee")
        l_ankle = joints.get("l_ankle")
        r_ankle = joints.get("r_ankle")
        if all(p is not None for p in (l_knee, r_knee, l_ankle, r_ankle)):
            knee_z = min(float(l_knee[2]), float(r_knee[2]))
            ankle_z = max(float(l_ankle[2]), float(r_ankle[2]))
            z_min = ankle_z + 0.06
            z_max = knee_z - 0.04
            fallback_z = knee_z - 0.30 * (knee_z - ankle_z)
            return z_min, z_max, fallback_z
    return height * 0.16, height * 0.26, None


def measure_wrist(mesh, height, joints=None, step=0.002):
    """Measure wrist circumference perpendicular to the forearm axis.

    ISO 8559-1 §5.3.19: girth of the wrist measured over the wrist bones
    (styloid processes of radius and ulna).

    When joints are provided, uses elbow→wrist to estimate the forearm axis
    and slices perpendicular to it at the wrist position. Sweeps a small
    range along the axis to find the minimum circumference (the bony wrist).

    Without joints, returns (0, 0, 0) — horizontal slicing is unreliable
    for arms in A-pose (~45° angle).

    Returns:
        (circ_cm, z, pct) or (0, 0, 0) if not found.
    """
    if not joints:
        return 0, 0, 0

    circs = []
    for side in ("l", "r"):
        elbow = joints.get(f"{side}_elbow")
        wrist = joints.get(f"{side}_wrist")
        if elbow is None or wrist is None:
            continue

        axis = wrist - elbow
        axis_len = np.linalg.norm(axis)
        if axis_len < 0.01:
            continue
        axis_unit = axis / axis_len

        # Sweep a small range near the wrist (last 15% of forearm)
        best_circ = float("inf")
        for t in np.arange(0.85, 1.01, step / axis_len):
            point = elbow + t * axis
            pts = _perpendicular_limb_contour(mesh, point, axis_unit, max_dist=0.10)
            if pts is None or len(pts) < 3:
                continue
            closed = np.vstack([pts, pts[:1]])
            circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
            if circ < 0.10:  # reject tiny fragments (real wrist > 12cm)
                continue
            if circ < best_circ:
                best_circ = circ

        if best_circ < float("inf"):
            circs.append(best_circ)

    if not circs:
        return 0, 0, 0

    avg_circ = float(np.mean(circs))
    # Approximate wrist Z from joint position (for metadata)
    wrist_pos = joints.get("l_wrist")
    if wrist_pos is None:
        wrist_pos = joints.get("r_wrist")
    wrist_z = float(wrist_pos[2]) if wrist_pos is not None else 0
    return avg_circ * 100, wrist_z, wrist_z / height * 100
