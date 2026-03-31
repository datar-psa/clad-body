"""
Shared body measurement infrastructure — plane sweep, limb measurement, rendering.

Both Anny and MHR measurement modules use these functions. All circumference
measurements follow ISO 8559-1 conventions with convex hull (tape-measure simulation).
"""

import json
import os

import numpy as np
from scipy.spatial import ConvexHull

import trimesh

# ── Constants ──

# Search regions as percentage of body height (ISO 8559-1 compliant).
REGIONS = {
    "hip":  {"low_pct": 0.46, "high_pct": 0.54, "mode": "max"},
    "bust": {"low_pct": 0.68, "high_pct": 0.76, "mode": "max"},
}

# ISO 8559-1 waist level: midway between lowest rib and iliac crest.
# Iliac crest ≈ top of hip region (54%), lowest rib ≈ bottom of bust region (68%).
# Validated against MHR skeleton: c_spine1 joint sits at ~62% height.
WAIST_HEIGHT_PCT = (REGIONS["hip"]["high_pct"] + REGIONS["bust"]["low_pct"]) / 2

# Maximum X extent (meters) for a valid torso contour.
# Contours wider than this include arms connected to the torso.
MAX_TORSO_X_EXTENT = 0.50

CONTOUR_COLORS = {
    "bust": "green", "underbust": "cyan", "waist": "blue",
    "stomach": "darkorange", "hip": "red",
    "thigh": "purple", "knee": "deeppink", "calf": "hotpink",
    "upperarm": "orange", "wrist": "gold", "neck": "teal",
}

# Canonical joint name mapping for linear measurements.
# Maps our standard names to model-specific joint name candidates.
# Used by extract_joints_from_names() to find the right indices.
# NOTE: Shoulder bones are INSIDE the body. upperarm01 HEAD gives the correct
# lateral (X) position for the shoulder/arm junction. The actual acromion
# (bony shoulder tip) is found by model-specific find_acromion() functions
# in measure/anny.py (max Z above bone tail) and measure/mhr.py (max |X|).
ANNY_JOINT_MAP = {
    "c7": ["neck01"],
    "neck_base": ["neck01"],      # base of neck (C7 level, ~85% height)
    "neck_mid": ["neck02"],       # neck01 tail = neck02 head (~86% height, Adam's apple)
    "head": ["head"],             # top of neck / base of skull
    "l_shoulder": ["upperarm01.L"],
    "r_shoulder": ["upperarm01.R"],
    "l_elbow": ["lowerarm01.L"],
    "r_elbow": ["lowerarm01.R"],
    "l_wrist": ["wrist.L"],
    "r_wrist": ["wrist.R"],
}

MHR_JOINT_MAP = {
    "c7": ["c_neck"],        # y=144 cm (~85% height) — cervicothoracic junction
    "l_shoulder": ["l_uparm"],
    "r_shoulder": ["r_uparm"],
    "l_elbow": ["l_lowarm"],
    "r_elbow": ["r_lowarm"],
    "l_wrist": ["l_wrist"],
    "r_wrist": ["r_wrist"],
}


# ── Fast vectorized slicer ──


class MeshSlicer:
    """Fast horizontal mesh slicer for body measurements.

    Pre-computes face topology once, then slices at any Z height using
    vectorized numpy operations + union-find for contour separation.
    ~10x faster than trimesh.section() for dense sweeps (2mm step).
    """

    def __init__(self, mesh):
        verts = mesh.vertices
        faces = mesh.faces

        # Vertex coordinates as flat arrays for fast indexing
        self._vx = np.ascontiguousarray(verts[:, 0])
        self._vy = np.ascontiguousarray(verts[:, 1])
        self._vz = np.ascontiguousarray(verts[:, 2])
        self._nv = len(verts)

        # Face edges: (F, 3, 2) — vertex index pairs for each of 3 edges per face
        self._face_edges = np.stack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], axis=1)  # (F, 3, 2)

        # Z values at each edge endpoint: (F, 3, 2)
        self._edge_z = verts[self._face_edges, 2]

        # Face Z ranges for fast filtering
        face_z = verts[faces, 2]  # (F, 3)
        self._face_min_z = face_z.min(axis=1)
        self._face_max_z = face_z.max(axis=1)

    def contours_at_z(self, z):
        """Slice mesh at height z, return separated 2D contours.

        Returns list of (pts_xy, x_extent, x_center) tuples, one per
        connected component with >= 3 points.
        """
        # 1. Find faces straddling z
        mask = (self._face_min_z < z) & (self._face_max_z > z)
        fids = np.nonzero(mask)[0]
        if len(fids) == 0:
            return []

        # 2. Find crossing edges (endpoints on opposite sides of z)
        ez = self._edge_z[fids]  # (C, 3, 2)
        crosses = (ez[:, :, 0] < z) != (ez[:, :, 1] < z)  # (C, 3)
        ci, ei = np.nonzero(crosses)
        if len(ci) == 0:
            return []

        # 3. Interpolate XY intersection points
        v0 = self._face_edges[fids[ci], ei, 0]
        v1 = self._face_edges[fids[ci], ei, 1]

        z0 = self._vz[v0]
        z1 = self._vz[v1]
        t = np.clip((z - z0) / (z1 - z0 + 1e-30), 0.0, 1.0)

        ix = self._vx[v0] + t * (self._vx[v1] - self._vx[v0])
        iy = self._vy[v0] + t * (self._vy[v1] - self._vy[v0])

        # 4. Canonical edge keys for deduplication
        lo = np.minimum(v0, v1)
        hi = np.maximum(v0, v1)
        ek = lo.astype(np.int64) * self._nv + hi.astype(np.int64)

        # 5. Unique intersection points (one per mesh edge)
        uniq_ek, first_occ, inv_idx = np.unique(
            ek, return_index=True, return_inverse=True)
        n_uniq = len(uniq_ek)
        pts_x = ix[first_occ]
        pts_y = iy[first_occ]

        if n_uniq < 3:
            return []

        # 6. Contour separation via union-find on shared edges
        # Group by face: sort points by their local face index (ci)
        order = np.argsort(ci)
        ci_s = ci[order]
        inv_s = inv_idx[order]

        # Find face boundaries — faces with exactly 2 crossing edges
        uci, starts, counts = np.unique(ci_s, return_index=True,
                                         return_counts=True)
        m2 = counts >= 2
        pairs_start = starts[m2]
        ea = inv_s[pairs_start]
        eb = inv_s[pairs_start + 1]

        # Union-find
        parent = np.arange(n_uniq, dtype=np.intp)
        for i in range(len(ea)):
            a, b = int(ea[i]), int(eb[i])
            while parent[a] != a:
                a = parent[a]
            while parent[b] != b:
                b = parent[b]
            if a != b:
                parent[max(a, b)] = min(a, b)

        # Path compression
        for i in range(n_uniq):
            r = i
            while parent[r] != r:
                r = parent[r]
            parent[i] = r

        # 7. Group points by component
        labels = parent
        unique_labels = np.unique(labels)

        result = []
        for lbl in unique_labels:
            idx = np.nonzero(labels == lbl)[0]
            if len(idx) < 3:
                continue
            pts = np.column_stack([pts_x[idx], pts_y[idx]])
            x_ext = pts[:, 0].max() - pts[:, 0].min()
            x_ctr = (pts[:, 0].max() + pts[:, 0].min()) / 2
            result.append((pts, x_ext, x_ctr))

        return result

    def circumference_at_z(self, z, max_x_extent=MAX_TORSO_X_EXTENT,
                           combine_fragments=False):
        """Convex hull circumference at height z.

        Args:
            max_x_extent: reject contours wider than this (filters arm-merged).
            combine_fragments: if True, merge all qualifying fragments into one
                hull.  Use on torso-only meshes where arm-face removal can split
                the contour at shoulder gaps.  On full meshes leave False to
                avoid merging finger/hand contours into the torso.

        Returns circumference in meters (or 0 if no valid contour).
        """
        contours = self.contours_at_z(z)
        qualifying = [pts for pts, x_ext, _ in contours if x_ext < max_x_extent]
        if not qualifying:
            return 0.0

        if combine_fragments:
            combined = np.vstack(qualifying)
            if len(combined) < 3:
                return 0.0
            try:
                hull = ConvexHull(combined)
                return hull.area
            except Exception:
                closed = np.vstack([combined, combined[:1]])
                return np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()

        best = 0.0
        for pts in qualifying:
            try:
                hull = ConvexHull(pts)
                circ = hull.area
            except Exception:
                closed = np.vstack([pts, pts[:1]])
                circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
            if circ > best:
                best = circ
        return best

    def limb_contours_at_z(self, z):
        """Get all contours at z with their properties.

        Returns list of (circumference_m, x_center, x_extent) tuples,
        sorted by circumference descending. Same output format as
        measure_limb_at_z().
        """
        contours = self.contours_at_z(z)
        limbs = []
        for pts, x_ext, x_ctr in contours:
            try:
                hull = ConvexHull(pts)
                circ = hull.area
            except Exception:
                closed = np.vstack([pts, pts[:1]])
                circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
            limbs.append((circ, x_ctr, x_ext))
        limbs.sort(key=lambda c: -c[0])
        return limbs


# ── Core plane sweep ──

def torso_circumference_at_z(mesh, z, max_x_extent=MAX_TORSO_X_EXTENT,
                             return_contour=False, convex_hull=True,
                             combine_fragments=False):
    """Compute circumference at a given Z height.

    Slices the mesh with a horizontal plane, filters by max_x_extent.

    Args:
        return_contour: If True, also return 3D contour points.
        convex_hull: If True, compute convex hull perimeter (tape-measure sim).
        combine_fragments: If True, merge all qualifying fragments before
            computing the hull.  Use on torso-only meshes where arm-face
            removal splits the contour at shoulder gaps.

    Returns:
        circumference in meters (or 0 if no valid contour found).
        If return_contour=True, returns (circumference, points_3d or None).
    """
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return (0, None) if return_contour else 0

    try:
        path2d, to_3D = section.to_2D()
    except Exception:
        return (0, None) if return_contour else 0

    # Collect qualifying contour fragments
    qualifying = []
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        x_extent = pts[:, 0].max() - pts[:, 0].min()
        if x_extent >= max_x_extent:
            continue
        qualifying.append(pts)

    if not qualifying:
        return (0, None) if return_contour else 0

    if combine_fragments:
        # Merge all fragments (torso mesh: arm gaps split the contour)
        combined = np.vstack(qualifying)
        if len(combined) < 3:
            return (0, None) if return_contour else 0
        if convex_hull:
            try:
                hull = ConvexHull(combined)
                circ = hull.area
                result_pts = combined[hull.vertices]
            except Exception:
                closed = np.vstack([combined, combined[:1]])
                circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
                result_pts = combined
        else:
            closed = np.vstack([combined, combined[:1]])
            circ = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
            result_pts = combined
    else:
        # Pick largest single contour (full mesh: avoid merging finger/hand)
        circ = 0
        result_pts = None
        for pts in qualifying:
            if convex_hull and len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    c = hull.area
                    c_pts = pts[hull.vertices]
                except Exception:
                    closed = np.vstack([pts, pts[:1]])
                    c = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
                    c_pts = pts
            else:
                closed = np.vstack([pts, pts[:1]])
                c = np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()
                c_pts = pts
            if c > circ:
                circ = c
                result_pts = c_pts

    if not return_contour:
        return circ

    # Convert 2D contour back to 3D
    pts_3d = None
    if result_pts is not None:
        pts_h = np.column_stack([result_pts, np.zeros(len(result_pts)),
                                 np.ones(len(result_pts))])
        pts_3d = (to_3D @ pts_h.T).T[:, :3]

    return circ, pts_3d


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

    Returns (bust_cm, bust_z, hips_cm, hip_z).
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
    hips_cm, hip_z = 0.0, 0.0

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
            hips_cm = hip_circs[idx] * 100
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

    return bust_cm, bust_z, hips_cm, hip_z


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


def measure_inseam(mesh, height, step=0.002):
    """Measure inside leg length (crotch to floor) via mesh geometry.

    Sweeps upward from 25% height looking for the Z where two separate
    leg contours merge into one (the crotch point). Inseam = crotch Z.
    ISO 8559-1 5.1.15: vertical distance from crotch to floor.

    Returns:
        (inseam_cm, crotch_z, crotch_pct) or (0, 0, 0) if not found.
    """
    slicer = MeshSlicer(mesh)
    last_two_legs_z = 0
    found_two_legs = False

    for pct in np.arange(0.25, 0.55, step / height):
        z = height * pct
        contours = slicer.limb_contours_at_z(z)

        # Count contours that look like legs (not torso — x_extent < 30cm)
        leg_contours = []
        for circ, x_center, x_extent in contours:
            if x_extent > 0.30:
                continue
            leg_contours.append((circ, x_center))

        # Two legs on opposite sides?
        has_two = False
        if len(leg_contours) >= 2:
            has_pos = any(x > 0 for _, x in leg_contours)
            has_neg = any(x < 0 for _, x in leg_contours)
            has_two = has_pos and has_neg

        if has_two:
            found_two_legs = True
            last_two_legs_z = z
        elif found_two_legs:
            # Was two legs, now merged → crotch at last_two_legs_z
            break

    if last_two_legs_z > 0:
        return last_two_legs_z * 100, last_two_legs_z, last_two_legs_z / height * 100
    return 0, 0, 0


def measure_crotch_length(mesh, height, waist_z, crotch_z, step=0.005):
    """Measure crotch length (total rise) via surface tracing — ISO 8559-1 §5.1.11.

    Traces the body surface along the midline (X~0) from waist (front) down
    through the perineum (crotch) and back up to waist (back).  This is the
    measurement a tailor takes by running a tape from front waist, between
    the legs, to back waist.

    Blue Eye Custom Tailor: "From waist down through crotch to buttocks and
    back up."  SAIA 3DLook provides front_crotch_length and back_crotch_length
    as separate values.

    Args:
        mesh: trimesh body mesh (Z-up, metres, floor-aligned)
        height: body height in metres
        waist_z: waist Z height in metres (from waist measurement)
        crotch_z: crotch Z height in metres (from measure_inseam)
        step: Z sampling interval in metres (default 5mm)

    Returns:
        (crotch_length_cm, front_rise_cm, back_rise_cm,
         front_pts, back_pts) where pts are (N, 3) arrays for rendering.
        Returns (0, 0, 0, None, None) if inputs are invalid.
    """
    if waist_z <= 0 or crotch_z <= 0 or waist_z <= crotch_z:
        return 0, 0, 0, None, None

    x_band = 0.05  # metres — midline band half-width
    slicer = MeshSlicer(mesh)

    # Sample Z levels from waist down to crotch
    z_levels = np.arange(waist_z, crotch_z - step / 2, -step)
    if len(z_levels) < 2:
        return 0, 0, 0, None, None

    # At each Z, use MeshSlicer to get actual surface contour points (not
    # raw vertices, which include interior mesh points at leg junctions).
    # From the contour XY points near X~0, take min-Y (front) and max-Y (back).
    front_ys = []
    back_ys = []
    valid_zs = []

    for z in z_levels:
        contours = slicer.contours_at_z(z)
        if not contours:
            continue

        # Collect all contour points near the midline
        midline_pts = []
        for pts_xy, _, _ in contours:
            near = pts_xy[np.abs(pts_xy[:, 0]) < x_band]
            if len(near) > 0:
                midline_pts.append(near)

        if not midline_pts:
            continue
        midline = np.vstack(midline_pts)
        if len(midline) < 2:
            continue

        front_ys.append(float(midline[:, 1].min()))
        back_ys.append(float(midline[:, 1].max()))
        valid_zs.append(z)

    if len(valid_zs) < 2:
        return 0, 0, 0, None, None

    valid_zs = np.array(valid_zs)
    front_ys = np.array(front_ys)
    back_ys = np.array(back_ys)

    # Build 3D polylines (X=0, Y=surface, Z=height)
    front_pts = np.column_stack([
        np.zeros(len(valid_zs)), front_ys, valid_zs])
    back_pts = np.column_stack([
        np.zeros(len(valid_zs)), back_ys, valid_zs])

    # Both paths share the crotch endpoint (perineum) — average the
    # front and back at the lowest point so paths connect cleanly.
    crotch_pt = (front_pts[-1] + back_pts[-1]) / 2
    front_pts[-1] = crotch_pt
    back_pts[-1] = crotch_pt

    # Sum Euclidean segment lengths
    front_len = float(np.linalg.norm(np.diff(front_pts, axis=0), axis=1).sum())
    back_len = float(np.linalg.norm(np.diff(back_pts, axis=0), axis=1).sum())
    total = front_len + back_len

    return (total * 100, front_len * 100, back_len * 100,
            front_pts.astype(np.float32), back_pts.astype(np.float32))


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


def measure_calf(mesh, height, step=0.002):
    """Measure calf circumference — maximum lower leg girth (ISO 8559-1 §5.3.24).

    Sweeps from 16-26% height looking for Z levels where there are exactly
    2 separate leg contours of similar size. Returns the maximum average
    circumference (fullest part of the calf muscle).

    Returns:
        (circ_cm, z, pct) or (0, 0, 0) if not found.
    """
    slicer = MeshSlicer(mesh)
    best_circ = 0
    best_z = 0

    for pct in np.arange(0.16, 0.26, step / height):
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

        avg_circ = (c1 + c2) / 2
        if avg_circ > best_circ:
            best_circ = avg_circ
            best_z = z

    if best_circ == 0:
        return 0, 0, 0
    return best_circ * 100, best_z, best_z / height * 100


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


def _shoulder_arc_polyline(verts, l_acromion, r_acromion, c7, n_samples=30):
    """Build a smooth surface-following polyline from R acromion → C7 → L acromion.

    Fits a cubic spline through the three waypoints in the XZ plane, then
    projects each sample point onto the posterior body surface. The result
    is a smooth arc that follows the upper back contour (like a real tape
    measure), not a V-shaped pair of straight lines.

    Args:
        verts: (V, 3) mesh vertices (Z-up, metres)
        l_acromion: (3,) left acromion position
        r_acromion: (3,) right acromion position
        c7: (3,) C7 vertebra position (back of neck)
        n_samples: number of points along the arc

    Returns:
        (N, 3) numpy array of surface-projected arc points, or None if
        surface projection fails for too many points.
    """
    from scipy.interpolate import CubicSpline

    # Parameterise waypoints by cumulative chord length: R → C7 → L
    waypoints = np.array([r_acromion, c7, l_acromion])
    dists = np.cumsum([0] + [np.linalg.norm(waypoints[i+1] - waypoints[i])
                              for i in range(len(waypoints) - 1)])
    # Cubic spline through the 3 waypoints (X and Z as functions of arc parameter)
    cs_x = CubicSpline(dists, waypoints[:, 0], bc_type='natural')
    cs_z = CubicSpline(dists, waypoints[:, 2], bc_type='natural')

    # Acromion points (first & last) are already on the skin surface — use
    # their actual Y so the arc starts/ends exactly at the landmark dots.
    # Only C7 (middle waypoint) needs surface projection because the bone
    # position is inside the body.
    wp_y = [float(r_acromion[1]), None, float(l_acromion[1])]
    c7_y = _surface_y_at(verts, c7[0], c7[2], x_tol=0.03, z_tol=0.01, side="back")
    if c7_y is None:
        c7_y = _surface_y_at(verts, c7[0], c7[2], x_tol=0.04, z_tol=0.03, side="back")
    if c7_y is None:
        c7_y = _surface_y_at(verts, c7[0], c7[2], x_tol=0.06, z_tol=0.06, side="back")
    wp_y[1] = c7_y if c7_y is not None else float(c7[1])

    cs_y = CubicSpline(dists, wp_y, bc_type='natural')

    t_vals = np.linspace(dists[0], dists[-1], n_samples)
    pts = []
    for t in t_vals:
        x = float(cs_x(t))
        z = float(cs_z(t))
        y = float(cs_y(t))
        pts.append([x, y, z])

    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.float64)


def measure_shoulder_width(joints, mesh=None, acromion_fn=None):
    """Measure shoulder width — arc length over upper back via C7.

    ISO 8559-1 5.4.2: tape follows contour of upper back between left and
    right acromion points, passing over the C7 vertebra (back neck point).

    When a mesh is provided, builds a smooth surface-following polyline from
    acromion to acromion via C7 and sums segment lengths (true tape-measure
    arc length). Without mesh, falls back to straight-line segments.

    Args:
        joints: dict with 'l_shoulder', 'r_shoulder', 'c7' as (3,) arrays (metres, Z-up).
        mesh: optional trimesh for surface acromion detection + arc measurement.
        acromion_fn: callable(verts, joint, side) → (3,) acromion position.
            Model-specific — Anny and MHR provide their own implementations.

    Returns:
        (shoulder_width_cm, arc_pts) — cm value and (N,3) polyline (or None).
        Returns (0, None) if joints unavailable.
    """
    l = joints.get("l_shoulder")
    r = joints.get("r_shoulder")
    c7 = joints.get("c7")
    if l is None or r is None:
        return 0, None

    if mesh is not None and acromion_fn is not None:
        verts = np.array(mesh.vertices)
        l = acromion_fn(verts, l, side="left")
        r = acromion_fn(verts, r, side="right")

        if c7 is not None:
            arc_pts = _shoulder_arc_polyline(verts, l, r, c7)
            if arc_pts is not None and len(arc_pts) >= 2:
                diffs = np.diff(arc_pts, axis=0)
                arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
                return arc_len * 100, arc_pts.astype(np.float32)

    if c7 is None:
        return float(np.linalg.norm(l - r) * 100), None
    arc = np.linalg.norm(c7 - r) + np.linalg.norm(l - c7)
    return float(arc * 100), None


def measure_sleeve_length(joints, mesh=None, acromion_fn=None):
    """Measure sleeve length — shoulder (acromion) to elbow to wrist.

    ISO 8559-1 5.7.8: distance from shoulder point (acromion) over the elbow
    to the wrist. Sum of Euclidean segment lengths. Averages left and right.

    When a mesh is provided, projects shoulder joints to the acromion (surface
    shoulder tip) via acromion_fn, matching shoulder_width behaviour.
    Without mesh, uses raw joint positions (inside the body — underestimates).

    Args:
        joints: dict with 'l/r_shoulder', 'l/r_elbow', 'l/r_wrist'
                as (3,) arrays (metres, Z-up).
        mesh: optional trimesh for surface acromion detection.
        acromion_fn: callable(verts, joint, side) → (3,) acromion position.

    Returns:
        sleeve_length_cm or 0 if joints unavailable.
    """
    verts = np.array(mesh.vertices) if mesh is not None else None

    lengths = []
    for side in ("l", "r"):
        shoulder = joints.get(f"{side}_shoulder")
        elbow = joints.get(f"{side}_elbow")
        wrist = joints.get(f"{side}_wrist")
        if shoulder is None or elbow is None or wrist is None:
            continue

        if verts is not None and acromion_fn is not None:
            side_name = "left" if side == "l" else "right"
            shoulder = acromion_fn(verts, shoulder, side=side_name)

        sleeve_len = (np.linalg.norm(shoulder - elbow) +
                      np.linalg.norm(elbow - wrist))
        lengths.append(sleeve_len)

    if not lengths:
        return 0
    return float(np.mean(lengths) * 100)


# Backwards compatibility alias
measure_arm_length = measure_sleeve_length




def _surface_y_at(verts, x, z, x_tol=0.05, z_tol=0.05, side="back"):
    """Y coordinate of the body surface at (x, z) from the given side.

    The Anny body faces -Y (nose at min Y, back of head at max Y).

    Args:
        verts: (V, 3) vertex array (Z-up, metres)
        x: target X position
        z: target Z position
        x_tol: half-width of X search band (metres)
        z_tol: half-width of Z search band (metres)
        side: "back" → max Y (posterior surface), "front" → min Y (anterior)

    Returns:
        Y value, or None if no vertices found in band.
    """
    mask = (np.abs(verts[:, 0] - x) < x_tol) & (np.abs(verts[:, 2] - z) < z_tol)
    near = verts[mask]
    if len(near) == 0:
        return None
    return float(near[:, 1].max() if side == "back" else near[:, 1].min())


def extract_linear_measurement_polylines(mesh, measurements, joints):
    """Build surface-projected 3D polylines for linear measurements.

    Computes open polylines that lie on the body surface for:
    - shoulder_width: pre-computed arc from measure_shoulder_width (via _shoulder_arc_pts)
    - sleeve_length: joint chain shoulder→elbow→wrist projected to back surface
    - inseam: inner leg seam from floor to crotch

    All points are projected to the posterior (back) body surface — the side
    that faces the camera in the standard 3D viewer (Three.js rotation.x=-π/2).

    Args:
        mesh: full body trimesh (Z-up, metres)
        measurements: dict from measure_body/measure_mhr (needs _inseam_z,
            _shoulder_arc_pts)
        joints: canonical joint dict (from extract_joints_from_names)

    Returns:
        dict mapping name → (N, 3) numpy array (open polyline on body surface).
        Empty dict if joint data unavailable.
    """
    if not joints:
        return {}

    verts = np.array(mesh.vertices)
    result = {}

    l_sh = joints.get("l_shoulder")
    l_elbow = joints.get("l_elbow")
    l_wrist = joints.get("l_wrist")

    # Shoulder width: reuse the arc polyline computed by measure_shoulder_width
    # so the rendered line and the cm value are guaranteed to match.
    arc_pts = measurements.get("_shoulder_arc_pts")
    if arc_pts is not None:
        result["shoulder_width"] = arc_pts

    # Sleeve length: left acromion → left elbow → left wrist, projected to surface.
    # Start from the acromion (shoulder arc endpoint) rather than the bone joint,
    # so the line starts at the visible shoulder tip on the skin surface.
    if l_sh is not None and l_elbow is not None and l_wrist is not None:
        # Use left acromion from shoulder arc if available (last point = L acromion)
        if arc_pts is not None and len(arc_pts) >= 2:
            start = arc_pts[-1]  # L acromion (already on skin surface)
        else:
            start = l_sh
        chain = [list(start)]
        for j in [l_elbow, l_wrist]:
            y = _surface_y_at(verts, j[0], j[2], x_tol=0.05, z_tol=0.05, side="back")
            if y is None:
                y = _surface_y_at(verts, j[0], j[2], x_tol=0.10, z_tol=0.10, side="back")
            if y is None:
                y = float(verts[:, 1].min()) - 0.005
            chain.append([j[0], y - 0.005, j[2]])
        result["sleeve_length"] = np.array(chain, dtype=np.float32)

    # Inseam: vertical line from crotch to floor (ISO 8559-1 5.1.15).
    # Positioned at the inner right leg surface at crotch height.
    inseam_z = measurements.get("_inseam_z", 0)
    if inseam_z > 0:
        # Find innermost right-leg vertices near crotch
        near = verts[
            (verts[:, 0] > 0.01) & (verts[:, 0] < 0.15) &
            (np.abs(verts[:, 2] - inseam_z) < 0.02)
        ]
        x_seam = float(near[:, 0].min()) if len(near) > 0 else 0.04
        # Y from inner-leg vertices (mean Y of the innermost strip)
        inner = near[near[:, 0] < x_seam + 0.02]
        y = float(inner[:, 1].mean()) if len(inner) > 0 else 0.0
        pts = np.array([
            [x_seam, y, 0.0],
            [x_seam, y, inseam_z],
        ], dtype=np.float32)
        result["inseam"] = pts

    # Crotch length: front and back midline paths from waist to perineum.
    crotch_front = measurements.get("_crotch_front_pts")
    crotch_back = measurements.get("_crotch_back_pts")
    if crotch_front is not None:
        result["crotch_front"] = crotch_front
    if crotch_back is not None:
        result["crotch_back"] = crotch_back

    return result


def extract_joints_from_names(joint_names, joint_positions, joint_map):
    """Map model-specific joint names to canonical joint positions.

    Args:
        joint_names: list of joint name strings from the body model.
        joint_positions: (N, 3) array of joint positions (metres, Z-up).
        joint_map: dict mapping canonical name → list of candidate model names.

    Returns:
        dict mapping canonical name → (3,) numpy array, or empty dict on failure.
    """
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    result = {}
    for canon_name, candidates in joint_map.items():
        for cand in candidates:
            if cand in name_to_idx:
                result[canon_name] = joint_positions[name_to_idx[cand]]
                break
    return result


# ── Utilities ──

def floor_align(v: np.ndarray) -> np.ndarray:
    """Align mesh vertices: feet at Z=0, X-centred, torso-centred in Y.

    Y centering uses only torso vertices (|x| < 15 cm from midline) to avoid
    arm-position-dependent shifts — different arm angles between MHR and Anny
    would otherwise pull the bounding-box centre forward/backward.

    Args:
        v: (V, 3) numpy array of vertex positions in metres.

    Returns:
        Aligned copy of v.
    """
    out = v.copy()
    x_center = (v[:, 0].max() + v[:, 0].min()) / 2
    out[:, 0] -= x_center
    torso_mask = np.abs(v[:, 0] - x_center) < 0.15
    if torso_mask.sum() > 100:
        y_center = (v[torso_mask, 1].max() + v[torso_mask, 1].min()) / 2
    else:
        y_center = (v[:, 1].max() + v[:, 1].min()) / 2
    out[:, 1] -= y_center
    out[:, 2] -= v[:, 2].min()
    return out


def load_target_measurements(path: str) -> dict:
    """Load target measurements JSON.

    Returns dict with only non-null measurement keys (height_cm, waist_cm, mass_kg).
    Keys starting with '_' are metadata and ignored.
    """
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items()
            if not k.startswith("_") and v is not None and v != ""}


def find_target_json(body_dir: str) -> str | None:
    """Auto-find target_measurements.json next to body OBJ."""
    candidate = os.path.join(body_dir, "target_measurements.json")
    return candidate if os.path.exists(candidate) else None


def print_comparison(current: dict, target: dict):
    """Print current vs target measurements with deltas."""
    display = {
        "height_cm":          ("Height",     "cm", ".1f"),
        "bust_cm":            ("Bust",       "cm", ".1f"),
        "underbust_cm":       ("Underbust",  "cm", ".1f"),
        "waist_cm":           ("Waist",      "cm", ".1f"),
        "stomach_cm":         ("Stomach",    "cm", ".1f"),
        "hips_cm":            ("Hips",       "cm", ".1f"),
        "hip_cm":             ("Hip",        "cm", ".1f"),
        "thigh_cm":           ("Thigh",      "cm", ".1f"),
        "knee_cm":            ("Knee",       "cm", ".1f"),
        "calf_cm":            ("Calf",       "cm", ".1f"),
        "upperarm_cm":        ("Upper arm",  "cm", ".1f"),
        "wrist_cm":           ("Wrist",      "cm", ".1f"),
        "neck_cm":            ("Neck",       "cm", ".1f"),
        "shoulder_width_cm":  ("Shoulder W", "cm", ".1f"),
        "sleeve_length_cm":   ("Sleeve len", "cm", ".1f"),
        "inseam_cm":          ("Inseam",     "cm", ".1f"),
        "crotch_length_cm":   ("Crotch len", "cm", ".1f"),
        "front_rise_cm":      ("Front rise", "cm", ".1f"),
        "back_rise_cm":       ("Back rise",  "cm", ".1f"),
        "mass_kg":            ("Mass",       "kg", ".1f"),
    }

    print(f"\n{'Measurement':>12s}  {'Current':>8s}  {'Target':>8s}  {'Delta':>8s}")
    print(f"{'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for key, (label, unit, fmt) in display.items():
        cur = current.get(key)
        tgt = target.get(key)
        # Cross-match hip_cm <-> hips_cm (MHR uses "hip", Anny uses "hips")
        if tgt is None and key == "hip_cm":
            tgt = target.get("hips_cm")
        elif tgt is None and key == "hips_cm":
            tgt = target.get("hip_cm")
        if cur is None:
            continue

        cur_str = f"{cur:{fmt}} {unit}"
        if tgt is not None:
            delta = cur - tgt
            sign = "+" if delta > 0 else ""
            tgt_str = f"{tgt:{fmt}} {unit}"
            delta_str = f"{sign}{delta:{fmt}} {unit}"
        else:
            tgt_str = "   —"
            delta_str = "   —"

        print(f"{label:>12s}  {cur_str:>8s}  {tgt_str:>8s}  {delta_str:>8s}")


# ── Rendering (pyrender GPU) ──

def _camera_pose(elev_deg, azim_deg, distance, target):
    """Convert matplotlib-style elev/azim to a 4×4 camera-to-world pose matrix.

    Matches matplotlib's view_init(elev, azim) convention so rendered views
    are identical to the old Poly3DCollection renderer.
    """
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    eye = target + distance * np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn
    actual_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = actual_up
    pose[:3, 2] = -forward  # OpenGL: camera looks along -Z
    pose[:3, 3] = eye
    return pose


def _project_3d_to_2d(pts_3d, cam_pose, xmag, ymag, w, h):
    """Project 3D world points to 2D pixel coords via orthographic camera."""
    pose_inv = np.linalg.inv(cam_pose)
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    pts_cam = (pose_inv @ pts_h.T).T
    px = (pts_cam[:, 0] / xmag + 1) / 2 * w
    py = (1 - pts_cam[:, 1] / ymag) / 2 * h
    return px, py


def _render_views_pyrender(mesh, views, center, height):
    """Render multiple views of a mesh using pyrender (GPU via EGL).

    Returns list of (image, cam_pose) tuples and (xmag, ymag) camera params.
    """
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.65, 0.55, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,  # fully rough = diffuse only, matches old flat shading
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

    view_w, view_h = 500, 900
    ymag = height / 2 * 1.15
    xmag = ymag * (view_w / view_h)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    renderer = pyrender.OffscreenRenderer(view_w, view_h)
    results = []
    for _, elev, azim in views:
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=[0.3, 0.3, 0.3],
        )
        scene.add(pr_mesh)
        cam_pose = _camera_pose(elev, azim, 3.0, center)
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
        scene.add(light, pose=cam_pose)
        color, _ = renderer.render(scene)
        results.append((color, cam_pose))
    renderer.delete()

    return results, xmag, ymag, view_w, view_h


def _find_contour_centroids_at_z(mesh, z, filter_fn=None):
    """Find 3D centroids of mesh cross-section contours at a given Z height."""
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []
    try:
        path2d, to_3D = section.to_2D()
    except Exception:
        return []
    centroids = []
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        if len(pts) < 3:
            continue
        if filter_fn is not None and not filter_fn(pts):
            continue
        pts_h = np.column_stack([pts, np.zeros(len(pts)), np.ones(len(pts))])
        pts_3d = (to_3D @ pts_h.T).T[:, :3]
        centroids.append(pts_3d.mean(axis=0))
    return centroids


def _perpendicular_limb_contour(mesh, centroid, axis, max_dist=0.10):
    """Slice mesh perpendicular to axis at centroid, return closest 3D contour.

    Applies ConvexHull to produce a clean polygon (like torso contours)
    instead of dense raw section points.
    """
    section = mesh.section(plane_origin=centroid, plane_normal=axis)
    if section is None:
        return None
    try:
        path2d, to_3D = section.to_2D()
    except Exception:
        return None
    best_pts = None
    best_dist = float('inf')
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        except Exception:
            hull_pts = pts
        pts_h = np.column_stack([hull_pts, np.zeros(len(hull_pts)),
                                 np.ones(len(hull_pts))])
        pts_3d = (to_3D @ pts_h.T).T[:, :3]
        dist = np.linalg.norm(pts_3d.mean(axis=0) - centroid)
        if dist < max_dist and dist < best_dist:
            best_dist = dist
            best_pts = pts_3d
    return best_pts


def _estimate_axes_and_extract(mesh, z, contour_filter, delta_z=0.04,
                                max_dist=0.10):
    """Estimate limb axes from centroids at z +/- delta, extract perpendicular contours."""
    c_lo = _find_contour_centroids_at_z(mesh, z - delta_z, contour_filter)
    c_hi = _find_contour_centroids_at_z(mesh, z + delta_z, contour_filter)
    c_mid = _find_contour_centroids_at_z(mesh, z, contour_filter)

    if not c_mid or not c_lo or not c_hi:
        return None

    contours = []
    for cm in c_mid:
        x_sign = np.sign(cm[0])

        same_lo = [c for c in c_lo if np.sign(c[0]) == x_sign]
        same_hi = [c for c in c_hi if np.sign(c[0]) == x_sign]

        if not same_lo or not same_hi:
            continue

        cl = min(same_lo, key=lambda c: np.linalg.norm(c - cm))
        ch = min(same_hi, key=lambda c: np.linalg.norm(c - cm))

        axis = ch - cl
        norm = np.linalg.norm(axis)
        if norm < 0.001:
            continue
        axis = axis / norm

        pts = _perpendicular_limb_contour(mesh, cm, axis, max_dist)
        if pts is not None:
            contours.append(pts)

    return contours if contours else None


def _extract_limb_contours_3d(mesh, z, max_x_extent=0.20, min_x_offset=0.15):
    """Extract 3D arm contour points using perpendicular plane slicing.

    Estimates arm axis direction from horizontal slice centroids with asymmetric
    deltas. Falls back to horizontal slicing if axis estimation fails.

    Returns list of 3D point arrays (one per arm contour found).
    """
    def arm_filter(pts):
        x_extent = pts[:, 0].max() - pts[:, 0].min()
        x_center = (pts[:, 0].max() + pts[:, 0].min()) / 2
        return x_extent <= max_x_extent and abs(x_center) >= min_x_offset

    c_lo = _find_contour_centroids_at_z(mesh, z - 0.10, arm_filter)
    c_mid = _find_contour_centroids_at_z(mesh, z, arm_filter)

    if c_mid and c_lo:
        contours = []
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
            if pts is not None:
                contours.append(pts)
        if contours:
            return contours

    # Fallback: horizontal slice
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []
    try:
        path2d, to_3D = section.to_2D()
    except Exception:
        return []
    limb_contours = []
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        if len(pts) < 3:
            continue
        x_extent = pts[:, 0].max() - pts[:, 0].min()
        x_center = (pts[:, 0].max() + pts[:, 0].min()) / 2
        if x_extent > max_x_extent or abs(x_center) < min_x_offset:
            continue
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        except Exception:
            hull_pts = pts
        pts_h = np.column_stack([hull_pts, np.zeros(len(hull_pts)),
                                 np.ones(len(hull_pts))])
        pts_3d = (to_3D @ pts_h.T).T[:, :3]
        limb_contours.append(pts_3d)
    return limb_contours


def _extract_thigh_contours_3d(mesh, z):
    """Extract 3D thigh contour points using perpendicular plane slicing.

    Estimates leg axis direction from horizontal slice centroids at z +/- delta,
    then re-slices perpendicular to each leg's axis.
    Falls back to horizontal slicing if axis estimation fails.

    Returns list of 3D point arrays for contours that look like legs.
    """
    def thigh_filter(pts):
        circ = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        return circ > 0.10

    result = _estimate_axes_and_extract(mesh, z, thigh_filter, delta_z=0.03,
                                         max_dist=0.15)
    if result is not None and len(result) >= 2:
        centers = [c.mean(axis=0) for c in result]
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if centers[i][0] * centers[j][0] < 0:
                    return [result[i], result[j]]

    # Fallback: horizontal slice
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return []
    try:
        path2d, to_3D = section.to_2D()
    except Exception:
        return []
    candidates = []
    for entity in path2d.entities:
        pts = path2d.vertices[entity.points]
        if len(pts) < 3:
            continue
        x_center = (pts[:, 0].max() + pts[:, 0].min()) / 2
        circ = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        pts_h = np.column_stack([pts, np.zeros(len(pts)), np.ones(len(pts))])
        pts_3d = (to_3D @ pts_h.T).T[:, :3]
        candidates.append((circ, x_center, pts_3d))
    candidates.sort(key=lambda c: -c[0])
    if len(candidates) >= 2:
        c1, xc1, p1 = candidates[0]
        c2, xc2, p2 = candidates[1]
        if xc1 * xc2 < 0 and min(c1, c2) > 0.5 * max(c1, c2):
            return [p1, p2]
    return []


def extract_measurement_contours(mesh, measurements, torso_mesh=None):
    """Extract 3D measurement contour polylines from a body mesh.

    Computes closed 3D polylines at each measurement height (bust, waist, hip,
    thigh, upperarm). These are the same contours used by render_4view().

    Args:
        mesh: full body trimesh (Z-up, metres)
        measurements: dict from measure_body/measure_mhr (needs _*_z keys)
        torso_mesh: torso-only mesh for bust contour (Anny); None → use full mesh

    Returns:
        dict mapping measurement name → list of (N, 3) numpy arrays.
        Each array is a closed 3D polyline (convex hull contour at that height).
        Thigh and upperarm have 2 arrays each (left/right).
    """
    contours = {}

    # Plane-sweep contours for bust, underbust, waist, stomach, hip
    for name in ["bust", "underbust", "waist", "stomach", "hip"]:
        z = measurements.get(f"_{name}_z", 0)
        if z > 0:
            if name == "hip":
                cmesh, max_x, combine = mesh, 0.60, False
            elif name in ("bust", "underbust"):
                cmesh = torso_mesh or mesh
                max_x = MAX_TORSO_X_EXTENT
                combine = torso_mesh is not None
            else:  # waist, stomach
                cmesh = torso_mesh or mesh
                max_x = 0.60
                combine = torso_mesh is not None
            _, pts_3d = torso_circumference_at_z(
                cmesh, z, max_x_extent=max_x, return_contour=True,
                combine_fragments=combine)
            if pts_3d is not None:
                contours[name] = [pts_3d]

    # Thigh contours (two legs) via perpendicular plane slicing
    thigh_z = measurements.get("_thigh_z", 0)
    if thigh_z > 0:
        thigh_pts = _extract_thigh_contours_3d(mesh, thigh_z)
        if thigh_pts:
            contours["thigh"] = thigh_pts

    # Knee contours (two legs) — same extraction as thigh at knee height
    knee_z = measurements.get("_knee_z", 0)
    if knee_z > 0:
        knee_pts = _extract_thigh_contours_3d(mesh, knee_z)
        if knee_pts:
            contours["knee"] = knee_pts

    # Calf contours (two legs) — same extraction as thigh at calf height
    calf_z = measurements.get("_calf_z", 0)
    if calf_z > 0:
        calf_pts = _extract_thigh_contours_3d(mesh, calf_z)
        if calf_pts:
            contours["calf"] = calf_pts

    # Upperarm contours (two arms) via perpendicular plane slicing
    arm_z = measurements.get("_upperarm_z", 0)
    if arm_z > 0:
        arm_pts = _extract_limb_contours_3d(mesh, arm_z)
        if arm_pts:
            contours["upperarm"] = arm_pts

    # Wrist contours (two arms) — smaller limbs, tighter filter
    wrist_z = measurements.get("_wrist_z", 0)
    if wrist_z > 0:
        wrist_pts = _extract_limb_contours_3d(
            mesh, wrist_z, max_x_extent=0.12, min_x_offset=0.10)
        if wrist_pts:
            contours["wrist"] = wrist_pts

    # Neck contour — reuse perpendicular contour from measurement if available,
    # otherwise fall back to horizontal slice at _neck_z.
    neck_pts = measurements.get("_neck_contour_pts")
    if neck_pts is not None:
        contours["neck"] = [neck_pts]
    else:
        neck_z = measurements.get("_neck_z", 0)
        if neck_z > 0:
            _, pts_3d = torso_circumference_at_z(
                mesh, neck_z, max_x_extent=0.25, return_contour=True)
            if pts_3d is not None:
                contours["neck"] = [pts_3d]

    return contours


def render_4view(mesh, measurements, output_path, title="", model_label="",
                 torso_mesh=None):
    """Render 4-view body mesh with measurement contours overlaid.

    Uses pyrender (GPU via EGL) for mesh rendering (~0.1s for 4 views)
    and matplotlib 2D for contour overlay + title compositing.

    Args:
        mesh: full body trimesh
        measurements: dict from measure_body/measure_mhr
        output_path: PNG file path
        title: body name (e.g. "aro")
        model_label: "Anny" or "MHR" (for the title)
        torso_mesh: torso-only mesh for bust contour (Anny); None → use full mesh
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Install clad-body[render] for rendering: pip install 'clad-body[render]'"
        )

    verts = np.array(mesh.vertices)

    contours = extract_measurement_contours(mesh, measurements,
                                            torso_mesh=torso_mesh)

    views = [
        ("Front", 10, -90),
        ("Side (R)", 10, 0),
        ("Back", 10, 90),
        ("3/4 View", 20, -60),
    ]

    # Render mesh views with pyrender (GPU)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    height = verts[:, 2].max() - verts[:, 2].min()
    rendered, xmag, ymag, vw, vh = _render_views_pyrender(
        mesh, views, center, height)

    # Compose with matplotlib 2D
    fig, axes = plt.subplots(1, 4, figsize=(22, 9))
    for i, ((img, cam_pose), (view_title, _, _)) in enumerate(
            zip(rendered, views)):
        ax = axes[i]
        ax.imshow(img)

        # Overlay projected contour lines (closed)
        for name, pts_list in contours.items():
            color = CONTOUR_COLORS[name]
            for pts in pts_list:
                loop = np.vstack([pts, pts[:1]])
                px, py = _project_3d_to_2d(
                    loop, cam_pose, xmag, ymag, vw, vh)
                ax.plot(px, py, c=color, linewidth=2.5)

        # Overlay linear measurement polylines (open, view-filtered).
        # No depth testing on 2D overlay → only show each line from views
        # where it's naturally visible (not occluded by the body).
        _linear_cfg = {
            "shoulder_width": ("cyan", {"Front", "Back", "Side (R)", "3/4 View"}),
            "sleeve_length": ("magenta", {"Back", "Side (R)"}),
            "inseam": ("yellow", {"Side (R)", "3/4 View"}),
            "crotch_front": ("lime", {"Side (R)", "Front"}),
            "crotch_back": ("lime", {"Side (R)", "Back"}),
        }
        for name, pts in measurements.get("_linear_polylines", {}).items():
            cfg = _linear_cfg.get(name)
            if cfg is None or view_title not in cfg[1]:
                continue
            px, py = _project_3d_to_2d(pts, cam_pose, xmag, ymag, vw, vh)
            ax.plot(px, py, c=cfg[0], linewidth=2.5)

        # Overlay debug joint dots (if present)
        _joint_colors = {
            "l_shoulder": "red", "r_shoulder": "red",
            "l_elbow": "blue", "r_elbow": "blue",
            "l_wrist": "green", "r_wrist": "green",
            "c7": "white",
        }
        for jname, jpos in measurements.get("_debug_joints", {}).items():
            color = _joint_colors.get(jname)
            if color is None:
                continue
            px, py = _project_3d_to_2d(
                jpos.reshape(1, 3), cam_pose, xmag, ymag, vw, vh)
            ax.plot(px, py, 'o', c=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=0.5)

        ax.set_title(view_title)
        ax.axis("off")

    # Build legend with all measurements
    meas_labels = []
    for name, keys in [("Bust", ["bust_cm"]), ("Waist", ["waist_cm"]),
                        ("Stomach", ["stomach_cm"]),
                        ("Hips", ["hips_cm", "hip_cm"]),
                        ("Thigh", ["thigh_cm"]), ("Knee", ["knee_cm"]),
                        ("Calf", ["calf_cm"]),
                        ("Arm", ["upperarm_cm"]), ("Wrist", ["wrist_cm"]),
                        ("Neck", ["neck_cm"]),
                        ("Shoulder W", ["shoulder_width_cm"]),
                        ("Sleeve", ["sleeve_length_cm"]),
                        ("Crotch", ["crotch_length_cm"])]:
        val = 0
        for key in keys:
            val = measurements.get(key, 0)
            if val > 0:
                break
        if val > 0:
            meas_labels.append(f"{name}: {val:.1f}cm")
    legend = " | ".join(meas_labels)

    label_prefix = f"{model_label} Measurements" if model_label else "Measurements"
    plt.suptitle(
        f'{label_prefix}{" — " + title if title else ""}\n'
        f'Height: {measurements["height_cm"]:.1f}cm | {legend}',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
