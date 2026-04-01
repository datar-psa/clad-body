"""Mesh slicing infrastructure — fast vectorized slicer and geometry helpers.

Provides MeshSlicer for horizontal plane sweeps, torso_circumference_at_z for
trimesh-based slicing, and perpendicular contour extraction helpers used by
circumference and length measurement modules.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import ConvexHull

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


# ── Geometry helpers (used by circumference and length modules) ──

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
