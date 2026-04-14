"""Linear body measurements — shoulder width, sleeve length, inseam, crotch, shirt.

All length measurements follow ISO 8559-1 conventions. Surface-projected
polylines are computed for rendering overlay.
"""
from __future__ import annotations

import numpy as np

from ._slicer import MeshSlicer


def extract_joints_from_names(joint_names, heads, joint_map, tails=None):
    """Map model-specific joint names to canonical joint positions.

    Each map entry is ``canonical_name -> [candidate, ...]`` where each
    candidate is either:
      - a plain string  → bone head position (default)
      - ``(name, "head")``     → bone head position (explicit)
      - ``(name, "tail")``     → bone tail position (requires tails)
      - ``(name, "midpoint")`` → average of head and tail (requires tails)

    Args:
        joint_names: list of joint name strings from the body model.
        heads: (N, 3) array of bone head positions (Z-up, metres).
        joint_map: dict mapping canonical name -> list of candidates (see above).
        tails: optional (N, 3) array of bone tail positions; required for
            "tail" and "midpoint" candidates.

    Returns:
        dict mapping canonical name -> (3,) numpy array, or empty dict on failure.
    """
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    result = {}
    for canon_name, candidates in joint_map.items():
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
                continue  # tails unavailable for tail/midpoint — try next candidate
            break
    return result


_BACK_TAPE_HALF_WIDTH = 0.025  # metres — half a 5-cm-wide bridging strip


def _interp_y_at_x(half, x_target):
    """Linear interpolation of y at `x_target` between the two points on
    `half` whose x values straddle `x_target` (closest below + closest
    above).  Returns None if one side is empty.
    """
    below = half[half[:, 0] <= x_target]
    above = half[half[:, 0] >= x_target]
    if len(below) == 0 or len(above) == 0:
        return None
    left = below[np.argmax(below[:, 0])]   # closest from below
    right = above[np.argmin(above[:, 0])]  # closest from above
    dx = right[0] - left[0]
    if dx < 1e-9:
        return float((left[1] + right[1]) / 2)
    t = (x_target - left[0]) / dx
    return float(left[1] + t * (right[1] - left[1]))


def _front_y_sagittal(contour_pts):
    """Front-surface y at x=0 on a body cross-section.

    For the front (abdomen / pubic) side, a tape measure physically
    follows the surface at the centerline — there is no midline
    concavity the tape would bridge over.  The belly is a smooth
    convex curve whose crossing of the ``x=0`` plane is the centerline
    surface point, which is exactly what this function returns.

    We do **not** offset off-axis on the front, because near the
    perineum the front half of a butterfly-shaped cross-section
    contains inner-thigh peninsulas at |x|≈1 cm that are not on the
    belly arc.  An off-axis interpolation can jump across a peninsula
    and land on a wrong surface — ``x=0`` always stays on the belly
    arc because the belly is the feature that crosses x=0.

    Args:
        contour_pts: (N, 2) XY points making up one closed body contour.

    Returns:
        Interpolated y at x=0 on the front half, or None.
    """
    cy = contour_pts[:, 1].mean()
    front = contour_pts[contour_pts[:, 1] < cy]
    if len(front) < 2:
        return None
    return _interp_y_at_x(front, 0.0)


def _back_y_tape_bridge(contour_pts):
    """Back-surface y averaged at x = ±2.5 cm on a body cross-section.

    Models a tailor's tape as a flat 5-cm-wide strip pressed against
    the body's back surface.  Instead of asking "where does the
    surface cross x=0" (which would dive into the gluteal cleft), we
    ask "where does the surface cross the lines x=+2.5 cm and
    x=-2.5 cm" and average.  The 2.5-cm offset puts the sampling
    columns well outside the gluteal cleft and near the cheek peaks —
    the outermost back surface a real tape would rest on when pulled
    across the buttocks.

    Why this works on the back but not on the front:

      * The **back** near the perineum has a clean topology — the
        gluteal cheeks curve monotonically from the cleft outwards.
        A 1-cm-off-axis interpolation lands on the cheek arc on each
        side.
      * The **front** near the perineum has a butterfly topology —
        inner-thigh peninsulas can poke up to |x|≈1 cm close to the
        centroid, so a 1-cm interpolation can jump across a peninsula
        and pick a wrong surface point.  Use :func:`_front_y_sagittal`
        for the front.

    Args:
        contour_pts: (N, 2) XY points making up one closed body contour.

    Returns:
        Average of y at x=+1 cm and y at x=-1 cm on the back half,
        or None if both sides fail.
    """
    cy = contour_pts[:, 1].mean()
    back = contour_pts[contour_pts[:, 1] > cy]
    if len(back) < 2:
        return None
    y_neg = _interp_y_at_x(back, -_BACK_TAPE_HALF_WIDTH)
    y_pos = _interp_y_at_x(back, +_BACK_TAPE_HALF_WIDTH)
    if y_neg is None and y_pos is None:
        return None
    if y_neg is None:
        return y_pos
    if y_pos is None:
        return y_neg
    return 0.5 * (y_neg + y_pos)


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


def c7_surface_point(verts, c7):
    """Project the C7 bone position onto the posterior (back) skin surface.

    C7 bone sits inside the body. This finds the nearest back-surface point
    at the same X and Z, using progressively wider search bands. Returns a
    (3,) point on the skin, or None if the surface cannot be found.

    This is the canonical projection used by both the shoulder-width arc and
    back-neck-to-waist, so both measurements share the exact same start point.

    Args:
        verts: (V, 3) mesh vertices (Z-up, metres)
        c7: (3,) C7 bone position (Z-up, metres)

    Returns:
        (3,) surface point (Z-up, metres) or None.
    """
    for x_tol, z_tol in [(0.03, 0.01), (0.04, 0.03), (0.06, 0.06)]:
        y = _surface_y_at(verts, float(c7[0]), float(c7[2]),
                          x_tol=x_tol, z_tol=z_tol, side="back")
        if y is not None:
            return np.array([float(c7[0]), y, float(c7[2])], dtype=np.float64)
    return None


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
    # position is inside the body. Use c7_surface_point so the arc and
    # back-neck-to-waist share the identical start landmark.
    wp_y = [float(r_acromion[1]), None, float(l_acromion[1])]
    c7_surf = c7_surface_point(verts, c7)
    wp_y[1] = float(c7_surf[1]) if c7_surf is not None else float(c7[1])

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


def measure_sleeve_length_from_joints(joints, upperarm_loop_cm):
    """ISO 8559-1 sleeve length — fully differentiable through Anny LBS.

    ISO 8559-1 §5.4.14 + §5.4.15: distance from shoulder point (acromion)
    over the elbow (olecranon, with arm slightly bent) to the wrist (ulnar
    styloid), measured along the body surface as a tape arc.

    The slow ISO reference (`measure_sleeve_length_iso_reference`) computes
    this exactly: it poses the body in rest pose with elbow flexed ~42°,
    detects acromion / olecranon / wrist styloid landmarks via skinning
    weights and bone-perpendicular geometry, slices the body with two
    planes (one through the upper-arm bone + acromion + olecranon, one
    through the forearm bone + olecranon + wrist styloid), and walks
    Dijkstra shortest paths along the resulting contours. None of that is
    differentiable.

    This function is the FAST DIFFERENTIABLE APPROXIMATION calibrated
    against that reference:

        sleeve_iso_cm ≈ bone_chain_cm + a*upperarm_loop_cm + bias

    where bone_chain_cm is the sum of upper arm + forearm bone lengths
    (pose-invariant: same in A-pose and rest pose), and upperarm_loop_cm
    is the differentiable vertex-loop circumference of the upperarm.

    Calibration: 2-param least squares on the 6 anny testdata bodies vs
    the surface-walk reference. RMS = 0.33 cm, max = 0.55 cm.

    The arithmetic is type-polymorphic — pass numpy arrays for reporting,
    torch tensors for gradient-based optimization.

    Args:
        joints: dict with 'l_shoulder_ball', 'r_shoulder_ball', 'l_elbow',
                'r_elbow', 'l_wrist', 'r_wrist' as (3,) arrays/tensors
                (Z-up, metres). The shoulder anchors MUST be the
                upperarm01.head ball joints, NOT the legacy 'l_shoulder'
                bone-tail anchors used by measure_sleeve_length.
        upperarm_loop_cm: differentiable vertex-loop circumference of the
                          upperarm in cm
                          (= compute_loop_circumference(verts, BASE_MESH_UPPERARM_VERTICES) * 100)

    Returns:
        sleeve_length_cm — scalar (numpy float or torch tensor depending
        on input dtype). Returns 0 if any joint is missing.
    """
    sums = []
    for side in ("l", "r"):
        sh = joints.get(f"{side}_shoulder_ball")
        el = joints.get(f"{side}_elbow")
        wr = joints.get(f"{side}_wrist")
        if sh is None or el is None or wr is None:
            continue
        sums.append(_norm(sh - el) + _norm(el - wr))

    if not sums:
        return 0
    bone_chain_m = sum(sums) / len(sums)
    bone_chain_cm = bone_chain_m * 100

    # Empirical fit (n=6, RMS=0.33 cm vs surface walk reference).
    # See findings/sleeve_length_iso_compliance.md for the calibration data.
    A_LOOP = 0.13803
    A_BIAS = 2.04621
    return bone_chain_cm + A_LOOP * upperarm_loop_cm + A_BIAS


def _norm(v):
    """Type-polymorphic 3-vector norm: numpy array or torch tensor."""
    if hasattr(v, "norm"):  # torch tensor
        return v.norm()
    return float(np.linalg.norm(v))


def find_side_neck_point(slicer, c7_z):
    """Find the ISO 8559-1 §3.1.7 side neck point (right side).

    Definition: crossing point of the neck base line and the anterior
    border of the trapezius muscle.

    Implementation: horizontal slice at ``c7_z`` (the neck base line)
    → rightmost contour point (maximum X > 0).  At this height the body
    cross-section is the upper-trapezius + neck mass; its lateral max-X
    sits exactly where the trapezius surface starts to slope down toward
    the acromion — the anterior trapezius border at the neck base line.

    Args:
        slicer: MeshSlicer pre-built on the body mesh (Z-up, metres).
        c7_z: C7 vertebra Z height in metres (neck base line).

    Returns:
        (3,) numpy array [X, Y, Z] in metres, or None if slice unavailable.
    """
    contours = slicer.contours_at_z(float(c7_z))
    if not contours:
        return None
    all_pts = np.vstack([pts for pts, _, _ in contours])
    right = all_pts[all_pts[:, 0] > 0]
    if len(right) == 0:
        return None
    lat = right[np.argmax(right[:, 0])]
    return np.array([float(lat[0]), float(lat[1]), float(c7_z)], dtype=np.float64)


def measure_shirt_length(joints, mesh, crotch_z, measurements=None, step=0.005):
    """Measure shirt length — side neck to crotch along front body contour.

    3DLook "jacket length": distance from Side Neck Point to the thigh
    centre at crotch level (ISO 8559-1 §3.1.7 + 3DLook definition).

    Algorithm:
      1. **Side neck point** — lateral max-X of the horizontal body
         contour at C7 height (``find_side_neck_point``).  This is the
         crossing point of the neck base line and the anterior border of
         the trapezius, per ISO 8559-1 §3.1.7.
      2. **Skip zone** — the 5 cm of Z immediately below the side neck
         are skipped.  The shoulder-cap topology transitions abruptly in
         this zone, producing spurious front-surface samples.  The lower
         convex hull bridges cleanly from the start to the first stable
         chest-level sample.
      3. **Front surface sweep** — from (c7_z − 5 cm) to 2 cm above
         crotch Z.  At each Z, take most-anterior Y within ±2.5 cm of
         ``side_neck_x``.
      4. **Lower convex hull ("not tight to skin")** — bridges
         concavities (waist dip), follows convex protrusions (bust, belly).
      5. **Single polyline** — measurement = sum of Euclidean segment
         lengths; polyline offset 0.5 mm anterior of skin for rendering.

    Args:
        joints: dict with 'c7' as (3,) array (metres, Z-up).
        mesh: trimesh of the full body (Z-up, metres).
        crotch_z: Z coordinate of crotch (from measure_inseam), in metres.
        measurements: unused (kept for API compatibility).
        step: vertical step between sample planes (metres, default 5 mm).

    Returns:
        (shirt_length_cm, polyline) — cm value and (N, 3) float32 array,
        or (0, None) on failure.
    """
    c7 = joints.get("c7")
    if c7 is None or crotch_z <= 0:
        return 0, None

    slicer = MeshSlicer(mesh)
    c7_z = float(c7[2])

    # ── Side neck point (ISO 8559-1 §3.1.7) ──
    side_neck = find_side_neck_point(slicer, c7_z)
    if side_neck is None:
        return 0, None
    snx = float(side_neck[0])
    sny = float(side_neck[1])
    snz = float(side_neck[2])

    if snz <= crotch_z:
        return 0, None

    # ── Front surface sweep ──
    # Skip 5 cm below side neck: shoulder-cap topology changes abruptly
    # there.  The lower convex hull bridges this gap with a single clean
    # diagonal from the trap top to the first stable chest-level sample.
    x_band = 0.025
    skip_zone = 0.05
    sweep_bottom = crotch_z + 0.02

    front_ys = [sny]
    valid_zs = [snz]

    for z in np.arange(snz - skip_zone - step, sweep_bottom - step / 2, -step):
        contours = slicer.contours_at_z(z)
        if not contours:
            continue
        nearby_pts = []
        for pts_xy, _, _ in contours:
            near = pts_xy[np.abs(pts_xy[:, 0] - snx) < x_band]
            if len(near) > 0:
                nearby_pts.append(near)
        if not nearby_pts:
            continue
        band = np.vstack(nearby_pts)
        front_ys.append(float(band[:, 1].min()))
        valid_zs.append(z)

    if len(valid_zs) < 2:
        return 0, None

    # ── Lower convex hull — "not tight to skin" ──
    n = len(valid_zs)
    z_arr = np.array(valid_zs)
    y_arr = np.array(front_ys)

    # Andrew's monotone chain (lower hull), Z-ascending pass.
    hull = []
    for i in range(n - 1, -1, -1):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = ((z_arr[a] - z_arr[o]) * (y_arr[i] - y_arr[o]) -
                     (y_arr[a] - y_arr[o]) * (z_arr[i] - z_arr[o]))
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    hull.reverse()

    hull_z = z_arr[hull]
    hull_y = y_arr[hull]
    tape_ys = np.interp(z_arr, hull_z[::-1], hull_y[::-1])

    # Offset 0.5 mm in front of skin so the line sits ON the surface
    tape_ys = np.minimum(tape_ys, y_arr) - 0.005

    # ── Single polyline for measurement + visualisation ──
    pts = np.column_stack([np.full(n, snx), tape_ys, z_arr])
    total_cm = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() * 100)

    return total_cm, pts.astype(np.float32)


def measure_back_neck_to_waist(joints, mesh, waist_z, step=0.005, c7_surface=None):
    """Measure back neck point to waist length — ISO 8559-1 §5.4.5.

    Distance from the cervicale (C7 vertebra prominens, "back neck point")
    down the centre back, following the body contour, to the waist level.
    The tape touches the skin and follows the curvature of the upper back
    and lumbar spine.

    Algorithm: starting from C7 (back-projected to skin), sweep down in Z
    from c7_z to waist_z. At each Z level, take the most-posterior (max Y)
    contour point near the midline (X≈0). Sum Euclidean segment lengths.

    Args:
        joints: dict with 'c7' as a (3,) array (metres, Z-up).
        mesh: trimesh body mesh (Z-up, metres).
        waist_z: waist Z height in metres (from waist measurement).
        step: vertical sampling interval in metres (default 5 mm).
        c7_surface: optional (3,) pre-computed C7 skin-surface point returned
            by measure_shoulder_width. When provided, skips the internal
            surface projection so the back-neck-to-waist line starts at the
            exact same landmark as the shoulder-width arc midpoint.

    Returns:
        (back_neck_to_waist_cm, polyline) where polyline is (N, 3) float32
        on the back body surface, or (0, None) if inputs are invalid.
    """
    c7 = joints.get("c7")
    if c7 is None or waist_z <= 0:
        return 0, None

    verts = np.array(mesh.vertices)
    c7_z = float(c7[2])
    if c7_z <= waist_z:
        return 0, None

    if c7_surface is not None:
        # Caller already projected c7 to the skin (from measure_shoulder_width).
        start_z = float(c7_surface[2])
        start_y = float(c7_surface[1])
    else:
        # Fallback: project C7 bone to nearest posterior midline vertex.
        near_c7 = verts[
            (np.abs(verts[:, 0]) < 0.04)
            & (np.abs(verts[:, 2] - c7_z) < 0.015)
        ]
        if len(near_c7) == 0:
            return 0, None
        start = near_c7[np.argmax(near_c7[:, 1])]  # most posterior
        start_z = float(start[2])
        start_y = float(start[1])

    slicer = MeshSlicer(mesh)
    x_band = 0.05  # midline half-width

    z_levels = np.arange(start_z, waist_z - step / 2, -step)
    if len(z_levels) < 2:
        return 0, None

    back_ys = [start_y]
    valid_zs = [start_z]
    for z in z_levels[1:]:
        contours = slicer.contours_at_z(z)
        if not contours:
            continue
        midline_pts = []
        for pts_xy, _, _ in contours:
            near = pts_xy[np.abs(pts_xy[:, 0]) < x_band]
            if len(near) > 0:
                midline_pts.append(near)
        if not midline_pts:
            continue
        midline = np.vstack(midline_pts)
        back_ys.append(float(midline[:, 1].max()))
        valid_zs.append(z)

    if len(valid_zs) < 2:
        return 0, None

    pts = np.column_stack([
        np.zeros(len(valid_zs)), np.array(back_ys), np.array(valid_zs)
    ])
    total_cm = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() * 100)
    return total_cm, pts.astype(np.float32)


# Anny perineum vertex pair — left/right symmetric, ~8 mm off the body
# centerline at the inguinal surface. Their height-from-floor tracks the
# ISO 8559-1 mesh-sweep crotch directly because they ARE the perineum
# surface, not a kinematic proxy. Stable across the parameter space:
# 118-case stress test (6 testdata bodies + 90 questionnaire grid + 12
# leg-length blendshape sweeps + 10 random local_changes perturbations)
# scored max 0.19 cm error, RMS 0.09 cm.
ANNY_PERINEUM_VERTEX_L = 6319
ANNY_PERINEUM_VERTEX_R = 12900


def measure_inseam_from_perineum_vertices(verts, height_axis):
    """ISO 8559-1 §5.1.15 inseam — height of the perineum vertex pair above the floor.

    Reads two stable Anny vertex indices on the inguinal surface (a left/right
    symmetric pair, ~8 mm off the body centerline). Differentiable through
    LBS+blendshapes by construction — the vertices ARE soft tissue, so any
    blendshape that moves the perineum moves them.

    This replaced an older bone-tail-plus-linear-correction formula
    (``upperleg01.tail.z + a*thigh + b*pelvis_w + c``) that drifted
    catastrophically (>10 cm) on bodies whose ``measure-{upper,lower}leg-
    height-incr`` blendshapes were pushed away from zero — the femur tail
    moved at only ~70 % the rate of the mesh perineum, and the calibration
    set never spanned the questionnaire body distribution. See
    ``findings/measure_inseam_from_joints_drift.md``.

    Validation: 118 stress-test bodies (testdata × leg-length blendshape
    sweeps + questionnaire grid + random local_changes) → max error
    0.19 cm vs the ISO mesh sweep, RMS 0.09 cm.

    Type-polymorphic — accepts a numpy array for reporting or a torch
    tensor for the gradient hot loop. Returns ``inseam_cm`` in the same
    type as ``verts``.

    Args:
        verts: ``(V, 3)`` or ``(1, V, 3)`` array/tensor of vertex positions
            in the raw Anny model frame (no XY-centering or floor alignment
            required — only Z-from-floor matters and is computed here).
        height_axis: ``1`` if Anny is in Y-up convention (vertical = old Y,
            with feet at max Y — same logic as ``_extract_anny_joints`` and
            ``load_anny_from_params``); ``2`` if already Z-up.

    Returns:
        ``inseam_cm`` — type-polymorphic scalar (preserves torch grad).
    """
    if verts.ndim == 3:
        verts = verts[0]

    # Project to 1D height-from-floor. Mirrors the Y-up→Z-up convention
    # used by load_anny_from_params and _extract_anny_joints: in the Y-up
    # frame, vertical = -Y (so feet at max Y become min "z"), and the
    # absolute height-from-floor is obtained by subtracting the new min.
    if height_axis == 1:  # Y-up
        v_z = -verts[:, 1]
    else:  # Z-up (Anny's current default)
        v_z = verts[:, 2]
    floor = v_z.min()
    z_l = v_z[ANNY_PERINEUM_VERTEX_L] - floor
    z_r = v_z[ANNY_PERINEUM_VERTEX_R] - floor
    crotch_z_m = (z_l + z_r) / 2
    return crotch_z_m * 100


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

    slicer = MeshSlicer(mesh)

    # Sample Z levels from waist down to crotch
    z_levels = np.arange(waist_z, crotch_z - step / 2, -step)
    if len(z_levels) < 2:
        return 0, 0, 0, None, None

    # At each Z, take the body cross-section's full contour and use its
    # min_y / max_y as front (abdomen) / back (lumbar / buttocks).  The
    # body contour is the unique closed contour with body-scale x-extent
    # — narrow enough to exclude torso+arm merges (>= 0.50 m) and wide
    # enough to exclude arm-only fragments and slicing-artifact islets
    # (< 0.10 m).
    #
    # Why min_y over the WHOLE contour, not a |x|<5cm midline strip:
    # the strip only contains ~50 of the contour's ~100 points, so a
    # single LBS-jitter point can dominate min_y and spike the trace
    # by 2 cm at one z (see findings/crotch_midline_artifact.md).
    # Sampling the full contour averages this out — min_y is now
    # robust to ~mm-scale floating-point jitter and is anatomically
    # the actual front of the body cross-section.
    #
    # Termination is topological: as soon as the slicer returns more
    # than one body-shaped contour, the legs have separated and we
    # have reached the perineum.  This avoids depending on
    # `crotch_z` rounding to land in the right `np.arange` bucket
    # (which previously made `crotch_length_cm` brittle to ~0.5 mm
    # perturbations of `crotch_z`).
    BODY_X_MIN = 0.10  # metres — exclude tiny artifact islets and arms
    BODY_X_MAX = 0.50  # metres — exclude torso+arm-merged contours
    BODY_XC_MAX = 0.10  # metres — body contour must be near the centerline
                       # (excludes arm contours which sit at |xc| > 0.4)

    front_ys = []
    back_ys = []
    valid_zs = []

    for z in z_levels:
        contours = slicer.contours_at_z(z)
        if not contours:
            continue

        body_contours = [
            pts for pts, xe, xc in contours
            if BODY_X_MIN < xe < BODY_X_MAX and abs(xc) < BODY_XC_MAX
        ]
        if len(body_contours) == 0:
            continue
        if len(body_contours) >= 2:
            # Legs separated → perineum reached, stop the trace.
            break

        body = body_contours[0]
        front_y = _front_y_sagittal(body)
        back_y = _back_y_tape_bridge(body)
        if front_y is None or back_y is None:
            continue
        front_ys.append(front_y)
        back_ys.append(back_y)
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
    # front_rise: front waist → perineum (visualises front_rise_cm)
    # back_rise:  back waist → perineum (visualises back_rise_cm)
    # crotch_length: combined U-path front waist → perineum → back waist
    #                (visualises crotch_length_cm = front_rise + back_rise)
    crotch_front = measurements.get("_crotch_front_pts")
    crotch_back = measurements.get("_crotch_back_pts")
    if crotch_front is not None:
        result["front_rise"] = crotch_front
    if crotch_back is not None:
        result["back_rise"] = crotch_back
    if crotch_front is not None and crotch_back is not None:
        result["crotch_length"] = np.concatenate([crotch_front, crotch_back[::-1]])

    # Shirt length: surface-traced polyline from measure_shirt_length().
    shirt_pts = measurements.get("_shirt_length_pts")
    if shirt_pts is not None:
        result["shirt_length"] = shirt_pts

    # Back neck point to waist (ISO 8559-1 §5.4.5).
    bnw_pts = measurements.get("_back_neck_to_waist_pts")
    if bnw_pts is not None:
        result["back_neck_to_waist"] = bnw_pts

    return result
