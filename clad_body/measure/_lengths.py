"""Linear body measurements — shoulder width, sleeve length, inseam, crotch, shirt.

All length measurements follow ISO 8559-1 conventions. Surface-projected
polylines are computed for rendering overlay.
"""
from __future__ import annotations

import numpy as np

from ._slicer import MeshSlicer


def extract_joints_from_names(joint_names, joint_positions, joint_map):
    """Map model-specific joint names to canonical joint positions.

    Args:
        joint_names: list of joint name strings from the body model.
        joint_positions: (N, 3) array of joint positions (metres, Z-up).
        joint_map: dict mapping canonical name -> list of candidate model names.

    Returns:
        dict mapping canonical name -> (3,) numpy array, or empty dict on failure.
    """
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    result = {}
    for canon_name, candidates in joint_map.items():
        for cand in candidates:
            if cand in name_to_idx:
                result[canon_name] = joint_positions[name_to_idx[cand]]
                break
    return result


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


def measure_shirt_length(joints, mesh, crotch_z, measurements=None, step=0.005):
    """Measure shirt length — shoulder to crotch along front body contour.

    SAIA 3DLook "jacket length": distance from Side Neck to crotch level.
    Blue Eye Custom Tailor #12: from shoulder seam at collar to bottom of fly.

    Algorithm:
      1. **Start point** — shoulder bone projected to nearest skin vertex.
         Anny: ``side_neck`` (shoulder01.L head = acromioclavicular joint).
         MHR: midpoint of C7 and ``l_shoulder``.
         The hit vertex's X becomes ``trace_x`` for the whole sweep.
      2. **Front surface sweep** — from 5 cm below start Z to 2 cm above
         crotch Z (5 mm steps).  At each Z, take most-anterior Y within
         ±4 cm of ``trace_x``.
      3. **Convex hull ("not tight to skin")** — lower convex hull of the
         (Z, Y) profile via Andrew's monotone chain.  Bridges concavities,
         follows convex curvature back toward the body after peaks.
      4. **Single path** — same polyline for measurement and visualisation.
         Measurement = sum of Euclidean segment lengths.  Polyline offset
         0.5 mm in front of skin for rendering.

    Args:
        joints: dict with 'c7' and optionally 'side_neck' / 'l_shoulder'.
        mesh: trimesh of the full body (Z-up, metres).
        crotch_z: Z coordinate of crotch (from measure_inseam), in metres.
        measurements: dict from the measurement pipeline (for shoulder arc).
        step: vertical step between sample planes (metres, default 5 mm).

    Returns:
        (shirt_length_cm, polyline) — cm value and (N, 3) float32 array,
        or (0, None) on failure.
    """
    c7 = joints.get("c7")
    if c7 is None or crotch_z <= 0:
        return 0, None

    verts = np.array(mesh.vertices)
    if measurements is None:
        measurements = {}

    # ── Shoulder bone → nearest skin vertex = start point ──
    # Anny: side_neck (acromioclavicular joint).
    # MHR: midpoint of C7 and l_shoulder (collar-shoulder junction).
    # Fallback: neck radius X at C7 - 3% height.
    slicer = MeshSlicer(mesh)
    side_neck_x = 0.055  # default neck radius
    c7_contours = slicer.contours_at_z(float(c7[2]))
    if c7_contours:
        all_pts = np.vstack([pts for pts, _, _ in c7_contours])
        neck_pts = all_pts[(all_pts[:, 0] > 0) & (all_pts[:, 0] < 0.08)]
        if len(neck_pts) > 0:
            side_neck_x = float(neck_pts[:, 0].max())

    shoulder_bone = joints.get("side_neck")
    if shoulder_bone is None:
        l_sh = joints.get("l_shoulder")
        if l_sh is not None:
            shoulder_bone = (c7 + l_sh) / 2
    if shoulder_bone is not None:
        bone_x = float(shoulder_bone[0])
        bone_y = float(shoulder_bone[1])
        bone_z = float(shoulder_bone[2])
    else:
        bone_x = side_neck_x
        bone_y = float(c7[1])
        bone_z = float(c7[2]) - 0.03 * verts[:, 2].max()

    # Project to nearest skin vertex
    bone_pos = np.array([bone_x, bone_y, bone_z])
    hit = verts[np.argmin(np.linalg.norm(verts - bone_pos, axis=1))]
    snz = float(hit[2])
    trace_x = float(hit[0])
    start_y = float(hit[1])

    if snz <= crotch_z:
        return 0, None

    # ── Front surface sweep (5 cm below shoulder → 2 cm above crotch) ──
    # Start point is prepended; hull bridges from shoulder to first bust contact.
    x_band = 0.04
    sweep_top = snz - 0.05
    sweep_bottom = crotch_z + 0.02
    z_levels = np.arange(sweep_top, sweep_bottom - step / 2, -step)

    front_ys = [start_y]
    valid_zs = [snz]

    for z in z_levels:
        contours = slicer.contours_at_z(z)
        if not contours:
            continue
        nearby_pts = []
        for pts_xy, _, _ in contours:
            near = pts_xy[np.abs(pts_xy[:, 0] - trace_x) < x_band]
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
    # Tape touches convex protrusions (chest, belly) and bridges
    # concavities (dip between them).  After a peak the tape follows the
    # outgoing convex curvature back toward the body.
    n = len(valid_zs)
    z_arr = np.array(valid_zs)   # Z descending (top → bottom)
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
    pts = np.column_stack([np.full(n, trace_x), tape_ys, z_arr])
    total_cm = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() * 100)

    return total_cm, pts.astype(np.float32)


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

    # Shirt length: surface-traced polyline from measure_shirt_length().
    shirt_pts = measurements.get("_shirt_length_pts")
    if shirt_pts is not None:
        result["shirt_length"] = shirt_pts

    return result
