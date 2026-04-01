"""Rendering and comparison — 4-view body renders, contour extraction, I/O.

Provides pyrender GPU rendering (via EGL), contour extraction for
visualising body measurements, and target measurement comparison helpers.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial import ConvexHull

from ._slicer import (
    CONTOUR_COLORS,
    MAX_TORSO_X_EXTENT,
    MeshSlicer,
    _find_contour_centroids_at_z,
    _perpendicular_limb_contour,
    torso_circumference_at_z,
)


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
            "shirt_length": ("dodgerblue", {"Side (R)", "Front"}),
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
                        ("Hips", ["hip_cm"]),
                        ("Thigh", ["thigh_cm"]), ("Knee", ["knee_cm"]),
                        ("Calf", ["calf_cm"]),
                        ("Arm", ["upperarm_cm"]), ("Wrist", ["wrist_cm"]),
                        ("Neck", ["neck_cm"]),
                        ("Shoulder W", ["shoulder_width_cm"]),
                        ("Sleeve", ["sleeve_length_cm"]),
                        ("Crotch", ["crotch_length_cm"]),
                        ("Shirt", ["shirt_length_cm"])]:
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


# ── Target measurement I/O and comparison ────────────────────────────────────


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
        "shirt_length_cm":    ("Shirt len",  "cm", ".1f"),
        "mass_kg":            ("Mass",       "kg", ".1f"),
    }

    print(f"\n{'Measurement':>12s}  {'Current':>8s}  {'Target':>8s}  {'Delta':>8s}")
    print(f"{'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

    for key, (label, unit, fmt) in display.items():
        cur = current.get(key)
        tgt = target.get(key)
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
