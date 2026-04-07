#!/usr/bin/env python3
"""
Measure MHR body mesh — extract anthropometric measurements via mesh-plane sweep.

Uses horizontal plane slicing (ISO 8559-1 compliant) to find circumferences:
- Bust: maximum horizontal circumference in bust region (torso-only, arms filtered)
- Waist: circumference at ISO 8559-1 waist level (61% of height)
- Hip: maximum horizontal circumference in hip region
- Thigh: maximum average of 2 separate leg contours (30-37% height)
- Upper arm: maximum average of 2 arm contours (72-80% height)

Usage:
    python -m clad_body.measure.mhr path/to/params.json
"""

import argparse
import os
import sys

import numpy as np
import trimesh

from clad_body.load.mhr import load_mhr_from_params

from clad_body.measure._slicer import (
    CONTOUR_COLORS,
    MAX_TORSO_X_EXTENT,
    MeshSlicer,
    REGIONS,
    WAIST_HEIGHT_PCT,
    torso_circumference_at_z,
)
from clad_body.measure._circumferences import (
    body_signature,
    find_measurement,
    measure_calf,
    measure_knee,
    measure_thigh,
    measure_upperarm,
    measure_wrist,
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
from clad_body.measure._render import load_target_measurements, print_comparison

# Canonical joint name mapping for MHR linear measurements.
MHR_JOINT_MAP = {
    "c7": ["c_neck"],        # y=144 cm (~85% height) — cervicothoracic junction
    "l_shoulder": ["l_uparm"],
    "r_shoulder": ["r_uparm"],
    "l_elbow": ["l_lowarm"],
    "r_elbow": ["r_lowarm"],
    "l_wrist": ["l_wrist"],
    "r_wrist": ["r_wrist"],
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_acromion_candidates(verts, x, z, x_reach, z_lo, side):
    """Search for shoulder cap vertices on one side."""
    if side == "left":
        mask = (
            (verts[:, 0] >= x) &
            (verts[:, 0] <= x + x_reach) &
            (verts[:, 2] >= z + z_lo) &
            (verts[:, 2] < z + 0.05)
        )
    else:
        mask = (
            (verts[:, 0] <= x) &
            (verts[:, 0] >= x - x_reach) &
            (verts[:, 2] >= z + z_lo) &
            (verts[:, 2] < z + 0.05)
        )
    return verts[mask]


def find_acromion(verts, shoulder_joint, side="left"):
    """Find acromion for MHR — most lateral + highest point on shoulder cap.

    MHR shoulder joints (l_uparm/r_uparm) are bone heads deep inside the
    glenohumeral socket. Two-step search: (1) find most lateral surface
    vertex (max |X|) up to 5cm outward from the bone, then (2) refine to
    the highest Z within 1cm of that lateral edge.

    Uses a tight Z band (-3cm) first; widens to -5cm if no candidates
    are found (some bodies have the shoulder cap further below the joint).

    Args:
        verts: (V, 3) vertex array (Z-up, metres, floor-aligned)
        shoulder_joint: (3,) bone head position (inside body)
        side: "left" (positive X) or "right" (negative X)
    """
    x = shoulder_joint[0]
    z = shoulder_joint[2]
    x_reach = 0.05

    # Try tight band first, widen if needed
    candidates = _find_acromion_candidates(verts, x, z, x_reach, -0.03, side)
    if len(candidates) == 0:
        candidates = _find_acromion_candidates(verts, x, z, x_reach, -0.05, side)
    if len(candidates) == 0:
        return shoulder_joint.copy()

    # Most lateral, then highest Z near that edge
    if side == "left":
        edge_x = candidates[:, 0].max()
        near_edge = candidates[candidates[:, 0] >= edge_x - 0.01]
    else:
        edge_x = candidates[:, 0].min()
        near_edge = candidates[candidates[:, 0] <= edge_x + 0.01]
    return near_edge[np.argmax(near_edge[:, 2])].copy()


def measure_mhr(mesh_or_body, render_path=None, title=""):
    """Extract body measurements from MHR mesh.

    .. deprecated::
        Use ``clad_body.measure.measure(body)`` instead::

            from clad_body.load import load_mhr_from_params
            from clad_body.measure import measure
            body = load_mhr_from_params("path/to/params.json")
            m = measure(body)

    Args:
        mesh_or_body: trimesh.Trimesh or MhrBody
        render_path: if set, save 4-view render with measurement contours
        title: title for the 4-view render

    Returns:
        dict with measurements in cm
    """
    from clad_body.load.mhr import MhrBody
    joints = None
    if isinstance(mesh_or_body, MhrBody):
        mesh = mesh_or_body.mesh
        joints = mesh_or_body.joints
    else:
        mesh = mesh_or_body

    height = mesh.vertices[:, 2].max()
    zs, circs = body_signature(mesh)

    measurements = {"height_cm": height * 100}

    # Bust: from body signature (default max_x=0.40 filters arm-merged contours)
    z, circ_cm, pct = find_measurement(zs, circs, height, "bust")
    measurements["bust_cm"] = circ_cm
    measurements["_bust_z"] = z
    measurements["_bust_pct"] = pct

    # Hip: dedicated sweep with max_x=0.60.  At hip level (~50% height) arms
    # are far above (elbows ~60%, shoulders ~75%), so wide contours are purely
    # torso.  The body_signature default max_x=0.40 clips plus-size hips.
    from clad_body.measure._slicer import MeshSlicer
    hip_region = REGIONS["hip"]
    hip_zs = np.arange(height * hip_region["low_pct"],
                       height * hip_region["high_pct"], 0.002)
    hip_slicer = MeshSlicer(mesh)
    hip_circs = np.array([
        hip_slicer.circumference_at_z(z, max_x_extent=0.60) for z in hip_zs
    ])
    hip_valid = hip_circs > 0.30
    if hip_valid.any():
        idx = np.argmax(np.where(hip_valid, hip_circs, -1))
        measurements["hip_cm"] = hip_circs[idx] * 100
        measurements["_hip_z"] = hip_zs[idx]
        measurements["_hip_pct"] = hip_zs[idx] / height * 100
    else:
        measurements["hip_cm"] = 0
        measurements["_hip_z"] = 0
        measurements["_hip_pct"] = 0

    # Waist: circumference at ISO 8559-1 waist level (fixed % of height).
    waist_z = height * WAIST_HEIGHT_PCT
    waist_circ = torso_circumference_at_z(mesh, waist_z)
    measurements["waist_cm"] = waist_circ * 100
    measurements["_waist_z"] = waist_z
    measurements["_waist_pct"] = WAIST_HEIGHT_PCT * 100

    # Stomach: circumference at max anterior protrusion between hip and waist.
    # MHR has no torso-only mesh, so we find the Z of max belly protrusion
    # (min Y vertex scan), then measure with torso_circumference_at_z which
    # filters by max_x_extent to exclude arms.
    hip_z = measurements.get("_hip_z", 0)
    if hip_z > 0 and waist_z > hip_z:
        stomach_zs = np.arange(hip_z, waist_z, 0.002)
        mesh_verts = np.array(mesh.vertices)
        best_z, best_front_y = None, float("inf")
        for sz in stomach_zs:
            band = mesh_verts[np.abs(mesh_verts[:, 2] - sz) < 0.002]
            if len(band) < 3:
                continue
            front_y = band[:, 1].min()  # most anterior
            if front_y < best_front_y:
                best_front_y = front_y
                best_z = sz
        if best_z is not None:
            circ = torso_circumference_at_z(mesh, best_z)
            if circ > 0.30:
                measurements["stomach_cm"] = circ * 100
                measurements["_stomach_z"] = best_z
                measurements["_stomach_pct"] = best_z / height * 100

    # Thigh: max circumference from 2 separate leg contours
    thigh_cm, thigh_z, thigh_pct = measure_thigh(mesh, height)
    measurements["thigh_cm"] = thigh_cm
    measurements["_thigh_z"] = thigh_z
    measurements["_thigh_pct"] = thigh_pct

    # Knee: circumference at mid-patella level
    knee_cm, knee_z, knee_pct = measure_knee(mesh, height)
    measurements["knee_cm"] = knee_cm
    measurements["_knee_z"] = knee_z
    measurements["_knee_pct"] = knee_pct

    # Calf: max circumference from 2 separate lower leg contours
    calf_cm, calf_z, calf_pct = measure_calf(mesh, height)
    measurements["calf_cm"] = calf_cm
    measurements["_calf_z"] = calf_z
    measurements["_calf_pct"] = calf_pct

    # Upper arm: max circumference from arm contours separate from torso
    upperarm_cm, upperarm_z, upperarm_pct = measure_upperarm(mesh, height)
    measurements["upperarm_cm"] = upperarm_cm
    measurements["_upperarm_z"] = upperarm_z
    measurements["_upperarm_pct"] = upperarm_pct

    # Wrist: circumference perpendicular to forearm axis
    wrist_cm, wrist_z, wrist_pct = measure_wrist(mesh, height, joints=joints)
    measurements["wrist_cm"] = wrist_cm
    measurements["_wrist_z"] = wrist_z
    measurements["_wrist_pct"] = wrist_pct

    # Inseam via mesh geometry (crotch detection)
    inseam_cm, inseam_z, inseam_pct = measure_inseam(mesh, height)
    measurements["inseam_cm"] = inseam_cm
    measurements["_inseam_z"] = inseam_z
    measurements["_inseam_pct"] = inseam_pct

    # Crotch length (total rise) via surface tracing
    crotch_len, front_rise, back_rise, crotch_f_pts, crotch_b_pts = \
        measure_crotch_length(
            mesh, height,
            measurements.get("_waist_z", 0), inseam_z)
    measurements["crotch_length_cm"] = crotch_len
    measurements["front_rise_cm"] = front_rise
    measurements["back_rise_cm"] = back_rise
    if crotch_f_pts is not None:
        measurements["_crotch_front_pts"] = crotch_f_pts
    if crotch_b_pts is not None:
        measurements["_crotch_back_pts"] = crotch_b_pts

    # Joint-based linear measurements (shoulder width, arm length)
    if joints:
        c7 = joints.get("c7")
        if c7 is not None:
            measurements["_c7_surface_pt"] = c7_surface_point(
                np.array(mesh.vertices), c7)
        sw_cm, sw_arc = measure_shoulder_width(
            joints, mesh=mesh, acromion_fn=find_acromion)
        measurements["shoulder_width_cm"] = sw_cm
        if sw_arc is not None:
            measurements["_shoulder_arc_pts"] = sw_arc
        measurements["sleeve_length_cm"] = measure_sleeve_length(
            joints, mesh=mesh, acromion_fn=find_acromion)

        # Shirt length: side neck → crotch along front body contour
        shirt_cm, shirt_pts = measure_shirt_length(
            joints, mesh, measurements.get("_inseam_z", 0),
            measurements=measurements)
        measurements["shirt_length_cm"] = shirt_cm
        if shirt_pts is not None:
            measurements["_shirt_length_pts"] = shirt_pts

    if joints:
        measurements["_debug_joints"] = {k: np.array(v) for k, v in joints.items()}
    measurements["_linear_polylines"] = extract_linear_measurement_polylines(
        mesh, measurements, joints or {})

    # Public mesh + contours for callers (e.g. OBJ export with measurement lines)
    measurements["mesh"] = mesh
    measurements["contours"] = extract_measurement_contours(mesh, measurements)

    if render_path:
        render_4view(mesh, measurements, render_path,
                     title=title, model_label="MHR")

    return measurements


def _measure_mhr(body, *, groups, render_path=None, title=""):
    """Internal: measure an MhrBody with selective computation groups.

    Called by ``clad_body.measure.measure()``. Do not call directly.
    """
    from clad_body.measure import (
        GROUP_A, GROUP_B, GROUP_C, GROUP_D, GROUP_E, GROUP_F, GROUP_G, GROUP_H,
    )
    mesh = body.mesh
    joints = body.joints
    height = mesh.vertices[:, 2].max()

    measurements = {"height_cm": height * 100}

    # ── Group A: Core torso ──────────────────────────────────────────────
    if GROUP_A in groups:
        zs, circs = body_signature(mesh)

        # Bust
        z, circ_cm, pct = find_measurement(zs, circs, height, "bust")
        measurements["bust_cm"] = circ_cm
        measurements["_bust_z"] = z
        measurements["_bust_pct"] = pct

        # Hip (dedicated sweep with wider x_extent for plus-size)
        hip_region = REGIONS["hip"]
        hip_zs = np.arange(height * hip_region["low_pct"],
                           height * hip_region["high_pct"], 0.002)
        hip_slicer = MeshSlicer(mesh)
        hip_circs = np.array([
            hip_slicer.circumference_at_z(z, max_x_extent=0.60) for z in hip_zs
        ])
        hip_valid = hip_circs > 0.30
        if hip_valid.any():
            idx = np.argmax(np.where(hip_valid, hip_circs, -1))
            measurements["hip_cm"] = hip_circs[idx] * 100
            measurements["_hip_z"] = hip_zs[idx]
            measurements["_hip_pct"] = hip_zs[idx] / height * 100
        else:
            measurements["hip_cm"] = 0
            measurements["_hip_z"] = 0
            measurements["_hip_pct"] = 0

        # Waist
        waist_z = height * WAIST_HEIGHT_PCT
        waist_circ = torso_circumference_at_z(mesh, waist_z)
        measurements["waist_cm"] = waist_circ * 100
        measurements["_waist_z"] = waist_z
        measurements["_waist_pct"] = WAIST_HEIGHT_PCT * 100

        # Stomach
        hip_z = measurements.get("_hip_z", 0)
        if hip_z > 0 and waist_z > hip_z:
            stomach_zs = np.arange(hip_z, waist_z, 0.002)
            mesh_verts = np.array(mesh.vertices)
            best_z, best_front_y = None, float("inf")
            for sz in stomach_zs:
                band = mesh_verts[np.abs(mesh_verts[:, 2] - sz) < 0.002]
                if len(band) < 3:
                    continue
                front_y = band[:, 1].min()
                if front_y < best_front_y:
                    best_front_y = front_y
                    best_z = sz
            if best_z is not None:
                circ = torso_circumference_at_z(mesh, best_z)
                if circ > 0.30:
                    measurements["stomach_cm"] = circ * 100
                    measurements["_stomach_z"] = best_z
                    measurements["_stomach_pct"] = best_z / height * 100

    # ── Group B: Limb sweeps ─────────────────────────────────────────────
    if GROUP_B in groups:
        thigh_cm, thigh_z, thigh_pct = measure_thigh(mesh, height)
        measurements["thigh_cm"] = thigh_cm
        measurements["_thigh_z"] = thigh_z
        measurements["_thigh_pct"] = thigh_pct

        knee_cm, knee_z, knee_pct = measure_knee(mesh, height)
        measurements["knee_cm"] = knee_cm
        measurements["_knee_z"] = knee_z
        measurements["_knee_pct"] = knee_pct

        calf_cm, calf_z, calf_pct = measure_calf(mesh, height)
        measurements["calf_cm"] = calf_cm
        measurements["_calf_z"] = calf_z
        measurements["_calf_pct"] = calf_pct

        upperarm_cm, upperarm_z, upperarm_pct = measure_upperarm(mesh, height)
        measurements["upperarm_cm"] = upperarm_cm
        measurements["_upperarm_z"] = upperarm_z
        measurements["_upperarm_pct"] = upperarm_pct

    # ── Group D: Perpendicular (wrist — MHR has no neck perpendicular) ──
    if GROUP_D in groups:
        wrist_cm, wrist_z, wrist_pct = measure_wrist(mesh, height, joints=joints)
        measurements["wrist_cm"] = wrist_cm
        measurements["_wrist_z"] = wrist_z
        measurements["_wrist_pct"] = wrist_pct

    # ── Group E: Mesh geometry (inseam, crotch) ──────────────────────────
    if GROUP_E in groups:
        inseam_cm, inseam_z, inseam_pct = measure_inseam(mesh, height)
        measurements["inseam_cm"] = inseam_cm
        measurements["_inseam_z"] = inseam_z
        measurements["_inseam_pct"] = inseam_pct

        crotch_len, front_rise, back_rise, crotch_f_pts, crotch_b_pts = \
            measure_crotch_length(
                mesh, height,
                measurements.get("_waist_z", 0), inseam_z)
        measurements["crotch_length_cm"] = crotch_len
        measurements["front_rise_cm"] = front_rise
        measurements["back_rise_cm"] = back_rise
        if crotch_f_pts is not None:
            measurements["_crotch_front_pts"] = crotch_f_pts
        if crotch_b_pts is not None:
            measurements["_crotch_back_pts"] = crotch_b_pts

    # ── Group C: Joint linear (shoulder, sleeve) ─────────────────────────
    if GROUP_C in groups and joints:
        c7 = joints.get("c7")
        if c7 is not None:
            measurements["_c7_surface_pt"] = c7_surface_point(
                np.array(mesh.vertices), c7)
        sw_cm, sw_arc = measure_shoulder_width(
            joints, mesh=mesh, acromion_fn=find_acromion)
        measurements["shoulder_width_cm"] = sw_cm
        if sw_arc is not None:
            measurements["_shoulder_arc_pts"] = sw_arc
        measurements["sleeve_length_cm"] = measure_sleeve_length(
            joints, mesh=mesh, acromion_fn=find_acromion)

    # ── Group F: Surface trace (shirt length) ────────────────────────────
    if GROUP_F in groups and joints:
        shirt_cm, shirt_pts = measure_shirt_length(
            joints, mesh, measurements.get("_inseam_z", 0),
            measurements=measurements)
        measurements["shirt_length_cm"] = shirt_cm
        if shirt_pts is not None:
            measurements["_shirt_length_pts"] = shirt_pts

    # ── Group H: Back neck to waist (ISO 5.4.5) ──────────────────────────
    if GROUP_H in groups and joints:
        bnw_cm, bnw_pts = measure_back_neck_to_waist(
            joints, mesh, measurements.get("_waist_z", 0),
            c7_surface=measurements.get("_c7_surface_pt"))
        measurements["back_neck_to_waist_cm"] = bnw_cm
        if bnw_pts is not None:
            measurements["_back_neck_to_waist_pts"] = bnw_pts

    # ── Visualization ────────────────────────────────────────────────────
    if joints:
        measurements["_debug_joints"] = {k: np.array(v) for k, v in joints.items()}
    has_linear = GROUP_C in groups or GROUP_E in groups or GROUP_H in groups
    if has_linear:
        measurements["_linear_polylines"] = extract_linear_measurement_polylines(
            mesh, measurements, joints or {})
    measurements["mesh"] = mesh
    measurements["contours"] = extract_measurement_contours(mesh, measurements)

    if render_path:
        render_4view(mesh, measurements, render_path,
                     title=title, model_label="MHR")

    return measurements


def render_body_signature(mesh, measurements, output_path, title=""):
    """Render body signature plot with measurement annotations."""
    import matplotlib.pyplot as plt
    zs, circs = body_signature(mesh)
    height = mesh.vertices[:, 2].max()
    pcts = zs / height * 100
    circs_cm = circs * 100

    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    ax.plot(circs_cm, pcts, 'b-', linewidth=1.5)

    colors = {"hip": "gold", "waist": "blue", "bust": "green",
              "thigh": "purple", "upperarm": "orange"}
    for name, color in colors.items():
        z = measurements.get(f"_{name}_z", 0)
        circ = measurements.get(f"{name}_cm", 0)
        pct = measurements.get(f"_{name}_pct", 0)
        if circ > 0:
            ax.axhline(pct, color=color, alpha=0.4, linestyle='--')
            ax.plot(circ, pct, 'o', color=color, markersize=14, zorder=10,
                    label=f'{name.title()}: {circ:.1f}cm ({pct:.0f}%)')

    ax.set_xlabel('Circumference (cm)', fontsize=13)
    ax.set_ylabel('Height (%)', fontsize=13)
    ax.set_title(f'MHR Body Signature{" — " + title if title else ""}\n'
                 f'Height: {measurements["height_cm"]:.1f}cm | '
                 f'Torso-only (X extent < {MAX_TORSO_X_EXTENT*100:.0f}cm)',
                 fontsize=13)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure MHR body mesh via horizontal plane sweep"
    )
    parser.add_argument(
        "input",
        help="Path to SAM3D params JSON or results directory"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate 4-view render with measurement contours"
    )
    parser.add_argument(
        "--signature", action="store_true",
        help="Generate body signature plot (circumference vs height)"
    )
    parser.add_argument(
        "--target", "-t", default=None,
        help="Path to target_measurements.json for comparison"
    )
    args = parser.parse_args()

    # Resolve input
    input_path = args.input
    if os.path.isdir(input_path):
        body_name = os.path.basename(os.path.normpath(input_path))
    else:
        body_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"Loading MHR from params...")
    body = load_mhr_from_params(input_path)
    mesh = body.mesh
    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    # Resolve render path
    render_path = None
    if args.plot:
        out_dir = os.path.join(_SCRIPT_DIR, "results", body_name)
        os.makedirs(out_dir, exist_ok=True)
        render_path = os.path.join(out_dir, "4view_mhr_base_measurements.png")

    print(f"\nMeasuring (plane sweep, 2mm steps)...")
    measurements = measure_mhr(body, render_path=render_path, title=body_name)

    print(f"\n=== MHR Body Measurements ===")
    print(f"Height:    {measurements['height_cm']:>6.1f} cm")
    print(f"Bust:      {measurements['bust_cm']:>6.1f} cm  (at {measurements['_bust_pct']:.0f}% height)")
    print(f"Waist:     {measurements['waist_cm']:>6.1f} cm  (at {measurements['_waist_pct']:.0f}% height)")
    print(f"Hip:       {measurements['hip_cm']:>6.1f} cm  (at {measurements['_hip_pct']:.0f}% height)")
    if measurements.get("stomach_cm", 0) > 0:
        print(f"Stomach:   {measurements['stomach_cm']:>6.1f} cm  (at {measurements['_stomach_pct']:.0f}% height)")
    if measurements.get("thigh_cm", 0) > 0:
        print(f"Thigh:     {measurements['thigh_cm']:>6.1f} cm  (at {measurements['_thigh_pct']:.0f}% height)")
    if measurements.get("upperarm_cm", 0) > 0:
        print(f"Upper arm: {measurements['upperarm_cm']:>6.1f} cm  (at {measurements['_upperarm_pct']:.0f}% height)")
    if measurements.get("shoulder_width_cm", 0) > 0:
        print(f"Shoulder W:{measurements['shoulder_width_cm']:>6.1f} cm")
    if measurements.get("sleeve_length_cm", 0) > 0:
        print(f"Sleeve len:{measurements['sleeve_length_cm']:>6.1f} cm")
    if measurements.get("inseam_cm", 0) > 0:
        print(f"Inseam:    {measurements['inseam_cm']:>6.1f} cm  (crotch at {measurements['_inseam_pct']:.0f}% height)")
    if measurements.get("crotch_length_cm", 0) > 0:
        print(f"Crotch len:{measurements['crotch_length_cm']:>6.1f} cm  (front {measurements['front_rise_cm']:.1f} + back {measurements['back_rise_cm']:.1f})")

    # Target comparison
    if args.target:
        target = load_target_measurements(args.target)
        print_comparison(measurements, target)

    # Body signature chart
    if args.signature:
        out_dir = os.path.join(_SCRIPT_DIR, "results", body_name)
        os.makedirs(out_dir, exist_ok=True)
        sig_path = os.path.join(out_dir, "mhr_signature.png")
        render_body_signature(mesh, measurements, sig_path, title=body_name)

    return measurements


if __name__ == "__main__":
    main()
