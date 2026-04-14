#!/usr/bin/env python3
"""Render a single measurement contour on a body → PNG for README table.

Usage:
    python tools/render_contour.py --measurement bust
    python tools/render_contour.py --all  # render all measurements
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

TESTDATA = os.path.join(
    os.path.dirname(__file__), "..", "clad_body", "measure", "testdata", "anny",
)

# Which view angle best shows each measurement
MEASUREMENT_VIEWS = {
    "bust":            ("Front", 10, -90),
    "underbust":       ("Front", 10, -90),
    "waist":           ("Front", 10, -90),
    "stomach":         ("Side",  10,   0),
    "hip":             ("Front", 10, -90),
    "thigh":           ("3/4",   15, -55),
    "knee":            ("Front", 10, -90),
    "calf":            ("Front", 10, -90),
    "upperarm":        ("Front", 10, -90),
    "wrist":           ("Front", 10, -90),
    "neck":            ("Front", 10, -90),
    "shoulder_width":  ("Back",  10,  90),
    "sleeve_length":   ("Back",  10,  90),
    "inseam":          ("Front", 10, -90),
    "crotch":          ("Side",  10,   0),
    "shirt_length":    ("Side",  10,   0),
    "back_neck_to_waist": ("Back", 10,  90),
}

CONTOUR_COLORS = {
    "bust": "green", "underbust": "cyan", "waist": "blue",
    "stomach": "darkorange", "hip": "red",
    "thigh": "purple", "knee": "deeppink", "calf": "hotpink",
    "upperarm": "orange", "wrist": "gold", "neck": "teal",
}

LINEAR_COLORS = {
    "shoulder_width": "cyan",
    "sleeve_length": "magenta",
    "inseam": "yellow",
    "shirt_length": "dodgerblue",
    "back_neck_to_waist": "orange",
}


def render_contour(subject: str, measurement: str, output_path: str):
    """Render a single measurement contour on the body, save as PNG."""
    import pyrender
    from PIL import Image, ImageDraw
    import matplotlib.colors as mcolors

    from clad_body.load.anny import load_anny_from_params
    from clad_body.measure import measure
    from clad_body.measure._render import (
        _camera_pose,
        _project_3d_to_2d,
        extract_measurement_contours,
    )

    params_path = os.path.join(TESTDATA, subject, "anny_params.json")
    with open(params_path) as f:
        params = json.load(f)

    measurements = measure(load_anny_from_params(params))
    mesh = measurements["_mesh_tri"]
    torso_mesh = measurements["_torso_mesh"]

    contours = extract_measurement_contours(mesh, measurements, torso_mesh=torso_mesh)
    linear_polylines = measurements.get("_linear_polylines", {})

    verts = np.array(mesh.vertices)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    height = verts[:, 2].max() - verts[:, 2].min()

    view_w, view_h = 250, 450
    ymag = height / 2 * 1.15
    xmag = ymag * (view_w / view_h)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.65, 0.55, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    renderer = pyrender.OffscreenRenderer(view_w, view_h)

    view = MEASUREMENT_VIEWS.get(measurement, ("Front", 10, -90))
    _, elev, azim = view

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
    renderer.delete()

    img = Image.fromarray(color)
    draw = ImageDraw.Draw(img)

    # Draw the specific contour
    if measurement == "crotch":
        # Crotch draws both front and back midline polylines
        rgb = tuple(int(c * 255) for c in mcolors.to_rgb("lime"))
        for key in ("crotch_front", "crotch_back"):
            pts = linear_polylines.get(key)
            if pts is not None:
                px, py = _project_3d_to_2d(pts, cam_pose, xmag, ymag, view_w, view_h)
                coords = list(zip(px.astype(int), py.astype(int)))
                if len(coords) > 1:
                    draw.line(coords, fill=rgb, width=3)
    elif measurement in contours:
        hex_color = CONTOUR_COLORS[measurement]
        rgb = tuple(int(c * 255) for c in mcolors.to_rgb(hex_color))
        for pts in contours[measurement]:
            loop = np.vstack([pts, pts[:1]])
            px, py = _project_3d_to_2d(loop, cam_pose, xmag, ymag, view_w, view_h)
            coords = list(zip(px.astype(int), py.astype(int)))
            if len(coords) > 1:
                draw.line(coords, fill=rgb, width=3)
    elif measurement in linear_polylines:
        hex_color = LINEAR_COLORS.get(measurement, "cyan")
        rgb = tuple(int(c * 255) for c in mcolors.to_rgb(hex_color))
        pts = linear_polylines[measurement]
        px, py = _project_3d_to_2d(pts, cam_pose, xmag, ymag, view_w, view_h)
        coords = list(zip(px.astype(int), py.astype(int)))
        if len(coords) > 1:
            draw.line(coords, fill=rgb, width=3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path, optimize=True)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  {measurement:>16s} → {output_path} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Render measurement contour PNG")
    parser.add_argument("measurement", nargs="?", help="Measurement name (e.g. bust, waist)")
    parser.add_argument("--all", action="store_true", help="Render all measurements")
    parser.add_argument("--subject", default="female_curvy",
                        help="Test subject (default: female_curvy)")
    parser.add_argument("--output-dir", default="assets/contours",
                        help="Output directory (default: assets/contours)")
    args = parser.parse_args()

    output_base = os.path.join(
        os.path.dirname(__file__), "..", args.output_dir,
    )

    if args.all:
        measurements = list(MEASUREMENT_VIEWS.keys())
    elif args.measurement:
        measurements = [args.measurement]
    else:
        parser.error("Specify a measurement name or --all")

    print(f"Rendering {len(measurements)} contour(s) for {args.subject}...")
    for m in measurements:
        output_path = os.path.join(output_base, f"{m}.png")
        render_contour(args.subject, m, output_path)


if __name__ == "__main__":
    main()
