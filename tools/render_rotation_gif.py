#!/usr/bin/env python3
"""Render a rotating body with measurement contours → GIF for README.

Usage:
    python tools/render_rotation_gif.py [--subject female_curvy] [--output assets/body_rotation.gif]
"""
import argparse
import json
import os
import sys

import numpy as np

# Ensure EGL backend before any pyrender/OpenGL import
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "clad_body", "measure", "testdata", "anny")


def render_frames(subject: str, n_frames: int = 36) -> list[np.ndarray]:
    """Load body, measure, render rotation frames with contour overlays."""
    import pyrender
    import trimesh
    from PIL import Image

    from clad_body.load.anny import load_anny_from_params
    from clad_body.measure import measure
    from clad_body.measure.common import (
        CONTOUR_COLORS,
        _camera_pose,
        _project_3d_to_2d,
        extract_measurement_contours,
    )

    # Load params, measure, and extract the trimesh from measurements dict
    params_path = os.path.join(TESTDATA, subject, "anny_params.json")
    with open(params_path) as f:
        params = json.load(f)

    measurements = measure(load_anny_from_params(params))
    mesh = measurements["_mesh_tri"]
    torso_mesh = measurements["_torso_mesh"]

    contours = extract_measurement_contours(mesh, measurements, torso_mesh=torso_mesh)

    # Pyrender setup
    verts = np.array(mesh.vertices)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    height = verts[:, 2].max() - verts[:, 2].min()

    view_w, view_h = 400, 720
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

    frames = []
    # Full rotation: front → right → back → left → front
    azimuths = np.linspace(-90, 270, n_frames, endpoint=False)

    for azim in azimuths:
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=[0.3, 0.3, 0.3],
        )
        scene.add(pr_mesh)
        cam_pose = _camera_pose(10, azim, 3.0, center)
        scene.add(camera, pose=cam_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
        scene.add(light, pose=cam_pose)
        color, _ = renderer.render(scene)

        # Overlay contours using PIL (no matplotlib needed for frames)
        img = Image.fromarray(color)
        # Draw contours as simple lines
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        for name, pts_list in contours.items():
            hex_color = CONTOUR_COLORS[name]
            # Convert matplotlib color string to RGB tuple
            import matplotlib.colors as mcolors
            rgb = tuple(int(c * 255) for c in mcolors.to_rgb(hex_color))

            for pts in pts_list:
                loop = np.vstack([pts, pts[:1]])
                px, py = _project_3d_to_2d(loop, cam_pose, xmag, ymag, view_w, view_h)
                coords = list(zip(px.astype(int), py.astype(int)))
                if len(coords) > 1:
                    draw.line(coords, fill=rgb, width=2)

        # Also draw linear measurement polylines
        _linear_visible = {
            "shoulder_width": {-90, 90, 0, -60},  # front, back, side, 3/4
            "sleeve_length": {90, 0},
            "inseam": {0, -60},
        }
        _linear_colors = {
            "shoulder_width": "cyan",
            "sleeve_length": "magenta",
            "inseam": "yellow",
        }
        # Skip linear lines for rotation — they look messy from most angles

        frames.append(np.array(img))

    renderer.delete()
    return frames


def frames_to_gif(frames: list[np.ndarray], output_path: str, fps: int = 12):
    """Convert frames to an optimized GIF using PIL."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    # Quantize to common palette for smaller file
    duration_ms = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved {output_path} ({len(frames)} frames, {size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Render rotating body GIF")
    parser.add_argument("--subject", default="female_curvy",
                        help="Test subject name (default: female_curvy)")
    parser.add_argument("--output", default="assets/body_rotation.gif",
                        help="Output GIF path (default: assets/body_rotation.gif)")
    parser.add_argument("--frames", type=int, default=36,
                        help="Number of frames (default: 36)")
    parser.add_argument("--fps", type=int, default=12,
                        help="GIF framerate (default: 12)")
    args = parser.parse_args()

    print(f"Rendering {args.frames} frames of {args.subject}...")
    frames = render_frames(args.subject, args.frames)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    frames_to_gif(frames, args.output, args.fps)


if __name__ == "__main__":
    main()
