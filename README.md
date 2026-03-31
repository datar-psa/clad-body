# clad-body

Body loaders and ISO 8559-1 measurements for [Anny](https://github.com/naver/anny) and [MHR](https://github.com/facebookresearch/MHR) parametric body models.

Neither Anny nor MHR ship with a measurement library. You get a mesh with 14–18K vertices and no standard way to extract waist circumference from it. This package fills that gap.

<p align="center">
  <img src="assets/body_rotation.gif" alt="Anny body rotating with ISO 8559-1 circumference measurement contours" width="400">
  <br>
  <img src="assets/4view_male_average.png" alt="Anny body — male average, 4-view with ISO 8559-1 circumference measurements" width="700">
</p>

## What it does

- **Load** Anny bodies from phenotype params, MHR bodies from SAM 3D Body params
- **Measure** circumferences (bust, waist, hips, thigh, knee, calf, upper arm, wrist, neck) via ISO 8559-1 plane sweep with convex hull tape-measure simulation
- **Linear measurements** — shoulder width, sleeve length, inseam
- **Body composition** — volume, mass, BMI, body fat estimation
- **Render** 4-view PNGs with measurement contour overlays

All bodies are normalised to the same coordinate convention: Z-up, metres, XY-centred, feet at Z=0, +Y=front.

## Install

```bash
pip install clad-body

# With Anny body loader (requires torch)
pip install 'clad-body[anny]'

# With MHR body loader (requires pymomentum)
pip install 'clad-body[mhr]'

# With 4-view rendering
pip install 'clad-body[render]'
```

## Quick start

### Measure an Anny body

```python
from clad_body.measure.anny import measure_body

params = {
    "gender": 0.5, "age": 0.5, "muscle": 0.5,
    "weight": 0.5, "height": 0.5, "proportions": 0.5,
    "cupsize": 0.5, "firmness": 0.5,
    "african": 0.33, "asian": 0.33, "caucasian": 0.33,
}

measurements = measure_body(params, render_path="body_4view.png")
print(f"Bust: {measurements['bust_cm']:.1f} cm")
print(f"Waist: {measurements['waist_cm']:.1f} cm")
print(f"Hips: {measurements['hips_cm']:.1f} cm")
```

### Measure an MHR body

```python
from clad_body.load.mhr import load_mhr_from_params
from clad_body.measure.mhr import measure_mhr

body = load_mhr_from_params("path/to/sam3d_params.json")
measurements = measure_mhr(body, render_path="mhr_4view.png")
```

### Use MeshSlicer directly (no body model needed)

```python
import trimesh
from clad_body.measure.common import MeshSlicer

mesh = trimesh.load("body.obj")
slicer = MeshSlicer(mesh)

# Circumference at a specific height (metres)
circumference = slicer.circumference_at_z(0.95)  # waist level
print(f"Circumference: {circumference * 100:.1f} cm")
```

## Measurements

All circumference measurements follow [ISO 8559-1:2017](https://www.iso.org/standard/61686.html) conventions. Circumferences use convex hull projection to simulate a tape measure wrapping around the body.

| Measurement | ISO 8559-1 | Method | |
|---|---|---|---|
| Bust | §5.3.4 | Maximum horizontal circumference in bust region (68–76% height) | <img src="assets/contours/bust.png" width="80"> |
| Underbust | §5.3.6 | Circumference directly below the bust prominence | <img src="assets/contours/underbust.png" width="80"> |
| Waist | §5.3.10 | Circumference at anatomical midpoint (~61% height) | <img src="assets/contours/waist.png" width="80"> |
| Stomach | — | Maximum circumference between waist and hips | <img src="assets/contours/stomach.png" width="80"> |
| Hips | §5.3.13 | Maximum horizontal circumference in hip region (46–54% height) | <img src="assets/contours/hip.png" width="80"> |
| Thigh | §5.3.20 | Maximum circumference from separate leg contours | <img src="assets/contours/thigh.png" width="80"> |
| Knee | §5.3.22 | Circumference at mid-patella level | <img src="assets/contours/knee.png" width="80"> |
| Calf | §5.3.24 | Maximum circumference of the lower leg | <img src="assets/contours/calf.png" width="80"> |
| Upper arm | §5.3.16 | Maximum circumference from separate arm contours | <img src="assets/contours/upperarm.png" width="80"> |
| Wrist | §5.3.19 | Circumference over the wrist bones (perpendicular to forearm) | <img src="assets/contours/wrist.png" width="80"> |
| Neck | §5.3.2 | Minimum circumference in neck region | <img src="assets/contours/neck.png" width="80"> |
| Shoulder width | §5.4.2 | Acromion-to-acromion arc over C7 vertebra | <img src="assets/contours/shoulder_width.png" width="80"> |
| Sleeve length | §5.7.8 | Acromion to wrist via elbow | <img src="assets/contours/sleeve_length.png" width="80"> |
| Inseam | §5.1.15 | Crotch point to floor | <img src="assets/contours/inseam.png" width="80"> |

## Optional extras

| Extra | What it enables |
|---|---|
| `[anny]` | `clad_body.load.anny` — generate Anny bodies from phenotype params |
| `[mhr]` | `clad_body.load.mhr` — generate MHR bodies from SAM 3D Body params |
| `[render]` | `render_4view()` — matplotlib/pyrender 4-view body renders |

Without extras, `MeshSlicer` and the core measurement utilities work with any trimesh mesh — only numpy, scipy, and trimesh are required.

## Demo

Try the full pipeline (body reconstruction + measurements) at [clad.you/size-aware/size-me](https://clad.you/size-aware/size-me).

## Background

This library was built for [Clad](https://clad.you)'s size-aware virtual try-on pipeline. Read the full story: [A 3D Body Scan for Nine Cents — Without SMPL](https://clad.you/blog/posts/body-pipeline/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
