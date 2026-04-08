# clad-body

Body loaders and ISO 8559-1 measurements for [Anny](https://github.com/naver/anny) and [MHR](https://github.com/facebookresearch/MHR) parametric body models.

Neither Anny nor MHR ship with a measurement library. You get a mesh with 14-18K vertices and no standard way to extract waist circumference from it. This package fills that gap.

<p align="center">
  <img src="https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/body_rotation.gif" alt="Anny body rotating with ISO 8559-1 circumference measurement contours" width="400">
  <br>
  <img src="https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/4view_male_average.png" alt="Anny body — male average, 4-view with ISO 8559-1 circumference measurements" width="700">
</p>

## What it does

- **Load** Anny bodies from phenotype params, MHR bodies from SAM 3D Body params
- **Measure** circumferences (bust, waist, hips, thigh, knee, calf, upper arm, wrist, neck) via ISO 8559-1 plane sweep with convex hull tape-measure simulation
- **Linear measurements** — shoulder width, sleeve length, inseam, crotch length, shirt length
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

```python
from clad_body.load import load_anny_from_params
from clad_body.measure import measure

body = load_anny_from_params(params)

m = measure(body)                                  # all measurements
m = measure(body, preset="core")                   # 4: height, bust, waist, hip
m = measure(body, preset="standard")               # 9: + thigh, upperarm, shoulder, sleeve, inseam
m = measure(body, preset="tops")                   # garment-relevant subset
m = measure(body, only=["bust_cm", "hip_cm"])       # specific keys
m = measure(body, tags={"type": "circumference", "region": "leg"})  # tag filter
m = measure(body, render_path="body.png")          # with 4-view render
```

MHR works the same way:

```python
from clad_body.load import load_mhr_from_params
body = load_mhr_from_params("path/to/sam3d_params.json")
m = measure(body)
```

## Public API

| Import | What |
|---|---|
| `clad_body.load.load_anny_from_params` | Load Anny body from phenotype params |
| `clad_body.load.load_mhr_from_params` | Load MHR body from SAM 3D Body params |
| `clad_body.load.AnnyBody`, `MhrBody` | Body dataclasses |
| `clad_body.measure.measure` | Measure a body (one entry point) |
| `clad_body.measure.REGISTRY` | All measurement definitions (`dict[str, MeasurementDef]`) |
| `clad_body.measure.list_measurements` | Query measurements by tags |
| `clad_body.measure.MeasurementDef` | Measurement definition type |

### Selection

`measure()` accepts `preset`, `only`, `tags`, `exclude`. Precedence: `only` > `preset` > `tags` > default (`"all"`). `exclude` is applied last. Only runs computation groups needed for the requested keys.

### Introspection

```python
from clad_body.measure import REGISTRY, list_measurements

REGISTRY["bust_cm"].description   # self-measurement instructions
REGISTRY["bust_cm"].iso_ref       # "5.3.4"
REGISTRY["bust_cm"].type          # "circumference"

list_measurements(type="circumference", region="leg")   # [thigh, knee, calf]
```

## Measurement registry

Every measurement is tagged across 5 dimensions. Each carries a human-readable `description` for self-measurement instructions and i18n key mapping.

### Tags

| Dimension | Values |
|---|---|
| **type** | `circumference`, `length`, `scalar` |
| **standard** | `iso` (ISO 8559-1), `tailor` (industry standard), `derived` (computed) |
| **region** | `neck`, `torso`, `abdomen`, `arm`, `leg`, `full_body` |
| **tier** | `core` > `standard` > `enhanced` > `fitted` (cumulative) |
| **garments** | `tops`, `bottoms`, `dresses`, `outerwear`, `underwear` |

### Tier presets

| Preset | Count | Adds |
|---|---|---|
| `core` | 4 | height, bust, waist, hip |
| `standard` | 9 | thigh, upperarm, shoulder_width, sleeve_length, inseam |
| `enhanced` | 18 | neck, underbust, stomach, mass, volume, bmi, body_fat, belly_depth, back_neck_to_waist |
| `fitted`/`all` | 25 | knee, calf, wrist, crotch_length, front_rise, back_rise, shirt_length |

### Full measurement table

Garment codes: **T**ops, **B**ottoms, **D**resses, **O**uterwear, **U**nderwear.

| Contour | Key | Description | ISO | Type | Std | Region | Tier | Grp | Gar |
|---|---|---|---|---|---|---|---|---|---|
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/bust.png) | `height_cm` | Vertical distance from floor to top of head. Stand erect, feet together. | 5.1.1 | scalar | iso | full_body | core | A | all |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/bust.png) | `bust_cm` | Horizontal circumference at the fullest part of the chest/bust. Tape under armpits, across bust prominence, level and snug. | 5.3.4 | circ | iso | torso | core | A | T,D,O,U |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/waist.png) | `waist_cm` | Horizontal circumference at natural waist, midway between lowest rib and hip bone. Tape at navel height, parallel to floor. | 5.3.10 | circ | iso | torso | core | A | all |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/hip.png) | `hip_cm` | Horizontal circumference at greatest buttock prominence. Feet together, tape around widest part of hips. | 5.3.13 | circ | iso | abdomen | core | A | B,D,O,U |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/thigh.png) | `thigh_cm` | Horizontal circumference at fullest part of upper thigh, just below gluteal fold. Stand with legs slightly apart. | 5.3.20 | circ | iso | leg | std | B | B |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/upperarm.png) | `upperarm_cm` | Circumference at fullest part of upper arm, midway between shoulder and elbow. Arm relaxed, not flexed. | 5.3.16 | circ | iso | arm | std | B | T,O |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/shoulder_width.png) | `shoulder_width_cm` | Distance between left and right shoulder points (acromion), measured across back over C7 vertebra. | 5.4.2 | length | iso | torso | std | C | T,D,O |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/sleeve_length.png) | `sleeve_length_cm` | Distance from shoulder point along outside of slightly bent arm, over elbow, to wrist bone. | 5.7.8 | length | iso | arm | std | C | T,O |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/inseam.png) | `inseam_cm` | Distance from crotch point straight down to floor. Stand erect, feet slightly apart. | 5.1.15 | length | iso | leg | std | E | B |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/neck.png) | `neck_cm` | Circumference just below Adam's apple, perpendicular to neck axis. Comfortably snug. | 5.3.2 | circ | iso | neck | enh | D | T |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/underbust.png) | `underbust_cm` | Horizontal circumference directly below breast tissue, at inframammary crease. Bra band size. | 5.3.6 | circ | iso | torso | enh | A | T,D,U |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/stomach.png) | `stomach_cm` | Horizontal circumference at maximum anterior protrusion of abdomen, usually at/below navel. | -- | circ | tailor | abdomen | enh | A | T,B |
| | `mass_kg` | Total body mass in kilograms. | 5.6.1 | scalar | iso | full_body | enh | G | -- |
| | `volume_m3` | Total body volume in cubic metres, from mesh geometry. | -- | scalar | derived | full_body | enh | G | -- |
| | `bmi` | Body mass index: mass (kg) / height (m)^2. | -- | scalar | derived | full_body | enh | G | -- |
| | `body_fat_pct` | Estimated body fat % via Navy/Weltman equations from circumferences. | -- | scalar | derived | full_body | enh | G | -- |
| | `belly_depth_cm` | How much belly protrudes forward vs underbust/ribcage. Negative = belly prominence. | -- | scalar | derived | abdomen | enh | A | T,B |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/knee.png) | `knee_cm` | Horizontal circumference at centre of kneecap. Bend knee slightly (~45 degrees). | 5.3.22 | circ | iso | leg | fit | B | B |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/calf.png) | `calf_cm` | Maximum horizontal circumference of the calf. Stand with legs slightly apart. | 5.3.24 | circ | iso | leg | fit | B | B |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/wrist.png) | `wrist_cm` | Circumference at wrist, at prominent bone on little finger side (ulnar styloid). | 5.3.19 | circ | iso | arm | fit | D | T |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/crotch.png) | `crotch_length_cm` | Distance from front waist centre, through crotch, to back waist centre. Follow body surface. | 5.4.18 | length | iso | leg | fit | E | B |
| | `front_rise_cm` | Front waist to crotch point, along front body surface. Trouser front panel length. | -- | length | tailor | leg | fit | E | B |
| | `back_rise_cm` | Back waist to crotch point, along back body surface. Trouser back panel length. | -- | length | tailor | leg | fit | E | B |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/shirt_length.png) | `shirt_length_cm` | Side neck point down along front body contour to crotch level. Follow chest/stomach curve. | -- | length | tailor | torso | fit | F | T |
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/back_neck_to_waist.png) | `back_neck_to_waist_cm` | Cervicale (C7) down centre back along body contour to waist level. Tape follows spine curvature. | 5.4.5 | length | iso | torso | enh | H | T,D,O |

Tier codes: **core**, **std** (standard), **enh** (enhanced), **fit** (fitted). Anny-only: underbust, mass, volume, bmi, body_fat, belly_depth.

### Computation groups

| Group | Measurements | Cost | Deps |
|---|---|---|---|
| **A** Core torso | height, bust, waist, hip, stomach, underbust, belly_depth | Cheap | -- |
| **B** Limb sweeps | thigh, knee, calf, upperarm | Expensive | -- |
| **C** Joint linear | shoulder_width, sleeve_length | Expensive | -- |
| **D** Perpendicular | neck, wrist | Medium | -- |
| **E** Mesh geometry | inseam, crotch_length, front_rise, back_rise | Medium | -- |
| **F** Surface trace | shirt_length | Medium | E |
| **G** Body composition | volume, mass, bmi, body_fat | Cheap | D |
| **H** Back length | back_neck_to_waist | Cheap | A |

## Performance

`measure()` only runs the computation groups needed for the requested keys — use `only=` or `preset=` to skip expensive groups:

```python
measure(body)                         # all groups — ~800 ms
measure(body, preset="core")          # group A only — ~100 ms
measure(body, only=["bust_cm"])       # group A only — ~100 ms
measure(body, only=["shoulder_width_cm"])  # groups A + C — ~200 ms
```

### GPU acceleration

`measure()` accepts a `device` parameter (`None` = auto-detect CUDA):

```python
measure(body, only=["bust_cm"], device="cuda")  # GPU forward pass
measure(body, device=None)                       # auto: CUDA if available
```

## Optional extras

| Extra | What it enables |
|---|---|
| `[anny]` | Anny body loader (requires torch) |
| `[mhr]` | MHR body loader (requires pymomentum) |
| `[render]` | 4-view body renders (requires matplotlib, pyrender) |

Without extras, only numpy, scipy, and trimesh are required.

## Demo

Try the full pipeline at [clad.you/size-aware/size-me](https://clad.you/size-aware/size-me).

## Background

This library was built for [Clad](https://clad.you)'s size-aware virtual try-on pipeline. Read the full story: [A 3D Body Scan for Nine Cents — Without SMPL](https://clad.you/blog/posts/body-pipeline/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
