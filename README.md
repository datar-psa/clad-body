# clad-body

ISO 8559-1 body measurements for [Anny](https://github.com/naver/anny) and [MHR](https://github.com/facebookresearch/MHR) parametric body models. Twelve keys are differentiable through PyTorch autograd for gradient-based body fitting.

Anny and MHR give you a 14–18K vertex mesh and nothing to measure it with. SMPL tooling doesn't port over, and the plane-sweep algorithms look simple until you hit convex-hull tape simulation, contour-fragment merging, and ISO-compliant landmark detection for bust/hip/crotch. `clad-body` is that work, done once — 25 anthropometric measurements over circumferences, lengths, and body composition (volume, mass, BMI, body fat), calibrated against real scan data. It's used in production at [Clad](https://clad.you) for size-aware virtual try-on.

> Also exposed as a REST API at [**api.clad.you**](https://api.clad.you) — questionnaire or photo in, GLB + measurements out. Free for now while we work out whether anyone actually wants this; key at [clad.you/developers](https://clad.you/developers).

<p align="center">
  <img src="https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/body_rotation.gif" alt="Anny body rotating with ISO 8559-1 circumference measurement contours" width="400">
  <br>
  <img src="https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/4view_male_average.png" alt="Anny body — male average, 4-view with ISO 8559-1 circumference measurements" width="700">
</p>

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

### Differentiable path — `measure_grad` (Anny only, experimental)

> **Under active development.** API surface and supported keys may change between minor versions. Fifteen keys are differentiable today; more will follow.

For autograd-based optimization of the body mesh, use `measure_grad(body)` instead of `measure(body)`. Same input, same key names — but the returned values are PyTorch tensors with autograd history, so you can put them directly into a loss and backprop into the Anny phenotype parameters.

Pass `requires_grad=True` to `load_anny_from_params` to create the body with gradient-enabled phenotype tensors (stored on `body.phenotype_kwargs`):

```python
import torch
from clad_body.load import load_anny_from_params
from clad_body.measure import measure_grad

body = load_anny_from_params(initial_params, requires_grad=True)
optimizer = torch.optim.Adam(list(body.phenotype_kwargs.values()), lr=0.01)

for step in range(500):
    optimizer.zero_grad()
    m = measure_grad(body, only=["bust_cm", "waist_cm", "inseam_cm"])
    loss = (m["bust_cm"] - 92.0) ** 2 + (m["waist_cm"] - 78.0) ** 2 + (m["inseam_cm"] - 82.0) ** 2
    loss.backward()
    optimizer.step()
```

Each `measure_grad(body)` call re-runs the forward pass using `body.phenotype_kwargs`, so after `optimizer.step()` updates the tensors the next iteration measures the new mesh. If you only want to optimize a subset of parameters, load without `requires_grad=True` and enable it per-tensor: `body.phenotype_kwargs["height"].requires_grad_(True)`.

Supported keys and their calibration error vs the ISO reference that `measure()` uses:

| Key | Error vs ISO |
|---|---|
| `height_cm`, `waist_cm` | exact (same loop / extent) |
| `bust_cm` | MAE 0.06 cm, max 0.18 cm |
| `underbust_cm` | MAE 0.39 cm, max 1.61 cm |
| `hip_cm` | MAE 0.46 cm, max 1.39 cm |
| `stomach_cm` | MAE 0.93 cm, P95 2.83 cm, max 3.61 cm (soft-argmin picks a different Z than the reference's 2 mm band scan; residual is inherent Z-choice noise) |
| `inseam_cm` | RMS 0.06 cm, max 0.10 cm |
| `sleeve_length_cm` | RMS 0.33 cm, max 0.55 cm |
| `shoulder_width_cm` | RMS 1.39 cm on 100 random bodies (91 % within ±2 cm). Max 5cm |
| `upperarm_cm` | ≤ 1 cm |
| `mass_kg` | ≤ 3 kg |
| `thigh_cm` | MAE 0.06 cm, max 0.18 cm (100 random bodies) |
| `knee_cm` | MAE 0.24 cm, P95 0.51 cm, max 0.69 cm (100 bodies). Perpendicular slice along femur–tibia bisector at `upperleg02.tail` (= ISO §3.1.17 kneecap centre on Anny A-pose meshes). |
| `calf_cm` | MAE 0.08 cm, P95 0.23 cm, max 0.92 cm (100 bodies). Per-leg Gaussian Z-binning + spread-proxy soft-argmax → perpendicular slice along tibia. Anatomical Gaussian prior (β=2000) at `knee_z − 0.30·(knee_z − ankle_z)` mirrors numpy's deflated-calf boundary fallback. |
| `neck_cm` | MAE 0.20 cm, P95 0.45 cm, max 1.01 cm (999 bodies). Tilted plane perpendicular to neck axis, anchor `0.0102 × body_height` below neck02 bone head (ISO §5.3.2 "just below the Adam's apple"). |

#### Circumference = convex hull perimeter, not contour perimeter

Both `measure()` and `measure_grad` report **convex hull circumference**, not the raw cross-section perimeter. This matches ISO 8559-1: a real measuring tape bridges across concavities (e.g., cleavage between breasts, armpit crease) rather than dipping into them. The convex hull perimeter is always ≤ the raw contour perimeter — the difference is most visible at the bust on larger cup sizes where the cleavage concavity can shorten the measurement by 1-3 cm compared to following the actual surface.

`measure_grad` builds a 72-point polygon via differentiable soft edge-plane intersection, then takes its `scipy.spatial.ConvexHull` perimeter. This is used for bust, underbust, hip, stomach, thigh, knee, calf, and neck — each at its respective anatomical level. For bust/hip/stomach/thigh the cutting plane is horizontal (`soft_circumference`); for knee, calf, and neck it tilts (`soft_circumference_plane`) because the limb axis is meaningfully off-vertical and horizontal slicing biases the result. Knee slices perpendicular to the femur–tibia bisector at the bone-anchored kneecap centre. Calf slices perpendicular to the tibia at a soft-argmax-resolved gastrocnemius Z (with a Gaussian prior centred at the anatomical 30 %-from-knee fallback). Neck tilts to perpendicular-to-the-neck-axis (the neck leans ~15-20° forward; horizontal slice would overestimate by 5-6 %). Every `measure_grad()` call recomputes the hull from scratch (new forward pass → new polygon → new hull). The hull decides *which* polygon vertices to keep (discrete, like `argmax` or `tensor[mask]`) — but the perimeter of those vertices is a plain sum of `torch.linalg.norm` over their positions, so `loss.backward()` flows gradients through the kept vertices back to the Anny phenotype params. Dropped vertices (inside the hull, e.g., cleavage or gluteal cleft bins) get zero gradient — correct, since they don't affect the tape measure. The hull indices can change between optimization steps if the body shape changes enough, but the perimeter value is continuous at those transitions so the loss doesn't jump. See `clad_body/measure/_soft_circ.py`.

For `stomach_cm` the cutting-plane height is itself differentiable: a soft-argmin over torso vertex Y (most anterior = most negative) in the belly Z-range picks the height of maximum anterior protrusion, and one `soft_circumference` is taken there. See `measure_stomach_soft` in the same file. The Z selection can drift by 1–2 cm from the reference's 2 mm band-scan argmax on bodies where multiple vertex clusters have near-identical anterior Y — which translates to ~3 cm of circumference error because the body tapers rapidly in the belly region. This is why `stomach_cm` has looser tolerance than the other soft-circ keys.

Strictly speaking, `stomach_cm` is differentiable almost everywhere, with numerically-tiny (~10⁻⁵ cm) step discontinuities where a vertex crosses the hard Z-mask boundary that gates out feet and bust from the belly soft-argmin.

Requesting any other key raises `ValueError`. There is no silent numpy fallback — it would break gradient flow without warning. For non-differentiable keys use `measure()`.

#### Parameter sensitivity

<p align="center">
  <img src="https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/sensitivity_local_changes.png" alt="Heatmap of |d(measurement)/d(local_change)| averaged across 6 reference bodies — rows are local_changes sorted by total normalised leverage, columns are the 12 supported measurements" width="700">
</p>

Jacobian heatmap for every Anny `local_changes` dimension against every supported measurement — `|d(measurement)/d(param)|` averaged across 6 reference bodies. Darker cells mean the local_change has strong leverage over that measurement; pale cells mean the gradient is near zero and Adam will barely move it. Useful when deciding which params to unfreeze for a given fit target (e.g. optimising `bust_cm` is pointless without `measure-bust-circ-incr` or `torso-scale-horiz-incr` in the active set). Regenerate with `python tools/sensitivity_map.py --output <path>`.

## Public API

| Import | What |
|---|---|
| `clad_body.load.load_anny_from_params` | Load Anny body from phenotype params |
| `clad_body.load.load_mhr_from_params` | Load MHR body from SAM 3D Body params |
| `clad_body.load.AnnyBody`, `MhrBody` | Body dataclasses |
| `clad_body.measure.measure` | Measure a body (numpy reporting path, ISO 8559-1) |
| `clad_body.measure.measure_grad` | Differentiable measurements for autograd loops (Anny only) |
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
| ![](https://raw.githubusercontent.com/datar-psa/clad-body/main/assets/contours/sleeve_length.png) | `sleeve_length_cm` | Distance from shoulder point along outside of slightly bent arm, over elbow, to wrist bone. (ISO §5.4.14 + §5.4.15 outer arm length, computed via plane-slice surface walk on rest pose; differentiable runtime is bone chain + linear correction.) | 5.7.8 | length | iso | arm | std | C | T,O |
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
| **C** Joint linear | shoulder_width, sleeve_length (ISO surface walk on re-posed body) | Very expensive | -- |
| **D** Perpendicular | neck, wrist | Medium | -- |
| **E** Mesh geometry | inseam (mesh sweep), crotch_length, front_rise, back_rise | Medium | -- |
| **F** Surface trace | shirt_length | Medium | E |
| **G** Body composition | volume, mass, bmi, body_fat | Cheap | D |
| **H** Back length | back_neck_to_waist | Cheap | A |

Groups B (thigh, knee, calf, upperarm), C (shoulder_width, sleeve_length), D (neck), and E (inseam) have differentiable alternatives in [`measure_grad`](#differentiable-path--measure_grad-anny-only-experimental) — use it for hot-loop optimization instead of calling `measure()` repeatedly.

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

## Try it

- **Consumer demo** — full pipeline (questionnaire → body → virtual try-on) at [clad.you/size-aware/size-me](https://clad.you/size-aware/size-me).
- **REST API** — same body model + these measurements behind a bearer-token API at [api.clad.you](https://api.clad.you). Swagger UI on the root. Free while we gauge whether it's useful; key management at [clad.you/developers](https://clad.you/developers).

## Background

This library was built for [Clad](https://clad.you)'s size-aware virtual try-on pipeline. Read the full story: [A 3D Body Scan for Nine Cents — Without SMPL](https://clad.you/blog/posts/body-pipeline/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
