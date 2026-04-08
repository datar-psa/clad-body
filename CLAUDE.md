# clad-body

ISO 8559-1 body measurement library for Anny and MHR parametric body models.

## API — use this, not the deprecated functions

```python
from clad_body.load.anny import load_anny_from_params
from clad_body.load.mhr import load_mhr_from_params
from clad_body.measure import measure

body = load_anny_from_params(params)
m = measure(body, only=["bust_cm"])     # fast: ~100 ms
m = measure(body, preset="core")        # 4 keys: height, bust, waist, hip
m = measure(body)                       # all keys: ~800 ms
m = measure(body, device="cuda")        # GPU acceleration
```

For optimisation hot loops where you already have a model and a fresh forward-pass vertex tensor and don't want to pay for `load_anny_from_params()` re-creating the model, use `load_anny_from_verts(verts, model, phenotype_kwargs=..., bone_heads=..., bone_tails=...)` which wraps the existing model + verts into an `AnnyBody` ready for `measure()`.

### Differentiable measurements (`measure_grad`)

```python
from clad_body.measure import measure, measure_grad

# Reporting path (numpy, JSON-safe)
body = load_anny_from_params(params)
m = measure(body, only=["waist_cm", "inseam_cm"])

# Gradient path (torch, for autograd loops)
# Requires a forward pass WITHOUT torch.no_grad() so the graph is preserved:
output = model(pose_parameters=a_pose, phenotype_kwargs=kwargs,
               return_bone_ends=True)
model._last_bone_heads = output["bone_heads"]
model._last_bone_tails = output["bone_tails"]
verts = output["vertices"]  # (1, V, 3) — keep computation graph, don't detach

m = measure_grad(model, verts, only=["waist_cm", "inseam_cm", "sleeve_length_cm"])
loss = (m["waist_cm"] - target_waist) ** 2 + (m["inseam_cm"] - target_inseam) ** 2
loss.backward()  # gradients flow back to phenotype_kwargs tensors
```

**When to use which API:**

| API | Use for | Returns |
|-----|---------|---------|
| `measure(body)` | Reporting, JSON export, all keys, MHR bodies | `dict[str, float]` |
| `measure_grad(model, verts)` | Gradient-based optimisation hot loops (tune.py) | `dict[str, torch.Tensor]` |

**`measure_grad` supported keys (Anny only, v1):**
- `height_cm` — `verts[:, :, height_axis].max() - .min()`
- `waist_cm` — vertex-loop circumference (waist loop)
- `thigh_cm` — vertex-loop circumference (thigh loop)
- `upperarm_cm` — vertex-loop circumference (upperarm loop)
- `inseam_cm` — `measure_inseam_from_joints` (hip joint Z + soft-tissue correction)
- `sleeve_length_cm` — `measure_sleeve_length_from_joints` (bone chain + upperarm correction)

Requesting any other key raises `ValueError` listing what IS supported. Silent numpy fallback is intentionally absent — it would break gradient flow without warning.

**`measure_grad` does NOT:**
- Support MHR (TODO: add `_measure_mhr_grad` when needed)
- Cache results (caller controls forward-pass frequency)
- Fall back to numpy for unsupported keys

The legacy entry points `generate_anny_mesh_from_params()`, `measure_body_from_verts()`, and `measure_body()` were **removed in 0.3.0** — they created a new Anny model (~400 ms) on every call.

## Performance rules

1. **Always use `only=` or `preset=`** when you don't need all measurements. `measure(body)` runs all 7 computation groups (~800 ms). `measure(body, only=["bust_cm"])` runs only GROUP_A (~100 ms).

2. **Never call `anny.create_fullbody_model()` in a loop.** It takes ~400 ms per call. `load_anny_from_params()` caches the model on the returned `AnnyBody`. For hot loops (optimization, batch processing), reuse the model:

   ```python
   # WRONG — creates model every iteration
   for params in param_list:
       body = load_anny_from_params(params)
       m = measure(body)

   # RIGHT — same speed, model is on each body
   # (load_anny_from_params caches model internally)
   for params in param_list:
       body = load_anny_from_params(params)
       m = measure(body, only=["bust_cm"])

   # FASTEST — share model across bodies
   first = load_anny_from_params(param_list[0])
   model = first.model
   for params in param_list:
       body = load_anny_from_params(params)
       body._model = model  # reuse, skip model creation
       m = measure(body, only=["bust_cm"])
   ```

3. **The Anny model is stateless.** Phenotype values are forward-pass kwargs, not model state. `model(phenotype_kwargs=A)` then `model(phenotype_kwargs=B)` on the same model is safe. What varies between model instances is which `local_changes` labels are enabled (blendshape basis dimensions).

## AnnyBody

`AnnyBody` is the primary data class. Created by `load_anny_from_params()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `vertices` | `np.ndarray` (N, 3) | Z-up, metres, XY-centred, feet at Z=0 |
| `faces` | `np.ndarray` (M, 3) | Triangulated face indices |
| `phenotype_params` | `dict` | Phenotype values + `_local_changes` |
| `mesh` | `trimesh.Trimesh` | Lazy cached property |
| `model` | Anny rigged model | Lazy — created from params if not set |
| `_xy_offset` | `np.ndarray` (2,) | XY centering offset (for joint alignment) |

**Coordinate gotcha**: `vertices` are XY-centred (via `reposition_apose`), but Anny bone positions from the forward pass are not. `_xy_offset` stores the centering offset so `measure()` can align joint positions to the mesh. This is handled automatically — callers don't need to worry about it.

## Computation groups

`measure()` only runs groups needed for the requested keys:

| Group | Keys | Cost |
|-------|------|------|
| **A** Core torso | height, bust, waist, hip, stomach, underbust, belly_depth | ~100 ms |
| **B** Limb sweeps | thigh, knee, calf, upperarm | ~300 ms |
| **C** Joint linear | shoulder_width, sleeve_length (differentiable) | ~100 ms |
| **D** Perpendicular | neck, wrist | ~200 ms |
| **E** Mesh geometry | inseam (joint-based, differentiable), crotch_length, front_rise, back_rise | ~100 ms |
| **F** Surface trace | shirt_length (depends on E) | ~100 ms |
| **G** Body composition | volume, mass, bmi, body_fat (depends on D) | ~50 ms |

### Group C notes — sleeve length (ISO 8559-1 §5.4.14 + §5.4.15)

**Sleeve length is differentiable through LBS.** `sleeve_length_cm` is computed from the bone-chain `||shoulder_ball − elbow|| + ||elbow − wrist||` (pose-invariant — same in A-pose and rest pose) plus a soft-tissue correction `offset = a*upperarm_loop + bias` where `upperarm_loop` is the differentiable vertex-loop upperarm circumference. All inputs flow through Anny's blendshapes + LBS skinning, so gradients propagate end-to-end. Calibrated to RMS 0.33 cm vs the slow plane-slice surface walk reference on the 6 testdata bodies. Implementation: `measure_sleeve_length_from_joints` in [`clad_body/measure/_lengths.py`](clad_body/measure/_lengths.py).

The shoulder anchor is `upperarm01.head` (the actual ball joint), exposed as the `l_shoulder_ball`/`r_shoulder_ball` keys in `ANNY_JOINT_MAP`. The legacy `l_shoulder` key (= `upperarm01.tail`, mid-bicep) is still in the map and used by `measure_shoulder_width` + `find_acromion`, which are unchanged.

**Slow ISO reference** ([`measure_sleeve_length_iso_reference`](clad_body/measure/anny.py) in `anny.py`): re-poses the body with `lowerarm01` rotation = 0° (Anny's natural rest pose has the elbow already flexed at ~42° — the convention for "elbow bent" in ISO §5.4.14/5.4.15), detects acromion / olecranon / wrist styloid via skinning weights and bone-perpendicular geometry, slices the body with two planes (upper-arm + forearm), and walks Dijkstra shortest paths along the resulting contours. ~1 second per body. Used only for calibration, not in the gradient hot loop.

The two-tier split mirrors `measure_inseam` (slow mesh sweep) / `measure_inseam_from_joints` (fast differentiable).

### Group E notes — inseam, crotch trace

**Inseam is differentiable through LBS.** `inseam_cm` is computed from the Z position of the `upperleg01.L/R` bone tails (the perineum-level articulation, not the femoral head) plus a soft-tissue correction `offset = a*thigh_loop + b*pelvis_width + c` where `pelvis_width = |l_hip.x − r_hip.x|` and `thigh_loop` is the vertex-loop thigh circumference. All inputs flow through Anny's blendshapes + LBS skinning, so gradients propagate end-to-end. Calibrated to RMS 0.06 cm vs the ISO 8559-1 mesh-sweep on the testdata bodies. Falls back to the mesh sweep only if the joints are missing. Implementation: `measure_inseam_from_joints` in [`clad_body/measure/_lengths.py`](clad_body/measure/_lengths.py).

**`crotch_length_cm` / `front_rise_cm` / `back_rise_cm` use an asymmetric tape-bridge model.** The trace samples z from waist down to the perineum and connects per-z surface points into front and back polylines. Per-z picks are different on front vs back because the physical tape behaves differently:

- **Front** (`_front_y_sagittal`): linear interpolation of the body contour at exactly `x=0`. The belly / pubic surface has no midline concavity so a real tape rests at the centerline; off-axis sampling would risk jumping onto inner-thigh "peninsulas" near the perineum on butterfly-shaped cross-sections.
- **Back** (`_back_y_tape_bridge`): average of contour `y` at `x = ±2.5 cm`. Models a 5-cm-wide tape that bridges the gluteal cleft (a narrow ~1-cm concavity up to 5 cm deep on Anny near the perineum) by sampling on the cheek surface either side of it. Pure `x=0` would dive into the cleft and over-measure back rise by several cm.

Termination is **topological**: the trace stops as soon as the slicer returns ≥2 body-shaped contours (legs separated → perineum reached). This avoids `np.arange` step-boundary aliasing where a 0.5 mm change in `crotch_z` could shift `crotch_length_cm` by 3 cm via a single included/excluded sample.

## Tests

```bash
cd hmr/clad-body
UV_PROJECT_ENVIRONMENT=venv uv sync --extra anny --extra render --extra dev
venv/bin/pytest tests/ -v
```

MHR tests require downloaded assets (189 MB):
```bash
curl -L https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip -o /tmp/mhr_assets.zip
unzip /tmp/mhr_assets.zip -d venv/lib/python3.12/site-packages/
```
