## clad-body — AI assistant notes

ISO 8559-1 body measurements for Anny and MHR.

> **Public API, install, quick-start, presets, full measurement table, computation groups, and `measure_grad` reference are in [README.md](README.md).** This file is loaded on every turn — it documents only gotchas, perf traps, and non-obvious internals. Do not duplicate README content here.

## Performance traps

1. **Never call `anny.create_fullbody_model()` in a loop** (~400 ms per call). `load_anny_from_params()` caches the model on the returned `AnnyBody`. To share one model across many bodies in an optimisation loop:

   ```python
   first = load_anny_from_params(param_list[0])
   model = first.model
   for params in param_list:
       body = load_anny_from_params(params)
       body._model = model  # reuse, skip model creation
       m = measure(body, only=["bust_cm"])
   ```

2. **The Anny model is stateless.** Phenotype values are forward-pass kwargs, not model state — `model(phenotype_kwargs=A)` then `model(phenotype_kwargs=B)` on the same model is safe. What varies between model instances is which `local_changes` labels are enabled (blendshape basis dimensions).

3. **For optimisation hot loops** where you already have a model and a fresh forward-pass vertex tensor, use `load_anny_from_verts(verts, model, phenotype_kwargs=..., bone_heads=..., bone_tails=...)` to wrap them into an `AnnyBody` without re-creating the model.

4. **Wall-clock cost mirrors token cost** when Claude reads test output. A pytest case calling `measure(body)` is ~800 ms / many groups; `only=["bust_cm"]` is ~100 ms / one group. Always pass `only=` or `preset=` when iterating.

## measure_grad limits

- **Anny only** — no MHR support (TODO).
- **No caching** — every call re-runs the forward pass; caller controls frequency.
- **No silent numpy fallback** for unsupported keys — raises `ValueError` (silent fallback would break gradient flow without warning).

The legacy entry points `generate_anny_mesh_from_params()`, `measure_body_from_verts()`, and `measure_body()` were **removed in 0.3.0** — they created a new Anny model (~400 ms) on every call.

## AnnyBody coordinate gotcha

`vertices` are XY-centred (via `reposition_apose`), but Anny bone positions from the forward pass are not. `_xy_offset` (np.ndarray, shape (2,)) stores the centering offset so `measure()` can align joint positions to the mesh. Handled automatically — callers don't need to think about it. `model` and `mesh` are lazy attributes.

## Soft circumference (differentiable bust / underbust / hip / thigh / neck)

Algorithm, per-key anchors, topology caches, calibration tables, and the
neck tilted-plane variant live in [`../findings/soft_circumference.md`](../findings/soft_circumference.md). Neck-specific
ISO/Navy research and the height-proportional-offset derivation are in
[`../findings/soft_neck.md`](../findings/soft_neck.md). Implementation:
[`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py).

## Group C — sleeve length (ISO 8559-1 §5.4.14 + §5.4.15)

**Differentiable through LBS.** `sleeve_length_cm` = bone-chain `||shoulder_ball − elbow|| + ||elbow − wrist||` (pose-invariant — same in A-pose and rest pose) plus a soft-tissue correction `offset = a*upperarm_loop + bias` where `upperarm_loop` is the differentiable vertex-loop upperarm circumference. All inputs flow through Anny's blendshapes + LBS skinning, so gradients propagate end-to-end. Calibrated to RMS 0.33 cm vs the slow plane-slice surface walk on the 6 testdata bodies. Implementation: `measure_sleeve_length_from_joints` in [`clad_body/measure/_lengths.py`](clad_body/measure/_lengths.py).

The shoulder anchor is `upperarm01.head` (the actual ball joint), exposed as `l_shoulder_ball`/`r_shoulder_ball` in `ANNY_JOINT_MAP`. The legacy `l_shoulder` key (= `upperarm01.tail`, mid-bicep) is still in the map and used by `measure_shoulder_width` + `find_acromion`, which are unchanged.

**Slow ISO reference** ([`measure_sleeve_length_iso_reference`](clad_body/measure/anny.py)): re-poses the body with `lowerarm01` rotation = 0° (Anny's natural rest pose has the elbow already flexed at ~42° — the convention for "elbow bent" in ISO §5.4.14/5.4.15), detects acromion / olecranon / wrist styloid via skinning weights and bone-perpendicular geometry, slices with two planes (upper-arm + forearm), walks Dijkstra shortest paths along the contours. ~1 s per body. Calibration only, never in the gradient hot loop. The two-tier split mirrors `measure_inseam` (slow) / `measure_inseam_from_perineum_vertices` (fast).

## Group E — inseam, crotch trace

**Inseam is differentiable through LBS via a curated perineum vertex pair.** `inseam_cm` = average Z (height-from-floor) of vertices `6319` and `12900`, a left/right symmetric pair on Anny's inguinal surface ~8 mm off the body centerline. The vertices ARE the perineum surface, so any blendshape that moves the perineum moves them — no kinematic-anchor + soft-tissue-correction approximation. Empirical max error vs the ISO mesh sweep across a 118-case stress matrix (testdata × leg-length blendshape sweeps + questionnaire grid + random local_changes): 0.19 cm, RMS 0.09 cm. Implementation: `measure_inseam_from_perineum_vertices` in [`clad_body/measure/_lengths.py`](clad_body/measure/_lengths.py).

**`crotch_length_cm` / `front_rise_cm` / `back_rise_cm` use an asymmetric tape-bridge model.** The trace samples z from waist down to the perineum and connects per-z surface points into front and back polylines. Per-z picks are different on front vs back because the physical tape behaves differently:

- **Front** (`_front_y_sagittal`): linear interpolation of the body contour at exactly `x=0`. The belly / pubic surface has no midline concavity so a real tape rests at the centerline; off-axis sampling would risk jumping onto inner-thigh "peninsulas" near the perineum on butterfly-shaped cross-sections.
- **Back** (`_back_y_tape_bridge`): average of contour `y` at `x = ±2.5 cm`. Models a 5-cm-wide tape that bridges the gluteal cleft (a narrow ~1 cm concavity up to 5 cm deep on Anny near the perineum) by sampling on the cheek surface either side of it. Pure `x=0` would dive into the cleft and over-measure back rise by several cm.

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
