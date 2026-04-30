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

   **⚠ Caveat (2026-04-30):** state leaks across bodies even with identical LC label sets. Empirically validated on 1000 bodies from `data_10k_42`: sharing the model produced ~3 kg systematic bias on `mass_kg` vs an unshared run, and per-body errors up to ~13 kg on the `measure_grad` path. Suspect cached forward-pass tensors / bone heads / soft-circ edge caches stored on the model object. Until the leak is rooted out, **only use this pattern when measurements are dominated by a single forward pass and you can prove independence in your specific call set** — verify by running a small unshared sample side-by-side. Don't blanket-apply to validation, training data generation, or anything where per-body accuracy matters.

2. **The Anny model is stateless.** Phenotype values are forward-pass kwargs, not model state — `model(phenotype_kwargs=A)` then `model(phenotype_kwargs=B)` on the same model is safe. What varies between model instances is which `local_changes` labels are enabled (blendshape basis dimensions). (Note: this is the *Anny model* itself. The `clad-body` measurement pipeline stores caches on the model object — see caveat in trap 1.)

3. **For optimisation hot loops** where you already have a model and a fresh forward-pass vertex tensor, use `load_anny_from_verts(verts, model, phenotype_kwargs=..., bone_heads=..., bone_tails=...)` to wrap them into an `AnnyBody` without re-creating the model.

4. **Wall-clock cost mirrors token cost** when Claude reads test output. A pytest case calling `measure(body)` is ~800 ms / many groups; `only=["bust_cm"]` is ~100 ms / one group. Always pass `only=` or `preset=` when iterating.

## measure_grad limits

- **Anny only** — no MHR support (TODO).
- **No caching** — every call re-runs the forward pass; caller controls frequency.
- **No silent numpy fallback** for unsupported keys — raises `ValueError` (silent fallback would break gradient flow without warning).

The legacy entry points `generate_anny_mesh_from_params()`, `measure_body_from_verts()`, and `measure_body()` were **removed in 0.3.0** — they created a new Anny model (~400 ms) on every call.

## AnnyBody coordinate gotcha

`vertices` are XY-centred (via `reposition_apose`), but Anny bone positions from the forward pass are not. `_xy_offset` (np.ndarray, shape (2,)) stores the centering offset so `measure()` can align joint positions to the mesh. Handled automatically — callers don't need to think about it. `model` and `mesh` are lazy attributes.

## Per-measurement implementation notes

Algorithm details, anchor choices, calibration data, and edge-case handling for individual measurement groups live in [MEASUREMENT_INTERNALS.md](MEASUREMENT_INTERNALS.md) — read it on demand when touching the corresponding code:

- **Soft circumferences** (differentiable bust / underbust / hip / thigh / knee / calf / neck) — see also [`../findings/soft_circumference.md`](../findings/soft_circumference.md), [`../findings/soft_neck.md`](../findings/soft_neck.md). Implementation: [`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py).
- **Group B — knee** (ISO §5.3.22 + §3.1.17, bone-anchored at `upperleg02.tail` = kneecap centre, perpendicular slice along femur–tibia bisector). Same plane definition for the numpy reference and the differentiable soft path. Implementation: `measure_knee` / `_measure_knee_perpendicular` in [`clad_body/measure/_circumferences.py`](clad_body/measure/_circumferences.py).
- **Group B — calf** (ISO §5.3.24, two-phase: joint-anchored max-girth Z search + perpendicular-to-tibia slice; deflated-calf fallback retained). Soft path uses Gaussian Z-binning + per-Z spread soft-argmax to mirror the numpy circumference-peak Z. Implementation: `measure_calf` / `_calf_perpendicular_at_z` in [`clad_body/measure/_circumferences.py`](clad_body/measure/_circumferences.py); `measure_calf_soft` in [`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py).
- **Group C — sleeve length** (ISO §5.4.14/§5.4.15, fast LBS bone-chain + slow ISO plane-slice reference). Implementation: [`clad_body/measure/_lengths.py`](clad_body/measure/_lengths.py).
- **Group E — inseam, crotch trace** (perineum vertex pair, asymmetric front/back tape-bridge model). Implementation: [`clad_body/measure/_lengths.py`](clad_body/measure/_lengths.py).

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
