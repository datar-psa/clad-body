# Measurement internals

Non-obvious implementation notes for specific measurement groups. Loaded on demand from [CLAUDE.md](CLAUDE.md) — only read when working on the relevant code path.

For the soft (differentiable) bust / underbust / hip / thigh / neck implementation see [`../findings/soft_circumference.md`](../findings/soft_circumference.md) and [`../findings/soft_neck.md`](../findings/soft_neck.md), with the implementation in [`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py).

## Group B — knee perpendicular slice (ISO 8559-1 §5.3.22 + §3.1.17)

**Anny** (joint-anchored, perpendicular). The kneecap-centre landmark
(ISO §3.1.17) is `upperleg02.tail` (= `lowerleg01.head` = the knee joint
articulation). On Anny A-pose meshes the patella's anterior prominence
sits at exactly this Z (verified empirically: at the joint Z the leg has
its most-negative Y in the knee region; +/- a few cm the front recedes).
The local circumference minimum is ~2 cm BELOW the kneecap (infrapatellar)
and is **not** the ISO landmark — `measure_knee` used to land near it via
a fixed-fraction sweep at 24-31 % of body height with target 27.5 %, which
also drifted off-landmark under `measure-{upper,lower}leg-height-incr`
blendshapes.

The slice is taken **perpendicular to the femur–tibia bisector** at the
joint, not horizontal: Anny legs sit 5–8 ° off vertical (femur 5°, tibia
8°), so a horizontal cut overestimates by ~1 % (~+0.3 cm). The bisector is
`(knee − hip) / |·| + (ankle − knee) / |·|`, computed from the joints
already plumbed in (`l_hip` = `upperleg02.head` = perineum, `l_knee`,
`l_ankle`). Per-leg perpendicular contour, convex-hull perimeter,
average L+R. Implementation: `measure_knee` → `_measure_knee_perpendicular`
in [`clad_body/measure/_circumferences.py`](clad_body/measure/_circumferences.py).

**MHR** path keeps the legacy fixed 24-31 % horizontal sweep at 27.5 %
(`_measure_knee_horizontal`) — `MHR_JOINT_MAP` doesn't expose hip/knee/ankle
for this purpose, and `mhr.py:measure_knee(mesh, height)` calls without
joints. Same drift caveats as the old Anny path apply if MHR is fed body
proportions far from the median; not currently a problem in practice.

The differentiable companion `measure_knee_soft` lives in
[`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py) and uses
`soft_circumference_plane` with the same per-leg origin/axis pair to keep
the gradient path consistent with the numpy reference. Calibration
A=0.9716, B=+0.4648 (MAE 0.24 cm, max 0.69 cm on 100 random bodies from
data_10k_42).

## Group B — calf two-phase joint-anchored search (ISO 8559-1 §5.3.24)

**Anny** (joint-anchored, two-phase). ISO §5.3.24 says "maximum **horizontal** circumference of the calf" — but the ISO protocol assumes a person standing erect with vertical legs, so on an Anny A-pose mesh (tibia ~8° off vertical) the principled interpretation of "horizontal" is "perpendicular to the tibia". Same logic as upperarm (45° in A-pose) and neck (15-20°): the standard's "horizontal" is shorthand for the standing-vertical case; for a posed mesh, perpendicular-to-limb is faithful.

Implementation:
- **Phase 1 — horizontal sweep** over `[ankle_z + 6 cm, knee_z − 4 cm]` from `ANNY_JOINT_MAP`'s `l_knee`/`r_knee` (= `upperleg02.tail`) and `l_ankle`/`r_ankle` (= `lowerleg02.tail`) finds the gastrocnemius-peak Z via `_two_leg_avg_circumference`. Cheap, vectorised through `MeshSlicer`.
- **Phase 2 — perpendicular slice** per leg at that Z, normal aligned with the tibia axis (`ankle − knee`), through `_perpendicular_limb_contour`. Convex-hull perimeter, average L+R.

Boundary fallback (deflated-calf safety net): if the Phase-1 max lands within one step of the upper bound, the lower leg is monotonically widening toward the knee — no real calf belly. This happens on tuned bodies where the optimizer has deflated the calf as a side effect of inflating the thighs. We fall back to `knee_z − 0.30 × (knee_z − ankle_z)` (gastrocnemius position) and slice perpendicular there. Untuned testdata bodies hit the interior-peak path (5.8-7 cm clearance from the upper bound) so the fallback is never engaged.

Implementation: `measure_calf` → `_calf_perpendicular_at_z` + `_calf_search_range` in [`clad_body/measure/_circumferences.py`](clad_body/measure/_circumferences.py).

**MHR** path keeps the legacy fixed 16-26 % height range with horizontal slicing — `MHR_JOINT_MAP` doesn't expose knee/ankle, and `mhr.py:measure_calf(mesh, height)` calls without joints. Same boundary-pathology risk applies if MHR is ever used with a tuned-deflated body; not currently a problem.

The differentiable companion `measure_calf_soft` lives in [`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py). Two-phase too: a per-leg soft-argmax over a Gaussian Z-binning of leg vertices weighted by squared distance from the per-leg axis centroid (mirrors numpy's circumference-peak search), then `soft_circumference_plane` perpendicular to the tibia at the resolved Z. An earlier per-vertex posterior-Y proxy was rejected because the Y peak and the circumference peak don't coincide on slim bodies (1.9 cm under-read on female_slim); the per-Z spread aggregate tracks circumference more faithfully.

A Gaussian anatomical prior `−β · (z − fallback_z)²` with `fallback_z = knee_z − 0.30 × (knee_z − ankle_z)` is added to the per-bin score (`β = CALF_PRIOR_BETA = 2000`). On normal bodies the gastrocnemius peak already sits at ~28-32 % from knee, so the prior is centred on the spread peak and adds essentially nothing. On bodies with no real peak (deflated calves — `weight × muscle` near zero) the prior provides a stable anatomical default, mirroring numpy's discrete boundary fallback differentiably. Calibration (100 bodies): A=1.0077, B=−0.4578, MAE 0.08 cm, max 0.92 cm. Without the prior the worst case was 1.72 cm on a single deflated body.

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
