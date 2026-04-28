# Measurement internals

Non-obvious implementation notes for specific measurement groups. Loaded on demand from [CLAUDE.md](CLAUDE.md) — only read when working on the relevant code path.

For the soft (differentiable) bust / underbust / hip / thigh / neck implementation see [`../findings/soft_circumference.md`](../findings/soft_circumference.md) and [`../findings/soft_neck.md`](../findings/soft_neck.md), with the implementation in [`clad_body/measure/_soft_circ.py`](clad_body/measure/_soft_circ.py).

## Group B — calf horizontal sweep (ISO 8559-1 §5.3.24)

**Anny** (joint-anchored): bounds `[ankle_z + 6 cm, knee_z − 4 cm]` from `ANNY_JOINT_MAP`'s `l_knee`/`r_knee` (= `upperleg02.tail`) and `l_ankle`/`r_ankle` (= `lowerleg02.tail`). After picking the global max, if it lands within one step of the upper bound there is no real calf-belly peak — the lower leg is monotonically widening toward the knee, which happens on tuned bodies where the optimizer has deflated the calf as a side effect of inflating the thighs. In that case we report girth at `knee_z − 0.30 × (knee_z − ankle_z)` (gastrocnemius position) instead of the patellar clip. The reported `calf_cm` is then no longer a true ISO max but is anatomically honest about the deflated geometry. Untuned testdata bodies hit the interior-peak path (5.8–7 cm clearance from the upper bound, 9.9–11.1 cm gap from knee bone) so the fallback is never engaged for them and baselines are unchanged. Implementation: `measure_calf` + `_calf_search_range` in [`clad_body/measure/_circumferences.py`](clad_body/measure/_circumferences.py).

**MHR** path keeps the legacy fixed 16–26 % height sweep — `MHR_JOINT_MAP` does not yet have knee/ankle entries, and `mhr.py:measure_calf(mesh, height)` calls without joints. Same boundary-pathology risk applies if MHR is ever used with a tuned-deflated body; not currently a problem.

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
