# Soft circumference — method, tuning, and back coverage

Experiment date: 2026-04-13

---

## 1. Method overview

`soft_circ` is a differentiable circumference estimator that approximates the
plane-sweep + convex-hull measurement used in `clad_body.measure`.

### 1.1 Inputs

- `verts`: (1, V, 3) body mesh vertices, Z-up, metres, XY-centred
- `edge_indices`: (E, 2) torso-only edges (arm vertices excluded via `build_arm_mask`)
- `z`: scalar — height of the horizontal cutting plane
- `n_bins`, `sigma_z`, `tau`: hyperparameters (see §2)

### 1.2 Torso edge construction

Faces where any vertex has its dominant bone in `ARM_HAND_BONES` (bone indices
49–73 and 75–99) are excluded. All unique edges of the remaining faces become
the torso edge set. This removes arms but keeps all torso, neck, and hip edges.

### 1.3 Cutting-plane heights

- **Bust z**: derived from `breast.L` and `breast.R` bone tail positions,
  projected to the mesh height axis via `bust_z_differentiable`.
- **Underbust z**: `verts[0, breast_idx, 2].min()` — minimum z of all vertices
  with breast bone skinning weight > 0.3. Approximates the inframammary fold.

### 1.4 Algorithm steps

**Step 1 — Edge-plane intersection (soft)**

For each torso edge (va, vb):
```
dz = zb − za
t  = (z − za) / (dz + ε)
```
Soft crossing weight:
```
w = sigmoid(t / tau) × sigmoid((1 − t) / tau)     # peaks when 0 < t < 1
w = w × sigmoid(|dz| / sigma_z − 1)               # down-weight near-horizontal edges
```
Intersection point (clamped):
```
t_c = clamp(t, 0, 1)
px  = va.x + t_c × (vb.x − va.x)
py  = va.y + t_c × (vb.y − va.y)
```

**Step 2 — Polar binning**

Convert intersection points to polar: `r = sqrt(px² + py²)`, `θ = atan2(py, px)`.
72 angular bins from −π to π, each 5° wide. Angular affinity to each bin:
```
sig_th   = bin_width × 0.6  ≈ 0.052 rad ≈ 3°
ang_diff = atan2(sin(θ_edge − θ_bin), cos(θ_edge − θ_bin))
ang_aff  = exp(−ang_diff² / (2 × sig_th²))
```

**Step 3 — Soft-max radius per bin (convex hull proxy)**

Per-bin radius uses a log-space softmax biased toward larger r:
```
comb_w   = w × ang_aff                                 # (E, N)
masked_w = comb_w × (w > 0.01)                         # gate negligible edges
log_w    = log(masked_w + ε) + r / tau                  # bias toward outermost
exp_w    = exp(log_w − max(log_w)) × (masked_w > 1e-20)
r_bin    = Σ(r × exp_w) / Σ(exp_w)                     # per-bin radius
```

**Step 4 — Spike clip (one-sided rolling median)**

For each bin, if its radius exceeds the median of its `2×clip_window`
neighbours by more than `clip_thresh` metres, clip it DOWN to that median.
Never clips upward (concave regions like armpits are preserved).
```
for i in range(n_bins):
    neighbours = [r_bin[(i − w + j) % n] for j ≠ i in window]
    if r_bin[i] > median(neighbours) + clip_thresh:
        r_bin[i] = median(neighbours)
```

**Step 5 — Circumference**

Convert per-bin radii back to Cartesian polygon and sum chord lengths:
```
pts  = [r_bin[i] × (cos θ_i, sin θ_i)]
circ = Σ ||pts[i+1] − pts[i]||
```

### 1.5 Calibration

Raw soft_circ output is mapped to cm via a linear fit on 100 bodies:
```
predicted_cm = A × raw_cm + B
```
Coefficients are fit per measurement (bust / underbust) using least-squares
regression against the reference `measure(body, only=[...])` values.

---

## 2. Best hyperparameters

### 2.1 Current best: recentered (2026-04-13)

| Parameter | Value | Notes |
|---|---|---|
| n_bins | 72 | 5° angular resolution |
| sigma_z | 0.005 m | Soft gate width for edge dz |
| tau | 0.050 m | Sigmoid gate width + softmax temperature |
| sigma_theta | scaled (bin_width × 0.6) | ≈ 0.052 rad ≈ 3° |
| recenter | True | Polar origin = weighted crossing centroid |
| clip_window | 0 | Not needed after recentering |
| clip_thresh | 0.0 | Not needed after recentering |

Calibration coefficients:
```
Bust:      A = 0.9700,  B = 2.16
Underbust: A = 0.9872,  B = 1.14
```

Dataset: `random.Random(42).sample(train.json, 100)`.

| Metric | Bust | Underbust | Combined max(|bust_err|, |ub_err|) |
|---|---|---|---|
| MAE | 0.60 cm | 0.35 cm | — |
| Max error | 2.33 cm | 1.32 cm | 2.33 cm |
| p90 | 1.36 cm | 0.95 cm | 1.36 cm |
| p97 | — | — | 2.15 cm |

### 2.2 Previous best: origin-centred + spike clip

| Parameter | Value | Notes |
|---|---|---|
| n_bins | 72 | 5° angular resolution |
| sigma_z | 0.005 m | Soft gate width for edge dz |
| tau | 0.050 m | Sigmoid gate width + softmax temperature |
| sigma_theta | scaled (bin_width × 0.6) | ≈ 0.052 rad ≈ 3° |
| recenter | False | Polar origin at mesh XY origin (0, 0) |
| clip_window | 3 | 6-neighbour rolling median |
| clip_thresh | 0.012 m | 12 mm above median triggers downward clip |

Calibration coefficients:
```
Bust:      A = 0.9936,  B = −0.9925
Underbust: A = 0.9758,  B =  0.4157
```

| Metric | Bust | Underbust | Combined max(|bust_err|, |ub_err|) |
|---|---|---|---|
| MAE | 1.155 cm | 0.810 cm | — |
| Max error | 3.63 cm | 4.78 cm | 4.78 cm |
| p90 | 2.54 cm | 1.63 cm | 2.61 cm |
| p97 | — | — | 3.08 cm |

### 2.3 Tuning history

Baseline (tau=0.100, no clip): bust MAE=1.32, UB MAE=1.42.

Tau reduction (0.100 → 0.050): tighter sigmoid gate, improved both metrics.

Spike clip (clip_window=3, clip_thresh=12mm): removes outlier bins from
long diagonal boundary edges. Main improvement on underbust.

Recentering (recenter=True): compute polar coordinates relative to the
weighted centroid of crossing points instead of the mesh XY origin (0, 0).
Eliminates empty back bins and outlier leakage. Spike clip has zero effect
after recentering (identical results with and without).

---

## 3. Back coverage investigation

### 3.1 Coordinate system

Anny body after `_apply_coordinate_transform`:
- Z-up, origin at ground level
- Positive Y = posterior (back), negative Y = anterior (front)
- XY-centred via `reposition_apose`

### 3.2 Cross-section geometry

At underbust height the body cross-section is D-shaped, not circular.
The XY origin sits close to the posterior surface.

Measured on average testdata body at uz=122.1 cm:

| Region | py range | r range | Notes |
|---|---|---|---|
| Front (anterior) | −25.4 to −5 cm | ~15–25 cm | Deep, many crossing edges |
| Sides | −5 to +1 cm | ~10–15 cm | Good coverage |
| Back (posterior) | +1 to +3 cm | ~2–5 cm | Shallow, sparse crossings |

### 3.3 Back crossing diagnostics

**Average testdata body** (uz=122.1 cm):

```
Back-half torso verts (py>0): 227
  z range: 80.0 to 139.6 cm
  py range: 0.0 to 3.0 cm

Back-half edges (both verts py>0): 589
Back-half edges that cross uz: 27
  |dz| range: 0.72 to 2.86 cm

Sample crossings:
  px=10.3 py=1.1  θ=atan2(1.1,10.3) ≈  6°
  px=-8.4 py=1.6  θ=atan2(1.6,-8.4) ≈ 169°
  px=1.9  py=1.2  θ=atan2(1.2,1.9)  ≈ 32°
  px=-2.5 py=1.4  θ=atan2(1.4,-2.5) ≈ 151°
```

27 back edges cross the plane, but their angular positions are at θ ≈ 6–32°
and θ ≈ 150–170°, NOT at θ ≈ 90° (pure back direction).

With sigma_theta ≈ 3°, the angular affinity from these crossings to bins
near θ=90° is exp(−(58°)²/(2×3²)) ≈ 0. Bins at θ ≈ 60–120° receive
effectively zero weight.

**p97 worst-case body** (uz=136.7 cm): 0 back-half crossings at that height.

### 3.4 Empty bin prevalence

71 out of 100 bodies have at least one angular bin where `exp_w.sum(0) = 0`
(no crossing data reaches that bin with nonzero weight).

These empty bins get r_bin ≈ 0, producing polygon vertices at the origin.

### 3.5 Root cause: r-biased softmax leaks side radii into back bins

The `(masked_w > 1e-20)` gate in the softmax is too permissive. A side
crossing at θ ≈ 0° with r = 12 cm has nonzero masked_w at bins up to
~25° away (masked_w ≈ 1e-13, still above 1e-20). The `r/tau` bias in
`log_w = log(masked_w) + r/tau` amplifies these faint contributions
because their large r adds a positive logit.

For back-facing bins (θ ≈ 10–25°) with no legitimate crossings, this
single faint contribution is the only signal, so the softmax gives it
weight ≈ 1. The bin gets r_bin ≈ 12 cm. The ring point at
`(12 × cos 20°, 12 × sin 20°) = (11.3, 4.1)` is 1–2 cm behind the
body back surface (at py ≈ 3 cm).

Beyond θ ≈ 25–30°, angular affinity finally drops below 1e-20 and the
bin goes to r = 0. The ring snaps from r = 12 to r = 0 — creating the
visible "ears" that jump behind the body and then collapse inward.

### 3.6 Circular interpolation experiment (failed)

Attempted fix: for each empty bin, linearly interpolate r from the nearest
filled neighbours on each side (circular wrap-around).

| Algorithm | Bust MAE | Bust max | UB MAE | UB max | Combined p97 | Combined max |
|---|---|---|---|---|---|---|
| No interp + calibrated coeffs | 1.155 | 3.63 | 0.810 | 4.78 | 3.08 | 4.78 |
| Interp + old coeffs | 1.256 | 5.21 | 1.558 | 9.77 | 5.73 | 9.77 |
| Interp + refitted coeffs | 1.252 | 5.18 | 1.663 | 8.84 | 5.18 | 8.84 |

All values in cm.

Circular interpolation fills empty back bins with r ≈ 8–12 cm (interpolated
from side bins). Actual back surface is at r ≈ 3–5 cm. The inflated back
radius increases raw circumference non-uniformly across bodies (depends on
how many bins are empty), which the linear calibration cannot compensate.

UB MAE degrades from 0.810 to 1.663 cm (+105%). Interpolation reverted.

### 3.7 Recentering fix (successful)

Root cause of the empty-bin / leakage problem: the XY origin sits near the
posterior surface (py ≈ +3 cm back vs py ≈ −25 cm front). All crossing
points are in the front half of the polar coordinate system, leaving back
bins empty or filled only by faint leakage.

Fix: compute polar coordinates relative to the weighted centroid of
crossing points instead of the mesh XY origin:
```
cx = Σ(w × px) / Σ(w)
cy = Σ(w × py) / Σ(w)
r  = sqrt((px − cx)² + (py − cy)²)
θ  = atan2(py − cy, px − cx)
```

Centroid is detached (no gradient flow) so it acts as a stable anchor.
After recentering, crossing points distribute evenly around 360° and
back bins receive direct data. Spike clip has zero effect (identical
results with and without).

100-body results:

| Algorithm | Bust MAE | Bust max | UB MAE | UB max | Combined p97 | Combined max |
|---|---|---|---|---|---|---|
| Origin + clip (old best) | 1.155 | 3.63 | 0.810 | 4.78 | 3.08 | 4.78 |
| **Recentered (new best)** | **0.60** | **2.33** | **0.35** | **1.32** | **2.15** | **2.33** |

Bust MAE: −48%. UB MAE: −57%. Combined max: −51%.

---

## 4. Scripts

| Script | Purpose |
|---|---|
| `soft_circ_experiment.py` | Original experiment: soft_circumference vs plane sweep |
| `soft_circ_improve.py` | Hyperparameter sweep + spike clip (canonical `soft_circ`) |
| `soft_circ_visualize.py` | Scatter plot: predicted vs reference for worst-case body |
| `soft_circ_compare.py` | Before/after spike clip 2×2 grid |
| `soft_circ_gallery.py` | 6 bodies × bust/underbust cross-section gallery |
| `soft_circ_avg.py` | Average body cross-section detail |
| `soft_circ_4view_loop.py` | 4-view pyrender body with soft ring overlay |
| `soft_circ_debug_back.py` | Back outlier diagnostic |

---

## 5. Open questions

1. For the p97 body (uz=136.7 cm), `breast_idx.z.min()` gives a height near
   the shoulder level. Is this underbust estimate correct for that body type,
   or is the breast bone vertex set misidentifying the inframammary fold?

2. The recentered centroid is detached (no gradients). Would allowing gradients
   through the centroid improve or harm downstream optimization?

3. The recentered algorithm no longer needs spike clip. Are there edge cases
   (extreme body types) where clip is still needed after recentering?
