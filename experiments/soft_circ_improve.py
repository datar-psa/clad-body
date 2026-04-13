"""Improvement experiment for SHAPY-lite soft circumference.

Protocol:
  1. Test each config on 6 testdata bodies (fast, <10s)
  2. Promote to 100-body dataset test only if better than baseline on both metrics

Baseline (n_bins=72, sigma_z=0.005, tau=0.100, scaled sigma_theta):
  6-body in-sample:   bust MAE=1.32 cm, underbust MAE=1.42 cm
  100-body in-sample: bust MAE=1.32 cm, underbust MAE=1.42 cm

Spike clip (round 2):
  After angular-bin softmax, apply a one-sided rolling-median clip to r_bin.
  For each bin whose radius exceeds (median-of-window-neighbors + thresh_m),
  clip it DOWN to the neighbor median.  Never clips upward → concave armpit
  bins are unaffected.  thresh_m is in metres (0.010 = 1 cm).

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_improve.py
"""

import json
import os
import random
import sys

import numpy as np
import torch

CLAD_BODY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLAD_BODY_ROOT)

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure import measure
from clad_body.measure.anny import BREAST_BONES, build_arm_mask, load_phenotype_params

sys.path.insert(0, os.path.join(CLAD_BODY_ROOT, "experiments"))
from soft_circ_experiment import (
    build_torso_edge_indices,
    bust_z_differentiable,
    _apply_coordinate_transform,
    TESTDATA_DIR,
    ALL_SUBJECTS,
)

DATA_PATH = os.path.normpath(
    os.path.join(
        CLAD_BODY_ROOT, "..", "body-tuning", "questionnaire",
        "data_10k_42", "train.json"
    )
)

# Baseline in-sample MAE on 100 bodies (n72, sigma_z=0.005, tau=0.100, scaled σ_θ)
BASELINE_BUST_MAE = 1.32
BASELINE_UB_MAE   = 1.42


# ── Core circumference with configurable sigma_theta ──────────────────────────

def soft_circ(verts, edge_indices, z, n_bins, sigma_z, tau,
              sigma_theta=None, clip_window=0, clip_thresh=0.0,
              recenter=False):
    """soft_circumference with optional fixed sigma_theta and spike clip.

    sigma_theta:  None → scaled default (bin_width * 0.6);
                  float (radians) → fixed regardless of n_bins.
    clip_window:  half-width of rolling-median window (bins).  0 = disabled.
    clip_thresh:  absolute threshold in metres above which a bin is clipped
                  down to its neighbor median.  Only clips downward.
    recenter:     if True, compute polar coordinates relative to the weighted
                  centroid of crossing points (fixes empty back bins on
                  D-shaped cross-sections).  Default False for backward compat.
    """
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=verts.dtype, device=verts.device)

    v   = verts[0]
    va  = v[edge_indices[:, 0]]
    vb  = v[edge_indices[:, 1]]
    za, zb = va[:, 2], vb[:, 2]
    dz  = zb - za
    t   = (z - za) / (dz + 1e-10)

    w   = torch.sigmoid(t / tau) * torch.sigmoid((1.0 - t) / tau)
    w   = w * torch.sigmoid(torch.abs(dz) / sigma_z - 1.0)

    t_c = t.clamp(0, 1)
    px  = va[:, 0] + t_c * (vb[:, 0] - va[:, 0])
    py  = va[:, 1] + t_c * (vb[:, 1] - va[:, 1])

    # ── Polar origin: mesh centre (0,0) or weighted crossing centroid ────────
    if recenter:
        w_sum = w.sum() + 1e-10
        cx = (w * px).sum() / w_sum
        cy = (w * py).sum() / w_sum
        # Detach centroid so it acts as a stable anchor, not a gradient sink
        cx = cx.detach()
        cy = cy.detach()
    else:
        cx = torch.tensor(0.0, dtype=verts.dtype, device=verts.device)
        cy = torch.tensor(0.0, dtype=verts.dtype, device=verts.device)

    dx  = px - cx
    dy  = py - cy
    r   = torch.sqrt(dx**2 + dy**2 + 1e-10)
    theta = torch.atan2(dy, dx)

    bin_centers = torch.linspace(
        -np.pi, np.pi * (1 - 2.0 / n_bins), n_bins,
        device=verts.device, dtype=verts.dtype
    )
    bin_width = 2 * np.pi / n_bins
    if sigma_theta is None:
        sig_th = bin_width * 0.6          # scaled default
    else:
        sig_th = float(sigma_theta)       # fixed

    ang_diff    = theta.unsqueeze(-1) - bin_centers.unsqueeze(0)
    ang_diff    = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
    ang_aff     = torch.exp(-ang_diff**2 / (2 * sig_th**2))

    comb_w      = w.unsqueeze(-1) * ang_aff
    masked_w    = comb_w * (w.unsqueeze(-1) > 0.01).float()
    log_w       = torch.log(masked_w + 1e-30) + r.unsqueeze(-1) / tau
    log_max     = log_w.max(dim=0, keepdim=True).values
    exp_w       = torch.exp(log_w - log_max) * (masked_w > 1e-20).float()
    r_bin       = (r.unsqueeze(-1) * exp_w).sum(0) / (exp_w.sum(0) + 1e-10)

    # ── Spike clip: one-sided rolling-median (downward only) ─────────────────
    if clip_window > 0 and clip_thresh > 0.0:
        r_np = r_bin.detach().cpu().numpy().copy()
        cw = clip_window
        n = n_bins
        for i in range(n):
            idxs = [(i - cw + j) % n for j in range(2 * cw + 1) if j != cw]
            nbr_med = float(np.median(r_np[idxs]))
            if r_np[i] > nbr_med + clip_thresh:
                r_np[i] = nbr_med
        r_bin = torch.tensor(r_np, dtype=verts.dtype, device=verts.device)

    # Convert back to Cartesian (relative to centroid, then offset back)
    pts  = torch.stack([cx + r_bin * torch.cos(bin_centers),
                        cy + r_bin * torch.sin(bin_centers)], dim=-1)

    # Convex hull perimeter (tape-measure behaviour: bridges concavities)
    # Hull index selection is non-differentiable but gradients flow through
    # the selected vertex positions (sum of edge norms is differentiable).
    from scipy.spatial import ConvexHull as _CH
    try:
        hull_idx = _CH(pts.detach().cpu().numpy()).vertices
        hp = pts[hull_idx]
        circ = torch.sum(torch.linalg.norm(torch.roll(hp, -1, 0) - hp, dim=-1))
    except Exception:
        circ = torch.sum(torch.linalg.norm(torch.roll(pts, -1, 0) - pts, dim=-1))
    return circ


# ── Shared model pack ─────────────────────────────────────────────────────────

def _make_model_pack(params):
    body  = load_anny_from_params(params, requires_grad=False)
    model = body.model
    _, torso_edges = build_torso_edge_indices(model, model.faces)

    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    bw  = np.zeros(vbw.shape[0])
    for bi in BREAST_BONES:
        bw += np.where(vbi == bi, vbw, 0).sum(axis=1)
    breast_idx = torch.tensor(np.where(bw > 0.3)[0], dtype=torch.long)

    pose = build_anny_apose(model, "cpu")
    return model, torso_edges, breast_idx, pose


def _forward(model, body_obj, pose):
    body_obj._model = model
    with torch.no_grad():
        out = model(
            pose_parameters=pose,
            phenotype_kwargs=body_obj.phenotype_kwargs,
            local_changes_kwargs=body_obj.local_changes_kwargs or {},
            pose_parameterization="root_relative_world",
            return_bone_ends=True,
        )
        model._last_bone_heads = out["bone_heads"]
        model._last_bone_tails = out["bone_tails"]
        verts_zup, _ = _apply_coordinate_transform(out["vertices"], model)
    return verts_zup


# ── Stage 1: 6 testdata bodies ────────────────────────────────────────────────

def _parse_cfg(cfg):
    """Extract config fields, supporting both old (4-6 element) and new (dict) formats."""
    if isinstance(cfg, dict):
        return cfg
    # Legacy tuple format
    d = {
        "n_bins": cfg[0], "sigma_z": cfg[1], "tau": cfg[2],
        "sigma_theta": cfg[3],
        "clip_window": cfg[4] if len(cfg) > 4 else 0,
        "clip_thresh": cfg[5] if len(cfg) > 5 else 0.0,
        "recenter": cfg[6] if len(cfg) > 6 else False,
    }
    return d


def run_on_6(cfg, model6, torso_edges, breast_idx, pose6):
    c = _parse_cfg(cfg)
    bust_raw, bust_ref = [], []
    ub_raw,   ub_ref   = [], []

    for subject in ALL_SUBJECTS:
        params_path = os.path.join(TESTDATA_DIR, subject, "anny_params.json")
        params  = load_phenotype_params(params_path)
        body    = load_anny_from_params(params, requires_grad=False)
        body._model = model6
        verts   = _forward(model6, body, pose6)

        ref = measure(body, only=["bust_cm", "underbust_cm"])
        ref_bust = ref["bust_cm"]
        ref_ub   = ref["underbust_cm"]

        bust_z_t = bust_z_differentiable(model6, verts)
        ub_z_t   = verts[0, breast_idx, 2].min()

        if bust_z_t is not None:
            sc_b = soft_circ(verts, torso_edges, bust_z_t,
                             c["n_bins"], c["sigma_z"], c["tau"], c["sigma_theta"],
                             c["clip_window"], c["clip_thresh"], c["recenter"])
            bust_raw.append(sc_b.item() * 100)
            bust_ref.append(ref_bust)

        sc_u = soft_circ(verts, torso_edges, ub_z_t,
                         c["n_bins"], c["sigma_z"], c["tau"], c["sigma_theta"],
                         c["clip_window"], c["clip_thresh"], c["recenter"])
        ub_raw.append(sc_u.item() * 100)
        ub_ref.append(ref_ub)

    return (np.array(bust_raw), np.array(bust_ref),
            np.array(ub_raw),   np.array(ub_ref))


# ── Stage 2: 100 dataset bodies ───────────────────────────────────────────────

def load_100(seed=42):
    with open(DATA_PATH) as f:
        all_entries = json.load(f)
    return random.Random(seed).sample(all_entries, 100)


def run_on_100(cfg, entries, model100, torso_edges, breast_idx, pose100):
    c = _parse_cfg(cfg)
    bust_raw, bust_ref = [], []
    ub_raw,   ub_ref   = [], []

    for entry in entries:
        m_ref = entry["measurements"]
        ref_b = m_ref.get("bust_cm")
        ref_u = m_ref.get("underbust_cm")
        if ref_b is None or ref_u is None:
            continue
        try:
            body   = load_anny_from_params(entry["params"], requires_grad=False)
            verts  = _forward(model100, body, pose100)

            bust_z = bust_z_differentiable(model100, verts)
            ub_z   = verts[0, breast_idx, 2].min()

            if bust_z is not None:
                sc_b = soft_circ(verts, torso_edges, bust_z,
                                 c["n_bins"], c["sigma_z"], c["tau"], c["sigma_theta"],
                                 c["clip_window"], c["clip_thresh"], c["recenter"])
                bust_raw.append(sc_b.item() * 100)
                bust_ref.append(ref_b)

            sc_u = soft_circ(verts, torso_edges, ub_z,
                             c["n_bins"], c["sigma_z"], c["tau"], c["sigma_theta"],
                             c["clip_window"], c["clip_thresh"], c["recenter"])
            ub_raw.append(sc_u.item() * 100)
            ub_ref.append(ref_u)
        except Exception:
            pass

    return (np.array(bust_raw), np.array(bust_ref),
            np.array(ub_raw),   np.array(ub_ref))


# ── Stats helpers ─────────────────────────────────────────────────────────────

def fit_linear(raw, ref):
    A = np.column_stack([raw, np.ones(len(raw))])
    ab, _, _, _ = np.linalg.lstsq(A, ref, rcond=None)
    return ab[0], ab[1]


def stats(raw, ref):
    a, b = fit_linear(raw, ref)
    err  = a * raw + b - ref
    return {
        "mae":  np.mean(np.abs(err)),
        "max":  np.max(np.abs(err)),
        "p90":  np.percentile(np.abs(err), 90),
        "bias": np.mean(err),
        "a": a, "b": b,
    }


def fmt(s):
    return (f"bust MAE={s[0]['mae']:.2f} max={s[0]['max']:.2f} p90={s[0]['p90']:.2f}  │"
            f"  ub MAE={s[1]['mae']:.2f} max={s[1]['max']:.2f} p90={s[1]['p90']:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def cfg_label(cfg):
    c = _parse_cfg(cfg) if not isinstance(cfg, dict) else cfg
    sig = f"σ_θ={np.degrees(c['sigma_theta']):.0f}°" if c.get("sigma_theta") else "scaled"
    clip = f" clip={c['clip_window']}×{c['clip_thresh']*1000:.0f}µm" if c.get("clip_window", 0) > 0 else ""
    rc = " RC" if c.get("recenter") else ""
    return f"n={c['n_bins']} sz={c['sigma_z']} τ={c['tau']} {sig}{clip}{rc}"


if __name__ == "__main__":
    # (n_bins, sigma_z, tau, sigma_theta, clip_window, clip_thresh_m, recenter)
    configs = [
        # ── Previous best (no recenter) — reference ──────────────
        (72,  0.005, 0.050, None, 3, 0.012, False),  # current best

        # ── Recentered: same configs, with recenter=True ─────────
        # No clip (recenter alone should fix the back bins)
        (72,  0.005, 0.050, None, 0, 0.0, True),
        (72,  0.005, 0.100, None, 0, 0.0, True),

        # Recenter + clip (does clip still help after recentering?)
        (72,  0.005, 0.050, None, 3, 0.012, True),
        (72,  0.005, 0.050, None, 3, 0.008, True),
        (72,  0.005, 0.050, None, 5, 0.010, True),
        (72,  0.005, 0.100, None, 3, 0.012, True),
    ]

    # ── Setup models ──────────────────────────────────────────────────────────
    print("Setting up model for 6-body test...")
    params6 = load_phenotype_params(
        os.path.join(TESTDATA_DIR, ALL_SUBJECTS[0], "anny_params.json")
    )
    model6, torso_edges6, breast_idx6, pose6 = _make_model_pack(params6)
    print(f"  Torso edges: {len(torso_edges6)}")

    print("\nLoading 100 dataset entries...")
    entries100 = load_100(seed=42)
    first_params = entries100[0]["params"]

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Config':<40s}  {'Stage':>7s}  Bust MAE  Bust max  Bust p90  │  UB MAE  UB max  UB p90")
    print("=" * 90)

    promoted = []

    for cfg in configs:
        label = cfg_label(cfg)

        # Stage 1: 6 bodies
        br6, bref6, ur6, uref6 = run_on_6(cfg, model6, torso_edges6, breast_idx6, pose6)
        sb6 = stats(br6, bref6)
        su6 = stats(ur6, uref6)
        tag6 = "  BETTER" if (sb6["mae"] < BASELINE_BUST_MAE and su6["mae"] < BASELINE_UB_MAE) else ""
        print(f"  {label:<40s}  {'6-body':>7s}  "
              f"{sb6['mae']:>8.2f}  {sb6['max']:>8.2f}  {sb6['p90']:>8.2f}  │"
              f"  {su6['mae']:>6.2f}  {su6['max']:>6.2f}  {su6['p90']:>6.2f}{tag6}")

        # Stage 2: promote to 100 if better on both
        if sb6["mae"] < BASELINE_BUST_MAE and su6["mae"] < BASELINE_UB_MAE:
            # Lazy-init 100-body model pack on first promotion
            if not promoted:
                print(f"\n  [Promoting to 100-body test — initialising dataset model...]\n")
                model100, torso_edges100, breast_idx100, pose100 = _make_model_pack(first_params)

            br100, bref100, ur100, uref100 = run_on_100(
                cfg, entries100, model100, torso_edges100, breast_idx100, pose100
            )
            sb100 = stats(br100, bref100)
            su100 = stats(ur100, uref100)
            won = sb100["mae"] < BASELINE_BUST_MAE and su100["mae"] < BASELINE_UB_MAE
            tag100 = "  *** NEW BEST ***" if won else "  (worse on 100)"
            print(f"  {label:<40s}  {'100-body':>7s}  "
                  f"{sb100['mae']:>8.2f}  {sb100['max']:>8.2f}  {sb100['p90']:>8.2f}  │"
                  f"  {su100['mae']:>6.2f}  {su100['max']:>6.2f}  {su100['p90']:>6.2f}{tag100}")
            if won:
                promoted.append((cfg, sb100, su100))

    print("\n" + "=" * 90)
    if promoted:
        print("PROMOTED configs (better than baseline on 100 bodies):")
        for cfg, sb, su in promoted:
            print(f"  {cfg_label(cfg)}")
            print(f"    bust:      MAE={sb['mae']:.2f}  max={sb['max']:.2f}  "
                  f"correction: y = {sb['a']:.4f}*x + {sb['b']:+.4f}")
            print(f"    underbust: MAE={su['mae']:.2f}  max={su['max']:.2f}  "
                  f"correction: y = {su['a']:.4f}*x + {su['b']:+.4f}")
    else:
        print("No config beat the baseline on 100 bodies.")
        print(f"  Baseline: bust MAE={BASELINE_BUST_MAE}  underbust MAE={BASELINE_UB_MAE}")
