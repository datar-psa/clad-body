"""Experiment: SHAPY-lite soft differentiable circumference vs plane sweep.

Implements a differentiable circumference function that uses:
1. Exact edge-plane intersection geometry (differentiable via linear interpolation)
2. Soft sigmoid gates instead of hard boolean edge-crossing masks
3. Angular binning with soft-max radius for convex-hull-equivalent perimeter

Compares against the reference plane sweep (MeshSlicer + ConvexHull) for bust
and underbust on all 6 testdata bodies.

Usage:
    cd hmr/clad-body
    venv/bin/python experiments/soft_circ_experiment.py
"""

import os
import sys

import numpy as np
import torch

# ── Setup paths ──
CLAD_BODY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CLAD_BODY_ROOT)

from clad_body.load.anny import load_anny_from_params, build_anny_apose
from clad_body.measure import measure
from clad_body.measure.anny import (
    BREAST_BONES,
    build_arm_mask,
    load_phenotype_params,
)

TESTDATA_DIR = os.path.join(CLAD_BODY_ROOT, "clad_body", "measure", "testdata", "anny")

ALL_SUBJECTS = [
    "male_average",
    "female_average",
    "male_plus_size",
    "female_curvy",
    "female_slim",
    "female_plus_size",
]


# ── Soft circumference implementation ──


def _edges_from_faces(faces_np):
    """Extract unique edges from face array."""
    edges_set = set()
    for f in faces_np:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges_set.add((min(a, b), max(a, b)))
    return torch.tensor(list(edges_set), dtype=torch.long)


def build_torso_edge_indices(model, faces_tensor):
    """Build edge index tensor for the torso-only mesh.

    Returns:
        torso_vertex_mask: (V,) bool numpy array — True for torso vertices
        edge_indices: (E, 2) int tensor — edges of torso-only faces
    """
    arm_mask = build_arm_mask(model)
    faces_np = faces_tensor.detach().cpu().numpy() if hasattr(faces_tensor, 'detach') else np.array(faces_tensor)

    # Exclude faces with any arm vertex
    face_has_arm = arm_mask[faces_np].any(axis=1)
    torso_faces = faces_np[~face_has_arm]

    edge_indices = _edges_from_faces(torso_faces)
    return ~arm_mask, edge_indices


def build_ribcage_edge_indices(model, faces_tensor):
    """Build edge indices for the ribcage (torso minus breast tissue).

    For underbust measurement: at the inframammary fold, breast tissue hangs below
    the ribcage, creating outlier points that inflate the circumference. Excluding
    breast-dominated faces gives a clean ribcage contour.

    Returns:
        edge_indices: (E, 2) int tensor
    """
    arm_mask = build_arm_mask(model)
    faces_np = faces_tensor.detach().cpu().numpy() if hasattr(faces_tensor, 'detach') else np.array(faces_tensor)

    # Breast bone skinning weight per vertex
    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    breast_weight = np.zeros(vbw.shape[0])
    for bone_idx in BREAST_BONES:
        mask = (vbi == bone_idx)
        breast_weight += np.where(mask, vbw, 0).sum(axis=1)

    # A vertex is "breast-dominated" if breast bones are its primary driver
    breast_heavy = breast_weight > 0.3

    # Exclude faces with arm OR breast-dominated vertices
    exclude_mask = arm_mask | breast_heavy
    face_has_exclude = exclude_mask[faces_np].any(axis=1)
    ribcage_faces = faces_np[~face_has_exclude]

    return _edges_from_faces(ribcage_faces)


def soft_circumference(verts, edge_indices, z, n_bins=72, sigma_z=0.005,
                       tau=0.003, method="softmax"):
    """Differentiable circumference at height z using soft edge-plane intersection.

    Args:
        verts: (1, V, 3) torch tensor — full mesh vertices (Z-up, metres)
        edge_indices: (E, 2) long tensor — edges to consider (torso-only)
        z: scalar torch tensor or float — cutting plane height
        n_bins: number of angular bins (default 72 = 5° resolution)
        sigma_z: Gaussian sigma for soft edge crossing (metres)
        tau: temperature for soft crossing gate and soft-max radius
        method: "softmax" (convex hull proxy) or "weighted_avg" (mean radius)

    Returns:
        circ: scalar tensor — circumference in metres (differentiable)
        min_mass: float — minimum bin mass (diagnostic)
    """
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=verts.dtype, device=verts.device)

    v = verts[0]  # (V, 3)

    # Edge endpoint coordinates
    va = v[edge_indices[:, 0]]  # (E, 3)
    vb = v[edge_indices[:, 1]]  # (E, 3)

    za = va[:, 2]  # (E,)
    zb = vb[:, 2]  # (E,)

    # Interpolation parameter: where the plane crosses each edge
    dz = zb - za
    t = (z - za) / (dz + 1e-10)  # (E,)

    # Soft crossing gate: sigmoid product — peaks when 0 < t < 1
    w = torch.sigmoid(t / tau) * torch.sigmoid((1.0 - t) / tau)  # (E,)

    # Down-weight near-tangent edges (small |dz| relative to sigma_z)
    dz_weight = torch.sigmoid(torch.abs(dz) / sigma_z - 1.0)
    w = w * dz_weight

    # Exact intersection points (xy) — same math as MeshSlicer
    t_clamped = t.clamp(0, 1)
    px = va[:, 0] + t_clamped * (vb[:, 0] - va[:, 0])  # (E,)
    py = va[:, 1] + t_clamped * (vb[:, 1] - va[:, 1])  # (E,)

    # Fixed center at origin (body is XY-centred by load_anny_from_params)
    dx = px
    dy = py
    r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-10)  # (E,)
    theta = torch.atan2(dy, dx)  # (E,) — [-π, π]

    # Angular bins
    bin_centers = torch.linspace(-np.pi, np.pi * (1 - 2.0 / n_bins), n_bins,
                                 device=verts.device, dtype=verts.dtype)
    bin_width = 2 * np.pi / n_bins
    sigma_theta = bin_width * 0.6

    # Angular affinity (circular distance)
    ang_diff = theta.unsqueeze(-1) - bin_centers.unsqueeze(0)  # (E, N)
    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))
    ang_affinity = torch.exp(-ang_diff ** 2 / (2 * sigma_theta ** 2))  # (E, N)

    # Combined weight: crosses the plane AND near this angular bin
    combined_w = w.unsqueeze(-1) * ang_affinity  # (E, N)

    r_expanded = r.unsqueeze(-1)  # (E, 1)

    if method == "softmax":
        # Soft-max radius per bin: selects outermost crossing point (convex hull proxy).
        # Use per-bin softmax over crossing-weighted radii. The temperature `softmax_T`
        # is in the same units as r (metres). A good value is ~2mm: two points 2mm apart
        # in radius get ~exp(1) weight ratio, so the outer one strongly dominates.
        softmax_T = tau  # reuse tau as temperature in metres
        # Only compute softmax over edges with meaningful crossing weight
        # Mask out negligible contributions to avoid noise from distant edges
        masked_w = combined_w * (w.unsqueeze(-1) > 0.01).float()
        scores = r_expanded / softmax_T  # (E, 1)
        log_w = torch.log(masked_w + 1e-30) + scores  # (E, N)
        log_max = log_w.max(dim=0, keepdim=True).values
        exp_w = torch.exp(log_w - log_max)
        # Zero out non-crossing edges in the exp weights
        exp_w = exp_w * (masked_w > 1e-20).float()
        r_bin = (r_expanded * exp_w).sum(dim=0) / (exp_w.sum(dim=0) + 1e-10)
    else:
        # Weighted average: simpler, less biased
        r_bin = (r_expanded * combined_w).sum(dim=0) / (combined_w.sum(dim=0) + 1e-10)

    # Diagnostic: minimum bin mass
    bin_mass = combined_w.sum(dim=0)
    min_mass = bin_mass.min().item()

    # Convert bin radii to Cartesian and compute perimeter
    bx = r_bin * torch.cos(bin_centers)
    by = r_bin * torch.sin(bin_centers)
    points = torch.stack([bx, by], dim=-1)  # (N, 2)
    edges_vec = torch.roll(points, -1, dims=0) - points
    circ = torch.sum(torch.linalg.norm(edges_vec, dim=-1))

    return circ, min_mass


def bust_z_differentiable(model, verts):
    """Get bust prominence Z as a differentiable torch scalar."""
    tails = getattr(model, '_last_bone_tails', None)
    heads = getattr(model, '_last_bone_heads', None)
    if tails is None or heads is None:
        return None

    tails_t = tails[0]  # (J, 3) torch tensor
    heads_t = heads[0]

    labels = model.bone_labels
    try:
        bl = labels.index("breast.L")
        br = labels.index("breast.R")
    except ValueError:
        return None

    with torch.no_grad():
        all_pts = torch.cat([heads_t, tails_t], dim=0)
        extents = all_pts.max(0).values - all_pts.min(0).values
        height_axis = int(extents.argmax().item())

    all_pts_diff = torch.cat([heads_t, tails_t], dim=0)
    raw_min = all_pts_diff[:, height_axis].min()
    raw_range = all_pts_diff[:, height_axis].max() - raw_min

    frac_l = (tails_t[bl, height_axis] - raw_min) / (raw_range + 1e-10)
    frac_r = (tails_t[br, height_axis] - raw_min) / (raw_range + 1e-10)

    mesh_height = verts[0, :, 2].max()
    return (frac_l + frac_r) / 2 * mesh_height


def underbust_z_differentiable(model, verts):
    """Get underbust Z as a differentiable torch scalar."""
    vbw = model.vertex_bone_weights.detach().cpu().numpy()
    vbi = model.vertex_bone_indices.detach().cpu().numpy()
    breast_weight = np.zeros(vbw.shape[0])
    for bone_idx in BREAST_BONES:
        mask = (vbi == bone_idx)
        breast_weight += np.where(mask, vbw, 0).sum(axis=1)
    breast_vert_mask = breast_weight > 0.3

    if not breast_vert_mask.any():
        return None

    breast_indices = np.where(breast_vert_mask)[0]
    breast_indices_t = torch.tensor(breast_indices, dtype=torch.long, device=verts.device)

    breast_z = verts[0, breast_indices_t, 2]
    return breast_z.min()


def _apply_coordinate_transform(verts_tensor, model):
    """Apply the same Z-up + floor-align transform as _anny_to_trimesh, in torch."""
    v = verts_tensor[0].clone()  # (V, 3)

    with torch.no_grad():
        extents = v.max(0).values - v.min(0).values
        height_axis = int(extents.argmax().item())

    if height_axis == 1:  # Y-up → Z-up
        v = v[:, [0, 2, 1]]
        v = v.clone()
        v[:, 2] = -v[:, 2]

    v = v - v[:, 2].min() * torch.tensor([0, 0, 1], dtype=v.dtype, device=v.device)

    return v.unsqueeze(0), height_axis


# ── Main experiment ──

def run_experiment():
    print("=" * 78)
    print("SHAPY-lite Soft Circumference vs Plane Sweep — Bust & Underbust")
    print("=" * 78)

    # Focus on the best config from previous sweep
    configs = [
        {"n_bins": 72, "sigma_z": 0.005, "tau": 0.100, "method": "softmax", "label": "n72_t100"},
    ]

    results = {c["label"]: {
        "bust_errors": [], "underbust_errors": [],
        "bust_raw": [], "underbust_raw": [],
        "bust_ref": [], "underbust_ref": [],
    } for c in configs}

    for subject in ALL_SUBJECTS:
        print(f"\n{'─' * 60}")
        print(f"Subject: {subject}")
        print(f"{'─' * 60}")

        # Load body
        params_path = os.path.join(TESTDATA_DIR, subject, "anny_params.json")
        params = load_phenotype_params(params_path)
        body = load_anny_from_params(params, requires_grad=True)
        model = body.model

        # Reference: plane sweep measurement
        ref = measure(body, only=["bust_cm", "underbust_cm"])
        ref_bust = ref["bust_cm"]
        ref_underbust = ref["underbust_cm"]
        ref_bust_z = ref.get("_bust_z", 0)
        ref_underbust_z = ref.get("_underbust_z", 0)
        print(f"  Plane sweep:  bust={ref_bust:.2f} cm  underbust={ref_underbust:.2f} cm")

        # Forward pass with gradients
        pose = build_anny_apose(model, "cpu")
        output = model(
            pose_parameters=pose,
            phenotype_kwargs=body.phenotype_kwargs,
            local_changes_kwargs=body.local_changes_kwargs or {},
            pose_parameterization="root_relative_world",
            return_bone_ends=True,
        )
        model._last_bone_heads = output["bone_heads"]
        model._last_bone_tails = output["bone_tails"]
        raw_verts = output["vertices"]

        # Z-up transform
        verts_zup, _ = _apply_coordinate_transform(raw_verts, model)

        # Build edge sets (once per model)
        _, torso_edges = build_torso_edge_indices(model, model.faces)
        ribcage_edges = build_ribcage_edge_indices(model, model.faces)
        print(f"  Edges:        torso={len(torso_edges)}  ribcage={len(ribcage_edges)}")

        # Z heights
        bust_z_t = bust_z_differentiable(model, verts_zup)
        underbust_z_t = underbust_z_differentiable(model, verts_zup)

        if bust_z_t is not None:
            print(f"  Bust Z:       diff={bust_z_t.item()*100:.2f} cm  ref={ref_bust_z*100:.2f} cm")
        if underbust_z_t is not None:
            print(f"  Underbust Z:  diff={underbust_z_t.item()*100:.2f} cm  ref={ref_underbust_z*100:.2f} cm")

        # Test each config
        for cfg in configs:
            label = cfg["label"]

            if bust_z_t is not None and ref_bust > 0:
                sc, _ = soft_circumference(
                    verts_zup, torso_edges, bust_z_t.detach(),
                    n_bins=cfg["n_bins"], sigma_z=cfg["sigma_z"],
                    tau=cfg["tau"], method=cfg["method"])
                sc_cm = sc.item() * 100
                results[label]["bust_raw"].append(sc_cm)
                results[label]["bust_ref"].append(ref_bust)
                results[label]["bust_errors"].append(sc_cm - ref_bust)

            if underbust_z_t is not None and ref_underbust > 0:
                sc, _ = soft_circumference(
                    verts_zup, torso_edges, underbust_z_t.detach(),
                    n_bins=cfg["n_bins"], sigma_z=cfg["sigma_z"],
                    tau=cfg["tau"], method=cfg["method"])
                sc_cm = sc.item() * 100
                results[label]["underbust_raw"].append(sc_cm)
                results[label]["underbust_ref"].append(ref_underbust)
                results[label]["underbust_errors"].append(sc_cm - ref_underbust)

        # Gradient flow test (first config)
        if bust_z_t is not None:
            sc_grad, _ = soft_circumference(
                verts_zup, torso_edges, bust_z_t,
                n_bins=72, sigma_z=0.005, tau=0.003, method="weighted_avg")
            loss = sc_grad * 100
            try:
                loss.backward(retain_graph=True)
                grads = any(
                    t.grad is not None and t.grad.abs().sum().item() > 0
                    for t in body.phenotype_kwargs.values()
                )
                print(f"  Gradient flow: {'YES' if grads else 'NO'}")
                for t in body.phenotype_kwargs.values():
                    if t.grad is not None:
                        t.grad.zero_()
            except Exception as e:
                print(f"  Gradient flow: FAILED ({e})")

    # ── Summary: raw errors ──
    cfg = configs[0]
    label = cfg["label"]
    bust_raw = np.array(results[label]["bust_raw"])
    underbust_raw = np.array(results[label]["underbust_raw"])
    bust_ref = np.array(results[label]["bust_ref"])
    underbust_ref = np.array(results[label]["underbust_ref"])

    print(f"\n{'=' * 78}")
    print(f"Config: {label}  (n_bins={cfg['n_bins']}, sigma_z={cfg['sigma_z']}m, tau={cfg['tau']}m)")
    print(f"{'=' * 78}")

    # Raw errors
    bust_err_raw = bust_raw - bust_ref
    ub_err_raw = underbust_raw - underbust_ref

    print(f"\nRAW (no correction):")
    print(f"  Bust:      MAE={np.mean(np.abs(bust_err_raw)):.2f}  max={np.max(np.abs(bust_err_raw)):.2f}  bias={np.mean(bust_err_raw):+.2f} cm")
    print(f"  Underbust: MAE={np.mean(np.abs(ub_err_raw)):.2f}  max={np.max(np.abs(ub_err_raw)):.2f}  bias={np.mean(ub_err_raw):+.2f} cm")

    # ── Linear correction: fit y = a*x + b on leave-one-out ──
    # LOOCV: train on N-1 bodies, test on the held-out one — avoids overfitting
    print(f"\nLINEAR CORRECTION (leave-one-out cross-validation):")
    print(f"  y_corrected = a * y_raw + b")

    def fit_and_correct_loocv(raw, ref):
        """Fit linear correction y=ax+b with LOOCV."""
        n = len(raw)
        corrected = np.zeros(n)
        coeffs = []
        for i in range(n):
            train_raw = np.delete(raw, i)
            train_ref = np.delete(ref, i)
            # Least squares: solve [[x 1]] [a, b]^T = ref
            A = np.column_stack([train_raw, np.ones(n - 1)])
            ab, _, _, _ = np.linalg.lstsq(A, train_ref, rcond=None)
            a, b = ab
            coeffs.append((a, b))
            corrected[i] = a * raw[i] + b
        return corrected, coeffs

    bust_corr, bust_coeffs = fit_and_correct_loocv(bust_raw, bust_ref)
    ub_corr, ub_coeffs = fit_and_correct_loocv(underbust_raw, underbust_ref)

    bust_err_corr = bust_corr - bust_ref
    ub_err_corr = ub_corr - underbust_ref

    print(f"  Bust:      MAE={np.mean(np.abs(bust_err_corr)):.2f}  max={np.max(np.abs(bust_err_corr)):.2f}  bias={np.mean(bust_err_corr):+.2f} cm")
    print(f"  Underbust: MAE={np.mean(np.abs(ub_err_corr)):.2f}  max={np.max(np.abs(ub_err_corr)):.2f}  bias={np.mean(ub_err_corr):+.2f} cm")

    # ── Global linear correction (all bodies) ──
    def fit_global(raw, ref):
        A = np.column_stack([raw, np.ones(len(raw))])
        ab, _, _, _ = np.linalg.lstsq(A, ref, rcond=None)
        return ab[0], ab[1]

    bust_a, bust_b = fit_global(bust_raw, bust_ref)
    ub_a, ub_b = fit_global(underbust_raw, underbust_ref)
    print(f"\nGLOBAL fit (all 6 bodies — for reference):")
    print(f"  Bust:      a={bust_a:.4f}  b={bust_b:+.4f}  →  y = {bust_a:.4f}*x + {bust_b:+.4f}")
    print(f"  Underbust: a={ub_a:.4f}  b={ub_b:+.4f}  →  y = {ub_a:.4f}*x + {ub_b:+.4f}")

    # ── Per-subject breakdown ──
    print(f"\nPER-SUBJECT:")
    print(f"  {'Subject':<22s}  {'Bust raw':>9s}  {'Bust corr':>9s}  │  {'UB raw':>9s}  {'UB corr':>9s}  │  Ref bust  Ref ub")
    print(f"  {'─' * 22}  {'─' * 9}  {'─' * 9}  │  {'─' * 9}  {'─' * 9}  │  {'─' * 9}  {'─' * 6}")
    for i, subj in enumerate(ALL_SUBJECTS):
        print(
            f"  {subj:<22s}  {bust_err_raw[i]:+9.2f}  {bust_err_corr[i]:+9.2f}  │"
            f"  {ub_err_raw[i]:+9.2f}  {ub_err_corr[i]:+9.2f}  │"
            f"  {bust_ref[i]:8.2f}  {underbust_ref[i]:6.2f}"
        )


def _load_dataset_entries(n=100, seed=42):
    """Load n random entries from data_10k_42/train.json."""
    import json
    import random

    data_path = os.path.join(
        CLAD_BODY_ROOT, "..", "body-tuning", "questionnaire",
        "data_10k_42", "train.json"
    )
    data_path = os.path.normpath(data_path)
    with open(data_path) as f:
        all_entries = json.load(f)
    rng = random.Random(seed)
    return rng.sample(all_entries, min(n, len(all_entries)))


def validate_on_dataset(n=100):
    """Validate soft circumference on n bodies from data_10k_42.

    Applies the global linear correction from the 6-body testdata calibration:
        bust      = 0.9450 * raw + 3.2416
        underbust = 1.0244 * raw - 5.1162
    Compares against the plane-sweep reference already stored in the dataset.
    """
    # These are the global-fit coefficients from the 6 testdata bodies
    BUST_A, BUST_B = 0.9450, 3.2416
    UB_A, UB_B = 1.0244, -5.1162

    print("\n" + "=" * 78)
    print(f"Dataset Validation: {n} bodies from data_10k_42/train.json")
    print("=" * 78)

    entries = _load_dataset_entries(n=n)

    bust_raw_all, bust_ref_all = [], []
    ub_raw_all, ub_ref_all = [], []
    failed = 0

    # We'll cache the model and torso edges after the first body
    # (all dataset bodies use the same local_changes keys → same model)
    shared_model = None
    shared_torso_edges = None
    shared_breast_indices = None  # fixed topology

    for i, entry in enumerate(entries):
        params = entry["params"]
        meas_ref = entry["measurements"]
        ref_bust = meas_ref.get("bust_cm")
        ref_ub = meas_ref.get("underbust_cm")

        if ref_bust is None or ref_ub is None:
            failed += 1
            continue

        try:
            body = load_anny_from_params(params, requires_grad=False)

            if shared_model is None:
                shared_model = body.model
                _, shared_torso_edges = build_torso_edge_indices(
                    shared_model, shared_model.faces
                )
                # Pre-compute breast vertex indices (fixed topology)
                vbw = shared_model.vertex_bone_weights.detach().cpu().numpy()
                vbi = shared_model.vertex_bone_indices.detach().cpu().numpy()
                breast_weight = np.zeros(vbw.shape[0])
                for bone_idx in BREAST_BONES:
                    mask = (vbi == bone_idx)
                    breast_weight += np.where(mask, vbw, 0).sum(axis=1)
                breast_mask = breast_weight > 0.3
                shared_breast_indices = torch.tensor(
                    np.where(breast_mask)[0], dtype=torch.long
                )
                print(f"  Model cached. Torso edges: {len(shared_torso_edges)}")
            else:
                body._model = shared_model

            pose = build_anny_apose(shared_model, "cpu")
            with torch.no_grad():
                output = shared_model(
                    pose_parameters=pose,
                    phenotype_kwargs=body.phenotype_kwargs,
                    local_changes_kwargs=body.local_changes_kwargs or {},
                    pose_parameterization="root_relative_world",
                    return_bone_ends=True,
                )
                shared_model._last_bone_heads = output["bone_heads"]
                shared_model._last_bone_tails = output["bone_tails"]
                raw_verts = output["vertices"]

                verts_zup, _ = _apply_coordinate_transform(raw_verts, shared_model)

                bust_z_t = bust_z_differentiable(shared_model, verts_zup)
                # Use cached breast indices for underbust Z
                breast_z = verts_zup[0, shared_breast_indices, 2]
                underbust_z_t = breast_z.min()

            if bust_z_t is not None:
                sc_bust, _ = soft_circumference(
                    verts_zup, shared_torso_edges, bust_z_t,
                    n_bins=72, sigma_z=0.005, tau=0.100, method="softmax"
                )
                bust_raw_all.append(sc_bust.item() * 100)
                bust_ref_all.append(ref_bust)

            sc_ub, _ = soft_circumference(
                verts_zup, shared_torso_edges, underbust_z_t,
                n_bins=72, sigma_z=0.005, tau=0.100, method="softmax"
            )
            ub_raw_all.append(sc_ub.item() * 100)
            ub_ref_all.append(ref_ub)

        except Exception as e:
            failed += 1
            print(f"  [skip] body {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n} bodies...")

    bust_raw = np.array(bust_raw_all)
    bust_ref = np.array(bust_ref_all)
    ub_raw = np.array(ub_raw_all)
    ub_ref = np.array(ub_ref_all)

    print(f"\n  Bodies processed: {len(bust_raw)}  (failed/skipped: {failed})")

    # ── Raw errors ──
    bust_err_raw = bust_raw - bust_ref
    ub_err_raw = ub_raw - ub_ref

    print(f"\nRAW (no correction):")
    print(f"  Bust:      MAE={np.mean(np.abs(bust_err_raw)):.2f}  "
          f"max={np.max(np.abs(bust_err_raw)):.2f}  bias={np.mean(bust_err_raw):+.2f} cm")
    print(f"  Underbust: MAE={np.mean(np.abs(ub_err_raw)):.2f}  "
          f"max={np.max(np.abs(ub_err_raw)):.2f}  bias={np.mean(ub_err_raw):+.2f} cm")

    # ── Apply 6-body calibration correction ──
    bust_corr_6 = BUST_A * bust_raw + BUST_B
    ub_corr_6 = UB_A * ub_raw + UB_B
    bust_err_corr_6 = bust_corr_6 - bust_ref
    ub_err_corr_6 = ub_corr_6 - ub_ref

    print(f"\nWITH 6-body calibration (a={BUST_A}, b={BUST_B:+.4f} / a={UB_A}, b={UB_B:+.4f}):")
    print(f"  Bust:      MAE={np.mean(np.abs(bust_err_corr_6)):.2f}  "
          f"max={np.max(np.abs(bust_err_corr_6)):.2f}  bias={np.mean(bust_err_corr_6):+.2f} cm")
    print(f"  Underbust: MAE={np.mean(np.abs(ub_err_corr_6)):.2f}  "
          f"max={np.max(np.abs(ub_err_corr_6)):.2f}  bias={np.mean(ub_err_corr_6):+.2f} cm")

    # ── In-sample fit on the 100 bodies (upper bound) ──
    def fit_global(raw, ref):
        A = np.column_stack([raw, np.ones(len(raw))])
        ab, _, _, _ = np.linalg.lstsq(A, ref, rcond=None)
        return ab[0], ab[1]

    bust_a100, bust_b100 = fit_global(bust_raw, bust_ref)
    ub_a100, ub_b100 = fit_global(ub_raw, ub_ref)
    bust_corr_100 = bust_a100 * bust_raw + bust_b100
    ub_corr_100 = ub_a100 * ub_raw + ub_b100
    bust_err_corr_100 = bust_corr_100 - bust_ref
    ub_err_corr_100 = ub_corr_100 - ub_ref

    print(f"\nIN-SAMPLE fit on {len(bust_raw)} bodies (upper bound):")
    print(f"  Bust:      a={bust_a100:.4f}  b={bust_b100:+.4f}  "
          f"MAE={np.mean(np.abs(bust_err_corr_100)):.2f}  max={np.max(np.abs(bust_err_corr_100)):.2f} cm")
    print(f"  Underbust: a={ub_a100:.4f}  b={ub_b100:+.4f}  "
          f"MAE={np.mean(np.abs(ub_err_corr_100)):.2f}  max={np.max(np.abs(ub_err_corr_100)):.2f} cm")

    # ── Error distribution ──
    print(f"\nERROR DISTRIBUTION (6-body correction):")
    for label, errs in [("Bust", bust_err_corr_6), ("Underbust", ub_err_corr_6)]:
        p25, p50, p75, p90, p95 = np.percentile(np.abs(errs), [25, 50, 75, 90, 95])
        print(f"  {label:<10s}: p25={p25:.2f}  p50={p50:.2f}  p75={p75:.2f}  p90={p90:.2f}  p95={p95:.2f} cm")


if __name__ == "__main__":
    import sys
    if "--dataset" in sys.argv:
        n = int(sys.argv[sys.argv.index("--dataset") + 1]) if len(sys.argv) > sys.argv.index("--dataset") + 1 else 100
        validate_on_dataset(n=n)
    else:
        run_experiment()
        validate_on_dataset(n=100)
