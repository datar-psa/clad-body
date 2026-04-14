"""Tests for measure_grad — differentiable Anny body measurements.

Verifies:
  1. Forward equivalence: measure_grad values match measure() within tolerance
     on all 6 testdata bodies.
  2. Gradient flow: loss.backward() produces non-zero .grad on phenotype tensors.
  3. ``only=`` filtering: returns exactly the requested keys.
  4. Error on unsupported key.
  5. Error when phenotype_kwargs are missing on the body.
  6. No spurious UserWarning from clad_body during the measurement call.

Run:
    pytest tests/test_measure_grad.py -v
"""

import os
import warnings

import pytest

from clad_body.load.anny import AnnyBody, load_anny_from_params
from clad_body.measure import measure
from clad_body.measure.anny import (
    SUPPORTED_KEYS,
    _MEDIAN_DENSITY,
    load_phenotype_params,
    measure_grad,
)

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "clad_body", "measure", "testdata", "anny"
)

ALL_SUBJECTS = [
    "male_average",
    "female_average",
    "male_plus_size",
    "female_curvy",
    "female_slim",
    "female_plus_size",
]

# Tight tolerance for the keys that share a code path with measure() — float32 drift only.
TOLERANCE_CM = 0.05

# bust_cm uses soft circumference (recentered + convex hull).  Calibrated to
# A≈1.0, MAE=0.06 cm on 100 bodies.  The convex hull matches the reference
# plane-sweep + convex-hull in measure() very closely.
TOLERANCE_BUST_CM = 0.5

# underbust_cm uses the same soft circumference.  Calibrated MAE=0.39 cm on
# 100 bodies.  Slightly noisier than bust because underbust z estimation
# (breast vertex min-z) varies more across body types.
TOLERANCE_UNDERBUST_CM = 2.0

# hip_cm uses soft circumference at the mean Z of BASE_MESH_HIP_VERTICES.
# The convex hull bridges the gluteal cleft.  Calibrated on 100-body dataset
# (A=1.0039, B=0.47): MAE 0.46 cm, max 1.39 cm.  Testdata max: 0.96 cm.
TOLERANCE_HIP_CM = 1.5

# thigh_cm uses BASE_MESH_THIGH_VERTICES which is a known-broken vertex loop that
# under-reports by 3–6 cm vs the plane-sweep measure().  The gradient signal is still
# useful for optimization (direction is correct), but absolute values diverge badly.
TOLERANCE_THIGH_CM = 10.0

# upperarm_cm vertex loop is reasonable as a proxy but not ISO-accurate; allow < 1.5 cm.
TOLERANCE_UPPERARM_CM = 1.5

# inseam_cm in measure_grad reads a curated Anny perineum vertex pair (6319/12900)
# whose Z directly tracks the ISO mesh-sweep crotch.  Empirical max error vs the
# mesh sweep across a 118-case stress matrix (testdata + leg-length blendshape
# sweeps + questionnaire grid + random local_changes): 0.187 cm.
TOLERANCE_INSEAM_CM = 0.20

# sleeve_length_cm in measure_grad uses the joint-chain approximation calibrated
# against measure_sleeve_length_iso_reference that measure() now uses.  Max error: 0.55 cm.
TOLERANCE_SLEEVE_CM = 0.65

# stomach_cm uses soft-argmin over torso vertex Y in [hip_z, waist_z] to pick
# the belly Z (the most-anterior torso vertex), then one soft_circumference
# at that Z.  Validated on a 100-body random sample from
# hmr/body-tuning/questionnaire/data_10k_42: MAE 0.93 cm, P95 2.83 cm,
# max 3.61 cm, bias −0.30 cm.  The residual error comes from the soft-argmin
# smoothly averaging over vertex clusters with near-identical anterior Y,
# whereas the non-diff argmax picks a single mesh-topology-specific spike.
TOLERANCE_STOMACH_CM = 4.0

# mass_kg in measure_grad uses V × _MEDIAN_DENSITY[gender] (no BF correction).
# measure() uses V × BF-corrected density (Navy/Weltman → Siri equation).  The
# two diverge most for plus-size bodies (high BF lowers density below the
# population median).  Empirically observed max on testdata: 2.98 kg
# (female_plus_size).  The strict V × _MEDIAN_DENSITY[gender] identity is
# checked separately in test_mass_kg_matches_volume_times_median_density.
TOLERANCE_MASS_KG = 3.1

_KEY_TOLERANCE = {
    "bust_cm": TOLERANCE_BUST_CM,
    "underbust_cm": TOLERANCE_UNDERBUST_CM,
    "hip_cm": TOLERANCE_HIP_CM,
    "stomach_cm": TOLERANCE_STOMACH_CM,
    "thigh_cm": TOLERANCE_THIGH_CM,
    "upperarm_cm": TOLERANCE_UPPERARM_CM,
    "inseam_cm": TOLERANCE_INSEAM_CM,
    "sleeve_length_cm": TOLERANCE_SLEEVE_CM,
    "mass_kg": TOLERANCE_MASS_KG,
}


def _load(name, *, requires_grad=False):
    """Load a testdata body."""
    params = load_phenotype_params(os.path.join(TESTDATA_DIR, name, "anny_params.json"))
    return load_anny_from_params(params, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# 1. Forward equivalence — all 6 bodies, all supported keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_SUBJECTS)
def test_forward_equivalence(name):
    """measure_grad values match measure() within tolerance on all subjects."""
    body = _load(name)

    m_ref = measure(body, only=list(SUPPORTED_KEYS))
    m_grad = measure_grad(body)

    errors = {}
    for key in SUPPORTED_KEYS:
        ref_val = m_ref.get(key)
        if ref_val is None:
            continue
        got = m_grad[key].item()
        err = abs(got - ref_val)
        tol = _KEY_TOLERANCE.get(key, TOLERANCE_CM)
        if err > tol:
            errors[key] = (got, ref_val, err, tol)

    assert not errors, (
        f"{name}: measure_grad diverges from measure() beyond tolerance:\n"
        + "\n".join(
            f"  {k}: got={v[0]:.4f}, ref={v[1]:.4f}, diff={v[2]:.4f} (tol={v[3]})"
            for k, v in sorted(errors.items())
        )
    )


# ---------------------------------------------------------------------------
# 1a. inseam_cm regression — leg-length blendshape must not break tracking
# ---------------------------------------------------------------------------
#
# Pins the fix for the bone-tail formula drift bug. The earlier
# bone-tail-plus-linear-correction `measure_inseam_from_joints` calibrated on
# the 6 testdata bodies at native shape, and broke catastrophically (>10 cm
# error) when `measure-{upper,lower}leg-height-incr` was pushed away from
# zero — those blendshapes were added to BODY_LOCAL_CHANGES for length tuning
# (see body-tuning/measurements_tuning/findings/anny_length_levers.md), but
# the formula's calibration set never spanned them.  The vertex-pair
# replacement reads the perineum directly so it can't drift like that.

@pytest.mark.parametrize(
    "delta", [0.0, 0.5, 1.0],
    ids=["leg+0.0", "leg+0.5", "leg+1.0"],
)
def test_inseam_tracks_mesh_sweep_under_leg_length_blendshape(delta):
    """measure_grad['inseam_cm'] must track measure()['inseam_cm'] across the
    full range of the `measure-{upper,lower}leg-height-incr` blendshapes."""
    params = load_phenotype_params(
        os.path.join(TESTDATA_DIR, "male_average", "anny_params.json")
    )
    lc = dict(params.get("_local_changes", {}))
    lc["measure-upperleg-height-incr"] = delta
    lc["measure-lowerleg-height-incr"] = delta
    params["_local_changes"] = lc
    body = load_anny_from_params(params)

    ref = measure(body, only=["inseam_cm"])["inseam_cm"]
    grad = measure_grad(body, only=["inseam_cm"])["inseam_cm"].item()

    err = abs(grad - ref)
    assert err < TOLERANCE_INSEAM_CM, (
        f"leg-length δ={delta:+.1f}: grad={grad:.3f} ref={ref:.3f} "
        f"err={err:.3f} cm > tol={TOLERANCE_INSEAM_CM} cm"
    )


# ---------------------------------------------------------------------------
# 1b. mass_kg identity — must match V × _MEDIAN_DENSITY[gender] exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_SUBJECTS)
def test_mass_kg_matches_volume_times_median_density(name):
    """measure_grad['mass_kg'] == measure()['volume_m3'] * _MEDIAN_DENSITY[gender].

    The cross-API equivalence test (test_forward_equivalence) compares against
    measure()'s BF-corrected mass with a generous tolerance.  This test pins
    the actual measure_grad formula: simple V × constant per gender.
    """
    body = _load(name)
    m_ref = measure(body, only=["volume_m3"])
    m_grad = measure_grad(body, only=["mass_kg"])

    gender_str = "female" if "female" in name else "male"
    expected = m_ref["volume_m3"] * _MEDIAN_DENSITY[gender_str]
    got = m_grad["mass_kg"].item()

    assert abs(got - expected) < 0.01, (
        f"{name}: mass_kg={got:.4f}, expected V×ρ={expected:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Every SUPPORTED_KEY produces non-zero .grad on at least one phenotype tensor."""
    body = _load("female_average", requires_grad=True)

    m = measure_grad(body)
    loss = sum(m.values())
    loss.backward()

    non_zero = {
        label
        for label, t in body.phenotype_kwargs.items()
        if t.grad is not None and t.grad.abs().sum().item() > 0
    }
    assert non_zero, (
        "No non-zero gradients on any phenotype tensor after loss.backward(). "
        f"Labels: {list(body.phenotype_kwargs.keys())}"
    )


# ---------------------------------------------------------------------------
# 3. ``only=`` filtering
# ---------------------------------------------------------------------------

def test_only_single_key():
    """only=['inseam_cm'] returns exactly one key."""
    body = _load("female_average")
    result = measure_grad(body, only=["inseam_cm"])
    assert set(result.keys()) == {"inseam_cm"}


def test_only_none_returns_all_supported():
    """only=None returns all SUPPORTED_KEYS."""
    body = _load("female_average")
    result = measure_grad(body, only=None)
    assert set(result.keys()) == SUPPORTED_KEYS


def test_only_unsupported_key_raises():
    """Requesting an unsupported key raises ValueError listing SUPPORTED_KEYS."""
    body = _load("female_average")
    with pytest.raises(ValueError) as exc_info:
        measure_grad(body, only=["shoulder_width_cm"])
    msg = str(exc_info.value)
    assert "shoulder_width_cm" in msg
    for key in SUPPORTED_KEYS:
        assert key in msg, f"SUPPORTED_KEYS entry '{key}' missing from error message"


# ---------------------------------------------------------------------------
# 4. Missing phenotype_kwargs
# ---------------------------------------------------------------------------

def test_missing_phenotype_kwargs_raises():
    """A body without phenotype_kwargs (e.g. hand-built) raises ValueError."""
    import numpy as np
    body = AnnyBody(
        vertices=np.zeros((10, 3), dtype=np.float32),
        faces=np.zeros((1, 3), dtype=np.int32),
        source="test",
    )
    with pytest.raises(ValueError, match="phenotype_kwargs"):
        measure_grad(body)


# ---------------------------------------------------------------------------
# 5. No spurious warnings from clad_body
# ---------------------------------------------------------------------------

def test_no_warnings():
    """measure_grad must not emit warnings from clad_body itself.

    The Anny library emits a UserWarning about LBS skinning fallback (no NVidia Warp
    installed) — expected infrastructure noise. We check that OUR code path doesn't
    produce warnings that would indicate a broken gradient flow.
    """
    body = _load("female_slim")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measure_grad(body)

    our_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "clad_body" in str(w.filename)
    ]
    assert not our_warnings, (
        "measure_grad emitted unexpected UserWarning(s) from clad_body:\n"
        + "\n".join(f"  {w.filename}:{w.lineno}: {w.message}" for w in our_warnings)
    )
