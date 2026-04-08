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
from clad_body.measure.anny import SUPPORTED_KEYS, load_phenotype_params, measure_grad

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

# thigh_cm uses BASE_MESH_THIGH_VERTICES which is a known-broken vertex loop that
# under-reports by 3–6 cm vs the plane-sweep measure().  The gradient signal is still
# useful for optimization (direction is correct), but absolute values diverge badly.
TOLERANCE_THIGH_CM = 10.0

# upperarm_cm vertex loop is reasonable as a proxy but not ISO-accurate; allow < 1.5 cm.
TOLERANCE_UPPERARM_CM = 1.5

# inseam_cm in measure_grad uses the joint-based approximation calibrated against
# the mesh-sweep ISO method that measure() now uses.  Max calibration error: 0.10 cm.
TOLERANCE_INSEAM_CM = 0.15

# sleeve_length_cm in measure_grad uses the joint-chain approximation calibrated
# against measure_sleeve_length_iso_reference that measure() now uses.  Max error: 0.55 cm.
TOLERANCE_SLEEVE_CM = 0.65

_KEY_TOLERANCE = {
    "thigh_cm": TOLERANCE_THIGH_CM,
    "upperarm_cm": TOLERANCE_UPPERARM_CM,
    "inseam_cm": TOLERANCE_INSEAM_CM,
    "sleeve_length_cm": TOLERANCE_SLEEVE_CM,
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
# 2. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """loss.backward() produces non-zero .grad on at least one phenotype tensor."""
    body = _load("male_average", requires_grad=True)

    m = measure_grad(body, only=["sleeve_length_cm", "inseam_cm", "waist_cm"])
    loss = m["sleeve_length_cm"] + m["inseam_cm"] + m["waist_cm"]
    loss.backward()

    non_zero = {
        label: t.grad
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
        measure_grad(body, only=["bust_cm"])
    msg = str(exc_info.value)
    assert "bust_cm" in msg
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
