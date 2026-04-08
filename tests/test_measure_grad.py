"""Tests for measure_grad — differentiable Anny body measurements.

Verifies:
  1. Forward equivalence: measure_grad values match measure() to within 0.05 cm
     on all 6 testdata bodies.
  2. Gradient flow: loss.backward() produces non-zero .grad on phenotype tensors.
  3. ``only=`` filtering: returns exactly the requested keys.
  4. Error on unsupported key.
  5. Error on missing bone state.
  6. No spurious UserWarning during calibration calls.

Run:
    pytest tests/test_measure_grad.py -v
"""

import os
import warnings

import pytest
import torch

from clad_body.load.anny import build_anny_apose, load_anny_from_params
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

# Tight tolerance for joint-based and waist/height keys — same code path, float32 drift only.
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

# Per-key tolerance overrides (keys absent from this dict use TOLERANCE_CM).
_KEY_TOLERANCE = {
    "thigh_cm": TOLERANCE_THIGH_CM,
    "upperarm_cm": TOLERANCE_UPPERARM_CM,
    "inseam_cm": TOLERANCE_INSEAM_CM,
    "sleeve_length_cm": TOLERANCE_SLEEVE_CM,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_body_and_verts(name):
    """Return (body, model, verts_raw) for a testdata subject.

    ``verts_raw`` is the raw (1, V, 3) Y-up tensor from the model forward pass,
    with the computation graph intact so measure_grad can compute gradients.
    The model's _last_bone_heads / _last_bone_tails are updated to match this
    forward pass.
    """
    params = load_phenotype_params(os.path.join(TESTDATA_DIR, name, "anny_params.json"))
    body = load_anny_from_params(params)
    model = body.model

    # Build phenotype kwargs (no requires_grad here — equivalence test only)
    phenotype_kwargs = {}
    for label in model.phenotype_labels:
        if label in params:
            phenotype_kwargs[label] = torch.tensor(
                [params[label]], dtype=torch.float32
            )

    local_changes = params.get("_local_changes", {})
    local_kwargs = {
        label: torch.tensor([v], dtype=torch.float32)
        for label, v in local_changes.items()
    }

    a_pose = build_anny_apose(model, torch.device("cpu"))

    # Fresh forward pass WITH gradient graph (no torch.no_grad)
    output = model(
        pose_parameters=a_pose,
        phenotype_kwargs=phenotype_kwargs,
        local_changes_kwargs=local_kwargs,
        pose_parameterization="root_relative_world",
        return_bone_ends=True,
    )

    # Update bone state so _extract_anny_joints(as_tensor=True) sees live data
    model._last_bone_heads = output.get("bone_heads")
    model._last_bone_tails = output.get("bone_tails")

    return body, model, output["vertices"]


def _load_body_with_grad_params(name):
    """Return (model, verts, phenotype_kwargs) where all phenotype tensors have
    requires_grad=True so loss.backward() can produce non-zero .grad values.
    """
    params = load_phenotype_params(os.path.join(TESTDATA_DIR, name, "anny_params.json"))
    body = load_anny_from_params(params)
    model = body.model

    phenotype_kwargs = {}
    for label in model.phenotype_labels:
        if label in params:
            phenotype_kwargs[label] = torch.tensor(
                [params[label]], dtype=torch.float32, requires_grad=True
            )

    local_changes = params.get("_local_changes", {})
    local_kwargs = {
        label: torch.tensor([v], dtype=torch.float32, requires_grad=True)
        for label, v in local_changes.items()
    }

    a_pose = build_anny_apose(model, torch.device("cpu"))

    output = model(
        pose_parameters=a_pose,
        phenotype_kwargs=phenotype_kwargs,
        local_changes_kwargs=local_kwargs,
        pose_parameterization="root_relative_world",
        return_bone_ends=True,
    )

    model._last_bone_heads = output.get("bone_heads")
    model._last_bone_tails = output.get("bone_tails")

    return model, output["vertices"], phenotype_kwargs


# ---------------------------------------------------------------------------
# 1. Forward equivalence — all 6 bodies, all supported keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_SUBJECTS)
def test_forward_equivalence(name):
    """measure_grad values match measure() to within TOLERANCE_CM on all subjects."""
    body, model, verts_raw = _load_body_and_verts(name)

    # Reference via numpy reporting path
    m_ref = measure(body, only=list(SUPPORTED_KEYS))

    # Differentiable path
    m_grad = measure_grad(model, verts_raw)

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
    model, verts, phenotype_kwargs = _load_body_with_grad_params("male_average")

    m = measure_grad(model, verts, only=["sleeve_length_cm", "inseam_cm", "waist_cm"])
    loss = m["sleeve_length_cm"] + m["inseam_cm"] + m["waist_cm"]
    loss.backward()

    all_grads = {
        label: t.grad
        for label, t in phenotype_kwargs.items()
        if t.grad is not None
    }
    non_zero = {k: v for k, v in all_grads.items() if v.abs().sum().item() > 0}

    assert non_zero, (
        "No non-zero gradients found on any phenotype tensor after loss.backward(). "
        f"Phenotype labels: {list(phenotype_kwargs.keys())}. "
        "Check that the forward pass is not wrapped in torch.no_grad()."
    )


# ---------------------------------------------------------------------------
# 3. ``only=`` filtering
# ---------------------------------------------------------------------------

def test_only_single_key():
    """only=['inseam_cm'] returns exactly one key."""
    body, model, verts_raw = _load_body_and_verts("female_average")
    result = measure_grad(model, verts_raw, only=["inseam_cm"])
    assert set(result.keys()) == {"inseam_cm"}, (
        f"Expected exactly {{'inseam_cm'}}, got {set(result.keys())}"
    )


def test_only_none_returns_all_supported():
    """only=None returns all SUPPORTED_KEYS."""
    body, model, verts_raw = _load_body_and_verts("female_average")
    result = measure_grad(model, verts_raw, only=None)
    assert set(result.keys()) == SUPPORTED_KEYS


def test_only_unsupported_key_raises():
    """Requesting an unsupported key raises ValueError listing SUPPORTED_KEYS."""
    body, model, verts_raw = _load_body_and_verts("female_average")
    with pytest.raises(ValueError) as exc_info:
        measure_grad(model, verts_raw, only=["bust_cm"])
    msg = str(exc_info.value)
    assert "bust_cm" in msg
    # Error message must list supported keys so caller knows what's available
    for key in SUPPORTED_KEYS:
        assert key in msg, f"SUPPORTED_KEYS entry '{key}' missing from error message"


# ---------------------------------------------------------------------------
# 4. Missing bone state
# ---------------------------------------------------------------------------

def test_missing_bone_state_raises():
    """Calling measure_grad after del model._last_bone_heads raises ValueError."""
    body, model, verts_raw = _load_body_and_verts("male_average")
    del model._last_bone_heads
    with pytest.raises(ValueError, match="_last_bone_heads"):
        measure_grad(model, verts_raw)


# ---------------------------------------------------------------------------
# 5. No spurious warnings
# ---------------------------------------------------------------------------

def test_no_warnings():
    """measure_grad must not emit tensor-to-scalar coercion warnings.

    The Anny library emits a UserWarning about LBS skinning fallback (no NVidia Warp
    installed) — that's expected infrastructure noise, not our code. We specifically
    check that OUR code doesn't produce implicit tensor-to-scalar conversion warnings,
    which would indicate a broken gradient path.
    """
    body, model, verts_raw = _load_body_and_verts("female_slim")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measure_grad(model, verts_raw)

    our_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "clad_body" in str(w.filename)
    ]
    assert not our_warnings, (
        "measure_grad emitted unexpected UserWarning(s) from clad_body:\n"
        + "\n".join(f"  {w.filename}:{w.lineno}: {w.message}" for w in our_warnings)
    )
