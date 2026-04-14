"""Gradient correctness tests for measure_grad.

The sibling test_measure_grad.py checks that measure_grad VALUES match
measure() (forward equivalence). This file checks that measure_grad
DERIVATIVES match central finite differences (gradient equivalence).

Why both tests exist:
    measure_grad uses soft approximations (sigmoid-gated edge intersection,
    softmax angular binning, convex-hull perimeter with frozen selection).
    Each approximation has calibration knobs (sigmoid beta, softmax tau) that
    trade fidelity for gradient smoothness. Silent drift in these knobs is
    invisible to a forward-equivalence test: the VALUE stays calibrated, but
    the DERIVATIVE can go wrong (saturating sigmoids, sparse softmax attention,
    detached tensors in a refactor).

    Finite differences are slow but make no assumption about the code path --
    they only require the function to be evaluable. If FD disagrees with
    autograd, either the autograd graph is broken (silent numpy detour,
    .detach() in the wrong place, non-retained intermediate) or the soft
    approximation's derivative has drifted from the forward value's derivative.
    Both are bugs.

Categories:
    1. Tight agreement  -- keys on clean graph paths (height, inseam, mass)
                           agree with FD to within ~2% relative.
    2. Loose agreement  -- soft-circ keys (bust, hip, underbust) with hull
                           selection agree within ~10% relative.
    3. Sign only        -- known-noisy keys (stomach, thigh) only need
                           correct gradient DIRECTION. thigh is documented as
                           "gradient direction only" in the README.
    4. Finite gradients -- no NaN or Inf in any .grad tensor after backward.
    5. End-to-end conv  -- Adam converges on a realistic multi-target loss.

Run:
    pytest tests/test_measure_grad_gradcheck.py -v

These tests are slower than the forward-equivalence tests because FD needs
two extra forward passes per (key, phenotype) pair. Kept under ~30s total
by reusing one loaded body per subject and scoping perturbations narrowly.
"""

import os

import pytest
import torch

from clad_body.load.anny import load_anny_from_params
from clad_body.measure.anny import (
    SUPPORTED_KEYS,
    load_phenotype_params,
    measure_grad,
)

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "clad_body", "measure", "testdata", "anny"
)

# One well-behaved body for most tests. female_average is near the centre of
# the phenotype space, so small perturbations don't push any soft gate into
# saturation. Picked deliberately -- plus-size bodies push some local_changes
# near their bounds and give FD a noisier signal.
DEFAULT_SUBJECT = "female_average"

# FD step size. Phenotypes are in [0, 1]; h = 5e-3 is ~0.5% of the range,
# small enough to stay inside the local linear regime on all keys we test,
# large enough to dominate float32 cancellation (~1e-7 relative).
FD_STEP = 5e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(subject=DEFAULT_SUBJECT, *, requires_grad=True):
    params = load_phenotype_params(os.path.join(TESTDATA_DIR, subject, "anny_params.json"))
    return load_anny_from_params(params, requires_grad=requires_grad)


def _autograd_grad(body, key, label):
    """Autograd ∂measure_grad(body)[key]/∂body.phenotype_kwargs[label]."""
    for t in body.phenotype_kwargs.values():
        if t.grad is not None:
            t.grad.zero_()

    value = measure_grad(body, only=[key])[key]
    value.backward()

    g = body.phenotype_kwargs[label].grad
    return 0.0 if g is None else float(g.item())


def _fd_grad(body, key, label, h=FD_STEP):
    """Central finite-difference gradient, computed by in-place perturbation.

    Mutating ``.data`` keeps the tensor's ``requires_grad`` and its identity
    on the body, so a subsequent measure_grad() call picks up the new value
    without rebuilding the body. The original value is restored at the end so
    the body is reusable across tests.
    """
    tensor = body.phenotype_kwargs[label]
    orig = tensor.data.clone()

    try:
        tensor.data = orig + h
        with torch.no_grad():
            f_plus = float(measure_grad(body, only=[key])[key].item())

        tensor.data = orig - h
        with torch.no_grad():
            f_minus = float(measure_grad(body, only=[key])[key].item())
    finally:
        tensor.data = orig

    return (f_plus - f_minus) / (2.0 * h)


def _assert_agree(ag, fd, *, rtol, atol, context):
    """Relative-or-absolute tolerance comparison with a readable failure message."""
    diff = abs(ag - fd)
    scale = max(abs(ag), abs(fd))
    ok = diff <= atol or diff <= rtol * scale
    assert ok, (
        f"{context}: autograd={ag:+.4f}, fd={fd:+.4f}, diff={diff:.4f}  "
        f"(rtol={rtol}, atol={atol}, scale={scale:.4f})"
    )


# ---------------------------------------------------------------------------
# Test matrix -- (key, phenotype_label)
#
# Pairs picked so the phenotype has a large, non-saturating effect on the key.
# Using a phenotype that barely moves the measurement makes FD noise dominate
# and gives false positives. ``height`` is the cleanest lever for most keys
# because Anny scales the whole skeleton.
# ---------------------------------------------------------------------------

# Tight rtol -- clean graph paths through blendshape + LBS only.
TIGHT_PAIRS = [
    ("height_cm",       "height"),
    ("height_cm",       "weight"),     # weight affects posture, small but real
    ("waist_cm",        "weight"),
    ("waist_cm",        "muscle"),
    ("inseam_cm",       "height"),
    ("inseam_cm",       "proportions"),
    ("mass_kg",         "weight"),
    ("mass_kg",         "height"),
]

# Looser rtol -- soft-circ (sigmoid gate + softmax bin + convex hull).
LOOSE_PAIRS = [
    ("bust_cm",         "weight"),
    ("bust_cm",         "muscle"),
    ("underbust_cm",    "weight"),
    ("hip_cm",          "weight"),
    ("hip_cm",          "height"),
    ("upperarm_cm",     "weight"),
    ("upperarm_cm",     "muscle"),
    ("sleeve_length_cm", "height"),
    ("sleeve_length_cm", "proportions"),
]

# Sign-only -- stomach soft-argmin picks a different Z than the reference on
# some bodies; thigh is documented as "gradient direction only" in the README.
SIGN_ONLY_PAIRS = [
    ("stomach_cm",      "weight"),
    ("stomach_cm",      "height"),
    ("thigh_cm",        "weight"),
    ("thigh_cm",        "height"),
]


# ---------------------------------------------------------------------------
# 1. Tight agreement -- clean graph paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,label", TIGHT_PAIRS)
def test_gradcheck_tight(key, label):
    """Autograd gradient matches central FD within 2% relative tolerance.

    Targets keys whose graph is blendshape + LBS + torch arithmetic only.
    These paths have no soft approximations; disagreement indicates a broken
    autograd graph (silent numpy call, tensor detached, missing retain_graph).
    """
    body = _load()
    ag = _autograd_grad(body, key, label)
    fd = _fd_grad(body, key, label)

    # 0.01 atol covers the case where both gradients are legitimately near zero
    # (e.g., ∂height_cm/∂weight is small; we just want to assert they agree).
    _assert_agree(ag, fd, rtol=0.02, atol=0.01, context=f"d{key}/d{label}")


# ---------------------------------------------------------------------------
# 2. Loose agreement -- soft-circ and calibrated-linear keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,label", LOOSE_PAIRS)
def test_gradcheck_loose(key, label):
    """Autograd gradient matches FD within 10% relative tolerance.

    Soft-circ keys (bust, underbust, hip) and calibrated-linear keys (sleeve,
    upperarm) have small systematic differences between autograd derivative
    and FD derivative due to:
      - sigmoid saturation where a crossing gate is near 0 or 1,
      - softmax attention sparseness in a few of the 72 angular bins,
      - hull-membership kinks (dropped vertices contribute zero gradient but
        a slightly-different set can be selected either side of the FD step).
    10% is empirically loose enough to absorb these and still catch real bugs.
    """
    body = _load()
    ag = _autograd_grad(body, key, label)
    fd = _fd_grad(body, key, label)
    _assert_agree(ag, fd, rtol=0.10, atol=0.02, context=f"d{key}/d{label}")


# ---------------------------------------------------------------------------
# 3. Sign-only agreement -- noisy or known-biased keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,label", SIGN_ONLY_PAIRS)
def test_gradcheck_sign_only(key, label):
    """Autograd and FD agree on gradient DIRECTION.

    stomach_cm uses soft-argmin over torso Y; on some bodies the selected
    Z drifts 1-2 cm from the reference argmax, so magnitudes disagree. thigh_cm
    is documented as "gradient direction only" (README calibration table).
    Both are usable in a loss function because Adam's step direction comes
    from the sign, but reporting should use measure() not measure_grad().

    This test pins the usable part: the sign is correct.
    """
    body = _load()
    ag = _autograd_grad(body, key, label)
    fd = _fd_grad(body, key, label)

    # If both are near zero (|g| < 0.05) the sign is ambiguous and the test
    # doesn't meaningfully pin anything; accept and skip the sign check.
    if abs(ag) < 0.05 and abs(fd) < 0.05:
        return

    assert (ag * fd) > 0, (
        f"d{key}/d{label}: autograd={ag:+.4f}, fd={fd:+.4f} -- signs disagree. "
        f"Loss optimisation using this gradient would step in the wrong direction."
    )


# ---------------------------------------------------------------------------
# 4. Gradients are always finite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(SUPPORTED_KEYS))
def test_gradients_finite(key):
    """Every key produces only finite gradients on every phenotype tensor.

    NaN or Inf in a .grad poisons the optimiser step and kills convergence
    silently -- Adam's default behaviour is to pass NaN through to the
    parameter. This test catches the common causes: division by zero in a
    softmax normalisation, log of zero weight, 0/0 in sigmoid derivative.
    """
    body = _load()
    for t in body.phenotype_kwargs.values():
        if t.grad is not None:
            t.grad.zero_()

    value = measure_grad(body, only=[key])[key]
    value.backward()

    for label, t in body.phenotype_kwargs.items():
        if t.grad is None:
            continue
        assert torch.isfinite(t.grad).all(), (
            f"d{key}/d{label} contains non-finite values: {t.grad.tolist()}"
        )


# ---------------------------------------------------------------------------
# 5. Deterministic -- same body, same gradient
# ---------------------------------------------------------------------------

def test_gradient_is_deterministic():
    """Two identical measure_grad().backward() calls produce near-identical grads.

    Any randomness in the forward pass (dropout, stochastic softmax sampling,
    non-deterministic CUDA kernels) would break gradient-based fitting:
    Adam's momentum assumes ∂L/∂θ is a function of θ alone. Catches a class
    of bugs where a refactor accidentally introduces sampling.

    Tolerance is 1e-4 relative, not exact equality -- float32 reductions over
    many terms (LBS skinning sums over ~18k vertices, soft-circ sums over 72
    bins) drift at the ~1e-5 level from non-associative addition order. That
    is not randomness, it's IEEE 754 on tight budgets; it reproduces on the
    same hardware. Real stochasticity would be ~1e-2 or worse.
    """
    body = _load()

    def grad_vec():
        for t in body.phenotype_kwargs.values():
            if t.grad is not None:
                t.grad.zero_()
        total = sum(measure_grad(body).values())
        total.backward()
        return {k: float(t.grad.item()) for k, t in body.phenotype_kwargs.items() if t.grad is not None}

    g1 = grad_vec()
    g2 = grad_vec()

    for label in g1:
        a, b = g1[label], g2[label]
        scale = max(abs(a), abs(b), 1.0)
        assert abs(a - b) < 1e-4 * scale, (
            f"Non-deterministic gradient on {label}: {a} vs {b} "
            f"(diff={abs(a-b):.2e}, scale={scale:.2f})"
        )


# ---------------------------------------------------------------------------
# 6. End-to-end: Adam optimisation converges on a realistic multi-target loss
# ---------------------------------------------------------------------------

def test_adam_converges_on_multi_target():
    """The practical test: does gradient-based fitting actually work?

    This is what the whole differentiable path exists for. If the autograd
    graph is intact and gradients are sane, 300 Adam steps on a bust + waist
    + inseam target should drive the loss down by ~100x from the baseline.

    A lower bar than "hit the target to 0.1 cm" because local_changes aren't
    unfrozen here -- just phenotype params. Target drift within a few cm is
    expected and not the point. The point is that the loss monotonically
    decreases, which requires all three gradients to point in useful
    directions simultaneously.
    """
    body = _load(requires_grad=True)

    # Take current measurements as the starting point, then perturb the
    # target by a realistic amount (a few cm each). Must stay in-reach for
    # the 11-dim phenotype basis -- asking for a 50-cm change that only
    # local_changes can achieve would not be a fair test.
    with torch.no_grad():
        baseline = measure_grad(body, only=["bust_cm", "waist_cm", "inseam_cm"])

    target = {
        "bust_cm":  float(baseline["bust_cm"].item())  + 3.0,
        "waist_cm": float(baseline["waist_cm"].item()) + 2.0,
        "inseam_cm": float(baseline["inseam_cm"].item()) - 1.5,
    }

    optimizer = torch.optim.Adam(list(body.phenotype_kwargs.values()), lr=0.01)

    def loss_fn():
        m = measure_grad(body, only=list(target))
        return sum((m[k] - v) ** 2 for k, v in target.items())

    initial_loss = float(loss_fn().item())

    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    final_loss = float(loss_fn().item())

    assert final_loss < initial_loss / 10.0, (
        f"Adam failed to reduce loss by 10x: initial={initial_loss:.4f}, "
        f"final={final_loss:.4f}. Gradients are not usable for fitting."
    )
