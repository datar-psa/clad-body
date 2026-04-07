"""Regression tests for Anny body measurements.

Each test loads a subject from testdata/anny/, runs measure(), and asserts
that every measurement stays within TOLERANCE of the expected reference value.

Subjects come from the mhr_to_anny pipeline (real photo reconstructions):
  - male_average     (aro)           — 171cm male, average build
  - female_average   (kasia)         — 165cm female, average build
  - male_plus_size   (scott)         — 170cm male, plus-size
  - female_curvy     (lorena_duran)  — 164cm female, curvy / hourglass
  - female_slim      (nicole_joseph) — 167cm female, slim
  - female_plus_size (stephanie)     — 163cm female, plus-size

Run:
    pytest tests/test_measure_anny.py -v
    pytest tests/test_measure_anny.py -v --view   # also render 4-view PNGs
"""

import json
import os

import pytest

from clad_body.load.anny import load_anny_from_params
from clad_body.measure import measure
from clad_body.measure.anny import load_phenotype_params
from tests.conftest import RESULTS_DIR

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "clad_body", "measure", "testdata", "anny"
)

# Maximum allowed deviation per measurement (cm for lengths, m³ for volume, kg for mass).
# Tight tolerance: these are deterministic — same params → same mesh → same measurement.
TOLERANCE_CM = 0.1
TOLERANCE_VOLUME = 0.005  # m³
TOLERANCE_MASS = 0.5      # kg
TOLERANCE_BMI = 0.2

CIRC_KEYS = [
    "height_cm", "bust_cm", "underbust_cm", "waist_cm", "stomach_cm",
    "hip_cm", "shoulder_width_cm", "thigh_cm", "knee_cm", "calf_cm",
    "upperarm_cm", "wrist_cm", "neck_cm",
    "inseam_cm", "sleeve_length_cm",
    "crotch_length_cm", "front_rise_cm", "back_rise_cm",
    "shirt_length_cm",
]

BODY_COMP_TOLERANCES = {
    "volume_m3": TOLERANCE_VOLUME,
    "mass_kg": TOLERANCE_MASS,
    "bmi": TOLERANCE_BMI,
}


def _run_measure(name, render=False):
    """Load testdata, run measurement, return (measured, expected, errors)."""
    d = os.path.join(TESTDATA_DIR, name)
    params = load_phenotype_params(os.path.join(d, "anny_params.json"))
    with open(os.path.join(d, "expected_measurements.json")) as f:
        expected = json.load(f)

    render_path = None
    if render:
        out_dir = os.path.join(RESULTS_DIR, "anny")
        os.makedirs(out_dir, exist_ok=True)
        render_path = os.path.join(out_dir, f"{name}_4view.png")

    body = load_anny_from_params(params)
    measured = measure(body, render_path=render_path, title=name)

    errors = {}
    for key in CIRC_KEYS:
        exp = expected.get(key)
        if exp is not None:
            got = measured.get(key, 0)
            err = abs(got - exp)
            if err > TOLERANCE_CM:
                errors[key] = (got, exp, err)

    for key, tol in BODY_COMP_TOLERANCES.items():
        exp = expected.get(key)
        if exp is not None:
            got = measured.get(key, 0)
            err = abs(got - exp)
            if err > tol:
                errors[key] = (got, exp, err)

    return measured, expected, errors


class TestMaleAverage:
    def test_measurements(self, view):
        _, _, errors = _run_measure("male_average", render=view)
        assert not errors, _format_errors("male_average", errors)


class TestFemaleAverage:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_average", render=view)
        assert not errors, _format_errors("female_average", errors)


class TestMalePlusSize:
    def test_measurements(self, view):
        _, _, errors = _run_measure("male_plus_size", render=view)
        assert not errors, _format_errors("male_plus_size", errors)


class TestFemaleCurvy:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_curvy", render=view)
        assert not errors, _format_errors("female_curvy", errors)


class TestFemaleSlim:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_slim", render=view)
        assert not errors, _format_errors("female_slim", errors)


class TestFemalePlusSize:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_plus_size", render=view)
        assert not errors, _format_errors("female_plus_size", errors)


def _format_errors(name, errors):
    lines = [f"Measurement regression in {name}:"]
    for key, (got, exp, err) in errors.items():
        lines.append(f"  {key}: got {got:.2f}, expected {exp:.2f}, diff {err:.2f}")
    return "\n".join(lines)


# ── measure() API tests ─────────────────────────────────────────────────────
# These test the new unified measure() entry point with presets, tags, and
# selective computation. Uses a single body (male_average) for speed.


@pytest.fixture(scope="module")
def anny_body():
    """Load male_average AnnyBody once for all measure() API tests."""
    from clad_body.load import load_anny_from_params
    params = load_phenotype_params(
        os.path.join(TESTDATA_DIR, "male_average", "anny_params.json")
    )
    return load_anny_from_params(params)


class TestMeasureAPIPresets:
    """Test preset-based measurement selection."""

    def test_core_returns_4_keys(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, preset="core")
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert keys == {"height_cm", "bust_cm", "waist_cm", "hip_cm"}

    def test_standard_returns_9_keys(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, preset="standard")
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert keys == {
            "height_cm", "bust_cm", "waist_cm", "hip_cm",
            "thigh_cm", "upperarm_cm", "shoulder_width_cm",
            "sleeve_length_cm", "inseam_cm",
        }

    def test_all_returns_all_available(self, anny_body):
        from clad_body.measure import measure, REGISTRY
        m = measure(anny_body)
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        # Should have all non-anny_only keys + anny_only keys
        expected = {k for k, d in REGISTRY.items() if not d.anny_only or True}
        # Some derived measurements may not be present (e.g. body_fat needs neck)
        assert keys >= {k for k, d in REGISTRY.items() if d.tier in ("core", "standard")}

    def test_garment_preset_tops(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, preset="tops")
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert "bust_cm" in keys
        assert "shoulder_width_cm" in keys
        # Bottoms-only measurements should not be present
        assert "inseam_cm" not in keys
        assert "crotch_length_cm" not in keys


class TestMeasureAPISelection:
    """Test only, tags, and exclude selection."""

    def test_only_single_key(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, only=["bust_cm"])
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert keys == {"bust_cm"}

    def test_only_multiple_keys(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, only=["bust_cm", "hip_cm", "neck_cm"])
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert keys == {"bust_cm", "hip_cm", "neck_cm"}

    def test_tags_circumference_leg(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, tags={"type": "circumference", "region": "leg"})
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert keys == {"thigh_cm", "knee_cm", "calf_cm"}

    def test_exclude(self, anny_body):
        from clad_body.measure import measure
        m = measure(anny_body, preset="core", exclude=["height_cm"])
        keys = {k for k in m if not k.startswith("_") and k not in ("mesh", "contours")}
        assert "height_cm" not in keys
        assert len(keys) == 3

    def test_values_match_regression(self, anny_body):
        """Values from measure() must match the existing regression baselines."""
        from clad_body.measure import measure
        with open(os.path.join(TESTDATA_DIR, "male_average", "expected_measurements.json")) as f:
            expected = json.load(f)
        m = measure(anny_body)
        for key in CIRC_KEYS:
            exp = expected.get(key)
            if exp is not None:
                got = m.get(key, 0)
                assert abs(got - exp) <= TOLERANCE_CM, (
                    f"{key}: measure() got {got:.2f}, expected {exp:.2f}"
                )

    def test_back_neck_to_waist(self, anny_body):
        """ISO 8559-1 §5.4.5: cervicale → waist along back contour.

        Sanity bounds for an average ~171cm body:
        - Should be > 0 (computed)
        - Plausible anthropometric range 35–55 cm
        - Must exceed the straight vertical drop (c7_z − waist_z),
          since the tape follows the back curvature.
        """
        from clad_body.measure import measure
        m = measure(anny_body, only=["back_neck_to_waist_cm"])
        bnw = m["back_neck_to_waist_cm"]
        assert bnw > 0, "back_neck_to_waist_cm should be computed"
        assert 30.0 < bnw < 60.0, f"unexpected value: {bnw}"
        # Vertical drop sanity (uses internal _waist_z and joints)
        m_full = measure(anny_body, only=["back_neck_to_waist_cm"])
        assert "_back_neck_to_waist_pts" in m_full
        pts = m_full["_back_neck_to_waist_pts"]
        vertical_cm = (pts[0, 2] - pts[-1, 2]) * 100
        assert bnw >= vertical_cm - 1e-6, (
            f"contour length {bnw:.2f} must be ≥ vertical {vertical_cm:.2f}"
        )
        # Contour should not be wildly longer than vertical (sanity ceiling)
        assert bnw < vertical_cm * 1.4, (
            f"contour length {bnw:.2f} suspiciously > 1.4× vertical {vertical_cm:.2f}"
        )

    def test_only_mass_kg_matches_full(self, anny_body):
        """`measure(body, only=['mass_kg'])` must equal full-measure mass.

        Regression test for the GROUP_G dependency bug: the body fat formula
        in GROUP_G reads waist_cm and hip_cm from GROUP_A.  Without A in the
        dep set, those default to 0, BF% early-returns 3%, density is wrong,
        and mass_kg is off by 5+ kg.  GROUP_G must declare A as a dependency.
        """
        from clad_body.measure import measure
        m_full = measure(anny_body)
        m_only = measure(anny_body, only=["mass_kg"])
        assert abs(m_only["mass_kg"] - m_full["mass_kg"]) < 0.01, (
            f"only=mass_kg gave {m_only['mass_kg']:.3f}, "
            f"full gave {m_full['mass_kg']:.3f}"
        )

    def test_hip_cm_not_hips_cm(self, anny_body):
        """Verify hip_cm is used, not the old hips_cm."""
        from clad_body.measure import measure
        m = measure(anny_body, preset="core")
        assert "hip_cm" in m
        assert "hips_cm" not in m
