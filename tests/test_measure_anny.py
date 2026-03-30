"""Regression tests for Anny body measurements.

Each test loads a subject from testdata/anny/, runs measure_body(), and asserts
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

from clad_body.measure.anny import measure_body, load_phenotype_params
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
    "hips_cm", "shoulder_width_cm", "thigh_cm", "upperarm_cm", "neck_cm",
    "inseam_cm", "sleeve_length_cm",
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

    measured = measure_body(params, render_path=render_path, title=name)

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
