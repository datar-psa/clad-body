"""Regression tests for MHR body measurements.

Each test loads a subject from testdata/mhr/, runs measure_mhr(), and asserts
that every measurement stays within TOLERANCE of the expected reference value.

Subjects come from SAM 3D Body (sam3d) and height-scaled (mhr_only) pipelines:
  - male_average    (aro, scaled 182cm)    — average male
  - male_plus_size  (scott, raw)           — plus-size male
  - female_slim     (nicole_joseph, scaled) — slim female
  - female_average  (kasia, raw)           — average female
  - female_curvy    (lorena_duran, raw)    — curvy female
  - female_plus_size(stephanie, raw)       — plus-size female

Run:
    pytest tests/test_measure_mhr.py -v
    pytest tests/test_measure_mhr.py -v --view   # also render 4-view PNGs
"""

import json
import os

import pytest

from clad_body.load.mhr import load_mhr_from_params
from clad_body.measure.mhr import measure_mhr
from tests.conftest import RESULTS_DIR

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "clad_body", "measure", "testdata", "mhr"
)

# MHR measurement is deterministic (same params → same mesh → same measurement).
TOLERANCE_CM = 0.1

CIRC_KEYS = [
    "height_cm", "bust_cm", "hip_cm", "waist_cm", "stomach_cm",
    "thigh_cm", "knee_cm", "calf_cm", "upperarm_cm", "wrist_cm",
    "inseam_cm", "shoulder_width_cm", "sleeve_length_cm",
    "crotch_length_cm", "front_rise_cm", "back_rise_cm",
    "shirt_length_cm",
]


def _run_measure(name, render=False):
    """Load testdata, run measurement, return (measured, expected, errors)."""
    d = os.path.join(TESTDATA_DIR, name)
    mhr = load_mhr_from_params(os.path.join(d, "mhr_params.json"))
    with open(os.path.join(d, "expected_measurements.json")) as f:
        expected = json.load(f)

    render_path = None
    if render:
        out_dir = os.path.join(RESULTS_DIR, "mhr")
        os.makedirs(out_dir, exist_ok=True)
        render_path = os.path.join(out_dir, f"{name}_4view.png")

    measured = measure_mhr(mhr, render_path=render_path, title=name)

    errors = {}
    for key in CIRC_KEYS:
        exp = expected.get(key)
        if exp is not None:
            got = measured.get(key, 0)
            err = abs(got - exp)
            if err > TOLERANCE_CM:
                errors[key] = (got, exp, err)

    return measured, expected, errors


class TestMalePlusSize:
    def test_measurements(self, view):
        _, _, errors = _run_measure("male_plus_size", render=view)
        assert not errors, _format_errors("male_plus_size", errors)


class TestFemaleSlim:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_slim", render=view)
        assert not errors, _format_errors("female_slim", errors)


class TestFemaleAverage:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_average", render=view)
        assert not errors, _format_errors("female_average", errors)


class TestFemaleCurvy:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_curvy", render=view)
        assert not errors, _format_errors("female_curvy", errors)


class TestFemalePlusSize:
    def test_measurements(self, view):
        _, _, errors = _run_measure("female_plus_size", render=view)
        assert not errors, _format_errors("female_plus_size", errors)


def _format_errors(name, errors):
    lines = [f"Measurement regression in {name}:"]
    for key, (got, exp, err) in errors.items():
        lines.append(f"  {key}: got {got:.2f}, expected {exp:.2f}, diff {err:.2f}")
    return "\n".join(lines)
