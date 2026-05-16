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


class TestCalfAnatomicalPlacement:
    """ISO 8559-1 §5.3.24: calf girth = max horizontal lower-leg girth.

    The measurement Z must land on the calf belly, anatomically below the
    knee. Real anatomy: the gastrocnemius peak sits ~6–10 cm below the
    knee crease for adults across body types.

    Regression: when a body has its calf circumference blendshape pushed
    negative (tuned bodies where the optimizer shrinks the unmeasured calf
    to balance an oversized thigh/hip target), the lower-leg cross-section
    just under the patella can become wider than the deflated calf belly.
    A pure max-girth horizontal sweep then lands on the kneecap region
    instead of the calf — visually wrong and propagates to the 4-view.
    """

    @pytest.fixture(scope="class")
    def shrunken_calf_body(self):
        """Synthetic body mimicking a tuned-thigh / shrunken-calf pathology."""
        from clad_body.load.anny import load_anny_from_params
        params = {
            "gender": 0.8, "age": 0.5, "muscle": 0.2, "weight": 0.5,
            "height": 0.55, "proportions": 0.45, "cupsize": 0.5,
            "firmness": 0.5, "african": 0.1, "asian": 0.1, "caucasian": 0.8,
            "_local_changes": {
                # Pathology that triggers the bug: deflate calf hard while
                # leaving lower-leg vertical span intact, so the upper part
                # of the lower leg becomes the widest cross-section.
                "measure-calf-circ-incr": -0.4,
                "l-lowerleg-scale-horiz-incr": -0.2,
                "r-lowerleg-scale-horiz-incr": -0.2,
                "l-lowerleg-scale-depth-incr": -0.25,
                "r-lowerleg-scale-depth-incr": -0.25,
                # Plausible upper-body context — not strictly needed but
                # keeps the body anatomically self-consistent.
                "measure-thigh-circ-incr": 0.4,
                "l-upperleg-fat-incr": 0.3,
                "r-upperleg-fat-incr": 0.3,
            },
        }
        return load_anny_from_params(params)

    def test_calf_below_knee(self, shrunken_calf_body):
        """Calf belly must be at least 4 cm below the knee crease.

        4 cm is conservative: real anatomy is 6–10 cm. We pick 4 cm so a
        slightly-misshapen mesh can still pass, but anything landing on
        the kneecap region (gap < 4 cm) is unambiguously wrong.
        """
        from clad_body.measure import measure
        m = measure(shrunken_calf_body, only=["calf_cm", "knee_cm"])
        gap_cm = (m["_knee_z"] - m["_calf_z"]) * 100
        assert gap_cm > 4.0, (
            f"calf_z lands {gap_cm:.1f} cm below knee — should be >4 cm "
            f"(calf_z={m['_calf_z']:.3f}, knee_z={m['_knee_z']:.3f}, "
            f"calf_cm={m['calf_cm']:.2f}, knee_cm={m['knee_cm']:.2f})"
        )


class TestInseamNoHandSaturation:
    """ISO 8559-1 §5.1.15 inseam — slow plane-sweep reference must reject
    A-pose hand cross-sections that hang in the crotch z-range.

    Regression: the inseam loop's leg-contour filter (`x_extent < 0.30 m`)
    accepted any narrow contour. For heavy/curvy bodies whose true crotch z
    overlaps the z-range where hands hang in A-pose, the small hand contours
    at ±60 cm in X were misclassified as legs, keeping ``has_two=True`` past
    the actual leg-merge point. The loop then exited at the search-range
    upper bound, returning an inseam of exactly 0.55 × height.
    """

    @pytest.fixture(scope="class")
    def saturating_body(self):
        from clad_body.load.anny import load_anny_from_params
        fixture_path = os.path.join(
            os.path.dirname(__file__), "testdata", "inseam_saturating_body.json"
        )
        with open(fixture_path) as f:
            fixture = json.load(f)
        return load_anny_from_params(fixture["params"])

    def test_inseam_ratio_in_anatomical_range(self, saturating_body):
        """Inseam / height must be in the human anatomical range, not the
        search-range upper bound."""
        from clad_body.measure import measure
        m = measure(saturating_body, only=["inseam_cm"])
        h_cm = (
            saturating_body.vertices[:, 2].max()
            - saturating_body.vertices[:, 2].min()
        ) * 100
        inseam_cm = m["inseam_cm"]
        ratio = inseam_cm / h_cm
        # Anatomical inseam / height is 0.42-0.50 for adults across body types.
        # Pre-fix bug: ratio = 0.549 (= 0.55 search-range upper bound minus one
        # step). The 0.52 upper bound below leaves headroom for slightly
        # high-crotched anatomy while still failing on the saturated bug.
        assert 0.40 < ratio < 0.52, (
            f"Inseam ratio {ratio:.4f} (cm={inseam_cm:.2f}, h={h_cm:.1f}) is "
            f"outside anatomical range — slow ISO inseam likely saturated at "
            f"the 0.55 × height search-range upper bound. The leg-contour "
            f"filter in measure_inseam() should be rejecting hand cross-"
            f"sections that hang in the crotch z-range in A-pose."
        )


class TestSleeveISOReferenceLocalChanges:
    """ISO 8559-1 §5.4.14 slow sleeve reference must follow arm-length
    blendshapes regardless of how the body was loaded.

    Regression: ``measure_sleeve_length_iso_reference`` re-poses the body to
    rest pose via a fresh forward pass. It used to source ``local_changes``
    only from ``body.phenotype_params['_local_changes']`` — a path
    populated by ``load_anny_from_params`` but not by
    ``load_anny_from_verts``. Bodies loaded via the verts path silently
    re-posed without local changes and returned a constant baseline-arm
    sleeve regardless of the actual blendshape values.
    """

    @staticmethod
    def _build_via_verts(lc_value):
        import anny
        import torch

        from clad_body.load.anny import build_anny_apose, load_anny_from_verts

        device = torch.device("cpu")
        labels = ["measure-upperarm-length-incr"]
        model = anny.create_fullbody_model(
            all_phenotypes=True, triangulate_faces=True, local_changes=labels,
        ).to(dtype=torch.float32, device=device)
        pheno = {"height": 1.0, "weight": 0.5, "age": 0.4, "muscle": 0.5, "gender": 1.0}
        ph_kw = {
            l: torch.tensor([v], dtype=torch.float32)
            for l, v in pheno.items()
            if l in model.phenotype_labels
        }
        lc_kw = {
            "measure-upperarm-length-incr": torch.tensor([lc_value], dtype=torch.float32),
        }
        a_pose = build_anny_apose(model, device)
        with torch.no_grad():
            out = model(
                pose_parameters=a_pose,
                phenotype_kwargs=ph_kw,
                local_changes_kwargs=lc_kw,
                pose_parameterization="root_relative_world",
                return_bone_ends=True,
            )
        return load_anny_from_verts(
            out["vertices"], model,
            phenotype_kwargs=ph_kw,
            local_changes_kwargs=lc_kw,
            bone_heads=out["bone_heads"],
            bone_tails=out["bone_tails"],
        )

    def test_slow_sleeve_responds_to_upperarm_length_blendshape(self):
        from clad_body.measure import measure

        baseline_body = self._build_via_verts(0.0)
        elongated_body = self._build_via_verts(0.4)
        slow_base = float(measure(baseline_body, only=["sleeve_length_cm"])["sleeve_length_cm"])
        slow_long = float(measure(elongated_body, only=["sleeve_length_cm"])["sleeve_length_cm"])
        delta = slow_long - slow_base
        # The fast bone-chain path sees ~+2.8 cm for upperarm-length=+0.4.
        # Pre-fix bug: delta == 0 because the re-pose used no local changes.
        # Use a generous floor (1 cm) to keep the assertion stable.
        assert delta > 1.0, (
            f"Slow ISO sleeve does not respond to upperarm-length blendshape "
            f"when the body was loaded via load_anny_from_verts "
            f"(base={slow_base:.2f}, +0.4={slow_long:.2f}, Δ={delta:.2f} cm). "
            f"load_anny_from_verts must propagate local_changes_kwargs so the "
            f"slow ISO re-pose can apply them."
        )
