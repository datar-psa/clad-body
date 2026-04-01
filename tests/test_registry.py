"""Tests for measurement registry — definitions, tags, presets, and query helpers.

Pure data tests: no body model, no GPU, no heavy deps.
"""

import pytest

from clad_body.measure.registry import (
    CORE,
    ENHANCED,
    FITTED,
    GARMENT_PRESETS,
    PRESETS,
    REGISTRY,
    STANDARD,
    TIER_PRESETS,
    VALID_GARMENTS,
    VALID_REGIONS,
    VALID_STANDARDS,
    VALID_TIERS,
    VALID_TYPES,
    MeasurementDef,
    list_measurements,
    resolve_keys,
)


# ── Registry integrity ──────────────────────────────────────────────────────


class TestRegistryIntegrity:
    """Every entry in REGISTRY must have valid, consistent metadata."""

    def test_registry_not_empty(self):
        assert len(REGISTRY) > 0

    def test_all_entries_are_measurement_defs(self):
        for key, m in REGISTRY.items():
            assert isinstance(m, MeasurementDef), f"{key}: not a MeasurementDef"

    def test_key_matches_dict_key(self):
        for key, m in REGISTRY.items():
            assert m.key == key, f"Dict key {key!r} != MeasurementDef.key {m.key!r}"

    def test_valid_type(self):
        for key, m in REGISTRY.items():
            assert m.type in VALID_TYPES, f"{key}: invalid type {m.type!r}"

    def test_valid_standard(self):
        for key, m in REGISTRY.items():
            assert m.standard in VALID_STANDARDS, f"{key}: invalid standard {m.standard!r}"

    def test_valid_region(self):
        for key, m in REGISTRY.items():
            assert m.region in VALID_REGIONS, f"{key}: invalid region {m.region!r}"

    def test_valid_tier(self):
        for key, m in REGISTRY.items():
            assert m.tier in VALID_TIERS, f"{key}: invalid tier {m.tier!r}"

    def test_valid_garments(self):
        for key, m in REGISTRY.items():
            assert m.garments <= VALID_GARMENTS, (
                f"{key}: invalid garments {m.garments - VALID_GARMENTS}"
            )

    def test_non_empty_name(self):
        for key, m in REGISTRY.items():
            assert m.name, f"{key}: empty name"

    def test_non_empty_description(self):
        for key, m in REGISTRY.items():
            assert m.description, f"{key}: empty description"

    def test_no_duplicate_keys(self):
        # Covered by dict nature, but verify explicitly
        keys = [m.key for m in REGISTRY.values()]
        assert len(keys) == len(set(keys))

    def test_unit_matches_key_suffix(self):
        suffix_to_unit = {"_cm": "cm", "_kg": "kg", "_m3": "m3", "_pct": "pct"}
        for key, m in REGISTRY.items():
            for suffix, expected_unit in suffix_to_unit.items():
                if key.endswith(suffix):
                    assert m.unit == expected_unit, (
                        f"{key}: key ends with {suffix!r} but unit is {m.unit!r}"
                    )
                    break

    def test_iso_ref_consistency(self):
        """ISO standard entries must have iso_ref; tailor/derived must not."""
        for key, m in REGISTRY.items():
            if m.standard == "iso":
                assert m.iso_ref is not None, f"{key}: iso standard but no iso_ref"
            else:
                assert m.iso_ref is None, (
                    f"{key}: standard={m.standard!r} but has iso_ref={m.iso_ref!r}"
                )


# ── Tier presets ─────────────────────────────────────────────────────────────


class TestTierPresets:
    """Tier presets must be cumulative: core ⊂ standard ⊂ enhanced ⊂ fitted."""

    def test_core_has_exactly_4(self):
        assert TIER_PRESETS[CORE] == frozenset({
            "height_cm", "bust_cm", "waist_cm", "hip_cm",
        })

    def test_cumulative_inclusion(self):
        tiers = [CORE, STANDARD, ENHANCED, FITTED]
        for i in range(len(tiers) - 1):
            lower = TIER_PRESETS[tiers[i]]
            upper = TIER_PRESETS[tiers[i + 1]]
            assert lower < upper, (
                f"{tiers[i]} is not a strict subset of {tiers[i + 1]}. "
                f"Missing: {lower - upper}"
            )

    def test_each_tier_adds_at_least_one(self):
        tiers = [CORE, STANDARD, ENHANCED, FITTED]
        for i in range(len(tiers) - 1):
            lower = TIER_PRESETS[tiers[i]]
            upper = TIER_PRESETS[tiers[i + 1]]
            assert len(upper) > len(lower), (
                f"{tiers[i + 1]} adds no keys over {tiers[i]}"
            )

    def test_fitted_equals_all(self):
        assert TIER_PRESETS[FITTED] == PRESETS["all"]

    def test_all_registry_keys_in_fitted(self):
        assert set(REGISTRY.keys()) == set(TIER_PRESETS[FITTED])


# ── Garment presets ──────────────────────────────────────────────────────────


class TestGarmentPresets:
    """Garment presets must match garment tags in the registry."""

    def test_garment_preset_matches_tags(self):
        for garment in VALID_GARMENTS:
            expected = frozenset(
                k for k, m in REGISTRY.items() if garment in m.garments
            )
            assert GARMENT_PRESETS[garment] == expected, (
                f"Preset {garment!r} doesn't match tagged entries"
            )

    def test_underwear_includes_bust_underbust_waist_hip_height(self):
        uw = GARMENT_PRESETS["underwear"]
        for key in ["bust_cm", "underbust_cm", "waist_cm", "hip_cm", "height_cm"]:
            assert key in uw, f"underwear preset missing {key!r}"

    def test_every_non_scalar_belongs_to_at_least_one_garment(self):
        """Measurements with garments=frozenset() should only be scalars."""
        for key, m in REGISTRY.items():
            if not m.garments:
                assert m.standard == "derived" or m.type == "scalar", (
                    f"{key}: no garment tags but not derived/scalar"
                )


# ── list_measurements ────────────────────────────────────────────────────────


class TestListMeasurements:
    def test_no_filters_returns_all(self):
        assert len(list_measurements()) == len(REGISTRY)

    def test_filter_by_type(self):
        circs = list_measurements(type="circumference")
        assert all(m.type == "circumference" for m in circs)
        assert len(circs) > 0

    def test_filter_by_region(self):
        legs = list_measurements(region="leg")
        assert all(m.region == "leg" for m in legs)
        assert len(legs) > 0

    def test_filter_by_garments(self):
        tops = list_measurements(garments="tops")
        assert all("tops" in m.garments for m in tops)
        assert len(tops) > 0

    def test_multiple_filters_and(self):
        leg_circs = list_measurements(type="circumference", region="leg")
        assert all(m.type == "circumference" and m.region == "leg" for m in leg_circs)
        # thigh, knee, calf
        keys = {m.key for m in leg_circs}
        assert keys == {"thigh_cm", "knee_cm", "calf_cm"}

    def test_empty_result_for_impossible_combo(self):
        result = list_measurements(type="scalar", region="neck")
        assert result == []

    def test_unknown_dimension_raises(self):
        with pytest.raises(ValueError, match="Unknown tag dimension"):
            list_measurements(color="red")


# ── resolve_keys ─────────────────────────────────────────────────────────────


class TestResolveKeys:
    def test_default_returns_all(self):
        assert resolve_keys() == PRESETS["all"]

    def test_preset_core(self):
        keys = resolve_keys(preset="core")
        assert keys == TIER_PRESETS[CORE]
        assert len(keys) == 4

    def test_preset_standard(self):
        keys = resolve_keys(preset="standard")
        assert keys == TIER_PRESETS[STANDARD]

    def test_preset_garment(self):
        keys = resolve_keys(preset="tops")
        assert keys == GARMENT_PRESETS["tops"]

    def test_only_explicit_keys(self):
        keys = resolve_keys(only=["bust_cm", "hip_cm"])
        assert keys == frozenset({"bust_cm", "hip_cm"})

    def test_only_takes_precedence_over_preset(self):
        keys = resolve_keys(preset="all", only=["bust_cm"])
        assert keys == frozenset({"bust_cm"})

    def test_tags_filter(self):
        keys = resolve_keys(tags={"type": "circumference", "region": "leg"})
        assert keys == {"thigh_cm", "knee_cm", "calf_cm"}

    def test_exclude_removes_keys(self):
        keys = resolve_keys(preset="core", exclude=["height_cm"])
        assert "height_cm" not in keys
        assert len(keys) == 3

    def test_unknown_key_in_only_raises(self):
        with pytest.raises(ValueError, match="Unknown measurement keys"):
            resolve_keys(only=["nonexistent_cm"])

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_keys(preset="nonexistent")

    def test_unknown_key_in_exclude_raises(self):
        with pytest.raises(ValueError, match="Unknown measurement keys in exclude"):
            resolve_keys(exclude=["nonexistent_cm"])

    def test_unknown_tag_dimension_raises(self):
        with pytest.raises(ValueError, match="Unknown tag dimension"):
            resolve_keys(tags={"color": "red"})
