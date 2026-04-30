"""Measurement registry — definitions, tags, presets, and query helpers.

Every body measurement in clad-body is registered here with structured metadata:
type (circumference/length/scalar), standard (ISO/tailor/derived), body region,
tier (core→fitted), garment relevance, and self-measurement description.

Public API::

    from clad_body.measure.registry import REGISTRY, list_measurements, resolve_keys

    REGISTRY["bust_cm"]                             # MeasurementDef(...)
    list_measurements(type="circumference")         # [MeasurementDef, ...]
    resolve_keys(preset="core")                     # {"height_cm", "bust_cm", ...}
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Tag value constants ──────────────────────────────────────────────────────

# MeasurementDef.type
CIRCUMFERENCE = "circumference"
LENGTH = "length"
SCALAR = "scalar"
VALID_TYPES = frozenset({CIRCUMFERENCE, LENGTH, SCALAR})

# MeasurementDef.standard
ISO = "iso"
TAILOR = "tailor"
DERIVED = "derived"
VALID_STANDARDS = frozenset({ISO, TAILOR, DERIVED})

# MeasurementDef.region
NECK = "neck"
TORSO = "torso"
ABDOMEN = "abdomen"
ARM = "arm"
LEG = "leg"
FULL_BODY = "full_body"
VALID_REGIONS = frozenset({NECK, TORSO, ABDOMEN, ARM, LEG, FULL_BODY})

# MeasurementDef.tier (cumulative: core ⊂ standard ⊂ enhanced ⊂ fitted)
CORE = "core"
STANDARD = "standard"
ENHANCED = "enhanced"
FITTED = "fitted"
VALID_TIERS = frozenset({CORE, STANDARD, ENHANCED, FITTED})
_TIER_ORDER = [CORE, STANDARD, ENHANCED, FITTED]

# MeasurementDef.garments
TOPS = "tops"
BOTTOMS = "bottoms"
DRESSES = "dresses"
OUTERWEAR = "outerwear"
UNDERWEAR = "underwear"
VALID_GARMENTS = frozenset({TOPS, BOTTOMS, DRESSES, OUTERWEAR, UNDERWEAR})

# Valid tag dimensions for tags= parameter
VALID_TAG_DIMS = frozenset({"type", "standard", "region", "tier", "garments"})


# ── MeasurementDef ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MeasurementDef:
    """Definition of a single body measurement.

    Attributes:
        key: Measurement dict key, e.g. ``"bust_cm"``.
        name: Human-readable name, e.g. ``"Bust girth"``.
        description: Self-measurement instructions (2-3 sentences) + standard ref.
        iso_ref: ISO 8559-1:2017 clause, e.g. ``"5.3.4"``, or ``None``.
        type: ``"circumference"`` | ``"length"`` | ``"scalar"``.
        standard: ``"iso"`` | ``"tailor"`` | ``"derived"``.
        region: Body region — ``"neck"`` | ``"torso"`` | ``"abdomen"`` | ``"arm"``
                | ``"leg"`` | ``"full_body"``.
        tier: Priority tier — ``"core"`` | ``"standard"`` | ``"enhanced"`` | ``"fitted"``.
        garments: Garment types this measurement is relevant for.
        unit: ``"cm"`` | ``"kg"`` | ``"m3"`` | ``"pct"`` | ``""``.
        needs_joints: Whether skeleton joint data is required.
        anny_only: Whether this measurement is only available on Anny bodies.
    """

    key: str
    name: str
    description: str
    iso_ref: str | None
    type: str
    standard: str
    region: str
    tier: str
    garments: frozenset[str]
    unit: str
    needs_joints: bool
    anny_only: bool


def _g(*tags: str) -> frozenset[str]:
    """Shorthand for frozenset of garment tags."""
    return frozenset(tags)


_ALL_GARMENTS = _g(TOPS, BOTTOMS, DRESSES, OUTERWEAR, UNDERWEAR)

# ── Registry ─────────────────────────────────────────────────────────────────

REGISTRY: dict[str, MeasurementDef] = {}


def _reg(m: MeasurementDef) -> None:
    REGISTRY[m.key] = m


# ── Core tier (4) ────────────────────────────────────────────────────────────

_reg(MeasurementDef(
    key="height_cm",
    name="Height (stature)",
    description=(
        "Vertical distance from the floor to the top of the head. "
        "Stand erect with feet together, looking straight ahead. "
        "ISO 8559-1 §5.1.1."
    ),
    iso_ref="5.1.1",
    type=SCALAR, standard=ISO, region=FULL_BODY, tier=CORE,
    garments=_ALL_GARMENTS, unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="bust_cm",
    name="Bust/chest girth",
    description=(
        "Horizontal circumference at the fullest part of the chest/bust. "
        "Pass the tape under the armpits and across the bust prominence. "
        "Keep the tape level and snug without compressing. "
        "ISO 8559-1 §5.3.4."
    ),
    iso_ref="5.3.4",
    type=CIRCUMFERENCE, standard=ISO, region=TORSO, tier=CORE,
    garments=_g(TOPS, DRESSES, OUTERWEAR, UNDERWEAR), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="waist_cm",
    name="Waist girth",
    description=(
        "Horizontal circumference at the natural waist level, midway between "
        "the lowest rib and the top of the hip bone. Wrap the tape at navel "
        "height, parallel to the floor, snug but not compressing. "
        "ISO 8559-1 §5.3.10."
    ),
    iso_ref="5.3.10",
    type=CIRCUMFERENCE, standard=ISO, region=TORSO, tier=CORE,
    garments=_ALL_GARMENTS, unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="hip_cm",
    name="Hip girth",
    description=(
        "Horizontal circumference at the level of greatest buttock prominence. "
        "Stand with feet together, tape around the widest part of the hips, "
        "parallel to the floor. ISO 8559-1 §5.3.13."
    ),
    iso_ref="5.3.13",
    type=CIRCUMFERENCE, standard=ISO, region=ABDOMEN, tier=CORE,
    garments=_g(BOTTOMS, DRESSES, OUTERWEAR, UNDERWEAR), unit="cm",
    needs_joints=False, anny_only=False,
))

# ── Standard tier (+5 = 9) ──────────────────────────────────────────────────

_reg(MeasurementDef(
    key="thigh_cm",
    name="Thigh girth",
    description=(
        "Horizontal circumference at the fullest part of the upper thigh, "
        "just below the gluteal fold. Stand erect with legs slightly apart. "
        "Measure both legs and use the larger value. ISO 8559-1 §5.3.20."
    ),
    iso_ref="5.3.20",
    type=CIRCUMFERENCE, standard=ISO, region=LEG, tier=STANDARD,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="upperarm_cm",
    name="Upper arm girth",
    description=(
        "Circumference at the fullest part of the upper arm, midway between "
        "shoulder and elbow. Arm hangs relaxed at the side, not flexed. "
        "ISO 8559-1 §5.3.16."
    ),
    iso_ref="5.3.16",
    type=CIRCUMFERENCE, standard=ISO, region=ARM, tier=STANDARD,
    garments=_g(TOPS, OUTERWEAR), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="shoulder_width_cm",
    name="Shoulder width (across back)",
    description=(
        "Distance between the left and right shoulder points (acromion), "
        "measured across the back over the C7 vertebra. The tape follows "
        "the contour of the upper back. ISO 8559-1 §5.4.2."
    ),
    iso_ref="5.4.2",
    type=LENGTH, standard=ISO, region=TORSO, tier=STANDARD,
    garments=_g(TOPS, DRESSES, OUTERWEAR), unit="cm",
    needs_joints=True, anny_only=False,
))

_reg(MeasurementDef(
    key="sleeve_length_cm",
    name="Sleeve length (shoulder to wrist)",
    description=(
        "Distance from the shoulder point (acromion) along the outside of "
        "a slightly bent arm, over the elbow, down to the wrist bone. "
        "ISO 8559-1 §5.7.8."
    ),
    iso_ref="5.7.8",
    type=LENGTH, standard=ISO, region=ARM, tier=STANDARD,
    garments=_g(TOPS, OUTERWEAR), unit="cm",
    needs_joints=True, anny_only=False,
))

_reg(MeasurementDef(
    key="inseam_cm",
    name="Inside leg length",
    description=(
        "Distance from the crotch point (perineum) straight down to the "
        "floor. Stand erect with feet slightly apart. Place the tape firmly "
        "at the highest point of the inner thigh and measure vertically to "
        "the ground. ISO 8559-1 §5.1.15."
    ),
    iso_ref="5.1.15",
    type=LENGTH, standard=ISO, region=LEG, tier=STANDARD,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

# ── Enhanced tier (+8 = 17) ─────────────────────────────────────────────────

_reg(MeasurementDef(
    key="neck_cm",
    name="Neck girth",
    description=(
        "Circumference of the neck just below the Adam's apple (thyroid "
        "cartilage), perpendicular to the neck axis. Keep the tape "
        "comfortably snug. ISO 8559-1 §5.3.2."
    ),
    iso_ref="5.3.2",
    type=CIRCUMFERENCE, standard=ISO, region=NECK, tier=ENHANCED,
    garments=_g(TOPS), unit="cm",
    needs_joints=True, anny_only=False,
))

_reg(MeasurementDef(
    key="underbust_cm",
    name="Under-bust girth",
    description=(
        "Horizontal circumference directly below the breast tissue, at the "
        "inframammary crease. This is the bra band size measurement. "
        "ISO 8559-1 §5.3.6."
    ),
    iso_ref="5.3.6",
    type=CIRCUMFERENCE, standard=ISO, region=TORSO, tier=ENHANCED,
    garments=_g(TOPS, DRESSES, UNDERWEAR), unit="cm",
    needs_joints=False, anny_only=True,
))

_reg(MeasurementDef(
    key="stomach_cm",
    name="Stomach circumference",
    description=(
        "Horizontal circumference at the level of maximum anterior protrusion "
        "of the abdomen, usually at or slightly below the navel. Tape passes "
        "around the body without compressing soft tissue. "
        "Industry standard (not in ISO 8559-1)."
    ),
    iso_ref=None,
    type=CIRCUMFERENCE, standard=TAILOR, region=ABDOMEN, tier=ENHANCED,
    garments=_g(TOPS, BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="mass_kg",
    name="Body mass",
    description=(
        "Total body mass in kilograms. When body fat estimation is available "
        "(requires neck circumference), uses volume × estimated tissue density "
        "(V×ρ, Siri two-component, tissue-only convention: 900 fat / 1100 FFM "
        "kg/m³) for realistic scale weight. Falls back to Anny fixed-density "
        "mass (V×980) otherwise — note that 980 sits between the two literature "
        "conventions (whole-body ~985 with lung air, tissue-only ~1030+ without) "
        "and is empirically calibrated rather than physically derived. "
        "ISO 8559-1 §5.6.1."
    ),
    iso_ref="5.6.1",
    type=SCALAR, standard=ISO, region=FULL_BODY, tier=ENHANCED,
    garments=frozenset(), unit="kg",
    needs_joints=False, anny_only=True,
))

_reg(MeasurementDef(
    key="volume_m3",
    name="Body volume",
    description=(
        "Total body volume in cubic metres, computed from mesh geometry. "
        "Derived from Anny anthropometry model."
    ),
    iso_ref=None,
    type=SCALAR, standard=DERIVED, region=FULL_BODY, tier=ENHANCED,
    garments=frozenset(), unit="m3",
    needs_joints=False, anny_only=True,
))

_reg(MeasurementDef(
    key="bmi",
    name="Body mass index",
    description=(
        "Body mass index: mass (kg) divided by height (m) squared. "
        "Derived from Anny anthropometry model."
    ),
    iso_ref=None,
    type=SCALAR, standard=DERIVED, region=FULL_BODY, tier=ENHANCED,
    garments=frozenset(), unit="",
    needs_joints=False, anny_only=True,
))

_reg(MeasurementDef(
    key="body_fat_pct",
    name="Body fat percentage",
    description=(
        "Estimated body fat percentage using Navy (male) or Weltman (female) "
        "equations from circumference measurements. "
        "Derived from neck, waist, hip, height, and mass."
    ),
    iso_ref=None,
    type=SCALAR, standard=DERIVED, region=FULL_BODY, tier=ENHANCED,
    garments=frozenset(), unit="pct",
    needs_joints=True, anny_only=True,
))

_reg(MeasurementDef(
    key="estimated_density",
    name="Estimated tissue density",
    description=(
        "Estimated whole-body tissue density in kg/m³, derived from body fat "
        "percentage via the Siri equation."
    ),
    iso_ref=None,
    type=SCALAR, standard=DERIVED, region=FULL_BODY, tier=ENHANCED,
    garments=frozenset(), unit="kg/m3",
    needs_joints=True, anny_only=True,
))

_reg(MeasurementDef(
    key="back_neck_to_waist_cm",
    name="Back neck point to waist length",
    description=(
        "Distance from the back neck point (cervicale, C7 vertebra prominens) "
        "down the centre back, following the body contour, to the waist level. "
        "Stand erect; the tape touches the skin and follows the curvature of "
        "the spine. ISO 8559-1 §5.4.5."
    ),
    iso_ref="5.4.5",
    type=LENGTH, standard=ISO, region=TORSO, tier=ENHANCED,
    garments=_g(TOPS, DRESSES, OUTERWEAR), unit="cm",
    needs_joints=True, anny_only=False,
))

_reg(MeasurementDef(
    key="belly_depth_cm",
    name="Belly depth",
    description=(
        "How much the belly protrudes forward compared to the underbust/ "
        "ribcage level. Negative values indicate belly prominence. "
        "Derived from mesh geometry (not a tape measurement)."
    ),
    iso_ref=None,
    type=SCALAR, standard=DERIVED, region=ABDOMEN, tier=ENHANCED,
    garments=_g(TOPS, BOTTOMS), unit="cm",
    needs_joints=False, anny_only=True,
))

# ── Fitted tier (+7 = 24) ───────────────────────────────────────────────────

_reg(MeasurementDef(
    key="knee_cm",
    name="Knee girth",
    description=(
        "Horizontal circumference at the centre of the kneecap. "
        "Bend the knee slightly (about 45 degrees) and wrap the tape "
        "around the centre of the kneecap. ISO 8559-1 §5.3.22."
    ),
    iso_ref="5.3.22",
    type=CIRCUMFERENCE, standard=ISO, region=LEG, tier=FITTED,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="calf_cm",
    name="Calf girth",
    description=(
        "Maximum horizontal circumference of the calf. Stand erect with "
        "legs slightly apart. Wrap the tape around the widest part. "
        "ISO 8559-1 §5.3.24."
    ),
    iso_ref="5.3.24",
    type=CIRCUMFERENCE, standard=ISO, region=LEG, tier=FITTED,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="wrist_cm",
    name="Wrist girth",
    description=(
        "Circumference at the wrist, at the level of the prominent bone "
        "on the little finger side (ulnar styloid). "
        "ISO 8559-1 §5.3.19."
    ),
    iso_ref="5.3.19",
    type=CIRCUMFERENCE, standard=ISO, region=ARM, tier=FITTED,
    garments=_g(TOPS), unit="cm",
    needs_joints=True, anny_only=False,
))

_reg(MeasurementDef(
    key="crotch_length_cm",
    name="Total crotch length",
    description=(
        "Distance from the front waist centre, through the crotch (inside "
        "leg level), to the back waist centre. The tape follows the body "
        "surface without constriction at the crotch. ISO 8559-1 §5.4.18."
    ),
    iso_ref="5.4.18",
    type=LENGTH, standard=ISO, region=LEG, tier=FITTED,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="front_rise_cm",
    name="Front rise",
    description=(
        "Distance from the front waist level to the crotch point, measured "
        "along the front body surface. Used for trouser front panel length. "
        "Industry standard (not in ISO 8559-1)."
    ),
    iso_ref=None,
    type=LENGTH, standard=TAILOR, region=LEG, tier=FITTED,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="back_rise_cm",
    name="Back rise",
    description=(
        "Distance from the back waist level to the crotch point, measured "
        "along the back body surface. Used for trouser back panel length. "
        "Industry standard (not in ISO 8559-1)."
    ),
    iso_ref=None,
    type=LENGTH, standard=TAILOR, region=LEG, tier=FITTED,
    garments=_g(BOTTOMS), unit="cm",
    needs_joints=False, anny_only=False,
))

_reg(MeasurementDef(
    key="shirt_length_cm",
    name="Shirt/jacket length",
    description=(
        "Distance from the side neck point (where collar meets shoulder seam) "
        "down along the front body contour to the crotch level. Let the tape "
        "follow the natural curve of the chest and stomach. "
        "Industry standard (not in ISO 8559-1)."
    ),
    iso_ref=None,
    type=LENGTH, standard=TAILOR, region=TORSO, tier=FITTED,
    garments=_g(TOPS), unit="cm",
    needs_joints=True, anny_only=False,
))

# ── Presets ──────────────────────────────────────────────────────────────────

def _keys_up_to_tier(tier: str) -> frozenset[str]:
    """Return all measurement keys at or below the given tier."""
    idx = _TIER_ORDER.index(tier)
    allowed = set(_TIER_ORDER[: idx + 1])
    return frozenset(k for k, m in REGISTRY.items() if m.tier in allowed)


def _keys_for_garment(garment: str) -> frozenset[str]:
    """Return all measurement keys relevant to a garment type."""
    return frozenset(k for k, m in REGISTRY.items() if garment in m.garments)


TIER_PRESETS: dict[str, frozenset[str]] = {
    CORE: _keys_up_to_tier(CORE),
    STANDARD: _keys_up_to_tier(STANDARD),
    ENHANCED: _keys_up_to_tier(ENHANCED),
    FITTED: _keys_up_to_tier(FITTED),
    "all": _keys_up_to_tier(FITTED),
}

GARMENT_PRESETS: dict[str, frozenset[str]] = {
    TOPS: _keys_for_garment(TOPS),
    BOTTOMS: _keys_for_garment(BOTTOMS),
    DRESSES: _keys_for_garment(DRESSES),
    OUTERWEAR: _keys_for_garment(OUTERWEAR),
    UNDERWEAR: _keys_for_garment(UNDERWEAR),
}

PRESETS: dict[str, frozenset[str]] = {**TIER_PRESETS, **GARMENT_PRESETS}


# ── Query helpers ────────────────────────────────────────────────────────────


def list_measurements(**filters: str) -> list[MeasurementDef]:
    """Return measurement definitions matching all given tag filters.

    Filters are AND-ed across dimensions::

        list_measurements()                                  # all
        list_measurements(type="circumference")              # circumferences only
        list_measurements(type="circumference", region="leg")  # leg circumferences

    For the ``garments`` dimension, pass a single garment tag::

        list_measurements(garments="tops")                   # tops-relevant
    """
    for dim in filters:
        if dim not in VALID_TAG_DIMS:
            raise ValueError(
                f"Unknown tag dimension {dim!r}. "
                f"Valid: {sorted(VALID_TAG_DIMS)}"
            )
    result = []
    for m in REGISTRY.values():
        match = True
        for dim, val in filters.items():
            if dim == "garments":
                if val not in getattr(m, dim):
                    match = False
                    break
            else:
                if getattr(m, dim) != val:
                    match = False
                    break
        if match:
            result.append(m)
    return result


def resolve_keys(
    *,
    preset: str | None = None,
    only: list[str] | None = None,
    tags: dict[str, str] | None = None,
    exclude: list[str] | None = None,
) -> frozenset[str]:
    """Resolve a set of measurement keys from selection parameters.

    Precedence: ``only`` > ``preset`` > ``tags`` > default (all).
    ``exclude`` is applied last, removing keys from the resolved set.

    Raises ``ValueError`` for unknown keys, presets, or tag dimensions.
    """
    # 1. Resolve base set
    if only is not None:
        unknown = set(only) - REGISTRY.keys()
        if unknown:
            raise ValueError(f"Unknown measurement keys: {sorted(unknown)}")
        keys = frozenset(only)
    elif preset is not None:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. "
                f"Valid: {sorted(PRESETS.keys())}"
            )
        keys = PRESETS[preset]
    elif tags is not None:
        matches = list_measurements(**tags)
        keys = frozenset(m.key for m in matches)
    else:
        keys = PRESETS["all"]

    # 2. Apply exclude
    if exclude is not None:
        unknown = set(exclude) - REGISTRY.keys()
        if unknown:
            raise ValueError(f"Unknown measurement keys in exclude: {sorted(unknown)}")
        keys = keys - frozenset(exclude)

    return keys
