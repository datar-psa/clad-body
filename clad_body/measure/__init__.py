"""Body measurement package — Anny and MHR body models.

Public API::

    from clad_body.load.anny import load_anny_from_params
    from clad_body.measure import measure

    body = load_anny_from_params(params)           # model cached on body
    m = measure(body)                              # all measurements (~800 ms)
    m = measure(body, preset="core")               # 4: height, bust, waist, hip
    m = measure(body, only=["bust_cm", "hip_cm"])  # specific keys (fastest)
    m = measure(body, device="cuda")               # GPU acceleration

**Performance**: Always use ``only=`` or ``preset=`` when you don't need all
measurements.  The ``AnnyBody`` from ``load_anny_from_params`` caches the Anny
model — ``measure()`` reuses it (~100 ms per call vs ~500 ms without cache).
"""

from __future__ import annotations

from clad_body.measure.registry import (
    REGISTRY,
    MeasurementDef,
    list_measurements,
    resolve_keys,
)

# ── Computation groups ───────────────────────────────────────────────────────
# Each measurement belongs to a computation group. Groups have dependencies.
# measure() resolves which groups need to run based on requested keys.

GROUP_A = "core_torso"       # height, bust, waist, hip, stomach, underbust, belly_depth
GROUP_B = "limb_sweeps"      # thigh, knee, calf, upperarm, wrist
GROUP_C = "joint_linear"     # shoulder_width, sleeve_length
GROUP_D = "perpendicular"    # neck
GROUP_E = "mesh_geometry"    # inseam, crotch_length, front_rise, back_rise
GROUP_F = "surface_trace"    # shirt_length
GROUP_G = "body_composition" # volume, mass, bmi, body_fat, estimated_density, density_corrected_mass
GROUP_H = "back_length"      # back_neck_to_waist (needs joints + waist_z from A)

_KEY_TO_GROUP = {
    "height_cm": GROUP_A,
    "bust_cm": GROUP_A,
    "waist_cm": GROUP_A,
    "hip_cm": GROUP_A,
    "stomach_cm": GROUP_A,
    "underbust_cm": GROUP_A,
    "belly_depth_cm": GROUP_A,
    "thigh_cm": GROUP_B,
    "knee_cm": GROUP_B,
    "calf_cm": GROUP_B,
    "upperarm_cm": GROUP_B,
    "shoulder_width_cm": GROUP_C,
    "sleeve_length_cm": GROUP_C,
    "neck_cm": GROUP_D,
    "wrist_cm": GROUP_B,
    "inseam_cm": GROUP_E,
    "crotch_length_cm": GROUP_E,
    "front_rise_cm": GROUP_E,
    "back_rise_cm": GROUP_E,
    "shirt_length_cm": GROUP_F,
    "back_neck_to_waist_cm": GROUP_H,
    "volume_m3": GROUP_G,
    "mass_kg": GROUP_G,
    "bmi": GROUP_G,
    "body_fat_pct": GROUP_G,
    "estimated_density": GROUP_G,
    "density_corrected_mass_kg": GROUP_G,
}

# Group dependencies:
#   F needs E (uses inseam_z to find shirt-length endpoint)
#   G needs D (BF% formula uses neck) AND A (BF% formula uses waist + hip).
#     Without A, waist/hip default to 0 → BF% formula early-returns 3% →
#     wrong density → mass_kg off by 5+ kg.  See test_only_mass_kg_matches_full.
#   H needs A (back_neck_to_waist surface trace endpoint is _waist_z)
_GROUP_DEPS = {
    GROUP_F: {GROUP_E},
    GROUP_G: {GROUP_A, GROUP_D},
    GROUP_H: {GROUP_A},
}


def _resolve_groups(requested_keys: frozenset[str]) -> frozenset[str]:
    """Determine which computation groups to run for the requested keys."""
    groups = set()
    for key in requested_keys:
        g = _KEY_TO_GROUP.get(key)
        if g:
            groups.add(g)
    # Add dependencies
    changed = True
    while changed:
        changed = False
        for g in list(groups):
            for dep in _GROUP_DEPS.get(g, set()):
                if dep not in groups:
                    groups.add(dep)
                    changed = True
    return frozenset(groups)


def measure(
    body,
    *,
    preset: str | None = None,
    only: list[str] | None = None,
    tags: dict[str, str] | None = None,
    exclude: list[str] | None = None,
    render_path: str | None = None,
    title: str = "",
    device: str | None = None,
) -> dict:
    """Measure a body model (Anny or MHR).

    Selection (precedence: only > preset > tags > default "all"):
        preset: "core", "standard", "enhanced", "fitted"/"all",
                or garment: "tops", "bottoms", "dresses", "outerwear", "underwear"
        only:   explicit key list (highest precedence)
        tags:   {"type": "circumference", "region": "leg"} (AND)
        exclude: remove from resolved set (applied last)

    Only runs the computation groups needed for the requested keys.

    Args:
        body: AnnyBody or MhrBody instance.
        preset: Named preset (tier or garment).
        only: Explicit list of measurement keys.
        tags: Tag-based filter (AND across dimensions).
        exclude: Keys to remove from resolved set.
        render_path: If set, save 4-view render PNG.
        title: Title for the render.
        device: ``"cpu"``, ``"cuda"``, or ``None`` (auto: CUDA if available).

    Returns:
        dict with measurement keys (e.g. "bust_cm" → float) plus internal
        metadata (keys prefixed with "_").
    """
    requested_keys = resolve_keys(
        preset=preset, only=only, tags=tags, exclude=exclude,
    )
    groups = _resolve_groups(requested_keys)

    # Dispatch based on body type
    from clad_body.load.anny import AnnyBody
    from clad_body.load.mhr import MhrBody

    if isinstance(body, AnnyBody):
        from clad_body.measure.anny import _measure_anny
        all_measurements = _measure_anny(
            body, groups=groups, render_path=render_path, title=title,
            device=device,
        )
    elif isinstance(body, MhrBody):
        from clad_body.measure.mhr import _measure_mhr
        all_measurements = _measure_mhr(
            body, groups=groups, render_path=render_path, title=title,
        )
    else:
        raise TypeError(
            f"Expected AnnyBody or MhrBody, got {type(body).__name__}"
        )

    # Filter to requested keys + internal metadata
    result = {}
    for k, v in all_measurements.items():
        if k.startswith("_") or k in requested_keys:
            result[k] = v
    # Always include mesh and contours if present (needed for rendering/export)
    for special in ("mesh", "contours"):
        if special in all_measurements:
            result[special] = all_measurements[special]

    return result


__all__ = [
    "measure",
    "REGISTRY",
    "MeasurementDef",
    "list_measurements",
]
