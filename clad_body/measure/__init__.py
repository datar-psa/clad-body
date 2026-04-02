"""Body measurement package — Anny and MHR body models.

Public API::

    from clad_body.measure import measure, REGISTRY, list_measurements, MeasurementDef

    body = load_anny_from_params(params)
    m = measure(body)                              # all measurements
    m = measure(body, preset="core")               # 4: height, bust, waist, hip
    m = measure(body, only=["bust_cm", "hip_cm"])  # specific keys
    m = measure(body, tags={"type": "circumference", "region": "leg"})
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
GROUP_B = "limb_sweeps"      # thigh, knee, calf, upperarm
GROUP_C = "joint_linear"     # shoulder_width, sleeve_length
GROUP_D = "perpendicular"    # neck, wrist
GROUP_E = "mesh_geometry"    # inseam, crotch_length, front_rise, back_rise
GROUP_F = "surface_trace"    # shirt_length
GROUP_G = "body_composition" # volume, mass, bmi, body_fat, estimated_density, density_corrected_mass

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
    "wrist_cm": GROUP_D,
    "inseam_cm": GROUP_E,
    "crotch_length_cm": GROUP_E,
    "front_rise_cm": GROUP_E,
    "back_rise_cm": GROUP_E,
    "shirt_length_cm": GROUP_F,
    "volume_m3": GROUP_G,
    "mass_kg": GROUP_G,
    "bmi": GROUP_G,
    "body_fat_pct": GROUP_G,
}

# Group dependencies: F needs E (for inseam_z), G needs D (for neck/BF%)
_GROUP_DEPS = {
    GROUP_F: {GROUP_E},
    GROUP_G: {GROUP_D},
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
