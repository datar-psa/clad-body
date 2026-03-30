"""Body loading module — canonical loaders for Anny and MHR body meshes.

Both loaders normalise to the same coordinate convention:
  - Z-up, metres
  - XY-centred (body axis on Z)
  - Feet at Z = 0
  - +Y = front (body faces +Y direction)

Load from params (deterministic orientation)::

    from clad_body.load.mhr  import load_mhr_from_params
    from clad_body.load.anny import load_anny_from_params

    mhr  = load_mhr_from_params("path/to/params.json")
    anny = load_anny_from_params({"gender": 0.5, "height": 0.6, ...})
"""

from .anny import AnnyBody, build_anny_apose, load_anny_from_arrays, load_anny_from_params

try:
    from .mhr import MhrBody, load_mhr_from_params
except ImportError:
    pass

__all__ = [
    "AnnyBody",
    "build_anny_apose",
    "load_anny_from_arrays",
    "load_anny_from_params",
    "MhrBody",
    "load_mhr_from_params",
]
