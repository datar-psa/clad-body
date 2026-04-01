"""Load Anny body meshes from phenotype params or raw arrays.

All outputs are Z-up, metres, XY-centred, feet at Z = 0 (Anny native /
Newton native convention).

Sources:
  * :func:`load_anny_from_params` — generate from phenotype params dict
  * :func:`load_anny_from_arrays` — create from raw numpy arrays
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnnyBody:
    """Loaded Anny body mesh in canonical A-pose convention.

    Coordinate system: Z-up, metres, XY-centred, feet at Z = 0.
    """

    vertices: np.ndarray          # (N, 3) float32
    faces: np.ndarray             # (M, 3) int32 — triangulated
    source: str                   # description of origin
    obj_path: Optional[str] = None

    # Phenotype params (11 Anny params, if available from JSON export)
    phenotype_params: Optional[dict] = None
    phenotype_labels: list = field(default_factory=lambda: [
        "gender", "age", "muscle", "weight", "height",
        "proportions", "cupsize", "firmness",
        "african", "asian", "caucasian",
    ])

    @property
    def height_m(self) -> float:
        """Body height in metres (Z extent)."""
        return float(self.vertices[:, 2].max() - self.vertices[:, 2].min())

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    def __repr__(self) -> str:
        params_str = f", params={len(self.phenotype_params)} keys" if self.phenotype_params else ""
        return (
            f"AnnyBody({self.n_vertices} verts, {self.n_faces} faces, "
            f"height={self.height_m:.3f}m, source='{self.source}'{params_str})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def reposition_apose(verts: np.ndarray) -> np.ndarray:
    """Normalise vertex positions to canonical A-pose convention.

    Applies in-place to a copy:
      * Feet at Z = 0
      * Body centred on XY origin

    Args:
        verts: (N, 3) array, Z-up, metres.

    Returns:
        New (N, 3) array with canonical positioning.
    """
    v = verts.copy()
    v[:, 2] -= v[:, 2].min()                                    # feet at Z=0
    center_xy = (v[:, :2].max(axis=0) + v[:, :2].min(axis=0)) / 2
    v[:, 0] -= center_xy[0]
    v[:, 1] -= center_xy[1]
    return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Anny bone indices for A-pose
_LOWERARM01_L = 52
_LOWERARM01_R = 78
_UPPERLEG01_L = 2
_UPPERLEG01_R = 22


def build_anny_apose(model, device, arm_angle_deg=-45.0, leg_angle_deg=0.0):
    """Build Anny A-pose tensor.

    Returns a (1, n_joints, 4, 4) pose tensor with identity everywhere except
    the specified arm and leg rotations on the X axis.

    Presets:
      * Default (-45° arms, 0° legs) — standard Anny A-pose.
      * (-40° arms, +3° legs) — MHR-matching pose for fitting.

    Args:
        model: Anny fullbody model (needs ``bone_count``).
        device: torch device.
        arm_angle_deg: X rotation for lowerarm01 bones (default: -45°).
        leg_angle_deg: X rotation for upperleg01 bones (default: 0°).

    Returns:
        ``torch.Tensor`` shape (1, n_joints, 4, 4).
    """
    import math

    import roma
    import torch

    n_joints = model.bone_count
    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose = pose.unsqueeze(0).unsqueeze(0).expand(1, n_joints, 4, 4).clone()

    if arm_angle_deg != 0.0:
        arm_rv = torch.tensor(
            [math.radians(arm_angle_deg), 0.0, 0.0],
            device=device, dtype=torch.float32,
        )
        arm_rot = roma.rotvec_to_rotmat(arm_rv)
        pose[0, _LOWERARM01_L, :3, :3] = arm_rot
        pose[0, _LOWERARM01_R, :3, :3] = arm_rot

    if leg_angle_deg != 0.0:
        leg_rv = torch.tensor(
            [math.radians(leg_angle_deg), 0.0, 0.0],
            device=device, dtype=torch.float32,
        )
        leg_rot = roma.rotvec_to_rotmat(leg_rv)
        pose[0, _UPPERLEG01_L, :3, :3] = leg_rot
        pose[0, _UPPERLEG01_R, :3, :3] = leg_rot

    return pose


def load_anny_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    source: str = "numpy_arrays",
    phenotype_params: Optional[dict] = None,
) -> AnnyBody:
    """Create :class:`AnnyBody` from raw numpy arrays (already Z-up, metres).

    Applies canonical A-pose positioning (feet-at-zero, XY-centred).
    """
    verts = reposition_apose(np.asarray(vertices, dtype=np.float32))
    return AnnyBody(
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int32),
        source=source,
        phenotype_params=phenotype_params,
    )


def load_anny_from_params(
    params: dict,
    device: str = "cpu",
) -> AnnyBody:
    """Generate Anny body from phenotype params dict.

    Creates an Anny model, builds an A-pose (lowerarm01 X=-45deg), runs the
    forward pass, and returns an :class:`AnnyBody` in canonical rest-pose.

    Supports ``_local_changes`` dict in *params* for local shape modifiers.

    Args:
        params: Phenotype params dict with keys like ``gender``, ``height``,
                ``weight``, etc.  All values in [0, 1].
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        :class:`AnnyBody` in canonical rest-pose (Z-up, m, +Y=front, XY-centred).
    """
    import anny
    import torch

    device = torch.device(device)

    # Extract local changes if present
    local_changes = params.get("_local_changes", {})
    local_change_labels = list(local_changes.keys()) if local_changes else False

    model = anny.create_fullbody_model(
        all_phenotypes=True, triangulate_faces=True,
        local_changes=local_change_labels,
    ).to(dtype=torch.float32, device=device)

    # Phenotype kwargs
    phenotype_kwargs = {}
    for label in model.phenotype_labels:
        if label in params:
            phenotype_kwargs[label] = torch.tensor(
                [params[label]], dtype=torch.float32, device=device,
            )

    # Local changes kwargs
    local_kwargs = {}
    for label, value in local_changes.items():
        local_kwargs[label] = torch.tensor(
            [value], dtype=torch.float32, device=device,
        )

    a_pose = build_anny_apose(model, device)

    with torch.no_grad():
        output = model(
            pose_parameters=a_pose,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_kwargs,
            pose_parameterization="root_relative_world",
        )

    verts_np = output["vertices"][0].cpu().numpy().astype(np.float32)
    faces_np = model.faces.cpu().numpy().astype(np.int32)

    # Z-up, feet at Z=0, XY-centred
    verts_np = reposition_apose(verts_np)

    return AnnyBody(
        vertices=verts_np,
        faces=faces_np,
        source="params",
        phenotype_params=dict(params),  # full copy including _local_changes
    )
