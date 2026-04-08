"""Load Anny body meshes from phenotype params or raw arrays.

All outputs are Z-up, metres, XY-centred, feet at Z = 0.

Preferred entry point::

    from clad_body.load.anny import load_anny_from_params
    from clad_body.measure import measure

    body = load_anny_from_params(params)   # model + mesh + bones cached
    m = measure(body, only=["bust_cm"])    # reuses cached model (~100 ms)

The returned :class:`AnnyBody` carries the Anny model, trimesh, and bone
data.  ``measure()`` reuses all of it — no redundant model creation or
forward passes.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
import trimesh


@dataclass
class AnnyBody:
    """Loaded Anny body mesh in canonical A-pose convention.

    Coordinate system: Z-up, metres, XY-centred, feet at Z = 0.

    Properties:
        mesh:   Cached ``trimesh.Trimesh`` (lazy, built on first access).
        model:  The Anny rigged model (lazy, created from ``phenotype_params``
                on first access if not already set).

    The preferred way to create an ``AnnyBody`` is via
    :func:`load_anny_from_params`, which runs the forward pass and caches
    the model + bone data on the returned object.  This makes subsequent
    ``measure()`` calls fast (~100 ms) because they skip model creation
    (~400 ms) and reuse the cached mesh and bone positions.

    **Model reuse**: The Anny model is **stateless** with respect to body
    params — phenotype values and local changes are passed as kwargs to
    ``model.forward()``, not stored as model state.  The model only holds
    fixed topology (faces, blendshape basis, skinning weights, bone
    hierarchy).  It is safe to call ``model(phenotype_kwargs=A)`` then
    ``model(phenotype_kwargs=B)`` on the same model object.  What varies
    between model instances is which ``local_changes`` labels are enabled
    (that changes the blendshape basis dimensions).

    **Coordinate gotcha**: ``vertices`` are XY-centred (via
    ``reposition_apose``), but Anny bone positions from the forward pass
    are not.  ``_xy_offset`` stores the centering offset so ``measure()``
    can align joint positions to the mesh.  This is handled automatically —
    callers don't need to worry about it.
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

    # Set by load_anny_from_params; used by .model property as cache.
    _model: object = field(default=None, repr=False)
    # XY centering offset (for aligning bone/joint positions to the mesh).
    _xy_offset: Optional[np.ndarray] = field(default=None, repr=False)

    # Torch tensors mirroring phenotype_params + _local_changes, created by
    # load_anny_from_params so measure_grad() can re-run the forward pass with
    # gradients.  For autograd optimisation, call .requires_grad_(True) on the
    # tensors you want to optimise (or pass requires_grad=True to
    # load_anny_from_params to enable it on all of them).
    phenotype_kwargs: Optional[dict] = field(default=None, repr=False)
    local_changes_kwargs: Optional[dict] = field(default=None, repr=False)

    def __hash__(self):
        return id(self)

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        """Trimesh built from vertices/faces (cached on first access)."""
        return trimesh.Trimesh(
            vertices=self.vertices, faces=self.faces, process=False)

    @property
    def model(self):
        """Anny rigged model (topology, blendshape basis, bone weights).

        Returned from cache if available (set by :func:`load_anny_from_params`),
        otherwise created lazily from ``phenotype_params``.  The model is
        stateless — safe to reuse across forward passes with different params.
        """
        if self._model is not None:
            return self._model
        if self.phenotype_params is None:
            raise ValueError(
                "Cannot create Anny model without phenotype_params. "
                "Use load_anny_from_params() to create the body."
            )
        # Re-create model from params via load_anny_from_params (which caches
        # the model on the returned body) and steal it.
        self._model = load_anny_from_params(self.phenotype_params)._model
        return self._model

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


def load_anny_from_verts(
    verts_torch,
    model,
    *,
    phenotype_kwargs: Optional[dict] = None,
    bone_heads=None,
    bone_tails=None,
    source: str = "from_verts",
) -> "AnnyBody":
    """Wrap a forward-pass output `(verts, model)` into an :class:`AnnyBody`.

    Use this when you already have an Anny model and a fresh `model(...)`
    forward pass and just want a body to feed to ``measure()`` — typically
    inside an optimisation loop where re-running ``load_anny_from_params``
    (which creates a fresh model every call, ~400 ms) would dominate.

    Applies the canonical transform (Y-up → Z-up, feet at Z=0, XY-centred)
    and stashes the centering offset on the body so :func:`measure` aligns
    joints to the mesh.

    Args:
        verts_torch: ``(1, V, 3)`` torch tensor — output of
            ``model(...)['vertices']``.
        model: the Anny model used for the forward pass.  Reused as-is
            (cached on the returned body).
        phenotype_kwargs: optional kwargs from the same forward pass — if
            given, they're stashed on the model so ``measure()``'s gender
            inference can read them.  May be omitted (gender will fall
            back to a mesh heuristic).
        bone_heads, bone_tails: outputs of ``model(..., return_bone_ends=True)``.
            If provided, stashed on the model so joint extraction works
            inside ``measure()``.
        source: provenance label.

    Returns:
        :class:`AnnyBody` ready for :func:`clad_body.measure.measure`.
    """
    verts_np = verts_torch[0].detach().cpu().numpy().astype(np.float32)
    extents = verts_np.max(0) - verts_np.min(0)
    if int(np.argmax(extents)) == 1:  # Y-up → Z-up
        verts_np = verts_np[:, [0, 2, 1]].copy()
        verts_np[:, 2] = -verts_np[:, 2]
    verts_np[:, 2] -= verts_np[:, 2].min()
    center_xy = (verts_np[:, :2].max(0) + verts_np[:, :2].min(0)) / 2
    verts_np[:, 0] -= center_xy[0]
    verts_np[:, 1] -= center_xy[1]

    faces_np = (
        model.faces.detach().cpu().numpy()
        if hasattr(model.faces, "detach")
        else np.asarray(model.faces)
    ).astype(np.int32)

    if phenotype_kwargs is not None:
        model._last_phenotype_kwargs = phenotype_kwargs
    if bone_heads is not None:
        model._last_bone_heads = bone_heads
    if bone_tails is not None:
        model._last_bone_tails = bone_tails

    return AnnyBody(
        vertices=verts_np,
        faces=faces_np,
        source=source,
        phenotype_params={},  # non-None placeholder; measure() requires this
        _model=model,
        _xy_offset=(-center_xy).astype(np.float32),
    )


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
    *,
    requires_grad: bool = False,
) -> AnnyBody:
    """Generate Anny body from phenotype params dict.

    Creates an Anny model, builds an A-pose, runs the forward pass, and
    returns an :class:`AnnyBody` in canonical rest-pose.  The model and
    bone data are cached on the returned body so that subsequent
    ``measure()`` calls skip model creation (~400 ms).

    Supports ``_local_changes`` dict in *params* for local shape modifiers.

    Args:
        params: Phenotype params dict with keys like ``gender``, ``height``,
                ``weight``, etc.  All values in [0, 1].  May include
                ``_local_changes`` sub-dict for shape blendshapes.
        device: ``"cpu"`` or ``"cuda"``.
        requires_grad: If ``True``, all phenotype and local-change tensors on
                the returned body are created with ``requires_grad=True`` so
                :func:`clad_body.measure.measure_grad` can backprop into them.
                Default ``False`` — enable per-tensor via
                ``body.phenotype_kwargs[label].requires_grad_(True)`` if you
                only want gradients on a subset.

    Returns:
        :class:`AnnyBody` with ``_model``, ``_xy_offset``, ``phenotype_kwargs``
        and ``local_changes_kwargs`` populated.

    Performance:
        This is the recommended entry point.  The body carries its own
        model, mesh, and bone data — ``measure(body)`` reuses all of it.
        If you need to measure many bodies with the **same local_changes
        labels** but different param values, the model is stateless and can
        be shared (see :attr:`AnnyBody.model` docstring).
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

    # Phenotype kwargs (stored on body for measure_grad to reuse)
    phenotype_kwargs = {}
    for label in model.phenotype_labels:
        if label in params:
            phenotype_kwargs[label] = torch.tensor(
                [params[label]], dtype=torch.float32, device=device,
                requires_grad=requires_grad,
            )

    # Local changes kwargs
    local_kwargs = {}
    for label, value in local_changes.items():
        local_kwargs[label] = torch.tensor(
            [value], dtype=torch.float32, device=device,
            requires_grad=requires_grad,
        )

    a_pose = build_anny_apose(model, device)

    with torch.no_grad():
        output = model(
            pose_parameters=a_pose,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs=local_kwargs,
            pose_parameterization="root_relative_world",
            return_bone_ends=True,
        )

    # Stash bone data + phenotype kwargs on model for joint extraction and
    # gender inference in measure(). _infer_gender reads _last_phenotype_kwargs
    # to pick male/female body fat formulas.
    model._last_bone_heads = output.get("bone_heads")
    model._last_bone_tails = output.get("bone_tails")
    model._last_phenotype_kwargs = phenotype_kwargs

    verts_np = output["vertices"][0].cpu().numpy().astype(np.float32)
    faces_np = model.faces.cpu().numpy().astype(np.int32)

    # Convert to Z-up before reposition so we can compute the XY offset.
    # Detect Y-up → Z-up (same logic as _anny_to_trimesh)
    extents = verts_np.max(0) - verts_np.min(0)
    height_axis = int(np.argmax(extents))
    if height_axis == 1:  # Y-up → Z-up
        verts_np = verts_np[:, [0, 2, 1]].copy()
        verts_np[:, 2] = -verts_np[:, 2]

    # Compute XY offset BEFORE centering (= what reposition_apose will subtract)
    verts_np[:, 2] -= verts_np[:, 2].min()  # feet at Z=0
    center_xy = (verts_np[:, :2].max(0) + verts_np[:, :2].min(0)) / 2
    xy_offset = -center_xy  # offset to apply to joints

    # XY-centre
    verts_np[:, 0] -= center_xy[0]
    verts_np[:, 1] -= center_xy[1]

    return AnnyBody(
        vertices=verts_np.astype(np.float32),
        faces=faces_np,
        source="params",
        phenotype_params=dict(params),  # full copy including _local_changes
        phenotype_kwargs=phenotype_kwargs,
        local_changes_kwargs=local_kwargs,
        _model=model,
        _xy_offset=xy_offset.astype(np.float32),
    )
