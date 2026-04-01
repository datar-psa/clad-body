"""Load MHR body meshes from SAM3D params JSON via pymomentum.

All outputs are Z-up, metres, XY-centred, feet at Z = 0, +Y=front.
Matches Anny convention — both loaders produce the same canonical orientation.

Uses :func:`load_mhr_from_params` — generates from SAM3D params JSON via
pymomentum subprocess.  Deterministic coordinate conversion (no heuristic
orientation detection).
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh


@dataclass
class MhrBody:
    """MHR body mesh in canonical rest-pose convention.

    Coordinate system: Z-up, metres, XY-centred, feet at Z = 0, +Y=front.
    Matches Anny convention — both loaders produce the same orientation.
    """

    mesh: trimesh.Trimesh        # canonical-positioned mesh
    source: str                  # e.g. "params:sam3d_mhr_restpose_params.json"
    obj_path: Optional[str] = None
    sam3d_params: Optional[dict] = None  # shape_params, scale_params, raw_height_m
    joints: Optional[dict] = None  # canonical joint name → (3,) position (Z-up, m)

    @property
    def height_m(self) -> float:
        verts = np.asarray(self.mesh.vertices)
        return float(verts[:, 2].max() - verts[:, 2].min())

    @property
    def n_vertices(self) -> int:
        return len(self.mesh.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.mesh.faces)

    def __repr__(self) -> str:
        params_str = " +params" if self.sam3d_params else ""
        return (
            f"MhrBody({self.n_vertices} verts, {self.n_faces} faces, "
            f"height={self.height_m:.3f}m, source='{self.source}'{params_str})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mhr_yup_cm_to_canonical(verts_yup_cm: np.ndarray) -> np.ndarray:
    """Convert pymomentum output (Y-up, cm) to canonical (Z-up, m, +Y=front).

    Deterministic transformation — no orientation detection needed.
    MHR native: Y=height, -Z=front.  Target: Z=height, +Y=front.
    """
    v = np.zeros_like(verts_yup_cm, dtype=np.float32)
    v[:, 0] = verts_yup_cm[:, 0] / 100.0        # X stays X, cm → m
    v[:, 1] = -verts_yup_cm[:, 2] / 100.0       # MHR -Z=front → +Y=front
    v[:, 2] = verts_yup_cm[:, 1] / 100.0        # MHR Y=height → Z=height
    v[:, 2] -= v[:, 2].min()                     # feet at Z = 0
    return v


def _resolve_params_json(path: str) -> str:
    """Resolve path to sam3d_mhr_restpose_params.json."""
    if path.endswith(".json") and os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "sam3d_mhr_restpose_params.json")
        if os.path.exists(candidate):
            return candidate
    # Try sibling of OBJ
    if path.endswith(".obj"):
        candidate = path.replace(".obj", "_params.json")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No params JSON found for: {path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mhr_from_params_dict(
    params: dict,
    elbow_bend: float = -0.5,
) -> MhrBody:
    """Generate MHR body from a SAM3D params dict (no file on disk required).

    Writes a temporary JSON file and delegates to :func:`load_mhr_from_params`.

    Args:
        params: SAM3D params dict with at least ``shape_params`` and ``scale_params``.
        elbow_bend: MHR elbow_bend parameter (default: -0.5).

    Returns:
        :class:`MhrBody` in canonical rest-pose (Z-up, m, +Y=front, XY-centred).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as jf:
        json.dump(params, jf)
        tmp_path = jf.name
    try:
        body = load_mhr_from_params(tmp_path, elbow_bend=elbow_bend)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return body


def load_mhr_from_params(
    path: str,
    elbow_bend: float = -0.5,
) -> MhrBody:
    """Generate MHR body from SAM3D params JSON via pymomentum.

    Deterministic coordinate conversion, no heuristic orientation detection.
    Runs pymomentum in a subprocess (import-order isolation:
    ``pymomentum.geometry`` must come before ``torch``).

    Args:
        path: Path to params JSON, SAM3D results directory, or restpose OBJ
              (auto-finds companion ``_params.json``).
        elbow_bend: MHR elbow_bend parameter (default: -0.5).

    Returns:
        :class:`MhrBody` in canonical rest-pose (Z-up, m, +Y=front, XY-centred).
    """
    params_json = _resolve_params_json(path)

    # Ensure LD_LIBRARY_PATH includes torch/lib for pymomentum-cpu
    import importlib.util
    _torch_spec = importlib.util.find_spec("torch")
    if _torch_spec and _torch_spec.origin:
        _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _torch_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{_ld}" if _ld else _torch_lib

    # Subprocess script — pymomentum.geometry MUST import before torch
    script = f"""\
import sys, os
import pymomentum.geometry  # noqa: F401 — MUST come before torch
import pymomentum.skel_state as pym_skel_state
import json, torch, numpy as np
from mhr.mhr import MHR

with open({params_json!r}) as f:
    params = json.load(f)

shape_t = torch.tensor(params["shape_params"], dtype=torch.float32).unsqueeze(0)

model = MHR.from_files(device="cpu", wants_pose_correctives=False)

# Rest pose: zero translation/rotation/pose, keep scale from model output.
# Layout: [trans(3)|rot(3)|pose(130)|scale(68)] = 204
mp = torch.zeros(1, 204, dtype=torch.float32)
mp[0, 36] = {elbow_bend}   # r_elbow_bend
mp[0, 46] = {elbow_bend}   # l_elbow_bend

# Scale params: either from full mhr_model_params[136:] or standalone scale_params
if "mhr_model_params" in params:
    mhr_params = torch.tensor(params["mhr_model_params"], dtype=torch.float32)
    mp[0, 136:] = mhr_params[136:]
elif "scale_params" in params:
    scale_t = torch.tensor(params["scale_params"], dtype=torch.float32)
    mp[0, 136:136+len(scale_t)] = scale_t

with torch.no_grad():
    verts, skel_state = model(shape_t, mp, None, apply_correctives=True)

v = verts[0].cpu().numpy().astype(np.float32)
faces = np.array(model.character.mesh.faces, dtype=np.int32)

# Extract joint positions from skeleton state
joint_positions, _, _ = pym_skel_state.split(skel_state)
joint_pos = joint_positions[0].cpu().numpy().astype(np.float32)
joint_names = model.character_torch.skeleton.joint_names

np.savez(sys.argv[1], vertices=v, faces=faces,
         joint_positions=joint_pos,
         joint_names=np.array(joint_names))
"""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as nf:
        npz_path = nf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
        sf.write(script)
        script_path = sf.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, npz_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            err = result.stderr[-800:] if result.stderr else "unknown error"
            raise RuntimeError(f"pymomentum subprocess failed:\n{err}")

        data = np.load(npz_path, allow_pickle=True)
        verts_yup_cm = data["vertices"]
        faces = data["faces"]
        joint_pos_raw = data.get("joint_positions")
        joint_names_raw = data.get("joint_names")
    finally:
        for p in (npz_path, script_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    # Deterministic conversion: Y-up cm → Z-up m, +Y=front
    verts = _mhr_yup_cm_to_canonical(verts_yup_cm)

    # XY-centre
    center_xy = (verts[:, :2].max(axis=0) + verts[:, :2].min(axis=0)) / 2
    verts[:, 0] -= center_xy[0]
    verts[:, 1] -= center_xy[1]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Extract canonical joint positions (same coordinate transform as vertices)
    joints = None
    if joint_pos_raw is not None and joint_names_raw is not None:
        from clad_body.measure.mhr import MHR_JOINT_MAP
        from clad_body.measure._lengths import extract_joints_from_names
        # Joint positions are in MHR native Y-up cm → convert to Z-up m
        joint_pos_canonical = _mhr_yup_cm_to_canonical(joint_pos_raw)
        # Apply same XY centering as vertices
        joint_pos_canonical[:, 0] -= center_xy[0]
        joint_pos_canonical[:, 1] -= center_xy[1]
        joint_names = list(joint_names_raw)
        joints = extract_joints_from_names(
            joint_names, joint_pos_canonical, MHR_JOINT_MAP)

    with open(params_json) as f:
        sam3d_params = json.load(f)

    return MhrBody(
        mesh=mesh,
        source=f"params:{os.path.basename(params_json)}",
        sam3d_params=sam3d_params,
        joints=joints,
    )
