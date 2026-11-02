# BSD 3-Clause License
#
# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Camera frustum visualization utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera


# Global debug draw interface (initialized on first use)
_draw_interface = None
_debug_draw_available = None

_USD_ROOT = "/World/GnpCameraFrustums"
# Accumulated USD segments across multi-camera draws in one env step.
_usd_pending_points: list = []
_usd_pending_counts: list = []
_usd_pending_colors: list = []


def _get_draw_interface():
    """Get or create the debug draw interface."""
    global _draw_interface, _debug_draw_available

    if _debug_draw_available is None:
        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            _draw_interface = omni_debug_draw.acquire_debug_draw_interface()
            _debug_draw_available = True
        except (ModuleNotFoundError, ImportError, AttributeError):
            _debug_draw_available = False
            _draw_interface = None

    if not _debug_draw_available:
        return None

    return _draw_interface


def _to_torch(x: Any, device: torch.device | str) -> torch.Tensor:
    """Convert Torch / Warp / numpy pose buffers to a float torch tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    # Warp arrays / Fabric ProxyArray (often CUDA-backed; avoid np.asarray on device).
    try:
        import warp as wp

        if isinstance(x, wp.array) or type(x).__name__ == "ProxyArray":
            return wp.to_torch(x).to(device=device, dtype=torch.float32)
    except Exception:
        pass
    if hasattr(x, "numpy") and callable(x.numpy):
        try:
            return torch.as_tensor(x.numpy(), device=device, dtype=torch.float32)
        except Exception:
            pass
    arr = np.asarray(x)
    if arr.dtype == object or (arr.ndim >= 1 and not np.issubdtype(arr.dtype, np.number)):
        arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in list(x)])
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def _as_torch_xyzn(x: Any, device: torch.device | str) -> torch.Tensor:
    """Convert robot/sensor pose buffers to ``(N, 3)`` float tensors."""
    t = _to_torch(x, device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.shape[-1] != 3 and t.numel() % 3 == 0:
        t = t.reshape(-1, 3)
    return t


def _as_torch_quatn(x: Any, device: torch.device | str) -> torch.Tensor:
    """Convert quaternion buffers to ``(N, 4)`` float tensors (wxyz or xyzw as stored)."""
    t = _to_torch(x, device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.shape[-1] != 4 and t.numel() % 4 == 0:
        t = t.reshape(-1, 4)
    return t


def _compute_corner_ray_indices(camera: MultiMeshRayCasterCamera, device: str) -> torch.Tensor:
    """Linear indices for the 4 FOV corner pixels (TL, TR, BL, BR)."""
    width = camera.cfg.pattern_cfg.width
    height = camera.cfg.pattern_cfg.height

    corner_pixel_indices = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ]
    return torch.tensor(
        [y * width + x for x, y in corner_pixel_indices], device=device, dtype=torch.long
    )


def _flush_usd_frustum_curves(line_width: float = 0.012) -> None:
    """Update USD BasisCurves in place (visible in RTX / KitVisualizer captures)."""
    if not _usd_pending_points:
        return
    try:
        import omni.usd
        from pxr import Sdf, UsdGeom
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    root = stage.GetPrimAtPath(_USD_ROOT)
    if not root.IsValid():
        UsdGeom.Xform.Define(stage, _USD_ROOT)

    curves_path = f"{_USD_ROOT}/lines"
    curves_prim = stage.GetPrimAtPath(curves_path)
    if not curves_prim.IsValid():
        curves = UsdGeom.BasisCurves.Define(stage, curves_path)
        curves.CreateTypeAttr("linear")
        curves.CreateBasisAttr("bezier")
        curves.CreateWrapAttr("nonperiodic")
        curves.CreatePointsAttr(_usd_pending_points)
        curves.CreateCurveVertexCountsAttr(_usd_pending_counts)
        curves.CreateWidthsAttr([float(line_width)] * len(_usd_pending_points))
        UsdGeom.PrimvarsAPI(curves.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
        ).Set(_usd_pending_colors)
    else:
        curves = UsdGeom.BasisCurves(curves_prim)
        curves.GetPointsAttr().Set(_usd_pending_points)
        curves.GetCurveVertexCountsAttr().Set(_usd_pending_counts)
        widths_attr = curves.GetWidthsAttr()
        if widths_attr:
            widths_attr.Set([float(line_width)] * len(_usd_pending_points))
        color_pv = UsdGeom.PrimvarsAPI(curves_prim).GetPrimvar("displayColor")
        if color_pv:
            color_pv.Set(_usd_pending_colors)


def visualize_camera_frustum(
    env: ManagerBasedEnv,
    camera_name: str = "depth_camera",
    frustum_distance: float = 2.0,
    line_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    line_thickness: float = 2.0,
    robot_name: str = "robot",
    clear_existing: bool = True,
) -> None:
    """Draw camera FOV frustums (origin→corners + far rectangle).

    Draws via USD BasisCurves (visible in KitVisualizer / RTX headless capture) and
    optionally via debug-draw for interactive viewports.

    Set ``clear_existing=False`` when stacking multiple cameras in one frame.
    """
    global _usd_pending_points, _usd_pending_counts, _usd_pending_colors

    camera = env.scene[camera_name]
    robot = env.scene[robot_name]

    corner_ray_indices = _compute_corner_ray_indices(camera, env.device)
    corner_directions_camera = camera.ray_directions[0][corner_ray_indices]  # Shape: (4, 3)

    base_pos_w = _as_torch_xyzn(robot.data.root_pos_w, env.device)
    base_quat_w = _as_torch_quatn(robot.data.root_quat_w, env.device)

    from general_navigation_policy.tasks.mdp.camera_buffers import ensure_camera_offset_buffers

    ensure_camera_offset_buffers(camera)
    offset_pos = _as_torch_xyzn(camera._offset_pos, env.device)
    offset_quat = _as_torch_quatn(camera._offset_quat, env.device)

    camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
        base_pos_w, base_quat_w, offset_pos, offset_quat
    )

    start_positions = []
    end_positions = []
    line_colors = []
    line_thicknesses = []

    if clear_existing:
        _usd_pending_points = []
        _usd_pending_counts = []
        _usd_pending_colors = []

    num_envs = env.num_envs
    rgb = (float(line_color[0]), float(line_color[1]), float(line_color[2]))

    for env_idx in range(num_envs):
        cam_pos = camera_pos_w[env_idx]
        cam_quat = camera_quat_w[env_idx]
        cam_pos_np = cam_pos.detach().float().cpu().numpy()

        corner_positions_world = []
        for corner_idx in range(4):
            corner_dir_cam = corner_directions_camera[corner_idx].to(env.device)
            corner_dir_world = math_utils.quat_apply(
                cam_quat.unsqueeze(0),
                corner_dir_cam.unsqueeze(0),
            ).squeeze(0)
            corner_dir_world = corner_dir_world / (torch.norm(corner_dir_world) + 1e-8)
            far_corner_pos = cam_pos + corner_dir_world * frustum_distance
            corner_positions_world.append(far_corner_pos.detach().float().cpu().numpy())

        for corner_pos in corner_positions_world:
            start_positions.append(cam_pos_np)
            end_positions.append(corner_pos)
            line_colors.append(list(line_color))
            line_thicknesses.append(line_thickness)

        corner_indices = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for i, j in corner_indices:
            start_positions.append(corner_positions_world[i])
            end_positions.append(corner_positions_world[j])
            line_colors.append(list(line_color))
            line_thicknesses.append(line_thickness)

    try:
        from pxr import Gf
    except Exception:
        Gf = None  # type: ignore

    if Gf is not None:
        for s, e in zip(start_positions, end_positions):
            _usd_pending_points.append(Gf.Vec3f(float(s[0]), float(s[1]), float(s[2])))
            _usd_pending_points.append(Gf.Vec3f(float(e[0]), float(e[1]), float(e[2])))
            _usd_pending_counts.append(2)
            _usd_pending_colors.append(Gf.Vec3f(*rgb))

    _flush_usd_frustum_curves(line_width=max(0.008, 0.006 * float(line_thickness)))

    draw_interface = _get_draw_interface()
    if draw_interface is not None and start_positions:
        if clear_existing:
            draw_interface.clear_lines()
        draw_interface.draw_lines(start_positions, end_positions, line_colors, line_thicknesses)
