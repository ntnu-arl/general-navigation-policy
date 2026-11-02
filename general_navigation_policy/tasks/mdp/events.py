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
"""Event functions for camera pose / intrinsics / active-camera randomization."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz
import isaaclab.utils.math as math_utils

from general_navigation_policy.tasks.mdp.camera_buffers import ensure_camera_offset_buffers


def randomize_camera_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | list[int] | slice,
    sensor_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
    orientation_range: dict[str, tuple[float, float]],
):
    """Randomize camera offset pose relative to the sensor base offset (pos m, Euler deg)."""
    # Get the camera sensor
    camera = env.scene[sensor_cfg.name]
    
    # Convert env_ids to tensor if needed
    if isinstance(env_ids, slice):
        if env_ids == slice(None):
            env_ids = torch.arange(env.num_envs, device=camera.device)
        else:
            env_ids = torch.arange(env.num_envs, device=camera.device)[env_ids]
    elif isinstance(env_ids, list):
        env_ids = torch.tensor(env_ids, device=camera.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=camera.device, dtype=torch.long)

    # Sample random positions
    num_envs = len(env_ids)
    device = camera.device

    # Sample positions relative to base offset
    pos_x = torch.empty(num_envs, device=device).uniform_(*position_range["x"])
    pos_y = torch.empty(num_envs, device=device).uniform_(*position_range["y"])
    pos_z = torch.empty(num_envs, device=device).uniform_(*position_range["z"])

    # Sample orientations (convert degrees to radians)
    roll = torch.deg2rad(
        torch.empty(num_envs, device=device).uniform_(*orientation_range["roll"])
    )
    pitch = torch.deg2rad(
        torch.empty(num_envs, device=device).uniform_(*orientation_range["pitch"])
    )
    yaw = torch.deg2rad(
        torch.empty(num_envs, device=device).uniform_(*orientation_range["yaw"])
    )

    # quat_from_euler_xyz → (x, y, z, w)
    quats = quat_from_euler_xyz(roll, pitch, yaw)

    ensure_camera_offset_buffers(camera)
    
    # Get base offset from config
    base_pos = torch.tensor(camera.cfg.offset.pos, device=device)
    base_quat = torch.tensor(camera.cfg.offset.rot, device=device).unsqueeze(0)

    # Convert base quaternion to world convention
    base_quat_world = math_utils.convert_camera_frame_orientation_convention(
        base_quat, origin=camera.cfg.offset.convention, target="world"
    )

    # Combine base offset with randomized offset
    # Position: add randomized offset to base position
    randomized_pos = base_pos.unsqueeze(0) + torch.stack([pos_x, pos_y, pos_z], dim=1)

    # Orientation: combine base quaternion with randomized quaternion
    randomized_quat_world = math_utils.quat_mul(base_quat_world.repeat(num_envs, 1), quats)

    # Update position offset
    camera._offset_pos[env_ids] = randomized_pos

    # Update orientation offset (already in world convention)
    camera._offset_quat[env_ids] = randomized_quat_world


def randomize_camera_intrinsics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    focal_length_range: tuple[float, float],
    horizontal_aperture_range: tuple[float, float] | None = None,
    vertical_aperture_range: tuple[float, float] | None = None,
    aperture_aspect_ratio_h_over_v_range: tuple[float, float] | None = None,
    horizontal_aperture_offset_range: tuple[float, float] | None = None,
    vertical_aperture_offset_range: tuple[float, float] | None = None,
):
    """Randomize pinhole intrinsics (focal length, apertures, optional offsets).

    ``vertical_aperture_range`` and ``aperture_aspect_ratio_h_over_v_range`` are
    mutually exclusive; if both None, vertical aperture matches image aspect ratio.
    """
    # Get the camera sensor
    camera = env.scene[sensor_cfg.name]

    # Check if camera supports intrinsic matrix setting
    if not hasattr(camera, "set_intrinsic_matrices"):
        raise AttributeError(
            f"Camera sensor '{sensor_cfg.name}' does not support intrinsic randomization. "
            "Expected MultiMeshRayCasterCamera with set_intrinsic_matrices method."
        )

    num_envs = len(env_ids)
    device = camera.device

    if vertical_aperture_range is not None and aperture_aspect_ratio_h_over_v_range is not None:
        raise ValueError(
            "Pass at most one of vertical_aperture_range and aperture_aspect_ratio_h_over_v_range."
        )

    # Get base values from config
    pattern_cfg = camera.cfg.pattern_cfg
    base_focal_length = pattern_cfg.focal_length
    base_horizontal_aperture = pattern_cfg.horizontal_aperture
    base_vertical_aperture = (
        pattern_cfg.vertical_aperture
        if pattern_cfg.vertical_aperture is not None
        else base_horizontal_aperture * pattern_cfg.height / pattern_cfg.width
    )
    base_horizontal_aperture_offset = (
        pattern_cfg.horizontal_aperture_offset if hasattr(pattern_cfg, "horizontal_aperture_offset") else 0.0
    )
    base_vertical_aperture_offset = (
        pattern_cfg.vertical_aperture_offset if hasattr(pattern_cfg, "vertical_aperture_offset") else 0.0
    )

    # Sample random focal length
    focal_lengths = torch.empty(num_envs, device=device).uniform_(*focal_length_range)

    # Sample random horizontal aperture
    if horizontal_aperture_range is not None:
        horizontal_apertures = torch.empty(num_envs, device=device).uniform_(*horizontal_aperture_range)
    else:
        horizontal_apertures = torch.full((num_envs,), base_horizontal_aperture, device=device)

    # Sample random vertical aperture (explicit range, aspect-ratio sampling, or square pixels)
    if vertical_aperture_range is not None:
        vertical_apertures = torch.empty(num_envs, device=device).uniform_(*vertical_aperture_range)
    elif aperture_aspect_ratio_h_over_v_range is not None:
        lo, hi = float(aperture_aspect_ratio_h_over_v_range[0]), float(aperture_aspect_ratio_h_over_v_range[1])
        if lo <= 0.0 or hi <= 0.0 or lo > hi:
            raise ValueError(
                f"aperture_aspect_ratio_h_over_v_range must be positive min/max with min <= max; got {(lo, hi)}."
            )
        aspect_ratios = torch.empty(num_envs, device=device).uniform_(lo, hi)
        vertical_apertures = horizontal_apertures / aspect_ratios
    else:
        # Match image aspect ratio so pixels are square in the pinhole model
        vertical_apertures = horizontal_apertures * pattern_cfg.height / pattern_cfg.width

    # Sample random aperture offsets (optional)
    if horizontal_aperture_offset_range is not None:
        horizontal_aperture_offsets = torch.empty(num_envs, device=device).uniform_(*horizontal_aperture_offset_range)
    else:
        horizontal_aperture_offsets = torch.full((num_envs,), base_horizontal_aperture_offset, device=device)

    if vertical_aperture_offset_range is not None:
        vertical_aperture_offsets = torch.empty(num_envs, device=device).uniform_(*vertical_aperture_offset_range)
    else:
        vertical_aperture_offsets = torch.full((num_envs,), base_vertical_aperture_offset, device=device)

    # Compute intrinsic matrices for each environment
    width = pattern_cfg.width
    height = pattern_cfg.height
    intrinsic_matrices = torch.zeros((num_envs, 3, 3), device=device)

    for i in range(num_envs):
        # Compute focal lengths in pixels
        f_x = width * focal_lengths[i] / horizontal_apertures[i]
        f_y = height * focal_lengths[i] / vertical_apertures[i]

        # Compute principal point (center + offset)
        c_x = horizontal_aperture_offsets[i] * f_x + width / 2
        c_y = vertical_aperture_offsets[i] * f_y + height / 2

        # Build intrinsic matrix
        intrinsic_matrices[i, 0, 0] = f_x
        intrinsic_matrices[i, 0, 2] = c_x
        intrinsic_matrices[i, 1, 1] = f_y
        intrinsic_matrices[i, 1, 2] = c_y
        intrinsic_matrices[i, 2, 2] = 1.0

    # Ray-caster uses matrices; mean focal_length is mainly for USD reference.
    mean_focal_length = focal_lengths.mean().item()
    camera.set_intrinsic_matrices(intrinsic_matrices, focal_length=mean_focal_length, env_ids=env_ids)


def randomize_camera_pose_and_intrinsics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
    orientation_range: dict[str, tuple[float, float]],
    focal_length_range: tuple[float, float],
    horizontal_aperture_range: tuple[float, float] | None = None,
    vertical_aperture_range: tuple[float, float] | None = None,
    aperture_aspect_ratio_h_over_v_range: tuple[float, float] | None = None,
    horizontal_aperture_offset_range: tuple[float, float] | None = None,
    vertical_aperture_offset_range: tuple[float, float] | None = None,
):
    """Convenience event: ``randomize_camera_pose`` then ``randomize_camera_intrinsics``."""
    # Randomize camera pose
    randomize_camera_pose(
        env=env,
        env_ids=env_ids,
        sensor_cfg=sensor_cfg,
        position_range=position_range,
        orientation_range=orientation_range,
    )

    # Randomize camera intrinsics
    randomize_camera_intrinsics(
        env=env,
        env_ids=env_ids,
        sensor_cfg=sensor_cfg,
        focal_length_range=focal_length_range,
        horizontal_aperture_range=horizontal_aperture_range,
        vertical_aperture_range=vertical_aperture_range,
        aperture_aspect_ratio_h_over_v_range=aperture_aspect_ratio_h_over_v_range,
        horizontal_aperture_offset_range=horizontal_aperture_offset_range,
        vertical_aperture_offset_range=vertical_aperture_offset_range,
        )
    
    
def _to_env_ids(env: ManagerBasedRLEnv, env_ids, device: torch.device) -> torch.Tensor:
    """Normalize env_ids to a 1D torch.long tensor on target device."""
    if isinstance(env_ids, slice):
        if env_ids == slice(None):
            return torch.arange(env.num_envs, device=device, dtype=torch.long)
        return torch.arange(env.num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, list):
        return torch.tensor(env_ids, device=device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.tensor(env_ids, device=device, dtype=torch.long)


def randomize_num_active_cameras(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | list[int] | slice,
    max_num_cameras: int,
    min_num_active_cameras: int = 0,
):
    """Sample number of active cameras per reset env.

    Cameras are always spawned 0..max_num_cameras-1.
    Active cameras are deterministic prefix: [0, ..., n_active-1].
    """
    if max_num_cameras < 1:
        raise ValueError(f"max_num_cameras must be >= 1, got {max_num_cameras}")
    if not (0 <= min_num_active_cameras <= max_num_cameras):
        raise ValueError(
            f"min_num_active_cameras must be in [0, {max_num_cameras}], got {min_num_active_cameras}"
        )

    device = env.device
    env_ids = _to_env_ids(env, env_ids, device=device)

    # Persistent buffers
    if (not hasattr(env, "_active_camera_mask")
            or env._active_camera_mask.shape != (env.num_envs, max_num_cameras)):
        env._active_camera_mask = torch.zeros(
            (env.num_envs, max_num_cameras), device=device, dtype=torch.bool
        )

    if (not hasattr(env, "_num_active_cameras")
            or env._num_active_cameras.shape != (env.num_envs,)):
        env._num_active_cameras = torch.zeros(
            env.num_envs, device=device, dtype=torch.long
        )

    # Sample active count per reset env in [min_num_active_cameras, max_num_cameras]
    n_active = torch.randint(
        low=min_num_active_cameras,
        high=max_num_cameras + 1,
        size=(len(env_ids),),
        device=device,
        dtype=torch.long,
    )
    env._num_active_cameras[env_ids] = n_active

    # Build deterministic active mask: camera index < n_active
    env._active_camera_mask[env_ids] = False
    cam_ids = torch.arange(max_num_cameras, device=device).view(1, -1)  # (1, C)
    env._active_camera_mask[env_ids] = cam_ids < n_active.view(-1, 1)    # (R, C)


def set_fixed_num_active_cameras(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | list[int] | slice,
    max_num_cameras: int,
    min_num_active_cameras: int = 0,
    fixed_num_active_cameras: int | None = None,
) -> None:
    """Reset event: set a fixed active-camera prefix mask."""
    del min_num_active_cameras
    fixed_n = int(fixed_num_active_cameras if fixed_num_active_cameras is not None else 1)

    device = env.device
    env_ids_t = _to_env_ids(env, env_ids, device=device)

    if (
        not hasattr(env, "_active_camera_mask")
        or env._active_camera_mask.shape != (env.num_envs, max_num_cameras)
    ):
        env._active_camera_mask = torch.zeros(
            (env.num_envs, max_num_cameras), device=device, dtype=torch.bool
        )
    if not hasattr(env, "_num_active_cameras") or env._num_active_cameras.shape != (env.num_envs,):
        env._num_active_cameras = torch.zeros(env.num_envs, device=device, dtype=torch.long)

    env._num_active_cameras[env_ids_t] = fixed_n
    cam_ids = torch.arange(max_num_cameras, device=device).view(1, -1)
    env._active_camera_mask[env_ids_t] = cam_ids < fixed_n


def configure_active_cameras(env_cfg, active_cameras: int | str) -> None:
    """Pin or randomize active camera count for multi-camera eval/viz CLI."""
    import inspect

    active_spec = str(active_cameras).strip().lower()
    if not hasattr(env_cfg, "max_num_cameras"):
        return

    max_cams = int(env_cfg.max_num_cameras)
    if active_spec == "random":
        min_n = 1
        max_active_n = max_cams
    else:
        min_n = int(active_spec)
        max_active_n = min_n
        if min_n not in {1, 2, 3, 4} or min_n > max_cams:
            raise ValueError(f"active_cameras must be random or 1..{max_cams}, got {active_cameras}")

    env_cfg.min_num_active_cameras = min_n
    if hasattr(env_cfg, "fixed_num_active_cameras"):
        env_cfg.fixed_num_active_cameras = min_n

    event = getattr(getattr(env_cfg, "events", None), "randomize_num_active_cameras", None)
    if event is None:
        return

    event.params["min_num_active_cameras"] = min_n
    event.params["max_num_cameras"] = max_cams
    randomize_fn = event.func
    if "max_num_active_cameras" in inspect.signature(randomize_fn).parameters:
        event.params["max_num_active_cameras"] = max_active_n
    elif active_spec != "random":
        event.params["fixed_num_active_cameras"] = int(max_active_n)
        event.func = set_fixed_num_active_cameras
