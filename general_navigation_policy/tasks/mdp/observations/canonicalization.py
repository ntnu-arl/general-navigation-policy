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
"""Canonicalized 360° depth observations (perspective → body-frame spherical map)."""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCamera
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape, record_dtype

from general_navigation_policy.tasks.mdp.camera_buffers import as_torch, get_camera_offset
from general_navigation_policy.tasks.mdp.observations.observation_noise import (
    _apply_depth_noise_nhwc,
    _apply_intrinsics_noise,
    _apply_pose_noise,
    _normalize_depth_noise_params,
    _normalize_intrinsics_noise_params,
    _normalize_pose_noise_params,
)
import isaaclab.utils.math as math_utils


@generic_io_descriptor(
    units="m", axes=["azimuth", "elevation"], observation_type="SensorState", on_inspect=[record_shape, record_dtype]
)
def canonicalized_360_depth(
    env: ManagerBasedEnv,
    camera_pos_b: torch.Tensor,  
    camera_quat_b: torch.Tensor, 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    azimuth_resolution: int = 1440,
    elevation_resolution: int = 720,
    max_range: float = 10.0,
    apply_depth_noise: bool = False,
    depth_noise_params: dict[str, Any] | None = None,
    apply_intrinsics_noise: bool = False,
    intrinsics_noise_params: dict[str, Any] | None = None,
    apply_pose_noise: bool = False,
    pose_noise_params: dict[str, Any] | None = None,
    use_inverse_depth: bool = False,
    inverse_depth_near_m: float = 0.25,
) -> torch.Tensor:
    """Body-frame spherical depth map from one perspective camera.

    Returns ``(N, az, el, 2)``: channel 0 normalized depth (or inverse depth),
    channel 1 FOV coverage mask. Caches metric depth / validity for rewards.
    """
    camera: RayCasterCamera = env.scene[sensor_cfg.name]
    # robot = env.scene[robot_cfg.name]

    depth_image = as_torch(camera.data.output["distance_to_image_plane"])  # (num_envs, H, W, 1)
    depth_image, resolved_noise_params = _apply_depth_noise_nhwc(
        env=env,
        depth_image_nhwc=depth_image,
        sensor_cfg=sensor_cfg,
        apply_depth_noise=apply_depth_noise,
        depth_noise_params=depth_noise_params,
    )
    num_envs, height, width, _ = depth_image.shape

    intrinsics = as_torch(camera.data.intrinsic_matrices)  # (num_envs, 3, 3)
    intrinsics, _ = _apply_intrinsics_noise(
        intrinsics=intrinsics,
        apply_intrinsics_noise=apply_intrinsics_noise,
        intrinsics_noise_params=intrinsics_noise_params,
    )
    camera_pos_b, camera_quat_b, _ = _apply_pose_noise(
        camera_pos_b=camera_pos_b,
        camera_quat_b=camera_quat_b,
        apply_pose_noise=apply_pose_noise,
        pose_noise_params=pose_noise_params,
    )

    # Ray-caster depth is distance_to_image_plane (camera X-forward).
    depth_flat = depth_image.squeeze(-1)  # (N, H, W)

    fx = intrinsics[:, 0, 0].view(num_envs, 1, 1)
    fy = intrinsics[:, 1, 1].view(num_envs, 1, 1)
    cx = intrinsics[:, 0, 2].view(num_envs, 1, 1)
    cy = intrinsics[:, 1, 2].view(num_envs, 1, 1)

    u = torch.arange(width, device=env.device, dtype=depth_flat.dtype).view(1, 1, width).expand(num_envs, height, width)
    v = torch.arange(height, device=env.device, dtype=depth_flat.dtype).view(1, height, 1).expand(num_envs, height, width)

    X = depth_flat
    Y = -((u - cx) / torch.clamp(fx, min=1e-8)) * X
    Z = -((v - cy) / torch.clamp(fy, min=1e-8)) * X

    points_camera = torch.stack([X, Y, Z], dim=-1).view(num_envs, height * width, 3)  # (N, H*W, 3)
    
    points_robot_b = math_utils.quat_apply(
        camera_quat_b.unsqueeze(1).repeat(1, height * width, 1),
        points_camera
    ) + camera_pos_b.unsqueeze(1)

    # When depth is 0/NaN, 3D points collapse; use ray through pixel for binning.
    dir_cam = torch.stack(
        [
            torch.ones_like(depth_flat),
            -((u - cx) / torch.clamp(fx, min=1e-8)),
            -((v - cy) / torch.clamp(fy, min=1e-8)),
        ],
        dim=-1,
    )
    dir_cam = dir_cam / torch.clamp(torch.norm(dir_cam, dim=-1, keepdim=True), min=1e-8)
    dir_robot_b = math_utils.quat_apply(
        camera_quat_b.unsqueeze(1).repeat(1, height * width, 1),
        dir_cam.view(num_envs, height * width, 3),
    )
    max_range_t = torch.tensor(float(max_range), device=env.device, dtype=dir_robot_b.dtype)
    anchor_b = camera_pos_b.unsqueeze(1) + max_range_t * dir_robot_b
    ax_a, ay_a, az_a = anchor_b[..., 0], anchor_b[..., 1], anchor_b[..., 2]
    azimuth_anchor = (torch.atan2(ay_a, ax_a) + 2 * torch.pi) % (2 * torch.pi)
    elevation_anchor = torch.atan2(az_a, torch.sqrt(ax_a * ax_a + ay_a * ay_a + 1e-20)) + torch.pi / 2
    
    x = points_robot_b[:, :, 0]  # (num_envs, H*W)
    y = points_robot_b[:, :, 1]  # (num_envs, H*W)
    z = points_robot_b[:, :, 2]  # (num_envs, H*W)

    distance = torch.sqrt(x**2 + y**2 + z**2)  # (num_envs, H*W)

    azimuth = torch.atan2(y, x)  # (num_envs, H*W), range [-π, π]
    azimuth = (azimuth + 2 * torch.pi) % (2 * torch.pi)  # [0, 2π]
    elevation = torch.atan2(z, torch.sqrt(x**2 + y**2))
    elevation = elevation + torch.pi / 2  # [0, π]

    depth_flat_reshaped = depth_flat.view(num_envs, height * width)  # (num_envs, H*W)
    finite_depth = ~torch.isnan(depth_flat_reshaped) & ~torch.isinf(depth_flat_reshaped)
    # No-return / beyond clip: zero-clipped or saturated at max_range.
    is_no_hit = (
        ~finite_depth
        | (finite_depth & (depth_flat_reshaped <= 1e-6))
        | (depth_flat_reshaped >= (max_range - 1e-3))
    )
    angle_ok = torch.isfinite(azimuth) & torch.isfinite(elevation)
    use_anchor_angles = is_no_hit | (~angle_ok)
    azimuth = torch.where(use_anchor_angles, azimuth_anchor, azimuth)
    elevation = torch.where(use_anchor_angles, elevation_anchor, elevation)

    min_valid_depth = max(
        float(resolved_noise_params["min_depth"]) if apply_depth_noise else 0.0,
        1e-6,
    )
    is_invalid_depth = (~finite_depth) | (depth_flat_reshaped <= min_valid_depth)
    observed_invalid_mask = is_no_hit | is_invalid_depth
    depth_valid = finite_depth & (depth_flat_reshaped > min_valid_depth) & (~is_no_hit)  # (num_envs, H*W)

    distance_valid = (
        ~torch.isnan(distance) &
        ~torch.isinf(distance) &
        (distance > 1e-6)
    )  # (num_envs, H*W)

    distance_to_use = torch.where(
        distance_valid,
        distance,
        depth_flat_reshaped
    )  # (num_envs, H*W)

    valid_mask = depth_valid  # (num_envs, H*W)

    # Observed invalids (no-hit / noise) get far depth but still cover the bin.
    distance_to_use = torch.where(
        observed_invalid_mask,
        max_range,
        distance_to_use
    )  # (num_envs, H*W)

    # Metric cache for rewards before obs remapping (norm / inverse-depth).
    metric_distance_to_use = torch.clamp(distance_to_use, 0.0, max_range)
    distance_to_use = torch.clamp(metric_distance_to_use / max_range, 0.0, 1.0)

    device = env.device
    depth_grid = torch.full(
        (num_envs, azimuth_resolution, elevation_resolution),
        0.0,  # Invalid pixels (not covered by camera) get 0. to distinguish from valid max_range
        dtype=torch.float32,
        device=device
    )
    validity_grid = torch.zeros(
        (num_envs, azimuth_resolution, elevation_resolution),
        dtype=torch.float32,
        device=device
    )
    
    azimuth_idx = (azimuth / (2 * torch.pi) * azimuth_resolution).long()  # (num_envs, H*W)
    azimuth_idx = torch.clamp(azimuth_idx, 0, azimuth_resolution - 1)
    elevation_idx = (elevation / torch.pi * elevation_resolution).long()  # (num_envs, H*W)
    elevation_idx = torch.clamp(elevation_idx, 0, elevation_resolution - 1)

    num_pixels = height * width
    grid_size = azimuth_resolution * elevation_resolution
    valid_mask_flat = valid_mask.view(-1)  # (num_envs * H*W,)
    valid_indices = torch.nonzero(valid_mask_flat, as_tuple=False).squeeze(-1)  # (N_total_valid,)

    if valid_indices.numel() > 0:
        env_ids_flat = valid_indices // num_pixels  # (N_total_valid,)
        pixel_ids_flat = valid_indices % num_pixels  # (N_total_valid,)
        azimuth_idx_flat = azimuth_idx.view(-1)[valid_indices]  # (N_total_valid,)
        elevation_idx_flat = elevation_idx.view(-1)[valid_indices]  # (N_total_valid,)
        distance_flat = distance_to_use.view(-1)[valid_indices]  # (N_total_valid,)
        grid_flat_idx_local = azimuth_idx_flat * elevation_resolution + elevation_idx_flat  # (N_total_valid,)
        env_offsets = env_ids_flat * grid_size  # (N_total_valid,)
        grid_flat_idx_global = grid_flat_idx_local + env_offsets  # (N_total_valid,)
        depth_grid_flat = depth_grid.view(-1)  # (num_envs * azimuth_res * elev_res,)
        depth_grid_flat.scatter_reduce_(
            dim=0,
            index=grid_flat_idx_global,
            src=distance_flat,
            reduce="amin",
            include_self=False
        )

    # No-hit / invalid: write far where no closer sample was fused.
    observed_invalid_flat = observed_invalid_mask.view(-1)  # (num_envs * H*W,)
    observed_invalid_indices = torch.nonzero(observed_invalid_flat, as_tuple=False).squeeze(-1)
    if observed_invalid_indices.numel() > 0:
        invalid_env_ids_flat = observed_invalid_indices // num_pixels
        invalid_azimuth_idx_flat = azimuth_idx.view(-1)[observed_invalid_indices]
        invalid_elevation_idx_flat = elevation_idx.view(-1)[observed_invalid_indices]
        invalid_grid_flat_idx_local = (
            invalid_azimuth_idx_flat * elevation_resolution + invalid_elevation_idx_flat
        )
        invalid_env_offsets = invalid_env_ids_flat * grid_size
        invalid_grid_flat_idx_global = invalid_grid_flat_idx_local + invalid_env_offsets

        depth_grid_flat = depth_grid.view(-1)
        # Do not overwrite bins that already received a closer metric hit.
        depth_still_empty = depth_grid_flat[invalid_grid_flat_idx_global] < 1e-6
        if torch.any(depth_still_empty):
            idx_far = invalid_grid_flat_idx_global[depth_still_empty]
            depth_grid_flat[idx_far] = 1.0

    # Coverage: every image pixel marks its spherical bin (no-hit still counts).
    grid_local_flat = (azimuth_idx * elevation_resolution + elevation_idx).reshape(-1)
    env_ids_flat = torch.arange(num_envs, device=device, dtype=torch.long).repeat_interleave(num_pixels)
    global_cov = grid_local_flat + env_ids_flat * grid_size
    validity_flat = validity_grid.view(-1)
    ones_cov = torch.ones(global_cov.shape[0], device=device, dtype=validity_grid.dtype)
    validity_flat.scatter_reduce_(0, global_cov, ones_cov, reduce="amax", include_self=False)

    # Covered bins must not stay at 0 (breaks multi-cam amin fusion).
    depth_grid = torch.where(
        (validity_grid > 0.5) & (depth_grid < 1e-6),
        torch.ones_like(depth_grid),
        depth_grid,
    )

    metric_depth_grid = depth_grid * max_range

    # Optional inverse-depth obs: far→0, near→1; uncovered stays 0.
    if use_inverse_depth:
        inv_depth_norm = _normalized_inverse_depth(
            metric_depth_grid, max_range, inverse_depth_near_m
        )
        depth_grid = torch.where(
            validity_grid > 0.0,
            inv_depth_norm,
            torch.zeros_like(depth_grid),
        )

    output = torch.stack([depth_grid, validity_grid], dim=-1)  # (num_envs, azimuth_res, elev_res, 2)

    if not hasattr(env, "_cached_validity_step") or env._cached_validity_step != env.common_step_counter:
        env._cached_validity_mask = validity_grid
        env._cached_metric_depth = metric_depth_grid
        env._cached_depth_is_inverse = use_inverse_depth
        env._cached_depth_max_range = float(max_range)
        env._cached_validity_step = env.common_step_counter
    
    return output

def canonicalized_360_depth_multi_camera(
    env: ManagerBasedEnv,
    max_num_cameras: int,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    azimuth_resolution: int = 90,
    elevation_resolution: int = 45,
    max_range: float = 10.0,
    obs_update_period_s: float | None = None,
    obs_decimation_steps: int | None = None,
    apply_depth_noise: bool = False,
    depth_noise_params: dict[str, Any] | None = None,
    apply_intrinsics_noise: bool = False,
    intrinsics_noise_params: dict[str, Any] | None = None,
    apply_pose_noise: bool = False,
    pose_noise_params: dict[str, Any] | None = None,
    use_inverse_depth: bool = False,
    inverse_depth_near_m: float = 0.25,
) -> torch.Tensor:
    """Fuse spherical depth over active cameras (``depth_camera_0..N-1``).

    Amin depth / amax validity over cameras gated by ``_active_camera_mask``.
    Supports observation decimation via ``obs_decimation_steps`` / ``obs_update_period_s``.
    """
    if max_num_cameras < 1:
        raise ValueError(f"max_num_cameras must be >= 1, got {max_num_cameras}")

    N = env.num_envs
    device = env.device
    step_count = int(getattr(env, "common_step_counter", 0))

    # Decimation priority: explicit steps > seconds > first camera update_period.
    if obs_decimation_steps is not None:
        obs_steps = max(1, int(obs_decimation_steps))
    else:
        period_s = obs_update_period_s
        if period_s is None:
            first_camera = env.scene["depth_camera_0"] if "depth_camera_0" in env.scene.keys() else None
            if first_camera is not None and hasattr(first_camera, "cfg"):
                period_s = float(first_camera.cfg.update_period)
            else:
                period_s = float(getattr(env, "step_dt", 0.0))
        step_dt = float(getattr(env, "step_dt", 0.0))
        if step_dt > 0.0:
            obs_steps = max(1, int(round(period_s / step_dt)))
        else:
            obs_steps = 1

    has_resets = bool(hasattr(env, "reset_buf") and torch.any(env.reset_buf))
    noise_params_for_signature = _normalize_depth_noise_params(depth_noise_params)
    intrinsics_noise_params_for_signature = _normalize_intrinsics_noise_params(intrinsics_noise_params)
    pose_noise_params_for_signature = _normalize_pose_noise_params(pose_noise_params)
    cache_signature = (
        max_num_cameras,
        azimuth_resolution,
        elevation_resolution,
        float(max_range),
        obs_steps,
        N,
        bool(apply_depth_noise),
        repr(noise_params_for_signature),
        bool(apply_intrinsics_noise),
        repr(intrinsics_noise_params_for_signature),
        bool(apply_pose_noise),
        repr(pose_noise_params_for_signature),
        bool(use_inverse_depth),
        float(inverse_depth_near_m),
    )
    if (
        not has_resets
        and hasattr(env, "_cached_obs_360_multi")
        and getattr(env, "_cached_obs_360_multi_signature", None) == cache_signature
        and (step_count - int(getattr(env, "_cached_obs_360_multi_step", -10**9))) < obs_steps
    ):
        cached_output = env._cached_obs_360_multi
        env._cached_validity_mask = cached_output[..., 1]
        env._cached_validity_step = step_count
        return cached_output

    # Before first reset: treat all cameras as active.
    if (not hasattr(env, "_active_camera_mask")
            or env._active_camera_mask.shape != (N, max_num_cameras)):
        active_mask = torch.ones((N, max_num_cameras), device=device, dtype=torch.bool)
    else:
        active_mask = env._active_camera_mask

    fused_depth = torch.ones(
        (N, azimuth_resolution, elevation_resolution),
        device=device,
        dtype=torch.float32,
    )  # far init for amin fusion
    fused_valid = torch.zeros_like(fused_depth)

    for cam_idx in range(max_num_cameras):
        sensor_name = f"depth_camera_{cam_idx}"
        camera_pos_b, camera_quat_b = _camera_extrinsics_in_robot_frame(env, sensor_name)
        
        cam_out = canonicalized_360_depth(
            env=env,
            camera_pos_b=camera_pos_b,
            camera_quat_b=camera_quat_b,
            sensor_cfg=SceneEntityCfg(sensor_name),
            robot_cfg=robot_cfg,
            azimuth_resolution=azimuth_resolution,
            elevation_resolution=elevation_resolution,
            max_range=max_range,
            apply_depth_noise=apply_depth_noise,
            depth_noise_params=depth_noise_params,
            apply_intrinsics_noise=apply_intrinsics_noise,
            intrinsics_noise_params=intrinsics_noise_params,
            apply_pose_noise=apply_pose_noise,
            pose_noise_params=pose_noise_params,
            use_inverse_depth=False,
            inverse_depth_near_m=inverse_depth_near_m,
        )  # (N, A, E, 2)

        cam_depth = cam_out[..., 0]   # (N, A, E)
        cam_valid = cam_out[..., 1]   # (N, A, E)

        cam_on = active_mask[:, cam_idx].view(N, 1, 1)  # bool
        cam_valid = cam_valid * cam_on.to(cam_valid.dtype)

        # Inactive / empty covered cells → far so amin does not wipe fused depth.
        cam_depth = torch.where(
            cam_valid > 0.0,
            torch.where(cam_depth < 1e-6, torch.ones_like(cam_depth), cam_depth),
            torch.ones_like(cam_depth),
        )

        fused_depth = torch.minimum(fused_depth, cam_depth)
        fused_valid = torch.maximum(fused_valid, cam_valid)

    fused_depth = torch.where(fused_valid > 0.0, fused_depth, torch.zeros_like(fused_depth))
    fused_metric_depth = fused_depth * float(max_range)

    if use_inverse_depth:
        inv_depth_norm = _normalized_inverse_depth(
            fused_metric_depth, max_range, inverse_depth_near_m
        )
        fused_depth = torch.where(
            fused_valid > 0.0,
            inv_depth_norm,
            torch.zeros_like(fused_depth),
        )

    env._cached_validity_mask = fused_valid
    env._cached_metric_depth = fused_metric_depth
    env._cached_depth_is_inverse = use_inverse_depth
    env._cached_depth_max_range = float(max_range)
    env._cached_validity_step = step_count

    output = torch.stack([fused_depth, fused_valid], dim=-1)
    env._cached_obs_360_multi = output
    env._cached_obs_360_multi_step = step_count
    env._cached_obs_360_multi_signature = cache_signature

    return output


def _camera_extrinsics_in_robot_frame(env: ManagerBasedEnv, sensor_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env camera extrinsics (position, quaternion) in robot body frame."""
    cam = env.scene[sensor_name]
    return get_camera_offset(cam)


def _normalized_inverse_depth(
    metric_depth: torch.Tensor,
    max_range: float,
    inverse_depth_near_m: float = 0.25,
) -> torch.Tensor:
    """Map metric depth (m) to normalized inverse depth: far→0, near→1."""
    far = float(max_range)
    near_m = max(float(inverse_depth_near_m), 1e-6)
    near_m = min(near_m, far)
    inv_min = 1.0 / far
    inv_max = 1.0 / near_m
    inv_den = max(inv_max - inv_min, 1e-6)
    inv_depth = 1.0 / torch.clamp(metric_depth, min=1e-6)
    return torch.clamp((inv_depth - inv_min) / inv_den, 0.0, 1.0)
