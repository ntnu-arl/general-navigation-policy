"""Reward functions for general navigation policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from general_navigation_policy.tasks.mdp.camera_buffers import as_torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_in_observed_direction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    azimuth_resolution: int = 90,
    elevation_resolution: int = 45,
    window_half_size: int = 5,
    action_name: str = "velocity_commands",
    use_validity_mask: bool = True,
) -> torch.Tensor:
    """Reward commanded speed scaled by observation density around the commanded direction.

    Converts the robot's commanded body-frame translational velocity to azimuth/elevation
    in the same coordinate frame as the canonicalized 360° range image, then computes the
    fraction of valid pixels in a (2*window_half_size+1)² square centered on that direction.

        reward_i = commanded_speed_i * (valid_pixels_in_window / total_pixels_in_window)

    The result is smooth in both commanded speed and observation coverage:
    - Commanding fast into a fully observed region -> high reward
    - Commanding fast into a partially observed region -> proportional reward
    - Commanding fast into a blind region -> near-zero reward
    - Commanding near-zero speed -> near-zero reward regardless of coverage

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the robot (kept for API compatibility).
        azimuth_resolution: Must match the observation's azimuth_resolution.
        elevation_resolution: Must match the observation's elevation_resolution.
        window_half_size: Half-size of the square window in grid cells.
            The full window is (2*h+1) x (2*h+1). With azimuth_resolution=90
            each cell spans 4 deg, so window_half_size=5 covers +/-20 deg (~40 deg total).
        action_name: Name of the navigation action term exposing processed commands
            (vx, vy, vz, yaw_rate), default "velocity_commands".
        use_validity_mask: If False (no-mask ablation), ignore FOV coverage and return
            commanded speed only (equivalent to all bins observed).

    Returns:
        (num_envs,) reward tensor >= 0.
    """
    # --- Commanded translational velocity in body frame ---
    action_term = env.action_manager.get_term(action_name)
    commands = action_term.prev_commands  # (num_envs, 4): [vx, vy, vz, yaw_rate]
    vx, vy, vz = commands[:, 0], commands[:, 1], commands[:, 2]
    speed = torch.sqrt(vx**2 + vy**2 + vz**2 + 1e-8)

    if not use_validity_mask:
        return speed

    # --- Convert commanded direction to azimuth / elevation (same as range image) ---
    vel_az = (torch.atan2(vy, vx) + 2 * torch.pi) % (2 * torch.pi)                 # [0, 2π]
    vel_el = torch.atan2(vz, torch.sqrt(vx**2 + vy**2 + 1e-8)) + torch.pi / 2      # [0, π]

    # Map to grid indices (center of the window)
    az_center = (vel_az / (2 * torch.pi) * azimuth_resolution).long()
    az_center = torch.clamp(az_center, 0, azimuth_resolution - 1)
    el_center = (vel_el / torch.pi * elevation_resolution).long()
    el_center = torch.clamp(el_center, 0, elevation_resolution - 1)

    # --- Get cached validity mask (set by canonicalized_360_depth*) ---
    validity = env._cached_validity_mask  # (num_envs, az, el)

    # --- Average validity in a square window via unfold ---
    h = window_half_size
    w = 2 * h + 1  # window side length

    # Pad azimuth circularly and elevation with zeros, then use unfold
    val_padded = torch.cat(
        [validity[:, -h:, :], validity, validity[:, :h, :]], dim=1
    )  # (N, az+2h, el)
    val_padded = F.pad(
        val_padded, (h, h), mode="constant", value=0.0
    )  # (N, az+2h, el+2h)

    # unfold both spatial dims -> (N, az, el, w, w)
    val_windows = val_padded.unfold(1, w, 1).unfold(2, w, 1)

    # Gather the window at each env's (az_center, el_center)
    N = validity.shape[0]
    env_ids = torch.arange(N, device=env.device)
    windows = val_windows[env_ids, az_center, el_center]  # (N, w, w)

    # Fraction of valid pixels in the window
    fraction_valid = windows.mean(dim=(1, 2))  # (N,)

    return speed * fraction_valid


def time_to_collision_penalty_from_canonicalized_depth(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ttc_magnitude: float = -3.0,
    ttc_exponent: float = 2.0,
    max_ttc: float = 10.0,
    eps: float = 1e-6,
    use_validity_mask: bool = True,
) -> torch.Tensor:
    """Penalize low time-to-collision (TTC) using cached canonicalized depth.

    TTC is computed over spherical bins as:
      ttc = range / max(v_along_ray, eps)  if v_along_ray > 0 else max_ttc

    where:
      - range is the metric depth (meters) cached by canonicalized depth observations
      - v_along_ray is body-frame linear velocity projected along each bin direction

    Args:
        use_validity_mask: If True, only FOV-covered bins contribute. If False (no-mask
            ablation), use depth-only gating ``depth_metric > eps`` (invalid bins are 0 m
            in the metric cache) so the explicit validity mask is unused.

    Returns:
      A negative reward in [ttc_magnitude, 0], strongest when imminent collision is likely.
      If cached depth/validity is unavailable, returns zeros.
    """
    if not hasattr(env, "_cached_metric_depth"):
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    depth_metric: torch.Tensor = env._cached_metric_depth  # (N, A, E), meters
    if depth_metric.ndim != 3:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)

    if use_validity_mask:
        if not hasattr(env, "_cached_validity_mask"):
            return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        validity: torch.Tensor = env._cached_validity_mask  # (N, A, E), {0, 1}
        if validity.ndim != 3:
            return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
        observed = validity > 0.0
    else:
        # Depth-only proxy: uncovered bins are cached as 0 m metric depth.
        observed = depth_metric > float(eps)

    asset: RigidObject = env.scene[asset_cfg.name]
    # Lab 3 ProxyArray: use explicit .torch (vec3 warp shape is (N,), torch is (N, 3)).
    vel_b = as_torch(asset.data.root_lin_vel_b)  # (N, 3)

    num_envs, az_res, el_res = depth_metric.shape
    device = depth_metric.device
    dtype = depth_metric.dtype

    # Reconstruct unit directions for each (azimuth, elevation) bin in robot body frame.
    az = (torch.arange(az_res, device=device, dtype=dtype) + 0.5) / az_res * (2.0 * torch.pi)
    el = (torch.arange(el_res, device=device, dtype=dtype) + 0.5) / el_res * torch.pi
    elev = el - (torch.pi / 2.0)  # [-pi/2, pi/2]

    cos_elev = torch.cos(elev).view(1, el_res)
    sin_elev = torch.sin(elev).view(1, el_res)
    cos_az = torch.cos(az).view(az_res, 1)
    sin_az = torch.sin(az).view(az_res, 1)

    dir_x = cos_elev * cos_az
    dir_y = cos_elev * sin_az
    dir_z = sin_elev.expand(az_res, el_res)
    dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)  # (A, E, 3)

    vel_proj = torch.sum(vel_b.view(num_envs, 1, 1, 3) * dirs.view(1, az_res, el_res, 3), dim=-1)

    approaching = (vel_proj > 0.0) & observed
    ttc = torch.where(
        approaching,
        depth_metric / torch.clamp(vel_proj, min=eps),
        torch.full_like(depth_metric, float(max_ttc)),
    )
    ttc = torch.clamp(ttc, 0.0, float(max_ttc))
    min_ttc = torch.min(ttc.view(num_envs, -1), dim=1).values

    return ttc_magnitude * torch.exp(-(min_ttc * min_ttc) * ttc_exponent)
