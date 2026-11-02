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
"""Observation-time depth / intrinsics / pose noise helpers."""

from __future__ import annotations

from typing import Any

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from general_navigation_policy.tasks.mdp.camera_buffers import as_torch
from general_navigation_policy.tasks.mdp.depth_noise import DepthNoise


_DEPTH_NOISE_MODELS: dict[str, DepthNoise] = {}


def _normalize_depth_noise_params(depth_noise_params: dict[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "focal_length": 25.0,
        "baseline": 0.12,
        "min_depth": 0.25,
        "max_depth": 10.0,
        "filter_size": 3,
        "inlier_thred_range": (0.01, 0.05),
        "prob_range": (0.4, 0.6),
        "invalid_disp": 1e7,
    }
    if depth_noise_params is None:
        return defaults
    merged = defaults.copy()
    merged.update(depth_noise_params)
    return merged


def _depth_noise_model_key(sensor_name: str, params: dict[str, Any]) -> str:
    key_values = (
        sensor_name,
        float(params["focal_length"]),
        float(params["baseline"]),
        float(params["min_depth"]),
        float(params["max_depth"]),
        int(params["filter_size"]),
        tuple(params["inlier_thred_range"]),
        tuple(params["prob_range"]),
        float(params["invalid_disp"]),
    )
    return repr(key_values)

def _apply_depth_noise_nhwc(
    env: ManagerBasedEnv,
    depth_image_nhwc: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    apply_depth_noise: bool,
    depth_noise_params: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply depth noise to an NHWC depth tensor and return params used."""
    params = _normalize_depth_noise_params(depth_noise_params)
    if not apply_depth_noise:
        return depth_image_nhwc, params

    model_key = _depth_noise_model_key(sensor_cfg.name, params)
    if model_key not in _DEPTH_NOISE_MODELS:
        _DEPTH_NOISE_MODELS[model_key] = DepthNoise(
            focal_length=float(params["focal_length"]),
            baseline=float(params["baseline"]),
            min_depth=float(params["min_depth"]),
            max_depth=float(params["max_depth"]),
            filter_size=int(params["filter_size"]),
            inlier_thred_range=tuple(params["inlier_thred_range"]),
            prob_range=tuple(params["prob_range"]),
            invalid_disp=float(params["invalid_disp"]),
        ).to(device=env.device)

    model = _DEPTH_NOISE_MODELS[model_key]
    depth_image_nhwc = as_torch(depth_image_nhwc)
    model = model.to(device=depth_image_nhwc.device, dtype=depth_image_nhwc.dtype)

    depth_nchw = depth_image_nhwc.permute(0, 3, 1, 2)
    noisy_nchw = model(depth_nchw, add_noise=True)
    return noisy_nchw.permute(0, 2, 3, 1), params


def _normalize_intrinsics_noise_params(intrinsics_noise_params: dict[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "fx_scale_range": (0.99, 1.01),
        "fy_scale_range": (0.99, 1.01),
        "cx_offset_px_range": (-1.0, 1.0),
        "cy_offset_px_range": (-1.0, 1.0),
        "min_focal_px": 1e-6,
        # "uniform_range": U[lo, hi]; "fixed_magnitude": random endpoint {lo, hi}.
        "sample_mode": "uniform_range",
    }
    if intrinsics_noise_params is None:
        return defaults
    merged = defaults.copy()
    merged.update(intrinsics_noise_params)
    return merged


def _sample_from_range(
    lo: float,
    hi: float,
    num_envs: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_mode: str,
) -> torch.Tensor:
    """Sample per-env scalars from ``[lo, hi]`` (uniform or endpoint ``fixed_magnitude``)."""
    lo_f = float(lo)
    hi_f = float(hi)
    if abs(hi_f - lo_f) < 1e-12:
        return torch.full((num_envs,), lo_f, device=device, dtype=dtype)
    mode = str(sample_mode).lower()
    if mode in ("fixed_magnitude", "fixed", "endpoints"):
        pick_hi = torch.randint(0, 2, (num_envs,), device=device, dtype=torch.bool)
        return torch.where(
            pick_hi,
            torch.full((num_envs,), hi_f, device=device, dtype=dtype),
            torch.full((num_envs,), lo_f, device=device, dtype=dtype),
        )
    if mode not in ("uniform_range", "uniform"):
        raise ValueError(f"Unknown noise sample_mode={sample_mode!r}")
    return torch.empty((num_envs,), device=device, dtype=dtype).uniform_(lo_f, hi_f)


def _apply_intrinsics_noise(
    intrinsics: torch.Tensor,
    apply_intrinsics_noise: bool,
    intrinsics_noise_params: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply canonicalization-time intrinsics perturbation without touching camera capture intrinsics."""
    params = _normalize_intrinsics_noise_params(intrinsics_noise_params)
    if not apply_intrinsics_noise:
        return intrinsics, params

    noisy_intrinsics = intrinsics.clone()
    num_envs = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype
    sample_mode = params.get("sample_mode", "uniform_range")

    fx_lo, fx_hi = float(params["fx_scale_range"][0]), float(params["fx_scale_range"][1])
    fy_lo, fy_hi = float(params["fy_scale_range"][0]), float(params["fy_scale_range"][1])
    fx_scale = _sample_from_range(fx_lo, fx_hi, num_envs, device=device, dtype=dtype, sample_mode=sample_mode)
    # Isotropic FOV (identical fx/fy ranges): share one draw so |s-1| matches for both axes.
    if abs(fx_lo - fy_lo) < 1e-12 and abs(fx_hi - fy_hi) < 1e-12:
        fy_scale = fx_scale
    else:
        fy_scale = _sample_from_range(fy_lo, fy_hi, num_envs, device=device, dtype=dtype, sample_mode=sample_mode)
    cx_offset = _sample_from_range(
        float(params["cx_offset_px_range"][0]),
        float(params["cx_offset_px_range"][1]),
        num_envs,
        device=device,
        dtype=dtype,
        sample_mode=sample_mode,
    )
    cy_offset = _sample_from_range(
        float(params["cy_offset_px_range"][0]),
        float(params["cy_offset_px_range"][1]),
        num_envs,
        device=device,
        dtype=dtype,
        sample_mode=sample_mode,
    )

    min_focal_px = float(params["min_focal_px"])
    noisy_intrinsics[:, 0, 0] = torch.clamp(noisy_intrinsics[:, 0, 0] * fx_scale, min=min_focal_px)
    noisy_intrinsics[:, 1, 1] = torch.clamp(noisy_intrinsics[:, 1, 1] * fy_scale, min=min_focal_px)
    noisy_intrinsics[:, 0, 2] = noisy_intrinsics[:, 0, 2] + cx_offset
    noisy_intrinsics[:, 1, 2] = noisy_intrinsics[:, 1, 2] + cy_offset
    return noisy_intrinsics, params


def _normalize_pose_noise_params(pose_noise_params: dict[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "x_m_range": (-0.01, 0.01),
        "y_m_range": (-0.01, 0.01),
        "z_m_range": (-0.005, 0.005),
        "roll_deg_range": (-1.0, 1.0),
        "pitch_deg_range": (-1.0, 1.0),
        "yaw_deg_range": (-1.0, 1.0),
        "sample_mode": "uniform_range",
    }
    if pose_noise_params is None:
        return defaults
    merged = defaults.copy()
    merged.update(pose_noise_params)
    return merged


def _apply_pose_noise(
    camera_pos_b: torch.Tensor,
    camera_quat_b: torch.Tensor,
    apply_pose_noise: bool,
    pose_noise_params: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply canonicalization-time pose perturbation without touching capture pose."""
    params = _normalize_pose_noise_params(pose_noise_params)
    if not apply_pose_noise:
        return camera_pos_b, camera_quat_b, params

    num_envs = camera_pos_b.shape[0]
    device = camera_pos_b.device
    dtype = camera_pos_b.dtype
    sample_mode = params.get("sample_mode", "uniform_range")

    pos_noise = torch.stack(
        [
            _sample_from_range(
                float(params["x_m_range"][0]),
                float(params["x_m_range"][1]),
                num_envs,
                device=device,
                dtype=dtype,
                sample_mode=sample_mode,
            ),
            _sample_from_range(
                float(params["y_m_range"][0]),
                float(params["y_m_range"][1]),
                num_envs,
                device=device,
                dtype=dtype,
                sample_mode=sample_mode,
            ),
            _sample_from_range(
                float(params["z_m_range"][0]),
                float(params["z_m_range"][1]),
                num_envs,
                device=device,
                dtype=dtype,
                sample_mode=sample_mode,
            ),
        ],
        dim=-1,
    )  # (N, 3)
    noisy_pos_b = camera_pos_b + pos_noise

    deg2rad = torch.pi / 180.0
    roll = (
        _sample_from_range(
            float(params["roll_deg_range"][0]),
            float(params["roll_deg_range"][1]),
            num_envs,
            device=device,
            dtype=dtype,
            sample_mode=sample_mode,
        )
        * deg2rad
    )
    pitch = (
        _sample_from_range(
            float(params["pitch_deg_range"][0]),
            float(params["pitch_deg_range"][1]),
            num_envs,
            device=device,
            dtype=dtype,
            sample_mode=sample_mode,
        )
        * deg2rad
    )
    yaw = (
        _sample_from_range(
            float(params["yaw_deg_range"][0]),
            float(params["yaw_deg_range"][1]),
            num_envs,
            device=device,
            dtype=dtype,
            sample_mode=sample_mode,
        )
        * deg2rad
    )
    delta_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    noisy_quat_b = math_utils.quat_mul(camera_quat_b, delta_quat)
    noisy_quat_b = noisy_quat_b / torch.clamp(noisy_quat_b.norm(dim=-1, keepdim=True), min=1e-8)
    return noisy_pos_b, noisy_quat_b, params
