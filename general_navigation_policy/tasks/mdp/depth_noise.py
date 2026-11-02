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
"""Stereo-inspired depth noise (adapted from sru-navigation-sim DepthNoise)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNoise(nn.Module):
    """PyTorch module to add stereo-inspired noise to depth images."""

    def __init__(
        self,
        focal_length: float = 25.0,
        baseline: float = 0.12,
        min_depth: float = 0.25,
        max_depth: float = 10.0,
        filter_size: int = 3,
        inlier_thred_range: tuple[float, float] = (0.01, 0.05),
        prob_range: tuple[float, float] = (0.4, 0.6),
        invalid_disp: float = 1e7,
    ):
        super().__init__()
        self.focal_length = float(focal_length)
        self.baseline = float(baseline)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.invalid_disp = float(invalid_disp)
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.filter_size = int(filter_size)

        weights, substitutes = self._compute_weights(self.filter_size)
        self.register_buffer("weights", weights.view(1, 1, self.filter_size, self.filter_size))
        self.register_buffer("substitutes", substitutes.view(1, 1, self.filter_size, self.filter_size))

    def _compute_weights(self, filter_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing="ij")
        sqr_radius = x_filter**2 + y_filter**2
        sqrt_radius = torch.sqrt(sqr_radius)
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()
        return weights, substitutes

    def filter_disparity(self, disparity: torch.Tensor) -> torch.Tensor:
        b, _, h, w = disparity.shape
        device = disparity.device
        center = self.filter_size // 2

        output_disparity = torch.full_like(disparity, self.invalid_disp)

        prob = torch.rand(b, 1, 1, 1, device=device) * (self.prob_range[1] - self.prob_range[0]) + self.prob_range[0]
        random_mask = torch.rand(b, 1, h, w, device=device) < prob

        weighted_disparity = F.conv2d(disparity, self.weights, padding=center)
        differences = torch.abs(disparity - weighted_disparity)

        differences_flat = differences.view(b, -1)
        mean_diff = torch.mean(differences_flat, dim=1, keepdim=True)
        std_diff = torch.std(differences_flat, dim=1, keepdim=True) + 1e-6
        normalized_differences_flat = (differences_flat - mean_diff) / std_diff
        normalized_differences = normalized_differences_flat.view_as(differences)

        threshold = (
            torch.rand(b, 1, 1, 1, device=device) * (self.inlier_thred_range[1] - self.inlier_thred_range[0])
            + self.inlier_thred_range[0]
        )
        update_mask = (normalized_differences < threshold) & random_mask

        disparity = torch.round(disparity * 32.0) / 32.0
        output_disparity = torch.where(update_mask, disparity, output_disparity)

        filled_values = F.conv2d(update_mask.float() * disparity, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        average_filled_values = filled_values / counts
        output_disparity = torch.where(counts >= 1, average_filled_values, output_disparity)
        return output_disparity

    def forward(
        self,
        depth: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        if depth.dim() != 4 or depth.shape[1] != 1:
            raise ValueError("Input depth tensor must have shape (B, 1, H, W).")

        # Track invalid-vs-far input semantics:
        # - invalid: NaN, -Inf, and non-positive values -> output 0.0
        # - far/no-hit: +Inf or > max_depth -> output max_depth (valid saturation)
        invalid_input_mask = torch.isnan(depth) | torch.isneginf(depth) | (depth <= 0.0)
        far_input_mask = torch.isposinf(depth) | (depth > self.max_depth)

        # Build a safe depth tensor for arithmetic while preserving invalid mask.
        safe_depth = depth.clone()
        safe_depth = torch.nan_to_num(
            safe_depth,
            nan=1.0 / self.invalid_disp,
            posinf=self.max_depth,
            neginf=1.0 / self.invalid_disp,
        )

        if add_noise:
            safe_depth = torch.clamp(safe_depth, min=1.0 / self.invalid_disp)
            disparity = self.focal_length * self.baseline / safe_depth
            filtered_disparity = self.filter_disparity(disparity)
            output_depth = self.focal_length * self.baseline / filtered_disparity
        else:
            output_depth = safe_depth

        # Depth outside measurable range:
        # - too near values are invalid
        # - too far values are valid but saturated at max depth
        output_depth[output_depth < self.min_depth] = 0.0
        output_depth = torch.clamp(output_depth, max=self.max_depth)

        # Preserve far/no-hit semantics and restore true invalids.
        output_depth[far_input_mask] = self.max_depth
        output_depth[invalid_input_mask] = 0.0
        return output_depth
