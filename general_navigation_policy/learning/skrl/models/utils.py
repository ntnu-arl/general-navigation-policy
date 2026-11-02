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
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Padding utilities
# ------------------------------


def _robot_state_dim_from_space(observation_space) -> int:
    """Infer robot_state size from observation space."""
    if hasattr(observation_space, "spaces"):
        if "robot_state" in observation_space.spaces:
            return int(observation_space.spaces["robot_state"].shape[0])
        if "policy" in observation_space.spaces:
            return int(observation_space.spaces["policy"].spaces["robot_state"].shape[0])
    raise ValueError("observation_space must have 'robot_state' or 'policy'['robot_state'].")

def _conv5_spatial_size(height: int, width: int) -> tuple[int, int]:
    """Spatial size after 5 stride-2 convs (k5s2p2 then four k3s2p1)."""
    h, w = height, width
    h = (h + 4 - 5) // 2 + 1
    w = (w + 4 - 5) // 2 + 1
    for _ in range(4):
        h = (h + 2 - 3) // 2 + 1
        w = (w + 2 - 3) // 2 + 1
    return h, w

def circular_pad_azimuth(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular-pad azimuth (H) of ``(B, C, H, W)`` range images."""
    if pad <= 0:
        return x
    return torch.cat([x[:, :, -pad:, :], x, x[:, :, :pad, :]], dim=2)


def pad_elevation(
    x: torch.Tensor,
    pad: int,
    *,
    semantic: bool,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """Pad elevation (W); semantic mode zeros depth/mask fringe channels."""
    if pad <= 0:
        return x

    if semantic:
        assert x.shape[1] >= 2, "Semantic padding requires depth+mask channels"

        depth = x[:, 0:1]
        mask  = x[:, 1:2]
        rest  = x[:, 2:] if x.shape[1] > 2 else None

        depth = F.pad(depth, (pad, pad, 0, 0), value=0.0)
        mask  = F.pad(mask,  (pad, pad, 0, 0), value=0.0)

        if rest is not None:
            rest = F.pad(rest, (pad, pad, 0, 0), mode=padding_mode)
            return torch.cat([depth, mask, rest], dim=1)

        return torch.cat([depth, mask], dim=1)

    return F.pad(x, (pad, pad, 0, 0), mode=padding_mode)


class CircularAzimuthConv2d(nn.Module):
    """Conv2d with circular azimuth (H) pad and configurable elevation (W) pad."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        *,
        semantic_elevation: bool = False,
        elevation_padding_mode: str = "reflect",
    ):
        super().__init__()

        self.pad_h = (kernel_size // 2) * dilation
        self.pad_w = (kernel_size // 2) * dilation
        self.semantic_elevation = semantic_elevation
        self.elevation_padding_mode = elevation_padding_mode

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = circular_pad_azimuth(x, self.pad_h)
        x = pad_elevation(
            x,
            self.pad_w,
            semantic=self.semantic_elevation,
            padding_mode=self.elevation_padding_mode,
        )
        return self.conv(x)
