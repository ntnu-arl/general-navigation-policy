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
"""Body-frame camera pose offsets (position, quaternion) as torch tensors."""

from __future__ import annotations

from typing import Any

import torch


def as_torch(x: Any) -> torch.Tensor:
    """Return a ``torch.Tensor`` view of ``x``."""
    if isinstance(x, torch.Tensor):
        return x
    torch_view = getattr(x, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view
    return torch.as_tensor(x)


def ensure_camera_offset_buffers(camera: Any) -> None:
    """Make ``camera._offset_pos`` / ``_offset_quat`` writable torch views."""
    if hasattr(camera, "_offset_pos") and hasattr(camera, "_offset_quat"):
        return

    if hasattr(camera, "_offset_pos_wp") and hasattr(camera, "_offset_quat_wp"):
        from isaaclab.utils.warp.proxy_array import ProxyArray

        camera._offset_pos = ProxyArray(camera._offset_pos_wp)
        camera._offset_quat = ProxyArray(camera._offset_quat_wp)
        return

    raise AttributeError(
        f"Camera '{getattr(getattr(camera, 'cfg', None), 'prim_path', type(camera).__name__)}' "
        "has no offset buffers (_offset_pos/_offset_quat or _offset_pos_wp/_offset_quat_wp). "
        "Expected MultiMeshRayCasterCamera."
    )


def get_camera_offset(camera: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(offset_pos, offset_quat)`` in the robot body frame."""
    ensure_camera_offset_buffers(camera)
    return as_torch(camera._offset_pos), as_torch(camera._offset_quat)
