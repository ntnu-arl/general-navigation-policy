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
"""Camera frustum visualization configuration."""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class CameraFrustumVisualizationCfg:
    """Configuration for camera frustum visualization.
    
    This configuration can be added to any environment config to enable frustum visualization.
    The visualization will be automatically updated each frame when enabled.
    """

    enabled: bool = False
    """Whether to enable frustum visualization. Defaults to False."""

    camera_name: str = "depth_camera"
    """Name of the camera sensor in the scene. Defaults to "depth_camera"."""

    robot_name: str = "robot"
    """Name of the robot asset in the scene. Defaults to "robot".
    
    Used to compute camera pose from robot base_link and camera offset.
    """

    frustum_distance: float = 2.0
    """Distance from camera to far plane in meters. Defaults to 2.0."""

    line_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    """RGBA color for the frustum lines. Defaults to red (1.0, 0.0, 0.0, 1.0)."""

    line_thickness: float = 2.0
    """Thickness of the frustum lines. Defaults to 2.0."""

