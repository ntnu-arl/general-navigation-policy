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
"""Apply identified robot sys-id parameters to navigation env configs."""

from __future__ import annotations

import math
from typing import Any


def apply_lmf2_sys_id(env_cfg: Any) -> None:
    """LMF2 velocity-controller and thruster parameters."""
    env_cfg.actions.velocity_commands.controller_cfg.K_vel_range = (
        (2.0, 2.0, 3.7),
        (2.2, 2.2, 3.9),
    )
    env_cfg.actions.velocity_commands.controller_cfg.K_rot_range = (
        (1.6788082694141405, 1.5900023655037197, 0.1789114470860935),
        (1.8788082694141405, 1.7900023655037197, 0.3789114470860935),
    )
    env_cfg.actions.velocity_commands.controller_cfg.K_angvel_range = (
        (0.35083859375098895, 0.3524356275040708, 0.02),
        (0.55083859375098895, 0.5524356275040708, 0.04),
    )
    env_cfg.actions.velocity_commands.controller_cfg.max_inclination_angle_rad = math.pi / 3.0
    env_cfg.actions.velocity_commands.controller_cfg.max_yaw_rate = math.pi / 3.0

    thr = env_cfg.scene.robot.actuators["thrusters"]
    thr.tau_inc_range = (0.05, 0.08)
    thr.tau_dec_range = (0.0046, 0.0046)
    thr.thrust_const_range = (9.26312e-06, 1.826312e-05)
    thr.thrust_range = (0.1, 10.0)
    thr.max_thrust_rate = 100_000.0

    env_cfg.scene.robot.allocation_matrix = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-0.13, -0.13, 0.13, 0.13],
        [-0.13, 0.13, 0.13, -0.13],
        [-0.07, 0.07, -0.07, 0.07],
    ]
