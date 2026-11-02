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
"""Robot-state observation terms for dict policy observations."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape, record_dtype


@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"], observation_type="SensorState", on_inspect=[record_shape, record_dtype]
)
def perspective_robot_state_flat(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    normalize: bool = True,
) -> torch.Tensor:
    """Concatenated robot state (goal, attitude, velocities, last action).

    ``sensor_cfg`` / ``normalize`` are unused API placeholders for ObsTerm compatibility.
    """
    from isaaclab.envs import mdp as core_mdp
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.contrib.drone_arl.mdp.observations import (
        base_roll_pitch,
        generated_drone_commands,
        last_action_navigation,
    )

    robot_state_terms = []

    base_link_position = generated_drone_commands(
        env=env,
        command_name="target_pose",
        asset_cfg=SceneEntityCfg("robot"),
    )
    robot_state_terms.append(base_link_position)
    robot_state_terms.append(base_roll_pitch(env=env))
    robot_state_terms.append(core_mdp.base_lin_vel(env=env))
    robot_state_terms.append(core_mdp.base_ang_vel(env=env))
    robot_state_terms.append(
        last_action_navigation(env=env, action_name="velocity_commands")
    )

    return torch.cat(robot_state_terms, dim=-1)
