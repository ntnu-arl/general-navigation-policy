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
from __future__ import annotations

import torch

from isaaclab_contrib.mdp.actions import NavigationAction


class ArbitrarySensorSetupNavigationAction(NavigationAction):
    """Clamp and magnitude-limit body-frame velocity commands."""

    @property
    def action_dim(self) -> int:
        return 4

    def apply_actions(self):
        clamped_action = torch.clamp(self.processed_actions, min=-1.0, max=1.0)

        vel = clamped_action[:, :3] * self.cfg.max_magnitude
        norm = vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel = torch.where(norm > self.cfg.max_magnitude, vel / norm * self.cfg.max_magnitude, vel)

        processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        processed_actions[:, :3] = vel
        processed_actions[:, 3] = clamped_action[:, 3] * self.cfg.max_yaw_command

        if not self._has_actions_updated:
            self._prev_commands[:] = self._commands
            self._commands[:] = processed_actions
            self._has_actions_updated = True

        wrench_command = self._lc.compute(processed_actions)
        # Use cached pseudo-inverse from NavigationAction (pinverse-every-step is
        # slower and can differ numerically from the parent's _allocation_pinv).
        allocation_pinv = getattr(self, "_allocation_pinv", None)
        if allocation_pinv is None:
            thrust_commands = (torch.pinverse(self._asset.allocation_matrix) @ wrench_command.T).T
        else:
            thrust_commands = wrench_command @ allocation_pinv.T
        self._asset.set_thrust_target(thrust_commands, thruster_ids=self._thruster_ids)
