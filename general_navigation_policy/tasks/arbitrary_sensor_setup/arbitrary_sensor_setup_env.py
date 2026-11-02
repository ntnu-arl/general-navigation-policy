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
"""Arbitrary sensor-setup navigation environment."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from general_navigation_policy.tasks.metrics import (
    DEFAULT_SUCCESS_RADIUS_M,
    add_obstacle_curriculum_metrics,
    episode_success_mask,
)
from .frustum_visualization import visualize_camera_frustum


class ArbitrarySensorSetupEnv(ManagerBasedRLEnv):
    """Navigation env with success/crash metrics and optional frustum visualization."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._success_radius = float(getattr(cfg, "success_radius", DEFAULT_SUCCESS_RADIUS_M))
        self._episode_goals_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_goal_times = torch.full((self.num_envs,), float("inf"), device=self.device)

    def step(self, action: torch.Tensor):
        result = super().step(action)

        if hasattr(self.scene, "robot") and hasattr(self.command_manager, "get_command"):
            robot = self.scene["robot"]
            target_poses = self.command_manager.get_command("target_pose")
            target_positions = target_poses[:, :3]
            robot_positions = robot.data.root_pos_w - self.scene.env_origins
            distances_to_goal = torch.norm(robot_positions - target_positions, dim=1)
            newly_reached = (
                (distances_to_goal <= self._success_radius) & ~self._episode_goals_reached
            )
            self._episode_goals_reached[newly_reached] = True
            self._episode_goal_times[newly_reached] = self.episode_length_buf[newly_reached]

        if hasattr(self.cfg, "frustum_visualization") and self.cfg.frustum_visualization.enabled:
            camera_names = []
            max_num_cameras = getattr(self.cfg, "max_num_cameras", 0)
            for cam_idx in range(max_num_cameras):
                sensor_name = f"depth_camera_{cam_idx}"
                if sensor_name in self.scene.keys():
                    camera_names.append(sensor_name)

            if not camera_names:
                camera_names = [self.cfg.frustum_visualization.camera_name]

            if hasattr(self, "_active_camera_mask") and len(camera_names) == max_num_cameras:
                active_per_camera = self._active_camera_mask.any(dim=0)
                camera_names = [
                    f"depth_camera_{cam_idx}"
                    for cam_idx in range(max_num_cameras)
                    if bool(active_per_camera[cam_idx].item())
                ]

            for cam_draw_idx, camera_name in enumerate(camera_names):
                visualize_camera_frustum(
                    env=self,
                    camera_name=camera_name,
                    frustum_distance=self.cfg.frustum_visualization.frustum_distance,
                    line_color=self.cfg.frustum_visualization.line_color,
                    line_thickness=self.cfg.frustum_visualization.line_thickness,
                    robot_name=self.cfg.frustum_visualization.robot_name,
                    clear_existing=(cam_draw_idx == 0),
                )
        return result

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            super()._reset_idx(env_ids)
            return

        robot = self.scene["robot"]
        target_poses = self.command_manager.get_command("target_pose")
        target_positions = target_poses[:, :3]
        robot_positions = robot.data.root_pos_w - self.scene.env_origins
        distances_to_goal = torch.norm(robot_positions - target_positions, dim=1)
        final_errors = distances_to_goal[env_ids].clone()
        goals_reached = self._episode_goals_reached[env_ids].clone()
        goal_times = self._episode_goal_times[env_ids].clone()

        if hasattr(self, "reset_terminated") and hasattr(self, "reset_time_outs"):
            crashed = self.reset_terminated[env_ids] & ~self.reset_time_outs[env_ids]
            timed_out = self.reset_time_outs[env_ids]
        else:
            crashed = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            timed_out = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        super()._reset_idx(env_ids)

        succeeded = episode_success_mask(goals_reached, final_errors, self._success_radius)

        num_resetting = len(env_ids)
        num_successful = succeeded.sum().item()
        num_crashed = crashed.sum().item()
        num_timeout = timed_out.sum().item()

        if num_resetting > 0:
            self.extras["log"]["Metrics/success_rate"] = num_successful / num_resetting
            self.extras["log"]["Metrics/crash_rate"] = num_crashed / num_resetting
            self.extras["log"]["Metrics/timeout_rate"] = num_timeout / num_resetting
            self.extras["log"]["Metrics/mean_final_position_error"] = final_errors.mean().item()
            self.extras["log"]["Metrics/success_radius_m"] = self._success_radius
            if goals_reached.any():
                goal_times_seconds = goal_times[goals_reached].float() * self.step_dt
                self.extras["log"]["Metrics/mean_time_to_goal"] = goal_times_seconds.mean().item()
            else:
                self.extras["log"]["Metrics/mean_time_to_goal"] = float("inf")
            add_obstacle_curriculum_metrics(self.extras["log"], self.cfg)

        self._episode_goals_reached[env_ids] = False
        self._episode_goal_times[env_ids] = float("inf")
