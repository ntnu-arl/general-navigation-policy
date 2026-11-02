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
"""Extend the obstacle navigation room while keeping spawn/goal/obstacle zones fixed."""

from __future__ import annotations

from typing import Any

from isaaclab_tasks.contrib.drone_arl.navigation.config.arl_robot_1.scenes.obstacle_scenes.obstacle_scene import (
    OBSTACLE_SCENE_CFG,
    generate_obstacle_collection,
)
from isaaclab_tasks.contrib.drone_arl.navigation.config.arl_robot_1.scenes.obstacle_scenes.obstacle_scene_cfg import (
    ObstaclesSceneCfg,
)


def extend_obstacle_room(env_cfg: Any, room_extension_m: tuple[float, float, float]) -> None:
    """Grow room walls outward; remap obstacle ratios so absolute zones stay fixed.

    Walls use ``(center_ratio - 0.5) * env_size``; ratios are remapped when env_size grows.
    """
    ext_x, ext_y, ext_z = room_extension_m

    if ext_x == 0.0 and ext_y == 0.0 and ext_z == 0.0:
        return

    sx_old, sy_old, sz_old = OBSTACLE_SCENE_CFG.env_size
    sx_new = sx_old + 2.0 * ext_x
    sy_new = sy_old + 2.0 * ext_y
    sz_new = sz_old + 2.0 * ext_z

    BoxCfg = ObstaclesSceneCfg.BoxCfg
    wall_thickness = 0.2

    def remap(r: float, s_old: float, s_new: float) -> float:
        return (r - 0.5) * s_old / s_new + 0.5

    new_cfg = ObstaclesSceneCfg(
        env_size=(sx_new, sy_new, sz_new),
        min_num_obstacles=OBSTACLE_SCENE_CFG.min_num_obstacles,
        max_num_obstacles=OBSTACLE_SCENE_CFG.max_num_obstacles,
        ground_offset=OBSTACLE_SCENE_CFG.ground_offset,
    )

    new_obstacle_cfgs = {}
    for name, cfg in OBSTACLE_SCENE_CFG.obstacle_cfgs.items():
        new_obstacle_cfgs[name] = BoxCfg(
            size=cfg.size,
            center_ratio_min=(
                remap(cfg.center_ratio_min[0], sx_old, sx_new),
                remap(cfg.center_ratio_min[1], sy_old, sy_new),
                remap(cfg.center_ratio_min[2], sz_old, sz_new),
            ),
            center_ratio_max=(
                remap(cfg.center_ratio_max[0], sx_old, sx_new),
                remap(cfg.center_ratio_max[1], sy_old, sy_new),
                remap(cfg.center_ratio_max[2], sz_old, sz_new),
            ),
        )
    new_cfg.obstacle_cfgs = new_obstacle_cfgs

    new_cfg.wall_cfgs = {
        "left_wall": BoxCfg(
            size=(sx_new, wall_thickness, sz_new),
            center_ratio_min=(0.5, 1.0, 0.5),
            center_ratio_max=(0.5, 1.0, 0.5),
        ),
        "right_wall": BoxCfg(
            size=(sx_new, wall_thickness, sz_new),
            center_ratio_min=(0.5, 0.0, 0.5),
            center_ratio_max=(0.5, 0.0, 0.5),
        ),
        "back_wall": BoxCfg(
            size=(wall_thickness, sy_new, sz_new),
            center_ratio_min=(0.0, 0.5, 0.5),
            center_ratio_max=(0.0, 0.5, 0.5),
        ),
        "front_wall": BoxCfg(
            size=(wall_thickness, sy_new, sz_new),
            center_ratio_min=(1.0, 0.5, 0.5),
            center_ratio_max=(1.0, 0.5, 0.5),
        ),
        "bottom_wall": BoxCfg(
            size=(sx_new, sy_new, wall_thickness),
            center_ratio_min=(0.5, 0.5, 0.0),
            center_ratio_max=(0.5, 0.5, 0.0),
        ),
        "top_wall": BoxCfg(
            size=(sx_new, sy_new, wall_thickness),
            center_ratio_min=(0.5, 0.5, 1.0),
            center_ratio_max=(0.5, 0.5, 1.0),
        ),
    }

    env_cfg.scene.object_collection = generate_obstacle_collection(new_cfg)

    reset_params = env_cfg.events.reset_obstacles.params
    reset_params["env_size"] = new_cfg.env_size
    reset_params["wall_configs"] = new_cfg.wall_cfgs
    reset_params["obstacle_configs"] = new_cfg.obstacle_cfgs
