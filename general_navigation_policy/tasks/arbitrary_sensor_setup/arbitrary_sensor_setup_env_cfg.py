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
"""Configuration for arbitrary multi-camera sensor-setup navigation."""

from __future__ import annotations

import copy
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.contrib.drone_arl.navigation.config.arl_robot_1.navigation_env_cfg import (
    NavigationVelocityFloatingObstacleEnvCfg,
)

from general_navigation_policy.tasks.mdp import observations as mdp_obs
from general_navigation_policy.tasks.mdp import rewards as gnp_rewards
from general_navigation_policy.tasks.mdp.actions import ArbitrarySensorSetupNavigationAction
from general_navigation_policy.tasks.mdp.events import (
    randomize_camera_pose_and_intrinsics,
    randomize_num_active_cameras,
)
from general_navigation_policy.tasks.metrics import DEFAULT_SUCCESS_RADIUS_M
from general_navigation_policy.tasks.mdp.obstacle_scene_extension import extend_obstacle_room
from general_navigation_policy.tasks.mdp.robot_sys_id import apply_lmf2_sys_id
from .frustum_visualization_cfg import CameraFrustumVisualizationCfg

# Template depth-camera resolution (cloned to depth_camera_0..N-1).
DEPTH_IMAGE_WIDTH = 90
DEPTH_IMAGE_HEIGHT = 50


@configclass
class PolicyCfgCanonicalized(ObsGroup):
    """Dict observations: 360° canonicalized range image + robot_state."""

    robot_state = ObsTerm(
        func=mdp_obs.perspective_robot_state_flat,
        params={"normalize": True},
    )

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class ArbitrarySensorSetupEnvCfg(NavigationVelocityFloatingObstacleEnvCfg):
    """Navigation with an arbitrary multi-camera sensor setup and spherical observations."""

    frustum_visualization: CameraFrustumVisualizationCfg = (
        CameraFrustumVisualizationCfg(enabled=False)
    )

    max_num_cameras: int = 4
    min_num_active_cameras: int = 1
    min_num_obstacles: int = 1
    success_radius: float = DEFAULT_SUCCESS_RADIUS_M

    depth_min_m: float = 0.60
    depth_max_m: float = 5.00

    room_extension_m: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def _setup_multi_cameras(self):
        """Create depth_camera_0..depth_camera_{N-1} from the template depth_camera."""
        base_cam_cfg = self.scene.depth_camera

        # Lab 3 OffsetCfg.rot is (x, y, z, w). Yaw about +Z: (0, 0, sin(θ/2), cos(θ/2)).
        yaw_quats = [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.70710678, 0.70710678),
            (0.0, 0.0, -0.70710678, 0.70710678),
            (0.0, 0.0, 1.0, 0.0),
        ]

        for i in range(self.max_num_cameras):
            cam_cfg_i = copy.deepcopy(base_cam_cfg)
            if i < len(yaw_quats):
                cam_cfg_i.offset.rot = yaw_quats[i]
            setattr(self.scene, f"depth_camera_{i}", cam_cfg_i)

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

        self.sim.physics = PhysxCfg(
            gpu_found_lost_pairs_capacity=8 * 1024 * 1024,
            gpu_max_rigid_contact_count=24 * 1024 * 1024,
            gpu_max_rigid_patch_count=8 * 1024 * 1024,
        )
        self.sim.use_newton_actuators = False

        pc = self.scene.depth_camera.pattern_cfg
        self.scene.depth_camera.pattern_cfg = PinholeCameraPatternCfg(
            width=DEPTH_IMAGE_WIDTH,
            height=DEPTH_IMAGE_HEIGHT,
            focal_length=pc.focal_length,
            horizontal_aperture=pc.horizontal_aperture,
            vertical_aperture=pc.vertical_aperture,
            horizontal_aperture_offset=pc.horizontal_aperture_offset,
            vertical_aperture_offset=pc.vertical_aperture_offset,
        )
        # Invalid / beyond max_distance → 0 (D455-like); spherical map remaps to far.
        self.scene.depth_camera.depth_clipping_behavior = "zero"
        self.scene.depth_camera.max_distance = self.depth_max_m
        # Lab 3 OffsetCfg.rot is (x, y, z, w). Parent EA navigation cfg still ships wxyz identity.
        off = self.scene.depth_camera.offset
        if tuple(off.rot) == (1.0, 0.0, 0.0, 0.0):
            off.rot = (0.0, 0.0, 0.0, 1.0)

        self.events.reset_obstacles.params["min_num_obstacles"] = self.min_num_obstacles

        extend_obstacle_room(self, self.room_extension_m)
        self._setup_multi_cameras()
        self.frustum_visualization.camera_name = "depth_camera_0"
        self.observations.policy = PolicyCfgCanonicalized()
        self.actions.velocity_commands.class_type = ArbitrarySensorSetupNavigationAction
        self.actions.velocity_commands.max_magnitude = 1.1
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi, math.pi)

        self.episode_length_s = 14.0
        self.commands.target_pose.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        self.commands.target_pose.ranges.pos_x = (5.0, 6.5)

        for i in range(self.max_num_cameras):
            setattr(
                self.events,
                f"randomize_camera_{i}",
                EventTerm(
                    func=randomize_camera_pose_and_intrinsics,
                    mode="reset",
                    params={
                        "sensor_cfg": SceneEntityCfg(f"depth_camera_{i}"),
                        "position_range": {
                            "x": (-0.2, 0.2),
                            "y": (-0.2, 0.2),
                            "z": (-0.1, 0.1),
                        },
                        "orientation_range": {
                            "roll": (-90.0, 90.0),
                            "pitch": (-45.0, 45.0),
                            "yaw": (-180.0, 180.0),
                        },
                        "focal_length_range": (0.10, 0.30),
                        "horizontal_aperture_range": (0.28, 0.44),
                        "aperture_aspect_ratio_h_over_v_range": (0.75, 2.0),
                    },
                ),
            )

        self.events.randomize_num_active_cameras = EventTerm(
            func=randomize_num_active_cameras,
            mode="reset",
            params={
                "max_num_cameras": self.max_num_cameras,
                "min_num_active_cameras": self.min_num_active_cameras,
            },
        )

        self.observations.policy.image = ObsTerm(
            func=mdp_obs.canonicalized_360_depth_multi_camera,
            params={
                "max_num_cameras": self.max_num_cameras,
                "robot_cfg": SceneEntityCfg("robot"),
                "azimuth_resolution": 90,
                "elevation_resolution": 45,
                "max_range": self.depth_max_m,
                "apply_depth_noise": True,
                "depth_noise_params": {
                    "focal_length": 25.0,
                    "baseline": 0.12,
                    "min_depth": self.depth_min_m,
                    "max_depth": self.depth_max_m,
                    "filter_size": 3,
                    "inlier_thred_range": (0.03, 0.08),
                    "prob_range": (0.7, 0.9),
                },
                "apply_intrinsics_noise": True,
                "intrinsics_noise_params": {
                    "fx_scale_range": (0.99, 1.01),
                    "fy_scale_range": (0.99, 1.01),
                    "cx_offset_px_range": (-1.0, 1.0),
                    "cy_offset_px_range": (-1.0, 1.0),
                },
                "apply_pose_noise": True,
                "pose_noise_params": {
                    "x_m_range": (-0.02, 0.02),
                    "y_m_range": (-0.02, 0.02),
                    "z_m_range": (-0.02, 0.02),
                    "roll_deg_range": (-2.0, 2.0),
                    "pitch_deg_range": (-2.0, 2.0),
                    "yaw_deg_range": (-2.0, 2.0),
                },
                "use_inverse_depth": True,
            },
        )

        self.rewards.action_in_observed_direction = RewTerm(
            func=gnp_rewards.action_in_observed_direction,
            weight=1.3,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "azimuth_resolution": 90,
                "elevation_resolution": 45,
                "window_half_size": 5,
            },
        )

        self.rewards.time_to_collision_penalty = RewTerm(
            func=gnp_rewards.time_to_collision_penalty_from_canonicalized_depth,
            weight=1.3,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ttc_magnitude": -8.0,
                "ttc_exponent": 3.0,
                "max_ttc": 10.0,
            },
        )

        self.rewards.termination_penalty.weight = -350.0

        apply_lmf2_sys_id(self)


@configclass
class ArbitrarySensorSetupEnvCfg_PLAY(ArbitrarySensorSetupEnvCfg):
    """Play/eval variant (single env, frustum on, fixed difficulty)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.0
        self.frustum_visualization.enabled = True
        # Shorter far-plane keeps FOV overlays readable in follow-cam play/video.
        self.frustum_visualization.frustum_distance = 0.6
        self.frustum_visualization.line_thickness = 1.5
        self.curriculum.obstacle_levels.params["min_difficulty"] = 9
        self.curriculum.obstacle_levels.params["max_difficulty"] = 9
        self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi / 6.0, math.pi / 6.0)
