#!/usr/bin/env python3
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
"""Top-down rollout visualization for arbitrary-sensor policies.

Usage: HEADLESS=1 isaaclab -p scripts/visualize_trajectories.py --checkpoint ... --active_cameras random
"""

from __future__ import annotations

import argparse
import datetime as dtm
import os
import random
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Top-down rollout screenshot with path/camera overlays (arbitrary sensor setup)."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-Play-v0",
    help="Gym task ID.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="skrl_spherical_range_cnn_gru_cfg_entry_point",
    help="Agent cfg entry-point key from the gym registry.",
)
parser.add_argument(
    "--model",
    type=str,
    default="spherical_range_cnn_gru",
    choices=["spherical_range_cnn_gru"],
    help="Model architecture.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt).")
parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes to run.")
parser.add_argument("--max_steps", type=int, default=1500, help="Max steps per episode.")
parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
parser.add_argument(
    "--obstacle_seed",
    type=int,
    default=None,
    help="Seed for the first-reset obstacle layout (default: --seed).",
)
parser.add_argument(
    "--success_radius",
    type=float,
    default=1.0,
    help="Goal radius [m] used for success.",
)
parser.add_argument("--frustum_stride", type=int, default=15, help="Draw camera axis every N steps.")
parser.add_argument("--frustum_distance", type=float, default=0.4, help="Camera axis length [m].")
parser.add_argument(
    "--active_cameras",
    type=str,
    default="random",
    help="'random' or fixed count 1–4 (colors encode camera-count systems).",
)
parser.add_argument("--path_height_offset", type=float, default=0.05, help="Lift paths above ground [m].")
parser.add_argument("--frustum_height_offset", type=float, default=0.08, help="Lift camera overlays [m].")
parser.add_argument(
    "--image_scale",
    type=float,
    default=1.0,
    help="Scale factor for saved PNG resolution.",
)
parser.add_argument(
    "--hide_top_wall",
    action="store_true",
    default=True,
    help="Hide top wall prims before screenshot.",
)
parser.add_argument("--show_top_wall", action="store_true", help="Keep top wall prims visible.")
parser.add_argument(
    "--output",
    type=str,
    default="outputs/analysis/topdown_rollouts_isaac.png",
    help="Output image path (a dated run folder is created next to it).",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.show_top_wall:
    args_cli.hide_top_wall = False

# Isaac Lab 3: viewport RGB capture needs RTX/offscreen cameras under HEADLESS.
args_cli.enable_cameras = True
# Offline PathTracing for stills: Lab experiences pin RealTimePathTracing, which
# looks grainy when NGX/DLSS fails headless. PathTracing + high totalSpp converges.
# OptiX denoiser needs /usr/share/nvidia/nvoptix.bin (often missing in Docker);
# leave it off — PathTracing + high totalSpp is enough for stills.
_kit_quality_args = (
    "--/rtx/rendermode=PathTracing "
    "--/rtx/pathtracing/spp=8 "
    "--/rtx/pathtracing/totalSpp=512 "
    "--/rtx/pathtracing/clampSpp=512 "
    "--/rtx/pathtracing/optixDenoiser/enabled=false "
    "--/rtx/pathtracing/optixDenoiser/temporalMode/enabled=false "
    "--/rtx/post/tvNoise/enabled=false"
)
existing_kit_args = getattr(args_cli, "kit_args", None) or ""
args_cli.kit_args = f"{existing_kit_args} {_kit_quality_args}".strip()

hydra_args = [a for a in hydra_args if a.strip()]
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import numpy as np
import skrl
import torch
from packaging import version

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from skrl.agents.torch.ppo import PPO, PPO_RNN
from skrl.memories.torch import RandomMemory
from general_navigation_policy.learning.skrl.load_agent_cfg import build_ppo_cfg

import general_navigation_policy.tasks.arbitrary_sensor_setup  # noqa: F401
from general_navigation_policy.learning.skrl.models.spherical_range_cnn_gru import SphericalRangeCNNGRU
from general_navigation_policy.tasks.metrics import is_episode_success
from general_navigation_policy.tasks.mdp.events import configure_active_cameras

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
except Exception:  # pragma: no cover
    omni_debug_draw = None


SKRL_VERSION = "2.1.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}.")

MODEL_CLASSES = {
    "spherical_range_cnn_gru": SphericalRangeCNNGRU,
}

# Hardcoded side-view options requested by user.
SIDE_VIEW_CAMERA_SIDE = "neg_y"
SIDE_VIEW_HIDE_WALL = "right"
ACTIVE_SYSTEM_CAMERA_COUNT = 1


def _extract_robot_and_goal_positions(raw_env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = raw_env.unwrapped.scene["robot"]
    target_poses = raw_env.unwrapped.command_manager.get_command("target_pose")
    target_positions = target_poses[:, :3]
    robot_positions = robot.data.root_pos_w - raw_env.unwrapped.scene.env_origins
    return robot_positions, target_positions


def _canonicalized_coverage_per_env(raw_env, obs_obj, observation_space) -> torch.Tensor | None:
    """Fraction of valid canonicalized bins per env (same as evaluate_skrl)."""
    mask = getattr(raw_env.unwrapped, "_cached_validity_mask", None)
    if torch.is_tensor(mask) and mask.ndim == 3 and mask.shape[0] == raw_env.unwrapped.num_envs:
        return mask.to(dtype=torch.float32).mean(dim=(1, 2))

    def _find_image_tensor(obj):
        if torch.is_tensor(obj) and obj.ndim == 4 and obj.shape[-1] == 2:
            return obj
        if isinstance(obj, dict):
            if (
                "image" in obj
                and torch.is_tensor(obj["image"])
                and obj["image"].ndim == 4
                and obj["image"].shape[-1] == 2
            ):
                return obj["image"]
            for value in obj.values():
                found = _find_image_tensor(value)
                if found is not None:
                    return found
        return None

    image = _find_image_tensor(obs_obj) if isinstance(obs_obj, dict) else None
    if image is None and torch.is_tensor(obs_obj) and obs_obj.ndim == 4 and obs_obj.shape[-1] == 2:
        image = obs_obj
    if image is None:
        return None
    return image[..., 1].to(dtype=torch.float32).mean(dim=(1, 2))


def _sanitize_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    return clean.strip("-") or "run"


def _derive_policy_tag_from_checkpoint(checkpoint_path: str) -> str:
    parts = os.path.normpath(os.path.abspath(checkpoint_path)).split(os.sep)
    try:
        logs_idx = parts.index("logs")
        if logs_idx + 2 < len(parts) and parts[logs_idx + 1] == "skrl":
            return _sanitize_name(parts[logs_idx + 2])
    except ValueError:
        pass
    return _sanitize_name(os.path.splitext(os.path.basename(checkpoint_path))[0])


def _prepare_output_paths() -> tuple[str, str, str, str]:
    """Create a dated run folder under the bind-mounted repo when possible.

    The container UID often differs from the host mount owner, so creating
    ``outputs/`` can raise PermissionError. Prefer the repo path (host-visible);
    only fall back to ``/tmp`` if nothing under the mount is writable.

    Returns:
        ``(run_folder, plot_path, stats_path, analysis_root)`` where ``analysis_root``
        is the parent used for stable "latest" copies outside the dated folder.
    """
    output_abs = os.path.abspath(args_cli.output)
    analysis_root = os.path.dirname(output_abs)
    plot_name = os.path.basename(output_abs)

    timestamp = dtm.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    policy_tag = _derive_policy_tag_from_checkpoint(args_cli.checkpoint)
    run_folder_name = f"{timestamp}__{policy_tag}"

    candidates = [analysis_root]
    # Always try the conventional bind-mounted path even if --output was absolute elsewhere.
    repo_analysis = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "analysis"))
    if repo_analysis not in candidates:
        candidates.append(repo_analysis)

    run_folder = None
    last_err: Exception | None = None
    for root in candidates:
        try:
            os.makedirs(root, exist_ok=True)
            # World-writable so host UID / container UID can both append later.
            try:
                os.chmod(root, 0o777)
                parent = os.path.dirname(root)
                if os.path.basename(parent) == "outputs":
                    os.chmod(parent, 0o777)
            except OSError:
                pass
            candidate = os.path.join(root, run_folder_name)
            os.makedirs(candidate, exist_ok=True)
            run_folder = candidate
            analysis_root = root
            if root != os.path.dirname(output_abs):
                print(f"[INFO] Writing visualization outputs under bind mount: {run_folder}")
            break
        except PermissionError as e:
            last_err = e
            continue

    if run_folder is None:
        fallback_root = os.path.join("/tmp", "gnp_visualize", "analysis")
        run_folder = os.path.join(fallback_root, run_folder_name)
        os.makedirs(run_folder, exist_ok=True)
        analysis_root = fallback_root
        print(
            f"[WARNING] Cannot write under the repo mount ({last_err}); "
            f"writing to {run_folder!r}. On the host, run:\n"
            f"  mkdir -p outputs/analysis && chmod 777 outputs outputs/analysis\n"
            f"then re-run, or copy with:\n"
            f"  docker cp isaac-lab-base:{run_folder} ./outputs/analysis/"
        )

    plot_path = os.path.join(run_folder, plot_name)
    stats_path = os.path.join(run_folder, "evaluation_stats.txt")
    return run_folder, plot_path, stats_path, analysis_root


def _publish_latest_outputs(run_folder: str, analysis_root: str, plot_name: str) -> list[str]:
    """Copy key artifacts to stable names under ``analysis_root`` (outside dated folder).

    So IDEs / file browsers can keep ``outputs/analysis/topdown_rollouts_isaac.png``
    open and see each new run without hunting timestamped directories.
    """
    import shutil

    published: list[str] = []
    names = [plot_name, "evaluation_stats.txt", "scene_side_raw.png"]
    for name in names:
        src = os.path.join(run_folder, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(analysis_root, name)
        try:
            shutil.copy2(src, dst)
            published.append(dst)
        except OSError as e:
            print(f"[WARNING] Could not publish latest {name!r} to {dst!r}: {e}")

    latest_link = os.path.join(analysis_root, "latest")
    try:
        if os.path.islink(latest_link) or os.path.isfile(latest_link):
            os.remove(latest_link)
        elif os.path.isdir(latest_link) and not os.path.islink(latest_link):
            # Don't delete a real directory named latest.
            latest_link = ""
        if latest_link:
            os.symlink(os.path.abspath(run_folder), latest_link)
            published.append(latest_link)
    except OSError as e:
        print(f"[WARNING] Could not update {analysis_root!r}/latest symlink: {e}")
    return published


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detect_camera_names(scene) -> list[str]:
    names = []
    for i in range(8):
        key = f"depth_camera_{i}"
        if key in scene.keys():
            names.append(key)
    if not names and "depth_camera" in scene.keys():
        names.append("depth_camera")
    return names


def _camera_frustum_segments_3d(
    raw_env,
    camera_name: str,
    active_camera_count: int,
    frustum_distance: float,
    progress: float,
    env_idx: int = 0,
):
    scene = raw_env.unwrapped.scene
    camera = scene[camera_name]
    robot = scene["robot"]

    width = int(camera.cfg.pattern_cfg.width)
    height = int(camera.cfg.pattern_cfg.height)
    dirs_cam = camera.ray_directions[env_idx]

    base_pos_w = robot.data.root_pos_w[env_idx : env_idx + 1]
    base_quat_w = robot.data.root_quat_w[env_idx : env_idx + 1]
    offset_pos = camera._offset_pos[env_idx : env_idx + 1]
    offset_quat = camera._offset_quat[env_idx : env_idx + 1]
    cam_pos_w, cam_quat_w = math_utils.combine_frame_transforms(base_pos_w, base_quat_w, offset_pos, offset_quat)

    origin = cam_pos_w[0]
    center_idx = (height // 2) * width + (width // 2)
    d_cam_center = dirs_cam[center_idx]
    d_w = math_utils.quat_apply(cam_quat_w, d_cam_center.unsqueeze(0)).squeeze(0)
    d_w = d_w / (torch.norm(d_w) + 1e-8)

    shaft_tip = cam_pos_w[0] + float(frustum_distance) * d_w
    head_len = max(0.12, 0.24 * float(frustum_distance))
    head_width = max(0.06, 0.13 * float(frustum_distance))
    head_base = shaft_tip - head_len * d_w

    # Build a stable orthonormal basis around the camera forward direction.
    ref = torch.tensor([0.0, 0.0, 1.0], device=d_w.device, dtype=d_w.dtype)
    if torch.abs(torch.dot(d_w, ref)) > 0.92:
        ref = torch.tensor([1.0, 0.0, 0.0], device=d_w.device, dtype=d_w.dtype)
    p1 = torch.cross(d_w, ref, dim=0)
    p1 = p1 / (torch.norm(p1) + 1e-8)
    p2 = torch.cross(d_w, p1, dim=0)
    p2 = p2 / (torch.norm(p2) + 1e-8)

    h1 = head_base + head_width * p1
    h2 = head_base - head_width * p1
    h3 = head_base + head_width * p2
    h4 = head_base - head_width * p2

    origin_np = origin.detach().cpu().numpy()
    shaft_tip_np = shaft_tip.detach().cpu().numpy()
    segments = [(origin_np, shaft_tip_np, int(active_camera_count), float(progress))]
    for h in (h1, h2, h3, h4):
        segments.append((h.detach().cpu().numpy(), shaft_tip_np, int(active_camera_count), float(progress)))
    return segments


def _camera_frustum_wireframe_segments_3d(
    raw_env,
    camera_name: str,
    active_camera_count: int,
    frustum_distance: float,
    progress: float,
    env_idx: int = 0,
):
    """Frustum wireframe segments (used for 3D legend schematic)."""
    scene = raw_env.unwrapped.scene
    camera = scene[camera_name]
    robot = scene["robot"]

    width = int(camera.cfg.pattern_cfg.width)
    height = int(camera.cfg.pattern_cfg.height)
    dirs_cam = camera.ray_directions[env_idx]
    corner_indices = [
        0 * width + 0,
        0 * width + (width - 1),
        (height - 1) * width + 0,
        (height - 1) * width + (width - 1),
    ]
    corner_dirs_cam = [dirs_cam[i] for i in corner_indices]

    base_pos_w = robot.data.root_pos_w[env_idx : env_idx + 1]
    base_quat_w = robot.data.root_quat_w[env_idx : env_idx + 1]
    offset_pos = camera._offset_pos[env_idx : env_idx + 1]
    offset_quat = camera._offset_quat[env_idx : env_idx + 1]
    cam_pos_w, cam_quat_w = math_utils.combine_frame_transforms(base_pos_w, base_quat_w, offset_pos, offset_quat)

    origin = cam_pos_w[0]
    corner_world = []
    for d_cam in corner_dirs_cam:
        d_w = math_utils.quat_apply(cam_quat_w, d_cam.unsqueeze(0)).squeeze(0)
        d_w = d_w / (torch.norm(d_w) + 1e-8)
        corner_world.append((cam_pos_w[0] + frustum_distance * d_w).detach().cpu().numpy())

    origin_np = origin.detach().cpu().numpy()
    segments = []
    for c in corner_world:
        segments.append((origin_np, c, int(active_camera_count), float(progress)))
    for i, j in [(0, 1), (1, 3), (3, 2), (2, 0)]:
        segments.append((corner_world[i], corner_world[j], int(active_camera_count), float(progress)))
    return segments


def _robot_orientation_frame_segments_3d(
    raw_env,
    env_idx: int = 0,
    axis_length: float = 0.18,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Return small XYZ frame segments using robot body orientation."""
    robot = raw_env.unwrapped.scene["robot"]
    origin = robot.data.root_pos_w[env_idx]
    quat_w = robot.data.root_quat_w[env_idx : env_idx + 1]

    axes_b = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        device=quat_w.device,
        dtype=quat_w.dtype,
    )
    ends = math_utils.quat_apply(quat_w.repeat(3, 1), axes_b) * float(axis_length) + origin.unsqueeze(0).repeat(3, 1)
    o = origin.detach().cpu().numpy()
    return [
        (o, ends[0].detach().cpu().numpy(), "x"),
        (o, ends[1].detach().cpu().numpy(), "y"),
        (o, ends[2].detach().cpu().numpy(), "z"),
    ]


def _collect_obstacle_xy(scene, env_idx: int = 0) -> np.ndarray:
    points = []
    for key in scene.keys():
        if key.startswith("obstacle_"):
            asset = scene[key]
            data = getattr(asset, "data", None)
            pos = getattr(data, "root_pos_w", None)
            if pos is not None and pos.ndim >= 2:
                points.append(pos[env_idx, :2].detach().cpu().numpy())
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(points, axis=0)


def _set_wall_visibility(
    hide_top: bool,
    hide_left: bool = False,
    hide_right: bool = False,
    hide_front: bool = False,
    hide_back: bool = False,
):
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if hide_top and "obstacle_top_wall" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
            if hide_left and "obstacle_left_wall" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
            if hide_right and "obstacle_right_wall" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
            if hide_front and "obstacle_front_wall" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
            if hide_back and "obstacle_back_wall" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
    except Exception as e:
        print(f"[WARNING] Could not update wall visibility: {e}")


def _frustum_color(active_camera_count: int, progress: float) -> list[float]:
    # Fixed, bright per-system color: same for all camera-axis arrows in one run.
    # Different systems (1/2/3/4 camera setups) get different colors.
    _ = progress
    cam_count = max(1, int(active_camera_count))
    palette = [
        (1.0, 0.55, 0.00),   # 1-camera system: orange
        (0.00, 0.95, 0.35),  # 2-camera system: green
        (0.10, 0.55, 1.00),  # 3-camera system: blue
        (1.00, 0.15, 0.85),  # 4-camera system: magenta
    ]
    r, g, b = palette[(cam_count - 1) % len(palette)]
    return [float(r), float(g), float(b), 1.0]


def _draw_debug_lines(
    episode_paths_xyz: list[list[np.ndarray]],
    frustum_segments: list[tuple[np.ndarray, np.ndarray, int, float]],
    orientation_segments: list[tuple[np.ndarray, np.ndarray, str]],
):
    if omni_debug_draw is None:
        print("[WARNING] Debug draw unavailable; overlay lines will be skipped.")
        return

    draw = omni_debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    starts = []
    ends = []
    colors = []
    sizes = []

    # Trajectory lines (episode-wise, no cross-episode connections).
    for ep_path in episode_paths_xyz:
        for i in range(1, len(ep_path)):
            p0 = ep_path[i - 1].copy()
            p1 = ep_path[i].copy()
            p0[2] += args_cli.path_height_offset
            p1[2] += args_cli.path_height_offset
            starts.append(p0.tolist())
            ends.append(p1.tolist())
            colors.append([1.0, 0.05, 0.05, 1.0])
            sizes.append(2.6)

    # Camera principal-axis arrows.
    for s, e, active_count, progress in frustum_segments:
        s3 = s.copy()
        e3 = e.copy()
        s3[2] += args_cli.frustum_height_offset
        e3[2] += args_cli.frustum_height_offset
        starts.append(s3.tolist())
        ends.append(e3.tolist())
        colors.append(_frustum_color(active_count, progress))
        sizes.append(1.8)

    # Robot orientation triads (XYZ) at camera sample points.
    for s, e, axis in orientation_segments:
        s3 = s.copy()
        e3 = e.copy()
        s3[2] += args_cli.frustum_height_offset
        e3[2] += args_cli.frustum_height_offset
        starts.append(s3.tolist())
        ends.append(e3.tolist())
        if axis == "x":
            colors.append([1.0, 0.0, 0.0, 1.0])
        elif axis == "y":
            colors.append([0.0, 1.0, 0.0, 1.0])
        else:
            colors.append([0.2, 0.45, 1.0, 1.0])
        sizes.append(2.8)

    if starts:
        draw.draw_lines(starts, ends, colors, sizes)


def _add_stage_line_overlays(
    episode_paths_xyz: list[list[np.ndarray]],
    frustum_segments: list[tuple[np.ndarray, np.ndarray, int, float]],
    orientation_segments: list[tuple[np.ndarray, np.ndarray, str]],
    start_points_xyz: list[np.ndarray],
    goal_points_xyz: list[np.ndarray],
    end_points_xyz: list[np.ndarray],
    crash_points_xyz: list[np.ndarray],
    timeout_points_xyz: list[np.ndarray],
) -> bool:
    """Add trajectory/camera overlays as USD geometry so they appear in camera renders."""
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom
    except Exception as e:
        print(f"[WARNING] USD overlay imports unavailable: {e}")
        return False

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARNING] USD stage unavailable; cannot add line overlays.")
        return False

    root_path = "/World/AnalysisOverlays"
    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim.IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    # Trajectory as one polyline.
    if episode_paths_xyz:
        traj_path = f"{root_path}/trajectory"
        traj = UsdGeom.BasisCurves.Define(stage, traj_path)
        traj.CreateTypeAttr("linear")
        traj.CreateBasisAttr("bezier")
        traj.CreateWrapAttr("nonperiodic")

        points = []
        counts = []
        for ep_path in episode_paths_xyz:
            if len(ep_path) < 2:
                continue
            counts.append(len(ep_path))
            for p in ep_path:
                points.append(
                    Gf.Vec3f(
                        float(p[0]),
                        float(p[1]),
                        float(p[2] + args_cli.path_height_offset),
                    )
                )
        traj.CreatePointsAttr(points)
        traj.CreateCurveVertexCountsAttr(counts)
        traj.CreateWidthsAttr([0.05] * len(points))
        traj.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(1.0, 0.05, 0.05)])

    # Camera principal-axis arrows as many 2-point linear curves in one prim.
    if frustum_segments:
        fr_path = f"{root_path}/frustums"
        fr = UsdGeom.BasisCurves.Define(stage, fr_path)
        fr.CreateTypeAttr("linear")
        fr.CreateBasisAttr("bezier")
        fr.CreateWrapAttr("nonperiodic")

        points = []
        counts = []
        fr_colors = []
        for s, e, active_count, progress in frustum_segments:
            points.append(Gf.Vec3f(float(s[0]), float(s[1]), float(s[2] + args_cli.frustum_height_offset)))
            points.append(Gf.Vec3f(float(e[0]), float(e[1]), float(e[2] + args_cli.frustum_height_offset)))
            counts.append(2)
            c = _frustum_color(active_count, progress)
            fr_colors.append(Gf.Vec3f(c[0], c[1], c[2]))
        fr.CreatePointsAttr(points)
        fr.CreateCurveVertexCountsAttr(counts)
        fr.CreateWidthsAttr([0.030] * len(points))
        UsdGeom.PrimvarsAPI(fr.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
        ).Set(fr_colors)

    # Orientation triads (x/y/z) as separate colored line sets.
    if orientation_segments:
        axis_to_color = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.2, 0.45, 1.0)}
        for axis in ("x", "y", "z"):
            segs = [(s, e) for (s, e, a) in orientation_segments if a == axis]
            if not segs:
                continue
            axis_path = f"{root_path}/robot_axes_{axis}"
            axis_curves = UsdGeom.BasisCurves.Define(stage, axis_path)
            axis_curves.CreateTypeAttr("linear")
            axis_curves.CreateBasisAttr("bezier")
            axis_curves.CreateWrapAttr("nonperiodic")
            points = []
            counts = []
            for s, e in segs:
                points.append(Gf.Vec3f(float(s[0]), float(s[1]), float(s[2] + args_cli.frustum_height_offset)))
                points.append(Gf.Vec3f(float(e[0]), float(e[1]), float(e[2] + args_cli.frustum_height_offset)))
                counts.append(2)
            axis_curves.CreatePointsAttr(points)
            axis_curves.CreateCurveVertexCountsAttr(counts)
            axis_curves.CreateWidthsAttr([0.045] * len(points))
            c = axis_to_color[axis]
            UsdGeom.PrimvarsAPI(axis_curves.GetPrim()).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
            ).Set([Gf.Vec3f(c[0], c[1], c[2])] * len(segs))

    def _add_points_prim(name: str, pts: list[np.ndarray], color_rgb: tuple[float, float, float], width: float):
        if not pts:
            return
        p_path = f"{root_path}/{name}"
        p_prim = UsdGeom.Points.Define(stage, p_path)
        p_data = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2] + args_cli.path_height_offset + 0.02)) for p in pts]
        p_prim.CreatePointsAttr(p_data)
        p_prim.CreateWidthsAttr([float(width)] * len(p_data))
        UsdGeom.PrimvarsAPI(p_prim.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        ).Set([Gf.Vec3f(*color_rgb)] * len(p_data))

    # Start/end in same color as requested.
    _add_points_prim("start_points", start_points_xyz, (0.0, 1.0, 0.0), 0.13)
    _add_points_prim("end_points", end_points_xyz, (0.0, 1.0, 0.0), 0.13)

    # Setpoints as X-shape markers in a distinguishable blue.
    if goal_points_xyz:
        cross_path = f"{root_path}/goal_crosses"
        cross = UsdGeom.BasisCurves.Define(stage, cross_path)
        cross.CreateTypeAttr("linear")
        cross.CreateBasisAttr("bezier")
        cross.CreateWrapAttr("nonperiodic")
        cross_points = []
        cross_counts = []
        cross_colors = []
        for p in goal_points_xyz:
            px = float(p[0])
            py = float(p[1])
            pz = float(p[2] + args_cli.path_height_offset + 0.03)
            s = 0.12
            cross_points.extend(
                [
                    Gf.Vec3f(px - s, py - s, pz),
                    Gf.Vec3f(px + s, py + s, pz),
                    Gf.Vec3f(px - s, py + s, pz),
                    Gf.Vec3f(px + s, py - s, pz),
                ]
            )
            cross_counts.extend([2, 2])
            cross_colors.extend([Gf.Vec3f(0.05, 0.35, 1.0), Gf.Vec3f(0.05, 0.35, 1.0)])
        cross.CreatePointsAttr(cross_points)
        cross.CreateCurveVertexCountsAttr(cross_counts)
        cross.CreateWidthsAttr([0.03] * len(cross_points))
        UsdGeom.PrimvarsAPI(cross.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
        ).Set(cross_colors)

    # Crash points as X-shape markers (to match legend).
    if crash_points_xyz:
        crash_path = f"{root_path}/crash_crosses"
        crash = UsdGeom.BasisCurves.Define(stage, crash_path)
        crash.CreateTypeAttr("linear")
        crash.CreateBasisAttr("bezier")
        crash.CreateWrapAttr("nonperiodic")
        crash_points = []
        crash_counts = []
        crash_colors = []
        for p in crash_points_xyz:
            px = float(p[0])
            py = float(p[1])
            pz = float(p[2] + args_cli.path_height_offset + 0.03)
            s = 0.10
            crash_points.extend(
                [
                    Gf.Vec3f(px - s, py - s, pz),
                    Gf.Vec3f(px + s, py + s, pz),
                    Gf.Vec3f(px - s, py + s, pz),
                    Gf.Vec3f(px + s, py - s, pz),
                ]
            )
            crash_counts.extend([2, 2])
            crash_colors.extend([Gf.Vec3f(1.0, 0.13, 0.67), Gf.Vec3f(1.0, 0.13, 0.67)])
        crash.CreatePointsAttr(crash_points)
        crash.CreateCurveVertexCountsAttr(crash_counts)
        crash.CreateWidthsAttr([0.03] * len(crash_points))
        UsdGeom.PrimvarsAPI(crash.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
        ).Set(crash_colors)

    # Timeout points as yellow diamond markers.
    if timeout_points_xyz:
        timeout_path = f"{root_path}/timeout_diamonds"
        timeout_prim = UsdGeom.BasisCurves.Define(stage, timeout_path)
        timeout_prim.CreateTypeAttr("linear")
        timeout_prim.CreateBasisAttr("bezier")
        timeout_prim.CreateWrapAttr("nonperiodic")
        timeout_points = []
        timeout_counts = []
        timeout_colors = []
        for p in timeout_points_xyz:
            px = float(p[0])
            py = float(p[1])
            pz = float(p[2] + args_cli.path_height_offset + 0.03)
            s = 0.10
            timeout_points.extend(
                [
                    Gf.Vec3f(px, py - s, pz),
                    Gf.Vec3f(px + s, py, pz),
                    Gf.Vec3f(px + s, py, pz),
                    Gf.Vec3f(px, py + s, pz),
                    Gf.Vec3f(px, py + s, pz),
                    Gf.Vec3f(px - s, py, pz),
                    Gf.Vec3f(px - s, py, pz),
                    Gf.Vec3f(px, py - s, pz),
                ]
            )
            timeout_counts.extend([2, 2, 2, 2])
            timeout_colors.extend(
                [Gf.Vec3f(1.0, 0.85, 0.0), Gf.Vec3f(1.0, 0.85, 0.0), Gf.Vec3f(1.0, 0.85, 0.0), Gf.Vec3f(1.0, 0.85, 0.0)]
            )
        timeout_prim.CreatePointsAttr(timeout_points)
        timeout_prim.CreateCurveVertexCountsAttr(timeout_counts)
        timeout_prim.CreateWidthsAttr([0.03] * len(timeout_points))
        UsdGeom.PrimvarsAPI(timeout_prim.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
        ).Set(timeout_colors)

    return True


def _maybe_scale_rgb(rgb: np.ndarray) -> np.ndarray:
    scale = max(float(args_cli.image_scale), 1.0)
    if scale <= 1.0:
        return rgb
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize(new_size, Image.Resampling.LANCZOS))
    except Exception:
        # Fallback without PIL: nearest-neighbor repeat.
        sy = max(1, int(round(scale)))
        sx = max(1, int(round(scale)))
        return np.repeat(np.repeat(rgb, sy, axis=0), sx, axis=1)


def _save_rgb_with_legend(
    rgb: np.ndarray,
    output_path: str,
    schematic_triad_segments: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
    schematic_camera_shafts: list[tuple[np.ndarray, np.ndarray, int, float]] | None = None,
    schematic_camera_frustums: list[tuple[np.ndarray, np.ndarray, int, float]] | None = None,
):
    """Save Isaac image with a compact legend overlay."""
    rgb = _maybe_scale_rgb(rgb)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    ax.imshow(rgb)
    ax.set_axis_off()

    handles = []
    labels = []
    handles.append(plt.Line2D([0], [0], color="#ff1a1a", lw=2.6))
    labels.append("trajectory")
    handles.append(plt.Line2D([0], [0], marker="o", color="#00ff00", lw=0, markersize=8))
    labels.append("start/end")
    handles.append(plt.Line2D([0], [0], marker="X", color="#0066ff", lw=0, markersize=8))
    labels.append("setpoint")
    handles.append(plt.Line2D([0], [0], marker="x", color="#ff22aa", lw=0, markersize=8))
    labels.append("crash")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="D",
            markerfacecolor="#ffd400",
            markeredgecolor="#664d00",
            color="#ffd400",
            lw=0,
            markersize=7,
        )
    )
    labels.append("timeout")
    handles.append(plt.Line2D([0], [0], color="#ff0000", lw=2.2))
    labels.append("robot x-axis")
    handles.append(plt.Line2D([0], [0], color="#00ff00", lw=2.2))
    labels.append("robot y-axis")
    handles.append(plt.Line2D([0], [0], color="#3377ff", lw=2.2))
    labels.append("robot z-axis")
    for k in [1, 2, 3, 4]:
        c = _frustum_color(k, 0.0)
        handles.append(plt.Line2D([0], [0], color=(c[0], c[1], c[2]), lw=2.0))
        labels.append(f"camera axis ({k} active cams)")

    ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.92, fontsize=8)

    # Add an isolated 3D schematic inset from actual sampled geometry.
    inset = ax.inset_axes([0.02, 0.02, 0.30, 0.30], projection="3d")  # [x0, y0, w, h] in axes fraction
    inset.set_facecolor((0.0, 0.0, 0.0, 0.32))
    inset.grid(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_zticks([])
    triad = schematic_triad_segments or []
    shafts = schematic_camera_shafts or []
    frustums = schematic_camera_frustums or []
    points_xyz = []
    for s, e, _axis in triad:
        points_xyz.append(np.asarray(s, dtype=np.float32))
        points_xyz.append(np.asarray(e, dtype=np.float32))
    for s, e, _cnt, _prog in shafts:
        points_xyz.append(np.asarray(s, dtype=np.float32))
        points_xyz.append(np.asarray(e, dtype=np.float32))
    for s, e, _cnt, _prog in frustums:
        points_xyz.append(np.asarray(s, dtype=np.float32))
        points_xyz.append(np.asarray(e, dtype=np.float32))

    if points_xyz:
        pts = np.stack(points_xyz, axis=0)
        center = pts.mean(axis=0)
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        scale = max(0.25, 0.85 * span)
        inset.set_xlim(float(center[0] - scale), float(center[0] + scale))
        inset.set_ylim(float(center[1] - scale), float(center[1] + scale))
        inset.set_zlim(float(center[2] - scale), float(center[2] + scale))

    # Match requested orientation: keep robot up visually up.
    inset.view_init(elev=24.0, azim=-58.0, roll=0.0, vertical_axis="z")

    for s, e, axis in triad:
        c = {"x": "#ff0000", "y": "#00ff00", "z": "#3377ff"}[axis]
        inset.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=c, linewidth=2.4, alpha=1.0)

    # Schematic uses frustum wireframes (as requested), colored per camera frustum.
    frustum_palette = [
        (1.00, 0.40, 0.00),  # orange
        (0.00, 0.95, 0.35),  # green
        (0.12, 0.55, 1.00),  # blue
        (1.00, 0.15, 0.85),  # magenta
        (1.00, 0.92, 0.15),  # yellow
        (0.00, 0.90, 0.90),  # cyan
    ]
    # Keep a single color per frustum wireframe.
    # _camera_frustum_wireframe_segments_3d returns 8 segments per frustum.
    segments_per_frustum = 8
    for i, (s, e, _active_count, _progress) in enumerate(frustums):
        frustum_idx = i // segments_per_frustum
        c = frustum_palette[frustum_idx % len(frustum_palette)]
        inset.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=c, linewidth=1.9, alpha=0.98)

    inset.set_title("Isolated triad + frustums (3D)", color="white", fontsize=8, pad=2)

    fig.tight_layout(pad=0.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _topdown_eye_target(
    points_xy: np.ndarray,
    center_xy: np.ndarray | None = None,
    radius_override: float | None = None,
):
    if points_xy.shape[0] == 0:
        center = np.array([0.0, 0.0], dtype=np.float32)
        radius = 12.0
    else:
        xy_min = points_xy.min(axis=0)
        xy_max = points_xy.max(axis=0)
        center = 0.5 * (xy_min + xy_max) if center_xy is None else np.asarray(center_xy, dtype=np.float32)
        span = float(np.max(xy_max - xy_min))
        radius = max(18.0, 1.20 * span + 10.0)

    if radius_override is not None:
        radius = float(radius_override)

    # Pure overhead framing (same as the publication topdown plots).
    # Any lateral offset needed for look-at singularity is applied in
    # ``_set_viewport_camera_view`` as a tiny epsilon, not here.
    eye = [float(center[0]), float(center[1]), float(radius)]
    target = [float(center[0]), float(center[1]), 0.0]
    return eye, target


def _set_viewport_camera_view(eye, target, camera_prim_path: str = "/OmniverseKit_Persp") -> None:
    """Move the viewport/Replicator camera under Isaac Lab 3 headless.

    ``SimulationContext.set_camera_view`` only forwards to active visualizers. In
    HEADLESS runs there is often no Kit visualizer, so the call is a no-op and
    screenshots stay at the default far camera (tiny speck). Author the pose on
    ``/OmniverseKit_Persp`` (and via Kit/Isaac viewport utils when available).
    """
    eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
    target_t = (float(target[0]), float(target[1]), float(target[2]))

    # Avoid look-at singularity when eye and target share XY (pure top-down).
    # Tiny -Y epsilon: world +X (spawn→goal) reads left→right in the image,
    # matching the publication topdown figures. (+Y flips the scene 180°.)
    dx = eye_t[0] - target_t[0]
    dy = eye_t[1] - target_t[1]
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        eye_t = (eye_t[0], eye_t[1] - 1e-3, eye_t[2])

    # Lab 2 / Isaac Sim path (needs an active viewport; often a no-op when HEADLESS).
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from isaacsim.core.utils.viewports import set_camera_view as isaac_set_camera_view

        if get_active_viewport() is not None:
            isaac_set_camera_view(eye=eye_t, target=target_t, camera_prim_path=camera_prim_path)
            return
    except Exception:
        pass

    # Lab 3 Kit helper (needs an active viewport).
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

        if get_active_viewport() is not None:
            set_kit_renderer_camera_view(eye=eye_t, target=target_t, camera_prim_path=camera_prim_path)
            return
    except Exception:
        pass

    # Headless / no-viewport path: author USD xform on the perspective camera.
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        eye_torch = torch.tensor([eye_t], dtype=torch.float32)
        lookat_torch = torch.tensor([target_t], dtype=torch.float32)
        up_axis = UsdGeom.GetStageUpAxis(stage)
        rotation_matrix = math_utils.create_rotation_matrix_from_view(
            eye_torch, lookat_torch, up_axis=up_axis, device="cpu"
        )
        if torch.isnan(rotation_matrix).any():
            raise ValueError(f"Degenerate camera look-at eye={eye_t} target={target_t}")
        quat_xyzw = math_utils.quat_from_matrix(rotation_matrix)[0]

        camera = UsdGeom.Camera.Define(stage, camera_prim_path)
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.ClearXformOpOrder()
        camera_xform.SetResetXformStack(True)
        prim = camera.GetPrim()
        t_attr = prim.GetAttribute("xformOp:translate")
        translate_op = UsdGeom.XformOp(t_attr) if t_attr else camera_xform.AddTranslateOp()
        o_attr = prim.GetAttribute("xformOp:orient")
        orient_op = UsdGeom.XformOp(o_attr) if o_attr else camera_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        camera_xform.SetXformOpOrder([translate_op, orient_op], camera_xform.GetResetXformStack())
        translate_op.Set(Gf.Vec3d(*eye_t))
        orient_op.Set(
            Gf.Quatd(
                float(quat_xyzw[3]),
                Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
            )
        )
    except Exception as e:
        print(f"[WARNING] Failed to author viewport camera pose on {camera_prim_path}: {e}")


def _side_eye_target(center_xyz: np.ndarray, span_xy: float):
    center = np.asarray(center_xyz, dtype=np.float32)
    span_xy = max(float(span_xy), 6.0)
    # Side view aligned with long side walls (XZ plane): look along +/-Y toward center.
    side_sign = -1.0 if SIDE_VIEW_CAMERA_SIDE == "neg_y" else 1.0
    eye = [
        float(center[0]),
        float(center[1] + side_sign * 1.12 * span_xy),
        float(center[2] + 0.65 * span_xy),
    ]
    # Slight upward tilt (less downward-looking).
    target = [float(center[0]), float(center[1]), float(center[2] + 0.16 * span_xy)]
    return eye, target


def _resolve_env_size_xyz(raw_env) -> tuple[float, float, float] | None:
    """Return (sx, sy, sz) from reset_obstacles, supporting scalar or tuple env_size."""
    try:
        reset_obst = getattr(getattr(raw_env.unwrapped.cfg, "events", None), "reset_obstacles", None)
        if reset_obst is None:
            return None
        env_size = reset_obst.params.get("env_size", None)
        if env_size is None:
            return None
        if isinstance(env_size, (list, tuple)):
            if len(env_size) < 2:
                return None
            sx = float(env_size[0])
            sy = float(env_size[1])
            sz = float(env_size[2]) if len(env_size) > 2 else 0.0
            return sx, sy, sz
        v = float(env_size)
        return v, v, v
    except Exception:
        return None


def _xy_span_from_walls(scene) -> float | None:
    """Full XY span from opposite wall prims, if available."""
    pairs = [
        ("obstacle_left_wall", "obstacle_right_wall"),
        ("obstacle_front_wall", "obstacle_back_wall"),
    ]
    spans = []
    for a, b in pairs:
        if a not in scene.keys() or b not in scene.keys():
            continue
        pa = getattr(getattr(scene[a], "data", None), "root_pos_w", None)
        pb = getattr(getattr(scene[b], "data", None), "root_pos_w", None)
        if pa is None or pb is None or pa.ndim < 2 or pb.ndim < 2:
            continue
        spans.append(float(torch.norm(pa[0, :2] - pb[0, :2]).item()))
    if not spans:
        return None
    return max(spans)


def _compute_full_env_topdown_radius(raw_env, obstacle_xy: np.ndarray) -> float:
    """Camera height for top-down capture.

    Keep the camera relatively high so 0.4 m camera-axis overlays read at the
    same on-screen size as the publication figures. FOV is tightened separately
    in ``_set_viewport_camera_view`` so the rectangular room still fills the frame.
    """
    scene = raw_env.unwrapped.scene
    wall_span = _xy_span_from_walls(scene)
    size = _resolve_env_size_xyz(raw_env)
    if size is not None:
        span = max(float(size[0]), float(size[1]))
    elif wall_span is not None:
        span = float(wall_span)
    elif obstacle_xy.shape[0] > 0:
        span = float(np.max(obstacle_xy.max(axis=0) - obstacle_xy.min(axis=0)))
    else:
        span = 12.0

    # Publication figures used ~19 m for a ~12 m room; scale with the long side.
    return max(12.0, 1.15 * span + 2.5)


_USD_DEFAULT_VERTICAL_APERTURE_MM = 15.2908


def _set_camera_focal_to_fit_room(
    camera_prim_path: str,
    eye_height: float,
    span_x: float,
    span_y: float,
    fill: float = 0.96,
    aspect: float = 16.0 / 9.0,
) -> None:
    """Tighten OmniverseKit_Persp FOV so ``span_x`` x ``span_y`` fills the frame."""
    import math

    import omni.usd
    from pxr import UsdGeom

    h = max(abs(float(eye_height)), 1e-3)
    # With the +Y nadir epsilon, world +X (long room axis) maps to image width.
    need_tan_v = (float(span_y) / fill) / (2.0 * h)
    need_tan_h = (float(span_x) / fill) / (2.0 * h)
    tan_half_v = max(need_tan_v, need_tan_h / aspect)
    tan_half_v = float(np.clip(tan_half_v, math.tan(math.radians(4.0)), math.tan(math.radians(35.0))))
    focal_mm = _USD_DEFAULT_VERTICAL_APERTURE_MM / (2.0 * tan_half_v)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    cam = UsdGeom.Camera.Get(stage, camera_prim_path)
    if not cam:
        cam = UsdGeom.Camera.Define(stage, camera_prim_path)
    # Only adjust focal length; keep existing apertures.
    cam.GetFocalLengthAttr().Set(float(focal_mm))
    print(
        f"[visualize_trajectories] viewport focal={focal_mm:.2f} mm "
        f"(eye_z={h:.2f}, span=({span_x:.2f},{span_y:.2f}), fill={fill})"
    )


def _room_spans_xy(raw_env, obstacle_xy: np.ndarray) -> tuple[float, float]:
    size = _resolve_env_size_xyz(raw_env)
    if size is not None:
        return float(size[0]), float(size[1])
    wall_span = _xy_span_from_walls(raw_env.unwrapped.scene)
    if obstacle_xy.shape[0] > 0:
        extents = obstacle_xy.max(axis=0) - obstacle_xy.min(axis=0)
        return float(extents[0]), float(extents[1])
    if wall_span is not None:
        return float(wall_span), float(wall_span)
    return 12.0, 8.0


def _compute_full_env_span(raw_env, obstacle_xy: np.ndarray) -> float:
    """Compute horizontal env span for side-view framing."""
    size = _resolve_env_size_xyz(raw_env)
    if size is not None:
        return max(float(size[0]), float(size[1]))
    wall_span = _xy_span_from_walls(raw_env.unwrapped.scene)
    if wall_span is not None:
        return float(wall_span)
    if obstacle_xy.shape[0] > 0:
        return float(np.max(obstacle_xy.max(axis=0) - obstacle_xy.min(axis=0)))
    return 12.0


def _compute_env_center_from_walls(scene) -> np.ndarray | None:
    wall_names = ["obstacle_left_wall", "obstacle_right_wall", "obstacle_front_wall", "obstacle_back_wall"]
    pts = []
    for name in wall_names:
        if name not in scene.keys():
            continue
        pos = getattr(getattr(scene[name], "data", None), "root_pos_w", None)
        if pos is None or pos.ndim < 2:
            continue
        pts.append(pos[0, :2].detach().cpu().numpy())
    if len(pts) >= 2:
        arr = np.asarray(pts, dtype=np.float32)
        return arr.mean(axis=0)
    return None


def _compute_env_center_xyz_from_walls(scene) -> np.ndarray | None:
    wall_names = [
        "obstacle_left_wall",
        "obstacle_right_wall",
        "obstacle_front_wall",
        "obstacle_back_wall",
        "obstacle_bottom_wall",
        "obstacle_top_wall",
    ]
    pts = []
    for name in wall_names:
        if name not in scene.keys():
            continue
        pos = getattr(getattr(scene[name], "data", None), "root_pos_w", None)
        if pos is None or pos.ndim < 2:
            continue
        pts.append(pos[0, :3].detach().cpu().numpy())
    if len(pts) >= 2:
        arr = np.asarray(pts, dtype=np.float32)
        return arr.mean(axis=0)
    return None


def _hide_builtin_command_visuals():
    """Hide built-in command visuals that may not align with logged start/goal markers."""
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/Visuals/Command/body_pose" in path or "/Visuals/Command/goal_pose" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
    except Exception as e:
        print(f"[WARNING] Could not hide built-in command visuals: {e}")


def _hide_robot_visuals():
    """Hide robot visuals for clean trajectory/camera screenshots."""
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/Robot" in path:
                UsdGeom.Imageable(prim).MakeInvisible()
    except Exception as e:
        print(f"[WARNING] Could not hide robot visuals: {e}")


def _save_rgb_image(rgb: np.ndarray, output_path: str):
    rgb = _maybe_scale_rgb(rgb)
    try:
        from PIL import Image

        img = Image.fromarray(rgb)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.save(output_path)
        return
    except Exception:
        pass

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.imsave(output_path, rgb)


# Lab 3 removed env.render(rgb_array) (always returns None). Cache a Replicator
# annotator on the viewport camera so we can still capture Isaac RGB + USD overlays.
_VIEWPORT_RGB_STATE: dict = {
    "annotator": None,
    "render_product": None,
    "camera_path": None,
    "resolution": None,
    "quality_configured": False,
}


def _configure_capture_render_quality(total_spp: int = 512, spp_per_frame: int = 8) -> None:
    """Force PathTracing and enough samples for a clean still."""
    state = _VIEWPORT_RGB_STATE
    total_spp = max(64, int(total_spp))
    spp_per_frame = max(1, int(spp_per_frame))
    try:
        import omni.replicator.core as rep

        # Sets rendermode=PathTracing and totalSpp.
        rep.settings.set_render_pathtraced(samples_per_pixel=total_spp)
    except Exception as e:
        print(f"[WARNING] set_render_pathtraced failed: {e}")
    try:
        import carb

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", spp_per_frame)
        settings.set("/rtx/pathtracing/totalSpp", total_spp)
        settings.set("/rtx/pathtracing/clampSpp", total_spp)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set("/rtx/pathtracing/optixDenoiser/temporalMode/enabled", False)
        settings.set("/rtx/post/tvNoise/enabled", False)
        if not state.get("quality_logged"):
            print(
                f"[INFO] Viewport capture render mode: {settings.get('/rtx/rendermode')!r} "
                f"spp/frame={settings.get('/rtx/pathtracing/spp')} "
                f"totalSpp={settings.get('/rtx/pathtracing/totalSpp')}"
            )
            state["quality_logged"] = True
    except Exception as e:
        print(f"[WARNING] PathTracing carb configure failed: {e}")
    # Frames needed ≈ totalSpp / spp_per_frame (+ margin for camera settle).
    state["capture_spp"] = int(np.ceil(total_spp / float(spp_per_frame))) + 16
    state["quality_configured"] = True


def _viewport_camera_and_resolution(raw_env) -> tuple[str, tuple[int, int]]:
    viewer = getattr(getattr(raw_env, "unwrapped", raw_env).cfg, "viewer", None)
    cam_path = getattr(viewer, "cam_prim_path", None) or "/OmniverseKit_Persp"
    resolution = getattr(viewer, "resolution", None) or (1280, 720)
    return str(cam_path), (int(resolution[0]), int(resolution[1]))


def _try_visualizer_rgb(raw_env) -> np.ndarray | None:
    """Prefer an active Kit/Newton visualizer's render_rgb_array() when present."""
    sim = getattr(getattr(raw_env, "unwrapped", raw_env), "sim", None)
    visualizers = getattr(sim, "visualizers", None) if sim is not None else None
    if not visualizers:
        return None
    for viz in visualizers:
        render_fn = getattr(viz, "render_rgb_array", None)
        if render_fn is None:
            continue
        try:
            frame = render_fn()
        except Exception as e:
            print(f"[WARNING] visualizer.render_rgb_array() failed: {e}")
            continue
        if frame is None:
            continue
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] >= 3 and arr.size > 0:
            return arr[:, :, :3].astype(np.uint8, copy=False)
    return None


def _capture_viewport_rgb(raw_env, warmup_renders: int = 24) -> np.ndarray | None:
    """Capture an RGB frame from the Isaac viewport camera (Lab 3 compatible).

    Isaac Lab 3 deprecated ``render_mode='rgb_array'`` (``env.render()`` always
    returns ``None``). The publication visualizations used the viewport camera
    with USD trajectory overlays baked into the stage — recreate that by reading
    Replicator RGB from ``/OmniverseKit_Persp`` (or ``env.cfg.viewer.cam_prim_path``).
    """
    _configure_capture_render_quality()
    # Warm up / apply latest set_camera_view before reading pixels.
    # Do not return early from the Kit visualizer here — under broken NGX that
    # path is typically a single under-sampled RealTimePathTracing frame.
    sim = getattr(getattr(raw_env, "unwrapped", raw_env), "sim", None)
    for _ in range(max(1, min(8, int(warmup_renders)))):
        if sim is not None:
            sim.render()

    camera_path, resolution = _viewport_camera_and_resolution(raw_env)
    w, h = resolution
    try:
        import omni.kit.app
        import omni.replicator.core as rep
        from isaaclab.app.settings_manager import get_settings_manager
    except Exception as e:
        print(f"[WARNING] Viewport RGB imports unavailable: {e}")
        return _try_visualizer_rgb(raw_env)

    state = _VIEWPORT_RGB_STATE
    try:
        if (
            state["annotator"] is None
            or state["camera_path"] != camera_path
            or state["resolution"] != resolution
        ):
            if state["annotator"] is not None:
                try:
                    state["annotator"].detach()
                except Exception:
                    pass
            if state["render_product"] is not None:
                try:
                    state["render_product"].destroy()
                except Exception:
                    pass
            state["render_product"] = rep.create.render_product(camera_path, (w, h))
            state["annotator"] = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            state["annotator"].attach([state["render_product"]])
            state["camera_path"] = camera_path
            state["resolution"] = resolution

        # Pump the app with physics paused so PathTracing can accumulate spp.
        settings = get_settings_manager()
        play_flag = settings.get("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        rgb = None
        spp = int(state.get("capture_spp") or 24)
        accumulate_frames = max(int(warmup_renders), spp)
        try:
            for _ in range(accumulate_frames):
                if sim is not None:
                    sim.render()
                omni.kit.app.get_app().update()
            # Read once after accumulation / AA settle — mid-loop reads can
            # interrupt PathTracing convergence.
            raw = state["annotator"].get_data()
            if isinstance(raw, dict):
                raw = raw.get("data", np.array([], dtype=np.uint8))
            raw = np.asarray(raw, dtype=np.uint8)
            if raw.size > 0:
                if raw.ndim == 1:
                    raw = raw.reshape(h, w, -1)
                if raw.ndim == 3 and raw.shape[-1] >= 3:
                    candidate = raw[:, :, :3]
                    if _rgb_looks_valid(candidate):
                        rgb = candidate
        finally:
            settings.set_bool("/app/player/playSimulations", bool(play_flag))

        return rgb
    except Exception as e:
        print(f"[WARNING] Viewport RGB capture failed: {e}")
        return _try_visualizer_rgb(raw_env)


def _rgb_looks_valid(rgb: np.ndarray) -> bool:
    """Reject blank / single-color warm-up frames from the RTX annotator."""
    if rgb is None or rgb.size == 0 or rgb.ndim != 3 or rgb.shape[-1] < 3:
        return False
    # Need some non-zero content and meaningful spatial variation.
    flat = rgb.reshape(-1, rgb.shape[-1]).astype(np.float32)
    if not np.any(flat):
        return False
    # Per-channel std over a subsample — uniform floor/sky fills fail this.
    sample = flat[:: max(1, flat.shape[0] // 5000)]
    return float(sample.std()) > 8.0


def _create_agent_and_env():
    if args_cli.model not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model '{args_cli.model}'.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    # Keep environment RNG tied to episode seed path. obstacle_seed is applied only for first-layout sampling.
    env_cfg.seed = args_cli.seed
    configure_active_cameras(env_cfg, args_cli.active_cameras)

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

    experiment_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    if not isinstance(experiment_cfg, dict):
        raise RuntimeError(f"Loaded agent config from '{args_cli.agent}' is not a YAML dictionary.")

    # Inference: don't open TensorBoard / checkpoint dirs (agent.init does that when intervals > 0).
    experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    model_class = MODEL_CLASSES[args_cli.model]
    device = env.device
    observation_space = env.observation_space
    action_space = env.action_space

    policy_kwargs = dict(experiment_cfg["models"]["policy"].get("kwargs", {}))
    policy_kwargs.setdefault("num_envs", env.num_envs)

    def _dbg(msg: str) -> None:
        print(f"[visualize_trajectories] {msg}", flush=True)

    _dbg(f"building model on {device} …")
    model = model_class(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **policy_kwargs,
    )
    _dbg("model built")
    models = {"policy": model, "value": model}

    memory_cfg = experiment_cfg.get("memory", {})
    memory_size = memory_cfg.get("memory_size", -1)
    if memory_size < 0:
        memory_size = experiment_cfg["agent"]["rollouts"]
    _dbg(f"building RandomMemory size={memory_size} …")
    memory = RandomMemory(num_envs=env.num_envs, device=device, memory_size=memory_size)
    _dbg("memory built")

    # build_ppo_cfg must see write_interval=0 so PPO_CFG / agent.init skip TensorBoard.
    ppo_source = dict(experiment_cfg["agent"])
    exp = dict(ppo_source.get("experiment") or {})
    exp["write_interval"] = 0
    exp["checkpoint_interval"] = 0
    ppo_source["experiment"] = exp

    _dbg("build_ppo_cfg …")
    ppo_cfg, _ = build_ppo_cfg(
        ppo_source,
        observation_space=observation_space,
        device=device,
    )
    _dbg("ppo_cfg ready")

    agent_class = PPO_RNN if args_cli.model in {"spherical_range_cnn_gru"} else PPO
    _dbg(f"constructing {agent_class.__name__} …")
    agent = agent_class(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )
    _dbg("agent.init() …")
    try:
        # Breadcrumbs inside skrl memory.create_tensor (where Dict obs can fail hard).
        _orig_create = memory.create_tensor

        def _create_tensor(name, *args, **kwargs):
            _dbg(f"  memory.create_tensor({name!r}) …")
            out = _orig_create(name, *args, **kwargs)
            _dbg(f"  memory.create_tensor({name!r}) ok")
            return out

        memory.create_tensor = _create_tensor  # type: ignore[method-assign]
        agent.init()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        memory.create_tensor = _orig_create  # type: ignore[method-assign]
    _dbg(f"agent.load({args_cli.checkpoint}) …")
    agent.load(retrieve_file_path(args_cli.checkpoint))
    agent.enable_training_mode(False)
    _dbg("agent ready")

    return raw_env, env, agent


def _freeze_obstacles_after_first_reset(raw_env):
    """Let first reset place obstacles, then prevent further obstacle re-randomization."""
    event_manager = getattr(raw_env.unwrapped, "event_manager", None)
    if event_manager is None:
        print("[WARNING] No event_manager found; cannot freeze obstacle layout.")
        return False
    try:
        cfg = event_manager.get_term_cfg("reset_obstacles")
    except Exception:
        print("[WARNING] reset_obstacles term not found; obstacle layout freeze skipped.")
        return False

    # Most robust freeze: replace event function with no-op after first layout sample.
    def _noop_reset_obstacles(*args, **kwargs):
        return None

    try:
        cfg.func = _noop_reset_obstacles
        if hasattr(cfg, "min_step_count_between_reset"):
            cfg.min_step_count_between_reset = int(1e12)
        event_manager.set_term_cfg("reset_obstacles", cfg)
        # Keep config object in sync for any downstream accesses.
        if hasattr(raw_env.unwrapped, "cfg") and hasattr(raw_env.unwrapped.cfg, "events"):
            setattr(raw_env.unwrapped.cfg.events, "reset_obstacles", cfg)
        return True
    except Exception as e:
        print(f"[WARNING] Could not freeze reset_obstacles term: {e}")
        return False


def _capture_obstacle_reference_poses(raw_env) -> dict[str, torch.Tensor]:
    """Capture obstacle root poses (xyz + quat) for env 0 as reference."""
    ref: dict[str, torch.Tensor] = {}
    scene = raw_env.unwrapped.scene
    for key in scene.keys():
        asset = scene[key]
        data = getattr(asset, "data", None)
        # RigidObjectCollection path (used by drone ARL obstacle scene).
        obj_pos = getattr(data, "object_pos_w", None)
        obj_quat = getattr(data, "object_quat_w", None)
        if obj_pos is not None and obj_quat is not None and obj_pos.ndim >= 3 and obj_quat.ndim >= 3:
            if "obstacle" in key or "object_collection" in key:
                pose = torch.cat([obj_pos[0:1], obj_quat[0:1]], dim=-1).clone()
                ref[key] = pose
            continue

        if "obstacle" not in key:
            continue
        pos = getattr(data, "root_pos_w", None)
        quat = getattr(data, "root_quat_w", None)
        if pos is None or quat is None or pos.ndim < 2 or quat.ndim < 2:
            continue
        pose = torch.cat([pos[0:1], quat[0:1]], dim=-1).clone()
        ref[key] = pose
    return ref


def _restore_obstacle_reference_poses(raw_env, ref: dict[str, torch.Tensor]) -> None:
    """Restore obstacle root poses for env 0 from a captured reference."""
    scene = raw_env.unwrapped.scene
    for key, pose in ref.items():
        if key not in scene.keys():
            continue
        asset = scene[key]
        # RigidObjectCollection restore.
        if hasattr(asset, "write_object_pose_to_sim"):
            try:
                asset.write_object_pose_to_sim(pose)
                if hasattr(asset, "write_object_velocity_to_sim"):
                    zeros_vel = torch.zeros((pose.shape[0], pose.shape[1], 6), device=pose.device, dtype=pose.dtype)
                    asset.write_object_velocity_to_sim(zeros_vel)
                continue
            except Exception:
                pass
        if hasattr(asset, "write_root_pose_to_sim"):
            asset.write_root_pose_to_sim(pose)
        if hasattr(asset, "write_root_velocity_to_sim"):
            zeros_vel = torch.zeros((pose.shape[0], 6), device=pose.device, dtype=pose.dtype)
            asset.write_root_velocity_to_sim(zeros_vel)


def _refresh_cameras_after_obstacle_restore(raw_env, camera_names: list[str]) -> None:
    """Best-effort camera/sensor refresh after obstacle pose restore."""
    try:
        raw_env.unwrapped.scene.write_data_to_sim()
        raw_env.unwrapped.sim.step(render=False)
        raw_env.unwrapped.scene.update(dt=raw_env.unwrapped.step_dt)
        for cam_name in camera_names:
            if cam_name in raw_env.unwrapped.scene.keys():
                cam = raw_env.unwrapped.scene[cam_name]
                if hasattr(cam, "update"):
                    cam.update(dt=raw_env.unwrapped.step_dt, force_recompute=True)
    except Exception:
        pass


def _reset_env_with_seed(env, raw_env, seed: int):
    """Reset env with explicit seed using wrapper-safe paths only."""
    # Always seed the raw env RNG first; many reset terms draw from this RNG path.
    if hasattr(raw_env.unwrapped, "seed"):
        try:
            raw_env.unwrapped.seed(int(seed))
        except Exception:
            pass

    try:
        return env.reset(seed=seed)
    except TypeError:
        pass
    except Exception:
        # Keep trying safe fallbacks below.
        pass

    # Avoid calling raw_env.reset() directly: it can desync wrapper state.
    return env.reset()


def _apply_reset_terms_with_seed(
    raw_env,
    seed: int,
    include_terms: set[str] | None = None,
    exclude_terms: set[str] | None = None,
) -> None:
    """Apply reset-mode event terms deterministically with an explicit seed."""
    event_manager = getattr(raw_env.unwrapped, "event_manager", None)
    if event_manager is None or "reset" not in getattr(event_manager, "active_terms", {}):
        return

    _set_seed(int(seed))
    if hasattr(raw_env.unwrapped, "seed"):
        try:
            raw_env.unwrapped.seed(int(seed))
        except Exception:
            pass

    env_ids = torch.tensor([0], dtype=torch.int64, device=raw_env.unwrapped.device)
    reset_terms = list(event_manager.active_terms.get("reset", []))
    for term_name in reset_terms:
        if include_terms is not None and term_name not in include_terms:
            continue
        if exclude_terms is not None and term_name in exclude_terms:
            continue
        try:
            cfg = event_manager.get_term_cfg(term_name)
            cfg.func(raw_env.unwrapped, env_ids, **cfg.params)
        except Exception as e:
            print(f"[WARNING] Could not apply reset term '{term_name}' with explicit seed: {e}")

    try:
        raw_env.unwrapped.scene.write_data_to_sim()
        raw_env.unwrapped.sim.forward()
        raw_env.unwrapped.scene.update(dt=raw_env.unwrapped.step_dt)
    except Exception:
        pass


def _obstacle_pose_deviation(raw_env, ref: dict[str, torch.Tensor]) -> float:
    """Return max L2 position deviation [m] from reference obstacle poses."""
    scene = raw_env.unwrapped.scene
    max_dev = 0.0
    for key, pose_ref in ref.items():
        if key not in scene.keys():
            continue
        data = getattr(scene[key], "data", None)
        obj_pos = getattr(data, "object_pos_w", None)
        if obj_pos is not None and obj_pos.ndim >= 3 and pose_ref.ndim >= 3:
            dev = torch.norm(obj_pos[0] - pose_ref[0, :, :3], dim=1).max().item()
            max_dev = max(max_dev, float(dev))
            continue
        pos = getattr(data, "root_pos_w", None)
        if pos is None or pos.ndim < 2:
            continue
        dev = torch.norm(pos[0] - pose_ref[0, :3]).item()
        max_dev = max(max_dev, float(dev))
    return max_dev


def main():
    if args_cli.success_radius <= 0:
        raise ValueError(f"--success_radius must be > 0, got {args_cli.success_radius}")

    def _dbg(msg: str) -> None:
        print(f"[visualize_trajectories] {msg}", flush=True)

    global ACTIVE_SYSTEM_CAMERA_COUNT
    _dbg("main: creating agent/env …")
    raw_env, env, agent = _create_agent_and_env()
    _dbg("main: agent/env ready")
    scene = raw_env.unwrapped.scene
    camera_names = _detect_camera_names(scene)
    _dbg(f"main: cameras={camera_names}")
    ACTIVE_SYSTEM_CAMERA_COUNT = max(len(camera_names), 1)
    run_folder, plot_path, stats_path, analysis_root = _prepare_output_paths()
    _dbg(f"main: output → {run_folder}")

    all_path_points_xyz: list[np.ndarray] = []
    all_path_points_xy: list[np.ndarray] = []
    episode_paths_xyz: list[list[np.ndarray]] = []
    all_frustum_segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    all_orientation_segments: list[tuple[np.ndarray, np.ndarray, str]] = []
    schematic_triad_segments: list[tuple[np.ndarray, np.ndarray, str]] = []
    schematic_camera_shafts: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    schematic_camera_frustum_segments: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    start_points_xyz: list[np.ndarray] = []
    goal_points_xyz: list[np.ndarray] = []
    end_points_xyz: list[np.ndarray] = []
    crash_points_xyz: list[np.ndarray] = []
    timeout_points_xyz: list[np.ndarray] = []
    debug_draw_used = omni_debug_draw is not None
    stage_overlay_used = False
    obstacles_frozen = False
    obstacle_ref_poses: dict[str, torch.Tensor] = {}
    obstacle_max_dev_m = 0.0

    # Metrics aligned with evaluate_skrl.py (success/crash are not mutually exclusive).
    episodes_done = 0
    crash_count = 0
    timeout_marker_count = 0
    success_count = 0
    success_time_sum_s = 0.0
    success_path_sum_m = 0.0
    mean_angvel_sum = 0.0
    mean_coverage_sum = 0.0
    coverage_available = False

    try:
        dt = float(getattr(raw_env.unwrapped, "step_dt", getattr(raw_env, "step_dt", 0.0)))
        # First reset: build obstacle layout using obstacle_seed only.
        obstacle_seed = args_cli.obstacle_seed if args_cli.obstacle_seed is not None else args_cli.seed
        _dbg(f"main: first reset (obstacle_seed={obstacle_seed}) …")
        _set_seed(obstacle_seed)
        _reset_env_with_seed(env, raw_env, obstacle_seed)
        _dbg("main: first reset done")
        _apply_reset_terms_with_seed(raw_env, obstacle_seed, include_terms={"reset_obstacles"})
        obstacles_frozen = _freeze_obstacles_after_first_reset(raw_env)
        obstacle_ref_poses = _capture_obstacle_reference_poses(raw_env)
        _dbg(f"main: obstacles frozen={obstacles_frozen}; starting {args_cli.num_episodes} episodes")

        for ep in range(args_cli.num_episodes):
            _dbg(f"main: episode {ep + 1}/{args_cli.num_episodes}")
            episode_seed = args_cli.seed + ep
            _set_seed(episode_seed)
            # Always reset episode with episode seed so --seed changes robot initialization.
            obs, _ = _reset_env_with_seed(env, raw_env, episode_seed)
            # Force non-obstacle reset randomization from episode_seed (robot/cameras/commands),
            # independent from obstacle_seed used for layout.
            _apply_reset_terms_with_seed(raw_env, episode_seed, exclude_terms={"reset_obstacles"})
            # Keep obstacle layout fixed by restoring captured first-layout poses.
            if obstacle_ref_poses:
                _restore_obstacle_reference_poses(raw_env, obstacle_ref_poses)
                _refresh_cameras_after_obstacle_restore(raw_env, camera_names)
                obstacle_max_dev_m = max(obstacle_max_dev_m, _obstacle_pose_deviation(raw_env, obstacle_ref_poses))

            ep_path = []
            robot0 = raw_env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            start_points_xyz.append(robot0)
            target_pose0 = raw_env.unwrapped.command_manager.get_command("target_pose")[0, :3]
            env_origin0 = raw_env.unwrapped.scene.env_origins[0]
            goal_points_xyz.append((target_pose0 + env_origin0).detach().cpu().numpy())
            ep_steps = 0
            ep_path_m = 0.0
            ep_angvel_sum = 0.0
            ep_coverage_sum = 0.0
            ep_reached_goal = False
            # Same spawn latch as evaluate_skrl: ignore goal checks until dist
            # has been observed outside the success radius at least once.
            ep_reach_armed = False
            ep_first_reach_step = 0
            ep_first_reach_path = 0.0
            dist_to_goal = float("inf")
            terminated_last = False
            truncated_last = False
            for step in range(args_cli.max_steps):
                robot = raw_env.unwrapped.scene["robot"]
                pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()
                ep_path.append(pos_w)
                all_path_points_xyz.append(pos_w)
                all_path_points_xy.append(pos_w[:2])

                robot_positions, target_positions = _extract_robot_and_goal_positions(raw_env)
                # Pre-step distance (evaluate_skrl uses this as final_error after done,
                # because step() auto-resets finished envs).
                dist_to_goal = float(torch.norm(robot_positions - target_positions, dim=1)[0].item())
                lin_speed = float(torch.norm(robot.data.root_lin_vel_w, dim=1)[0].item())
                ang_speed = float(torch.norm(robot.data.root_ang_vel_w, dim=1)[0].item())
                ep_steps += 1
                if dt > 0.0:
                    ep_path_m += lin_speed * dt
                ep_angvel_sum += ang_speed
                coverage = _canonicalized_coverage_per_env(raw_env, obs, env.observation_space)
                if coverage is not None:
                    coverage_available = True
                    ep_coverage_sum += float(coverage[0].item())

                if dist_to_goal > float(args_cli.success_radius):
                    ep_reach_armed = True
                if (
                    ep_reach_armed
                    and (not ep_reached_goal)
                    and (dist_to_goal <= float(args_cli.success_radius))
                ):
                    ep_reached_goal = True
                    ep_first_reach_step = ep_steps
                    ep_first_reach_path = ep_path_m

                if args_cli.frustum_stride > 0 and step % args_cli.frustum_stride == 0:
                    active_count = len(camera_names)
                    if hasattr(raw_env.unwrapped, "_active_camera_mask"):
                        try:
                            active_mask = raw_env.unwrapped._active_camera_mask[0]
                            active_count = int(active_mask.sum().item())
                        except Exception:
                            active_count = len(camera_names)

                    # Keep camera-arrow plotting consistent with policy input: only active cameras.
                    if hasattr(raw_env.unwrapped, "_active_camera_mask"):
                        try:
                            active_mask = raw_env.unwrapped._active_camera_mask[0]
                            active_camera_names = [name for i, name in enumerate(camera_names) if i < len(active_mask) and bool(active_mask[i].item())]
                        except Exception:
                            active_camera_names = list(camera_names)
                    else:
                        active_camera_names = list(camera_names)

                    if active_camera_names:
                        capture_schematic_this_sample = (not schematic_triad_segments) and (not schematic_camera_shafts)
                        triad_segments = _robot_orientation_frame_segments_3d(
                            raw_env,
                            env_idx=0,
                            axis_length=0.32,
                        )
                        all_orientation_segments.extend(triad_segments)
                        if capture_schematic_this_sample:
                            schematic_triad_segments = [(s.copy(), e.copy(), a) for (s, e, a) in triad_segments]

                    progress = (
                        (float(ep) * float(args_cli.max_steps) + float(step))
                        / max(float(args_cli.num_episodes * args_cli.max_steps - 1), 1.0)
                    )
                    for cam_name in active_camera_names:
                        cam_segments = _camera_frustum_segments_3d(
                            raw_env,
                            camera_name=cam_name,
                            active_camera_count=active_count,
                            frustum_distance=args_cli.frustum_distance,
                            progress=progress,
                            env_idx=0,
                        )
                        all_frustum_segments.extend(cam_segments)
                        if capture_schematic_this_sample and cam_segments:
                            # First segment is the principal-axis shaft for this camera.
                            s0, e0, cnt0, p0 = cam_segments[0]
                            schematic_camera_shafts.append((s0.copy(), e0.copy(), int(cnt0), float(p0)))
                            schematic_camera_frustum_segments.extend(
                                _camera_frustum_wireframe_segments_3d(
                                    raw_env,
                                    camera_name=cam_name,
                                    active_camera_count=active_count,
                                    frustum_distance=args_cli.frustum_distance,
                                    progress=progress,
                                    env_idx=0,
                                )
                            )

                with torch.inference_mode():
                    outputs = agent.act(
                        obs, env.state() if hasattr(env, "state") else None, timestep=0, timesteps=0
                    )
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    obs, _, terminated, truncated, _ = env.step(actions)
                    done = bool((terminated | truncated)[0].item())
                    terminated_last = bool(terminated[0].item())
                    truncated_last = bool(truncated[0].item())

                    if hasattr(agent, "_rnn") and agent._rnn and done:
                        for rnn_state in agent._rnn_final_states["policy"]:
                            rnn_state[:, 0] = 0
                        agent._rnn_initial_states = agent._rnn_final_states

                if done:
                    break

            # Metric definitions match evaluate_skrl.py:
            # - crash: terminated and not truncated
            # - success: is_episode_success(reached_during_episode, final_error, radius)
            #   (independent of crash; final_error is last pre-step distance)
            was_crash = bool(terminated_last and not truncated_last)
            final_error_m = float(dist_to_goal)
            was_success = is_episode_success(
                bool(ep_reached_goal),
                final_error_m,
                float(args_cli.success_radius),
            )
            # Plot-only residual marker (evaluate_skrl does not report timeout rate).
            plot_timeout = bool((not was_crash) and (not was_success))
            crash_count += int(was_crash)
            success_count += int(was_success)
            timeout_marker_count += int(plot_timeout)
            mean_angvel_sum += float(ep_angvel_sum) / float(max(ep_steps, 1))
            mean_coverage_sum += float(ep_coverage_sum) / float(max(ep_steps, 1))
            if was_success:
                success_time_sum_s += float(ep_first_reach_step) * dt
                success_path_sum_m += float(ep_first_reach_path)
            episodes_done += 1
            if ep_path:
                end_points_xyz.append(ep_path[-1].copy())
                episode_paths_xyz.append([p.copy() for p in ep_path])
                if was_crash:
                    crash_points_xyz.append(ep_path[-1].copy())
                elif plot_timeout:
                    timeout_points_xyz.append(ep_path[-1].copy())

            outcome_bits = []
            if was_crash:
                outcome_bits.append("crash")
            if was_success:
                outcome_bits.append("success")
            if plot_timeout:
                outcome_bits.append("timeout")
            outcome = "+".join(outcome_bits) if outcome_bits else "unknown"
            print(
                f"[INFO] Episode {ep + 1}/{args_cli.num_episodes} finished after {len(ep_path)} steps. "
                f"outcome={outcome} final_error={final_error_m:.3f}m"
            )

        all_xy = np.array(all_path_points_xy, dtype=np.float32) if all_path_points_xy else np.zeros((0, 2), dtype=np.float32)
        all_xyz = np.array(all_path_points_xyz, dtype=np.float32) if all_path_points_xyz else np.zeros((0, 3), dtype=np.float32)
        obstacle_xy = _collect_obstacle_xy(scene, env_idx=0)
        if obstacle_xy.shape[0] > 0:
            all_xy = np.concatenate([all_xy, obstacle_xy], axis=0) if all_xy.shape[0] > 0 else obstacle_xy

        _hide_builtin_command_visuals()
        _hide_robot_visuals()
        _set_wall_visibility(hide_top=bool(args_cli.hide_top_wall))
        stage_overlay_used = _add_stage_line_overlays(
            episode_paths_xyz,
            all_frustum_segments,
            all_orientation_segments,
            start_points_xyz,
            goal_points_xyz,
            end_points_xyz,
            crash_points_xyz,
            timeout_points_xyz,
        )
        _draw_debug_lines(episode_paths_xyz, all_frustum_segments, all_orientation_segments)

        # Center top-down view on environment center (walls), fallback to env origin.
        view_xy = obstacle_xy if obstacle_xy.shape[0] > 0 else all_xy
        env_center_xy = _compute_env_center_from_walls(scene)
        if env_center_xy is None:
            env_center_xy = raw_env.unwrapped.scene.env_origins[0, :2].detach().cpu().numpy()
        env_radius = _compute_full_env_topdown_radius(raw_env, obstacle_xy)
        eye, target = _topdown_eye_target(view_xy, center_xy=env_center_xy, radius_override=env_radius)
        span_x, span_y = _room_spans_xy(raw_env, obstacle_xy)
        raw_env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        _set_viewport_camera_view(eye, target)
        _set_camera_focal_to_fit_room(
            "/OmniverseKit_Persp",
            eye_height=eye[2],
            span_x=span_x,
            span_y=span_y,
            fill=0.96,
        )

        # Recreate the Replicator product after optics change so FOV is picked up.
        global _VIEWPORT_RGB_STATE
        if _VIEWPORT_RGB_STATE.get("annotator") is not None:
            try:
                _VIEWPORT_RGB_STATE["annotator"].detach()
            except Exception:
                pass
            try:
                if _VIEWPORT_RGB_STATE.get("render_product") is not None:
                    _VIEWPORT_RGB_STATE["render_product"].destroy()
            except Exception:
                pass
            _VIEWPORT_RGB_STATE["annotator"] = None
            _VIEWPORT_RGB_STATE["render_product"] = None
            _VIEWPORT_RGB_STATE["camera_path"] = None
            _VIEWPORT_RGB_STATE["resolution"] = None

        rgb = _capture_viewport_rgb(raw_env, warmup_renders=24)
        if rgb is None:
            raise RuntimeError(
                "Could not capture RGB frame from Isaac Lab viewport. "
                "Ensure HEADLESS runs use --enable_cameras (forced by this script)."
            )
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        if not (debug_draw_used or stage_overlay_used):
            raise RuntimeError(
                "Isaac viewport capture succeeded but no trajectory overlays were drawn "
                "(debug draw and USD stage overlays both unavailable)."
            )

        # Side view capture from Isaac scene (full-env framing).
        side_span = _compute_full_env_span(raw_env, obstacle_xy)
        side_center_xyz = _compute_env_center_xyz_from_walls(scene)
        if side_center_xyz is None:
            side_center_xyz = np.array([env_center_xy[0], env_center_xy[1], 0.0], dtype=np.float32)
        raw_scene_side_path = ""
        hidden_wall_name = SIDE_VIEW_HIDE_WALL
        side_eye, side_target = _side_eye_target(side_center_xyz, side_span)
        hide_left = SIDE_VIEW_HIDE_WALL == "left"
        hide_right = SIDE_VIEW_HIDE_WALL == "right"
        _set_wall_visibility(
            hide_top=bool(args_cli.hide_top_wall),
            hide_left=hide_left,
            hide_right=hide_right,
            hide_front=False,
            hide_back=False,
        )
        raw_env.unwrapped.sim.set_camera_view(eye=side_eye, target=side_target)
        _set_viewport_camera_view(side_eye, side_target)
        rgb_side = _capture_viewport_rgb(raw_env, warmup_renders=24)
        if rgb_side is not None:
            if rgb_side.shape[-1] == 4:
                rgb_side = rgb_side[:, :, :3]
            raw_scene_side_path = os.path.join(run_folder, "scene_side_raw.png")
            _save_rgb_image(rgb_side, raw_scene_side_path)

        _save_rgb_with_legend(
            rgb,
            plot_path,
            schematic_triad_segments=schematic_triad_segments,
            schematic_camera_shafts=schematic_camera_shafts,
            schematic_camera_frustums=schematic_camera_frustum_segments,
        )

        total_eps = max(episodes_done, 1)
        crash_rate = crash_count / total_eps
        success_rate = success_count / total_eps
        mean_time_to_goal = (success_time_sum_s / success_count) if success_count > 0 else float("nan")
        mean_path_to_goal = (success_path_sum_m / success_count) if success_count > 0 else float("nan")
        mean_angular_velocity = mean_angvel_sum / total_eps
        mean_coverage = (mean_coverage_sum / total_eps) if coverage_available else float("nan")

        stats_lines = [
            "=== Evaluation Results ===",
            "(Metrics aligned with scripts/evaluate_skrl.py; success/crash are not mutually exclusive.)",
            f"Task: {args_cli.task}",
            f"Agent entry: {args_cli.agent}",
            f"Model: {args_cli.model}",
            f"Checkpoint: {os.path.abspath(args_cli.checkpoint)}",
            f"Seed(base): {args_cli.seed}",
            f"Obstacle seed: {obstacle_seed}",
            f"Episodes evaluated: {episodes_done}",
            f"Success radius [m]: {float(args_cli.success_radius):.3f}",
            f"Crash rate: {crash_rate:.4f} ({crash_count}/{episodes_done})",
            f"Success rate: {success_rate:.4f} ({success_count}/{episodes_done})",
            f"Mean time to goal [s]: {mean_time_to_goal:.4f}",
            f"Mean path length to goal [m]: {mean_path_to_goal:.4f}",
            f"Mean angular velocity [rad/s]: {mean_angular_velocity:.4f}",
            f"Mean canonicalized coverage: {mean_coverage:.4f}",
            f"Crash markers plotted: {len(crash_points_xyz)}",
            f"Timeout markers plotted (non-crash & non-success): {len(timeout_points_xyz)}",
            "",
            "=== Visualization Settings ===",
            f"Camera-axis stride: {args_cli.frustum_stride}",
            f"Camera-axis distance [m]: {float(args_cli.frustum_distance):.3f}",
            f"Saved image scale factor: {float(args_cli.image_scale):.2f}",
            f"Topdown radius (full-env): {env_radius:.3f}",
            f"Side-view span (full-env): {side_span:.3f}",
            "Camera-axis colors keyed by active camera count (1..4).",
            f"Active cameras setting: {args_cli.active_cameras}",
            f"Top wall hidden: {bool(args_cli.hide_top_wall)}",
            "Built-in body/goal visuals hidden: True",
            "Robot visuals hidden: True",
            f"Side-view camera side: {SIDE_VIEW_CAMERA_SIDE}",
            f"Side wall hidden in side view: {hidden_wall_name}",
            f"Debug draw used: {debug_draw_used}",
            f"USD stage overlays used: {stage_overlay_used}",
            f"Obstacles frozen after first reset: {obstacles_frozen}",
            "Obstacle restore-after-reset enabled: True",
            f"Obstacle reference poses captured: {len(obstacle_ref_poses)}",
            f"Max obstacle deviation from reference [m]: {obstacle_max_dev_m:.6f}",
            f"Plot path: {os.path.abspath(plot_path)}",
            f"Raw scene side screenshot path: {os.path.abspath(raw_scene_side_path) if raw_scene_side_path else 'N/A'}",
            f"Run folder: {os.path.abspath(run_folder)}",
        ]
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stats_lines) + "\n")

        plot_name = os.path.basename(plot_path)
        published = _publish_latest_outputs(run_folder, analysis_root, plot_name)
        latest_plot = os.path.join(analysis_root, plot_name)

        print(f"[INFO] Saved IsaacLab top-down screenshot: {os.path.abspath(plot_path)}")
        print(f"[INFO] Saved evaluation stats: {os.path.abspath(stats_path)}")
        print(f"[INFO] Analysis folder: {os.path.abspath(run_folder)}")
        if published:
            print(f"[INFO] Latest copies (stable paths): {os.path.abspath(latest_plot)}")
            print(f"[INFO] Latest run symlink: {os.path.abspath(os.path.join(analysis_root, 'latest'))}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
