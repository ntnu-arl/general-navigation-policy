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
"""skrl eval: crash/success rates, time/path-to-goal, coverage; optional --sensor_dropout.

Usage: isaaclab -p scripts/evaluate_skrl.py --task ... --checkpoint ...
"""


from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments (copied/adapted from play script)
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--model",
    type=str,
    default="spherical_range_cnn_gru",
    choices=["spherical_range_cnn_gru"],
    help="Model architecture to use (must match the one used for training).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help="Number of completed episodes to evaluate.",
)
parser.add_argument(
    "--success_radius",
    type=float,
    default=1.0,
    help="Goal radius in meters used to determine success.",
)
parser.add_argument(
    "--active_cameras",
    type=str,
    default="random",
    help="Number of active cameras: 'random' or fixed count in {1,2,3,4}.",
)
parser.add_argument(
    "--save_policy_obs_debug",
    action="store_true",
    default=False,
    help="Save canonicalized policy observation images with per-camera FOV outlines.",
)
parser.add_argument(
    "--policy_obs_debug_dir",
    type=str,
    default=None,
    help="Directory for policy observation debug images (default: <log_dir>/policy_obs_debug).",
)
parser.add_argument(
    "--policy_obs_debug_env_id",
    type=int,
    default=0,
    help="Environment index to visualize in policy observation debug images.",
)
parser.add_argument(
    "--sensor_dropout",
    action="store_true",
    default=False,
    help=(
        "Enable mid-episode sensor dropout: once per episode, stop feeding one "
        "active camera into the canonicalization pipeline until episode end. "
        "Requires --active_cameras in {2,3,4}."
    ),
)
parser.add_argument(
    "--sensor_dropout_fraction",
    type=float,
    default=0.5,
    help="Fraction of max episode length at which dropout is applied (0, 1). Default: 0.5.",
)
parser.add_argument(
    "--sensor_dropout_camera",
    type=str,
    default="random",
    help=(
        "Which active camera to drop: 'random' (default, re-sampled per env/episode) "
        "or a fixed camera index among the initially active prefix (e.g. 0)."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Drop blank argv leftovers — Hydra Lexers choke on " ".
hydra_args = [a for a in hydra_args if a.strip()]
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (largely copied from IsaacLab's skrl/play.py and local play_skrl.py)."""

import gymnasium as gym
import skrl
import torch
import yaml
from packaging import version
from skrl.utils.spaces.torch import unflatten_tensorized_space

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401

# Import custom navigation tasks so their Gym environments are registered
import general_navigation_policy.tasks.arbitrary_sensor_setup  # noqa: F401

from general_navigation_policy.tasks.metrics import is_episode_success
from general_navigation_policy.tasks.mdp.events import configure_active_cameras
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# check for minimum supported skrl version
SKRL_VERSION = "2.1.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    sys.exit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    from skrl.agents.torch.ppo import PPO, PPO_RNN
    from skrl.memories.torch import RandomMemory
    from general_navigation_policy.learning.skrl.load_agent_cfg import build_ppo_cfg
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

# Import custom models
from general_navigation_policy.learning.skrl.models.spherical_range_cnn_gru import SphericalRangeCNNGRU
from general_navigation_policy.tasks.mdp import observations as mdp_obs


# Map --model names to classes
MODEL_CLASSES = {
    "spherical_range_cnn_gru": SphericalRangeCNNGRU,
}

# Tasks that require custom model instantiation (bypass Runner)
CUSTOM_MODEL_TASKS = {
    "Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-v0",
    "Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-Play-v0",
}

AGENT_ENTRY_ALIASES = {
    "spherical_range_cnn_gru": "skrl_spherical_range_cnn_gru_cfg_entry_point",
}
DEFAULT_AGENT_ENTRY = "skrl_spherical_range_cnn_gru_cfg_entry_point"


def _extract_robot_and_goal_positions(raw_env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return robot and goal positions in environment frame."""
    robot = raw_env.scene["robot"]
    target_poses = raw_env.command_manager.get_command("target_pose")
    target_positions = target_poses[:, :3]
    robot_positions = robot.data.root_pos_w - raw_env.scene.env_origins
    return robot_positions, target_positions


def _extract_policy_image_tensor(obs_obj, observation_space) -> torch.Tensor | None:
    """Return policy image tensor shaped (N, A, E, 2), or None if unavailable."""

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

    if isinstance(obs_obj, dict):
        return _find_image_tensor(obs_obj)
    if torch.is_tensor(obs_obj):
        if obs_obj.ndim == 4 and obs_obj.shape[-1] == 2:
            return obs_obj
        if obs_obj.ndim == 2:
            states = unflatten_tensorized_space(observation_space, obs_obj)
            if isinstance(states, dict):
                policy_obs = states.get("policy", states)
                return _find_image_tensor(policy_obs)
    return None


def _canonicalized_coverage_per_env(
    raw_env,
    obs_obj,
    observation_space,
) -> torch.Tensor | None:
    """Fraction of valid canonicalized bins per env: mean(mask) over azimuth×elevation."""
    mask = getattr(raw_env, "_cached_validity_mask", None)
    if torch.is_tensor(mask) and mask.ndim == 3 and mask.shape[0] == raw_env.num_envs:
        return mask.to(dtype=torch.float32).mean(dim=(1, 2))

    image = _extract_policy_image_tensor(obs_obj, observation_space)
    if image is None:
        return None
    return image[..., 1].to(dtype=torch.float32).mean(dim=(1, 2))


def _binary_outline(mask_2d: torch.Tensor) -> torch.Tensor:
    """Return boundary pixels (1.0) for a 2D binary mask tensor."""
    m = mask_2d.to(dtype=torch.bool)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={tuple(m.shape)}")
    h, w = int(m.shape[0]), int(m.shape[1])
    padded = torch.zeros((h + 2, w + 2), device=m.device, dtype=torch.bool)
    padded[1:-1, 1:-1] = m
    eroded = torch.ones((h, w), device=m.device, dtype=torch.bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return (m & ~eroded).to(dtype=torch.float32)


def _save_policy_observation_debug_image(
    raw_env_unwrapped,
    obs_from_policy,
    observation_space,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    output_dir: str | os.PathLike[str],
    env_id: int = 0,
) -> Path:
    """Save the policy canonicalized image with per-camera FOV outlines."""
    import matplotlib.pyplot as plt
    import numpy as np

    image_term = getattr(getattr(env_cfg, "observations", None), "policy", None)
    image_term = getattr(image_term, "image", None)
    if image_term is None:
        raise RuntimeError("Could not find env_cfg.observations.policy.image for debug visualization.")
    image_params = dict(getattr(image_term, "params", {}))

    max_num_cameras = int(image_params.get("max_num_cameras", getattr(env_cfg, "max_num_cameras", 1)))
    azimuth_resolution = int(image_params.get("azimuth_resolution", 90))
    elevation_resolution = int(image_params.get("elevation_resolution", 45))
    max_range = float(image_params.get("max_range", 10.0))
    robot_cfg = image_params.get("robot_cfg", SceneEntityCfg("robot"))

    def _extract_policy_image(obs_obj):
        found = _extract_policy_image_tensor(obs_obj, observation_space)
        if found is not None:
            return found
        if isinstance(obs_obj, dict):
            raise RuntimeError(f"Could not find policy image tensor in obs dict keys={list(obs_obj.keys())}")
        if torch.is_tensor(obs_obj):
            raise RuntimeError(f"Obs tensor does not look like image NHWC, shape={tuple(obs_obj.shape)}")
        raise RuntimeError(f"Unsupported obs object type for debug extraction: {type(obs_obj)}")

    fused = _extract_policy_image(obs_from_policy)

    if not (0 <= env_id < fused.shape[0]):
        raise ValueError(f"policy_obs_debug_env_id={env_id} out of range for num_envs={fused.shape[0]}")

    fused_depth = fused[env_id, :, :, 0]
    fused_valid = fused[env_id, :, :, 1]

    active_mask = None
    if hasattr(raw_env_unwrapped, "_active_camera_mask"):
        if raw_env_unwrapped._active_camera_mask.shape == (raw_env_unwrapped.num_envs, max_num_cameras):
            active_mask = raw_env_unwrapped._active_camera_mask[env_id]
    if active_mask is None:
        active_mask = torch.ones((max_num_cameras,), device=raw_env_unwrapped.device, dtype=torch.bool)

    outlines = []
    for cam_idx in range(max_num_cameras):
        sensor_name = f"depth_camera_{cam_idx}"
        camera_pos_b, camera_quat_b = mdp_obs._camera_extrinsics_in_robot_frame(raw_env_unwrapped, sensor_name)
        cam_out = mdp_obs.canonicalized_360_depth(
            env=raw_env_unwrapped,
            camera_pos_b=camera_pos_b,
            camera_quat_b=camera_quat_b,
            sensor_cfg=SceneEntityCfg(sensor_name),
            robot_cfg=robot_cfg,
            azimuth_resolution=azimuth_resolution,
            elevation_resolution=elevation_resolution,
            max_range=max_range,
        )
        cam_valid = cam_out[env_id, :, :, 1]
        outlines.append(_binary_outline(cam_valid))

    depth_np = fused_depth.detach().cpu().numpy().T
    valid_np = fused_valid.detach().cpu().numpy().T

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"policy_obs_env{env_id}_with_outlines.png"

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    cmap = plt.get_cmap("tab10" if max_num_cameras <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(max_num_cameras)]

    # Left: exact fused depth channel that goes to the policy image term.
    ax0 = axes[0]
    im = ax0.imshow(depth_np, origin="lower", aspect="auto", cmap="turbo", vmin=0.0, vmax=1.0)
    ax0.set_title("Policy image depth channel (fused)")
    ax0.set_xlabel("Azimuth")
    ax0.set_ylabel("Elevation")
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label="normalized depth")

    # Right: fused validity with per-camera footprint outlines.
    ax1 = axes[1]
    ax1.imshow(valid_np, origin="lower", aspect="auto", cmap="gray", vmin=0.0, vmax=1.0)
    ax1.set_title("Policy validity + camera FOV outlines")
    ax1.set_xlabel("Azimuth")
    ax1.set_ylabel("Elevation")

    from matplotlib.lines import Line2D

    legend_lines = []
    for cam_idx, outline in enumerate(outlines):
        outline_np = outline.detach().cpu().numpy().T
        ys, xs = np.where(outline_np > 0.5)
        is_active = bool(active_mask[cam_idx].item())
        alpha = 1.0 if is_active else 0.25
        size = 1.0 if is_active else 0.4
        ax1.scatter(xs, ys, c=[colors[cam_idx]], s=size, marker="s", linewidths=0, alpha=alpha)
        legend_lines.append(
            Line2D(
                [0],
                [0],
                color=colors[cam_idx],
                lw=2.0,
                alpha=alpha,
                label=f"cam {cam_idx} ({'active' if is_active else 'inactive'})",
            )
        )
    ax1.legend(handles=legend_lines, loc="upper right", fontsize=8)

    active_count = int(active_mask.to(dtype=torch.long).sum().item())
    fig.suptitle(f"Canonicalized policy observation (env={env_id}, active_cameras={active_count}/{max_num_cameras})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _invalidate_multi_camera_obs_cache(raw_env) -> None:
    """Force the next multi-camera canonicalization call to recompute."""
    for attr in ("_cached_obs_360_multi", "_cached_obs_360_multi_step", "_cached_obs_360_multi_signature"):
        if hasattr(raw_env, attr):
            delattr(raw_env, attr)


def _overlay_checkpoint_agent_cfg(experiment_cfg: dict, checkpoint_path: str) -> dict:
    """Prefer the training-run ``params/agent.yaml`` so model kwargs match the checkpoint.

    Current Hydra agent defaults (e.g. ``use_mask_channel: false`` for no-mask ablations)
    often differ from older checkpoints trained with a depth+mask encoder.
    """
    params_path = Path(checkpoint_path).expanduser().resolve().parents[1] / "params" / "agent.yaml"
    if not params_path.is_file():
        print(f"[WARN] No saved agent.yaml next to checkpoint ({params_path}); using Hydra agent cfg.")
        return experiment_cfg
    with params_path.open("r", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    if not isinstance(saved, dict):
        return experiment_cfg
    print(f"[INFO] Overlaying agent config from {params_path}")
    if "models" in saved:
        experiment_cfg["models"] = saved["models"]
        # Older runs omit use_mask_channel; keep the model default (True) when
        # the checkpoint was trained with a 2-channel encoder/decoder.
        policy_kwargs = (
            experiment_cfg.get("models", {}).get("policy", {}).get("kwargs") or {}
        )
        if (
            isinstance(policy_kwargs, dict)
            and "use_mask_channel" not in policy_kwargs
            and int(policy_kwargs.get("image_channels", 2)) >= 2
        ):
            policy_kwargs["use_mask_channel"] = True
            experiment_cfg["models"]["policy"]["kwargs"] = policy_kwargs
    if "agent" in saved:
        keep_exp = experiment_cfg.get("agent", {}).get("experiment", {})
        experiment_cfg["agent"] = saved["agent"]
        experiment_cfg["agent"]["experiment"] = keep_exp
    if "memory" in saved:
        experiment_cfg["memory"] = saved["memory"]
    return experiment_cfg


def _sensor_dropout_mid_step(raw_env) -> int:
    """Env step index at which mid-episode sensor dropout is applied."""
    max_len = int(getattr(raw_env, "max_episode_length", 0))
    if max_len < 2:
        step_dt = float(getattr(raw_env, "step_dt", 0.0))
        episode_s = float(getattr(getattr(raw_env, "cfg", None), "episode_length_s", 14.0))
        if step_dt > 0.0:
            max_len = max(2, int(round(episode_s / step_dt)))
        else:
            max_len = 2
    return max(1, int(max_len * float(args_cli.sensor_dropout_fraction)))


def _parse_sensor_dropout_camera(active_n: int) -> int | None:
    """Return fixed camera index, or None for per-episode random selection."""
    spec = str(args_cli.sensor_dropout_camera).strip().lower()
    if spec == "random":
        return None
    try:
        cam_idx = int(spec)
    except ValueError as e:
        raise ValueError(
            "--sensor_dropout_camera must be 'random' or an integer camera index."
        ) from e
    if not (0 <= cam_idx < active_n):
        raise ValueError(
            f"--sensor_dropout_camera={cam_idx} must be in [0, {active_n - 1}] "
            f"(one of the initially active cameras for --active_cameras={active_n})."
        )
    return cam_idx


def _maybe_apply_sensor_dropout(raw_env, *, fixed_drop_camera: int | None) -> int:
    """Deactivate one camera mid-episode via ``_active_camera_mask``.

    Returns the number of envs that just experienced dropout.
    """
    mid_step = _sensor_dropout_mid_step(raw_env)
    if not hasattr(raw_env, "_sensor_dropout_applied"):
        raw_env._sensor_dropout_applied = torch.zeros(
            raw_env.num_envs, device=raw_env.device, dtype=torch.bool
        )

    episode_len = getattr(raw_env, "episode_length_buf", None)
    if episode_len is None:
        return 0

    at_mid = (episode_len == mid_step) & (~raw_env._sensor_dropout_applied)
    if not at_mid.any():
        return 0

    env_ids = at_mid.nonzero(as_tuple=False).flatten()
    max_num_cameras = int(getattr(raw_env.cfg, "max_num_cameras", getattr(raw_env, "max_num_cameras", 4)))
    device = raw_env.device

    if (
        not hasattr(raw_env, "_active_camera_mask")
        or raw_env._active_camera_mask.shape != (raw_env.num_envs, max_num_cameras)
    ):
        return 0

    active_mask = raw_env._active_camera_mask
    n_env = int(env_ids.numel())
    if fixed_drop_camera is not None:
        drop_idx = torch.full((n_env,), int(fixed_drop_camera), device=device, dtype=torch.long)
        still_active = active_mask[env_ids, drop_idx]
        if not bool(still_active.all().item()):
            # Skip envs where the chosen camera is already inactive.
            env_ids = env_ids[still_active]
            drop_idx = drop_idx[still_active]
            if env_ids.numel() == 0:
                return 0
    else:
        # Sample uniformly among currently active cameras for each env.
        active_rows = active_mask[env_ids]  # (R, C)
        n_active = active_rows.sum(dim=1)
        eligible = n_active >= 2
        if not bool(eligible.any().item()):
            return 0
        env_ids = env_ids[eligible]
        active_rows = active_rows[eligible]
        n_active = n_active[eligible]
        # Inverse-CDF sample over the True entries of each row.
        # active_rows is a prefix mask in normal eval, but keep this general.
        rand = torch.rand(env_ids.numel(), device=device) * n_active.to(dtype=torch.float32)
        # rank k among active cams -> camera index of the k-th True
        cumsum = active_rows.to(dtype=torch.long).cumsum(dim=1)
        # For each env, find the first column where cumsum > floor(rand)
        target = rand.floor().to(dtype=torch.long).unsqueeze(1)  # 0-based rank
        drop_idx = (cumsum > target).to(dtype=torch.long).argmax(dim=1)

    active_mask[env_ids, drop_idx] = False
    if hasattr(raw_env, "_num_active_cameras") and raw_env._num_active_cameras.shape == (raw_env.num_envs,):
        raw_env._num_active_cameras[env_ids] = active_mask[env_ids].sum(dim=1)

    raw_env._sensor_dropout_applied[env_ids] = True
    _invalidate_multi_camera_obs_cache(raw_env)
    return int(env_ids.numel())


# config shortcuts (IsaacLab-style)
task_id = args_cli.task or ""
if args_cli.agent is None:
    agent_cfg_entry_point = DEFAULT_AGENT_ENTRY
elif args_cli.agent in AGENT_ENTRY_ALIASES:
    agent_cfg_entry_point = AGENT_ENTRY_ALIASES[args_cli.agent]
else:
    agent_cfg_entry_point = args_cli.agent
algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Evaluate a skrl policy for a fixed number of episodes."""
    if args_cli.num_episodes <= 0:
        raise ValueError(f"--num_episodes must be > 0, got {args_cli.num_episodes}")
    if args_cli.success_radius <= 0:
        raise ValueError(f"--success_radius must be > 0, got {args_cli.success_radius}")
    active_spec = str(args_cli.active_cameras).strip().lower()
    if active_spec != "random":
        try:
            active_n = int(active_spec)
        except ValueError as e:
            raise ValueError("--active_cameras must be one of: random, 1, 2, 3, 4.") from e
        if active_n not in {1, 2, 3, 4}:
            raise ValueError(f"--active_cameras must be one of: random, 1, 2, 3, 4. Got {args_cli.active_cameras}")
    else:
        active_n = None

    fixed_drop_camera = None
    if args_cli.sensor_dropout:
        if not (0.0 < float(args_cli.sensor_dropout_fraction) < 1.0):
            raise ValueError(
                f"--sensor_dropout_fraction must be in (0, 1), got {args_cli.sensor_dropout_fraction}"
            )
        if active_n is None or active_n < 2:
            raise ValueError(
                "--sensor_dropout requires --active_cameras in {2,3,4} "
                "(need at least two cameras so one can drop)."
            )
        fixed_drop_camera = _parse_sensor_dropout_camera(active_n)

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # Configure active camera sampling behavior before environment creation.
    configure_active_cameras(env_cfg, args_cli.active_cameras)

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    experiment_cfg = _overlay_checkpoint_agent_cfg(experiment_cfg, resume_path)

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(raw_env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        raw_env = multi_agent_to_single_agent(raw_env)

    env = SkrlVecEnvWrapper(raw_env, ml_framework=args_cli.ml_framework)

    use_custom_model = task_id in CUSTOM_MODEL_TASKS

    if use_custom_model and args_cli.ml_framework.startswith("torch"):
        model_class = MODEL_CLASSES[args_cli.model]
        print(f"[INFO] Using custom model: {model_class.__name__} (bypassing Runner)")

        device = env.device
        observation_space = env.observation_space
        action_space = env.action_space

        if "models" not in experiment_cfg:
            raise ValueError("Missing 'models' section in experiment_cfg. Check your skrl config YAML.")
        if "policy" not in experiment_cfg["models"]:
            raise ValueError("Missing 'models.policy' section in experiment_cfg. Check your skrl config YAML.")

        policy_kwargs = experiment_cfg["models"]["policy"].get("kwargs", {})
        if not policy_kwargs:
            raise ValueError("Missing 'models.policy.kwargs' in experiment_cfg.")

        policy_kwargs = dict(policy_kwargs)
        policy_kwargs.setdefault("num_envs", env.num_envs)

        model = model_class(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **policy_kwargs,
        )
        models = {"policy": model, "value": model}

        memory_cfg = experiment_cfg.get("memory", {})
        memory_size = memory_cfg.get("memory_size", -1)
        if memory_size < 0:
            memory_size = experiment_cfg["agent"]["rollouts"]
        memory = RandomMemory(num_envs=env.num_envs, device=device, memory_size=memory_size)

        ppo_cfg, _ = build_ppo_cfg(
            experiment_cfg["agent"],
            observation_space=observation_space,
            device=device,
        )
        experiment = ppo_cfg.setdefault("experiment", {})
        if isinstance(experiment, dict):
            experiment["write_interval"] = 0
            experiment["checkpoint_interval"] = 0

        rnn_models = {"spherical_range_cnn_gru"}
        agent_class = PPO_RNN if args_cli.model in rnn_models else PPO
        agent = agent_class(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        agent.init()
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
        agent.enable_training_mode(False)
    else:
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False)
        agent = runner.agent

    obs, _ = env.reset()

    if args_cli.sensor_dropout:
        mid_step = _sensor_dropout_mid_step(raw_env.unwrapped)
        drop_desc = (
            f"camera {fixed_drop_camera}"
            if fixed_drop_camera is not None
            else "a random active camera"
        )
        print(
            f"[INFO] Sensor dropout enabled: drop {drop_desc} at step {mid_step} "
            f"({float(args_cli.sensor_dropout_fraction):.0%} of max episode length); "
            f"active_cameras={active_n}"
        )
        raw_env.unwrapped._sensor_dropout_applied = torch.zeros(
            raw_env.unwrapped.num_envs, device=raw_env.unwrapped.device, dtype=torch.bool
        )

    if args_cli.save_policy_obs_debug:
        debug_dir = args_cli.policy_obs_debug_dir
        if debug_dir is None:
            debug_dir = os.path.join(log_dir, "policy_obs_debug")
        try:
            out_path = _save_policy_observation_debug_image(
                raw_env_unwrapped=raw_env.unwrapped,
                obs_from_policy=obs,
                observation_space=env.observation_space,
                env_cfg=env_cfg,
                output_dir=debug_dir,
                env_id=int(args_cli.policy_obs_debug_env_id),
            )
            print(f"[INFO] Saved policy observation debug image to: {out_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save policy observation debug image: {e}")

    num_envs = env.num_envs
    device = env.device
    dt = float(getattr(raw_env.unwrapped, "step_dt", getattr(raw_env, "step_dt", 0.0)))
    success_radius = float(args_cli.success_radius)

    # Per-active-episode accumulators
    ep_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
    ep_path = torch.zeros(num_envs, device=device, dtype=torch.float32)
    ep_angvel_sum = torch.zeros(num_envs, device=device, dtype=torch.float32)
    ep_coverage_sum = torch.zeros(num_envs, device=device, dtype=torch.float32)
    ep_reached_goal = torch.zeros(num_envs, device=device, dtype=torch.bool)
    # After reset, dist_to_goal can return stale values for a few steps.
    # We arm the reach check only after we observe dist_to_goal > success_radius
    # at least once for each env, preventing spurious "success" at episode start.
    ep_reach_armed = torch.zeros(num_envs, device=device, dtype=torch.bool)
    ep_first_reach_step = torch.zeros(num_envs, device=device, dtype=torch.long)
    ep_first_reach_path = torch.zeros(num_envs, device=device, dtype=torch.float32)

    # Aggregate counters
    episodes_done = 0
    crash_count = 0
    success_count = 0
    success_time_sum_s = 0.0
    success_path_sum_m = 0.0
    mean_angvel_sum = 0.0
    mean_coverage_sum = 0.0
    coverage_available = False
    sensor_dropout_events = 0

    # Fair episode accounting across parallel envs:
    # each env contributes a fixed quota to avoid "first K finished episodes" bias.
    base_quota = args_cli.num_episodes // num_envs
    remainder = args_cli.num_episodes % num_envs
    episode_quota_per_env = torch.full((num_envs,), base_quota, device=device, dtype=torch.long)
    if remainder > 0:
        episode_quota_per_env[:remainder] += 1
    completed_per_env = torch.zeros(num_envs, device=device, dtype=torch.long)

    while simulation_app.is_running() and episodes_done < args_cli.num_episodes:
        with torch.inference_mode():
            robot_positions, target_positions = _extract_robot_and_goal_positions(raw_env.unwrapped)
            dist_to_goal = torch.norm(robot_positions - target_positions, dim=1)

            robot = raw_env.unwrapped.scene["robot"]
            lin_speed = torch.norm(robot.data.root_lin_vel_w, dim=1)
            ang_speed = torch.norm(robot.data.root_ang_vel_w, dim=1)

            ep_steps += 1
            if dt > 0.0:
                ep_path += lin_speed * dt
            ep_angvel_sum += ang_speed
            coverage = _canonicalized_coverage_per_env(raw_env.unwrapped, obs, env.observation_space)
            if coverage is not None:
                coverage_available = True
                ep_coverage_sum += coverage

            ep_reach_armed |= (dist_to_goal > success_radius)
            newly_reached = (dist_to_goal <= success_radius) & ~ep_reached_goal & ep_reach_armed
            ep_reached_goal[newly_reached] = True
            ep_first_reach_step[newly_reached] = ep_steps[newly_reached]
            ep_first_reach_path[newly_reached] = ep_path[newly_reached]

            outputs = agent.act(
                obs, env.state() if hasattr(env, "state") else None, timestep=0, timesteps=0
            )
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            obs, _, terminated, truncated, _ = env.step(actions)

            if args_cli.sensor_dropout:
                sensor_dropout_events += _maybe_apply_sensor_dropout(
                    raw_env.unwrapped, fixed_drop_camera=fixed_drop_camera
                )

            # skrl IsaacLabWrapper returns terminated/truncated as (N, 1). Using
            # nonzero(...).flatten() would yield [env_i, 0, ...] and double-count
            # every finished env (spurious env 0). Squeeze to env indices only.
            done = (terminated | truncated).squeeze(-1)
            finished_ids = done.nonzero(as_tuple=False).flatten()

            if hasattr(agent, "_rnn") and agent._rnn:
                if finished_ids.numel():
                    for rnn_state in agent._rnn_final_states["policy"]:
                        rnn_state[:, finished_ids] = 0
                agent._rnn_initial_states = agent._rnn_final_states

            if finished_ids.numel() == 0:
                continue

            # Reset-event restores ``_active_camera_mask``; clear dropout latch for new episodes.
            if args_cli.sensor_dropout and hasattr(raw_env.unwrapped, "_sensor_dropout_applied"):
                raw_env.unwrapped._sensor_dropout_applied[finished_ids] = False

            for env_id_t in finished_ids.tolist():
                if episodes_done >= args_cli.num_episodes:
                    break
                env_id = int(env_id_t)

                # Skip logging for envs that already met their quota.
                if completed_per_env[env_id] >= episode_quota_per_env[env_id]:
                    ep_steps[env_id] = 0
                    ep_path[env_id] = 0.0
                    ep_angvel_sum[env_id] = 0.0
                    ep_coverage_sum[env_id] = 0.0
                    ep_reached_goal[env_id] = False
                    ep_reach_armed[env_id] = False
                    ep_first_reach_step[env_id] = 0
                    ep_first_reach_path[env_id] = 0.0
                    continue

                was_crash = bool(terminated[env_id].item() and not truncated[env_id].item())
                # Pre-step distance: step() auto-resets finished envs; post-step is next spawn.
                final_error_m = float(dist_to_goal[env_id].item())
                was_success = is_episode_success(
                    bool(ep_reached_goal[env_id].item()),
                    final_error_m,
                    success_radius,
                )

                crash_count += int(was_crash)
                success_count += int(was_success)

                steps_this_ep = max(int(ep_steps[env_id].item()), 1)
                mean_angvel_sum += float(ep_angvel_sum[env_id].item()) / float(steps_this_ep)
                mean_coverage_sum += float(ep_coverage_sum[env_id].item()) / float(steps_this_ep)

                if was_success:
                    success_steps = int(ep_first_reach_step[env_id].item())
                    success_time_sum_s += float(success_steps) * dt
                    success_path_sum_m += float(ep_first_reach_path[env_id].item())

                episodes_done += 1
                completed_per_env[env_id] += 1

                ep_steps[env_id] = 0
                ep_path[env_id] = 0.0
                ep_angvel_sum[env_id] = 0.0
                ep_coverage_sum[env_id] = 0.0
                ep_reached_goal[env_id] = False
                ep_reach_armed[env_id] = False
                ep_first_reach_step[env_id] = 0
                ep_first_reach_path[env_id] = 0.0

    total_eps = max(episodes_done, 1)
    crash_rate = crash_count / total_eps
    success_rate = success_count / total_eps
    mean_time_to_goal = (success_time_sum_s / success_count) if success_count > 0 else float("nan")
    mean_path_to_goal = (success_path_sum_m / success_count) if success_count > 0 else float("nan")
    mean_angular_velocity = mean_angvel_sum / total_eps
    mean_coverage = (mean_coverage_sum / total_eps) if coverage_available else float("nan")

    print("\n=== Evaluation Results ===")
    print(f"Episodes evaluated: {episodes_done}")
    print(f"Success radius [m]: {success_radius:.3f}")
    print(f"Crash rate: {crash_rate:.4f} ({crash_count}/{episodes_done})")
    print(f"Success rate: {success_rate:.4f} ({success_count}/{episodes_done})")
    print(f"Mean time to goal [s]: {mean_time_to_goal:.4f}")
    print(f"Mean path length to goal [m]: {mean_path_to_goal:.4f}")
    print(f"Mean angular velocity [rad/s]: {mean_angular_velocity:.4f}")
    print(f"Mean canonicalized coverage: {mean_coverage:.4f}")
    if args_cli.sensor_dropout:
        print(f"Sensor dropout events: {sensor_dropout_events}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
