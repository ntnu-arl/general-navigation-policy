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
import argparse
import os
import sys
import time
from datetime import datetime
import shutil
import subprocess
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Unified skrl training script (PPO / PPO_RNN).")
parser.add_argument("--task", type=str, default="Isaac-Arbitrary-Sensor-Setup-ARL-Robot-1-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument(
    "--agent",
    type=str,
    default="spherical_range_cnn_gru",
    choices=["spherical_range_cnn_gru"],
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Override trainer timesteps.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Drop blank argv leftovers (broken `\` continuations) — Hydra Lexers choke on " ".
hydra_args = [a for a in hydra_args if a.strip()]
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports (Isaac Sim must be initialised first)
# ---------------------------------------------------------------------------
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from skrl.utils import set_seed
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory

from general_navigation_policy.learning.skrl.load_agent_cfg import build_ppo_cfg, filter_trainer_cfg
from general_navigation_policy.learning.skrl.trainers.custom_sequential_trainer import (
    CustomSequentialTrainer,
)
from general_navigation_policy.learning.skrl.models.spherical_range_cnn_gru import SphericalRangeCNNGRU

from general_navigation_policy.learning.skrl.agents.custom_ppo_rnn import PPO_RNN_AuxRecon

# Register custom Gym environments
import general_navigation_policy.tasks.arbitrary_sensor_setup  # noqa: F401

set_seed(args_cli.seed)

# ---------------------------------------------------------------------------
# Look-up tables (agent name -> Hydra entry point / model class)
# ---------------------------------------------------------------------------
TASK = args_cli.task

AGENT_ENTRY = {
    "spherical_range_cnn_gru": "skrl_spherical_range_cnn_gru_cfg_entry_point",
}[args_cli.agent]

MODEL_CLASS = {
    "spherical_range_cnn_gru": SphericalRangeCNNGRU,
}

def _save_git_context(output_dir: str) -> None:
    """Save git commit hash and diff (relative to that hash) if available."""
    script_dir = Path(__file__).resolve().parent

    try:
        rev_parse = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_hash = rev_parse.stdout.strip()
    except Exception as err:
        print(f"[train_skrl] Skipping git context export (no hash): {err}")
        return

    try:
        diff_proc = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=script_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        diff_text = diff_proc.stdout
    except Exception as err:
        print(f"[train_skrl] Skipping git diff export: {err}")
        diff_text = ""

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "git_commit_hash.txt"), "w", encoding="utf-8") as f:
            f.write(commit_hash + "\n")
        with open(os.path.join(output_dir, "git_diff.patch"), "w", encoding="utf-8") as f:
            f.write(diff_text)
    except Exception as err:
        print(f"[train_skrl] Failed to write git context files: {err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra_task_config(TASK, AGENT_ENTRY)
def main(env_cfg, agent_cfg):
    # ---- CLI overrides ---------------------------------------------------
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed

    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # ---- logging directory -----------------------------------------------
    log_root = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch"
    exp_name = agent_cfg["agent"]["experiment"].get("experiment_name")
    if exp_name:
        log_dir_name += "_" + exp_name
    log_dir = os.path.join(log_root, log_dir_name)
    env_cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name

    # ---- save params (agent_cfg IS the config used — no hidden override) -
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    scene = getattr(env_cfg, "scene", None)
    depth_camera_cfg = getattr(scene, "depth_camera", None) if scene else None
    if depth_camera_cfg is not None:
        dump_yaml(os.path.join(params_dir, "camera.yaml"), depth_camera_cfg)
    _save_git_context(params_dir)

    # ---- create environment ----------------------------------------------
    env = gym.make(TASK, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env.device
    observation_space = env.observation_space
    action_space = env.action_space

    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if getattr(args_cli, "checkpoint", None)
        else None
    )
    model_class = MODEL_CLASS[args_cli.agent]

    # ---- read agent class and sharing mode from YAML ---------------------
    agent_class_name = agent_cfg.get("agent", {}).get("class", "PPO")
    use_ppo_rnn = agent_class_name == "PPO_RNN"
    separate_models = agent_cfg.get("models", {}).get("separate", False)

    # ---- model kwargs from YAML ------------------------------------------
    policy_kwargs = dict(
        agent_cfg.get("models", {}).get("policy", {}).get("kwargs", {})
    )
    policy_kwargs.setdefault("num_envs", env.num_envs)

    set_seed(agent_cfg["seed"])
    
    # ---- instantiate model(s) --------------------------------------------
    if separate_models:
        models = {
            "policy": model_class(observation_space, action_space, device, **policy_kwargs),
            "value":  model_class(observation_space, action_space, device, **policy_kwargs),
        }
    else:
        shared = model_class(observation_space, action_space, device, **policy_kwargs)
        models = {"policy": shared, "value": shared}

    # ---- rollout memory --------------------------------------------------
    rollouts = agent_cfg.get("agent", {}).get("rollouts", 24)
    mem_cfg = agent_cfg.get("memory", {})
    memory_size = mem_cfg.get("memory_size", -1)
    if memory_size < 0:
        memory_size = rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    # ---- PPO config (sole source of truth: agent_cfg) --------------------
    ppo_cfg, aux_recon_loss_scale = build_ppo_cfg(
        agent_cfg.get("agent", {}),
        observation_space=observation_space,
        device=device,
    )
    
    # ---- instantiate agent -----------------------------------------------
    AgentClass = PPO_RNN_AuxRecon if use_ppo_rnn else PPO
    agent_kwargs = dict(
        models=models,
        memory=memory,
        cfg=ppo_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )
    if use_ppo_rnn:
        agent_kwargs["aux_recon_loss_scale"] = aux_recon_loss_scale
    agent = AgentClass(**agent_kwargs)

    if resume_path:
        agent.load(resume_path)
   
    # ---- train -----------------------------------------------------------
    trainer_cfg = filter_trainer_cfg(agent_cfg.get("trainer", {}))
    if getattr(args_cli, "headless", False) or os.environ.get("HEADLESS", "0") == "1":
        trainer_cfg["headless"] = True
    trainer = CustomSequentialTrainer(
        cfg=trainer_cfg, env=env, agents=agent
    )
    start_time = time.time()
    trainer.train()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()