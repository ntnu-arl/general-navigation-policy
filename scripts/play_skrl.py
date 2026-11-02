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
"""skrl play for GNP Gym envs (based on IsaacLab skrl/play.py).

Usage: isaaclab -p scripts/play_skrl.py --task ... --model spherical_range_cnn_gru --checkpoint ...
"""

import argparse
import sys
import os
import random
import time

from isaaclab.app import AppLauncher

# add argparse arguments (copied from IsaacLab's skrl/play.py)
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument(
    "--active_cameras",
    type=str,
    default="random",
    help="Active cameras for Play: random, or 1/2/3/4.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

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

"""Rest everything follows (largely copied from IsaacLab's skrl/play.py)."""

import gymnasium as gym
import skrl
import torch
from packaging import version

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401

# Import custom navigation tasks so their Gym environments are registered
import general_navigation_policy.tasks.arbitrary_sensor_setup  # noqa: F401

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
from general_navigation_policy.tasks.mdp.events import configure_active_cameras


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
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    configure_active_cameras(env_cfg, args_cli.active_cameras)

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
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

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # Check if this task uses a custom model that can't be auto-instantiated by Runner
    # (task-based and cfg-based checks keep this robust to task name or config evolution).
    use_custom_model = (
        task_id in CUSTOM_MODEL_TASKS
        or experiment_cfg.get("models", {}).get("use_spherical_range_cnn_gru", False)
    )

    if use_custom_model and args_cli.ml_framework.startswith("torch"):
        # Manual instantiation for custom model tasks (bypass Runner's model instantiator)
        model_class = MODEL_CLASSES[args_cli.model]
        print(f"[INFO] Using custom model: {model_class.__name__} (bypassing Runner)")
        
        device = env.device
        observation_space = env.observation_space
        action_space = env.action_space
        
        # Extract model kwargs from experiment_cfg
        if "models" not in experiment_cfg:
            raise ValueError("Missing 'models' section in experiment_cfg. Check your skrl config YAML.")
        if "policy" not in experiment_cfg["models"]:
            raise ValueError("Missing 'models.policy' section in experiment_cfg. Check your skrl config YAML.")
        
        policy_kwargs = experiment_cfg["models"]["policy"].get("kwargs", {})
        if not policy_kwargs:
            raise ValueError("Missing 'models.policy.kwargs' in experiment_cfg. Model requires kwargs (range_H, range_W, etc.).")
        
        policy_kwargs = dict(policy_kwargs)  # make a mutable copy
        policy_kwargs.setdefault("num_envs", env.num_envs)
        
        # Instantiate model (shared: same instance for policy and value)
        print(f"[INFO] Instantiating {model_class.__name__}...")
        print(f"[INFO] Model kwargs: {policy_kwargs}")
        model = model_class(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **policy_kwargs
        )
        models = {"policy": model, "value": model}
        
        # Instantiate memory
        memory_cfg = experiment_cfg.get("memory", {})
        memory_size = memory_cfg.get("memory_size", -1)
        if memory_size < 0:
            memory_size = experiment_cfg["agent"]["rollouts"]
        print(f"[INFO] Instantiating RandomMemory with size={memory_size}...")
        memory = RandomMemory(num_envs=env.num_envs, device=device, memory_size=memory_size)
        
        # Configure PPO agent using the same config processing as Runner
        ppo_cfg, _ = build_ppo_cfg(
            experiment_cfg["agent"],
            observation_space=observation_space,
            device=device,
        )
        # Inference: skip TensorBoard / checkpoint dir creation in agent.init().
        experiment = ppo_cfg.setdefault("experiment", {})
        if isinstance(experiment, dict):
            experiment["write_interval"] = 0
            experiment["checkpoint_interval"] = 0

        RNN_MODELS = {"spherical_range_cnn_gru"}
        AgentClass = PPO_RNN if args_cli.model in RNN_MODELS else PPO
        print(f"[INFO] Instantiating {AgentClass.__name__} agent...")
        agent = AgentClass(
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
        # Use standard Runner for baseline tasks or non-torch frameworks
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False)
        agent = runner.agent

    # reset environment
    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            states = env.state() if hasattr(env, "state") else None
            outputs = agent.act(obs, states, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, terminated, truncated, _ = env.step(actions)

            # Carry RNN hidden states forward (PPO_RNN only)
            if hasattr(agent, "_rnn") and agent._rnn:
                done = (terminated | truncated).squeeze(-1)
                finished_ids = done.nonzero(as_tuple=False).flatten()
                if finished_ids.numel():
                    for rnn_state in agent._rnn_final_states["policy"]:
                        rnn_state[:, finished_ids] = 0
                agent._rnn_initial_states = agent._rnn_final_states

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
