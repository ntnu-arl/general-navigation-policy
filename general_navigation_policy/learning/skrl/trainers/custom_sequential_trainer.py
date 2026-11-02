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
from skrl.trainers.torch import SequentialTrainer


class CustomSequentialTrainer(SequentialTrainer):
    """SequentialTrainer with plain-scalar extras logging + RNN reset on episode end.

    Also prefixes Info / tags so TensorBoard keys match pre-skrl2 runs.
    """

    def _log_environment_info(self, infos: dict) -> None:
        """Log all env.extras['log'] scalars to tensorboard (env-agnostic)."""
        import torch

        if self.cfg.environment_info not in infos:
            return
        log_dict = infos[self.cfg.environment_info]
        if not isinstance(log_dict, dict):
            return
        for k, v in log_dict.items():
            # Always prefix so tags match pre-skrl2 runs (e.g. Info / Curriculum/...).
            # skrl 2 skips the prefix when "/" is already in k, which broke TB comparisons.
            key = f"Info / {k}"
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                self.agents.track_data(key, v.item())
            elif isinstance(v, (int, float)):
                self.agents.track_data(key, float(v))
            elif hasattr(v, "item") and hasattr(v, "dtype"):
                try:
                    self.agents.track_data(key, float(v.item()))
                except (AttributeError, ValueError):
                    pass

    def train(self) -> None:
        """Train a single agent (skrl 2 Trainer.train + GNP extras)."""
        import sys

        import torch
        import tqdm
        from skrl.utils import ScopedTimer

        assert self.num_simultaneous_agents == 1, (
            "CustomSequentialTrainer only supports a single simultaneous agent."
        )

        self.agents.enable_training_mode(True)
        # Rollouts must use single-step RNN; keep modules in eval between updates.
        self.agents.enable_models_training_mode(False)

        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                with ScopedTimer() as timer:
                    actions, _outputs = self.agents.act(
                        observations, states, timestep=timestep, timesteps=self.cfg.timesteps
                    )
                    self.agents.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    self.agents.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                self.agents.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.cfg.timesteps,
                )
                self._log_environment_info(infos)

                if (terminated.any() or truncated.any()) and hasattr(self.agents.policy, "reset"):
                    env_ids = torch.where(terminated | truncated)[0]
                    self.agents.policy.reset(env_ids)

            self.agents.post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    if hasattr(self.agents.policy, "reset"):
                        self.agents.policy.reset(None)
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states
