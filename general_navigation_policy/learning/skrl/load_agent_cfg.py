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
"""Load YAML agent/trainer sections into skrl 2 ``PPO_CFG`` / trainer kwargs."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from skrl.agents.torch.ppo import PPO_CFG
from skrl.trainers.torch import SequentialTrainerCfg

# Keys that used to appear in skrl 1 YAMLs. Reject them so configs stay skrl-2-native.
_DEPRECATED_AGENT_KEYS = {
    "lambda": "gae_lambda",
    "lambda_": "gae_lambda",
    "clip_predicted_values": "value_clip (>0 to enable clipping)",
    "state_preprocessor": "observation_preprocessor",
    "state_preprocessor_kwargs": "observation_preprocessor_kwargs",
    "shared_state_preprocessor": "state_preprocessor",
    "shared_state_preprocessor_kwargs": "state_preprocessor_kwargs",
}


def filter_ppo_cfg(agent: dict) -> dict[str, Any]:
    """Keep only keys accepted by ``PPO_CFG``."""
    allowed = {f.name for f in dataclasses.fields(PPO_CFG)}
    return {k: v for k, v in agent.items() if k in allowed}


def filter_trainer_cfg(trainer: dict) -> dict[str, Any]:
    """Keep only keys accepted by ``SequentialTrainerCfg``."""
    allowed = {f.name for f in dataclasses.fields(SequentialTrainerCfg)}
    return {k: v for k, v in trainer.items() if k in allowed}


def _resolve_agent_section(agent: dict) -> dict:
    """Resolve string class names and YAML-only shorthands the way skrl ``Runner`` does."""
    from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    agent = copy.deepcopy(agent)
    agent.pop("class", None)

    deprecated = sorted(k for k in agent if k in _DEPRECATED_AGENT_KEYS)
    if deprecated:
        hints = ", ".join(f"{k} → {_DEPRECATED_AGENT_KEYS[k]}" for k in deprecated)
        raise ValueError(
            "Agent YAML still uses deprecated skrl 1 keys. "
            f"Update to skrl 2 names: {hints}"
        )

    eval_namespace = {
        "KLAdaptiveLR": KLAdaptiveLR,
        "RunningStandardScaler": RunningStandardScaler,
        "GaussianNoise": GaussianNoise,
        "OrnsteinUhlenbeckNoise": OrnsteinUhlenbeckNoise,
    }
    eval_namespace.update(
        __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    )

    class_name_keys = (
        "learning_rate_scheduler",
        "observation_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
        "amp_observation_preprocessor",
        "noise",
        "smooth_regularization_noise",
        "exploration_noise",
    )

    for key, value in list(agent.items()):
        if key in class_name_keys and isinstance(value, str):
            agent[key] = eval(value, eval_namespace)
        elif key.endswith("_kwargs"):
            agent[key] = value if value is not None else {}
        elif key == "rewards_shaper_scale":
            scale = value
            if scale is not None and scale != 1.0:
                agent["rewards_shaper"] = (
                    lambda rewards, *args, scale=scale, **kwargs: rewards * scale
                )
            del agent["rewards_shaper_scale"]

    return agent


def build_ppo_cfg(
    agent_section: dict,
    *,
    observation_space=None,
    device=None,
) -> tuple[dict[str, Any], float]:
    """Build skrl-2 ``PPO_CFG`` kwargs from a YAML agent section.

    Returns ``(ppo_cfg_dict, aux_recon_loss_scale)``.
    ``aux_recon_loss_scale`` is a GNP extension (not part of ``PPO_CFG``).
    """
    agent = _resolve_agent_section(agent_section)
    aux_scale = float(agent.pop("aux_recon_loss_scale", 0.1))

    if agent.get("observation_preprocessor") is not None and observation_space is not None:
        kwargs = dict(agent.get("observation_preprocessor_kwargs") or {})
        kwargs.setdefault("size", observation_space)
        if device is not None:
            kwargs.setdefault("device", device)
        agent["observation_preprocessor_kwargs"] = kwargs

    if agent.get("state_preprocessor") is not None and observation_space is not None:
        kwargs = dict(agent.get("state_preprocessor_kwargs") or {})
        kwargs.setdefault("size", observation_space)
        if device is not None:
            kwargs.setdefault("device", device)
        agent["state_preprocessor_kwargs"] = kwargs

    if agent.get("value_preprocessor") is not None:
        kwargs = dict(agent.get("value_preprocessor_kwargs") or {})
        kwargs.setdefault("size", 1)
        if device is not None:
            kwargs.setdefault("device", device)
        agent["value_preprocessor_kwargs"] = kwargs

    return filter_ppo_cfg(agent), aux_scale
