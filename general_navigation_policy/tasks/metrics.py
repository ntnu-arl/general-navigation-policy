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
"""Episode success criteria and env extras['log'] metric helpers."""

from __future__ import annotations

import torch

# Matches ``evaluate_skrl.py`` / success logging default.
DEFAULT_SUCCESS_RADIUS_M = 1.0


def episode_success_mask(
    reached_during_episode: torch.Tensor,
    final_errors: torch.Tensor,
    success_radius: float,
) -> torch.Tensor:
    """Episode success mask aligned with offline skrl evaluation."""
    return reached_during_episode | (final_errors <= success_radius)


def is_episode_success(
    reached_during_episode: bool,
    final_error_m: float,
    success_radius: float,
) -> bool:
    """Scalar episode success check (same definition as :func:`episode_success_mask`)."""
    return reached_during_episode or final_error_m <= success_radius


def add_obstacle_curriculum_metrics(extras_log: dict, cfg) -> None:
    """Log Metrics/num_obstacles and Metrics/curriculum_level when curriculum is present."""
    try:
        # Prefer Lab dict logging (level/success_rate); fall back to scalar term state.
        level = extras_log.get("Curriculum/obstacle_levels/level")
        if level is None:
            level = extras_log.get("Curriculum/obstacle_levels")
        if level is None:
            return
        level = float(level.item() if hasattr(level, "item") else level)
        events = getattr(cfg, "events", None)
        curriculum = getattr(cfg, "curriculum", None)
        reset_obst = getattr(events, "reset_obstacles", None) if events else None
        obst_levels = getattr(curriculum, "obstacle_levels", None) if curriculum else None
        if reset_obst is None or obst_levels is None:
            return
        params = getattr(reset_obst, "params", None) or {}
        obst_params = getattr(obst_levels, "params", None) or {}
        min_num = params.get("min_num_obstacles")
        max_num = params.get("max_num_obstacles")
        max_diff = obst_params.get("max_difficulty")
        if min_num is None or max_num is None or max_diff is None or max_diff == 0:
            return
        num_obstacles = min_num + (level / max_diff) * (max_num - min_num)
        extras_log["Metrics/num_obstacles"] = float(num_obstacles)
        extras_log["Metrics/curriculum_level"] = level
    except (AttributeError, TypeError, KeyError):
        pass
