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
"""PPO_RNN with auxiliary reconstruction loss + skrl 1.4.3 GAE/done semantics."""

from __future__ import annotations

import skrl.agents.torch.ppo.ppo_rnn as ppo_rnn_mod
from skrl.agents.torch.ppo import PPO_RNN


class PPO_RNN_AuxRecon(PPO_RNN):
    """PPO_RNN that adds policy ``_aux_recon_loss`` (scale via ``aux_recon_loss_scale``).

    Restores skrl 1.4.3 episode-boundary semantics for GAE:

    - skrl 1.4.3: ``dones = terminated | truncated`` always cuts GAE.
    - skrl 2.1.0 ``PPO_RNN.compute_gae`` only looks at ``terminated``. With Isaac
      Lab auto-reset and ``time_limit_bootstrap=False`` (~85% timeouts at medium
      curriculum), that bootstraps value targets across episode boundaries into
      the *reset* observation.

    Enabling ``time_limit_bootstrap`` is not a safe substitute on skrl 2.1 RNN:
    ``record_transition`` still adds ``gamma * V(s_t)`` on truncation (current
    value), whereas non-RNN PPO was fixed to use next-state values.

    We monkey-patch ``compute_gae`` for the duration of ``update`` so GAE sees
    ``terminated | truncated`` without mutating the live memory tensor (which
    previously made the restore path a no-op when ``get_tensor_by_name`` returned
    a view that ``copy_`` had already overwritten).
    """

    def __init__(self, *, aux_recon_loss_scale: float = 0.1, **kwargs):
        cfg = kwargs.get("cfg", {})
        if isinstance(cfg, dict) and "aux_recon_loss_scale" in cfg:
            aux_recon_loss_scale = float(cfg.pop("aux_recon_loss_scale"))
        self.aux_recon_loss_scale = float(aux_recon_loss_scale)
        super().__init__(**kwargs)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Inject aux recon loss; use ``terminated | truncated`` for GAE."""
        _original_scale = self.scaler.scale
        aux_scale = self.aux_recon_loss_scale
        _original_compute_gae = ppo_rnn_mod.compute_gae

        def _patched_scale(loss):
            aux_loss = getattr(self.policy, "_aux_recon_loss", None)
            if aux_loss is not None:
                loss = loss + aux_scale * aux_loss
                self._last_aux_recon_loss = aux_loss.item()
            for attr, key in [
                ("_aux_depth_loss", "_last_aux_depth_loss"),
                ("_aux_mask_loss", "_last_aux_mask_loss"),
            ]:
                val = getattr(self.policy, attr, None)
                if val is not None:
                    setattr(self, key, val)
            return _original_scale(loss)

        def _compute_gae_skrl1(*, rewards, terminated, values, next_values,
                               discount_factor=0.99, lambda_coefficient=0.95):
            # skrl 1.4.3: cut GAE on terminated OR truncated (Isaac Lab auto-reset).
            truncated = self.memory.get_tensor_by_name("truncated")
            done = terminated | truncated
            self.track_data("Loss / Timeout fraction (GAE)", truncated.float().mean().item())
            return _original_compute_gae(
                rewards=rewards,
                terminated=done,
                values=values,
                next_values=next_values,
                discount_factor=discount_factor,
                lambda_coefficient=lambda_coefficient,
            )

        self.scaler.scale = _patched_scale
        ppo_rnn_mod.compute_gae = _compute_gae_skrl1
        try:
            super().update(timestep=timestep, timesteps=timesteps)
        finally:
            self.scaler.scale = _original_scale
            ppo_rnn_mod.compute_gae = _original_compute_gae

        if hasattr(self, "_last_aux_recon_loss"):
            self.track_data("Loss / Auxiliary total", self._last_aux_recon_loss)
        if hasattr(self, "_last_aux_depth_loss"):
            self.track_data("Loss / Depth reconstruction", self._last_aux_depth_loss)
        if hasattr(self, "_last_aux_mask_loss"):
            self.track_data("Loss / Mask reconstruction", self._last_aux_mask_loss)
