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
"""Spherical-range CNN-GRU actor-critic (skrl PPO_RNN)."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

from .utils import _robot_state_dim_from_space, _conv5_spatial_size, CircularAzimuthConv2d


class SphericalRangeCNNGRU(GaussianMixin, DeterministicMixin, Model):
    """Spherical-range CNN-GRU for dict obs: ``image`` (B, az, el, C) + ``robot_state``.

    Image: H=azimuth, W=elevation; ch0=depth, ch1=validity (optional via ``use_mask_channel``).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        *,
        range_H: int = 360,
        range_W: int = 180,
        image_channels: int = 2,
        use_mask_channel: bool = True,
        num_envs: int = 1,
        num_layers: int = 1,
        hidden_size: int = 64,
        sequence_length: int = 8,
        initial_log_std: float = 0.0,
        pretrained_encoder_path: str = None,
        freeze_encoder: bool = False,
        mask_loss_scale: float = 1.0,
        **kwargs,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
        )
        DeterministicMixin.__init__(self)

        self.device = device
        self.range_H = range_H
        self.range_W = range_W
        self._use_mask_channel = bool(use_mask_channel)
        # True 1-channel architecture when mask is disabled.
        if self._use_mask_channel:
            if int(image_channels) < 2:
                raise ValueError(
                    "image_channels must be >= 2 when use_mask_channel=True "
                    f"(got {image_channels})"
                )
            self.image_channels = int(image_channels)
        else:
            self.image_channels = 1

        print(
            f"[INFO] Spherical CNN-GRU: use_mask_channel={self._use_mask_channel}, "
            f"image_channels={self.image_channels}",
            flush=True,
        )

        # GRU params
        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        self._mask_loss_scale = mask_loss_scale

        self.robot_state_dim = _robot_state_dim_from_space(observation_space)
        num_actions = self.num_actions
        assert num_actions > 0

        # ------------------------------------------------------------------
        # Spherical CNN encoder (circular azimuth + elevation padding)
        # ------------------------------------------------------------------
        conv_h, conv_w = _conv5_spatial_size(range_H, range_W)
        image_encoder_flat = conv_h * conv_w * 128
        # Semantic elevation padding needs depth+mask; use reflect pad for depth-only.
        semantic_elevation = self._use_mask_channel

        self.image_encoder = nn.Sequential(
            CircularAzimuthConv2d(
                self.image_channels, 32,
                kernel_size=5, stride=2,
                semantic_elevation=semantic_elevation,
            ),
            nn.ReLU(inplace=True),
            CircularAzimuthConv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            CircularAzimuthConv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            CircularAzimuthConv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            CircularAzimuthConv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(image_encoder_flat, 128),
            nn.ELU(),
        )
        
        decoder_out_channels = 2 if self._use_mask_channel else 1
        self.decoder = nn.Sequential(
            nn.Linear(128, conv_h * conv_w * 128),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, conv_h, conv_w)),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32, decoder_out_channels, kernel_size=5, stride=2, padding=2, output_padding=(1, 0)
            ),
        )

        # ------------------------------------------------------------------
        # Optionally load pretrained encoder weights
        # ------------------------------------------------------------------
        self._freeze_encoder = freeze_encoder
        self._aux_recon_loss = None
        self._aux_depth_loss = None
        self._aux_mask_loss = None

        if pretrained_encoder_path is not None:
            if not os.path.isabs(pretrained_encoder_path):
                project_root = os.path.normpath(os.path.join(
                    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir,
                ))
                pretrained_encoder_path = os.path.join(project_root, pretrained_encoder_path)
            checkpoint = torch.load(
                pretrained_encoder_path,
                map_location=device,
                weights_only=True,
            )
            self.image_encoder.load_state_dict(checkpoint["encoder"])
            print(f"[INFO] Loaded pretrained encoder from "
                  f"{pretrained_encoder_path} (epoch {checkpoint.get('epoch', '?')})")

            if "decoder" in checkpoint:
                self.decoder.load_state_dict(checkpoint["decoder"])
                print("[INFO] Loaded pretrained decoder")

        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()
            self.decoder.eval()
            print("[INFO] Encoder and decoder frozen (requires_grad=False)")

        # ------------------------------------------------------------------
        # Robot state MLP
        # ------------------------------------------------------------------
        self.robot_state_net = nn.Sequential(
            nn.Linear(self.robot_state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

        # ------------------------------------------------------------------
        # Fusion MLP
        # ------------------------------------------------------------------
        mlp_in = 128 + 32  # image_encoder_out + robot_state_out
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        # ------------------------------------------------------------------
        # GRU
        # ------------------------------------------------------------------
        self.pre_gru_norm = nn.LayerNorm(64)
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # Actor / Critic heads
        # ------------------------------------------------------------------
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((num_actions,), initial_log_std))

        self.to(device)
    
    # ------------------------------------------------------------------
    # Train mode (make sure the encoder stays frozen)
    # ------------------------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        if self._freeze_encoder and mode:
            self.image_encoder.eval()
            self.decoder.eval()
        return self
    
    # ------------------------------------------------------------------
    # Required by PPO_RNN
    # ------------------------------------------------------------------
    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }

    # ------------------------------------------------------------------
    # CNN + robot state + MLP backbone (no GRU)
    # ------------------------------------------------------------------
    def _extract_features(self, obs: dict, role: str) -> torch.Tensor:
        image = obs["image"]
        robot_state = obs["robot_state"]
        if image.device != self.device:
            image = image.to(self.device, non_blocking=True)
            robot_state = robot_state.to(self.device, non_blocking=True)
        # (B, H, W, C) -> (B, C, H, W)
        image = image.permute(0, 3, 1, 2)

        # Keep GT mask only when the mask channel is part of the architecture.
        gt_mask = None
        if self._use_mask_channel and image.shape[1] >= 2:
            gt_mask = image[:, 1:2]
        if not self._use_mask_channel:
            image = image[:, 0:1]

        if self._freeze_encoder:
            with torch.no_grad():
                img_feat = self.image_encoder(image)
        else:
            img_feat = self.image_encoder(image)

            # Auxiliary losses (only during training, unfrozen encoder)
            if self.training and role == "policy":
                recon = self.decoder(img_feat)           # (B, C_out, H, W)
                depth = image[:, 0:1]                    # (B, 1, H, W)
                recon_depth = recon[:, 0:1]

                if self._use_mask_channel:
                    # Depth: masked MSE on valid pixels; mask: BCE over all pixels.
                    mask = gt_mask if gt_mask is not None else image[:, 1:2]
                    loss_depth = ((recon_depth - depth) ** 2 * mask).sum() / (
                        mask.sum() + 1e-8
                    )
                    recon_mask = torch.sigmoid(recon[:, 1:2])
                    loss_mask = F.binary_cross_entropy(recon_mask, mask)
                    # Combined auxiliary loss (individual scales applied here,
                    # global aux_recon_loss_scale applied by PPO_RNN_AuxRecon)
                    self._aux_recon_loss = (
                        loss_depth
                        + self._mask_loss_scale * loss_mask
                    )
                    self._aux_mask_loss = loss_mask.item()
                else:
                    # No-mask ablation: plain depth MSE, no validity weighting / mask BCE.
                    loss_depth = F.mse_loss(recon_depth, depth)
                    self._aux_recon_loss = loss_depth
                    self._aux_mask_loss = None

                self._aux_depth_loss = loss_depth.item()

        state_feat = self.robot_state_net(robot_state)

        return self.mlp(torch.cat([img_feat, state_feat], dim=1))

    # ------------------------------------------------------------------
    # act() dispatch
    # ------------------------------------------------------------------
    def act(self, inputs, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role=role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        else:
            raise ValueError(f"Unknown role: {role}")

    def get_entropy(self, *, role: str = "") -> torch.Tensor:
        # skrl 2.x sums over action dims before PPO's .mean(); skrl 1.x returned
        # per-dim entropy so .mean() averaged over actions too (~1/num_actions).
        # Match skrl 1 / seed_43 entropy bonus scale.
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)

    # ------------------------------------------------------------------
    # compute() — called by act() via the Mixin
    # ------------------------------------------------------------------
    def compute(self, inputs, role):
        # Unpack observations (skrl 2.x: "observations"; fallback: "states")
        obs_tensor = inputs.get("observations", inputs.get("states"))
        states = unflatten_tensorized_space(
            self.observation_space,
            obs_tensor,
        )
        obs = states.get("policy", states) if isinstance(states, dict) else states

        terminated = inputs.get("terminated", None)
        truncated = inputs.get("truncated", None)
        if terminated is not None and truncated is not None:
            done = terminated | truncated
        else:
            done = terminated
        # skrl wrappers store dones as (N, 1); squeeze so view(B, T) is unambiguous.
        if done is not None and done.ndim > 1:
            done = done.reshape(done.shape[0], -1).any(dim=-1)
        hidden = inputs["rnn"][0]  # (num_layers, B_or_BxT, hidden_size)

        # CNN + MLP features
        feats = self._extract_features(obs, role)  # (B, 64)  or  (B*T, 64)

        # Stabilize feature scale for GRU
        feats = self.pre_gru_norm(feats)

        # Sequence vs single-step MUST NOT key off ``self.training``.
        # skrl rollouts often leave the module in train mode; using sequence packing
        # then reinterprets (num_envs,) as (num_envs/seq_len, seq_len) and breaks control.
        # PPO_RNN only passes terminated/truncated during the update's sequence batches.
        use_sequence = done is not None

        # ==============================================================
        # TRAINING UPDATE: process sequences with termination-aware reset
        # ==============================================================
        if use_sequence:
            B = feats.shape[0] // self.sequence_length

            rnn_in = feats.view(B, self.sequence_length, -1)

            # hidden arrives as (num_layers, B*T, hidden_size) — take t=0 slice
            hidden = hidden.view(
                self.num_layers,
                B,
                self.sequence_length,
                self.hidden_size,
            )[:, :, 0, :].contiguous()

            if torch.any(done):
                outputs = []
                done = done.view(B, self.sequence_length)

                # Find split points where any env terminated/truncated
                idx = (
                    [0]
                    + (done[:, :-1].any(0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(idx) - 1):
                    i0, i1 = idx[i], idx[i + 1]
                    out, hidden = self.gru(rnn_in[:, i0:i1], hidden)
                    hidden[:, done[:, i1 - 1]] = 0
                    outputs.append(out)

                rnn_out = torch.cat(outputs, dim=1)
            else:
                rnn_out, hidden = self.gru(rnn_in, hidden)

        # ==============================================================
        # ROLLOUT: single-step
        # ==============================================================
        else:
            rnn_in = feats.unsqueeze(1)  # (B, 1, 64)
            rnn_out, hidden = self.gru(rnn_in, hidden)

        # Flatten sequence dim: (B, T, hidden_size) -> (B*T, hidden_size)
        rnn_out = torch.flatten(rnn_out, 0, 1)

        # ==============================================================
        # Heads
        # ==============================================================
        if role == "policy":
            mu = self.actor(rnn_out)
            return mu, {"log_std": self.log_std, "rnn": [hidden]}
        elif role == "value":
            return self.critic(rnn_out), {"rnn": [hidden]}
        else:
            raise ValueError(f"Unknown role: {role}")