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
"""GNP package — import-side-effect free until AppLauncher initializes Kit."""

__version__ = "0.1.0"
__all__ = [
    "ArbitrarySensorSetupEnvCfg",
    "ArbitrarySensorSetupEnvCfg_PLAY",
    "register_tasks",
]


def register_tasks() -> None:
    """Import task package to register Gym environments."""
    from . import tasks as _tasks  # noqa: F401


def __getattr__(name: str):
    """Lazily resolve task config symbols on first access."""
    if name in {"ArbitrarySensorSetupEnvCfg", "ArbitrarySensorSetupEnvCfg_PLAY"}:
        from .tasks import ArbitrarySensorSetupEnvCfg, ArbitrarySensorSetupEnvCfg_PLAY

        return {
            "ArbitrarySensorSetupEnvCfg": ArbitrarySensorSetupEnvCfg,
            "ArbitrarySensorSetupEnvCfg_PLAY": ArbitrarySensorSetupEnvCfg_PLAY,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
