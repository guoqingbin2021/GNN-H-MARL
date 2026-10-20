# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


SHARED_OBSERVATION_KEYS = (
    "resource",
    "pair",
    "demand_regions",
    "demand_region_mask",
    "uav_region_pair",
)


@dataclass
class CommunicationObservationFilter:
    """Apply evaluation-time communication availability to shared observations.

    Complete communication returns the current observation unchanged. Intermittent
    communication keeps local state and true feasibility masks current, while
    shared/remote observations may be replaced by their last communicated values.
    """

    mode: str = "complete"
    drop_prob: float = 0.5
    seed: int = 1
    shared_keys: tuple[str, ...] = SHARED_OBSERVATION_KEYS
    _cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False)
    _generator: torch.Generator = field(init=False, repr=False)
    calls: int = field(default=0, init=False)
    drops: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"complete", "intermittent"}:
            raise ValueError("mode must be 'complete' or 'intermittent'")
        if not 0.0 <= float(self.drop_prob) <= 1.0:
            raise ValueError("drop_prob must be in [0, 1]")
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(self.seed))

    def _communication_available(self) -> bool:
        if not self._cache:
            return True
        sample = torch.rand((), generator=self._generator).item()
        return sample >= float(self.drop_prob)

    def __call__(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.mode == "complete":
            return obs
        available = self._communication_available()
        self.calls += 1
        if not available:
            self.drops += 1
        filtered = dict(obs)
        for key in self.shared_keys:
            if key not in obs:
                continue
            if available or key not in self._cache:
                self._cache[key] = obs[key].detach().clone()
                filtered[key] = obs[key]
            else:
                filtered[key] = self._cache[key].to(
                    device=obs[key].device,
                    dtype=obs[key].dtype,
                )
        return filtered

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "drop_prob": float(self.drop_prob),
            "calls": int(self.calls),
            "drops": int(self.drops),
            "observed_drop_fraction": (
                float(self.drops / self.calls) if self.calls else 0.0
            ),
        }
