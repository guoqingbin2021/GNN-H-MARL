# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import yaml


class ConfigNode:
    """Read-only attribute view over a nested configuration dictionary."""

    def __init__(self, values: Mapping[str, Any]):
        self._values = {
            key: ConfigNode(value) if isinstance(value, Mapping) else value
            for key, value in values.items()
        }

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self._values.items():
            result[key] = value.to_dict() if isinstance(value, ConfigNode) else copy.deepcopy(value)
        return result


@dataclass(frozen=True)
class Config:
    root: ConfigNode
    source_path: Path

    def __getattr__(self, name: str) -> Any:
        return getattr(self.root, name)

    def to_dict(self) -> dict[str, Any]:
        return self.root.to_dict()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    parent = payload.get("inherits")
    if parent:
        parent_path = (path.parent / str(parent)).resolve()
        return _deep_merge(_load_yaml(parent_path), payload)
    return payload


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    source = Path(path).resolve()
    values = _load_yaml(source)
    if overrides:
        values = _deep_merge(values, overrides)
    return Config(ConfigNode(values), source)


def resolve_path(config: Config, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = config.source_path.parent.parent
    return (project_root / path).resolve()


def choose_num_envs(config: Config, device_type: str, gpu_memory_gb: float | None = None) -> int:
    configured = config.train.num_envs
    if configured != "auto":
        return int(configured)
    if device_type != "cuda":
        return 16
    memory = float(gpu_memory_gb or 16.0)
    if memory >= 36.0:
        return 512
    if memory >= 20.0:
        return 256
    if memory >= 10.0:
        return 128
    return 64


def training_load_multipliers(config: Config) -> list[float]:
    values = config.train.get("load_multiplier_choices", [1.0])
    if isinstance(values, (int, float)):
        return [float(values)]
    result = [float(value) for value in values]
    return result or [1.0]


def sample_training_load_multipliers(
    config: Config,
    batch: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    values = torch.tensor(
        training_load_multipliers(config),
        dtype=torch.float32,
        device=device,
    )
    if len(values) == 1:
        return values[0].expand(int(batch))
    indices = torch.randint(
        0,
        len(values),
        (int(batch),),
        generator=generator,
        device=device,
    )
    return values.index_select(0, indices)
