# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TerrainMap:
    lats: torch.Tensor
    lons: torch.Tensor
    elevation: torch.Tensor

    @classmethod
    def from_json(cls, path: str | Path, device: torch.device) -> "TerrainMap":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        grid_n = int(payload["grid_n"])
        elevation = torch.tensor(payload["elevation"], dtype=torch.float32, device=device)
        elevation = elevation.reshape(grid_n, grid_n)
        return cls(
            lats=torch.tensor(payload["lats"], dtype=torch.float32, device=device),
            lons=torch.tensor(payload["lons"], dtype=torch.float32, device=device),
            elevation=elevation,
        )

    def sample(self, latlon: torch.Tensor) -> torch.Tensor:
        lat = latlon[..., 0]
        lon = latlon[..., 1]
        lat_pos = (lat - self.lats[0]) / (self.lats[-1] - self.lats[0]) * (len(self.lats) - 1)
        lon_pos = (lon - self.lons[0]) / (self.lons[-1] - self.lons[0]) * (len(self.lons) - 1)
        lat0 = lat_pos.floor().long().clamp(0, len(self.lats) - 2)
        lon0 = lon_pos.floor().long().clamp(0, len(self.lons) - 2)
        lat1 = lat0 + 1
        lon1 = lon0 + 1
        wy = (lat_pos - lat0.float()).clamp(0.0, 1.0)
        wx = (lon_pos - lon0.float()).clamp(0.0, 1.0)
        z00 = self.elevation[lat0, lon0]
        z01 = self.elevation[lat0, lon1]
        z10 = self.elevation[lat1, lon0]
        z11 = self.elevation[lat1, lon1]
        return (
            z00 * (1.0 - wx) * (1.0 - wy)
            + z01 * wx * (1.0 - wy)
            + z10 * (1.0 - wx) * wy
            + z11 * wx * wy
        )

    def blocked(
        self,
        start_latlon: torch.Tensor,
        end_latlon: torch.Tensor,
        start_height_m: float,
        end_height_m: float,
        samples: int = 7,
        clearance_m: float = 2.0,
    ) -> torch.Tensor:
        fractions = torch.linspace(
            0.1,
            0.9,
            samples,
            dtype=start_latlon.dtype,
            device=start_latlon.device,
        )
        view_shape = (1,) * (start_latlon.ndim - 1) + (samples, 1)
        fractions = fractions.view(view_shape)
        points = start_latlon.unsqueeze(-2) * (1.0 - fractions) + end_latlon.unsqueeze(-2) * fractions
        terrain = self.sample(points)
        start_ground = self.sample(start_latlon)
        end_ground = self.sample(end_latlon)
        start_altitude = start_ground + float(start_height_m)
        end_altitude = end_ground + float(end_height_m)
        line_fraction = fractions[..., 0]
        line_altitude = start_altitude.unsqueeze(-1) * (1.0 - line_fraction) + end_altitude.unsqueeze(-1) * line_fraction
        return (terrain + clearance_m > line_altitude).any(dim=-1)

