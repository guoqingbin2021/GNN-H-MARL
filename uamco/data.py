# SPDX-License-Identifier: MIT
# MIT License
# Copyright (c) 2026 Qingbin Guo
# Paper: UAMCO-DAG: UAV-Assisted DAG Offloading for Intermittently Connected Vehicular Edge Computing
# Full license text is available in the root LICENSE file of this repository
# This code is originally released for academic research purposes only
# Any published work utilizing this code must cite the above paper

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .config import Config, resolve_path


EARTH_RADIUS_M = 6_371_000.0


def _latlon_to_xy(coords: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(coords[:, 0])
    lon = np.deg2rad(coords[:, 1])
    lat0 = float(lat.mean())
    lon0 = float(lon.mean())
    x = (lon - lon0) * math.cos(lat0) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
    return np.stack([x, y], axis=-1).astype(np.float32)


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.nanpercentile(values, 5.0, axis=0, keepdims=True)
    high = np.nanpercentile(values, 95.0, axis=0, keepdims=True)
    return np.clip((values - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def _nearest_sample(sensor_payload: dict[str, Any], samples: int, start_hour: int, step_minutes: int):
    counts = np.zeros(samples, dtype=np.float32)
    if not sensor_payload:
        return counts
    minutes = []
    raw_counts = []
    for payload in sensor_payload.values():
        hour, minute = payload.get("timestamp", [start_hour, 0])
        minutes.append((int(hour) - start_hour) * 60 + int(minute))
        raw_counts.append(float(payload.get("count", 0.0)))
    minutes_np = np.asarray(minutes, dtype=np.float32)
    count_np = np.asarray(raw_counts, dtype=np.float32)
    targets = np.arange(samples, dtype=np.float32) * step_minutes
    nearest = np.abs(targets[:, None] - minutes_np[None, :]).argmin(axis=1)
    counts[:] = count_np[nearest]
    return counts


def _select_region(xy: np.ndarray, adjacency: np.ndarray, traffic: np.ndarray, radius_m: float) -> np.ndarray:
    mean_flow = traffic.mean(axis=0)
    degree = adjacency.sum(axis=1)
    score = mean_flow / max(float(mean_flow.max()), 1e-6) + degree / max(float(degree.max()), 1.0)
    center_idx = int(score.argmax())
    distance = np.linalg.norm(xy - xy[center_idx], axis=1)
    selected = np.flatnonzero(distance <= radius_m)
    if selected.size < 32:
        selected = np.argsort(distance)[: min(96, len(distance))]
    return np.sort(selected)


def _padded_neighbors(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighbor_lists = [np.flatnonzero(row > 0).astype(np.int64) for row in adjacency]
    max_degree = max(max((len(row) for row in neighbor_lists), default=0), 1)
    neighbors = np.zeros((len(neighbor_lists), max_degree), dtype=np.int64)
    mask = np.zeros((len(neighbor_lists), max_degree), dtype=np.bool_)
    for idx, row in enumerate(neighbor_lists):
        if len(row) == 0:
            neighbors[idx, 0] = idx
            mask[idx, 0] = True
        else:
            neighbors[idx, : len(row)] = row
            mask[idx, : len(row)] = True
    return neighbors, mask


def prepare_trace_cache(config: Config, force: bool = False) -> Path:
    cache_path = resolve_path(config, config.data.processed_cache)
    if cache_path.exists() and not force:
        return cache_path

    graph_path = resolve_path(config, config.data.streets_graph)
    traffic_path = resolve_path(config, config.data.streets_traffic_zip)
    uci_path = resolve_path(config, config.data.uci_mat)
    with graph_path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    sensor_ids = sorted(int(value) for value in graph["sensor-dictionary"])
    latlon = np.asarray(
        [graph["sensor-dictionary"][str(sensor_id)][0] for sensor_id in sensor_ids],
        dtype=np.float64,
    )
    xy = _latlon_to_xy(latlon)
    adjacency = np.asarray(graph["adjacency-matrix"], dtype=np.float32)

    with zipfile.ZipFile(traffic_path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as handle:
            traffic_dictionary = json.load(handle)

    samples_per_day = (
        (int(config.data.traffic_end_hour) - int(config.data.traffic_start_hour))
        * 60
        // int(config.data.traffic_sample_minutes)
    )
    count_days = []
    for date_key in sorted(traffic_dictionary):
        day = traffic_dictionary[date_key]
        counts = np.zeros((samples_per_day, len(sensor_ids)), dtype=np.float32)
        for column, sensor_id in enumerate(sensor_ids):
            sensor_counts = _nearest_sample(
                day.get(str(sensor_id), {}),
                samples_per_day,
                int(config.data.traffic_start_hour),
                int(config.data.traffic_sample_minutes),
            )
            counts[:, column] = sensor_counts
        count_days.append(counts)
    traffic = np.concatenate(count_days, axis=0)

    selected = _select_region(xy, adjacency, traffic, float(config.data.map_radius_m))
    split = str(config.data.get("spatial_split", "all"))
    if split != "all":
        if split not in {"train", "heldout"}:
            raise ValueError("spatial_split must be all, train, or heldout")
        # Geographic holdout fixed before training, with disjoint sensor IDs.
        order = np.argsort(xy[:, 0], kind="stable")
        cut = max(2, int(0.7 * len(order)))
        selected = np.sort(order[:cut] if split == "train" else order[cut:])
        if len(selected) < 2:
            raise ValueError("Insufficient sensors for a disjoint spatial holdout")
        local_adjacency = adjacency[np.ix_(selected, selected)]
        flow = traffic[:, selected].mean(axis=0)
        degree = local_adjacency.sum(axis=1)
        center = int((flow / max(float(flow.max()), 1.) + degree / max(float(degree.max()), 1.)).argmax())
        distances = np.linalg.norm(xy[selected] - xy[selected[center]], axis=1)
        within = np.flatnonzero(distances <= float(config.data.map_radius_m))
        if len(within) < 4:
            within = np.argsort(distances)[:4]
        selected = np.sort(selected[within])
    xy = xy[selected]
    latlon = latlon[selected]
    adjacency = adjacency[np.ix_(selected, selected)]
    traffic = traffic[:, selected]
    xy -= xy.mean(axis=0, keepdims=True)
    # Exact inverse of the equirectangular projection (no fitted transform).
    original_latitude = np.mean([graph["sensor-dictionary"][str(i)][0][0] for i in sensor_ids])
    degrees_per_m = 180.0 / (math.pi * EARTH_RADIUS_M)
    xy_to_latlon = np.asarray([
        [0.0, degrees_per_m / math.cos(math.radians(original_latitude))],
        [degrees_per_m, 0.0],
        latlon.mean(axis=0),
    ], dtype=np.float64)
    neighbors, neighbor_mask = _padded_neighbors(adjacency)

    uci = loadmat(uci_path)
    uci_volume = np.concatenate(
        [
            np.asarray(uci["tra_Y_tr"], dtype=np.float32),
            np.asarray(uci["tra_Y_te"], dtype=np.float32),
        ],
        axis=1,
    ).T
    uci_scale = _robust_unit_scale(uci_volume.mean(axis=1, keepdims=True)).reshape(-1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        xy=xy.astype(np.float32),
        latlon=latlon.astype(np.float32),
        adjacency=adjacency.astype(np.float32),
        neighbors=neighbors,
        neighbor_mask=neighbor_mask,
        traffic=traffic.astype(np.float32),
        traffic_unit=_robust_unit_scale(traffic).astype(np.float32),
        uci_scale=uci_scale.astype(np.float32),
        selected_sensor_ids=np.asarray(sensor_ids, dtype=np.int64)[selected],
        xy_to_latlon=xy_to_latlon.astype(np.float32),
    )
    return cache_path


@dataclass(frozen=True)
class TraceData:
    xy: np.ndarray
    latlon: np.ndarray
    adjacency: np.ndarray
    neighbors: np.ndarray
    neighbor_mask: np.ndarray
    traffic: np.ndarray
    traffic_unit: np.ndarray
    uci_scale: np.ndarray
    selected_sensor_ids: np.ndarray
    xy_to_latlon: np.ndarray


def load_trace_data(config: Config) -> TraceData:
    cache_path = prepare_trace_cache(config)
    with np.load(cache_path, allow_pickle=False) as payload:
        return TraceData(**{key: payload[key] for key in TraceData.__dataclass_fields__})
