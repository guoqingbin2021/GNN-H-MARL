"""Measured Pegasus workflow traces, with explicit edge-simulation scaling.

The source JSONs are WfCommons execution traces, not vehicular measurements.
We preserve their dependency edges, wall runtimes and file sizes.  Runtime is
converted to *equivalent* gigacycles using one declared reference CPU and one
global time scale; it is not a claim that hardware cycle counters were measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, resolve_path


LOADER_VERSION = "pegasus-induced-windows-v1"
DEFAULT_FAMILIES = ("montage", "cycles", "seismology")


@dataclass(frozen=True)
class WorkflowBank:
    adjacency: np.ndarray
    node_mask: np.ndarray
    cycles_g: np.ndarray
    input_mb: np.ndarray
    output_mb: np.ndarray
    criticality: np.ndarray
    edge_data_mb: np.ndarray
    family_ids: np.ndarray
    metadata: dict[str, Any]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _settings(config: Config) -> dict[str, Any]:
    families = tuple(str(v).lower() for v in config.data.get("workflow_families", DEFAULT_FAMILIES))
    if not families or len(set(families)) != len(families):
        raise ValueError("workflow_families must contain distinct nonempty family names")
    if any(f not in DEFAULT_FAMILIES for f in families):
        raise ValueError(f"Unsupported Pegasus workflow family: {families}")
    settings = {
        "families": families,
        "holdout_fraction": float(config.data.get("workflow_holdout_fraction", 0.25)),
        "split_seed": int(config.data.get("workflow_split_seed", 1729)),
        "windows_per_trace": int(config.data.get("workflow_windows_per_trace", 4)),
        "num_subtasks": int(config.system.num_subtasks),
        "runtime_scale": float(config.task.get("workflow_runtime_scale", 0.03)),
        "reference_cpu_ghz": float(config.task.get("workflow_reference_cpu_ghz", 2.0)),
        "data_scale": float(config.task.get("workflow_data_scale", 0.02)),
        "loader_version": LOADER_VERSION,
    }
    if not 0.0 < settings["holdout_fraction"] < 1.0:
        raise ValueError("workflow_holdout_fraction must be strictly between zero and one")
    if settings["windows_per_trace"] < 1 or settings["num_subtasks"] < 2:
        raise ValueError("Workflow banks need at least one window and two nodes")
    for key in ("runtime_scale", "reference_cpu_ghz", "data_scale"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Workflow {key} must be finite and positive")
    return settings


@lru_cache(maxsize=128)
def _source_hash(path: str, size: int, mtime_ns: int) -> str:
    # File metadata forms the cache key; the fingerprint itself hashes content.
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workflow_dataset_manifest(config: Config) -> dict[str, Any]:
    """Hash all source files and partition whole traces before window extraction.

    Identical source-content hashes are grouped to prevent a duplicated JSON
    from appearing on both sides of the split.  The seed is experiment-global,
    independent of policy seeds, channel preferences and evaluation outcomes.
    """
    settings = _settings(config)
    root = resolve_path(config, config.data.get("workflow_root", "data/pegasus"))
    sources: list[dict[str, Any]] = []
    for family in settings["families"]:
        paths = sorted((root / family).glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No Pegasus JSON traces in {root / family}")
        family_sources = []
        for path in paths:
            stat = path.stat()
            family_sources.append({
                "family": family,
                "path": path.relative_to(root).as_posix(),
                "sha256": _source_hash(str(path), stat.st_size, stat.st_mtime_ns),
                "bytes": stat.st_size,
            })
        groups = sorted(
            {v["sha256"] for v in family_sources},
            key=lambda sha: _digest([settings["split_seed"], family, sha]),
        )
        if len(groups) < 2:
            raise ValueError(f"{family} needs at least two distinct traces for a leakage-free split")
        count = max(1, min(len(groups) - 1, int(math.ceil(len(groups) * settings["holdout_fraction"]))))
        heldout = set(groups[:count])
        for source in family_sources:
            source["split"] = "heldout" if source["sha256"] in heldout else "train"
        sources.extend(family_sources)
    payload = {"settings": settings, "sources": sources,
               "source_fingerprint": workflow_source_fingerprint(config)}
    return {**payload, "fingerprint": _digest(payload), "root": str(root)}


def workflow_source_fingerprint(config: Config) -> str:
    """Source identity independent of window size, scales and selected families.

    This identity allows checkpoint transfer to family/size scenarios while the
    stricter ``workflow_fingerprint`` still guards per-experiment resume caches.
    All three original source families are included even for a family-only run.
    """
    root = resolve_path(config, config.data.get("workflow_root", "data/pegasus"))
    hashes = []
    for family in DEFAULT_FAMILIES:
        paths = sorted((root / family).glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No Pegasus JSON traces in {root / family}")
        for path in paths:
            stat = path.stat()
            hashes.append((family, _source_hash(str(path), stat.st_size, stat.st_mtime_ns)))
    return _digest({"loader_version": LOADER_VERSION, "sources": sorted(set(hashes))})


def workflow_fingerprint(config: Config) -> str:
    return str(workflow_dataset_manifest(config)["fingerprint"])


def _topological_order(children: list[set[int]], parents: list[set[int]]) -> list[int]:
    indegree = [len(p) for p in parents]
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(children):
        raise ValueError("Pegasus task dependencies contain a cycle")
    return order


def _parse_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    workflow = payload["workflow"]
    specification = workflow["specification"]
    tasks = sorted(specification["tasks"], key=lambda task: str(task["id"]))
    ids = [str(task["id"]) for task in tasks]
    index = {value: i for i, value in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError(f"Duplicate task identifiers in {path}")
    children = [set() for _ in tasks]
    parents = [set() for _ in tasks]
    declared_parents: set[tuple[int, int]] = set()
    declared_children: set[tuple[int, int]] = set()
    for i, task in enumerate(tasks):
        for parent in task.get("parents", []):
            if str(parent) not in index:
                raise ValueError(f"Unknown parent {parent!r} in {path}")
            declared_parents.add((index[str(parent)], i))
        for child in task.get("children", []):
            if str(child) not in index:
                raise ValueError(f"Unknown child {child!r} in {path}")
            declared_children.add((i, index[str(child)]))
    if all("parents" in t and "children" in t for t in tasks) and declared_parents != declared_children:
        raise ValueError(f"Parent/child dependency declarations disagree in {path}")
    for source, target in declared_parents | declared_children:
        children[source].add(target)
        parents[target].add(source)
    order = _topological_order(children, parents)
    runtimes = {str(t["id"]): float(t["runtimeInSeconds"]) for t in workflow["execution"]["tasks"]}
    runtime = np.asarray([runtimes[task_id] for task_id in ids], dtype=np.float64)
    if not np.isfinite(runtime).all() or (runtime < 0).any():
        raise ValueError(f"Invalid measured runtimes in {path}")
    file_bytes: dict[str, int] = {}
    for item in specification["files"]:
        size = item["sizeInBytes"]
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0 or int(size) != size:
            raise ValueError(f"Invalid file size in {path}: {item['id']}")
        name = str(item["id"])
        if name in file_bytes and file_bytes[name] != int(size):
            raise ValueError(f"Conflicting file sizes in {path}: {name}")
        file_bytes[name] = int(size)
    inputs = [set(str(f) for f in t.get("inputFiles", [])) for t in tasks]
    outputs = [set(str(f) for f in t.get("outputFiles", [])) for t in tasks]
    for names in inputs + outputs:
        missing = names - file_bytes.keys()
        if missing:
            raise ValueError(f"Missing file sizes in {path}: {sorted(missing)[:3]}")
    return {
        "ids": ids, "children": children, "parents": parents, "order": order,
        "runtime": runtime, "inputs": inputs, "outputs": outputs,
        "file_bytes": file_bytes, "name": payload.get("name"),
        "description": payload.get("description"), "schema_version": payload.get("schemaVersion"),
    }


def _connected_windows(trace: dict[str, Any], size: int, count: int, seed: int) -> list[list[int]]:
    """Deterministic BFS windows on the undirected *existing* dependency graph."""
    neighbors = [p | c for p, c in zip(trace["parents"], trace["children"])]
    remaining = set(range(len(neighbors)))
    eligible: list[int] = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = [start]
        remaining.remove(start)
        for node in queue:
            unseen = neighbors[node] & remaining
            remaining.difference_update(unseen)
            component.update(unseen)
            queue.extend(sorted(unseen))
        if len(component) >= size:
            eligible.extend(sorted(component))
    if not eligible:
        raise ValueError(f"Trace has no connected component with {size} nodes")
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(eligible).tolist()
    windows = []
    seen: set[tuple[int, ...]] = set()
    for start in candidates:
        chosen = {start}
        queue = [start]
        for node in queue:
            for neighbor in rng.permutation(sorted(neighbors[node])).tolist():
                if neighbor not in chosen:
                    chosen.add(neighbor)
                    queue.append(neighbor)
                if len(chosen) == size:
                    break
            if len(chosen) == size:
                break
        # Global topological order makes the retained adjacency upper triangular.
        nodes = [node for node in trace["order"] if node in chosen]
        key = tuple(nodes)
        if key not in seen:
            windows.append(nodes)
            seen.add(key)
        if len(windows) == count:
            break
    return windows


def load_workflow_bank(config: Config, split: str | None = None) -> WorkflowBank:
    """Load trace-disjoint training or heldout windows; never fabricate a DAG.

    ``input_mb`` contains external inputs to the retained window.  Input files
    produced by a retained direct predecessor occur in ``edge_data_mb`` only,
    so an environment must add that edge payload once, not all parent outputs.
    Dropped predecessors are declared boundary conditions: their measured files
    are present at window arrival.  Thus windows are real induced subworkflows,
    not claims to execute an entire original scientific workflow.
    """
    manifest = workflow_dataset_manifest(config)
    requested = str(split or config.data.get("workflow_split", "train")).lower()
    if requested in {"test", "eval", "validation"}:
        requested = "heldout"
    if requested not in {"train", "heldout"}:
        raise ValueError("workflow_split must be train or heldout")
    return _load_bank(json.dumps(manifest, sort_keys=True), requested)


@lru_cache(maxsize=8)
def _load_bank(manifest_json: str, split: str) -> WorkflowBank:
    manifest = json.loads(manifest_json)
    settings = manifest["settings"]
    n = settings["num_subtasks"]
    entries: list[dict[str, Any]] = []
    sources = []
    seen_sources: set[str] = set()
    for source in manifest["sources"]:
        if source["split"] != split or source["sha256"] in seen_sources:
            continue
        seen_sources.add(source["sha256"])
        path = Path(manifest["root"]) / source["path"]
        trace = _parse_trace(path)
        seed = int(_digest([settings["split_seed"], source["sha256"]])[:16], 16)
        windows = _connected_windows(trace, n, settings["windows_per_trace"], seed)
        sources.append({**source, "num_tasks": len(trace["ids"]), "name": trace["name"],
                        "description": trace["description"], "schema_version": trace["schema_version"]})
        for nodes in windows:
            adjacency = np.zeros((n, n), dtype=np.bool_)
            edge_bytes = np.zeros((n, n), dtype=np.float64)
            external_bytes = np.zeros(n, dtype=np.float64)
            output_bytes = np.zeros(n, dtype=np.float64)
            retained = {node: i for i, node in enumerate(nodes)}
            for target_i, target in enumerate(nodes):
                internal_inputs: set[str] = set()
                for parent in trace["parents"][target]:
                    if parent not in retained:
                        continue
                    source_i = retained[parent]
                    adjacency[source_i, target_i] = True
                    transferred = trace["outputs"][parent] & trace["inputs"][target]
                    # Trace files have one producer; reject ambiguous provenance.
                    if transferred & internal_inputs:
                        raise ValueError(f"Multiple retained producers for an input in {path}")
                    internal_inputs.update(transferred)
                    edge_bytes[source_i, target_i] = sum(trace["file_bytes"][f] for f in transferred)
                external_bytes[target_i] = sum(trace["file_bytes"][f] for f in trace["inputs"][target] - internal_inputs)
                output_bytes[target_i] = sum(trace["file_bytes"][f] for f in trace["outputs"][target])
            criticality = np.ones(n, dtype=np.float32)
            for node in range(n - 1, -1, -1):
                child_ids = np.flatnonzero(adjacency[node])
                if len(child_ids):
                    criticality[node] += criticality[child_ids].max()
            criticality /= criticality.max()
            runtime = trace["runtime"][nodes]
            entries.append({
                "adjacency": adjacency,
                "cycles_g": runtime * settings["reference_cpu_ghz"] * settings["runtime_scale"],
                "input_mb": external_bytes * settings["data_scale"] / 1e6,
                "output_mb": output_bytes * settings["data_scale"] / 1e6,
                "edge_data_mb": edge_bytes * settings["data_scale"] / 1e6,
                "criticality": criticality,
                "family_id": settings["families"].index(source["family"]),
                "provenance": {"family": source["family"], "source": source["path"],
                    "sha256": source["sha256"], "task_ids": [trace["ids"][i] for i in nodes],
                    "runtime_seconds": runtime.tolist(), "external_input_bytes": external_bytes.tolist(),
                    "output_bytes": output_bytes.tolist(), "boundary_parent_edges": sum(
                        len(trace["parents"][i] - retained.keys()) for i in nodes)},
            })
    arrays = {
        key: np.stack([entry[key] for entry in entries]).astype(np.bool_ if key == "adjacency" else np.float32)
        for key in ("adjacency", "cycles_g", "input_mb", "output_mb", "edge_data_mb", "criticality")
    }
    arrays["family_ids"] = np.asarray([entry["family_id"] for entry in entries], dtype=np.int64)
    arrays["node_mask"] = np.ones((len(entries), n), dtype=np.bool_)
    for array in arrays.values():
        array.setflags(write=False)
    metadata = {
        "fingerprint": manifest["fingerprint"], "split": split,
        "source_fingerprint": manifest["source_fingerprint"],
        "loader_version": LOADER_VERSION, "settings": settings, "sources": sources,
        "windows": [entry["provenance"] for entry in entries],
        "interpretation": "Measured scientific-workflow induced windows; equivalent scaled compute work, not measured vehicular cycles.",
        "boundary_rule": "Dropped predecessors are pre-materialized external inputs at window arrival.",
        "megabyte_bytes": 1000000,
    }
    return WorkflowBank(**arrays, metadata=metadata)
