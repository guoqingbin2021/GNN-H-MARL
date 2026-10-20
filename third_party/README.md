# Third-party baseline components

This directory includes the source components used by the UAMCO-DAG baseline adapters. Files are copied from the local research project without changing their numerical implementation. The upstream revisions below are those recorded by the project; nested Git metadata was not retained, so this snapshot does not independently certify pristine upstream source identity.

| Baseline | Recorded upstream revision | Included components |
| --- | --- | --- |
| MAPPO | `de66d7a4b23fac2513f56f96f73b3f5cb96695ac` | `R_MAPPOPolicy`, `R_MAPPO`, recurrent actor/critic networks, shared rollout buffer, ValueNorm, and supporting utilities. |
| HAPPO | `b1af98b0dbab72a2eee9d160751cd09aedbb8ce2` | HARL actors/critics, network definitions, buffers, ValueNorm, and shared utilities. |
| CAVDN | `17126934c71e64b0669ad662cdbd6b1e569a8864` | `network/base_net.py`, `network/coord_net.py`, and `network/qmix_net.py`, providing `RNN_FC`, `CoordNet`, `Q_out`, and `QMixNet`. |
| DROO | `563d6c7723463c3e6b77b470e3200268c5fad4f3` | PyTorch `MemoryDNN` implementation in `memoryPyTorch.py`. |

The adapters in `uamco/official_baselines.py` and `uamco/sota_baselines.py` provide the shared-environment bridge. For CAVDN, the vectorized recurrent replay/training logic is in the project adapter, while the network classes above are imported from this directory. For DROO, the adapter uses `MemoryDNN` together with the project's RSU/UAV candidate evaluation and vectorized order-preserving decoding. These are adaptations to the shared simulator, not executions of the upstream benchmark environments.

HARL package registries eagerly import other HARL actor and critic classes, so those classes and their dependencies are retained even though the experiment adapter selects HAPPO. `harl/envs/env_wrappers.py` is retained because the common shape utilities import it. Its package initializer also imports benchmark logger classes, so those logger definitions are retained; the game-specific and robot-specific simulator implementations and assets are not bundled. The MAPPO package similarly requires its `envs` and `scripts` initializers and a `runner` namespace directory; their benchmark launchers are not included.

## Provenance and license status

- MAPPO: the existing MIT license file is retained at `on-policy/LICENSE`.
- HAPPO/HARL: the retained upstream packaging file `HARL/setup.py` declares MIT and identifies PKU-MARL. No separate license text was present in the local source tree; none has been invented here.
- DROO: the existing MIT license file is retained at `SOTA/DROO/DROO/LICENSE`.
- CAVDN: the project provenance record states that no upstream license file was declared. The local network sources and upstream README are included for the requested local package, but public redistribution permission has not been established.

Some local third-party Python files contain project-added paper/SPDX headers. Those headers are not evidence of a license grant from the upstream authors, do not transfer their ownership, and must not be read as relicensing their work under the root project license. Original third-party ownership and license status take precedence.

`SOURCES.json` preserves the project's upstream provenance record. `SNAPSHOT.json` records SHA-256 hashes of every copied file in this directory, excluding the newly written documentation and the snapshot file itself. It identifies the local source subset, not a new experimental result or a claim that upstream licensing has been cleared.

## Scope

No pretrained weights, result plots, datasets, shell launchers, or original benchmark-training entry points are bundled. This subset supports the project adapters; it is not intended to run every environment or utility advertised in the retained upstream READMEs. Use the package-level `requirements.txt` instead of installing every dependency from the historic upstream benchmark requirements.
