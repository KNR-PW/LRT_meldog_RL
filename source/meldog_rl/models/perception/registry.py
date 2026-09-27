# Copyright (c) 2022-2025, Meldog Project
# SPDX-License-Identifier: BSD-3-Clause

"""Perception model registry: one table that names every perception model.

Scripts select a model by its registry name (``--model v3``) and build it with
:func:`build_model`; nothing else should dispatch on model-name strings.

Print the table:  ``python -m meldog_rl.models.perception``, or without the package (no torch):
``python source/meldog_rl/models/perception/registry.py``

Naming history (read before adding or renaming a model)
--------------------------------------------------------
There are two numbering schemes and they DIFFER:

- Thesis names (``v1``, ``v2``, ``v3``) use the model numbers of the master thesis
  (section 2.6).
- Archived names (``<code-era name>_archived``) keep the version number the code used when
  the model was written. Archived models are frozen: their files are not edited again.

Code-era name -> registry name:

- code-era "V1" = ``SimpleMapper``, which read raw depth images. Registered as
  ``v1_archived``: a record only (no class on main, cannot run in this pipeline). It is NOT
  thesis ``v1``.
- code-era "V4" / "V4.1" = thesis ``v1`` / ``v2``. Their names appear only on the archive
  branches ``archive/perception_v2_dev`` / ``archive/perception_v3_dev`` (the branch numbers do
  not match the model numbers), never on main. The names V4 / V4.1 are not registered; the
  classes were copied in as ``v1`` / ``v2``.
- code-era "V5" = thesis ``v3``. The original class stays frozen as ``v5_archived``
  (``heightmap_convgru.py``); ``v3`` (``heightmap_v3.py``) is a copy with identical layer
  names, so their state_dicts are interchangeable. Later work edits only ``v3``.
- code-era "V6" = the autoregressive model. It was never in the thesis. Registered as
  ``v6_archived``.

The old command-line names ``v5``, ``v6``, ``heightmap_v5``, ``heightmap_v6``, ``hmv5`` and
``hmv6`` are rejected with a message naming the new one. Run folders and checkpoints made
before the registry use the code-era names (``PM_v5_*``, ``PM_v6_*``, checkpoint
``config["model"] == "heightmap_v5"``).

Naming new variants:

- a decimal (``v3.1``) for one change on the same backbone;
- a new integer for a new backbone or a new input contract.

This module imports only the standard library; model classes are imported when a model is
built, so the table can be read without torch.
"""

import importlib
import inspect
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# --- vocabulary -------------------------------------------------------------------------------

# kind: what a model consumes per step, i.e. which evaluation/training loop it needs
SINGLE_FRAME = "single_frame"  # (sparse, mask, grav) -> pred
RECURRENT = "recurrent"  # (sparse, mask, grav, hidden) -> (pred, hidden)
AUTOREGRESSIVE = "autoregressive"  # (sparse, mask, prev_pred, prev_valid, grav) -> pred
BASELINE = "baseline"  # non-learned mapper, (sparse, mask, pos, yaw) -> pred

# status
THESIS = "thesis"  # thesis model, copied in; may be edited
ACTIVE = "active"  # the thesis model current work trains and edits
ARCHIVED = "archived"  # frozen, code-era name
BASELINE_STATUS = "baseline"  # non-learned comparison

ACTIVE_MODEL = "v3"

_PKG = "meldog_rl.models.perception"
_DIR = "source/meldog_rl/models/perception"


class ModelRegistryError(ValueError):
    """A model name that does not resolve, or a model that cannot do what was asked."""


@dataclass(frozen=True)
class ModelEntry:
    """One registry row.

    Attributes:
        name: Registry name, used on the command line and in run-folder names.
        class_path: ``"module:Class"`` import path, or None for a record-only entry.
        file: Source file relative to the repository root, or None.
        params: Trainable parameter count with default arguments, or None if not learned.
        kind: ``single_frame``, ``recurrent``, ``autoregressive``, ``baseline``, or None.
        status: ``thesis``, ``active``, ``archived`` or ``baseline``.
        summary: One-line description.
        thesis: Thesis section describing the model, or None.
        source: Where the code came from.
        runnable: The model can be built and run in the current pipeline.
        trainable: ``train_perception.py`` can train it.
    """

    name: str
    class_path: str | None
    file: str | None
    params: int | None
    kind: str | None
    status: str
    summary: str
    thesis: str | None
    source: str
    runnable: bool
    trainable: bool

    @property
    def learned(self) -> bool:
        """True for neural models, False for non-learned baselines."""
        return self.kind != BASELINE

    def load_class(self) -> type:
        """Import and return the model class."""
        _require_runnable(self)
        module_name, _, class_name = self.class_path.partition(":")
        return getattr(importlib.import_module(module_name), class_name)


_ENTRIES = (
    ModelEntry(
        name="v1",
        class_path=f"{_PKG}.heightmap_v1:HeightmapV1",
        file=f"{_DIR}/heightmap_v1.py",
        params=132_403,
        kind=SINGLE_FRAME,
        status=THESIS,
        summary="single-frame U-Net, 2 encoder stages, 10x10 bottleneck",
        thesis="2.6.1",
        source=(
            "archive/perception_v2_dev (21dbd0a) scripts/rsl_rl/train_perception_v0.py, "
            "class SparseMapRefiner (code-era V4)"
        ),
        runnable=True,
        trainable=False,
    ),
    ModelEntry(
        name="v2",
        class_path=f"{_PKG}.heightmap_v2:HeightmapV2",
        file=f"{_DIR}/heightmap_v2.py",
        params=570_387,
        kind=SINGLE_FRAME,
        status=THESIS,
        summary="single-frame U-Net, 3 encoder stages, 5x5 bottleneck",
        thesis="2.6.2",
        source=(
            "archive/perception_v3_dev (ae6c491) scripts/rsl_rl/train_perception_v0.py, "
            "class SparseMapRefinerDeep without attention (code-era V4.1)"
        ),
        runnable=True,
        trainable=False,
    ),
    ModelEntry(
        name="v3",
        class_path=f"{_PKG}.heightmap_v3:HeightmapV3",
        file=f"{_DIR}/heightmap_v3.py",
        params=2_496_275,
        kind=RECURRENT,
        status=ACTIVE,
        summary="3-stage U-Net with a 2-layer ConvGRU at the 5x5 bottleneck",
        thesis="2.6.3",
        source="copy of heightmap_convgru.py, class HeightmapConvGRU (v5_archived)",
        runnable=True,
        trainable=True,
    ),
    ModelEntry(
        name="v5_archived",
        class_path=f"{_PKG}.heightmap_convgru:HeightmapConvGRU",
        file=f"{_DIR}/heightmap_convgru.py",
        params=2_496_275,
        kind=RECURRENT,
        status=ARCHIVED,
        summary="frozen original of v3 (ConvGRU U-Net)",
        thesis=None,
        source="main, class HeightmapConvGRU (code-era V5)",
        runnable=True,
        trainable=False,
    ),
    ModelEntry(
        name="v6_archived",
        class_path=f"{_PKG}.heightmap_autoreg:HeightmapAutoregressive",
        file=f"{_DIR}/heightmap_autoreg.py",
        params=573_910,
        kind=AUTOREGRESSIVE,
        status=ARCHIVED,
        summary="U-Net fed its gated previous output, warped to the current frame",
        thesis=None,
        source="main, class HeightmapAutoregressive (code-era V6, never in the thesis)",
        runnable=True,
        trainable=False,
    ),
    ModelEntry(
        name="v1_archived",
        class_path=None,
        file=None,
        params=781_857,
        kind=None,
        status=ARCHIVED,
        summary="SimpleMapper read raw depth images, not the sparse map",
        thesis=None,
        source=(
            "archive/perception_simplest_dev (78789cc) scripts/rsl_rl/train_perception_v0.py, "
            "class SimpleMapper (code-era V1), never on main"
        ),
        runnable=False,
        trainable=False,
    ),
    ModelEntry(
        name="elevation",
        class_path=f"{_PKG}.elevation_mapper:ElevationMapper",
        file=f"{_DIR}/elevation_mapper.py",
        params=None,
        kind=BASELINE,
        status=BASELINE_STATUS,
        summary="world-frame elevation mapping with Kalman fusion (non-learned)",
        thesis=None,
        source="main, class ElevationMapper",
        runnable=True,
        trainable=False,
    ),
    ModelEntry(
        name="legacy_slam",
        class_path=f"{_PKG}.slam_baseline:SLAMBaseline",
        file=f"{_DIR}/slam_baseline.py",
        params=None,
        kind=BASELINE,
        status=BASELINE_STATUS,
        summary="robot-frame shift-and-composite map (non-learned)",
        thesis=None,
        source="main, class SLAMBaseline",
        runnable=True,
        trainable=False,
    ),
)

MODELS: "MappingProxyType[str, ModelEntry]" = MappingProxyType({e.name: e for e in _ENTRIES})
"""All registry entries by name, in table order (read-only)."""

_V5_RENAMED = "'{old}' is now 'v5_archived'; the thesis model is 'v3'"
_V6_RENAMED = "'{old}' is now 'v6_archived' (the autoregressive model, not in the thesis)"
_RENAMED = {
    "v5": _V5_RENAMED,
    "heightmap_v5": _V5_RENAMED,
    "hmv5": _V5_RENAMED,
    "v6": _V6_RENAMED,
    "heightmap_v6": _V6_RENAMED,
    "hmv6": _V6_RENAMED,
}


def _require_runnable(entry: ModelEntry) -> None:
    if not entry.runnable:
        record = "archived record only: " if entry.status == ARCHIVED else ""
        raise ModelRegistryError(
            f"'{entry.name}' is not runnable: {record}{entry.summary}; "
            "it cannot run in this pipeline"
        )


def resolve_model(name: str) -> ModelEntry:
    """Return the registry entry for ``name`` (case-insensitive).

    Raises:
        ModelRegistryError: for a retired code-era name (the message names its replacement),
            for a record-only entry that cannot run, or for an unknown name (the message lists
            the valid names).
    """
    key = name.strip().lower()
    if key in _RENAMED:
        raise ModelRegistryError(_RENAMED[key].format(old=name))
    entry = MODELS.get(key)
    if entry is None:
        raise ModelRegistryError(f"unknown model '{name}'; valid names: {', '.join(MODELS)}")
    _require_runnable(entry)
    return entry


def resolve_trainable(name: str) -> ModelEntry:
    """Like :func:`resolve_model`, but also require that the model can be trained."""
    entry = resolve_model(name)
    if entry.trainable:
        return entry
    if entry.status == ARCHIVED:
        reason = f"archived models are frozen; train {ACTIVE_MODEL}"
    elif not entry.learned:
        reason = "it is a non-learned baseline; there is nothing to train"
    else:
        reason = "training support comes in a later step"
    trainable = ", ".join(e.name for e in MODELS.values() if e.trainable)
    raise ModelRegistryError(f"'{entry.name}' cannot be trained: {reason} (trainable: {trainable})")


def build_model(entry: ModelEntry | str, **kwargs) -> "torch.nn.Module":
    """Instantiate a learned model from its registry entry (or name).

    Keyword arguments the class does not accept are dropped, so a caller can pass e.g.
    ``gru_hidden`` / ``gru_layers`` for every model and only the recurrent ones use them.
    """
    if isinstance(entry, str):
        entry = resolve_model(entry)
    if not entry.learned:
        raise ModelRegistryError(
            f"'{entry.name}' is a non-learned baseline; construct {entry.class_path} directly"
        )
    cls = entry.load_class()
    accepted = inspect.signature(cls).parameters
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def format_table() -> str:
    """Return the registry as a plain-text table."""
    header = ("name", "status", "kind", "params", "thesis", "summary")
    rows = [
        (
            e.name,
            e.status,
            e.kind or "-",
            f"{e.params:,}" if e.params is not None else "-",
            e.thesis or "-",
            e.summary,
        )
        for e in MODELS.values()
    ]
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header) - 1)]

    def fmt(row):
        cells = [
            cell.rjust(w) if i == 3 else cell.ljust(w)
            for i, (cell, w) in enumerate(zip(row, widths))
        ]
        return "  ".join(cells + [row[-1]])

    lines = [fmt(header), fmt(tuple("-" * w for w in widths) + ("-" * len(header[-1]),))]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def print_models(file=None) -> None:
    """Print the registry table (default: stdout)."""
    print(format_table(), file=file or sys.stdout)


if __name__ == "__main__":
    print_models()
