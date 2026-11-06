"""Contracts shared by the dft/, mlp/, and tps/ packages.

These are the ONLY things workflow.py is allowed to depend on. Whoever owns
the DFT, MLP, or TPS/AIMMD package can change everything about their
implementation as long as they keep satisfying the ABC here - the workflow
engine, the CLI, and (eventually) a GUI never import mace / cp2k /
openpathsampling directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


class NeedsExternalJob(RuntimeError):
    """Raised by a backend when it has prepared work that must be run outside
    this process (e.g. submit a CP2K job to a cluster, run an MD engine by
    hand) before the workflow can continue.

    The workflow engine turns this into a manual gate rather than crashing,
    so a person (or a script) can complete the external step and then resume
    the same generation.
    """

    def __init__(self, stage: str, instructions: str):
        super().__init__(instructions)
        self.stage = stage
        self.instructions = instructions


@dataclass
class TrainedCommittee:
    """Output of MLPBackend.train_committee - a set of model files that were
    trained on different seeds/splits, used both for MD forces and for
    uncertainty estimation (spread across committee members).
    """

    model_paths: list[Path] = field(default_factory=list)

    def __post_init__(self):
        self.model_paths = [Path(p) for p in self.model_paths]

    @property
    def size(self) -> int:
        return len(self.model_paths)


@dataclass
class FrameUncertainty:
    """Per-frame committee disagreement, used to select frames for labeling."""

    frame_index: int
    energy_variance: float
    force_variance: float


class DFTEngine(ABC):
    """Reference (ab initio) labeling backend, e.g. CP2K."""

    @abstractmethod
    def label(self, structures: Path | Sequence[Path], workdir: Path) -> Path:
        """Compute reference energy/forces for every frame in `structures`
        (an xyz/extxyz file, or a list of them - one parallel job/structure_N each) and return the path to a combined extxyz file
        with `energy`/`forces` arrays set.

        May raise NeedsExternalJob if labeling requires a step outside this
        process (e.g. a cluster job the user must submit and wait for).
        """


class MLPBackend(ABC):
    """Machine-learned potential training/inference backend, e.g. MACE."""

    @abstractmethod
    def train_committee(
        self,
        train_file: Path,
        test_file: Path,
        workdir: Path,
        seeds: Sequence[int],
    ) -> TrainedCommittee:
        """Train one model per seed on the same data and return their paths."""

    @abstractmethod
    def predict_uncertainty(
        self, committee: TrainedCommittee, structures: Path
    ) -> list[FrameUncertainty]:
        """Evaluate every committee member on every frame in `structures` and
        return per-frame energy/force variance across the committee.
        """


class TPSSampler(ABC):
    """Rare-event trajectory sampler, e.g. OpenPathSampling/AIMMD driven by an
    MLPBackend-trained committee as the MD force field.
    """

    @abstractmethod
    def sample(
        self,
        committee: TrainedCommittee,
        initial_trajectory: Path,
        workdir: Path,
        n_trajectories: int,
    ) -> Path:
        """Run TPS shooting moves seeded from `initial_trajectory` and return
        the path to an xyz file containing the pool of candidate frames
        (accepted + extended trajectories) for the next selection step.

        May raise NeedsExternalJob if sampling requires a step outside this
        process.
        """
