"""Adapter from the legacy MACE_MLP class (legacy_mace.py) to the new
MLPBackend interface.
"""

from __future__ import annotations

from pathlib import Path

from core.interfaces import FrameUncertainty, MLPBackend, TrainedCommittee

from mlp.legacy_mace import MACE_MLP


class MACEBackend(MLPBackend):
    def __init__(self, e0s: dict, config_file: str | None = None):
        self.e0s = e0s
        self.config_file = config_file

    def train_committee(self, train_file: Path, test_file: Path, workdir: Path, seeds) -> TrainedCommittee:
        mlp = MACE_MLP(config_file=self.config_file, E0s=self.e0s)
        model_paths = mlp.train(
            seeds=list(seeds),
            output_dir=str(workdir),
            train_file=str(train_file),
            test_file=str(test_file),
            absolute_paths=True,
            prepare_only=False,
        )
        return TrainedCommittee(model_paths=model_paths)

    def predict_uncertainty(self, committee: TrainedCommittee, structures: Path) -> list[FrameUncertainty]:
        from ase.io import read

        mlp = MACE_MLP(config_file=self.config_file, E0s=self.e0s)
        committee_calcs = mlp.load_committee(committee.model_paths)

        frames = read(structures, index=":")
        results = []
        for idx, atoms in enumerate(frames):
            atoms = MACE_MLP.evaluate_committee(atoms, committee_calcs)
            results.append(
                FrameUncertainty(
                    frame_index=idx,
                    energy_variance=float(atoms.info["energy_variance"]),
                    force_variance=float(atoms.info["force_variance"]),
                )
            )
        return results
