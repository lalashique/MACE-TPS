"""Instant, zero-dependency stand-in for real MACE training.

Doesn't train anything. "Training" just records one small, seed-dependent
bias per committee member; "prediction" evaluates the same toy harmonic
potential the mock DFT backend uses (duplicated here on purpose - mlp/ and
dft/ are owned by different people and shouldn't import each other) plus
that bias, so committee members disagree by a bounded, reproducible amount.
That disagreement is what the frame-selection step in the workflow engine
actually exercises.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from core.interfaces import FrameUncertainty, MLPBackend, TrainedCommittee
from core.xyz_io import Frame, read_frames

_K = 0.05  # must match dft/mock_backend.py's toy potential for a meaningful test


def _biased_energy_and_forces(frame: Frame, bias: float) -> tuple[float, list[tuple[float, float, float]]]:
    energy = 0.0
    forces = []
    for x, y, z in frame.positions:
        k = _K * (1.0 + bias)
        energy += 0.5 * k * (x * x + y * y + z * z)
        forces.append((-k * x, -k * y, -k * z))
    return energy, forces


class MockMLPBackend(MLPBackend):
    def train_committee(
        self, train_file: Path, test_file: Path, workdir: Path, seeds
    ) -> TrainedCommittee:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        model_paths = []
        for seed in seeds:
            bias = (random.Random(seed).random() - 0.5) * 0.2  # +/-10% spring constant
            model_path = workdir / f"mock_model_seed{seed}.json"
            model_path.write_text(json.dumps({"seed": seed, "bias": bias}))
            model_paths.append(model_path)
        return TrainedCommittee(model_paths=model_paths)

    def predict_uncertainty(
        self, committee: TrainedCommittee, structures: Path
    ) -> list[FrameUncertainty]:
        frames = read_frames(structures)
        members = [json.loads(Path(p).read_text()) for p in committee.model_paths]

        results = []
        for idx, frame in enumerate(frames):
            energies = []
            force_components = []
            for member in members:
                energy, forces = _biased_energy_and_forces(frame, member["bias"])
                energies.append(energy)
                force_components.extend(v for triple in forces for v in triple)

            n = len(energies)
            mean_e = sum(energies) / n
            energy_var = sum((e - mean_e) ** 2 for e in energies) / n

            n_f = len(force_components)
            mean_f = sum(force_components) / n_f if n_f else 0.0
            force_var = (
                sum((f - mean_f) ** 2 for f in force_components) / n_f if n_f else 0.0
            )

            results.append(
                FrameUncertainty(frame_index=idx, energy_variance=energy_var, force_variance=force_var)
            )
        return results
