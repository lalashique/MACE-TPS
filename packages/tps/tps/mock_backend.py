"""Instant, zero-dependency stand-in for real OpenPathSampling TPS.

IMPORTANT: this is NOT a scientific TPS implementation. It does not run
two-way shooting, does not check state definitions, does not use the MLP
committee's forces. It exists purely to exercise the generational workflow
(file formats, gates, frame-selection plumbing) quickly and deterministically
without OpenPathSampling/OpenMM/MACE installed.

What it actually does: pick a random frame from the initial trajectory as a
"shooting point" for each of `n_trajectories`, jitter its positions a little
(standing in for "a short MD integration diverged from here"), and hand back
the pool (original trajectory + jittered frames) as candidates. Real
uncertainty-driven selection happens downstream in the workflow engine via
MLPBackend.predict_uncertainty - this mock only needs to produce *some*
variety of frames for that step to have something to chew on.
"""

from __future__ import annotations

import random
from pathlib import Path

from core.interfaces import TPSSampler, TrainedCommittee
from core.xyz_io import Frame, read_frames, write_frames


class MockTPSSampler(TPSSampler):
    def __init__(self, jitter: float = 0.05, seed: int = 0):
        self.jitter = jitter
        self.seed = seed

    def sample(
        self,
        committee: TrainedCommittee,
        initial_trajectory: Path,
        workdir: Path,
        n_trajectories: int,
    ) -> Path:
        frames = read_frames(initial_trajectory)
        if not frames:
            raise ValueError(f"No frames found in initial trajectory: {initial_trajectory}")

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(self.seed)
        candidates = []
        for traj_idx in range(n_trajectories):
            shooting_frame = frames[rng.randrange(len(frames))]
            jittered_positions = [
                (
                    x + rng.uniform(-self.jitter, self.jitter),
                    y + rng.uniform(-self.jitter, self.jitter),
                    z + rng.uniform(-self.jitter, self.jitter),
                )
                for x, y, z in shooting_frame.positions
            ]
            candidates.append(
                Frame(
                    symbols=shooting_frame.symbols,
                    positions=jittered_positions,
                    info={"trajectory": traj_idx, "source": "mock_tps"},
                )
            )

        out_path = workdir / "candidate_frames.xyz"
        write_frames(out_path, frames + candidates)
        return out_path
