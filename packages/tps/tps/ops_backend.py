"""Adapter from the legacy OpenPathSampling implementation (legacy_ops.py)
to the new TPSSampler interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from core.interfaces import TPSSampler, TrainedCommittee

from tps.legacy_ops import Sampler, TPS_setup, extend_and_select_frames


class OPSTPSSampler(TPSSampler):
    def __init__(self, config_template: dict):
        """config_template mirrors the legacy tps.json - everything except
        `mace_model_path`, which this adapter fills in per generation from
        the trained committee.
        """
        self.config_template = config_template

    def sample(
        self,
        committee: TrainedCommittee,
        initial_trajectory: Path,
        workdir: Path,
        n_trajectories: int,
    ) -> Path:
        # Resolve everything to absolute paths before the os.chdir(workdir)
        # below - legacy_ops.py reads several config paths (init_path,
        # top_file) relative to the process cwd at call time, so anything
        # still relative here would silently break once we've chdir'd,
        # exactly as happened with a relative WorkflowConfig.initial_trajectory
        # while building this adapter (OSError: No such file: tps.xyz).
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        initial_trajectory = Path(initial_trajectory).resolve()

        config = dict(self.config_template)
        if "top_file" in config:
            config["top_file"] = str(Path(config["top_file"]).resolve())
        config["mace_model_path"] = [str(Path(p).resolve()) for p in committee.model_paths]
        config["init_path"] = str(initial_trajectory)
        config.setdefault("tot_mc_steps", n_trajectories)

        config_path = workdir / "tps.json"
        config_path.write_text(json.dumps(config, indent=2))

        traj_dir = workdir / "traj"
        traj_dir.mkdir(parents=True, exist_ok=True)

        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            setup = TPS_setup(config_path.name)
            sampler = Sampler(
                setup_class=setup,
                directory=".",
                output_path=".",
                final_traj_storage_path=str(traj_dir),
            )
            sampler.run()
            return extend_and_select_frames(
                sampler=sampler,
                mace_model_paths=config["mace_model_path"],
                workdir=workdir,
            )
        finally:
            os.chdir(cwd)
