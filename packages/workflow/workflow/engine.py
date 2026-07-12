"""The generational active-learning loop.

Replaces the legacy mlp_path/workflow.py. Differences that matter:

  - No blocking input()/exit(0). Every point that used to ask a human a
    question is now a named gate (core.gates.GateController); step() raises
    WaitingForApproval instead of hanging, so a CLI, a cron job, or a GUI
    can all drive this the same way.
  - Explicit, inspectable state (core.state.WorkflowState) instead of
    DONE/PAUSE sentinel files scattered across the directory tree.
  - Zero direct imports of mace/cp2k/openpathsampling - backends are
    resolved through dft.create_dft_engine / mlp.create_mlp_backend /
    tps.create_tps_sampler based on WorkflowConfig, so this file works
    unmodified whether you're running the mock smoke test or a real
    CP2K+MACE+OpenPathSampling generation.
"""

from __future__ import annotations

from pathlib import Path

from core.gates import GateController, WaitingForApproval
from core.interfaces import NeedsExternalJob, TrainedCommittee
from core.schemas import WorkflowConfig
from core.state import GenerationRecord, GenerationStage, WorkflowState
from core.xyz_io import extract_by_indices, train_test_split_raw


class WorkflowEngine:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.state = WorkflowState.load_or_create(self.root_dir)
        self.gates = GateController(config.gates, self.root_dir)
        self._dft = None
        self._mlp = None
        self._tps = None

    # -- lazy backend resolution: only the package for the configured backend
    # needs to be installed in the environment actually running this. --
    def _dft_engine(self):
        if self._dft is None:
            from dft import create_dft_engine

            self._dft = create_dft_engine(self.config.dft_backend, self.config.dft_settings)
        return self._dft

    def _mlp_backend(self):
        if self._mlp is None:
            from mlp import create_mlp_backend

            self._mlp = create_mlp_backend(self.config.mlp_backend, self.config.mlp_settings)
        return self._mlp

    def _tps_sampler(self):
        if self._tps is None:
            from tps import create_tps_sampler

            self._tps = create_tps_sampler(self.config.tps_backend, self.config.tps_settings)
        return self._tps

    # -- generation bookkeeping --
    def _gen_dir(self, gen_idx: int) -> Path:
        d = self.root_dir / str(gen_idx)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _current_structures_path(self, gen_idx: int) -> Path:
        if gen_idx == 0:
            return self.config.initial_structures
        prev = self.state.generation(gen_idx - 1)
        if not prev.selected_frames_file:
            raise RuntimeError(f"Generation {gen_idx - 1} has no selected_frames_file yet.")
        return Path(prev.selected_frames_file)

    def is_done(self) -> bool:
        return self.state.current_generation >= self.config.max_generations

    def step(self) -> GenerationRecord:
        """Advance exactly one stage of the current generation.

        Raises WaitingForApproval if the next stage is blocked on a gate
        (either a config-declared manual gate, or a NeedsExternalJob from a
        real backend waiting on a human to run something outside Python).
        """
        if self.is_done():
            raise RuntimeError("Workflow already reached max_generations.")

        gen_idx = self.state.current_generation
        record = self.state.generation(gen_idx)
        gen_dir = self._gen_dir(gen_idx)

        try:
            if record.stage == GenerationStage.PENDING:
                self._run_dft(record, gen_idx, gen_dir)
            elif record.stage == GenerationStage.DFT_DONE:
                self._run_mlp(record, gen_idx, gen_dir)
            elif record.stage == GenerationStage.MLP_DONE:
                self._run_tps(record, gen_idx, gen_dir)
            elif record.stage == GenerationStage.TPS_DONE:
                self._run_selection(record, gen_idx, gen_dir)
            elif record.stage == GenerationStage.COMPLETE:
                self._advance_generation()
            else:
                raise RuntimeError(f"Unexpected stage: {record.stage}")
            record.blocked = False
            record.pending_gate = None
            record.pending_instructions = None
        except WaitingForApproval as e:
            record.blocked = True
            record.pending_gate = e.gate_id
            record.pending_instructions = e.instructions
            self.state.save()
            raise
        except NeedsExternalJob as e:
            gate_id = f"gen{gen_idx}:{e.stage}"
            record.blocked = True
            record.pending_gate = gate_id
            record.pending_instructions = e.instructions
            self.state.save()
            raise WaitingForApproval(e.stage, gate_id, e.instructions) from e
        except Exception as e:
            record.stage = GenerationStage.FAILED
            record.error = str(e)
            self.state.save()
            raise

        self.state.save()
        return record

    def run_to_completion(self, max_steps: int = 10_000) -> None:
        for _ in range(max_steps):
            if self.is_done():
                return
            self.step()

    # -- stages --
    def _run_dft(self, record: GenerationRecord, gen_idx: int, gen_dir: Path) -> None:
        gate_id = f"gen{gen_idx}:dft_submit"
        self.gates.check("dft_submit", gate_id, f"Approve DFT labeling for generation {gen_idx}")

        structures = self._current_structures_path(gen_idx)
        labeled = self._dft_engine().label(structures, gen_dir / "dft")
        record.labeled_file = str(labeled)

        train_file, test_file = train_test_split_raw(
            labeled,
            test_fraction=self.config.mlp_settings.get("valid_fraction", 0.25),
            train_out=gen_dir / "mlp" / "data" / "train.xyz",
            test_out=gen_dir / "mlp" / "data" / "test.xyz",
            seed=gen_idx,
        )
        record.train_file, record.test_file = str(train_file), str(test_file)
        record.stage = GenerationStage.DFT_DONE

    def _run_mlp(self, record: GenerationRecord, gen_idx: int, gen_dir: Path) -> None:
        committee = self._mlp_backend().train_committee(
            train_file=Path(record.train_file),
            test_file=Path(record.test_file),
            workdir=gen_dir / "mlp" / "runs",
            seeds=self.config.committee_seeds,
        )
        record.model_paths = [str(p) for p in committee.model_paths]
        record.stage = GenerationStage.MLP_DONE

    def _run_tps(self, record: GenerationRecord, gen_idx: int, gen_dir: Path) -> None:
        gate_id = f"gen{gen_idx}:tps_run"
        self.gates.check("tps_run", gate_id, f"Approve TPS sampling for generation {gen_idx}")

        committee = TrainedCommittee(model_paths=[Path(p) for p in record.model_paths])
        candidate_frames = self._tps_sampler().sample(
            committee=committee,
            initial_trajectory=self.config.initial_trajectory,
            workdir=gen_dir / "tps",
            n_trajectories=self.config.n_tps_trajectories,
        )
        record.candidate_frames_file = str(candidate_frames)
        record.stage = GenerationStage.TPS_DONE

    def _run_selection(self, record: GenerationRecord, gen_idx: int, gen_dir: Path) -> None:
        committee = TrainedCommittee(model_paths=[Path(p) for p in record.model_paths])
        uncertainties = self._mlp_backend().predict_uncertainty(
            committee, Path(record.candidate_frames_file)
        )
        ranked = sorted(uncertainties, key=lambda u: u.force_variance, reverse=True)
        top_k = [u.frame_index for u in ranked[: self.config.top_k_frames_per_generation]]

        selected_path = gen_dir / f"gen_{gen_idx}_selected_frames.xyz"
        extract_by_indices(Path(record.candidate_frames_file), top_k, selected_path)
        record.selected_frames_file = str(selected_path)
        record.stage = GenerationStage.COMPLETE

    def _advance_generation(self) -> None:
        gate_id = f"gen{self.state.current_generation}:continue_next_gen"
        self.gates.check(
            "continue_next_gen", gate_id, f"Continue to generation {self.state.current_generation + 1}?"
        )
        self.state.current_generation += 1
