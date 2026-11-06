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

import shutil
from pathlib import Path

from core.gates import GateController, WaitingForApproval
from core.interfaces import NeedsExternalJob, TrainedCommittee
from core.schemas import WorkflowConfig
from core.state import GenerationRecord, GenerationStage, WorkflowState
from core.xyz_io import concat_xyz, extract_by_indices, iter_raw_blocks, split_blocks, train_test_split_raw


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

    def _init_md_outputs(self) -> list[Path]:
        """Paper SI initial-data generation: one foundation-model MD per input xyz (each with its box).

        init_md_settings: inputs (list of xyz), model ("mace_mp:medium"), auto_run (run here; else the person
        runs the printed per-file commands, e.g. on a GPU node), plus MD settings of mlp.init_md
        (defaults: 400 K, 1 fs, 1e6 steps, 400 frames/input). Outputs: <root>/init_md/init_N.xyz (N = 0-based input
        index); the engine resumes/skips finished files. Gen-0 DFT then labels all of them (one structure_N each).
        """
        import json
        import subprocess
        import sys

        from mlp.init_md import DEFAULTS

        im = self.config.init_md_settings
        inputs = [Path(x) if Path(x).is_absolute() else self.root_dir / x for x in im["inputs"]]
        for x in inputs:
            if not x.exists():
                raise FileNotFoundError(f"init_md input not found: {x}")
        out_dir = self.root_dir / "init_md"
        out_dir.mkdir(parents=True, exist_ok=True)
        keys = set(DEFAULTS) | {"cell"}
        settings = {k: v for k, v in im.items() if k in keys}
        n_frames = int(settings.get("n_frames", DEFAULTS["n_frames"]))
        model = im.get("model", "mace_mp:medium")
        (out_dir / "init_md.json").write_text(json.dumps(settings, indent=2))
        outputs = [out_dir / f"init_{i}.xyz" for i in range(len(inputs))]

        def n_done(f: Path) -> int:
            return len(iter_raw_blocks(f)) if f.exists() and f.stat().st_size else 0

        todo = [(i, x, o) for i, (x, o) in enumerate(zip(inputs, outputs)) if n_done(o) < n_frames]
        if not todo:
            return outputs
        self.gates.check("init_md", "init_md", f"Approve initial-data MD ({len(inputs)} structures, model {model})")
        cmds = [
            f"cd {out_dir} && {sys.executable} -m mlp.init_md --model {model} --input {x} --output {o} "
            f"--settings init_md.json"
            for _, x, o in todo
        ]
        if im.get("auto_run"):
            for (i, x, o), cmd in zip(todo, cmds):
                log = out_dir / f"init_{i}.log"
                with open(log, "w") as fh:
                    proc = subprocess.run(
                        [sys.executable, "-m", "mlp.init_md", "--model", model, "--input", str(x),
                         "--output", str(o), "--settings", "init_md.json"],
                        cwd=out_dir, stdout=fh, stderr=subprocess.STDOUT,
                    )
                if proc.returncode != 0:
                    raise RuntimeError(f"init MD failed for {x} (exit {proc.returncode}); see {log}")
            if any(n_done(o) < n_frames for o in outputs):
                raise RuntimeError("init MD finished but outputs are incomplete")
            return outputs
        raise NeedsExternalJob(
            stage="init_md",
            instructions="Run the initial-data MD (foundation model), then re-run this generation:\n  "
            + "\n  ".join(cmds),
        )

    def _current_structures_path(self, gen_idx: int) -> Path | list[Path]:
        if gen_idx == 0:
            if self.config.init_md_settings:
                outs = self._init_md_outputs()
                return outs[0] if len(outs) == 1 else outs
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
        n_splits = self.config.n_dft_splits
        if gen_idx > 0 and self.config.dft_frames_per_split > 0:
            n_frames = len(iter_raw_blocks(structures))
            n_splits = -(-n_frames // self.config.dft_frames_per_split)  # ceil
        if gen_idx > 0 and n_splits > 1:
            # legacy: coord.xyz was split into N chunks for parallel DFT (N came from input())
            structures = split_blocks(structures, n_splits, gen_dir / "dft")
        elif isinstance(structures, list) and len(structures) == 1:
            structures = structures[0]
        labeled = self._dft_engine().label(structures, gen_dir / "dft")
        record.labeled_file = str(labeled)

        split_source = labeled
        if self.config.cumulative_training:
            # legacy: each generation retrains on everything labeled so far
            prior = [Path(self.state.generation(i).labeled_file) for i in range(gen_idx)]
            split_source = concat_xyz(prior + [Path(labeled)], gen_dir / "mlp" / "data" / "all_labeled.xyz")

        train_file, test_file = train_test_split_raw(
            split_source,
            test_fraction=self.config.mlp_settings.get("valid_fraction", 0.25),
            train_out=gen_dir / "mlp" / "data" / "train.xyz",
            test_out=gen_dir / "mlp" / "data" / "test.xyz",
            seed=gen_idx,
        )
        record.train_file, record.test_file = str(train_file), str(test_file)
        record.stage = GenerationStage.DFT_DONE

    def _run_mlp(self, record: GenerationRecord, gen_idx: int, gen_dir: Path) -> None:
        foundation = self.config.mlp_settings.get("foundation_models")
        if foundation:
            # MACE-MP foundation potential(s), used as-is (no fine-tuning): skip training.
            from mlp.foundation import resolve_model_spec

            specs = [f if f.startswith("mace_mp:") or "/" in f else f"mace_mp:{f}" for f in foundation]
            record.model_paths = [str(resolve_model_spec(sp).local_path()) for sp in specs]
            record.stage = GenerationStage.MLP_DONE
            return
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
        initial_trajectory = self._prepare_initial_path(committee, gen_idx, gen_dir)
        candidate_frames = self._tps_sampler().sample(
            committee=committee,
            initial_trajectory=initial_trajectory,
            workdir=gen_dir / "tps",
            n_trajectories=self.config.n_tps_trajectories,
        )
        record.candidate_frames_file = str(candidate_frames)
        record.stage = GenerationStage.TPS_DONE

    def _prepare_initial_path(self, committee: TrainedCommittee, gen_idx: int, gen_dir: Path) -> Path:
        """Initial TPS path for this generation.

        Without `smd_settings` this is the fixed `initial_trajectory`. With it,
        replicates the legacy steered-MD step: stage the SMD inputs in
        <gen>/tps/smd, wait (smd_run gate / NeedsExternalJob) for the person to run
        it with the generation's first committee model, then use its result.

        smd_settings keys:
          inputs:  {source_file: name_in_smd_dir}, e.g. {"smd.xyz": "coord.xyz", "smd.json": "smd.json",
                   "plumed.dat": "plumed.dat"}  (sources are relative to root_dir)
          result:  file (in the smd dir) that holds the path, default "tps.xyz"
          command: shell template, {model} is replaced by the first committee model
        """
        smd = self.config.smd_settings
        if not smd:
            return self.config.initial_trajectory

        smd_dir = gen_dir / "tps" / "smd"
        smd_dir.mkdir(parents=True, exist_ok=True)
        for src, dest in smd["inputs"].items():
            src_path = Path(src) if Path(src).is_absolute() else self.root_dir / src
            if not src_path.exists():
                raise FileNotFoundError(f"SMD input not found: {src_path}")
            shutil.copyfile(src_path, smd_dir / dest)

        self.gates.check("smd_run", f"gen{gen_idx}:smd_run", f"Approve steered MD for generation {gen_idx}")
        result = smd_dir / smd.get("result", "tps.xyz")
        if not result.exists() and smd.get("auto_run"):
            self._run_smd_auto(smd, smd_dir, committee, result)
        if not result.exists():
            command = smd.get("command", "mlp run_md --model_path {model} --md_params smd.json --plumed plumed.dat")
            command = command.replace("{model}", str(committee.model_paths[0]))
            raise NeedsExternalJob(
                stage="smd_run",
                instructions=(
                    f"Run steered MD, then place the resulting path at {result}:\n"
                    f"  cd {smd_dir} && {command}"
                ),
            )
        return result

    def _run_smd_auto(self, smd: dict, smd_dir: Path, committee: TrainedCommittee, result: Path) -> None:
        """Run the ASE+PLUMED SMD runner (mlp.run_md) and build `result` from its trajectory.

        smd_settings extras: auto_run (bool), device ("auto"|"cpu"|"cuda"), plumed ("plumed.dat" or null),
        md_params ("smd.json"), traj ("md_out.xyz"), stride (int, keep every Nth frame, default 1).
        Fails loud on a non-zero exit or an empty trajectory.
        """
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "mlp.run_md",
            "--model", str(committee.model_paths[0]),
            "--md_params", smd.get("md_params", "smd.json"),
            "--device", smd.get("device", "auto"),
        ]
        if smd.get("plumed", "plumed.dat"):
            cmd += ["--plumed", smd.get("plumed", "plumed.dat")]
        log = smd_dir / "run_md.log"
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, cwd=smd_dir, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"SMD run failed (exit {proc.returncode}); see {log}")
        traj = smd_dir / smd.get("traj", "md_out.xyz")
        if not traj.exists():
            raise FileNotFoundError(f"SMD produced no trajectory: {traj}")
        blocks = iter_raw_blocks(traj)
        if not blocks:
            raise RuntimeError(f"SMD trajectory is empty: {traj}")
        stride = int(smd.get("stride", 1))
        with open(result, "w") as out:
            for b in blocks[::stride]:
                out.write(b if b.endswith("\n") else b + "\n")

    def _run_selection(self, record: GenerationRecord, gen_idx: int, gen_dir: Path) -> None:
        selected_path = gen_dir / f"gen_{gen_idx}_selected_frames.xyz"
        if self.config.selection_mode == "sampler":
            # legacy: the TPS stage already applied its threshold/weighted selection
            shutil.copyfile(record.candidate_frames_file, selected_path)
        elif self.config.selection_mode == "topk":
            committee = TrainedCommittee(model_paths=[Path(p) for p in record.model_paths])
            uncertainties = self._mlp_backend().predict_uncertainty(
                committee, Path(record.candidate_frames_file)
            )
            ranked = sorted(uncertainties, key=lambda u: u.force_variance, reverse=True)
            top_k = [u.frame_index for u in ranked[: self.config.top_k_frames_per_generation]]
            extract_by_indices(Path(record.candidate_frames_file), top_k, selected_path)
        else:
            raise ValueError(f"Unknown selection_mode: {self.config.selection_mode!r}")
        record.selected_frames_file = str(selected_path)
        record.stage = GenerationStage.COMPLETE

    def _advance_generation(self) -> None:
        gate_id = f"gen{self.state.current_generation}:continue_next_gen"
        self.gates.check(
            "continue_next_gen", gate_id, f"Continue to generation {self.state.current_generation + 1}?"
        )
        self.state.current_generation += 1
