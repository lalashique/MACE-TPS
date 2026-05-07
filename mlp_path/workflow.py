"""
workflow.py — Active learning workflow for MLP-driven TPS simulations.

Orchestrates the iterative loop:
    DFT → MLP training → TPS sampling → frame selection → repeat

Each generation lives in its own subdirectory:
    <workdir>/
        0/
            dft/         ← CP2K inputs + outputs
            mlp/         ← MACE training runs
            tps/         ← OPS storage + sampled trajectories
            DONE         ← sentinel: this generation is complete
        1/
            ...
        init/            ← initial coord files (updated each generation)
        workflow_state.json

Usage:
    python workflow.py --config workflow.json
    python workflow.py --config workflow.json --gen 2          # resume at generation 2
    python workflow.py --config workflow.json --stage mlp      # run only one stage
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess as sp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StageStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    SKIPPED  = "skipped"
    FAILED   = "failed"


class StageName(str, Enum):
    DFT          = "dft"
    MLP          = "mlp"
    TPS          = "tps"
    FRAME_SELECT = "frame_select"


# ---------------------------------------------------------------------------
# State dataclasses (serialised to JSON)
# ---------------------------------------------------------------------------

@dataclass
class StageState:
    status: StageStatus = StageStatus.PENDING
    notes: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status.value, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "StageState":
        return cls(
            status=StageStatus(d.get("status", "pending")),
            notes=d.get("notes", ""),
        )


@dataclass
class GenerationState:
    generation: int
    stages: dict[str, StageState] = field(
        default_factory=lambda: {s.value: StageState() for s in StageName}
    )
    done: bool = False

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "done": self.done,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationState":
        obj = cls(generation=d["generation"], done=d.get("done", False))
        obj.stages = {k: StageState.from_dict(v) for k, v in d.get("stages", {}).items()}
        return obj

    def stage(self, name: StageName) -> StageState:
        return self.stages.setdefault(name.value, StageState())

    def mark(self, name: StageName, status: StageStatus, notes: str = "") -> None:
        s = self.stages.setdefault(name.value, StageState())
        s.status = status
        s.notes = notes


@dataclass
class WorkflowState:
    current_generation: int = 0
    generations: dict[int, GenerationState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "current_generation": self.current_generation,
            "generations": {str(k): v.to_dict() for k, v in self.generations.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowState":
        obj = cls(current_generation=d.get("current_generation", 0))
        obj.generations = {
            int(k): GenerationState.from_dict(v)
            for k, v in d.get("generations", {}).items()
        }
        return obj

    def get_or_create_gen(self, gen: int) -> GenerationState:
        if gen not in self.generations:
            self.generations[gen] = GenerationState(generation=gen)
        return self.generations[gen]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorkflowConfig:
    """
    Mirrors the JSON config file. All fields have sensible defaults.

    Minimal example JSON
    --------------------
    {
        "workdir": ".",
        "max_generations": 5,
        "structures": ["init/coord.xyz"],
        "top_file": "top.pdb",
        "dft": {
            "engine": "cp2k",
            "input_files": ["cp2k.inp", "force_eval.inp"],
            "settings": { "command": "mpirun -np 20 cp2k.popt" },
            "workdir": "single_reftraj",
            "reftraj": true,
            "run_mode": "prepare"
        },
        "mlp": {
            "config_file": "mlp.yaml",
            "E0s": {"1": -13.5, "6": -145.7, "8": -428.9},
            "seeds": [123, 234, 345],
            "test_fraction": 0.25
        },
        "tps": {
            "config_file": "tps.json",
            "run_command": "cd {tps_dir} && python -m tps --start_type new"
        },
        "auto_confirm": {
            "remove_pause": false,
            "continue_generation": false,
            "run_smd": false
        }
    }
    """

    workdir: Path
    max_generations: int
    structures: list[str]
    top_file: str
    dft: dict
    mlp: dict
    tps: dict
    auto_confirm: dict
    log_file: str = "workflow_debug.log"
    state_file: str = "workflow_state.json"

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkflowConfig":
        with open(path) as f:
            raw = json.load(f)

        workdir = Path(raw.get("workdir", ".")).resolve()
        auto_confirm_defaults = {
            "remove_pause": False,
            "continue_generation": False,
            "run_smd": False,
        }
        auto_confirm = {**auto_confirm_defaults, **raw.get("auto_confirm", {})}

        return cls(
            workdir=workdir,
            max_generations=raw.get("max_generations", 5),
            structures=raw.get("structures", []),
            top_file=raw.get("top_file", "top.pdb"),
            dft=raw.get("dft", {}),
            mlp=raw.get("mlp", {}),
            tps=raw.get("tps", {}),
            auto_confirm=auto_confirm,
            log_file=raw.get("log_file", "workflow_debug.log"),
            state_file=raw.get("state_file", "workflow_state.json"),
        )

    def confirm(self, key: str, prompt: str) -> bool:
        """Return the auto-confirm value if set; otherwise ask the user."""
        if key in self.auto_confirm:
            answer = bool(self.auto_confirm[key])
            logger.debug(f"[auto_confirm] {key} = {answer}  (prompt: '{prompt}')")
            return answer
        return input(f"{prompt} (y/N): ").strip().lower() == "y"


# ---------------------------------------------------------------------------
# Abstract stage base
# ---------------------------------------------------------------------------

class Stage(ABC):
    """Base class for a single workflow stage within one generation."""

    name: StageName  # must be set by each subclass

    def __init__(self, gen: int, config: WorkflowConfig, state: GenerationState):
        self.gen = gen
        self.config = config
        self.gen_state = state
        self.gen_dir = config.workdir / str(gen)
        self.init_dir = config.workdir / "init"

    @property
    def stage_state(self) -> StageState:
        return self.gen_state.stage(self.name)

    def is_done(self) -> bool:
        return self.stage_state.status == StageStatus.DONE

    def run(self) -> bool:
        """
        Entry point called by the generation runner.
        Skips if already done. Returns True on success, False if blocked.
        """
        if self.is_done():
            logger.info(f"[Gen {self.gen}][{self.name.value}] Already done — skipping.")
            return True

        logger.info(f"[Gen {self.gen}][{self.name.value}] Starting.")
        self.gen_state.mark(self.name, StageStatus.RUNNING)

        try:
            result = self._run()
        except Exception as exc:
            logger.error(f"[Gen {self.gen}][{self.name.value}] FAILED: {exc}")
            self.gen_state.mark(self.name, StageStatus.FAILED, str(exc))
            raise

        if result:
            self.gen_state.mark(self.name, StageStatus.DONE)
            logger.info(f"[Gen {self.gen}][{self.name.value}] Done.")
        else:
            self.gen_state.mark(
                self.name, StageStatus.PENDING,
                "Blocked — external action required."
            )
            logger.info(f"[Gen {self.gen}][{self.name.value}] Blocked; manual step needed.")

        return result

    @abstractmethod
    def _run(self) -> bool:
        """Stage logic. Return True if complete, False if blocked on an external step."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"Copied {src} → {dst}")

    @staticmethod
    def _touch_done(directory: Path) -> None:
        (directory / "DONE").touch()

    @staticmethod
    def run_cmd(cmd: str, capture: bool = False) -> str | int:
        if capture:
            try:
                return sp.check_output(cmd, shell=True).decode().strip()
            except sp.CalledProcessError:
                return ""
        return sp.call(cmd, shell=True)


# ---------------------------------------------------------------------------
# DFT Stage
# ---------------------------------------------------------------------------

class DFTStage(Stage):
    name = StageName.DFT

    def _run(self) -> bool:
        from dft import DFTFactory

        cfg = self.config.dft
        dft_dir = self.gen_dir / "dft"
        dft_dir.mkdir(parents=True, exist_ok=True)

        input_files = [str((self.config.workdir / f).resolve()) for f in cfg.get("input_files", [])]

        dft_engine = DFTFactory.create(
            engine=cfg.get("engine", "cp2k"),
            settings=cfg.get("settings", {}),
            input_files=input_files,
            reftraj=cfg.get("reftraj", True),
            run_mode=cfg.get("run_mode", "prepare"),
            workdir=str(dft_dir / cfg.get("workdir", "single_reftraj")),
        )

        # Resolve structures for this generation
        if self.gen == 0:
            structures = [str((self.config.workdir / s).resolve()) for s in self.config.structures]
        else:
            coord = self.init_dir / "coord.xyz"
            if not coord.exists():
                raise FileNotFoundError(f"coord.xyz not found: {coord}")
            structures = [str(coord.resolve())]

        state_file = Path(dft_engine.workdir) / "state.json"

        # --- Prepare inputs if not done yet ---
        if not state_file.exists():
            logger.info(f"[DFT] Preparing inputs in {dft_engine.workdir}")
            dft_engine.run(structures)
            cmd = cfg.get("settings", {}).get("command", "cp2k")
            inp = input_files[0] if input_files else "cp2k.inp"
            logger.info(f"[DFT] Run CP2K in each structure folder:\n"
                        f"  cd <structure_folder> && {cmd} -i {inp} -o cp2k.out")
            return False

        # --- Check completion of each structure ---
        dft_engine.state = dft_engine._load_state()
        all_complete = True

        for key in dft_engine.state:
            if "structure_" not in key:
                continue
            struct_dir = Path(dft_engine.workdir) / key
            out_xyz   = struct_dir / "cp2k-pos-1.xyz"
            coord_xyz = struct_dir / "coord.xyz"
            pause     = struct_dir / "PAUSE"

            if not out_xyz.exists():
                logger.warning(f"[DFT] Output missing: {out_xyz}")
                all_complete = False
                continue

            out_lines   = self.run_cmd(f"wc -l {out_xyz}", capture=True).split()[0]
            coord_lines = self.run_cmd(f"wc -l {coord_xyz}", capture=True).split()[0]

            if out_lines == coord_lines:
                if pause.exists() and self.config.confirm(
                    "remove_pause", f"DFT complete for {key}. Remove PAUSE file?"
                ):
                    pause.unlink()
                    logger.info(f"[DFT] Removed PAUSE for {key}.")
            else:
                logger.warning(f"[DFT] {key} not complete ({out_lines}/{coord_lines} lines).")
                all_complete = False

        if not all_complete:
            return False

        # --- Parse outputs and combine ---
        logger.info("[DFT] All calculations done — parsing outputs.")
        dft_engine._parse_outputs()

        dft_workdir = Path(dft_engine.workdir)
        combined = dft_workdir / "output" / "combined_extdata.xyz"
        combined.parent.mkdir(parents=True, exist_ok=True)

        with open(combined, "wb") as dst:
            for src in sorted(dft_workdir.glob("structure_*/extdata.xyz")):
                with open(src, "rb") as s:
                    shutil.copyfileobj(s, dst)

        logger.info(f"[DFT] Combined extxyz → {combined}")
        self._touch_done(dft_dir)
        return True


# ---------------------------------------------------------------------------
# MLP Stage
# ---------------------------------------------------------------------------

class MLPStage(Stage):
    name = StageName.MLP

    def _run(self) -> bool:
        import yaml
        from mlp import create_mlp
        import util.traj_util as tu

        cfg = self.config.mlp
        mlp_dir = self.gen_dir / "mlp"
        mlp_dir.mkdir(parents=True, exist_ok=True)

        # Locate DFT combined output
        dft_combined = (
            self.gen_dir / "dft" /
            self.config.dft.get("workdir", "single_reftraj") /
            "output" / "combined_extdata.xyz"
        )
        if not dft_combined.exists():
            raise FileNotFoundError(f"[MLP] DFT output not found: {dft_combined}")

        # Copy + split data (idempotent)
        data_dir = mlp_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / "combined_extdata.xyz"
        if not dest.exists():
            self._copy(dft_combined, dest)
            logger.info("[MLP] Splitting data into train/test sets.")
            tu.train_test_split(str(dest), test_fraction=cfg.get("test_fraction", 0.25))

        train_file = data_dir / "train.xyz"
        test_file  = data_dir / "test.xyz"

        # Write train YAML config (idempotent)
        train_yaml = mlp_dir / "train.yaml"
        if not train_yaml.exists():
            src_yaml = self.config.workdir / cfg.get("config_file", "mlp.yaml")
            if src_yaml.exists():
                self._copy(src_yaml, train_yaml)
            else:
                with open(train_yaml, "w") as f:
                    yaml.safe_dump(cfg, f)

        # Validate E0s
        E0s = cfg.get("E0s")
        if not E0s:
            raise ValueError("[MLP] E0s must be provided in the mlp config section.")
        E0s = {int(k): float(v) for k, v in E0s.items()}

        seeds = cfg.get("seeds", [123, 234, 345])
        orig_dir = Path.cwd()
        os.chdir(mlp_dir)
        try:
            mlp = create_mlp("mace", str(train_yaml), E0s=E0s)
            mlp.train(
                train_file=str(train_file.resolve()),
                test_file=str(test_file.resolve()),
                seeds=seeds,
                prepare_only=True,
                absolute_paths=True,
            )
        finally:
            os.chdir(orig_dir)

        logger.info("[MLP] Training configs written. Submit jobs and rerun when complete.")

        # Check for completion (models exist, no PAUSE files blocking)
        runs_dir = mlp_dir / "runs"
        pause_files = list(runs_dir.glob("committee_*/PAUSE"))
        all_trained = True

        for pause in pause_files:
            committee_dir = pause.parent
            mace_dir = committee_dir / "MACE_models"
            model = next(mace_dir.glob("*.model"), None) if mace_dir.exists() else None
            if model:
                if self.config.confirm(
                    "remove_pause",
                    f"Model found at {model.name}. Remove PAUSE file at {pause}?"
                ):
                    pause.unlink()
                    logger.info(f"[MLP] Removed PAUSE: {pause}")
            else:
                logger.warning(f"[MLP] No model in {committee_dir} — training incomplete.")
                all_trained = False

        if not all_trained:
            return False

        model_files = (
            list(runs_dir.glob("committee_*/MACE_models/*.model")) +
            list(runs_dir.glob("committee_*/*.model"))
        )
        if not model_files:
            logger.warning("[MLP] No trained models found yet.")
            return False

        logger.info(f"[MLP] {len(model_files)} model(s) ready.")
        self._touch_done(mlp_dir)
        return True


# ---------------------------------------------------------------------------
# TPS Stage
# ---------------------------------------------------------------------------

class TPSStage(Stage):
    name = StageName.TPS

    def _run(self) -> bool:
        cfg = self.config.tps
        tps_dir = self.gen_dir / "tps"
        tps_dir.mkdir(parents=True, exist_ok=True)

        # --- Gather model paths ---
        mlp_runs = self.gen_dir / "mlp" / "runs"
        models_dir = tps_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_paths: list[str] = []

        for committee in sorted(mlp_runs.glob("committee_*")):
            mace_dir = committee / "MACE_models"
            src = next(
                (list(mace_dir.glob("*.model")) if mace_dir.exists() else []) +
                list(committee.glob("*.model")),
                None,
            )
            if src:
                dst = models_dir / f"{committee.name}_{src.name}"
                if not dst.exists():
                    self._copy(src, dst)
                model_paths.append(str(dst))

        if not model_paths:
            raise FileNotFoundError(f"[TPS] No trained models found in {mlp_runs}")
        logger.info(f"[TPS] {len(model_paths)} model(s) available.")

        # --- SMD: generate initial path if needed ---
        tps_xyz = self.config.workdir / "tps.xyz"
        if not tps_xyz.exists():
            smd_dir = tps_dir / "smd"
            smd_dir.mkdir(parents=True, exist_ok=True)

            required = {"smd.json": "smd.json", "smd.xyz": "coord.xyz", "plumed.dat": "plumed.dat"}
            for fname, dest_name in required.items():
                src = self.config.workdir / fname
                dst = smd_dir / dest_name
                if not src.exists():
                    raise FileNotFoundError(f"[SMD] Required file not found: {src}")
                if not dst.exists():
                    self._copy(src, dst)

            smd_cmd = (
                f"cd {smd_dir} && mlp run_md "
                f"--model_path {model_paths[0]} "
                f"--md_params smd.json "
                f"--plumed plumed.dat"
            )
            logger.info(f"[SMD] Run command:\n  {smd_cmd}")

            if self.config.confirm("run_smd", "Run SMD now?"):
                self.run_cmd(smd_cmd)

            if not tps_xyz.exists():
                logger.info(f"[SMD] Copy the result trajectory to {tps_xyz} and rerun.")
                return False

        # --- Prepare TPS directory ---
        tps_json_src = self.config.workdir / cfg.get("config_file", "tps.json")
        if not tps_json_src.exists():
            raise FileNotFoundError(f"[TPS] Config not found: {tps_json_src}")

        # Write tps.json with updated model paths (always refresh)
        with open(tps_json_src) as f:
            tps_config = json.load(f)
        tps_config["mace_model_path"] = model_paths
        tps_json_dst = tps_dir / "tps.json"
        with open(tps_json_dst, "w") as f:
            json.dump(tps_config, f, indent=4)
        logger.info("[TPS] Written tps.json with current model paths.")

        for src_path, dst_name in [
            (tps_xyz, "tps.xyz"),
            (self.config.workdir / self.config.top_file, "top.pdb"),
        ]:
            dst = tps_dir / dst_name
            if not dst.exists():
                self._copy(Path(src_path), dst)

        # Provide the run command
        run_cmd = cfg.get("run_command", f"cd {tps_dir} && python -m tps --start_type new")
        logger.info(f"[TPS] Run command:\n  {run_cmd}")

        if not self.config.confirm("continue_generation", "Continue after TPS sampler?"):
            return False

        # Verify output exists
        traj_folder = tps_dir / "traj"
        extend_folder = traj_folder / "extend"
        if not extend_folder.exists():
            logger.warning("[TPS] Extend folder not found — run TPS and extension first.")
            return False

        self._touch_done(tps_dir)
        return True


# ---------------------------------------------------------------------------
# Frame Selection Stage
# ---------------------------------------------------------------------------

class FrameSelectStage(Stage):
    name = StageName.FRAME_SELECT

    def _run(self) -> bool:
        from util.traj_util import select_frames

        tps_dir    = self.gen_dir / "tps"
        extend_dir = tps_dir / "traj" / "extend"
        output_dir = tps_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not extend_dir.exists():
            raise FileNotFoundError(f"[FrameSelect] Extend folder not found: {extend_dir}")

        with open(tps_dir / "tps.json") as f:
            tps_params = json.load(f)

        # Combine extended trajectories
        extended_files = sorted(extend_dir.glob("*_extended.xyz"))
        if not extended_files:
            raise FileNotFoundError(f"[FrameSelect] No *_extended.xyz files in {extend_dir}")

        combined = output_dir / "tps_all_extended.xyz"
        logger.info(f"[FrameSelect] Combining {len(extended_files)} trajectories...")
        with open(combined, "wb") as dst:
            for src in extended_files:
                with open(src, "rb") as s:
                    shutil.copyfileobj(s, dst)

        # Frame selection
        out_file = output_dir / "tps_selected_frames.xyz"
        logger.info("[FrameSelect] Selecting high-uncertainty frames...")
        select_frames(xyz_path=str(combined), tps_params=tps_params, out_file=str(out_file))
        logger.info(f"[FrameSelect] Selected frames → {out_file}")

        # Archive for reference
        final = self.config.workdir / str(self.gen) / f"gen_{self.gen}_selected_frames.xyz"
        self._copy(out_file, final)

        # Backup init/ and update coord.xyz for the next generation
        backup = self.config.workdir / str(self.gen) / "init"
        try:
            if not backup.exists():
                shutil.copytree(self.init_dir, backup)
                logger.info(f"[FrameSelect] Backed up init/ → {backup}")
        except Exception as exc:
            logger.warning(f"[FrameSelect] Could not back up init/: {exc}")

        for old in self.init_dir.glob("*.xyz"):
            old.unlink()
        self._copy(final, self.init_dir / "coord.xyz")
        logger.info(f"[FrameSelect] Updated init/coord.xyz for generation {self.gen + 1}.")

        return True


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------

class Generation:
    """Runs all stages for a single generation in order."""

    STAGES: list[type[Stage]] = [DFTStage, MLPStage, TPSStage, FrameSelectStage]

    def __init__(self, gen: int, config: WorkflowConfig, state: GenerationState):
        self.gen = gen
        self.config = config
        self.gen_state = state
        self.gen_dir = config.workdir / str(gen)
        self.gen_dir.mkdir(parents=True, exist_ok=True)

    def run(self, only_stage: Optional[StageName] = None) -> bool:
        """
        Execute all stages in order (or just `only_stage` if given).
        Returns True when the generation is fully complete.
        """
        for StageClass in self.STAGES:
            stage = StageClass(self.gen, self.config, self.gen_state)

            if only_stage and stage.name != only_stage:
                continue

            if not stage.run():
                logger.info(
                    f"[Gen {self.gen}] Halted at '{stage.name.value}'. "
                    "Rerun after completing the required manual step."
                )
                return False

        if not only_stage:
            self.gen_state.done = True
            (self.gen_dir / "DONE").touch()
            logger.info(f"[Gen {self.gen}] Complete.")

        return True


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------

class Workflow:
    """
    Top-level controller. Loads config + state, iterates over generations.

    Parameters
    ----------
    config_path : path to the JSON config file
    start_gen   : generation to start from (overrides saved state)
    only_stage  : run only this stage name for pending generations
    """

    def __init__(
        self,
        config_path: str | Path,
        start_gen: Optional[int] = None,
        only_stage: Optional[str] = None,
    ):
        self.config = WorkflowConfig.from_json(config_path)
        self.state_path = self.config.workdir / self.config.state_file
        self.state = self._load_state()
        self.start_gen = start_gen if start_gen is not None else self.state.current_generation
        self.only_stage = StageName(only_stage) if only_stage else None

        (self.config.workdir / "init").mkdir(parents=True, exist_ok=True)

        logger.add(
            self.config.workdir / self.config.log_file,
            level="DEBUG",
            rotation="10 MB",
        )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> WorkflowState:
        if self.state_path.exists():
            with open(self.state_path) as f:
                return WorkflowState.from_dict(json.load(f))
        return WorkflowState()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.debug(f"State saved → {self.state_path}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            f"Workflow starting | workdir={self.config.workdir} | "
            f"max_gen={self.config.max_generations} | start_gen={self.start_gen}"
        )

        for gen_idx in range(self.start_gen, self.config.max_generations):
            gen_state = self.state.get_or_create_gen(gen_idx)

            if gen_state.done and not self.only_stage:
                logger.info(f"[Gen {gen_idx}] Already complete — skipping.")
                continue

            logger.info(f"{'=' * 60}")
            logger.info(f"  Generation {gen_idx}")
            logger.info(f"{'=' * 60}")

            self.state.current_generation = gen_idx
            self._save_state()

            generation = Generation(gen_idx, self.config, gen_state)
            completed = generation.run(only_stage=self.only_stage)
            self._save_state()

            if not completed:
                logger.info("Workflow paused — resolve the blocked stage and rerun.")
                return

            # Prompt (or auto-confirm) before advancing
            if not self.config.confirm(
                "continue_generation",
                f"Generation {gen_idx} complete. Continue to generation {gen_idx + 1}?",
            ):
                logger.info("Stopping at user request.")
                return

            self.state.current_generation = gen_idx + 1
            self._save_state()

        logger.info("All generations complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="MLPath active-learning workflow.")
    p.add_argument("--config", required=True, help="Path to workflow JSON config file.")
    p.add_argument("--gen", type=int, default=None,
                   help="Override starting generation (default: resume from saved state).")
    p.add_argument("--stage", type=str, default=None,
                   choices=[s.value for s in StageName],
                   help="Run only this stage for the current generation.")
    args = p.parse_args()

    Workflow(config_path=args.config, start_gen=args.gen, only_stage=args.stage).run()
