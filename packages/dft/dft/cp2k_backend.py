"""Adapter from the legacy CP2K implementation (legacy_cp2k.py) to the new
DFTEngine interface. This is where the "submit a job and wait for a human"
part of the old workflow.py becomes an explicit NeedsExternalJob instead of
an input()/exit(0) pair.
"""

from __future__ import annotations

from pathlib import Path

from core.interfaces import DFTEngine, NeedsExternalJob
from core.xyz_io import concat_xyz

from dft.legacy_cp2k import DFTFactory
from dft.reftraj_check import reftraj_missing


class CP2KDFTEngine(DFTEngine):
    def __init__(self, settings: dict, input_files: list[str]):
        self.settings = settings
        self.input_files = input_files

    def label(self, structures, workdir: Path) -> Path:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        structs = [Path(x) for x in (structures if isinstance(structures, (list, tuple)) else [structures])]

        combined = workdir / "output" / "combined_extdata.xyz"
        if combined.exists():
            return combined

        # run_mode="prepare" is deliberate, not "run": DFTBase.run() with
        # run_mode="run" calls _run_calculation() (a no-op placeholder here -
        # CP2K execution is external) immediately followed by
        # _parse_outputs(), which then fails because CP2K hasn't produced
        # any output yet. "prepare" stops right after writing inputs, which
        # is the only thing this process can actually do before a human/
        # scheduler runs CP2K.
        engine = DFTFactory.create(
            engine="cp2k",
            settings=self.settings,
            input_files=self.input_files,
            reftraj=True,
            run_mode="prepare",
            workdir=str(workdir),
        )

        state_file = workdir / "state.json"
        pending_note = ""
        if not state_file.exists():
            # First time seeing this generation: prepare one structure_N per
            # input file and stop - someone has to run CP2K before we can parse.
            engine.run(structs)
        else:
            # Already prepared; only parse once EVERY structure's CP2K run is complete
            # (parsing a half-written trajectory would yield truncated training data).
            engine.state = engine._load_state()
            struct_dirs = sorted(
                (Path(k) for k in engine.state if "structure_" in k),
                key=lambda f: int(f.name.split("_")[-1]),
            )
            incomplete = {d.name: r for d in struct_dirs if (r := reftraj_missing(d))}
            if struct_dirs and not incomplete:
                engine._parse_outputs()  # errors here are real (bad output) - let them surface
                ext_files = [d / "extdata.xyz" for d in struct_dirs]
                concat_xyz(ext_files, combined)  # legacy: output/combined_extdata.xyz
                return combined
            pending_note = "\n".join(f"  (not finished) {n}: {r}" for n, r in incomplete.items())

        # _write_reftraj_input copies each input file to struct_dir under its
        # basename - the printed command must reference that basename, not the
        # original (possibly relative-to-repo-root) input_files[0] path.
        input_basename = Path(self.input_files[0]).name
        cmd = self.settings.get("command", "<cp2k-command>")
        struct_dirs = sorted(workdir.glob("structure_*"), key=lambda f: int(f.name.split("_")[-1]))
        runs = "\n".join(f"  cd {d} && {cmd} -i {input_basename} -o cp2k.out" for d in struct_dirs)
        raise NeedsExternalJob(
            stage="dft_submit",
            instructions=f"Run CP2K in each structure dir and then re-run this generation:\n{runs}"
            + (f"\n{pending_note}" if pending_note else ""),
        )
