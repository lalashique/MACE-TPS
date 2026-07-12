"""Adapter from the legacy CP2K implementation (legacy_cp2k.py) to the new
DFTEngine interface. This is where the "submit a job and wait for a human"
part of the old workflow.py becomes an explicit NeedsExternalJob instead of
an input()/exit(0) pair.
"""

from __future__ import annotations

from pathlib import Path

from core.interfaces import DFTEngine, NeedsExternalJob

from dft.legacy_cp2k import DFTFactory


class CP2KDFTEngine(DFTEngine):
    def __init__(self, settings: dict, input_files: list[str]):
        self.settings = settings
        self.input_files = input_files

    def label(self, structures: Path, workdir: Path) -> Path:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

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
        struct_dir = workdir / "structure_0"
        out_file = struct_dir / "extdata.xyz"

        if out_file.exists():
            return out_file

        if not state_file.exists():
            # First time seeing this generation: prepare CP2K inputs and stop -
            # someone has to actually run CP2K before we can parse anything.
            engine.run([structures])
        else:
            # We were prepared already; see if the external CP2K run finished.
            engine.state = engine._load_state()
            try:
                engine._parse_outputs()
            except Exception:
                pass  # CP2K output not there/complete yet - fall through to the gate below
            if out_file.exists():
                return out_file

        # _write_reftraj_input copies each input file to struct_dir under its
        # basename (see legacy_cp2k.CP2KDFT._write_reftraj_input) - the
        # printed command must reference that basename, not the original
        # (possibly relative-to-repo-root) input_files[0] path.
        input_basename = Path(self.input_files[0]).name
        raise NeedsExternalJob(
            stage="dft_submit",
            instructions=(
                f"Run CP2K in {struct_dir} and then re-run this generation:\n"
                f"  cd {struct_dir} && {self.settings.get('command', '<cp2k-command>')}"
                f" -i {input_basename} -o cp2k.out"
            ),
        )
