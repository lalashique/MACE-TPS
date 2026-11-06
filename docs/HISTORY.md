# Development history: porting and real-backend verification

This workbench's `dft/`, `mlp/`, and `tps/` packages started from an existing, unpublished research
codebase (a DFT -> MACE committee -> Transition Path Sampling active-learning pipeline). This document
records what was carried over, what was fixed, and what was verified by actually running it - useful
context if you're modifying those packages, not needed to just use the workbench (see the main
[README](../README.md) for that).

## What's carried over vs. genuinely new

Ported with adapters, not rewritten (these files are the original logic, lightly reorganized - the
domain owners should keep iterating on them):
- `dft/legacy_cp2k.py`, `mlp/legacy_mace.py`, `tps/legacy_ops.py`, `tps/legacy_aimmd.py`

Genuinely new:
- `core/` package (interfaces, state, gates, config schema, xyz_io)
- All three `mock_backend.py` files
- `workflow/engine.py` (the generational state machine)
- `cli/` package
- `tests/test_workflow_smoke.py`

## Real bugs found and fixed while porting

- `mlp/legacy_mace.py`: `MACE_MLP._train_func`/`_calc_cls` were referenced but never assigned (the
  imports were commented out) - any real training or model-loading call raised `AttributeError`. Now
  wired up in `__init__`.
- `mlp/legacy_mace.py::_run_training`: built a shell command string
  (`"cd <dir> && python -m mace.cli.run_train --config <file>"`) and fed it to `str.split()` to
  construct `sys.argv` for an **in-process** call to `mace.cli.run_train.main()`. That command string
  is only valid for real shell/HPC execution - in-process, `sys.argv[0]` ended up being `"cd"`, which
  argparse used as the program name, and `--config` resolved against the wrong cwd. Confirmed by
  actually running it (see "Real-backend verification" below) - it failed with argparse printing
  `usage: cd ...`. Fixed to `os.chdir(member_dir)` + a real `sys.argv`; the shell-command string is
  still returned separately for the HPC/`prepare_only` path, where it's genuinely meant to be run as a
  shell command by a human/scheduler.
- `mlp/legacy_mace.py::train`: `next(member_dir.glob("*.model"), None)` only looked in the top level
  of the training directory, but MACE writes the compiled model under `model_dir`
  (`MACE_models/mace01_compiled.model` by default) - the original glob never found it, so
  `committee_models`/`model_paths` came back empty even after training succeeded. Confirmed by running
  it: training completed with real loss curves, then the assertion on committee size failed. Fixed to
  check `model_dir` first, falling back to a recursive glob.
- `mlp/legacy_mace.py::evaluate_committee` was missing `atoms.arrays["force_var_per_atom"]` (present
  in the original, dropped when this was first ported) - `tps/util/mlp_helper.py::process_frame`
  needs it and raised a `KeyError` without it.
- `dft/mock_backend.py`: originally wrote labeled data using `core.xyz_io`'s minimal format (a bare
  `energy=` comment, no `Properties=` header). That's fine for mock-to-mock communication, but real
  `mace.cli.run_train` (via `ase.io.read`) requires a proper extxyz `Properties=` header and a comment
  key matching `energy_key`/`forces_key` (`energy_cp2k`/`forces_cp2k` by default) - without it,
  training fails with `"None of 'energy_cp2k', 'forces_cp2k', ... found"`. Confirmed by actually
  running mock DFT + real MLP together. Fixed `MockDFTEngine` to write genuine ASE-compatible extxyz,
  so "mock DFT feeding a real MLP backend" actually works, not just in the all-mock or all-real cases.
- `tps/legacy_ops.py::extend_and_select_frames`: this logic (committee variance evaluation, trajectory
  extension, threshold filtering, weighted frame selection) existed only inside an
  `if __name__ == "__main__":` block in the original, so it could never be called programmatically -
  it's now a proper function the OPS backend adapter calls.
- `tps/util/cv.py`'s `bond`/`CN`/`bond_count` static methods imported
  `MDAnalysis.lib.distances.distance_array` at module scope and referenced it as a closed-over global.
  OpenPathSampling's storage layer refuses to serialize a CV function with unresolvable globals
  ("Cannot store function! ... globals: ['distance_array']") - moved to local imports inside each
  function, one of the fixes OPS's own error message suggests.
- `tps/legacy_ops.py::ISetup._setup_system` hardcoded `MLPotential("mace", ..., device="cuda")`, and
  `extend_and_select_frames` hardcoded `.lower()` on `platform_name` for the MACE device - both
  unrunnable on any non-NVIDIA machine. Now mapped through `{"CUDA": "cuda", "CPU": "cpu"}` with a
  `cpu` fallback.
- `tps/ops_backend.py::OPSTPSSampler.sample()` never resolved `initial_trajectory`/`top_file`/model
  paths to absolute paths before its `os.chdir(workdir)` - a relative
  `WorkflowConfig.initial_trajectory` (the realistic case) broke with `OSError: No such file: tps.xyz`
  once inside the generation's working directory. Fixed to resolve everything upfront.
- `dft/legacy_cp2k.py::_parse_outputs` imported `cpoker.iparser`/`cpoker.genfun`, not a dependency
  published anywhere - rewritten using plain ase/numpy for the energy/force parsing
  (`cp2k-pos-1.xyz`/`cp2k-frc-1.xyz` comment lines), same Hartree->eV / Hartree/Bohr->eV/Angstrom
  conversion factors as the original. The cell (`&CELL ABC ...`) initially used a flat regex over
  `cp2k.inp`, which is wrong for real production decks - they commonly `@include` the `&CELL` section
  from a separate file. Once the original `cpoker` library was made available, its `iparser` class -
  which resolves CP2K's `@SET`/`@include` directives - was vendored (trimmed, no `genfun` dependency)
  into `dft/cpoker_iparser.py` and replaces the regex. `genfun.py` itself was intentionally NOT
  vendored: its only use in the original was `unconv()`'s grep/awk convergence check, superseded by
  parsing energies directly as above; pulling in `genfun` (and its `numba`/`uncertainties`
  dependencies) for that would add nothing. Simplification vs. the original: still no
  convergence-flag filtering - every reftraj frame is kept.
- `dft/legacy_cp2k.py::_write_reftraj_input` stored the state-dict key as just `"structure_0"` (a bare
  basename) while `_parse_outputs` did `Path(struct_key).resolve()` - that resolves against whatever
  the process's cwd happens to be at parse time, not the actual structure directory. Fixed to store
  the full resolved path as the key.
- `dft/cp2k_backend.py::CP2KDFTEngine.label()` created its engine with `run_mode="run"`, but
  `DFTBase.run()` in that mode calls `_run_calculation()` (a no-op here - CP2K execution is external)
  immediately followed by `_parse_outputs()`, which then fails because CP2K hasn't produced anything
  yet. Fixed to `run_mode="prepare"`, which stops right after writing inputs; `_parse_outputs()` is
  now called explicitly only on the second call, after CP2K has actually run.

## Real-backend verification

Every stage - DFT, MLP training, TPS, trajectory extension, committee (QBC) frame selection, and
advancing to the next generation - has actually been run for real, CPU-only, no cluster. `real_test/`
holds the scripts that did this and surfaced the bugs above.

**MLP** (`real_test/run_real_mlp_test.py`, `run_mixed_workflow_test.py`): trains a real 2-model MACE
committee on small Cu clusters labeled with ASE's EMT calculator, runs real committee-uncertainty
prediction, and drives 2 full `WorkflowEngine` generations with mock DFT + mock TPS + real MACE.

**DFT** (`real_test/run_real_dft_test.py`): drives `CP2KDFTEngine.label()` against a real, tiny
(3-atom water) CP2K reftraj calculation - actual quantum-chemistry PBE/GTH energies and forces.
Confirms `NeedsExternalJob` fires before CP2K has run and the engine correctly parses output and
resumes after it has.

**TPS/extension/QBC** (`real_test/tps_real/`): a synthetic 2-atom double-well potential (built so
there's an actual rare event to sample) is labeled and used to train a real MACE committee, which
drives real OpenMM dynamics inside real two-way-shooting TPS - genuine Monte Carlo accept/reject.
`extend_and_select_frames` then extends the accepted paths, evaluates committee (QBC) energy/force
variance, and weighted-samples the highest-uncertainty frames - which clustered right around the
barrier region, the physically expected place for committee disagreement to peak.
`real_test/tps_real/run_full_real_workflow.py` wires this through the actual
`ops_backend`/`mace_backend` adapters (not just the raw legacy classes) inside `WorkflowEngine`, for 2
full generations - the exact DFT -> MLP -> TPS -> extension -> QBC -> next-generation loop.

The full formic-acid tutorial (see [tutorials/formic_acid_decomposition/](../tutorials/formic_acid_decomposition/))
is the larger-scale, still-real follow-up to this: a system with two competing decomposition pathways,
run for 4 generations with a 4-member committee - both pathways came out as genuinely distinct,
smoothly-evolving trajectories (verified in that tutorial's `analysis.ipynb`, not just asserted).

The OpenPathSampling stack (OpenMM, openmm-ml, openmmtools, mdtraj) isn't on PyPI for a compatible
combo - it's conda-forge only. See the main [README](../README.md) and
[environment.yml](../environment.yml) for how to set that up. If you hit an OpenMP double-init crash
on macOS (conda's libomp vs. torch's bundled one), set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Not done yet

- `tps/legacy_aimmd.py` is ported but not wired into a `TPSSampler` - there's no `AIMMDTPSSampler` yet.
  Building one means swapping `legacy_ops.Sampler`'s `UniformSelector` for
  `AIMMD_setup.setup_selector()`'s RCModel-based selector.
- No GUI yet, by design - `workflow.WorkflowEngine` was built so one can be added later as its own
  package driving the same `step()`/`run_to_completion()`/`GateController` API the CLI uses, without
  touching orchestration logic.
- Real CP2K/TPS running on a real multi-hundred-atom system with periodic water/electrode chemistry
  hasn't been exercised here - only small test systems (water, a synthetic diatomic, formic acid).

## Real-system workflow extensions (paper reproduction)

Added so the pipeline can run the Cu(100)/water CO2 study ([arXiv:2605.03737](https://arxiv.org/abs/2605.03737))
end to end. All options default to the earlier behaviour.

- **Inputs and DFT:** `initial_structures` may be a list (one `structure_N` per file); `n_dft_splits` /
  `dft_frames_per_split` split generation >0 DFT into parallel chunks; `cumulative_training` trains on all
  labelled data so far; the CP2K backend parses only after every `structure_N` has finished
  (`dft/reftraj_check.py`) and can drop frames whose SCF did not converge (`filter_unconverged`).
- **Selection:** `selection_mode` (`topk` | `sampler`) and `selection_scheme` (`legacy` | `paper`, with
  `lambda`); the paper scheme is `tps.util.mlp_helper.paper_selection_weights`.
- **Steered MD:** `smd_settings` with an ASE + PLUMED runner (`mlp/run_md.py`, `auto_run`) producing
  `<gen>/tps/smd/tps.xyz`, gate `smd_run`.
- **Foundation models:** `mlp_settings.foundation_models` uses MACE-MP (`mlp/foundation.py`, specs like
  `mace_mp:medium` or a path) and skips training.
- **Initial data:** `init_md_settings` runs Langevin MD with a foundation model on each initial structure
  (`mlp/init_md.py`, gate `init_md`, resumable); the frames become the generation-0 DFT inputs.
- **Fixes found by real runs:** CV functions in `tps/util/cv.py` must not use module-level globals (OPS cannot
  store them); E0s given as a dict are converted to the string MACE expects.
- **Verified by running:** CP2K + MACE + OpenPathSampling end to end for three generations, the SMD ->
  TPS chain with MACE-MP, and the initial-data MD stage (short runs). **Not run:** the full 5000-step
  steered MD, the 1 ns initial MD and the production TPS (too slow on CPU), GPU training.
- **Tooling and data:** `scripts/make_config.py` builds a config from a project directory; `examples/Cu_CO2/`
  holds the inputs, final train/test data and trained committees; usage is in [HOWTO_RUN.md](HOWTO_RUN.md).
