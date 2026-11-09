# How to run the workflow (and what to expect)

## 1. Setup

Create the environment once (see the README; it provides torch, mace, openmm(+ml), openpathsampling, ase, plumed, click):

```bash
mamba env create -f environment.yml && mamba activate mace-tps-workbench
python scripts/check_environment.py          # reports what is usable
```

Then, in every shell:

```bash
export PY=$(which python)           # python of that env
source scripts/env.sh               # sets REPO, PYTHONPATH, KMP_DUPLICATE_LIB_OK
mt(){ $PY -m cli.main "$@"; }       # short alias for the CLI used below
# only for steered MD (smd_settings):
export PLUMED_KERNEL=<env>/lib/libplumedKernel.dylib   # .so on Linux
```

CP2K (`cp2k.psmp`) must be on PATH on the machine where you run DFT; mock-only runs need no extras.

The CLI is `python -m cli.main` (or `mtps` if the packages are pip-installed):

| command | purpose |
|---|---|
| `run --config CFG` | advance until a gate blocks or all generations finish |
| `status --config CFG` | per-generation stage, blocked gate, error |
| `approve --config CFG GATE_ID` | release a manual gate |

Gate ids: `init_md` (only with init MD), `gen<N>:dft_submit`, `gen<N>:smd_run`, `gen<N>:tps_run`, `gen<N>:continue_next_gen`.

## 2. Quick check (mock, ~1 s, tested)

```bash
mt run --config example_mock_config.json          # stops: Waiting for approval: [gen0:dft_submit]
mt approve --config example_mock_config.json gen0:dft_submit
mt run --config example_mock_config.json          # gen 0 dft_done, mlp_done, tps_done, complete ... then blocks at gen1
mt status --config example_mock_config.json
```
Expected: `All 2 generations complete.` after approving `gen1:dft_submit` as well.

## 3. Real run (paper settings)

A ready-made project for the Cu(100)/water CO2 system is in `examples/Cu_CO2/`:

| path | content |
|---|---|
| `init/` | 13 initial structures (`coord*.xyz`, each with its box) and the CP2K inputs |
| `cp2k_data/` | CP2K basis/potential files named in `force_eval.inp` |
| `dft.json`, `mlp.yaml`, `tps.json`, `top.pdb`, `tps.xyz` | DFT, MACE-training and TPS settings, topology, initial path |
| `smd/path0`, `smd/path5` | steered-MD inputs (`smd.xyz`, `smd.json`, `plumed.dat`) for the two pathways |
| `data/gen0-5/` | cumulative train/test sets per generation (`.xyz.gz` for gens 0-2, `.xyz.xz` for gens 3-5) |
| `models/gen0-5/c0-2/` | trained MACE committee (3 models) per generation |

`MANIFEST.md` there lists every file with checksum and origin. Gen 3 in the source run was built from incomplete CP2K output (fewer labelled frames than planned). Use the data and models
for analysis or as a starting point; the workflow regenerates them itself.

```bash
export CP2K_DATA_DIR=$PWD/examples/Cu_CO2/cp2k_data
$PY scripts/make_config.py examples/Cu_CO2 cfg/workflow_config.json \
    --mlp examples/Cu_CO2/mlp.yaml --root-dir <run_dir> --paper --all-init
```
For your own system, pass your project directory (same layout) instead of `examples/Cu_CO2`.
`--paper`: 1000 frames/gen, 2000 MC steps, lambda=0.7 selection, 400 K smearing (patched *copies* in
`cfg/paper_inputs/`), 6 generations. `--all-init`: all `init/coord*.xyz` (dft.json may list fewer).
Add `--foundation medium --smd-auto` plus `smd.json/smd.xyz/plumed.dat` in the project to run
steered MD (ASE+PLUMED) with MACE-MP instead of training. The config stores absolute paths: regenerate
it on the machine/cluster where you run.

Loop for every generation:

1. `run` -> blocks at `gen<N>:dft_submit`; `approve` it; `run` again.
2. DFT is external: the engine prepares one `structure_N` dir per input (gen>0: per split) and prints
   `cd .../structure_N && mpirun -np ... cp2k.psmp -i cp2k.inp -o cp2k.out` for each. Run them (cluster).
   The gate is already approved, so after CP2K finishes just `run` again (no new approve needed).
   If any dir is incomplete the engine lists `(not finished) structure_N: reason` and blocks again.
3. Training (3 MACE seeds 123/234/345, GPU recommended), TPS (OpenPathSampling), selection run in
   process; approve `tps_run` / `smd_run` / `continue_next_gen` if those gates are `manual`.

## 3a. Initial-data generation (foundation-model MD, paper SI)

Start from **n raw xyz files, each with its box** (extxyz `Lattice=...`; or set `cell` in the settings).
Every file is run through Langevin MD with the foundation model (MACE-MP medium by default); the frames
become the generation-0 DFT inputs (one `structure_N` per file). Paper values (defaults): 400 K, 1 fs step,
1 ns (1,000,000 steps), 400 equally spaced frames per structure.

```bash
$PY scripts/make_config.py <project> cfg/workflow_config.json --mlp <project>/mlp.yaml \
    --root-dir <run_dir> --paper --init-md a.xyz b.xyz c.xyz        # add --init-md-auto to run inside the engine
```
Config block (`init_md_settings`): `inputs`, `model` (`mace_mp:medium` or a model path), `temperature`, `time_step`,
`total_steps`, `n_frames`, `friction_per_fs` (Langevin friction in fs^-1, default 1e-3 = 1 ps^-1), `cell`, `seed`, `device`, `auto_run`.

Flow: `run` -> blocks at gate `init_md` -> `approve ... init_md` -> `run`. Without `auto_run` it prints one
command per input. Run them on a GPU node: 1 ns of ~420 atoms takes days on a CPU.

```
python -m mlp.init_md --model mace_mp:medium --input a.xyz --output <run_dir>/init_md/init_0.xyz --settings init_md.json
```
then `run` again. Output: `<run_dir>/init_md/init_<N>.xyz` (N = input order, 0-based), `init_md.json`, and
`init_<N>.log` with `auto_run`. Interrupted MD resumes from the last written frame (velocities re-drawn);
finished files are skipped. A missing box raises an error (no silent default).


## 4. Expected output

Run directory layout:

```
<run_dir>/workflow_state.json     # stage per generation, file paths, blocked gate, error
<run_dir>/gate_approvals.json
<run_dir>/init_md/init_<N>.xyz    # only with init_md_settings: foundation-MD frames (gen-0 DFT inputs)
<run_dir>/<N>/dft/structure_K/    # coord.xyz cp2k.inp force_eval.inp (+ cp2k-pos-1.xyz, cp2k-frc-1.xyz, cp2k.out after CP2K)
<run_dir>/<N>/dft/split_K.xyz     # gen>0 only, when split for parallel DFT
<run_dir>/<N>/mlp/data/{train,test}.xyz
<run_dir>/<N>/mlp/runs/c0..c2/MACE_models/mace01.model
<run_dir>/<N>/tps/smd/            # only with smd_settings: md_out.xyz, COLVAR, tps.xyz, run_md.log
<run_dir>/<N>/tps/output/tps_selected_frames.xyz   # TPS candidates (extension + selection)
<run_dir>/<N>/gen_<N>_selected_frames.xyz          # frames sent to DFT in generation N+1
```

Stage progression per generation: `pending -> dft_done -> mlp_done -> tps_done -> complete`.
`status` shows `[BLOCKED]` with the gate and instructions; a stage error becomes `failed` + message.

## 5. Checklist after each generation

- `mt status` shows `complete`; no `error:` line.
- `labeled` count = sum of structure frames minus unconverged ones; train+test = cumulative total.
- 3 model files exist under `mlp/runs/c0..c2` (or the foundation model path in foundation mode).
- `gen_<N>_selected_frames.xyz` is non-empty (`selection_mode: "sampler"`: equals the TPS candidate file).
- Force/energy RMSE on the test set falls over generations; committee variance projected on the CVs
  (CN_H(O), d_CO, A(OH), d_OO) shifts towards new regions (see paper Fig. 3).

## 6. Troubleshooting

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: click` | `pip install click` in the env |
| torch import aborts (libomp) | `export KMP_DUPLICATE_LIB_OK=TRUE` (set by `scripts/env.sh`) |
| OPS: "Cannot store function ... globals" | a custom CV function uses module-level globals; import numpy etc. inside the function |
| OPS: `IndexError` in `PathSampling` | the initial path does not connect state A and B (e.g. SMD too short) |
| SMD: "missing input file coord.pdb" | the PLUMED input needs `coord.pdb`; `mlp.run_md` writes it from `coord.xyz`, so check the SMD directory is the one the run uses |
| CP2K cannot find `BASIS_SET` etc. | set `CP2K_DATA_DIR` (e.g. `examples/Cu_CO2/cp2k_data`) |
| DFT gate re-blocks with "(not finished)" | CP2K output incomplete: finish/rerun that `structure_N` |
| `NaN`/no frames from TPS selection | check `e_var_max`/`f_var_max` thresholds and `num_select` |

