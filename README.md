# MACE-TPS Workbench

> **Data for the paper** ([arXiv:2605.03737](https://arxiv.org/abs/2605.03737)) is in [`examples/`](examples/):
> see [`examples/Cu_CO2/`](examples/Cu_CO2/) for the inputs, the training/test data and the trained MACE models
> of generations 0-5.
> To run the workflow on it, follow [docs/HOWTO_RUN.md](docs/HOWTO_RUN.md).

An active-learning pipeline for discovering reaction mechanisms: DFT labeling -> MACE committee
training -> Transition Path Sampling (TPS) -> uncertainty-driven frame selection -> next generation,
repeated until the committee converges on the pathway(s) a system actually takes. Built so:

1. The DFT, MLP, and TPS/AIMMD parts each live in their own package and can be worked on (or swapped
   for a different implementation) without needing each other's heavy dependencies installed.
2. The generational loop is driven by an explicit, resumable state machine instead of blocking
   `input()` prompts and sentinel files - automatable via the CLI (or, later, a GUI) instead of
   requiring someone to babysit a terminal.
3. There's a fast, dependency-free mock mode that runs the entire multi-generation loop in well under
   a second, to sanity-check the plumbing without paying for CP2K, MACE training, or OpenPathSampling.

See [docs/CREATING_A_NEW_SYSTEM.md](docs/CREATING_A_NEW_SYSTEM.md) for how to set up your own system. To run the pipeline on a real periodic system (Cu(100)/water with CO2, as in
[arXiv:2605.03737](https://arxiv.org/abs/2605.03737)), follow [docs/HOWTO_RUN.md](docs/HOWTO_RUN.md);
the inputs, final training data and trained models are in [examples/Cu_CO2/](examples/Cu_CO2/).

Features beyond the basic loop:

- **Foundation-model mode:** use MACE-MP instead of training (`mlp_settings.foundation_models`).
- **Initial-data generation:** run foundation-model MD on any number of initial structures to create the
  generation-0 DFT inputs (`init_md_settings`, gate `init_md`).
- **Steered MD:** ASE + PLUMED steered MD to build the initial TPS path (`smd_settings`).
- **Frame selection:** the per-metric weighted draw, or the uncertainty-weighted scheme of the paper
  (`selection_scheme: "paper"`, `lambda`).
- **Parallel and cumulative DFT/training:** several initial structures, DFT split into chunks, training on
  all labelled data so far.

## Layout

```
packages/
├── core/       # shared contracts. No heavy deps - everyone installs this.
│   └── core/
│       ├── interfaces.py   # DFTEngine, MLPBackend, TPSSampler ABCs + NeedsExternalJob
│       ├── schemas.py      # WorkflowConfig
│       ├── state.py        # WorkflowState / GenerationRecord (persisted JSON, not sentinel files)
│       ├── gates.py        # GateController / WaitingForApproval (replaces input())
│       └── xyz_io.py       # stdlib-only extxyz-ish IO for the mock backends + generic block ops
│
├── dft/        # DFT labeling backends
│   └── dft/
│       ├── mock_backend.py    # instant toy-potential labeling, zero deps
│       ├── pyscf_backend.py   # real, non-periodic (molecules) - pip-installable, in-process
│       ├── cp2k_backend.py    # real, periodic (slabs/solvated) - external CP2K process
│       ├── legacy_cp2k.py     # CP2K reftraj engine implementation
│       ├── reftraj_check.py   # checks that a CP2K run finished before its output is parsed
│       └── cpoker_iparser.py  # @SET/@include-aware CP2K input-deck parser
│
├── mlp/        # MLP (MACE) training/inference backends
│   └── mlp/
│       ├── mock_backend.py    # instant "committee", zero deps
│       ├── legacy_mace.py     # MACE_MLP: committee training, loading, evaluation
│       ├── foundation.py      # MACE-MP foundation models (`mace_mp:<size>`), no training
│       ├── init_md.py         # initial-data generation: MD with a foundation model
│       ├── run_md.py          # ASE + PLUMED steered-MD runner
│       └── mace_backend.py    # adapter: MACE_MLP -> MLPBackend interface
│
├── tps/        # TPS/AIMMD rare-event sampling backends
│   └── tps/
│       ├── mock_backend.py    # instant toy shooting-point jitter, zero deps
│       ├── legacy_ops.py      # OpenPathSampling setup/sampler + extend_and_select_frames()
│       ├── legacy_aimmd.py    # AIMMD RCModel/selector setup (not yet wired into a sampler)
│       └── ops_backend.py     # adapter: OpenPathSampling logic -> TPSSampler interface
│
├── workflow/   # orchestration only - imports core + whichever backend package is configured
│   └── workflow/engine.py     # WorkflowEngine: the generational state machine
│
└── cli/        # thin wrapper around workflow.engine
    └── cli/main.py            # `mtps run|status|approve`

examples/
└── Cu_CO2/        # inputs, train/test data and MACE committees (generations 0-5) of the Cu(100)/water CO2 system

scripts/
├── env.sh               # sets PYTHONPATH etc. so the packages run from a checkout
├── check_environment.py # reports which backends are usable
├── make_config.py       # project directory -> workflow config JSON
└── verify_config.py     # checks a generated config (no backend is run)

docs/              # HOWTO_RUN.md (real run), CREATING_A_NEW_SYSTEM.md, HISTORY.md

tests/
└── test_workflow_smoke.py     # full mock end-to-end loop + manual-gate resume test
```

**The rule that keeps the three backend packages independent:** `workflow/` and `cli/` only ever
import `core.interfaces.{DFTEngine,MLPBackend,TPSSampler}` and the factory functions
(`dft.create_dft_engine`, `mlp.create_mlp_backend`, `tps.create_tps_sampler`). They never import
`mace`, `cp2k`, or `openpathsampling` directly. Whoever owns `dft/`, `mlp/`, or `tps/` can change
absolutely everything about their implementation as long as they keep satisfying the ABC in
`core/interfaces.py`.

## Installation

**Mock-only** (no scientific dependencies, for developing/testing the workflow engine itself):

```bash
python3 -m venv .venv
./.venv/bin/pip install -e packages/core -e packages/dft -e packages/mlp \
                         -e packages/tps -e packages/workflow -e packages/cli
./.venv/bin/python -m pytest tests/ -v
```

**Real MLP + PySCF DFT** (plain pip, no conda needed):

```bash
./.venv/bin/pip install -e "packages/dft[real,pyscf]" -e "packages/mlp[real]"
```

**Everything, including real TPS** (needs conda/mamba - the OpenPathSampling stack, OpenMM +
openmm-ml + openmmtools + mdtraj, isn't reliably pip-installable together):

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
mamba activate mace-tps-workbench
python scripts/check_environment.py   # reports what's actually usable, with install links for gaps
```

CP2K (for periodic/slab systems) is a separate scientific-computing binary, not installed by
`environment.yml` or any pip extra - `scripts/check_environment.py` warns if it's not on PATH rather
than failing silently later; see [cp2k.org/download](https://www.cp2k.org/download) to install it,
or use the PySCF backend instead if your system is non-periodic (a single molecule).

## Quick mock run (< 1 second)

```bash
mtps run --config example_mock_config.json
# ... hits a manual gate ...
mtps approve --config example_mock_config.json gen0:dft_submit
mtps run --config example_mock_config.json
mtps status --config example_mock_config.json
```

See `tests/test_workflow_smoke.py` for a config example and for how gates, resumption, and generation
state are meant to be used/asserted on.

## Switching a stage to the real backend

Set `"dft_backend": "cp2k"` or `"pyscf"`, `"mlp_backend": "mace"`, `"tps_backend": "ops"` in the
workflow config (see `core/schemas.py:WorkflowConfig` for all fields), with the corresponding
`dft_settings`/`mlp_settings`/`tps_settings` dicts - these are opaque to `workflow/` and `core/`; only
the matching backend package interprets them. Mix and match freely, e.g. mock DFT + real MLP to test
training behavior without needing DFT running at all.

## Approval gates

Every point in the pipeline that would otherwise need a human to babysit a prompt is a named gate in
`WorkflowConfig.gates`:

| gate                  | when it fires                                  |
|-----------------------|-------------------------------------------------|
| `dft_submit`          | before labeling a generation's structures        |
| `smd_run` / `tps_run` | before launching TPS sampling for a generation    |
| `continue_next_gen`   | before moving on to the next generation           |

Set a gate to `"auto"` to never block (what the mock config does) or `"manual"` to require
`mtps approve <gate_id>` before that stage proceeds. Real backends can also raise
`core.interfaces.NeedsExternalJob` mid-stage (e.g. CP2K's `label()` does this when it has prepared
inputs but nothing has actually run CP2K yet) - the engine turns that into the same
wait-for-approval mechanism automatically, so "please go run this on the cluster and come back" and
"please explicitly confirm before I proceed" both look the same from the CLI's (or, later, a GUI's)
point of view.

## Further reading

- [docs/HOWTO_RUN.md](docs/HOWTO_RUN.md) - running the full pipeline on a real system: setup, config
  generation, the DFT/approval loop, initial-data generation, expected output and troubleshooting.
- [docs/CREATING_A_NEW_SYSTEM.md](docs/CREATING_A_NEW_SYSTEM.md) - how to set up a new system: choosing a
  DFT backend, designing a reaction-progress CV, bootstrapping a seed trajectory, scaling up the
  generational loop, and structuring an analysis notebook.
- [docs/HISTORY.md](docs/HISTORY.md) - development history: what was carried over from an earlier
  codebase this was built from, real bugs found and fixed along the way, and what's been verified by
  actually running it end-to-end (not just assumed to work).

## License

MIT - see [LICENSE](LICENSE).
