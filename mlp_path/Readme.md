# MLPath 🔬

**MLPath** is a Python framework for active-learning-driven molecular dynamics simulations. It orchestrates an iterative loop of *ab initio* DFT calculations → machine learning potential (MLP) training → transition path sampling (TPS) → intelligent frame selection, progressively refining a committee of MACE neural network potentials for rare-event simulation.

---

## Table of Contents

- [MLPath 🔬](#mlpath-)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Key Features](#key-features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from source](#install-from-source)
  - [Quick Start](#quick-start)
    - [1. Prepare your initial structure](#1-prepare-your-initial-structure)
    - [2. Write a workflow config](#2-write-a-workflow-config)
    - [3. Run the workflow](#3-run-the-workflow)
    - [4. Evaluate a trained committee](#4-evaluate-a-trained-committee)
  - [Configuration](#configuration)
    - [Workflow Config](#workflow-config)
    - [TPS Config](#tps-config)
    - [MLP Training Config](#mlp-training-config)
  - [Module Reference](#module-reference)
    - [`workflow.py`](#workflowpy)
    - [`mlp.py`](#mlppy)
    - [`dft.py`](#dftpy)
    - [`util/cv.py`](#utilcvpy)
    - [`util/traj_util.py`](#utiltraj_utilpy)
    - [`util/parser.py`](#utilparserpy)
    - [`util/storage_helper.py`](#utilstorage_helperpy)
    - [`util/initial_state.py`](#utilinitial_statepy)
  - [Collective Variables](#collective-variables)
  - [Testing](#testing)
  - [HPC / Cluster Workflow](#hpc--cluster-workflow)
  - [Project Structure](#project-structure)
  - [Todo](#todo)
    - [Core Functionality](#core-functionality)
    - [TPS](#tps)
    - [MLP / Active Learning](#mlp--active-learning)
    - [Workflow \& Orchestration](#workflow--orchestration)
    - [Collective Variables](#collective-variables-1)
    - [Testing \& CI](#testing--ci)
    - [Documentation](#documentation)
    - [Performance \& Robustness](#performance--robustness)
  - [Contributing](#contributing)
  - [License](#license)

---

## Overview

MLPath automates the active-learning cycle used to build reliable machine learning interatomic potentials for rare-event simulations (e.g. proton transfer, conformational transitions, chemical reactions). The workflow is fully resumable, JSON-driven, and designed to run on HPC clusters where DFT jobs and MLP training are submitted externally (e.g. via SLURM).

```
┌────────────────────────────────────────────────────────────────┐
│                      MLPath Active Learning Loop               │
│                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   DFT    │───▶│   MLP    │───▶│   TPS    │───▶│   QbC    │  │
│  │  (CP2K)  │    │  (MACE)  │    │  (OPS)   │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                                 │      │
│       └──────────────── repeat ◀────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘
```

Each **generation** lives in its own numbered subdirectory. A `DONE` sentinel file marks completed generations so the workflow resumes cleanly after interruption. `PAUSE` sentinel files gate HPC job submission — remove them to allow the workflow to proceed.

---

## Architecture

| Layer | Component | Purpose |
|---|---|---|
| **Orchestration** | `workflow.py` | State machine; drives the DFT → MLP → TPS → FrameSelect pipeline |
| **MLP** | `mlp.py` | Abstract MLP base + MACE committee training, MD, minimisation, evaluation |
| **DFT** | `dft.py` | Abstract DFT base + CP2K driver (prepare / parse reftraj outputs) |
| **Sampling** | `tps.py` | OpenPathSampling TPS setup and execution |
| **CVs** | `util/cv.py` | Collective variable definitions (bond, CN, plane angle, …) |
| **Trajectories** | `util/traj_util.py` | Train/test split, frame selection, trajectory splitting |
| **Config** | `util/parser.py` | JSON config loading and argparse builder |
| **Storage** | `util/storage_helper.py` | OPS storage → XYZ extraction |
| **Init** | `util/initial_state.py` | FAIRChem-based slab/solvent/adsorbate system builder |

---

## Key Features

- **Resumable workflow**: JSON state file tracks every stage of every generation. Re-run the same command to pick up where you left off.
- **MACE committee models**: Train N models with different random seeds and compute per-frame energy/force uncertainty for principled active learning.
- **Flexible CV library**: Bond distance (COM, min, max, axis projection), coordination number (rational switching), bond count, plane angle, and more — all configurable from JSON.
- **MDAnalysis atom selections**: CV group definitions can be plain integer lists *or* MDAnalysis selection strings (e.g. `"name CA"`).
- **CP2K reftraj mode**: Batch DFT calculations, one subdirectory per structure, with `PAUSE` sentinels for HPC gating.
- **OPS TPS integration**: Builds OpenPathSampling `CVDefinedVolume` states and disk-cached CV objects automatically.
- **FAIRChem initial structures**: Generate periodic slab + solvent + adsorbate systems via the Open Catalyst dataset utilities.
- **Pluggable DFT engines**: `DFTFactory` registry allows new engines (VASP stub included) to be registered without modifying core code.
- **Comprehensive test suite**: 100 + tests covering all modules; heavy deps (MACE, OpenMM, OPS, MDAnalysis) mocked so tests run on any machine without a GPU.

---

## Installation

### Prerequisites

- Python ≥ 3.10
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [MACE](https://github.com/ACEsuit/mace) (for MLP training/inference)
- [OpenPathSampling](https://openpathsampling.org/) (for TPS)
- [OpenMM](https://openmm.org/) + [openmmtools](https://openmmtools.readthedocs.io/) (for MD engine)
- [MDAnalysis](https://www.mdanalysis.org/)
- [mdtraj](https://mdtraj.org/)
- [loguru](https://loguru.readthedocs.io/)
- [tqdm](https://tqdm.github.io/)
- [FAIRChem](https://github.com/FAIR-Chem/fairchem) (optional, for initial structure generation)
- [CP2K](https://www.cp2k.org/) (optional, for DFT calculations)

### Install from source

```bash
git clone https://github.com/your-org/mlpath.git
cd mlpath
pip install -e ".[dev]"
```

> **Tip:** Heavy simulation dependencies (MACE, OpenMM, OPS) are only required at runtime for the respective modules. The core orchestration and utilities can be used independently.

---

## Quick Start

### 1. Prepare your initial structure

```python
from util.initial_state import ReactionInterface
from ase.io import write

interface = ReactionInterface(
    bulk_id="mp-23",
    adsorbates_list=["*OH"],
    miller_indices=(1, 1, 1),
    solvent_density=1.0,
)
write("init/coord.xyz", interface.atoms_list[0])
```

### 2. Write a workflow config

Copy `workflow_config_example.json` and edit for your system:

```bash
cp workflow_config_example.json workflow.json
# edit workflow.json — set workdir, DFT command, E0s, seeds, …
```

### 3. Run the workflow

```bash
# First run — prepares DFT inputs and pauses
python workflow.py --config workflow.json

# After removing PAUSE files and running CP2K jobs externally:
python workflow.py --config workflow.json

# Resume at a specific generation:
python workflow.py --config workflow.json --gen 2

# Run only one stage of a generation:
python workflow.py --config workflow.json --gen 1 --stage mlp
```

### 4. Evaluate a trained committee

```python
from mlp import create_mlp

mlp = create_mlp("mace")
results = mlp.evaluate(
    model_path="gen0/mlp/seed_123/model.model",
    input_file="tps/traj.xyz",
    output_file="tps/traj_evaluated.xyz",
)
```

---

## Configuration

### Workflow Config

`workflow.json` controls the full active-learning loop.

```json
{
    "workdir": ".",
    "max_generations": 5,
    "top_file": "top.pdb",
    "structures": ["init/coord.xyz"],

    "dft": {
        "engine": "cp2k",
        "input_files": ["cp2k.inp", "force_eval.inp"],
        "settings": {"command": "mpirun -np 20 cp2k.popt"},
        "workdir": "single_reftraj",
        "reftraj": true,
        "run_mode": "prepare"
    },

    "mlp": {
        "config_file": "mlp.yaml",
        "E0s": {"1": -13.509, "8": -428.893},
        "seeds": [123, 234, 345],
        "test_fraction": 0.25
    },

    "tps": {
        "config_file": "tps.json",
        "run_command": "cd {tps_dir} && python -m tps --start_type new"
    },

    "auto_confirm": {
        "remove_pause":        false,
        "continue_generation": false,
        "run_smd":             false
    }
}
```

| Key | Description |
|---|---|
| `workdir` | Root directory for all outputs |
| `max_generations` | Number of active-learning iterations |
| `structures` | List of initial coordinate files |
| `top_file` | PDB topology file (used by OPS/mdtraj) |
| `dft.run_mode` | `"prepare"` writes inputs only; `"run"` also executes |
| `mlp.seeds` | List of random seeds — one MACE model trained per seed |
| `auto_confirm` | Set keys to `true` to bypass interactive prompts on HPC |

### TPS Config

`tps.json` configures the OpenPathSampling TPS run.

```json
{
    "system_name": "my_reaction",
    "platform_name": "CUDA",
    "top_file": "top.pdb",
    "mace_model_path": ["gen0/mlp/seed_123/model.model"],
    "Integrator_settings": {
        "temperature": 330,
        "dt": 0.001,
        "friction": 1.0,
        "constant_variable": "V"
    },
    "TPS_settings": {"n_frames_max": 1000, "n_steps_per_frame": 10},
    "CV": {
        "d_OH": {
            "type": "bond",
            "params": {"group_A_atoms": [0], "group_B_atoms": [1]},
            "bound":   [0.0, 1.5],
            "unbound": [2.0, 5.0]
        }
    },
    "tot_mc_steps": 500,
    "save_every": 10,
    "num_select": 50,
    "selection_indices": [0, 1, 2],
    "local_range": 5,
    "init_path": "tps.xyz",
    "save_rejected": false
}
```

### MLP Training Config

`mlp.yaml` is passed directly to MACE's training CLI. Common fields:

```yaml
model: MACE
r_max: 5.0
num_radial_basis: 8
num_cutoff_basis: 5
max_ell: 3
num_interactions: 2
hidden_irreps: "128x0e + 128x1o"
batch_size: 10
max_num_epochs: 500
lr: 0.01
```

---

## Module Reference

### `workflow.py`

Implements a stage-based state machine over active-learning generations.

```
WorkflowConfig   — loads and validates workflow.json
WorkflowState    — serialisable state across all generations
GenerationState  — per-generation stage tracking
Stage (ABC)      — base class: run() → skip / execute / fail
  DFTStage       — wraps dft.DFTFactory
  MLPStage       — wraps mlp.MACE_MLP
  TPSStage       — submits TPS run command
  FrameSelectStage — wraps traj_util.select_frames
Generation       — runs all four stages in order, halts on first block
Workflow         — top-level entry point; saves state after every stage
```

**CLI:**
```bash
python workflow.py --config workflow.json [--gen N] [--stage dft|mlp|tps|frame_select]
```

---

### `mlp.py`

Abstract MLP base with a full MACE implementation.

| Method | Description |
|---|---|
| `MACE_MLP.train(seeds, train_file, test_file, …)` | Write YAML configs + `PAUSE` sentinels; one directory per seed |
| `MACE_MLP.evaluate_committee(atoms, calculators)` | Compute energy/force variance across the committee |
| `MACE_MLP.evaluate(model_path, input_file, …)` | Run a single model over an XYZ file |
| `MACE_MLP.run_md(model_path, md_params)` | NVT/NVE MD with ASE Langevin |
| `MACE_MLP.run_minimization(model_path, coord_file)` | BFGS geometry minimisation |

**CLI:**
```bash
python mlp.py --method run_md --model_path model.model --md_params md.json
python mlp.py --method evaluate --model_path m.model --input_file data.xyz --output_file out.xyz
```

---

### `dft.py`

Factory-based DFT engine interface.

```python
from dft import DFTFactory

engine = DFTFactory.create(
    engine="cp2k",
    settings={"command": "mpirun -np 20 cp2k.popt"},
    input_files=["cp2k.inp"],
    reftraj=True,
    run_mode="prepare",
    workdir="dft_output",
)
engine.run(["coord_0.xyz", "coord_1.xyz"])
```

| Engine | Status | Reftraj |
|---|---|---|
| `cp2k` | ✅ Full support | ✅ |
| `vasp` | 🚧 Stub only | ❌ |

**Registering a custom engine:**
```python
from dft import DFTBase, DFTFactory

class MyEngine(DFTBase):
    supports_reftraj = False
    def _prepare_structure(self, structures): ...
    def _run_calculation(self, structures): ...
    def _parse_outputs(self): ...

DFTFactory.register("myengine", MyEngine)
```

---

### `util/cv.py`

Builds OPS `CoordinateFunctionCV` objects from JSON config and plots each CV along the initial trajectory.

**Available CV types:**

| Type | Description | Key parameters |
|---|---|---|
| `bond` | Distance between two atom groups (COM, min, max, or axis projection) | `group_A_atoms`, `group_B_atoms`, `type`, `axis` |
| `bondfunction` | Linear combination of two inter-group distances | `coeff` |
| `CN` | Coordination number via rational switching function | `d0`, `n`, `m` |
| `bond_count` | Count pairs at a given distance | `value`, `rounding` |
| `bond_plane_angle` | Angle between COM vector and a Cartesian plane | `axis` |
| `distance_to_OH` | Distance from atom to the other hydroxide oxygen | `bond_cutoff`, `box` |

Atom selections can be integer arrays **or** MDAnalysis selection strings:
```json
"params": {
    "group_A_atoms": "name CA",
    "group_B_atoms": [10, 11, 12]
}
```

---

### `util/traj_util.py`

| Function | Description |
|---|---|
| `train_test_split(xyz, test_fraction, seed)` | Shuffle + split; writes `train.xyz` / `test.xyz` |
| `select_frames(xyz_path, tps_params, out_file)` | Pick high-uncertainty frames by energy/force variance |
| `split_times(xyz, split, seed)` | Divide a trajectory into N equal segments |

---

### `util/parser.py`

| Function | Description |
|---|---|
| `read_config(file_path)` | Load a JSON file → dict |
| `print_config(config)` | Recursively pretty-print nested config |
| `create_parser(arguments_list)` | Build argparse from dict/tuple spec list |
| `global_arguments` | Shared argument definitions (flags, types, help) |

---

### `util/storage_helper.py`

Extract accepted/rejected trajectories and shooting-point snapshots from an OPS `.db` storage file:

```bash
python -m util.storage_helper \
    --storage tps_storage.db \
    --config  tps.json \
    --out     traj/
```

Writes per-step XYZ files and a `shooting_traj.xyz` annotated with acceptance status. Resumes automatically from the last written step.

---

### `util/initial_state.py`

Build periodic slab + solvent + adsorbate systems via FAIRChem:

```python
from util.initial_state import ReactionInterface

interface = ReactionInterface(
    bulk_id="mp-23",          # Materials Project ID
    adsorbates_list=["*OH"],  # SMILES
    solvent=0,                # FAIRChem solvent DB id
    ions_list=[{"atoms": "Na"}],
    miller_indices=(1, 1, 1),
    solvent_density=1.0,
)
```

---

## Collective Variables

CV blocks in the TPS JSON config follow this schema:

```json
"CV": {
    "<cv_name>": {
        "type":   "<method_name>",
        "params": { "<key>": "<value_or_selection_string>" },
        "bound":   [min, max],
        "unbound": [min, max]
    }
}
```

Multiple CVs are combined with logical AND for the bound/unbound state definitions. A CV plot (`<cv_name>_initial_trajectory.png`) is saved automatically for each defined CV.

---

## Testing

The test suite uses `pytest` and runs without any simulation packages installed — all heavy dependencies are mocked in `conftest.py`.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run a specific module
pytest tests/test_cv.py -v
pytest tests/test_workflow.py -v
```

**Test coverage by module:**

| Module | Test file | What's covered |
|---|---|---|
| `workflow.py` | `test_workflow.py` | Config loading, state machine, stage skip/fail/block, generation loop, resume |
| `mlp.py` | `test_mlp.py` | Factory, config, train prepare, committee evaluation, MD, evaluate |
| `dft.py` | `test_dft.py` | Factory, reftraj validation, file layout, state round-trip, VASP stub |
| `util/cv.py` | `test_cv.py` | All CV static methods, atom selection, init path compat |
| `util/traj_util.py` | `test_traj_util.py` | Split, select, split_times, reproducibility, edge cases |
| `util/parser.py` | `test_parser.py` | read_config, print_config, create_parser, global_arguments |
| `util/mlp_helper.py` | `test_mlp_helper.py` | process_frame, local region selection, distance correctness |

---

## HPC / Cluster Workflow

MLPath uses **`PAUSE` sentinel files** as a hand-off mechanism with HPC job schedulers. The typical cycle looks like this:

```
1. python workflow.py --config workflow.json
   → Writes CP2K input directories, creates PAUSE files, exits.

2. Submit CP2K jobs to cluster (e.g. SLURM array job).
   → Each structure_{N}/cp2k.out is produced.

3. Remove PAUSE files (manually or via job completion hook):
   find single_reftraj -name PAUSE -delete

4. python workflow.py --config workflow.json
   → Parses CP2K outputs, writes extdata.xyz, trains MACE committee.
   → Creates PAUSE files in MLP training dirs, exits.

5. Submit MACE training jobs.
   → Remove PAUSE files when done.

6. python workflow.py --config workflow.json
   → Runs TPS, selects frames, advances to next generation.
   → Repeat from step 1.
```

Set `auto_confirm.remove_pause: true` in your workflow config to skip the interactive removal prompt in automated pipeline environments.

---

## Project Structure

```
mlpath/
├── workflow.py                 # Main orchestration entry point
├── mlp.py                      # MACE MLP wrapper
├── dft.py                      # CP2K / VASP DFT wrappers
├── tps.py                      # OpenPathSampling TPS setup
├── workflow_config_example.json
│
├── util/
│   ├── cv.py                   # Collective variable definitions
│   ├── traj_util.py            # Trajectory splitting and frame selection
│   ├── parser.py               # JSON config and argparse utilities
│   ├── storage_helper.py       # OPS storage → XYZ extraction
│   ├── initial_state.py        # FAIRChem initial structure builder
│   └── mlp_helper.py           # process_frame and local variance utilities
│
└── tests/
    ├── conftest.py             # Fixtures + heavy-dep mocks
    ├── test_workflow.py
    ├── test_mlp.py
    ├── test_dft.py
    ├── test_cv.py
    ├── test_traj_util.py
    ├── test_parser.py
    ├── test_mlp_helper.py
    └── test_traj_util.py
```

---

## Todo

### Core Functionality
- [ ] **VASP DFT engine** — implement `_prepare_structure`, `_run_calculation`, and `_parse_outputs` for VASP (POSCAR/OUTCAR parsing)
- [ ] **Quantum ESPRESSO engine** — add a third DFT backend to `DFTFactory`

### TPS
- [ ] **AIMMD support** - test AIIMD workflow
- [ ] **TIS support** — extend the workflow to support Transition Interface Sampling alongside TPS
- [ ] **WTE/AIMMD integration** — wire up the aimmd committor model training into the workflow stages

### MLP / Active Learning
- [ ] **Additional MLP backends** — add support for DEEPMD or others via the abstract `MLP` base
- [ ] **Query-by-committee improvement** — use better QbC methods.
- [ ] **Online learning mode** — update the MLP incrementally without full retraining each generation
- [ ] **Model evaluation report** — auto-generate energy/force parity plots and RMSE tables after each training run

### Workflow & Orchestration
- [ ] **SLURM job submission** — native integration with `sbatch` for DFT and MLP jobs (replacing manual PAUSE removal)

### Collective Variables

### Testing & CI
- [ ] **Integration test** — end-to-end workflow smoke test with CO2 protonation data
- [ ] **`test_storage_helper.py`** — unit tests for OPS storage extraction utilities
- [ ] **`test_initial_state.py`** — unit tests for `ReactionInterface` with mocked FAIRChem

### Documentation
- [ ] **Sphinx API docs** — auto-generate HTML docs from docstrings
- [ ] **Worked tutorial** — Jupyter notebook demonstrating a full generation cycle on alanine dipeptide
- [ ] **CV cookbook** — examples of all CV types in a reference notebook
- [ ] **HPC setup guide** — SLURM script templates and cluster-specific tips

### Performance & Robustness
- [ ] **Graceful OOM handling** — catch GPU out-of-memory errors during MLP training and retry with smaller batch size
- [ ] **Config schema validation** — use `pydantic` or `jsonschema` to validate config files at startup with helpful error messages
- [ ] **Logging improvements** — structured JSON logs for machine-readable pipeline monitoring

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Add tests for any new functionality — all tests must pass with no real GPU/DFT/MLP installed.
3. Follow the existing code style (type hints, NumPy-style docstrings, `loguru` for logging).
4. Open a pull request with a clear description of the change.

To run the full test suite locally:

```bash
pip install pytest pytest-cov
pytest --tb=short -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for computational chemistry on HPC clusters.<br>
  MACE · OpenPathSampling · CP2K · ASE · MDAnalysis
</p>