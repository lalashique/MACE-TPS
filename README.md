# MACE-TPS: Machine Learning-Enhanced Transition Path Sampling

A comprehensive framework for efficient rare event sampling using machine learning potentials trained using transition path sampling (TPS). This project integrates MACE (Machine Learning Interatomic Potential) models with OpenPathSampling to explore complex molecular transitions.

## Overview

MACE-TPS combines:
- **MACE Models**: Machine learning interatomic potentials for fast force field evaluations
- **Transition Path Sampling**: Statistical method for exploring rare events in molecular dynamics
- **DFT Integration**: Density functional theory calculations for reference data
- **AIMMD**: Adaptive importance sampling molecular dynamics for enhanced sampling
- **Automated Workflow**: End-to-end pipeline for system setup, sampling, and analysis

## Project Structure

```
MACE-TPS/
├── main.py                 # Entry point for TPS simulations
├── mlp_path/               # Core simulation modules
│   ├── tps.py              # Transition path sampling implementation
│   ├── mlp.py              # Machine learning potential setup
│   ├── dft.py              # DFT calculation interface
│   ├── workflow.py         # Automated workflow orchestration
│   ├── AIMMD_utils.py      # AIMMD utilities
│   ├── util/               # Helper utilities
│   │   ├── cv.py           # Collective variables
│   │   ├── parser.py       # Configuration parsing
│   │   ├── mlp_helper.py   # MLP helper functions
│   │   ├── initial_state.py # Initial state preparation
│   │   ├── traj_util.py    # Trajectory utilities
│   │   └── storage_helper.py # Storage management
│   └── test/               # Test data and examples
│       ├── mlp/            # MLP test cases
│       ├── dft/            # DFT test cases
│       ├── tps/            # TPS test cases
│       └── workflow/       # Workflow examples
├── example/                # Example implementations
│   ├── v1/                 # Version 1 example
│   └── v2/                 # Version 2 example (AIMMD-ready)
└── README.md               # This file
```

## Key Components

### `mlp_path/tps.py`
Core TPS implementation using OpenPathSampling. Handles:
- Path ensemble definitions
- Shooting point selection and sampling
- Order parameter tracking
- Trajectory storage and analysis

**Key Dependencies:**
- OpenPathSampling
- OpenMM for MD engine
- PyTorch for ML models
- ASE for molecular structure handling
- MDAnalysis for trajectory analysis

### `mlp_path/mlp.py`
Machine learning potential setup:
- MACE model loading and evaluation
- Force/energy calculations
- Integration with OpenMM

### `mlp_path/dft.py`
DFT calculation interface:
- CP2K integration
- Single point energy/force calculations
- Geometry optimization

### `mlp_path/workflow.py`
Automated workflow management:
- Multi-stage execution (DFT → MLP Training → TPS)
- Job submission and monitoring
- Directory structure management
- Configuration handling

### `mlp_path/util/`
Utility modules:
- **cv.py**: Collective variable definitions for path distinction
- **parser.py**: Configuration file parsing
- **mlp_helper.py**: MLP-specific utilities
- **traj_util.py**: Trajectory manipulation and analysis
- **initial_state.py**: Initial state/structure preparation

## Configuration

Simulations are configured using JSON files with the following structure:

**TODO**

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch
- OpenPathSampling
- OpenMM/OpenMMTools
- MACE models
- CP2K (for DFT calculations)
- MDAnalysis
- ASE (Atomic Simulation Environment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MACE-TPS

# Create conda environment
conda env create -f mlp_path/mace_tps_aimmd_ready.yml

# Install in development mode
pip install -e .
```

## Usage

### Basic TPS Run

```bash
python main.py -dir "." -cfg "config.json" -nr 1000 -traj "p1.xyz"
```

### AIMMD-Enhanced TPS

```bash
cd mlp_path
python run_AIMMD_TPS.py -cfg config.json -traj initial_traj.xyz
```

### Examples

See `example/` directory for complete examples:
- **v1/**: Basic TPS setup with collective variables
- **v2/**: AIMMD-integrated workflow with automated configuration

Run examples:
```bash
cd example/v2
python run.py
```

## Workflow Stages

The automated workflow (`workflow.py`) executes stages in sequence:

1. **DFT Stage**: Calculate reference data using DFT
2. **MLP Training Stage**: Train MACE models on DFT data
3. **TPS Sampling Stage**: Run transition path sampling with MLP
4. **Analysis Stage**: Post-process trajectories and compute statistics

Each stage is configured independently and can be run separately or as a full pipeline.

## Test Suite

Test data is provided in `mlp_path/test/`:

# TODO #
```bash
# Test MLP setup
python -m pytest mlp_path/test/mlp/

# Test DFT interface
python -m pytest mlp_path/test/dft/

# Test TPS configuration
python -m pytest mlp_path/test/tps/

# Test full workflow
python mlp_path/test/workflow/run_test_workflow.py
```

## Output

Simulation results are stored in a hierarchical directory structure under `init/` and numbered trajectory directories (`0/`, `1/`, etc.):

```
workflow_name/
├── init/
│   ├── coord.xyz           # Initial structure
│   └── CV.json             # Initial CV values
├── tps.json                # TPS configuration
├── dft.json                # DFT configuration
├── mlp.json                # MLP configuration
├── 0/
│   ├── tps/
│   │     └── traj/
│   │          └── accepted_*.xyz  # Accepted trajectories
│   ├── dft/                # DFT calculations
│   └── mlp/                # MLP evaluations
├── 1/
│   └── ...
└── analysis/               # Analysis results
```

## Key Features

- **Efficient Sampling**: Combines TPS with ML potentials for faster exploration
- **Accuracy**: DFT reference calculations for model training
- **Scalability**: Committee ensemble of ML models for uncertainty quantification
- **Automation**: Full workflow management from structure to analysis
- **Flexibility**: Modular design allows custom CVs, engines, and analysis

## Contributing

1. Ensure code follows PEP 8 style guidelines
2. Add tests for new functionality
3. Update documentation for API changes
4. Submit pull requests with clear descriptions

## References

- **OpenPathSampling**: [openpathsampling.org](https://openpathsampling.org/)
- **MACE Models**: Batatia et al., [arXiv:2206.07697](https://arxiv.org/abs/2206.07697)
- **TPS Theory**: Bolhuis et al., Annual Review of Physical Chemistry, 2002

## License

[Specify your license here]

## Contact

For questions or issues, please refer to the project repository.
