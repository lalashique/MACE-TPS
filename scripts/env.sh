# Environment for running the workflow from a repo checkout (bash/zsh). Usage: source scripts/env.sh
# Set PY to the python of an env that has torch, mace, openmm(+ml), openpathsampling, ase, plumed, click.
if [ -n "$ZSH_VERSION" ]; then _here="${${(%):-%x}:A:h}"; else _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
export REPO="$(cd "$_here/.." && pwd)"
export PY="${PY:-python}"
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS: two libomp copies make `import torch` abort otherwise
export PYTHONPATH=$REPO/packages/core:$REPO/packages/dft:$REPO/packages/mlp:$REPO/packages/tps:$REPO/packages/workflow:$REPO/packages/cli${PYTHONPATH:+:$PYTHONPATH}
