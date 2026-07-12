# Tutorial: formic acid decomposition

A complete, real run of the active-learning loop (DFT -> MLP training -> TPS -> extension ->
committee/QBC frame selection -> next generation) on a system small enough to run on a laptop CPU in
minutes, chosen specifically because it has **two distinct, real reaction pathways** from the same
starting molecule - a good match for what this whole workbench is for: discovering reaction
mechanisms, not just fitting a potential.

Formic acid, HCOOH, has two well-documented competing unimolecular decomposition channels:

- **Dehydrogenation**: HCOOH -> CO2 + H2
- **Dehydration**: HCOOH -> CO + H2O

Non-periodic (a single small molecule, no slab/solvent), so this uses **PySCF** for DFT labeling
(`dft_backend: "pyscf"`) rather than CP2K - PySCF is a molecular-orbital code, plain `pip install
pyscf`, no external binary or job-submission step.

## Files

- `setup_system.py` - builds the HCOOH geometry, relaxes it with MACE-OFF23 (a pretrained organic-
  molecule foundation model), and bootstraps a seed reactive trajectory by running short high-T MD
  bursts until a spontaneous decomposition event is observed. Writes `top.pdb`, `tps.xyz`, `tps.json`.
- `make_initial_structures.py` - samples a diverse set of geometries along that seed trajectory
  (bound well, near the barrier, product) for generation-0 DFT labeling.
- `run_tutorial.py [channel]` - builds a `WorkflowConfig` (PySCF DFT + real MACE + real
  OpenPathSampling TPS) and runs `WorkflowEngine` for several generations. With no argument, uses the
  top-level `top.pdb`/`tps.xyz`/`tps.json` (dehydrogenation); `python run_tutorial.py dehydration`
  uses `dehydration/`'s files instead, writing to `run_dehydration/`.
- `analysis.ipynb` - loads *both* runs' output, plots the reaction-coordinate CV and committee (QBC)
  uncertainty for each, classifies which channel each explored frame belongs to, sanity-checks that
  classified frames come from smoothly-evolving trajectories rather than numerical artifacts, and
  summarizes what was found. Already executed against real runs - open it to see the results without
  re-running anything.
- `mace_train_config.yaml` - MACE hyperparameters (16 channels, 500 epochs - training still takes
  well under a minute per committee member).

`top.pdb`/`tps.xyz`/`tps.json` (dehydrogenation) and `dehydration/{top.pdb,tps.xyz,tps.json}`
(dehydration) are the two seeds this tutorial's own real runs actually used - both found by
`setup_system.py`'s high-T MD bootstrap search, at different temperatures/attempt counts (see below).

## Running it yourself

Real MLP training only needs plain pip installs:

```bash
pip install ase mace-torch pyscf
```

Real TPS needs OpenPathSampling, which (with its dependency openmm-ml) is conda-forge only - see the
main workbench README for how to set up an isolated conda env for this. Once that's done:

```bash
# only if you want to regenerate the seeds yourself - dehydrogenation tends to turn up within the
# first ~10-15 attempts around 2500 K; dehydration is rarer and needed ~40 attempts at 2800 K here
python setup_system.py --temperature 2500 --attempts 15
python make_initial_structures.py
python run_tutorial.py

python setup_system.py --temperature 2800 --attempts 40   # writes to dehydration/ (see setup_system.py)
python -c "from ase.io import read, write; import numpy as np; f=read('dehydration/tps.xyz', index=':'); i=np.unique(np.linspace(0,len(f)-1,40).astype(int)); write('dehydration/initial_structures.xyz', [f[j] for j in i])"
python run_tutorial.py dehydration

jupyter nbconvert --to notebook --execute analysis.ipynb  # or open it in Jupyter directly
```

Each run (4 generations, 4-member committee, ~40 PySCF single points/generation, real TPS with 40 MC
cycles/generation) takes on the order of 10-20 minutes on a laptop CPU.

## What this run actually found

**Both channels, for real.** The dehydrogenation-seeded run stays almost entirely in the bound well
plus dehydrogenation frames (24 out of ~24,700 total, across 4 generations) with only a single stray
"dehydration-labeled" frame - which the notebook's sanity-check cell shows is a literal one-step
hydrogen teleport (2.7 A to 1.0 A between consecutive MD steps), i.e. a numerical artifact from an
imprecise committee extrapolating badly, not a real event. That's worth stating plainly rather than
quietly dropping: a rare artifact like that is exactly the kind of thing a single flagged frame can
hide, which is why the notebook checks for it explicitly instead of taking classification counts at
face value.

The dehydration-seeded run (a *separately bootstrapped* seed trajectory - two-way shooting samples
locally around the path it's given, it doesn't search configuration space for a different product
channel on its own) found genuine, smoothly-evolving dehydration trajectories - 488 frames across 4
generations, confirmed by sub-0.4-Angstrom step-to-step changes in every accepted-and-extended
trajectory checked. Better still: the dehydration share of unbound frames climbs from 47% (gen 0) to
99% (gen 3) as the dedicated committee gets retrained generation over generation - real evidence of
the active-learning loop converging on that specific pathway, not just a lucky one-off crossing.

## Known rough edges

- MACE-OFF23 is licensed under the Academic Software License (non-commercial). It's used here only at
  setup time, to bootstrap a seed trajectory - not distributed with this repo, not used in the actual
  DFT/MLP/TPS loop itself (that uses PySCF-labeled data and a freshly-trained MACE committee).
- The bootstrap MD occasionally sends the system into complete, unphysical fragmentation shortly
  after a genuine crossing (the foundation model extrapolating badly into extreme geometries) -
  `setup_system.py` guards against saving a corrupted seed trajectory when that happens, but if you
  see "no channel found" across all attempts, try a different `--temperature` or more `--attempts`.
