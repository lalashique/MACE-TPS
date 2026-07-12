# Creating a new tutorial system

A worked example: [tutorials/formic_acid_decomposition/](../tutorials/formic_acid_decomposition/) -
read that folder's own README first if you want a concrete reference to follow alongside this guide.

This document is the general recipe for setting up *any* new system: what files a tutorial needs,
how to design a reaction-progress collective variable (CV), how to bootstrap a seed trajectory without
needing expensive ab initio MD, and how to structure the analysis notebook. It applies to molecules
(PySCF DFT backend) and periodic/slab systems (CP2K DFT backend) alike - the workflow engine and TPS
machinery don't care which.

## 1. Pick a system with something to discover

The best tutorial systems have **more than one plausible reaction pathway** from the same starting
point - that's what this workbench's active-learning loop (committee uncertainty -> targeted DFT
labeling -> retraining -> more TPS) is actually for. A system with only one accessible mechanism still
exercises the pipeline, but doesn't demonstrate the "discover multiple mechanisms" story.

Practical constraints for a *quick* tutorial (not a production campaign):
- Small (a handful of atoms for a molecule; a small slab + a few adsorbates/solvent molecules for a
  periodic system) - DFT labeling cost and MD step cost both scale with atom count.
- A real, chemically identifiable reaction, not a made-up potential - lets you sanity-check results
  against literature.
- Low enough real barriers, or a willingness to run at an elevated (clearly labeled as non-physical)
  temperature, that a handful of MC cycles can actually produce transitions.

## 2. Pick a DFT backend

- **Non-periodic** (a single molecule, no slab/solvent): use `dft_backend: "pyscf"` - PySCF is a
  molecular-orbital code, pure `pip install pyscf`, no external binary, runs in-process and
  synchronously (no job-submission step).
- **Periodic** (a slab, a solvated interface): use `dft_backend: "cp2k"` - needs a real CP2K install
  (see the main [README](../README.md)'s setup section) and drives it as an external process, so its
  `label()` calls raise `NeedsExternalJob` until you've actually run CP2K.

Either way, pick `E0s` (isolated-atom reference energies) at the *same level of theory* you're using
for labeling. Getting this wrong (e.g. leaving it at 0) makes MACE fit raw total energies (thousands
of eV, dominated by chemically-irrelevant core electrons) instead of the small atomization-energy
variations that actually distinguish reactant from products - training will look like it's converging
on the loss curve while producing a useless model. Compute isolated-atom energies once, at your
chosen basis/functional, and hardcode them into your `mlp_settings["E0s"]`.

## 3. Design a reaction-progress CV

`core`'s CV machinery (via `tps/util/cv.py`) already implements a `bond` CV type with several useful
modes - you likely don't need a new CV function, just the right parameters:

- `type=1` (max distance) over a list of atom-pair indices gives you "has *any* of these bonds
  broken" - useful as a single combined bound/unbound indicator when multiple different bonds could
  be the one that breaks (e.g. two competing reaction channels breaking different bonds from the same
  reactant). Pass the shared atom index twice if you need to pair it against two different partners
  (see the formic acid tutorial's `group_A_atoms: [1, 1], group_B_atoms: [2, 3]` for exactly this).
- `type=-1` (min distance) is the complementary "closest approach" mode.
- `CN` (coordination number, a smooth switching function) is useful when "bonded or not" is ambiguous
  near the cutoff and you want a differentiable proxy instead of a hard distance threshold.

Set `bound`/`unbound` ranges with real headroom - don't place the boundary right where you expect
normal thermal fluctuation to reach; check this empirically (plot the CV along a short trial MD run
before committing to thresholds) rather than guessing.

**Distinguishing *which* mechanism occurred is a job for post-hoc analysis, not the CV.** TPS only
needs "intact" vs. "decomposed" to sample paths; the analysis notebook classifies specific mechanisms
from full geometry after the fact (see the formic acid tutorial's `analysis.ipynb` for the pattern:
classify by which *new* bond has formed, not just which old bond broke - concerted/asynchronous
transitions can have the new bond form slightly before the old one fully separates).

## 4. Bootstrap a seed trajectory

TPS needs a valid reactant -> product path handed to it before two-way shooting can start sampling
paths near it - it does not discover the first crossing on its own. Three ways to get one, cheapest
first:

1. **Foundation-model-driven high-T MD** (recommended for molecules): relax your system with a
   pretrained model (e.g. MACE-OFF for organic molecules, MACE-MP for inorganic/materials systems -
   both free via `mace.calculators`), then run short high-temperature MD bursts until you observe a
   spontaneous transition. No training data needed. See
   `tutorials/formic_acid_decomposition/setup_system.py` for a complete, reusable implementation of
   this, including two things you'll likely also need:
   - **A blow-up guard.** Foundation models are approximate and can occasionally send a shooting
     trajectory into complete, unphysical fragmentation shortly after a genuine crossing. Don't save
     a seed trajectory that captures that - discard the attempt and retry.
   - **A short "tail" after detection**, not a long one. You want the seed to show a real, settled
     excursion into the product state, not just graze the boundary - but a long tail risks capturing
     the instability above instead. A few hundred steps is usually enough.
   - Expect to need **many attempts, and possibly a higher temperature or more attempts for rarer
     channels** - a system with two pathways of different barrier height will find the lower-barrier
     one far more readily. If you want a specific rarer channel, run the search again with a higher
     temperature and a larger attempt budget specifically targeting it, rather than assuming one
     search will find everything.
2. **Steered/constrained MD**: pull the system from reactant toward a guessed product along your CV,
   using whatever potential you have (foundation model or a partially-trained committee).
3. **Geometric interpolation** (linear or an actual path-interpolation method): a last resort - only
   as good as your guessed product geometry, and doesn't reflect real dynamics at all, but sometimes
   sufficient just to bootstrap the very first generation before any MD-based data exists.

**A note on reproducibility**: if you use ASE's `Langevin` (or similar stochastic integrators), seeding
the initial-velocity distribution does *not* make the whole trajectory reproducible - the integrator
itself injects its own unseeded randomness at every step from ASE's global random state. Don't rely on
"same seed -> same trajectory" when re-running a search.

## 5. Write the workflow config

See `tutorials/formic_acid_decomposition/run_tutorial.py` for a complete example building a
`WorkflowConfig` (DFT/MLP/TPS backend choice, committee size, generations, gates) and driving
`WorkflowEngine.run_to_completion()`. Scale-up knobs, roughly in order of what to increase first if
your system isn't finding a mechanism robustly:
- `committee_seeds` (more members -> better uncertainty estimates, catches more of the phase space)
- `tps.json`'s `tot_mc_steps` / `extend_steps` (more shooting attempts, longer extensions)
- `max_generations` (more retraining cycles - the mechanism-identification signal to watch for is a
  pathway's share of "unbound" frames *growing* over generations as its dedicated data accumulates,
  not just whether it appears at all)
- `tps.json`'s `Integrator_settings.temperature` (higher -> more transitions, but also more risk of
  unphysical excursions with an undertrained committee - watch for it)

## 6. Write the analysis notebook

Load every generation's `tps_all_extended.xyz` (the full pool of shooting-trial frames, already
carrying committee energy/force variance from `evaluate_committee`), plot the CV over "visitation
order" to see where sampling actually went, classify frames into mechanisms, and **sanity-check
suspicious findings** rather than taking classification counts at face value - a single
mechanism-classified frame with no smooth history behind it (check consecutive-frame distance jumps
within one accepted/rejected trajectory file) is more likely a numerical artifact from an approximate
potential than a real event. See `tutorials/formic_acid_decomposition/analysis.ipynb`'s "sanity check"
cell for the pattern; it exists because exactly this happened once during that tutorial's development.
