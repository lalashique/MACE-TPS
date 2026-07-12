"""PySCF-based DFT backend for non-periodic (molecular) systems.

Unlike CP2K, PySCF is a molecular-orbital code with no natural notion of a
periodic cell, and it's pure pip-installable with no external binary or
job-submission step - so this backend runs fully in-process and
synchronously: no NeedsExternalJob, no prepare/run split, just a direct
calculation per structure. This makes it the simplest DFT backend to get
running, and the natural fit for a single small molecule (as opposed to a
periodic slab/solvated system, which is what CP2K is for).
"""

from __future__ import annotations

from pathlib import Path

from core.interfaces import DFTEngine

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class PySCFDFTEngine(DFTEngine):
    def __init__(
        self,
        basis: str = "6-31g",
        xc: str = "b3lyp",
        charge: int = 0,
        spin: int = 0,
        energy_key: str = "energy_cp2k",
        forces_key: str = "forces_cp2k",
    ):
        """xc: any PySCF-supported functional name. Set xc=None to run plain
        Hartree-Fock instead of DFT (faster, less accurate - fine for
        bootstrapping/testing, per the same "doesn't need to be accurate to
        be useful" spirit as the mock/foundation-model bootstrap elsewhere
        in this workbench).
        spin: 2S, i.e. number of unpaired electrons (0 for closed-shell).
        energy_key/forces_key match whatever the MLP backend's training
        config expects (see mlp/legacy_mace.py's DEFAULT_CONFIG) - kept
        consistent with the CP2K backend's default key names so DFT
        backends are interchangeable without touching MLP config.
        """
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.energy_key = energy_key
        self.forces_key = forces_key

    def label(self, structures: Path, workdir: Path) -> Path:
        from ase.io import read, write
        from pyscf import gto, dft, scf

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        frames = read(structures, index=":")
        labeled = []
        for atoms in frames:
            mol = gto.M(
                atom=list(zip(atoms.get_chemical_symbols(), atoms.get_positions())),
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit="Angstrom",
                verbose=0,
            )
            if self.xc:
                mf = dft.UKS(mol) if self.spin else dft.RKS(mol)
                mf.xc = self.xc
            else:
                mf = scf.UHF(mol) if self.spin else scf.RHF(mol)

            energy_hartree = mf.kernel()
            forces_ev_per_ang = -mf.nuc_grad_method().kernel() * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM

            atoms = atoms.copy()
            atoms.info[self.energy_key] = float(energy_hartree) * HARTREE_TO_EV
            atoms.arrays[self.forces_key] = forces_ev_per_ang
            labeled.append(atoms)

        out_path = workdir / "labeled.xyz"
        write(str(out_path), labeled, format="extxyz")
        return out_path
