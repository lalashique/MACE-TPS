"""
util/initial_state.py — Initial structure preparation for MD simulations.

Builds periodic slab/solvent/adsorbate systems using the FAIRChem Open Catalyst
library (``fairchem.data.oc.core``).  Packmol must be installed and on PATH.

Typical usage
-------------
    from util.initial_state import ReactionInterface

    interface = ReactionInterface(
        bulk_id="mp-23",
        adsorbates_list=["*OH"],
        solvent=0,
        ions_list=[{"atoms": "Na"}],
        miller_indices=(1, 1, 1),
        solvent_density=1.0,
    )
    from ase.io import write
    write("initial_structure.xyz", interface.atoms_list[0])
"""

from __future__ import annotations

import fairchem.data.oc.core as oc
from ase import Atoms
from ase.io import write
from loguru import logger


class ReactionInterface(oc.InterfaceConfig):
    """
    Prepare a periodic interface structure for MD simulations.

    Constructs a slab + solvent + adsorbate system using the FAIRChem Open
    Catalyst dataset utilities.  The resulting ``atoms_list`` attribute holds
    one or more ``ase.Atoms`` objects ready for use as initial configurations.

    Parameters
    ----------
    bulk_id : str
        Materials Project bulk ID (e.g. ``"mp-23"`` for Pt).
    adsorbates_list : list[str | None]
        SMILES strings for adsorbates, or ``[None]`` for a bare slab.
    solvent : int
        FAIRChem solvent database ID.
    ions_list : list[dict]
        Ion specifications.  Each dict should have an ``"atoms"`` key with a
        chemical formula string (e.g. ``{"atoms": "Na"}``).
    miller_indices : tuple[int, int, int]
        Miller indices of the surface slab.
    solvent_density : float
        Solvent density in g/cm³ (default 1.0).
    """

    def __init__(
        self,
        bulk_id: str,
        adsorbates_list: list[str | None] | None = None,
        solvent: int = 0,
        ions_list: list[dict] | None = None,
        miller_indices: tuple[int, int, int] = (1, 0, 0),
        solvent_density: float = 1.0,
    ) -> None:
        if adsorbates_list is None:
            adsorbates_list = [None]
        if ions_list is None:
            ions_list = []

        logger.info(f"Building bulk from ID: {bulk_id}")
        bulk = oc.Bulk(bulk_src_id_from_db=bulk_id)
        slab = oc.Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller_indices)
        logger.info(f"Slab created with Miller indices {miller_indices}.")

        adsorbates = []
        for smiles in adsorbates_list:
            if smiles is not None:
                adsorbates.append(oc.Adsorbate(adsorbate_smiles_from_db=smiles))
                logger.debug(f"Added adsorbate: {smiles}")

        solvent_obj = oc.Solvent(solvent_id_from_db=solvent, solvent_density=solvent_density)

        ions = []
        for ion_spec in ions_list:
            ions.append(oc.Ion(Atoms(ion_spec["atoms"])))
            logger.debug(f"Added ion: {ion_spec['atoms']}")

        super().__init__(
            slab=slab[0],
            adsorbates=adsorbates if adsorbates else [None],
            solvent=solvent_obj,
            ions=ions,
            num_sites=1,
        )
        logger.info(f"ReactionInterface built with {len(self.atoms_list)} configuration(s).")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    structure = ReactionInterface(bulk_id="mp-23")
    write("initial_structure.xyz", structure.atoms_list[0], format="xyz")
    logger.info("Written initial_structure.xyz")
