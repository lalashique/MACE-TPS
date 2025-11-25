# --- Initial State Preparation ---
import fairchem.data.oc.core as oc
from ase import Atoms
from ase.io import write

class ReactionInterface(oc.InterfaceConfig):
    """
    Class to prepare the initial state for MD simulations in xyz format, including bulk, slab, solvent, and adsorbate. Look into
    the repo https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/data/oc/core for bulk_id and solvent_id.
    Adsorbtates and ions should be given as SMILES string for this implementation. Packmol is used to pack solvent and adsorbates around the slab,
    so make sure packmol is installed and in your PATH. 
    """
    def __init__(self, bulk_id, adsorbates_list=[None], solvent=0, ions_list=[], miller_indices=(1,0,0), solvent_density=1.0):
        bulk = oc.Bulk(bulk_src_id_from_db=bulk_id)
        slab = oc.Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller_indices)
        adsorbates = []
        for adsorbate_ind in adsorbates_list:
            adsorbate = oc.Adsorbate(adsorbate_smiles_from_db=adsorbate_ind)
            adsorbates.append(adsorbate)
        solvent = oc.Solvent(solvent_id_from_db=solvent, solvent_density=solvent_density)
        ions = []
        for ion in ions_list:
            ion_obj = oc.Ion(Atoms(ion['atoms']))
            ions.append(ion_obj)
        super().__init__(slab=slab[0], adsorbates=adsorbates, solvent=solvent, ions=ions, num_sites=1)

# For testing purposes
if __name__ == "__main__":
    structure = ReactionInterface(bulk_id="mp-23")
    write("initial_structure.xyz", structure.atoms_list[0], format="xyz")
