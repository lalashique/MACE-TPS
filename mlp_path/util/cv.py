

import numpy as np

import openpathsampling as paths
import matplotlib.pyplot as plt
import openpathsampling.engines.openmm as ops_openmm

from MDAnalysis.lib.distances import distance_array
import MDAnalysis as mda
from .parser import read_config
import mdtraj
from loguru import logger

class CV:
    def __init__(self,config,traj_path="p1.xyz"):
        logger.debug("Defining states and interfaces...")
        self.config = config #read_config(config_path)
        cv_config = self.config.get('CV')

        logger.debug(f"Reading top file")
        mdt = mdtraj.load(self.config.get("top_file","top.pdb"))
        traj = mdtraj.load(traj_path,top=mdt)
        traj.unitcell_vectors = np.array(list(mdt.unitcell_vectors)*len(traj.xyz))
        init_traj = ops_openmm.trajectory_from_mdtraj(traj,velocities=[np.zeros_like(frame) for frame in traj.xyz])

        cvs = []

        logger.debug(f"Defining CVs from config: {list(cv_config.keys())}")

        for cv in cv_config.keys():
            func = getattr(self,cv_config[cv]['type'])

            for param in cv_config[cv]['params']:
                # print( type(cv_config[cv]['params'][param]) )

                if type(cv_config[cv]['params'][param]) is str:
                    file = traj_path
                    h = mda.Universe(file).select_atoms(cv_config[cv]['params'][param]).indices
                    
                    cv_config[cv]['params'][param] = h
            # print(f"Defining CV: {cv} with params {cv_config[cv]['params']}")
            cv_b= paths.CoordinateFunctionCV(name=cv,
                f=func,
                **cv_config[cv]['params']
            ).with_diskcache()
            cvs.append(cv_b)
            fig, ax = plt.subplots(1,1)
            ax.plot(cv_b(init_traj))
            ax.set_xlabel("frames")
            ax.set_ylabel(f"CV {cv}")
            plt.savefig(f"{cv}_initial_trajectory.png", dpi=300)
            plt.close()
        def descriptor_transform(s, cvs):
            import numpy as np
            return np.array([cv(s) for cv in cvs])
        bound =[]
        unbound =[]
        for i,cv in enumerate(cvs):
            if cv_config[cv.name].get('bound') is not None:
                print("Defining bound for CV:", cv.name)
                print(cv_config[cv.name]['bound'])
                bound_cv = paths.CVDefinedVolume(cv, lambda_min=cv_config[cv.name]['bound'][0], lambda_max=cv_config[cv.name]['bound'][1]).named(f'bound_{cv.name}')
                bound.append(bound_cv)

            if cv_config[cv.name].get('unbound') is not None:
                print("Defining unbound for CV:", cv.name)
                print(cv_config[cv.name]['unbound'])
                unbound_cv = paths.CVDefinedVolume(cv, lambda_min=cv_config[cv.name]['unbound'][0], lambda_max=cv_config[cv.name]['unbound'][1]).named(f'unbound_{cv.name}')    
                unbound.append(unbound_cv)

        print(len(bound), len(unbound))
        for i in range(len(bound)-1):
            bound[0]= bound[0] & bound[i+1]
        for i in range(len(unbound)-1):
            unbound[0]= unbound[0] & unbound[i+1]

        self.states = (bound[0],unbound[0])

    @staticmethod
    def bondfunction(snapshot, group_A_atoms, group_B_atoms,group_C_atoms, group_D_atoms,coeff=-1):
        import numpy as np

        A_com = snapshot.xyz[group_A_atoms, :]
        B_com = snapshot.xyz[group_B_atoms, :]
        C_com = snapshot.xyz[group_C_atoms, :]
        D_com = snapshot.xyz[group_D_atoms, :]

        ab = A_com - B_com
        cd = C_com - D_com



        # Calculate the angle between the two vectors
        angle = np.linalg.norm(ab) + coeff * np.linalg.norm(cd)
        return angle

    @staticmethod
    def bond(snapshot, group_A_atoms, group_B_atoms,type=0,axis=-1): 
        import numpy as np

        A_com = snapshot.xyz[group_A_atoms, :]
        B_com = snapshot.xyz[group_B_atoms, :]

        # # make sure A_com and B_com are 2d arrays
        A_com = np.atleast_2d(A_com)
        B_com = np.atleast_2d(B_com)

        if type==0:
            A_com = np.mean(A_com, axis=0)
            B_com = np.mean(B_com, axis=0)
            ab = A_com - B_com
            if axis >= 0:
                angle = ab[axis]
            else:
                angle = np.linalg.norm(ab)
            return angle

        elif type==-1:
            A_com = np.reshape(A_com, ( -1, 3))
            B_com = np.reshape(B_com, ( -1, 3))
            dists = distance_array(A_com, B_com)
            if axis >= 0:
                return np.min(dists[:, axis])
            else:
                return np.min(dists)
        elif type==1:
            A_com = np.reshape(A_com, ( -1, 3))
            B_com = np.reshape(B_com, ( -1, 3))
            dists = distance_array(A_com, B_com)
            if axis >= 0:
                return np.max(dists[:, axis])
            else:   
                return np.max(dists)
        else:
            raise ValueError(f"Unknown bond type: {type}")
        
    @staticmethod
    def CN(snapshot, group_A_atoms, group_B_atoms, d0=1.25, n=12, m=24):
        d0 = d0/10  # in nm
        o1 = snapshot.xyz[group_A_atoms, :]
        h = snapshot.xyz[group_B_atoms, :]

        dist = distance_array(o1, h)


        s_ij = (1 - (dist / d0)**n) / (1 - (dist / d0)**m)

        cn = np.sum(s_ij)
        return cn
    
    @staticmethod
    def bond_count(snapshot, group_A_atoms, group_B_atoms,value=1.5,rounding=2):
        import numpy as np

        A_com = snapshot.xyz[group_A_atoms, :]
        B_com = snapshot.xyz[group_B_atoms, :]

        # # make sure A_com and B_com are 2d arrays
        A_com = np.atleast_2d(A_com)
        B_com = np.atleast_2d(B_com)

        A_com = np.reshape(A_com, ( -1, 3))
        B_com = np.reshape(B_com, ( -1, 3))
        dists = distance_array(A_com, B_com)
        count = np.sum(np.round(dists,rounding) == value)
        return count

    # ! TO BE DELETED : already implemented in bond function with axis=2
    @staticmethod
    def atom_surface(snapshot, group_A_atoms, group_B_atoms): 
        import numpy as np

        A_com = snapshot.xyz[group_A_atoms, :]
        B_com = snapshot.xyz[group_B_atoms, :]

        # # make sure A_com and B_com are 2d arrays
        A_com = np.atleast_2d(A_com)
        B_com = np.atleast_2d(B_com)


        A_com = np.mean(A_com[:,2], axis=0)
        B_com = np.mean(B_com[:,2], axis=0)
        ab = A_com - B_com
        # Calculate the angle between the two vectors
        angle = np.linalg.norm(ab)
        return angle
    
    @staticmethod
    def bond_plane_angle(snapshot, group_A_atoms, group_B_atoms,axis=2):
        import numpy as np

        A_com = snapshot.xyz[group_A_atoms, :]
        B_com = snapshot.xyz[group_B_atoms, :]

        # # make sure A_com and B_com are 2d arrays
        A_com = np.atleast_2d(A_com)
        B_com = np.atleast_2d(B_com)

        A_com = np.mean(A_com, axis=0)
        B_com = np.mean(B_com, axis=0)
        ab = A_com - B_com
        angle = np.degrees(np.arccos(ab[axis]/np.linalg.norm(ab)))-90
        return angle

    @staticmethod
    def distance_to_OH(snapshot,O_indices,H_indices, atom, bond_cutoff=1.3):
        from MDAnalysis.lib.distances import capped_distance
        import numpy as np

        """
        If 'atom' is the distance to be calculated from,
        return distance to the other OH oxygen.
        Otherwise return np.inf.
        """

        pairs, _ = capped_distance(
            snapshot.xyz[0, O_indices, :],
            snapshot.xyz[0, H_indices, :],
            max_cutoff=bond_cutoff/10,
            box=np.array([14.46, 14.46, 40,90,90,90]) # ! URGENT: Update to take from topology
        )

        if len(pairs) == 0:
            return -1

        O_indice_new, h_counts = np.unique(pairs[:, 0], return_counts=True)

        hydroxide_mask = (h_counts == 1)
        oh_group_indices = O_indice_new[hydroxide_mask]

        if len(oh_group_indices) < 2:
            return np.inf
        
        oh_oxygens_indices = O_indices[oh_group_indices]
        oh_oxygens = snapshot.xyz[0, oh_oxygens_indices, :]

        # Get the other OH oxygen
        other_O = oh_oxygens[oh_oxygens_indices != atom]

        if len(other_O) == 0:
            return np.inf

        dist = np.linalg.norm(snapshot.xyz[0][atom] - other_O[0])
        return dist