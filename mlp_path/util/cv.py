

import numpy as np

import openpathsampling as paths
import matplotlib.pyplot as plt
import openpathsampling.engines.openmm as ops_openmm

from MDAnalysis.lib.distances import distance_array
import MDAnalysis as mda
from .parser import read_config
import mdtraj

class CV:
    def __init__(self,config_path,traj_path="p1.xyz"):
        print("Defining states and interfaces...")
        self.config = read_config(config_path)
        cv_config = self.config.get('CV')
        mdt = mdtraj.load('../top.pdb')
        traj = mdtraj.load(traj_path,top=mdt)
        traj.unitcell_vectors = np.array(list(mdt.unitcell_vectors)*len(traj.xyz))
        init_traj = ops_openmm.trajectory_from_mdtraj(traj,velocities=[np.zeros_like(frame) for frame in traj.xyz])

        cvs = []



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
            ax.set_ylabel("CV B3")
            plt.savefig(f"{cv}_initial_trajectory.png", dpi=300)
            plt.close()

        bound =[]
        unbound =[]
        for i,cv in enumerate(cvs):
            bound_cv = paths.CVDefinedVolume(cv, lambda_min=cv_config[cv.name]['bound'][0], lambda_max=cv_config[cv.name]['bound'][1]).named(f'bound_{cv.name}')
            unbound_cv = paths.CVDefinedVolume(cv, lambda_min=cv_config[cv.name]['unbound'][0], lambda_max=cv_config[cv.name]['unbound'][1]).named(f'unbound_{cv.name}')
            bound.append(bound_cv)
            unbound.append(unbound_cv)

        # bound2 = paths.CVDefinedVolume(cv_b2, lambda_min=0.13, lambda_max=1).named('bound2')
        # unbound2 = paths.CVDefinedVolume(cv_b2, lambda_min=-1, lambda_max=-0.2).named('unbound2')
        # print(bound2,unbound2)
        for i in range(len(bound)-1):
            bound[0]= bound[0] & bound[i+1]
            unbound[0]= unbound[0] & unbound[i+1]
        # bound = bound2
        # unbound = unbound2
        self.states = (bound[0],unbound[0])
        # return (bound[0],unbound[0])

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
    def bond(snapshot, group_A_atoms, group_B_atoms,):
        import numpy as np

        A_com = snapshot.xyz[group_A_atoms, :]
        B_com = snapshot.xyz[group_B_atoms, :]

        ab = A_com - B_com

        # Calculate the angle between the two vectors
        angle = np.linalg.norm(ab)
        return angle
    @staticmethod
    def CN(snapshot, group_A_atoms, group_B_atoms, d0=1.25, n=12, m=24):
        d0 = d0/10  # in nm
        o1 = snapshot.xyz[group_A_atoms, :]
        h = snapshot.xyz[group_B_atoms, :]

        dist = distance_array(o1, h)


        s_ij = (1 - (dist / d0)**n) / (1 - (dist / d0)**m)

        cn = np.sum(s_ij)
        return cn


