#!/bin/bash
#SBATCH --job-name=mace_ops_test         
#SBATCH --output=out.txt                  
#SBATCH --error=err.txt                   
#SBATCH --partition=gpuRTX-1st,gpuRTX-2nd,gpuRTX-3rd,gpuRTX-4th            
##SBATCH --nodelist=gn1,gn2,gn3              
#SBATCH --ntasks=1                        
#SBATCH --gpus-per-node=1                 
#SBATCH --gpu-bind=single:1
#SBATCH --time=7-00:00:00                 

# initialize conda
# Base directories and configurations
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mlp-tps2

python run_AIMMD_TPS.py -dir "." -cfg "config.json" -nr 100 -traj "p1.xyz" -aimmd_store "aimmd_model_storage.h5"

echo "Job completed!"
