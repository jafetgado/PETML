#!/bin/bash
#SBATCH --job-name=hyperparameter_tuning
#SBATCH --account=bpms
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --qos=high
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --output=./experiment/stdout/hyperparameter_tuning.out
#SBATCH --error=./experiment/stdout/hyperparameter_tuning.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run python script
python ./experiment/python_scripts/hyperparameter_tuning.py