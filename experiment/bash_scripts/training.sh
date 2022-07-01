#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=bpms
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=./experiment/stdout/training.out
#SBATCH --error=./experiment/stdout/training.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run python script
python ./experiment/python_scripts/training.py