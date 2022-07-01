#!/bin/bash
#SBATCH --job-name=finetuning-1e-7
#SBATCH --account=bpms
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --time=04:00:00
##SBATCH --partition=debug
#SBATCH --output=./experiment/stdout/finetuning/finetuning_1e-7.out
#SBATCH --error=./experiment/stdout/finetuning/finetuning_1e-7.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run python script
python ./experiment/python_scripts/finetuning/finetuning_1e-7.py
