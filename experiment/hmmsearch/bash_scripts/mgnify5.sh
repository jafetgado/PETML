#!/bin/bash
#SBATCH --job-name=mgnify5
#SBATCH --account=bpms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=./experiment/hmmsearch/stdout/mgnify5.out
#SBATCH --error=./experiment/hmmsearch/stdout/mgnify5.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run hmmsearch
#/projects/bpms/jgado/hmmer-3.2.1/src/hmmsearch \
#-o ./experiment/hmmsearch/data/output/mgnify5_output.txt \
#--tblout ./experiment/hmmsearch/data/output/mgnify5_tabout.txt \
#-A ./experiment/hmmsearch/data/output/mgnify5_aln.sto --noali \
#-T 40 --domT 40 --incT 40 --incdomT 40 \
#./experiment/hmmsearch/data/petase-hmm/hmm.txt \
#/scratch/jgado/sequence_databases/mgnify/fasta/mgy_proteins_5.fa

# Convert stockholm to fasta
python -c "import sys;
sys.path.insert(1, './');
from module import utils;
aln_file = './experiment/hmmsearch/data/output/mgnify5_aln.sto';
fasta_file = './experiment/hmmsearch/data/output/mgnify5_aln.fasta';
utils.sto_to_fasta(aln_file, fasta_file)"
