#!/bin/bash
#SBATCH --job-name=mgnify3
#SBATCH --account=bpms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --output=./experiment/stdout/mgnify/mgnify3.out
#SBATCH --error=./experiment/stdout/mgnify/mgnify3.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run hmmsearch
/projects/bpms/jgado/hmmer-3.2.1/src/hmmsearch \
-o ./experiment/data/hmmsearch/output/mgnify/mgnify3_output.txt \
--tblout ./experiment/data/hmmsearch/output/mgnify/mgnify3_tabout.txt \
-A ./experiment/data/hmmsearch/output/mgnify/mgnify3_aln.sto --noali \
-T 100 --domT 100 --incT 100 --incdomT 100 \
./experiment/data/hmmsearch/petase_hmm/hmm.txt \
/scratch/jgado/sequence_databases/mgnify/fasta/mgy_proteins_3.fa

# Convert stockholm to fasta
python -c "import sys;
sys.path.insert(1, './');
from module import utils;
aln_file = './experiment/data/hmmsearch/output/mgnify/mgnify3_aln.sto';
fasta_file = './experiment/data/hmmsearch/output/mgnify/mgnify3_aln.fasta';
utils.sto_to_fasta(aln_file, fasta_file)"
