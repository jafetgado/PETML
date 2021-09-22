"""
Retrieve PETase-like sequences from databases with HMM
"""




import subprocess
import sys
sys.path.insert(1, './')
from module import utils




HMMSEARCH_EXEC = '/projects/bpms/jgado/hmmer-3.2.1/src/hmmsearch'




#================================================================#
# Build PETase-HMM from MSA of experimentally verified PETases
#================================================================#

# Align sequences
seqfile = './experiment/hmmsearch/data/petase-hmm/current_petases.fasta'
msafile = './experiment/hmmsearch/data/petase-hmm/current_petases_msa.fasta'
utils.mafft_MSA(seqfile, msafile)

# Build HMM
hmmfile = './experiment/hmmsearch/data/petase-hmm/hmm.txt'
_ = utils.hmmbuild_fxn(msafile, hmmfile)  # 61 seqs, 892 positions




#========================================================#
# Search PETase-HMM against sequence databases with HPC
#========================================================#

# Full path to databases
databases = {'ncbi': '/scratch/jgado/sequence_databases/ncbi_nr/nr.fasta',
             'jgi': '/scratch/jgado/sequence_databases/JGI_Saroj/'\
                     'combined_hotsprings_metagenome.fasta'}
for i in range(1,7):
    databases[f'mgnify{i}'] = '/scratch/jgado/sequence_databases/mgnify/fasta'\
                              f'/mgy_proteins_{i}.fa'


# Write bash scripts to implement hmmsearch with HPC
for key in databases.keys():
    
    bashscript = f'./experiment/hmmsearch/bash_scripts/{key}.sh'
    output_file = f'./experiment/hmmsearch/data/output/{key}_output.txt'
    tab_file = f'./experiment/hmmsearch/data/output/{key}_tabout.txt'
    aln_file = f'./experiment/hmmsearch/data/output/{key}_aln.sto'
    fasta_file = f'./experiment/hmmsearch/data/output/{key}_aln.fasta'
    full_length_file = f'./experiment/hmmsearch/data/output/{key}_full_length.fasta'
        
    with open(bashscript, 'w') as bash:
        bash.write(f'''#!/bin/bash
#SBATCH --job-name={key}
#SBATCH --account=bpms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=./experiment/hmmsearch/stdout/{key}.out
#SBATCH --error=./experiment/hmmsearch/stdout/{key}.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run hmmsearch
{HMMSEARCH_EXEC} \\
-o {output_file} \\
--tblout {tab_file} \\
-A {aln_file} --noali \\
-T 40 --domT 40 --incT 40 --incdomT 40 \\
{hmmfile} \\
{databases[key]}

# Convert stockholm to fasta
python -c "import sys;
sys.path.insert(1, './');
from module import utils;
aln_file = '{aln_file}';
fasta_file = '{fasta_file}';
utils.sto_to_fasta(aln_file, fasta_file)"
''')


    # Run hmmsearch job on HPC
    subprocess.call(f'sbatch {bashscript}', shell=True)