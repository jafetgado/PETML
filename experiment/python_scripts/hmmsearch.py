"""
Retrieve PETase-like sequences from databases with HMM
"""




#============#
# Imports
#============#

import subprocess
import sys
sys.path.insert(1, './')
from module import utils






#============================#
# Path to executable
#============================#

hmmbuild_exec = '/projects/bpms/jgado/hmmer-3.2.1/src/hmmbuild'
hmmsearch_exec = '/projects/bpms/jgado/hmmer-3.2.1/src/hmmsearch'
mafft_exec = '/usr/local/bin/mafft'






#================================================================#
# Build PETase-HMM from MSA of 61 experimentally verified PETases
#================================================================#

# Align sequences
seqfile = './experiment/data/hmmsearch/petase_hmm/current_petases.fasta'
msafile = './experiment/data/hmmsearch/petase-hmm/current_petases_msa.fasta'
_ = utils.mafft_MSA(seqfile, msafile, mafft_exec)


# Build HMM
hmmfile = './experiment/data/hmmsearch/petase_hmm/hmm.txt'
_ = utils.hmmbuild_fxn(msafile, hmmfile, hmmbuild_exec)  # 61 seqs, 892 positions


# Search HMM against PETase sequences 
output = './experiment/hmmsearch/data/petase-hmm/output'
_ = utils.hmmsearch_fxn(hmmfile, seqfile, hmmsearch_exec, threshold=0, 
                        tempdir=output) # Hmm score of all cutinase-like PETases > 100





#==================================================================================#
# Search PETase-HMM against 1.6 billion sequences in NCBI/MGNify/JGI with HPC
#==================================================================================#


threshold = 100 


# Sequence database paths
databases = {'ncbi': '/scratch/jgado/sequence_databases/ncbi_nr/nr.fasta',
             'jgi': '/scratch/jgado/sequence_databases/JGI_Saroj/'\
                     'combined_hotsprings_metagenome.fasta'}
for i in range(1,7):
    databases[f'mgnify{i}'] = '/scratch/jgado/sequence_databases/mgnify/fasta'\
                              f'/mgy_proteins_{i}.fa'


# Write bash scripts to implement hmmsearch with HPC
for key in databases.keys():
    
    bashscript = f'./experiment/bash_scripts/{key}.sh'
    output_file = f'./experiment/data/hmmsearch/output/{key}_output.txt'
    tab_file = f'./experiment/data/hmmsearch/output/{key}_tabout.txt'
    aln_file = f'./experiment/data/hmmsearch/output/{key}_aln.sto'
    fasta_file = f'./experiment/data/hmmsearch/output/{key}_aln.fasta'
        
    with open(bashscript, 'w') as bash:
        bash.write(f'''#!/bin/bash
#SBATCH --job-name={key}
#SBATCH --account=bpms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=high
#SBATCH --time=04:00:00
#SBATCH --output=./experiment/stdout/{key}.out
#SBATCH --error=./experiment/stdout/{key}.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=japheth.hpc@gmail.com


# Work directory and virtual environment
cd /scratch/jgado/deepPETase
source activate /home/jgado/condaenvs/tfgpu


# Run hmmsearch
{hmmsearch_exec} \\
-o {output_file} \\
--tblout {tab_file} \\
-A {aln_file} --noali \\
-T {threshold} --domT {threshold} --incT {threshold} --incdomT {threshold} \\
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
    
    
    