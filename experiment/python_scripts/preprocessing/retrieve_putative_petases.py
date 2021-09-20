"""
Retrieve PETase-like sequences from databases with HMM
"""




import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import os
import subprocess
import joblib
import time
import sys


from module import utils
import utils




HMMSEARCH_EXEC = '/usr/local/bin/hmmsearch'




# Build PETase-HMM from MSA of experimentally verified PETases
#================================================================#

# Align sequences
seqfile = './experiment/data/sequence/current_petases.fasta'
msafile = './experiment/data/sequence/current_petases_msa.fasta'
utils.mafft_MSA(seqfile, msafile)

# Build HMM
hmmfile = './experiment/data/hmm/petase-hmm.txt'
stdout = utils.hmmbuild_fxn(msafile, hmmfile)  # 61 seqs, 892 positions





# Write bash scripts to search PETase-HMM against sequence databases
#=====================================================================#

#!/bin/bash                                                                                                                                                                                    
#SBATCH --job-name=hmmncbi                                                                                                                                                                     
#SBATCH --account=bpms                                                                                                                                                                         
#SBATCH --nodes=1                                                                                                                                                                              
#SBATCH --ntasks=1                                                                                                                                                                             
##SBATCH --partition=debug                                                                                                                                                                     
#SBATCH --time=1:00:00                                                                                                                                                                         
#SBATCH --output=./work_bash_scripts/stdout/petase_database_ncbi_hmmsearch.out                                                                                                                 
#SBATCH --error=./work_bash_scripts/stdout/petase_database_ncbi_hmmsearch.err                                                                                                                  
#SBATCH --mail-type=ALL                                                                                                                                                                        
#SBATCH --mail-user=japheth.hpc@gmail.com                                                                                                                                                      

# Work directory and virtual environment                                                                                                                                                       
cd /scratch/jgado/petase_DL
source activate /home/jgado/condaenvs/tfgpu

# Source directories                                                                                                                                                                           
EXEC="/projects/bpms/jgado/hmmer-3.2.1/src/hmmsearch"
db_file="/scratch/jgado/sequence_databases/ncbi_nr/nr.fasta"
hmm_file="data/petase_database/hmm/petase_database.hmm"
output_file="data/hmmsearch/ncbi/petase_database_output.txt"
tab_file="data/hmmsearch/ncbi/petase_database_tab.txt"
aln_file="data/hmmsearch/ncbi/petase_database_hits.sto"
fasta_file="data/hmmsearch/ncbi/petase_database_hits.fasta"

# Run hmmsearch                                                                                                                                                                                
$EXEC -o ${output_file} --tblout ${tab_file} -A ${aln_file} --noali -T 70 --domT 70 --incT 70 --incdomT 70 ${hmm_file} ${db_file}

# Convert Stockholm alignment file to fasta file                                                                                                                                               
python -c "import sys; sys.path.insert(1, './work_python_scripts'); import utils; utils.sto_to_fasta('${aln_file}', '${fasta_file}')"









from Bio import SeqIO
sequences = SeqIO.parse(seqfile, 'fasta')
ids = [seq.id for seq in sequences]


fastafile = '/Users/jgado/Dropbox/research/projects/petase_project74/combined_hotsprings_metagenome.fasta'
outfile = 'fasta.fasta'
seqids = seqids[:100000]


import time
start = time.time()
utils.extract_sequences(fastafile, seqids, outfile)
stop = time.time()
print(stop - start)




#===========================#
# Compile activity data
#===========================#
'''
# Retrieve experimental data from csv (52 petcans)
csvdir = 'data/activity_data/assay_data'
csvnames = sorted(os.listdir(csvdir))
activity_data = {}

for csvname in csvnames:
    if '.csv' not in csvname:
        continue
    df = pd.read_csv(f'{csvdir}/{csvname}', index_col=0)
    df['buffer'] = 'naphos'
    df.loc['7.5 (H)', 'buffer'] = 'hepes'
    df.index = [9.0, 8.0, 7.5, 7.5, 7.0, 6.0]
    df.columns = [30., 40., 50., 60., 70., 'buffer']
    df['ph'] = df.index.values
    df['naphos'] = (df['buffer'] == 'naphos').astype(int)
    df['hepes'] = (df['buffer'] == 'hepes').astype(int)
    df = df.drop(columns='buffer')
    df = df.melt(id_vars=['ph', 'naphos', 'hepes'], var_name='temp', 
                 value_name='activity').astype(float)
    activity_data[csvname.replace('.csv', '')] = df


# Experimental values for all samples in one dataframe/csv
allsamples = pd.DataFrame()
for petcan, df in activity_data.items():
    df['petcan'] = petcan
    df = df.loc[:, ['petcan', 'naphos', 'hepes', 'ph', 'temp', 'activity']]
    allsamples = pd.concat([allsamples, df], axis=0, ignore_index=True)
allsamples.to_csv('data/activity_data/all_activity.csv')


# Maximum activity for all petcans
max_activity = pd.DataFrame(columns=allsamples.columns)
petcans = set(allsamples['petcan'])
for petcan in petcans:
    df = allsamples.iloc[(allsamples['petcan']==petcan).values, :]
    df = df.sort_values('activity')
    best = pd.DataFrame(df.iloc[-1,:]).transpose()
    max_activity = pd.concat([max_activity, best], axis=0, ignore_index=True)
max_activity['ispetase'] = (max_activity['activity'] > 0).astype(int)
max_activity.to_csv('data/activity_data/max_activity.csv')


# Normalize activity data to a 0-1 scale
normalized = allsamples.copy()
for col in normalized.columns:
    if col == 'petcan':
        continue
    array = normalized[col].values
    if col == 'activity':
        array = np.log(array + 1e-4)
    minval, maxval = min(array), max(array)
    array = (array - minval) / (maxval - minval)
    normalized[col] = array
normalized.to_csv('data/activity_data/all_activity_normalized.csv')


# Normalized Maximum activity for all petcans
max_activity = pd.DataFrame(columns=allsamples.columns)
petcans = set(allsamples['petcan'])
for petcan in petcans:
    df = allsamples.iloc[(allsamples['petcan']==petcan).values, :]
    df = df.sort_values('activity')
    best = pd.DataFrame(df.iloc[-1,:]).transpose()
    max_activity = pd.concat([max_activity, best], axis=0, ignore_index=True)
max_activity['ispetase'] = (max_activity['activity'] > 0).astype(int)
max_activity.to_csv('data/activity_data/max_activity_normalized.csv')


# Normalize activity data to a 0-1 scale
normalized = allsamples.copy()
for col in normalized.columns:
    array = normalized[col].values
    if col in ['petcan', 'naphos', 'hepes']:
        continue
    elif col in ['ph', 'temp']:
        minval, maxval = min(array), max(array)
        array = (array - minval) / (maxval - minval)
    elif col == 'activity':
        array_nonzero = array[array>0]
        minval, maxval = min(array_nonzero), max(array_nonzero)
        minval, maxval = np.log(minval), np.log(maxval)
        array = [(np.log(val) - minval)/(maxval - minval) if val!=0 else -1 for val in array]   
    normalized[col] = array
normalized.to_csv('data/activity_data/all_activity_normalized2.csv')


# Sequence fasta file of petcans with activity data
heads_p74, seqs_p74 = utils.read_fasta('data/fasta/p74_seqs/all_petcans.fasta')  # 74 p74 seqs + 6 controls
heads_p74 = utils.get_accession('data/fasta/p74_seqs/all_petcans.fasta')
has_data = pd.Series(heads_p74).isin(max_activity['petcan'].values).values
heads_data, seqs_data = np.array(heads_p74)[has_data], np.array(seqs_p74)[has_data]
utils.write_fasta(heads_data, seqs_data, 'data/fasta/p74_seqs/assayed_petcans.fasta')
'''


#====================================#
# Compile new/recent activity data
#====================================#
data = pd.read_csv('data/activity_data/recent_all_activity.csv')
buffer_array = data['ph'].values
temp_array = data['temp'].values
buffer_encoder = {'C6':[1,0,0,0,0,0], 
                  'NP7':[0,1,0,0,0,0], 
                  'NP7.5':[0,0,1,0,0,0],
                  'H7.5':[0,0,0,1,0,0],
                  'B8':[0,0,0,0,1,0],
                  'G9':[0,0,0,0,0,1]}
newdata = []
for col in data.columns[2:]:
    print(col)
    column = data[col].values
    
    for i in range(len(column)):
        # Sequence/PETcan name
        try:
            name = 'PETcan' + str(int(col))
        except:
            name = col
        # Buffer/pH
        buffer = buffer_encoder[buffer_array[i].replace(' ','')]
        # Temp
        temp = temp_array[i]
        # Activity
        activity = column[i]
        newdata.append([name] + buffer + [temp, activity])

data = pd.DataFrame(newdata)
data.columns = ['petcan', 'C6', 'NP7', 'NP7.5', 'H7.5', 'B8', 'G9', 'temp', 'activity']

data.to_csv('data/activity_data/recent_all_activity_compiled.csv')
plt.hist(data.activity.values); plt.title('Raw activity'); plt.show(); plt.close()
plt.hist(np.log(data.activity.values + 1)); plt.show(); plt.close()

# Log-normalize data
activity = np.log(data.activity.values + 1)
plt.scatter(activity, data.activity.values); plt.show(); plt.close()
data['temp'] = (data['temp'] - min(data['temp'])) / (max(data['temp']) - min(data['temp']))
data['activity'] = (activity - min(activity)) / (max(activity) - min(activity))
data.to_csv('data/activity_data/recent_all_activity_normalized.csv')




#=====================================================#
# Build HMM from all known PETases (published + P74)
#=====================================================#

# Align PETases
#fastafile = 'data/petase_database/p74_and_published_petases.fasta'
#msafile = 'data/petase_database/p74_and_published_petases_msa.fasta'
#utils.mafft_MSA(fastafile, msafile)
msafile = 'recent_p74/fasta/published+p74_noredun_structmsa_full.fasta'
fastafile = 'recent_p74/fasta/published+p74.fasta'
msafile2 = 'recent_p74/fasta/published+p74_mafft.fasta'
utils.mafft_MSA(fastafile, msafile2)

# Build HMM
hmmbuild = '/usr/local/bin/hmmbuild'
hmmfile = 'data/petase_database/hmm/petase_database.hmm'
hmmfile2 = 'data/petase_database/hmm/petase_database_mafft.hmm'
#output = subprocess.check_output(f'{hmmbuild} {hmmfile} {msafile}', shell=True)
output = subprocess.check_output(f'{hmmbuild} {hmmfile2} {msafile2}', shell=True)

# Align PETcans to PETase HMM
hmmsearch = '/usr/local/bin/hmmsearch'
#petcans_fasta = 'data/fasta/p74_seqs/all_petcans.fasta'
#cmd = f'{hmmsearch} -o output.out --tblout tab_output.out {hmmfile} {fastafile}'
cmd = f'{hmmsearch} -o output2.out --tblout tab_output2.out {hmmfile2} {fastafile}'
output = subprocess.check_output(cmd, shell=True)

# Convert HMM output alignment from stockholm to fasta format
stofile = 'data/hmmsearch/ncbi/petase_database_hits.sto'
fastafile = 'data/hmmsearch/ncbi/petase_database_hits.fasta'
utils.sto_to_fasta(stofile, fastafile)  #7,099 sequences

# Convert lower case letters to upper case
[heads, seqs] = utils.read_fasta(fastafile)
seqs = [seq.upper() for seq in seqs]
utils.write_fasta(heads, seqs, fastafile)

# Remove gaps
nogaps = 'data/hmmsearch/ncbi/petase_database_hits_nogaps.fasta'
utils.remove_gaps(fastafile, nogaps)

# Get hamming weights for HMM hits
start = time.time()
seqidmat = utils.seqid_matrix(fastafile)
#hamming_weights = utils.hamming_weights(fastafile, 0.2)
stop = time.time()
seqidmat.to_csv('data/hmmsearch/ncbi/petase_database_hits_seqidmat.csv')
weights_hamming = utils.hamming_weights(seqidmat=seqidmat, threshold=0.2)
counts = 1 / weights_hamming
plt.hist(counts); plt.yscale('log')





#====================================================================#
# Combine HMM search (PETase database) hits from NCBI/MGnify/JGI
#====================================================================#

# Fasta files of hits
fastas = ['data/hmmsearch/ncbi/petase_database_hits.fasta',
          'data/hmmsearch/jgi/petase_database_hits.fasta']
fastas += [f'data/hmmsearch/mgnify/petase_database_hits{i}.fasta' for i in range(1,7)]

# All hits in one fasta file
allheads, allseqs = [], []

for fasta in fastas:
    heads, seqs = utils.read_fasta(fasta)
    seqs = [seq.replace('-','').upper() for seq in seqs]
    allheads += heads
    allseqs += seqs

# Write combined fasta file
fasta = 'data/hmmsearch/combined/all_hits.fasta'
utils.write_fasta(allheads, allseqs, fasta)


# Cluster all PETase HMM hits
#===============================#
MMSEQS = '/home/jgado/condaenvs/tfgpu/bin/mmseqs'
fastafile = 'data/hmmsearch/combined/all_hits.fasta'

_ = utils.cluster_sequences(fastafile=fastafile, clusterfile=None,
                            minseqid=0.95, mmseqs_exec=MMSEQS, parse=False,
                            delete_temp_files=False)
clusters = pd.read_csv('./tempdir/seqdb_cluster.tsv', index_col=0, header=None,
                       sep='\t')
clusterfile = 'data/hmmsearch/combined/clusters.csv'
clusters.to_csv(clusterfile)
count = np.unique(clusters.index)
print(f'Num. of unique clusters = {len(count)}')
_ = subprocess.call('rm -rfv ./tempdir', shell=True)




# Get hamming-distance sequence weights                                                                                                                                                 
#==========================================#                                                                                                                                            
'''
MMSEQS = '/home/jgado/condaenvs/tfgpu/bin/mmseqs'
#storepath = 'data/hmmsearch/ncbi'
storepath = 'data/hmmsearch/mgnify'

#fasta_nogaps = f'{storepath}/petase_database_hits_nogaps.fasta'
fasta_nogaps = f'{storepath}/petase_database_hits_all.fasta'

heads, seqs = utils.read_fasta(fasta_nogaps)
seqs = [seq.replace('-','').upper() for seq in seqs]
utils.write_fasta(heads, seqs, fasta_nogaps)

weights_file = f'{storepath}/hamming_weights.csv'
weights = utils.hamming_weights(fastafile=fasta_nogaps, minseqid=0.8, maxseqs=None,                                                                                                    
                                output_file=weights_file, delete_temp_files=True,                                                                                                      
                                mmseqs_exec=MMSEQS)                                                                                                                                    
weights = pd.read_csv(weights_file, index_col=0)
print('Number of effective sequences = {:.2f}'.format(np.sum(weights.iloc[:,-1])))




# Cluster sequences and split clusters to training (80%) and testing (20%) sets                                                                                                         
#====================================================================================#                                                                                                  

MMSEQS = '/home/jgado/condaenvs/tfgpu/bin/mmseqs'
#storepath = 'data/hmmsearch/ncbi'
storepath = 'data/hmmsearch/mgnify'

clusterfile=f'{storepath}/clusters.txt'
#fasta_nogaps = f'{storepath}/petase_database_hits_nogaps.fasta'
fasta_nogaps = f'{storepath}/petase_database_hits_all.fasta'

clusters = utils.cluster_sequences(fastafile=fasta_nogaps, clusterfile=clusterfile,                                                                                                    
                                   minseqid=0.7, mmseqs_exec=MMSEQS)                                                                                                                   
trainseqs, testseqs = utils.split_cluster(clusterfile=clusterfile, testsize=0.2,
                                          random_seed=None, verbose=True)
# Write train/test sequence data to disk                                                                                                                                                
with open(f'{storepath}/trainseqs.txt', 'w') as trainfile:
    trainfile.write('\n'.join(trainseqs))
with open(f'{storepath}/testseqs.txt', 'w') as testfile:
    testfile.write('\n'.join(testseqs))
'''
