"""
Prepare sequence data for machine learning
"""




#============#
# Imports
#============#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, './')
from module import utils






#===============================================================#
# Combine all sequences from HMM search output (score > 100)
#===============================================================#

# Retreive output of hmmsearch against sequence databases
allheads, allseqs = [], []
databases = ['ncbi', 'jgi'] + [f'mgnify{i}' for i in range(1,7)]

for key in databases:
    fastafile = f'./experiment/data/hmmsearch/output/{key}_aln.fasta'
    [heads, seqs] = utils.read_fasta(fastafile)
    seqs = [seq.replace('*','').replace('-','').upper() for seq in seqs]
    allheads.extend(heads)
    allseqs.extend(seqs)

# Combine sequences
allfasta = './experiment/data/hmmsearch/output/all_hmmsearch_seqs.fasta'
utils.write_fasta(allheads, allseqs, allfasta)  
print(f'No. of retrieved sequences = {len(allheads)}')  #10,633 sequences




        

#==============================#
# Remove redundant sequences 
#==============================#

mmseqs_exec = '/home/jgado/condaenvs/tfgpu/bin/mmseqs'

# Cluster with MMSEQS at 100% identity threshold, keeping the longest seq. in each cluster
fasta = './experiment/data/hmmsearch/output/all_hmmsearch_seqs.fasta'
repfasta = './experiment/data/preprocessing/hmmsearch_noredun.fasta'
_ = utils.cluster_sequences(fastafile=fasta, 
                            mmseqs_exec=mmseqs_exec,
                            save_rep_seqs=True,
                            repfasta=repfasta,
                            write_cluster_data=False,
                            clusterfile=None,
                            write_hamming_weights=False,
                            hamming_weights_file=None,
                            tempdir='tempdir',
                            delete_temp_files=True,
                            seqidmode=1, # Seq-id is relative to shorter sequence
                            minseqid=0.9999,
                            maxseqs=20,
                            maxevalue=1e-3,
                            cluster_mode=2, #CD-HIIT clustering to keep longest sequence
                            cover=0.5,
                            cov_mode=1,
                            sensitivity=4.0)
heads, seqs = utils.read_fasta(repfasta)  
print(len(heads)) # 8,081 sequences remaining






#========================================================================#
# Split sequences to train/test sets with 80% sequence identity threshold
#========================================================================#

fasta = './experiment/data/preprocessing/hmmsearch_noredun.fasta'
hamming_weights_file =  './experiment/data/preprocessing/hamming_weights.csv'
clusterfile = './experiment/data/preprocessing/clusters.txt'

# Cluster at 80% identity threshold
_ = utils.cluster_sequences(fastafile=fasta, 
                            mmseqs_exec=mmseqs_exec,
                            save_rep_seqs=False,
                            repfasta=None,
                            write_cluster_data=True,
                            clusterfile=clusterfile,
                            write_hamming_weights=True,
                            hamming_weights_file=hamming_weights_file,
                            tempdir='tempdir',
                            delete_temp_files=True,
                            seqidmode=1, # Seq-id is relative to shorter sequence
                            minseqid=0.80, # 80% seq. identity threshold
                            maxseqs=20,
                            maxevalue=1e-3,
                            cluster_mode=0, #Greedy clustering
                            cover=0.5,
                            cov_mode=1,
                            sensitivity=4.0)


# Split clusters into training and testing sets (80% training, 20% testing)
trainaccs, testaccs = utils.split_cluster(clusterfile=clusterfile, 
                                          testsize=0.2,
                                          random_seed=0,
                                          verbose=True)

# Write training and testing sequences
heads, seqs = utils.read_fasta(fasta)
accs = utils.get_accession(fasta)
heads_data = dict(zip(accs, heads))
seqs_data = dict(zip(accs, seqs))
train_heads = [heads_data[acc] for acc in trainaccs]
train_seqs = [seqs_data[acc] for acc in trainaccs]
test_heads = [heads_data[acc] for acc in testaccs]
test_seqs = [seqs_data[acc] for acc in testaccs]
trainfasta = './experiment/data/preprocessing/train_sequences.fasta'
testfasta = './experiment/data/preprocessing/test_sequences.fasta'
utils.write_fasta(train_heads, train_seqs, trainfasta)   # 6,463 sequences 
utils.write_fasta(test_heads, test_seqs, testfasta)      # 1,618 sequences 






#=================================================#
# Combine and align train/test/labeled sequences 
#=================================================#

# Retrieve add, train, and test sequences
labelfasta = './experiment/data/preprocessing/label_sequences.fasta' 
trainfasta = './experiment/data/preprocessing/train_sequences.fasta'
testfasta = './experiment/data/preprocessing/test_sequences.fasta'

heads_label, seqs_label = utils.read_fasta(labelfasta)   # 428 seqs
heads_train, seqs_train = utils.read_fasta(trainfasta) # 6,463 seqs
heads_test, seqs_test = utils.read_fasta(testfasta)  # 1,618 seqs


# Combine all sequences in a single file
heads_all = heads_train + heads_test + heads_label 
seqs_all = seqs_train + seqs_test + seqs_label

allfasta = './experiment/data/preprocessing/combined_sequences.fasta'
utils.write_fasta(heads_all, seqs_all, allfasta)


# Align sequences
msafasta = './experiment/data/preprocessing/combined_msa.fasta'
utils.mafft_MSA(allfasta, msafasta)  






#========================================#
# Prepare alignment for VAE training
#========================================#

heads_msa, seqs_msa = utils.read_fasta(msafasta)  
heads_msa = np.array(heads_msa, dtype=np.object)
seqs_msa = np.array(seqs_msa, dtype=np.object)
print(len(seqs_msa[0]))  # 1,852 positions


# Location of subalignment sequences in combined alignment
train_idx = 0, len(heads_train)
test_idx = len(heads_train), len(heads_train) + len(heads_test)
label_idx = len(heads_train) + len(heads_test), len(heads_all)


# Check correctness of sequence index
assert np.equal(heads_msa[label_idx[0]:label_idx[1]], np.array(heads_label)).any()
assert np.equal(heads_msa[train_idx[0]:train_idx[1]], np.array(heads_train)).any()
assert np.equal(heads_msa[test_idx[0]:test_idx[1]], np.array(heads_test)).any()


# Exclude positions with occupancy < 5% (i.e. >95% gaps in the position)
df = utils.fasta_to_df(msafasta)
gapfreqs = [(df.iloc[:,i]=='-').values.sum() for i in range(df.shape[1])]
gapfreqs = np.array(gapfreqs) / df.shape[0]
select_locs = [i for i in range(len(gapfreqs)) if gapfreqs[i] < 0.95]
print(len(select_locs)) # 437 positions
df = df.iloc[:, select_locs]


# Save selection
aln_select = './experiment/data/preprocessing/vae_aln_positions.csv'
sel = pd.DataFrame(select_locs)
sel.to_csv(aln_select)


# Save aligned gapless sequences (i.e. with positions having +95% gaps removed)
gapless_path = './experiment/data/preprocessing/combined_msa_gapless.fasta'
utils.df_to_fasta(df, gapless_path)
heads_msa, seqs_msa = utils.read_fasta(gapless_path)
    

# Train/test/label gapless alignment
train_gapless = './experiment/data/preprocessing/train_msa_gapless.fasta'
utils.write_fasta(heads_msa[train_idx[0]:train_idx[1]],
                  seqs_msa[train_idx[0]:train_idx[1]], 
                  train_gapless)

test_gapless = './experiment/data/preprocessing/test_msa_gapless.fasta'
utils.write_fasta(heads_msa[test_idx[0]:test_idx[1]],
                  seqs_msa[test_idx[0]:test_idx[1]], 
                  test_gapless)

label_gapless = './experiment/data/preprocessing/label_msa_gapless.fasta'
utils.write_fasta(heads_msa[label_idx[0]:label_idx[1]],
                  seqs_msa[label_idx[0]:label_idx[1]], 
                  label_gapless)


#===========================================================#
# Derive sample (hamming) weights to reweight VAE training
#===========================================================#

'''
The hamming weight for each sequence is computed as the reciprocal of the number of 
sequences in the dataset with a hamming distance of less than 0.2 (i.e. >80% identity)
from the sequence.

Sequences are clustered at 80% identity and split into training and testing sets so
that all sequences in the test set have less than 80% identity with all sequences in the 
training set.
'''

# Accession of training/testing data
trainaccs = utils.get_accession(trainfasta)
testaccs = utils.get_accession(testfasta)

# Hamming weights (inverse count of sequences with > 80% identity to sequence)
hamming_weights_file = './experiment/data/preprocessing/hamming_weights.csv'
hamming_weights =  pd.read_csv(hamming_weights_file, index_col=0)
trainweights = hamming_weights.reindex(trainaccs)
testweights =  hamming_weights.reindex(testaccs)
print(trainweights.values.sum())   # 1,248.0
print(testweights.values.sum())    # 296.0 

# Write weights to file
trainweights.to_csv('./experiment/data/preprocessing/train_hamming_weights.csv')
testweights.to_csv('./experiment/data/preprocessing/test_hamming_weights.csv')



