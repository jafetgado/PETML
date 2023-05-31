"""
Train supervised ML model to predict PET hydrolase activity
"""




import numpy as np
import pandas as pd
import subprocess
import os
import joblib
from sklearn.linear_model import LogisticRegression
import petml.helper as helper




MAFFT_EXE = '/usr/local/bin/mafft' # Change this
HMMSEARCH_EXE = '/usr/local/bin/hmmsearch' # Change this
TMP = 'petml/data/tmp'
DELETE_TMP_FILES = False
if not os.path.exists(TMP): 
    os.makedirs(TMP)






#================================================#
# Prepare pairwise dataset for learning-to-rank
#================================================#

datasets = [
    'Bell et al, 2022', 
    'Zeng et al, 2022', 
    'Tournier et al, 2020 (Naturals)',
    'Xi et al, 2021',
    'Brott et al, 2021',
    'Pfaff et al, 2022', 
    'Han et al, 2017', 
    'Li Z. et al, 2022', 
    'Zhang et al, 2021', 
    'Sonnendecker et al, 2021',
    'Lu et al, 2022',
    'Chen et al, 2021', 
    'Then et al, 2016', 
    'Joo et al, 2018', 
    'Guo et al, 2022', 
    'Ma et al, 2018', 
    'Sagong et al, 2022',
    'Tournier et al, 2020',
    'Wang et al, 2022', 
    'Son et al, 2019', 
    'Furukawa et al, 2019', 
    'Wei et al, 2016', 
    'Nakamura et al, 2021', 
    'Liu et al, 2018', 
    'Li Q. et al, 2022',
    'Cui et al, 2021 (NanoPET)', 'Cui et al, 2021 (PET film)', 
    'Erickson et al, 2022 (Project74)'
    ]


# Get sequence data
headers, sequences = helper.read_fasta('petml/data/activity_datasets/sequences.fasta')
short_names = [f"Seq{i+1}" for i in range(len(headers))]
short_to_full = dict(zip(short_names, headers))
full_to_short = dict(zip(headers, short_names))
seqdict = dict(zip(short_names, sequences))
helper.write_fasta(seqdict, 'petml/data/tmp/sequences.fasta')
pd.Series(short_to_full).to_csv('petml/data/tmp/seq_name_key.csv')


# Pairwise data from all activity datasets
paired_data = {}
seqid = 0

for dataset in datasets:

    # Retrieve experiment labels
    df = pd.read_excel(f'petml/data/activity_datasets/{dataset}.xlsx', index_col=0, engine='openpyxl')
    proteins = df['Protein'].values
    activities = df['Activity'].values
    
    # Form all possible pairwise 
    for i in range(len(df)):
        for k in range(len(df)):
            if i != k:
                paired_data[seqid] = dict(
                    seqid = seqid,
                    name1 = full_to_short[proteins[i]],
                    name2 = full_to_short[proteins[k]],
                    study = dataset,
                    y = int(activities[i] > activities[k]), 
                    seq1 = seqdict[full_to_short[proteins[i]]],
                    seq2 = seqdict[full_to_short[proteins[k]]],
                    )
                seqid += 1


# Write pairwise data to csv
dfpaired = pd.DataFrame(paired_data).transpose()
dfpaired.index = dfpaired.seqid
dfpaired.to_csv('petml/data/tmp/paired_data.csv')






#===================================================#
# Prepare sequence alignment of sequences 
#===================================================#

# Label sequences
heads_label, seqs_label = helper.read_fasta('petml/data/tmp/sequences.fasta')


# Search label sequences with HMM of evolutionary sequences
# Do this to remove flanking regions from sequence
helper.search_with_HMM(seq_file='petml/data/tmp/sequences.fasta', 
                       hmm_file='petml/data/unsupervised/jackhmmer_hmm_b04.txt', 
                       threshold=0, 
                       outdir='petml/data/hmmoutput', 
                       hmmsearch_exe=HMMSEARCH_EXE)


# Sequences returned by hmmsearch
subprocess.check_output(
     'mv petml/data/hmmoutput/aln_no_gaps.fasta petml/data/tmp/sequences_hmm.fasta',
     shell=True
     ) # Rename sequence file
subprocess.check_output('rm -rf petml/data/hmmoutput', shell=True)


# Edit sequence file returned by HMM to ensure the same headings as original file
heads, seqs = helper.read_fasta('petml/data/tmp/sequences_hmm.fasta')
heads = [head.split('/')[0] for head in heads]
helper.write_fasta(dict(zip(heads, seqs)), 'petml/data/tmp/sequences_hmm.fasta')


# Select a subset of evolutionary alignment (to save time in alignment)
'''
heads, seqs = helper.read_fasta('petml/data/unsupervised/jackhmmer_seqs_b04.fasta')
lengths = [len(seq.replace('-','')) for seq in seqs]
select1 = np.argsort(lengths)[-1000:] # 500 longest sequences
np.random.seed(0)
select2 = np.random.choice(list(set(np.arange(len(seqs))) - set(select1)),
                           size=1000, replace=False)
select = list(select1) + list(select2)
heads_select, seqs_select = np.array(heads)[select], np.array(seqs)[select]
helper.write_fasta(dict(zip(heads_select, seqs_select)),
                   'petml/data/tmp/jackhmmer_seqs_b04_small.fasta')
'''


# Add label sequences to evolutionary alignment
helper.align_with_MSA(seq_file='petml/data/tmp/sequences_hmm.fasta', 
                      msa_file='petml/data/unsupervised/jackhmmer_seqs_b04_small.fasta',
                      out_file='petml/data/tmp/sequences_msa.fasta',
                      mafft_exe=MAFFT_EXE)
assert len(helper.read_fasta('petml/data/tmp/sequences_msa.fasta')[1][0]) == 1813






#================================================#
# One hot encode sequence alignment
#================================================#

# First, trim MSA, dropping positions with <10% coverage in jackhmmer MSA
df = helper.fasta_to_df('petml/data/tmp/sequences_msa.fasta')
locs = pd.read_csv('petml/data/supervised/aln_positions_469.csv', index_col=None, 
                   header=None).values.flatten()
assert len(locs) == 469
df = df.iloc[:,locs]
helper.df_to_fasta(df, 'petml/data/tmp/sequences_msa_trimmed.fasta')


# One hot encode trimmed MSA
headers, sequences = helper.read_fasta('petml/data/tmp/sequences_msa_trimmed.fasta')
df = pd.DataFrame(sequences, index=headers)
ohe = helper.OneHotEncoder()
df = pd.DataFrame(ohe.encode_from_df(df, col=0), index=headers)
df.to_csv('petml/data/tmp/onehot.csv')






#================================================#
# Train logistic regression model (pairwise)
#================================================#

# Get data
dfpaired = pd.read_csv('petml/data/tmp/paired_data.csv', index_col=0)
dfonehot = pd.read_csv('petml/data/tmp/onehot.csv', index_col=0)

# Standardize one-hot encoding representation
mean_std_onehot = pd.DataFrame({'means':dfonehot.mean(axis=0).values, 
                                'stds':dfonehot.std(axis=0).values})
mean_std_onehot.to_csv('petml/data/supervised/onehot_mean_std.csv')
dfonehot = (dfonehot - mean_std_onehot['means'].values) / (mean_std_onehot['stds'].values + 1e-8)

# Prepare training data as pairwise data
X1 = dfonehot.loc[dfpaired['name1'].values,:].values
X2 = dfonehot.loc[dfpaired['name2'].values,:].values
Z = X1 - X2
y = dfpaired['y'].values


# Train model
model = LogisticRegression(C=1e-5,
                           class_weight='balanced', 
                           solver='saga', 
                           random_state=0,
                           n_jobs=-1)
model = model.fit(Z, y)

# Save model
joblib.dump(model, 'petml/data/supervised/onehot_log_reg.pkl')






#================================================#
# Prepare data for unsupervised prediction
#================================================#

# Get consensus sequence
fasta = 'petml/data/unsupervised/jackhmmer_seqs_b04_small.fasta'    
df = helper.fasta_to_df(fasta)
consensus = ''
for col in df.columns:
    res =  df[col].value_counts()
    consensus += str(res.index[0])
helper.write_fasta(dict(zip(['jackhmmer_consensus_b04'], 
                            [consensus])),
                   'petml/data/unsupervised/jackhmmer_consensus_b04.fasta')






#========================#
# Delete temp. files
#========================#

if DELETE_TMP_FILES:
    subprocess.call('rm -rfv petml/data/tmp/', shell=True)
    
    
    
    
    
    