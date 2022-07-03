"""
Cluster sequences into families and split to train/test sets
"""




import numpy as np
import pandas as pd
import sys
sys.path.insert(1, './')
from module import utils




#====================================#
# Exclude very short/long sequences
#====================================#

seqfile = './experiment/preprocessing/data/hmmsearch99.fasta'

# Sequence lengths
heads, seqs = utils.read_fasta(seqfile)
lengths = np.array([len(seq) for seq in seqs])
utils.array_to_csv(lengths, './experiment/preprocessing/data/seq_lengths_orig.csv')

# Length thresholds
minlength = 70
maxlength = 400
below_min = lengths[lengths<minlength].shape  # 176 seqs, 0.060%
above_max = lengths[lengths>maxlength].shape  # 286seqs, 0.099%

# Remove sequences beyond the threshold
heads_new, seqs_new = [], []
for i, (head, seq, length) in enumerate(zip(heads, seqs, lengths)):
    if (minlength < length < maxlength):
        heads_new.append(head)
        seqs_new.append(seq)
        
# Save selected sequences
trimfile = './experiment/preprocessing/data/hmmsearch99_trimmed.fasta'
utils.write_fasta(heads_new, seqs_new, trimfile)




#=============================#
# Cluster into families (1E-2)
#=============================#

# Cluster with MMSEQS
clusterfile = './experiment/preprocessing/data/cluster_families.txt'
_ = utils.cluster_sequences(fastafile=seqfile, clusterfasta=None, clusterfile=clusterfile,
                            minseqid=0., maxseqs=300000, maxevalue=1e-1, cluster_mode=2, 
                            cover=0.33, cov_mode=1, seqidmode=1, sensitivity=25.0, 
                            write_cluster_data=True, save_rep_seqs=False, 
                            delete_temp_files=True)

# Analyze cluster results



#===============================================================#
# Split each family cluster to train (90%) and test (10%) sets
#===============================================================#
'''
# Retrieve family cluster data
cl_info, cl_data = utils.read_fasta(clusterfile)
cl_info = [each.split('::') for each in cl_info]
cl_data = [each.split(', ') for each in cl_data]

# Cluster info as dataframe
dfinfo = pd.DataFrame(cl_info)
dfinfo.iloc[:,-1] = dfinfo.iloc[:,-1].astype(int)
dfinfo.columns = ['name', 'rep. accession', 'size']
dfinfo.to_csv('experiment/preprocessing/data/cluster_families_all.csv')



# Collapse clusters with less than 50 sequences
cluster_data = []
collapsed = []
for i in range(len(dfinfo)):
    size = dfinfo.iloc[i,-1]
    if size > 50:
        cluster_data.append(cl_data[i]) # 285,477 seqs from 351 clusters
    else:
        collapsed.extend(cl_data[i])  # 2,460 seqs collapsed from 135 clusters
cluster_data.append(collapsed)
sizes = [len(each) for each in cluster_data]
dfnew = pd.DataFrame()
        
        
        





for i in range(len(dfinfo)):
clusterfile = '../cluster_families.txt'

# Retrieve family cluster data
cl_info, cl_data = utils.read_fasta(clusterfile)
cl_info = [each.split('::') for each in cl_info]
cl_data = [each.split(', ') for each in cl_data]



thresh = 50
select = sizes[sizes<thresh]
print(select.shape[0], sizes.shape[0], sizes.shape[0] - select.shape[0])
print(select.sum())


#plt.plot(np.log(df.iloc[:,-1].values))
plt.plot(np.log(sizes))
plt.axhline(np.log(50), color='black', linestyle='--')
'''