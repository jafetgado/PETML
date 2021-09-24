"""
Combine non-redundant PETase-like sequences from hmmsearch in a single file
"""




import sys
sys.path.insert(1, './')
from module import utils




#==============================================#
# Combine all sequences from hmmsearch output
#===============================================#

# Retreive output of hmmsearch of databases
allheads, allseqs = [], []
databases = ['ncbi', 'jgi'] + [f'mgnify{i}' for i in range(1,7)]

for key in databases:
    fastafile = f'./experiment/hmmsearch/data/output/{key}_aln.fasta'
    [heads, seqs] = utils.read_fasta(fastafile)
    seqs = [seq.replace('*','').replace('-','').upper() for seq in seqs]
    allheads.extend(heads)
    allseqs.extend(seqs)

# Combine sequences
allfasta = './experiment/preprocessing/data/all_hmmsearch_seqs.fasta'
utils.write_fasta(allheads, allseqs, allfasta)  # 555,290 sequences




#=============================#
# Remove redundant sequences 
#=============================#

# Cluster with MMSEQS at 99% identity (302,087 seqs)
allfasta = './experiment/preprocessing/data/all_hmmsearch_seqs.fasta'
clusterfasta = './experiment/preprocessing/data/hmmsearch99.fasta'
_ = utils.cluster_sequences(fastafile=allfasta, clusterfasta=clusterfasta, minseqid=0.99, 
                            maxseqs=500000, maxevalue=10, cluster_mode=2, cov_mode=1, 
                            seqidmode=1, write_cluster_data=False,
                            delete_temp_files=True)




