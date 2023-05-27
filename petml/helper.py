"""
Helper functions 
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from scipy.stats import rankdata
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

import Bio
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MafftCommandline

import sys
import os
import subprocess
import json








def read_fasta(fasta, return_as_dict=False):
    '''Return the protein sequences contained in a fasta file. 
    
    fasta: string of path to file containing sequences in fasta format
    
    return_as_dict: If True, return a dictionary with headers as keys and sequences as
    values. If False, return a tuple (headers, sequences)
    '''
    
    headers, sequences = [], []

    with open(fasta, 'r') as fast:
        
        for line in fast:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq

    if return_as_dict:
        return dict(zip(headers, sequences))
    else:
        return (headers, sequences)








def write_fasta(seqdict, fastapath):
    '''Write sequences to a file in fasta format
    
    seqdict: dictionary with headers as keys and sequences as values
    
    fastapath: path which sequences will be written
    '''
    
    with open(fastapath, 'w') as f:
        for header, sequence in seqdict.items():
            f.write('>' + header + '\n' + sequence + '\n')








def get_accession(path):
    ''' Return a list of accession codes for all sequences in a fasta file (path).'''
    
    [heads, seqs] = read_fasta(path)
    accessions =  [head.split()[0] for head in heads]
    
    return accessions






def remove_gaps(input_file, output_file, gap='-', replacewith=''):
    '''Remove all gap characters from sequences in input_file and write a sequences to
    output_file'''
    
    [head, seq] = read_fasta(input_file)
    new_seq = [x.replace('-', replacewith) for x in seq]
    write_fasta(dict(zip(head, new_seq)), output_file)








def sto_to_fasta(sto_file, fasta_file):
    '''Convert jackhmmer alignment file in stockholm format (sto_file) to fasta format
    (fasta_file)'''
    
    align = AlignIO.read(sto_file, 'stockholm')
    _ = AlignIO.write(align, fasta_file, 'fasta')
    [heads, seqs] = read_fasta(fasta_file)
    seqs = [seq.upper() for seq in seqs]
    _ = write_fasta(dict(zip(heads, seqs)), fasta_file)
    
    
    
    
    
    
    
    
def hmmsearch_fxn(hmmfile, seqfile, hmmsearch_exec, threshold=0, tempdir='./tempdir'):
    '''Search a profile hidden markov Model against a set of sequences. Output files
    are written to tempdir.'''
    
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
        
    command = f'{hmmsearch_exec} -o {tempdir}/output.txt --tblout {tempdir}/tblout.txt '\
              f'-A {tempdir}/aln.sto -T {threshold} --domT {threshold} '\
              f'--incT {threshold}  --incdomT {threshold} --noali {hmmfile} {seqfile}'
    stdout = subprocess.check_output(command, shell=True)
    sto_to_fasta(f'{tempdir}/aln.sto', f'{tempdir}/aln.fasta')
    
    return stdout








def search_with_HMM(seq_file,
                 hmm_file,
                 threshold=100,
                 outdir='./outdir',
                 hmmsearch_exe='/usr/local/bin/hmmsearch'):
    '''Search sequences in seq_file with an HMM (hmm_file) using hmmsearch. Do this to 
    exclude flanking regions in the sequence such as auxiliary domains'''
    
    
    # Check paths
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    assert os.path.exists(seq_file), 'Cannot find seq_file'
    assert os.path.exists(hmm_file), 'Cannot find hmm_file'
    assert os.path.exists(hmmsearch_exe), 'Cannot find hmmsearch executable'
    
    
    # Run hmmsearch with subprocess 
    command = f'{hmmsearch_exe} -o {outdir}/output.txt '\
              f'--tblout {outdir}/tab_output.txt -A {outdir}/aln.sto '\
              f'-T {threshold} --domT {threshold} --incT {threshold} '\
              f'--incdomT {threshold} --noali {hmm_file} {seq_file}'
    _ = subprocess.check_output(command, shell=True)
    
    
    # Convert alignment output from stockholm to fasta
    sto_to_fasta(f'{outdir}/aln.sto', f'{outdir}/aln.fasta')
    
    # Write unaligned hits in separate fasta file
    remove_gaps(f'{outdir}/aln.fasta', f'{outdir}/aln_no_gaps.fasta')
    
    # Remove other HMM output files
    _ = [subprocess.check_output(f'rm -rfv {outdir}/{file}', shell=True) for file in \
         ['aln.fasta', 'aln.sto']] 
        
        
        
        
        
        
        
        
def parse_hmm_tabout(tabout):
    '''Parse a hmmer tab output file and return a dataframe of search results'''
    
    data = []
    with open(tabout, 'r') as tab:
        for line in tab:
            if not line.startswith('#'):
                line = line.split()
                data.append(line)
    data = pd.DataFrame(data)
    
    return data








def align_with_MSA(seq_file,
                   msa_file,
                   out_file,
                   mafft_exe='/usr/local/bin/mafft',
                   verbose=True):
    '''Add sequences to an existing alignment with MAFFT'''
    
    # Check paths/files
    assert os.path.exists(mafft_exe), 'Cannot find mafft executable'
    assert os.path.exists(seq_file), 'Cannot find seq_file'
    
    # Add sequences to VAE MSA with mafft, keeping the original MSA positions
    command = f'{mafft_exe} --keeplength --add {seq_file} {msa_file} > {out_file}'
    if not verbose:
        command += ' 2> .mafft_stdout.txt'
    _ = subprocess.check_output(command, shell=True)
    
    # Select only added sequences from full alignment
    heads_msa, seqs_msa = read_fasta(msa_file)
    heads, seqs = read_fasta(out_file) 
    heads_select, seqs_select = heads[len(heads_msa):], seqs[len(seqs_msa):]
    write_fasta(dict(zip(heads_select, seqs_select)), out_file)        
    
    if not verbose:
        os.remove('.mafft_stdout.txt')





    
def fasta_to_df(fasta):
    '''Read aligned sequences from a fasta file and return a Pandas dataframe with
    sequences as rows and residues as columns'''
    
    heads, sequences = read_fasta(fasta)
    data = [list(seq) for seq in sequences]
    lengths = [len(seq) for seq in data]
    assert len(set(lengths)) == 1, f'Sequences in {fasta} are of unequal length!'
    df = pd.DataFrame(data)
    df.index = [head.replace('>', '') for head in heads]
    
    return df            






def df_to_fasta(df, fasta):
    '''Write a 2D array (dataframe) of aligned sequences (with sequences as rows and 
    residues as columns) to a fasta file'''
    
    heads = df.index.values
    sequences = [''.join(df.iloc[i,:].values) for i in range(len(df))]
    write_fasta(dict(zip(heads, sequences)), fasta)






class OneHotEncoder():

    def __init__(self):

        self.amino_letters = list('-ACDEFGHIKLMNPQRSTVWYX')
        self.amino_dict = {char:num for (num,char) in enumerate(self.amino_letters)}
        self.reverse_amino_dict = {num:char for (num,char) in enumerate(self.amino_letters)}
        

    def encode(self, sequence):
        '''Encode as a 2D array'''
        
        for char in 'BJOUXZ':
            sequence = sequence.replace(char, 'X')
        array = np.zeros((len(sequence),len(self.amino_letters))) 
        for (i, char) in enumerate(sequence):
            array[i, self.amino_dict[char]] = 1

        return array
    
    
    def encode_from_df(self, df, col=0, alphabet='ACDEFGHIKLMNPQRSTVWY-'):
        '''One-hot encode protein sequences.
        df: Dataframe column of sequences in string format
        col: Column name containing sequences
        Credit: Ada Shaw
        '''
    
        aa_to_oh = dict(zip(list(alphabet),np.eye(len(alphabet))))
        oh_seqs = np.vstack( df[col].apply(lambda x: np.ravel([aa_to_oh[i] for i in list(x)])).values)
    
        return oh_seqs
    
    
    
    
    
"""

#===========================#
# Helper functions
#===========================#

def mutate_sequence(sequence, mutations):
    '''
    Mutate a wild-type sequence and return the mutant sequence.  

    sequence: sequence to be mutated as string
    
    mutations: a string of mutations seperated by '/', e.g 'M101A/Y102K/D103S' 
    '''

    seqlist = list(sequence)
    mutations = mutations.split('/')
    for name in mutations:
        wt_aa, pos, mut_aa = name[0], int(name[1:-1]), name[-1]
        assert (sequence[pos-1] == wt_aa), f'Position {pos} is not {wt_aa} in {name}'
        seqlist[pos-1] = mut_aa
    mutant_seq = ''.join(seqlist)
    
    return mutant_seq







            
            



















    
    
    
    
    

    






def write_json(writedict, path, indent=4, sort_keys=False):
    '''Save dictionary as json file in path'''
    
    f = open(path, 'w')
    _ = f.write(json.dumps(writedict, indent=indent, sort_keys=sort_keys))
    f.close()
    
    




def read_json(path):
    '''Return a dictionry read from a json file'''
    
    f = open(path, 'r')
    readdict = json.load(f)
    f.close()
    
    return readdict 






def cluster_sequences(fastafile, 
                      mmseqs_exec,
                      save_rep_seqs=False, 
                      repfasta='reps.fasta', 
                      write_cluster_data=False, 
                      clusterfile='clusters.txt', 
                      write_hamming_weights=False,
                      hamming_weights_file='hamming_weights.csv',
                      tempdir='tempdir', 
                      delete_temp_files=True,
                      seqidmode=0,
                      minseqid=0.0, 
                      maxseqs=20, 
                      maxevalue=1e-3, 
                      cluster_mode=0, 
                      cov_mode=0, 
                      cover=0.8, 
                      sensitivity=4.0):
    '''
    Cluster sequences in fastafile with MMSEQS.

    Parameters
    ----------
    fastafile : str
        Path to file containing sequences in fasta format.
    save_rep_seqs : bool
        Whether to save representative sequences of clusters.
    repfasta : str
        File to write representative sequences of clusters in fasta format.
    write_cluster_data : bool
        Whether to write cluster data to clusterfile in fasta format.
    clusterfile : str
        File to write cluster data in fasta format. The cluster identifier, representative
        sequence, and size are written as the header of the fasta file in the form, 
        cluster_no::cluster_rep::cluster_size. The member sequences of the clusters
        are written as the fasta sequence data.
    write_hamming_weights : bool
        Whether to write hamming weights for each sequence to csv file. The hamming weight
        for each sequence is computed as the reciprocal of the number of sequences in the 
        dataset with a hamming distance of less than (1 - minseqid) from the sequence.
    hamming_weights_file : str
        Path of file to write hamming weights.
    tempdir : str
        Directory to store temporary files generated during clustering.
    delete_temp_files : bool
        Whether to delete temporary files after clustering.
    seqidmode : int, [`seq-id-mode` in MMSEQS, default=0]
        Method of determing sequence identity from alignment
        0: alignment length,
        1: shorter,
        2: longer sequence.
    minseqid : float, [`min-seq-id` in MMSEQS, default=0.0]
        List matches above this sequence identity (for clustering) (range 0.0-1.0).
    maxseqs : int, [`max-seqs` in MMSEQS, default=20]
        Maximum results per query sequence allowed to pass the prefilter 
        (affects sensitivity).
    maxevalue : float, [`e` in MMSEQS, default=1e-3].
        List matches below this E-value.
    cluster_mode : int, ['cluster-mode' in MMSEQS, default=0]
        Method of clustering.
        0: Set-Cover (greedy),
        1: Connected component (BLASTclust),
        2,3: Greedy clustering by sequence length (CDHIT).
    cover : float, [`c` in MMSEQS, default=0.8]
        List matches above this fraction of aligned (covered) residues.
    cov_mode : int, [`cov-mode` in MMSEQS, default=0]
        Method of determining coverage in pairwise alignment.
        0: coverage of query and target,
        1: coverage of target,
        2: coverage of query,
        3: target seq. length has to be at least x% of query length,
        4: query seq. length has to be at least x% of target length,
        5: short seq. needs to be at least x% of the other seq. length.
    sensitivity : float, [`s` in MMSEQS, default=4.0]
        Sensitivity of clustering (affects speed).
        1.0: faster,
        4.0: fast,
        7.5: sensitive.
    '''
  
    # Prepare directory for clustering files
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    subprocess.call(f'rm -rfv ./{tempdir}/*', shell=True) # Empty directory
    
    # Make mmseqs database from sequence fasta file
    subprocess.call(f'{mmseqs_exec} createdb {fastafile} ./{tempdir}/seqdb --shuffle 0',
                    shell=True)
    
    # Cluster database
    subprocess.call(f'{mmseqs_exec} cluster ./{tempdir}/seqdb ./{tempdir}/seqdb_cluster '\
                    f'./{tempdir}/tmp --min-seq-id {minseqid}  --max-seqs {maxseqs} '\
                    f'-e {maxevalue} --cluster-mode {cluster_mode} --cov-mode {cov_mode} '\
                    f'-c {cover} --seq-id-mode {seqidmode} -s {sensitivity}', shell=True)
    
    # Create tsv file from cluster output
    subprocess.call(f'{mmseqs_exec} createtsv ./{tempdir}/seqdb ./{tempdir}/seqdb '\
                    f'./{tempdir}/seqdb_cluster ./{tempdir}/seqdb_cluster.tsv',
                    shell=True)
    
    # Extract representative sequences
    if save_rep_seqs:
        subprocess.call(f'{mmseqs_exec} createsubdb ./{tempdir}/seqdb_cluster '\
                        f'./{tempdir}/seqdb ./{tempdir}/seqdb_rep', shell=True)
        subprocess.call(f'{mmseqs_exec} convert2fasta ./{tempdir}/seqdb_rep {repfasta}',
                        shell=True)
        
    # Write cluster data in fasta-like format
    if write_cluster_data:
        # Parse cluster tsv file
        df = pd.read_csv(f'./{tempdir}/seqdb_cluster.tsv', index_col=0, 
                         header=None, sep='\t')
        reps = np.unique(df.index)
        clusters = [df.loc[(df.index==rep),:].values.reshape(-1) for rep in reps]
        
        # Write cluster data
        with open(clusterfile, 'w') as file:
            for i,cluster in enumerate(clusters):
                file.write(f'>cluster{i}::{reps[i]}::{len(cluster)}\n')
                file.write(f"{', '.join(cluster)}\n")
    
    # Compute hamming weights
    if write_hamming_weights:
        df = pd.read_csv(f'./{tempdir}/seqdb_cluster.tsv', index_col=0, header=None, sep='\t')
        weights = [(df.index==df.index[i]).sum() for i in range(len(df))]
        weights = 1 / np.array(weights)    
        dfweights = pd.DataFrame(weights, index=df.iloc[:,0])
        dfweights.to_csv(hamming_weights_file)
    
    # Remove temp files
    if delete_temp_files:
        subprocess.call(f'rm -rfv ./{tempdir}', shell=True)
    
    return






def split_cluster(clusterfile, testsize=0.2, random_seed=None, verbose=True):
    '''Randomly split clustered sequences in clusterfile into training and testing sets'''
    
    # Read cluster data
    heads, clusters = read_fasta(clusterfile)
    counts = [int(each_head.split('::')[-1]) for each_head in heads]
    totalcount = np.sum(counts)

    # Randomly shuffle data
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle_locs = np.random.choice(range(len(heads)), len(heads), replace=False)
    heads = np.array(heads)[shuffle_locs]
    clusters = np.array(clusters)[shuffle_locs]
    
    
    # Split to train/test sets
    trainset, testset = [], []
    max_test_count = np.int(testsize * totalcount)
    test_count = 0
    
    for i,cluster in enumerate(clusters):
        seqs = cluster.split(', ')
        size = len(seqs)

        if test_count <= max_test_count:
            testset.extend(seqs)
            test_count += size
        else:
            trainset.extend(seqs)
                    
    if verbose:
        print('Test set has {} sequences ({:.1f}%)'.format(
                len(testset), len(testset)/totalcount * 100))
        print('Train set has {} sequences ({:.1f}%)'.format(
                len(trainset), len(trainset)/totalcount * 100))
    
    return (trainset, testset)






def mafft_MSA(read_path, write_path, mafft_exec='/usr/local/bin/mafft'):
    '''Align sequences in fasta file (read_path) with MAFFT (executable in mafft_exec), 
    and write aligned sequences to write_path.'''
    
    # Align sequences    
    heads1, sequences1 = read_fasta(read_path)
    accs1 = get_accession(read_path)
    Mafft_cline = MafftCommandline(mafft_exec, input=read_path)
    stdout, stderr = Mafft_cline()
    
    # Write Mafft output
    with open(write_path, 'w') as store:
        store.write(stdout)
    
    # Rearrange alignment output to original order
    heads2, sequences2 = read_fasta(write_path)    
    accs2 = get_accession(write_path)
    with open(write_path, 'w') as fileobj:
        for acc in accs1:
            loc = accs2.index(acc)
            fileobj.write('>' + heads2[loc] + '\n' + sequences2[loc] + '\n')
            
    
        
        
    
    
    




        
        
        
def hmmsearch_fxn(hmmfile, seqfile, hmmsearch_exec, threshold=0, tempdir='./tempdir'):
    '''Search a profile hidden markov Model against a set of sequences. Output files
    are written to tempdir.'''
    
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
        
    command = f'{hmmsearch_exec} -o {tempdir}/output.txt --tblout {tempdir}/tblout.txt '\
              f'-A {tempdir}/aln.sto -T {threshold} --domT {threshold} '\
              f'--incT {threshold}  --incdomT {threshold} --noali {hmmfile} {seqfile}'
    stdout = subprocess.check_output(command, shell=True)
    sto_to_fasta(f'{tempdir}/aln.sto', f'{tempdir}/aln.fasta')
    
    return stdout













def seq_identity(seq1, seq2, gap = '-', short=False):
    ''' Return the percentage sequence identity between 2 aligned sequences. 
    If short==True, seq_identity is calculated relative to the shortest sequence, else, 
    relative to the longest sequence.'''
    
    arr1, arr2, gapseq = [np.fromstring(arr, dtype='S1') \
                          for arr in [seq1, seq2, gap * len(seq1)]]
    counts = np.sum((arr1 == arr2) * (arr1 != gapseq))
    if short:
        seqlen = min(len(seq1.replace('-','')), len(seq2.replace('-','')))
    else:
        seqlen = max(len(seq1.replace('-','')), len(seq2.replace('-','')))
    
    return counts / seqlen * 100






def seqid_matrix(fasta, align=False, mafft_exec='/usr/local/bin/mafft'):
    '''Return a matrix (pandas dataframe) of pairwise sequence identities for all 
    sequences in a fasta file. The accession codes of sequences are the index and column
    names of the dataframe.  If align is True, sequences are aligned with Mafft before 
    analysis.'''
    
    # Retrieved aligned sequences
    [heads, seqs] = read_fasta(fasta)
    acc_list = get_accession(fasta)
    lengths = [len(seq) for seq in seqs]
    lengths = set(lengths)

    if (len(lengths) > 1):
        
        if align:
            msafasta = '_' + fasta
            mafft_MSA(fasta, msafasta, mafft_exec)
            [heads, seqs] = read_fasta(msafasta)
            subprocess.call(f'rm -rf {msafasta}', shell=True)
        
        else:
            error = 'Sequences are not of equal length and must be first aligned.'
            raise ValueError(error)
    
    # Sequence identity matrix of aligned sequences
    iden_mat = np.zeros((len(seqs), len(seqs)))

    for i in range(len(seqs)):

        for k in range(i, len(seqs)):
            
            if i == k:
                iden_mat[i,k] = 100
            else:
                iden = seq_identity(seqs[i], seqs[k])
                iden_mat[i,k] = iden
                iden_mat[k,i] = iden
    
    # Return identity matrix as dataframe
    df = pd.DataFrame(iden_mat, index=acc_list, columns=acc_list)
    
    return df
    











def calculate_NDCG(y_pred, y_true, normalize=True, power=2):
    '''Return the normalized discounted cumulative gain. If normalize is True, data in
    y_pred and y_true are first normalized to a mean of 0 and std. dev of 1.'''
    
    assert len(y_true) == len(y_pred), 'y_true and y_pred are not of same length'
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if (len(set(y_true)) == 1) or (len(set(y_pred)) == 1):
        return 0

    if normalize:
        y_true = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-8)
    if power is not None:
        y_true = np.power(power, y_true)
    rank_pred = rankdata(y_pred, method='ordinal') 
    rank_pred = max(rank_pred) - rank_pred + 1
    rank_true = rankdata(y_true, method='ordinal')
    rank_true = max(rank_true) - rank_true + 1
    DCG = np.sum(y_true / np.log2(rank_pred + 1))
    IDCG = np.sum(y_true / np.log2(rank_true + 1))
    NDCG = DCG / IDCG
    
    return NDCG






def corrAverage(corrs, sizes, fisher=True):
    '''Return the weighted average value of correlations from multiple samples.
    If fisher is True, correlation values are converted  to Fisher's Z before computing 
    the weighted average, to mitigate the the bias from the skewness of
    correlation values.
    Based on Corey, Dunlap, and Burke, 1998 (DOI: 10.1080/00221309809595548), and 
    Alexander, 1990 (DOI: 10.3758/BF03334037).
    
    corrs: An array of correlation values of different samples
    
    sizes: An array of the sizes of the samples for which correlation values were derived
    
    fisher: Whether to convert correlation values to Fisher Z values.
    '''
    
    corrs, sizes = np.asarray(corrs).flatten(), np.asarray(sizes).flatten()

    # Transform correlation values into a Fisher's Z
    if fisher:
        corrs = 0.5 * np.log((1 + corrs + 1e-6) / np.abs(1 - corrs + 1e-6))

    # Weighted average
    avg_corr = np.sum(corrs * sizes) / np.sum(sizes)
    
    # Convert Fisher's Z back to correlation values
    if fisher:
        avg_corr = (np.exp(2*avg_corr) - 1) / (np.exp(2 * avg_corr) + 1)
    
    
    return avg_corr






def corrAverageConfInt(corrs, sizes, bootstrap=1000, seed=0, fisher=True, multiplier=1.96):
    '''Return the confidence interval of fisher-transformed weighted average of 
    correlations'''
    
    corrs, sizes = np.asarray(corrs), np.asarray(sizes)
    probs = sizes / np.sum(sizes) # Account for sample size in bootstrapping
    N = len(corrs)
    avg_corrs = []
    np.random.seed(seed)
    for i in range(bootstrap):
        locs = np.random.choice(range(N), size=N, replace=True, p=probs) 
        avg_corrs.append(corrAverage(corrs[locs], sizes[locs], fisher=fisher))
    confint =  np.std(avg_corrs) / np.sqrt(len(corrs)) * multiplier

    return confint       
    
    




def meanResidualError(ytrue, ypred, normalize=False):
    '''Fit a linear line scale predicted values to normalized ground truth values and 
    compute the mean residual error'''
    
    ytrue, ypred = np.asarray(ytrue), np.asarray(ypred)
    if normalize:
        ytrue = (ytrue - np.max(ytrue)) / (np.max(ytrue) - np.min(ytrue) + 1e-8)
    linear = LinearRegression()
    linear = linear.fit(ypred.reshape(-1,1), ytrue)
    ypred_fitted = linear.predict(ypred.reshape(-1,1))
    mre = np.mean(np.abs(ypred_fitted - ytrue))

    return mre, ypred_fitted        






def mann_whitney_test(df, use_size=True):
    '''Return a dataframe of p-values for two-sided Mann-Whittney U-tests comparing the
    distribution of all columns in df. If samples are of different sizes, a 'size'column
   should include the sizes in df'''
   
    columns = list(df.columns)
    if use_size:
        assert 'size' in columns
        columns.remove('size')
    pvalues = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:

        for col2 in columns:

            if use_size:
                x = np.repeat(df[col1].values, df['size'].values.astype(int))
                y = np.repeat(df[col2].values, df['size'].values.astype(int))

            else:
                x = df[col1].values
                y = df[col2].values

            u1, p = mannwhitneyu(x, y, alternative='two-sided', method='asymptotic')
            pvalues.loc[col1, col2] = p
    pvalues = pvalues.astype(float)
    
    return pvalues








def get_rank_pairs(scores):
    '''Derive pairwise rank classification scores (0/1) for all pairs in a dataset'''

    pairs = np.array(list(itertools.permutations(range(len(scores)), 2)))
    index = [f'{scores.index[pairs[i,0]]} : {scores.index[pairs[i,1]]}' \
             for i in range(len(pairs))]
    ranked = pd.DataFrame(index=index)
    ranked['X1'] = scores.iloc[pairs[:,0]].values
    ranked['X2'] = scores.iloc[pairs[:,1]].values
    ranked['diff'] = ranked['X1'] - ranked['X2']
    ranked['y'] = (ranked['diff'] > 0).astype(int)
    
    return ranked

    
    
def bootstrap_locs(y, rep=100):

    return [np.random.choice(range(len(y)), size=len(y), replace=True) \
            for i in range(rep)]



            
def rank_performance(ytrue, ypred, weights=None, rep=100):

    out = {}
    bootlocs = bootstrap_locs(ytrue, rep=rep)
    out['accuracy'] = metrics.accuracy_score(ytrue, ypred)
    out['accuracy_err'] = np.std([metrics.accuracy_score(ytrue[loc], ypred[loc]) \
                                  for loc in bootlocs]) / np.sqrt(rep) * 1.96
    out['mcc'] = metrics.matthews_corrcoef(ytrue, ypred, sample_weight=weights)
    out['mcc_err'] = np.std([metrics.matthews_corrcoef(ytrue[loc], ypred[loc], 
                                                       sample_weight=weights) \
                                  for loc in bootlocs]) / np.sqrt(rep) * 1.96
    
    return out
    
    
    
'''
locs = ['size'] + [item for item in dfperf.columns if 'rho' in item]
df = dfperf.loc[:,locs]
pvalues = mann_whitney_test(df, use_size=True)
'''



"""