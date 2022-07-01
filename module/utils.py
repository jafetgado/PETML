"""
Utility functions for preprocessing and analysis
"""




#============#
# Imports
#============#
import numpy as np
import pandas as pd
import subprocess
import os
from scipy.stats import spearmanr
from scipy.stats import rankdata
from Bio.Align.Applications import MafftCommandline
from Bio import AlignIO, SeqIO








#===============#
# Constants
#===============#

AMINO_LETTERS = list('-ACDEFGHIKLMNPQRSTVWYX') # Non-canonical (X), gap/stop char (-)
NONCANONICAL = list('BJOUZ')
AMINO_DICT = {char:num for (num,char) in enumerate(AMINO_LETTERS)}
REVERSE_AMINO_DICT = {num:char for (num,char) in enumerate(AMINO_LETTERS)}








#============#
# Functions
#============#

def deliner(path):
    '''Remove line breaks within sequence data of a fasta file (path)'''

    data = []

    with open(path, 'r') as file_input:
        
        for line in file_input:
            text = line.strip()
    
            if text.startswith('>'):
                data.append(text)
                is_header = True
            else:
                if is_header == True:
                    data.append(text)						
                elif is_header == False:
                    data[-1] += text
                is_header = False

    with open(path, 'w') as file_output:
        for ele in data:
            file_output.write(ele + '\n')
    
    return






def read_fasta(fasta, return_as_dict=False):
    '''Read the protein sequences in a fasta file. If return_as_dict is True, return a 
    dictionary with headers as keys and sequences as values, else return a tuple, 
    (list_of_headers, list_of_sequences).'''
    
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






def write_fasta(headers, seqdata, path):
    '''Write a fasta file (path) from a list of headers and a corresponding list 
    of sequences (seqdata)'''
    
    with open(path, 'w') as pp:
        for i in range(len(headers)):
            pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')






def get_accession(path):
    ''' Return a list of accession codes for all sequences in a fasta file (path).'''
    
    [heads, seqs] = read_fasta(path)
    accessions =  [head.split()[0] for head in heads]
    
    return accessions






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
    write_fasta(heads, sequences, fasta)
    
    return





    
def mafft_MSA(read_path, write_path, mafft_exec='/usr/local/bin/mafft'):
    '''Align sequences in fasta file (read_path) with MAFFT (executable in mafft_exec), 
    and write aligned sequences to write_path.'''
    
    # Align sequences    
    [heads1, sequences1] = read_fasta(read_path)
    Mafft_cline = MafftCommandline(mafft_exec, input=read_path)
    stdout, stderr = Mafft_cline()
    
    # Write Mafft output
    with open(write_path, 'w') as store:
        store.write(stdout)
    
    # Rearrange alignment output to original order
    [heads2, sequences2] = read_fasta(write_path)    
    with open(write_path, 'w') as fileobj:
        for i in range(len(heads1)):
            posi = heads2.index(heads1[i])
            fileobj.write('>' + heads2[posi] + '\n' + sequences2[posi] + '\n')            






def hmmbuild_fxn(msafile, hmmfile, hmmbuild_exec):
    '''Build a profile hidden markov Model from aligned sequences in msafile with the
    HMMER hmmbuild function. Return the stdout as a string'''
    
    command = f'{hmmbuild_exec} {hmmfile} {msafile}'
    stdout = subprocess.check_output(command, shell=True)
    
    return stdout






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





def sto_to_fasta(sto_file, fasta_file):
    '''Convert hmmer sequence output file from stockholm format to fasta format'''
    
    align = AlignIO.read(sto_file, 'stockholm')
    _ = AlignIO.write(align, fasta_file, 'fasta')
    _ = deliner(fasta_file)
    [heads, seqs] = read_fasta(fasta_file)
    seqs = [seq.upper() for seq in seqs]
    _ = write_fasta(heads, seqs, fasta_file)
    
    return






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






def one_hot_encode_sequence(sequence):
    '''Convert an amino acid sequence to a 2D array with amino acid position as rows and 
    amino acids as one-hot encoded columns'''
    
    array = np.zeros((len(sequence),len(AMINO_DICT))) 
    
    # Replace all non-canonical amino acids with X
    for char in NONCANONICAL:
        sequence = sequence.replace(char, 'X')
    
    # One-hot encode sequence
    for (i, char) in enumerate(sequence):
        array[i, AMINO_DICT[char]] = 1

    return array






def calculate_NDCG(y_pred, y_true, normalize=True):
    '''Return the normalized discounted cumulative gain. If normalize is True, data in
    y_pred and y_true are first normalized to a mean of 0 and std. dev of 1.'''
    
    assert len(y_true) == len(y_pred), 'y_true and y_pred are not of same length'
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if normalize:
        y_true = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-8)
    rank_pred = rankdata(y_pred, method='ordinal') 
    rank_pred = max(rank_pred) - rank_pred + 1
    rank_true = rankdata(y_true, method='ordinal')
    rank_true = max(rank_true) - rank_true + 1
    DCG = np.sum(y_true / np.log2(rank_pred + 1))
    IDCG = np.sum(y_true / np.log2(rank_true + 1))
    NDCG = DCG / IDCG
    
    return NDCG