"""
Utility functions for preprocessing and analysis
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import subprocess

from Bio.Align.Applications import MafftCommandline
from Bio import Entrez, SeqIO, AlignIO







AMINO_LETTERS = list('-ACDEFGHIKLMNPQRSTVWYX') # Non-canonical (X), gap/stop char (-)
NONCANONICAL = list('BJOUZ')
AMINO_DICT = {char:num for (num,char) in enumerate(AMINO_LETTERS)}
REVERSE_AMINO_DICT = {num:char for (num,char) in enumerate(AMINO_LETTERS)}







def deliner(path):
    '''Remove line breaks within sequence data of a fasta file (path) and save the result
    by overwritting the original file'''

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
    '''Read the protein sequences in a fasta file. If return_as_dict, return a dictionary
    with headers as keys and sequences as values, else return a tuple, 
    (list_of_headers, list_of_sequences)'''
    
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
    
    return
            





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
    
    
    
    
    
    
def remove_gaps(input_file, output_file, gap='-', replacewith=''):
    '''Remove all gap characters from sequences in input_file and write a sequences to
    output_file'''
    
    [head, seq] = read_fasta(input_file)
    new_seq = [x.replace('-', replacewith) for x in seq]
    write_fasta(head, new_seq, output_file)
    
    return






def sto_to_fasta(sto_file, fasta_file):
    '''Convert hmmer sequence output file from stockholm format to fasta format'''
    
    align = AlignIO.read(sto_file, 'stockholm')
    _ = AlignIO.write(align, fasta_file, 'fasta')
    _ = deliner(fasta_file)
    [heads, seqs] = read_fasta(fasta_file)
    seqs = [seq.upper() for seq in seqs]
    _ = write_fasta(heads, seqs, fasta_file)
    
    return






def mafft_MSA(read_path, write_path, Mafft_exe='/usr/local/bin/mafft'):
    '''Align sequences in fasta file (read_path) with MAFFT (executable in Mafft_exe). Write
    aligned sequences to write_path.
    '''
    
    # Align sequences    
    [heads1, sequences1] = read_fasta(read_path)
    Mafft_cline = MafftCommandline(Mafft_exe, input=read_path)
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

    return 
             





def seq_identity(seq1, seq2, gap = '-', short=True):
    ''' Calculates the percentage sequence identity between 2 aligned sequences. If short
    is true, seq_identity is calculated relative to the shortest sequence, else, relative
    to the longest sequence.'''
    
    arr1, arr2, gapseq = [np.fromstring(arr, dtype='S1') \
                          for arr in [seq1, seq2, gap * len(seq1)]]
    counts = np.sum((arr1 == arr2) * (arr1 != gapseq))
    if short:
        seqlen = min(len(seq1.replace('-','')), len(seq2.replace('-','')))
    else:
        seqlen = max(len(seq1.replace('-','')), len(seq2.replace('-','')))
    
    return counts / seqlen






def seqid_matrix(fasta, align=False):
    '''Returns a matrix (DataFrame) of pairwise sequence identities for all 
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
            deliner(fasta)
            msafasta = '_' + fasta
            mafft_MSA(fasta, msafasta)
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
                iden_mat[i,k] = 1.0
            else:
                iden = seq_identity(seqs[i], seqs[k])
                iden_mat[i,k] = iden
                iden_mat[k,i] = iden
    
    # Return identity matrix as dataframe
    df = pd.DataFrame(iden_mat, index=acc_list, columns=acc_list)
    
    return df






def pad_sequence(sequence, maxlen=600, padtype='post', padchar='-'):
    '''Pad amino acid sequence to by adding a pad character to the sequence'''
    
    sequence = sequence.upper()
    seqlen = len(sequence)
    assert seqlen <= maxlen, 'Sequence cannot be longer than maxlen'
    addlen = maxlen - seqlen
    if padtype == 'pre':
        sequence = (padchar * addlen) + sequence
    elif padtype == 'post':
        sequence = sequence + (padchar * addlen)
    else:
        raise ValueError("padtype must be 'pre' or 'post'")
        
    return sequence






def categorical_encode_sequence(sequence):
    '''Convert an amino acid sequence to a 1D array of integer encodings'''

    for char in NONCANONICAL:
        sequence = sequence.replace(char, 'X')
    seq_encoding = np.array([AMINO_DICT[char] for char in sequence])
    
    return seq_encoding

    
    


    
def categorical_to_one_hot(array, maxlen=600, feature_dim=22):
    '''Convert a 2D categorical encoded array (None, maxlen) to a one-hot encoded 3D 
    array (None, maxlen, feature_dim)'''
    
    array3d = np.zeros((len(array), maxlen, feature_dim))
    for dim1 in range(len(array)):
        for dim2, dim3 in enumerate(array[dim1,:]):
            array3d[dim1, dim2, dim3] = 1
            
    return array3d
            
            
    



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





    

def reverse_one_hot_encode_sequence(array):
    '''Convert a 2D sequence encoding (amino acid index as rows, encoding/probability as
    columns) to a sequence string'''
    
    seq_as_int = np.argmax(array, axis=-1).reshape(-1)
    seq_as_str = ''.join([REVERSE_AMINO_DICT[num] for num in seq_as_int])
    
    return seq_as_str









    

    

def hamming_weights(fastafile, minseqid=0.8, maxseqs=None, 
                    output_file='hamming_weights.csv', delete_temp_files=False,
                    MMSEQS_EXEC='/home/jgado/condaenvs/tfgpu/bin/mmseqs'):
    '''Return weights for each sequence in a fasta file. Weights are calculated as the
    inverse count of number of sequences with a hamming distance below the specified 
    threshold. Sequence identity is computed with mmseqs2'''
    
    # Get accessions of sequences in fasta file
    accessions = get_accession(fastafile)
    if maxseqs is None:
        maxseqs = len(accessions)
    
    # Create folder for temporarily storing results
    subprocess.call('mkdir ./tempdir', shell=True)
    
    # Create mmseqs database from sequences
    subprocess.call(f'{MMSEQS_EXEC} createdb {fastafile} ./tempdir/seqdb --shuffle 0',
                    shell=True)
    
    # Search database for hits with hamming distance above threshold
    subprocess.call(f'{MMSEQS_EXEC} search ./tempdir/seqdb ./tempdir/seqdb '\
                    f'./tempdir/results ./tempdir/tmp -s 1 --min-seq-id {minseqid} '\
                    f'--max-seqs {maxseqs}', shell=True)
    
    # Create tsv file from search output
    subprocess.call(f'{MMSEQS_EXEC} createtsv ./tempdir/seqdb ./tempdir/seqdb '\
                    f'./tempdir/results ./tempdir/results.tsv', shell=True)
    
    
    # Count number of hits for each sequence
    counts = np.ones((len(accessions),))
    for (i,acc) in enumerate(accessions):
        output = subprocess.check_output(f'grep "^{acc}" ./tempdir/results.tsv | wc',shell=True)
        counts[i] = int(output.decode('utf-8').split()[0])
    weights = 1 / counts
    weights = pd.DataFrame([accessions, weights]).transpose()
    
    # Write weights to disk
    if output_file is not None:
        weights.to_csv(output_file)
    
    # Delete temp files
    if delete_temp_files:
        subprocess.call('rm -rfv ./tempdir', shell=True)
        
    return weights
    
   
    
   
def cluster_sequences(fastafile, clusterfile, minseqid=0.7, 
                      delete_temp_files=True, parse=False,
                      MMSEQS_EXEC='/home/jgado/condaenvs/tfgpu/bin/mmseqs'):
    '''Cluster sequences in fastafile to clusters with minimum sequence identity of 
    minseqid. Accession codes of sequences in each cluster are written to clusterfile in 
    a fasta-like format'''
    
    # Make mmseqs database from sequence fasta file
    subprocess.call('mkdir ./tempdir', shell=True)
    subprocess.call(f'{MMSEQS_EXEC} createdb {fastafile} ./tempdir/seqdb --shuffle 0',
                    shell=True)
    
    # Cluster database
    subprocess.call(f'{MMSEQS_EXEC} cluster ./tempdir/seqdb ./tempdir/seqdb_cluster '\
                    f'./tempdir/tmp --min-seq-id {minseqid}', shell=True)
    
    # Create tsv file from cluster output
    subprocess.call(f'{MMSEQS_EXEC} createtsv ./tempdir/seqdb ./tempdir/seqdb '\
                    f'./tempdir/seqdb_cluster ./tempdir/seqdb_cluster.tsv', shell=True)
    
    if parse:
        # Parse cluster tsv file
        df = pd.read_csv('./tempdir/seqdb_cluster.tsv', index_col=0, header=None, sep='\t')
        reps = np.unique(df.index)
        clusters = [df.loc[(df.index==rep),:].values.reshape(-1) for rep in reps]
        
        # Write cluster data in fasta-like format
        with open(clusterfile, 'w') as file:
            for i,cluster in enumerate(clusters):
                file.write(f'>cluster{i}::{reps[i]}::{len(cluster)}\n')
                file.write(f"{', '.join(cluster)}\n")
        
        # Remove temp files
        if delete_temp_files:
            subprocess.call('rm -rfv ./tempdir', shell=True)
        return clusters




def split_cluster(clusterfile, testsize=0.2, random_seed=None, verbose=True):
    '''Randomly split clustered sequences in clusterfile into training and testing sets
    '''
    
    # Read cluster data
    heads, clusters = read_fasta(clusterfile)
    counts = [int(each_head.split('::')[-1]) for each_head in heads]
    totalcount = np.sum(counts)

    # Randomly shuffle data
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle_locs = np.random.choice(range(len(heads)), len(heads), replace=False)
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
    
  
    
  


#=========================================#
# Functions for model analysis
#=========================================#


def plotHistory(history, losstype='loss', savepath=None):
    fig, ax1 = plt.subplots()
    ax1.plot(history.history[losstype], label='Training', color='dodgerblue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(f'Training {losstype}')
    ax2 = ax1.twinx()
    ax2.plot(history.history[f'val_{losstype}'], label='Validation', 
             color='indianred')
    ax2.set_ylabel(f'Validation {losstype}')
    fig.legend(bbox_to_anchor=(0.86, 0.46), loc='right', ncol=1)
    plt.title(f'{losstype} in model history')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show(); plt.close()
