"""
Custom losses for training Tensorflow/Keras models
"""




import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error


"""
Utility functions for preprocessing and analysis
"""




#==============#
# Imports
#==============#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import subprocess

from Bio.Align.Applications import MafftCommandline
from Bio import Entrez, SeqIO, AlignIO






#===============#
# Variables
#===============#

amino_letters = list('-ACDEFGHIKLMNPQRSTVWYX') # X: all non-canonical AAs; -: stop char
amino_dict = {char:num for (num,char) in enumerate(amino_letters)}
reverse_amino_dict = {num:char for (num,char) in enumerate(amino_letters)}






#=========================================#
# Functions for processing sequence data
#=========================================#

def deliner(path):
	'''Remove line breaks within sequence data of a fasta file (path) and save the result
    by overwritting the original file.'''

	file_input = open(path, 'r')
	data = []
	
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
	
	file_input.close()

	with open(path, 'w') as file_output:
		for ele in data:
			file_output.write(ele + '\n')




def read_fasta(fasta, return_as_dict=False):
    '''
    Read the protein sequences in a fasta file. If return_as_dict, return a dictionary
    with headers as keys and sequences as values, else return a tuple: 
        (list_of_headers, list_of_sequences).    	
    '''
    
    with open(fasta, 'r') as fast:
        headers, sequences = [], []
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
	'''
    Write a fasta file (path) from a list of headers and a corresponding list 
    of sequences (seqdata).
    '''
    
	with open(path, 'w') as pp:
		for i in range(len(headers)):
			pp.write('>' + headers[i] + '\n' + seqdata[i] + '\n')
            



def mafft_MSA(read_path, write_path, Mafft_exe='/usr/local/bin/mafft'):
    '''
    Align sequences in fasta file (read_path) with MAFFT (executable in Mafft_exe). Write
    aligned sequences to write_path.
    '''
    
    # Align sequences    
    [heads1, sequences1] = read_fasta(read_path)
    Mafft_cline = MafftCommandline(Mafft_exe, input=read_path)
    stdout, stderr = Mafft_cline()
    
    # Write Mafft output
    with open(write_path, 'w') as store:
        store.write(stdout)
    
    # Reorder alignment output
    [heads2, sequences2] = read_fasta(write_path)    
    with open(write_path, 'w') as fileobj:
        for i in range(len(heads1)):
            posi = heads2.index(heads1[i])
            fileobj.write('>' + heads2[posi] + '\n' + sequences2[posi] + '\n')            
            
       
            
       
def fasta_to_df(fasta):
    '''
    Read aligned sequences in fasta format and return a pandas Dataframe with
    sequences as indexes and residues as columns.
    '''
    
    heads, sequences = read_fasta(fasta)
    data = [list(seq) for seq in sequences]
    lengths = [len(seq) for seq in data]
    assert len(set(lengths)) == 1, f'Sequences in {fasta} are of unequal length!'
    df = pd.DataFrame(data)
    df.index = [head.replace('>', '') for head in heads]
    
    return df            




def df_to_fasta(df, fasta):
    '''
    Write an MSA dataframe (with sequences as indexes and residues as columns) to a 
    fasta file
    '''
    
    heads = df.index.values
    sequences = [''.join(df.iloc[i,:].values) for i in range(len(df))]
    write_fasta(heads, sequences, fasta)




def get_sequence(Accession):
    ''' Retrieves the sequence for a given accession number from the NCBI
    database. Returns [description, sequence].'''
    
    
    Entrez.email = 'japhethgado@gmail.com'
    handle = Entrez.efetch(db='protein', id=Accession, rettype='fasta', \
						   retmode='text')
    record = SeqIO.read(handle, 'fasta')
    desc = record.description
    sequ = str(record.seq)
    
    return [desc, sequ]




def remove_gaps(input_file, output_file, gap='-', replacewith=''):
    '''Remove all gaps from sequences in input_file and saves the result in 
    output_file.'''
    
    [head, seq] = read_fasta(input_file)
    new_seq = [x.replace('-', replacewith) for x in seq]
    write_fasta(head, new_seq, output_file)




def get_accession(path):
    ''' Returns a list of all accession codes from a genbank fasta file (path).'''
    
    [h,s] = read_fasta(path)
    
    return [x.split()[0] for x in h]




def seq_identity(seq1, seq2, gap = '-', short=True):
    ''' Calculates the percentage sequence identity between 2 aligned sequences
    relative to the shortest sequence. If short is set to false, the identity is 
    calculated relative to the longest sequence.'''
    
    arr1, arr2, gapseq = [np.fromstring(arr, dtype='S1') \
                          for arr in [seq1, seq2, gap * len(seq1)]]
    counts = np.sum((arr1 == arr2) * (arr1 != gapseq))
    if short:
        seqlen = min(len(seq1.replace('-','')), len(seq2.replace('-','')))
    else:
        seqlen = max(len(seq1.replace('-','')), len(seq2.replace('-','')))
    
    return counts / seqlen




def seqid_matrix(fasta):
    '''Returns a matrix (DataFrame) of pairwise sequence identities for all 
    sequences in a fasta file (fasta). The index and column names of the dataframe
    are the accesion codes of the sequences in the fasta file. If align=True,
    the a Mafft MSA is carried out before alignment.'''
    
    # Align, if sequences are not of same length
    [heads,seqs] = read_fasta(fasta)
    acc_list = get_accession(fasta)
    lengths = [len(seq) for seq in seqs]
    lengths = set(lengths)

    if len(lengths) > 1:
        # Multiple sequence alignment
        deliner(fasta)
        mafft_MSA(fasta,fasta)

        # Read aligned sequences
        [heads,seqs] = read_fasta(fasta)
        
        # Then remove gaps to return to unaligned
        remove_gaps(fasta, fasta)
    
    # Sequence identity matrix
    iden_mat = np.zeros((len(seqs), len(seqs)))

    for i in range(len(seqs)):

        for k in range(i, len(seqs)):
            
            if i == k:
                iden_mat[i,k] = 1.0
            else:
                iden = seq_identity(seqs[i], seqs[k])
                iden_mat[i,k] = iden
                iden_mat[k,i] = iden
    
    # Return as dataframe
    df = pd.DataFrame(iden_mat, index=acc_list, columns=acc_list)
    
    return df




def pad_sequence(sequence, maxlen=600, padtype='pre', padchar='-'):
    '''Pad amino acid sequence to maxlen by adding padchar before (pre) or after (post)
    the sequence'''
    
    sequence = sequence.upper()
    seqlen = len(sequence)
    assert seqlen <= maxlen
    addlen = maxlen - seqlen
    if padtype == 'pre':
        sequence = (padchar * addlen) + sequence
    elif padtype == 'post':
        sequence = sequence + (padchar * addlen)
    else:
        raise ValueError("padtype must be 'pre' or 'post'")
        
    return sequence




def categorical_encode_sequence(sequence, amino_dict=amino_dict):
    '''Convert an amino acid sequence to a 1D array with amino acids encoded as 
    integers'''

    for char in list('BJOUZ'):
        sequence = sequence.replace(char, 'X')
    seq_encoding = np.array([amino_dict[char] for char in sequence])
    
    return seq_encoding

    
    
    
def one_hot_encode_sequence(sequence, amino_dict=amino_dict, canonical=True):
    '''Convert an amino acid sequence to a 2D array with position as rows and amino acids
    as one-hot encoded columns'''
    
    # Initialize empty 2D array
    array = np.zeros((len(sequence),len(amino_dict))) 
    
    # Replace noncanonical characters with X
    if canonical:
        for char in list('BJOUZ'):
            sequence = sequence.replace(char, 'X')
    
    # One-hot encode sequence
    for (i, char) in enumerate(sequence):
        array[i, amino_dict[char]] = 1

    return array




def categorical_to_one_hot(array, maxlen=600, feature_dim=22):
    '''Convert a categorical encoded array (2D) to a 3D one-hot encoded array'''
    
    array3d = np.zeros((len(array), maxlen, feature_dim))
    for dim1 in range(len(array)):
        for dim2, dim3 in enumerate(array[dim1,:]):
            array3d[dim1, dim2, dim3] = 1
            
    return array3d
            
            
    assert len(array) == maxlen
    out_array = np.zeros((maxlen, feature_dim))
    for i,intval in enumerate(array):
        out_array[i, intval] = 1
    
    return out_array
    



def reverse_one_hot_encode_sequence(array, reverse_amino_dict=reverse_amino_dict):
    seq_as_int = array.argmax(axis=-1).reshape(-1)
    seq_as_str = ''.join([reverse_amino_dict[num] for num in seq_as_int])
    
    return seq_as_str




def jackhmmer_routine(seq_file, db_file, outpath='./jackhmmer_test_output', 
                      executable='/usr/local/bin/jackhmmer', threshold=100, 
                      iterations=2, return_hits=True):
    '''Perform iterative HMM search with jackhmmer and write results to outpath'''
    
    # Directory check
    assert os.path.exists(seq_file)
    assert os.path.exists(db_file)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    # Run jackhmmer
    command = f'{executable} -o {outpath}/jackhmmer_output.txt '\
              f'-A {outpath}/jackhmmer_alignment_stockholm.txt '\
              f'--tblout {outpath}/jackhmmer_tabout.txt '\
              f'-T {threshold} --incT {threshold} -N {iterations} --noali '\
              f'{seq_file} {db_file}'
    subprocess.call(command, shell=True)
    
    # Convert stockholm alignment to fasta
    align = AlignIO.read(f'{outpath}/jackhmmer_alignment_stockholm.txt', 'stockholm')
    _ = AlignIO.write(align, f'{outpath}/jackhmmer_alignment_fasta.txt', 'fasta')
    deliner(f'{outpath}/jackhmmer_alignment_fasta.txt')
    
    # Return aligned hits as tuple
    if return_hits:
        return read_fasta(f'{outpath}/jackhmmer_alignment_fasta.txt')




def sto_to_fasta(sto_file, fasta_file):
    '''Convert jackhmmer alignment file in stockholm format (sto_file) to fasta format
    (fasta_file)'''
    
    align = AlignIO.read(sto_file, 'stockholm')
    _ = AlignIO.write(align, fasta_file, 'fasta')
    _ = deliner(fasta_file)
    [heads, seqs] = read_fasta(fasta_file)
    seqs = [seq.upper() for seq in seqs]
    _ = write_fasta(heads, seqs, fasta_file)
    

    

def hamming_weights(fastafile, minseqid=0.8, maxseqs=None, 
                    output_file='hamming_weights.csv', delete_temp_files=False,
                    mmseqs_exec='/home/jgado/condaenvs/tfgpu/bin/mmseqs'):
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
    subprocess.call(f'{mmseqs_exec} createdb {fastafile} ./tempdir/seqdb --shuffle 0',
                    shell=True)
    
    # Search database for hits with hamming distance above threshold
    subprocess.call(f'{mmseqs_exec} search ./tempdir/seqdb ./tempdir/seqdb '\
                    f'./tempdir/results ./tempdir/tmp -s 1 --min-seq-id {minseqid} '\
                    f'--max-seqs {maxseqs}', shell=True)
    
    # Create tsv file from search output
    subprocess.call(f'{mmseqs_exec} createtsv ./tempdir/seqdb ./tempdir/seqdb '\
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
                      mmseqs_exec='/home/jgado/condaenvs/tfgpu/bin/mmseqs'):
    '''Cluster sequences in fastafile to clusters with minimum sequence identity of 
    minseqid. Accession codes of sequences in each cluster are written to clusterfile in 
    a fasta-like format'''
    
    # Make mmseqs database from sequence fasta file
    subprocess.call('mkdir ./tempdir', shell=True)
    subprocess.call(f'{mmseqs_exec} createdb {fastafile} ./tempdir/seqdb --shuffle 0',
                    shell=True)
    
    # Cluster database
    subprocess.call(f'{mmseqs_exec} cluster ./tempdir/seqdb ./tempdir/seqdb_cluster '\
                    f'./tempdir/tmp --min-seq-id {minseqid}', shell=True)
    
    # Create tsv file from cluster output
    subprocess.call(f'{mmseqs_exec} createtsv ./tempdir/seqdb ./tempdir/seqdb '\
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
