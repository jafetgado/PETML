"""
Helper functions 
"""




import numpy as np
import pandas as pd
from Bio import AlignIO
import os
import subprocess






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






def canonize(seq, remove=list('BJOUXZ'), replace='-'):
    '''Replace non-canonical characters with a specified character'''
    
    for aa in remove:
        seq = seq.replace(aa, replace)
        
    return seq






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







    