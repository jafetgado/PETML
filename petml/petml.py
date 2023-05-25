"""
PETML: predict PET hydrolase activity with machine learning
"""




import numpy as np
import pandas as pd
import subprocess
import os
import tensorflow as tf

import deepPETase.utils as utils
from deepPETase.generator import DataGenerator




















class DeepPETase():
    '''
    Class for prediction and engineering of PET-hydrolases with deep semi-supervised
    learning
    
    
    Parameters
    ----------
    outdir: str (default is './msaout')
        Path of directory to which MSA files will be written
    '''
    
    
    
    
    def __init__(self, path=None):
        '''Initialize DeepPETase'''
        
        # Paths       
        if path is None:
            path = '/Users/jgado/Dropbox/research/projects/deepPETase_project/'\
                   'project740_selection/deepPETase/deepPETase'
        self.petase_hmm_file = f'{path}/data/petase_hmm.txt'
        self.vae_msa_file = f'{path}/data/vae_msa.fasta'
        self.vae_raw_file = f'{path}/data/vae_raw.fasta'
        self.vae_positions_file = f'{path}/data/vae_aln_positions.csv'
        self.ensemble_model_path = f'{path}/data/deepPETase_ensemble_model.h5'
        self.canonical_petases = f'{path}/data/vae_msa_canonical.fasta'
        for path in [self.petase_hmm_file, self.vae_msa_file, self.vae_raw_file, 
                     self.vae_positions_file, self.ensemble_model_path, 
                     self.canonical_petases]:
            assert os.path.exists(path), f'{path} not found'
                          
        # Load ensemble model
        self.ensemble_model = tf.keras.models.load_model(self.ensemble_model_path)

    
    
    
    def alignWithHMM(self, 
                     seq_file,
                     threshold=100,
                     outdir='./hmm_outdir',
                     hmmsearch_exe='/usr/local/bin/hmmsearch'):
        '''
        Align sequences with the PETase HMM using hmmsearch. 
        Use this method to exclude flanking regions in the putative PETase sequence such 
        as auxiliary domains.

        Parameters
        ----------
        seq_file : str
            Path of file containing sequences to be aligned in fasta format
        threshold : float (default is 100)
            Score threshold in hmmsearch. Only sequences with scores above this threshold
            are considered.
        hmmsearch_exe : str (default is '/usr/local/bin/hmmsearch')
            Path to hmmsearch executable. Ensure that HMMER is properly installed on 
            your machine. 
        '''
        
        
        # Check paths
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        assert os.path.exists(seq_file), 'Cannot find seq_file'
        assert os.path.exists(hmmsearch_exe), 'Cannot find hmmsearch executable'
        
        
        # Run hmmsearch with subprocess 
        command = f'{hmmsearch_exe} -o {outdir}/output.txt '\
                  f'--tblout {outdir}/tab_output.txt -A {outdir}/aln.sto '\
                  f'-T {threshold} --domT {threshold} --incT {threshold} '\
                  f'--incdomT {threshold} --noali {self.petase_hmm_file} {seq_file}'
        _ = subprocess.check_output(command, shell=True)
        
        
        # Convert alignment output from stockholm to fasta
        utils.sto_to_fasta(f'{outdir}/aln.sto', f'{outdir}/aln.fasta')
        
        # Write unaligned hits in separate fasta file
        utils.remove_gaps(f'{outdir}/aln.fasta', f'{outdir}/aln_no_gaps.fasta')
        
        
        
    
    def alignWithMSA(self, 
                     seq_file,
                     outdir='./msa_outdir',
                     mafft_exe='/usr/local/bin/mafft'):
        '''
        Add sequences to the alignment used in training the VAE with mafft. The mafft 
        program should be installed on your machine.

        Parameters
        ----------
        seq_file : str
            Path of file containing unaligned sequences to be aligned.
        mafft_exe : str (default is '/usr/local/bin/mafft')
            Path to mafft executable
        '''
        
        # Check paths
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        assert os.path.exists(mafft_exe), 'Cannot find mafft executable'
        assert os.path.exists(seq_file), 'Cannot find seq_file'
        
        # Add sequences to VAE MSA with mafft, keeping the original MSA positions
        command = f'{mafft_exe} --keeplength --add {seq_file} {self.vae_msa_file} '\
                  f'> {outdir}/msa_all_positions.fasta'
        _ = subprocess.check_output(command, shell=True)
        
        
        # Select only added sequences from full alignment
        heads_vae, seqs_vae = utils.read_fasta(self.vae_msa_file)  # Sequences in VAE MSA
        heads, seqs = utils.read_fasta(f'{outdir}/msa_all_positions.fasta') 
        heads_select, seqs_select = heads[len(heads_vae):], seqs[len(seqs_vae):]
        utils.write_fasta(heads_select, seqs_select, f'{outdir}/msa_all_positions.fasta')
        
        # Select 437 positions used in training VAE from alignment (with > 5% occupancy)
        locs = pd.read_csv(self.vae_positions_file).values[:,-1]
        df = utils.fasta_to_df(f'{outdir}/msa_all_positions.fasta')
        assert df.shape[1] == len(seqs_vae[0]), \
            'MSA does not have the same number of positions as VAE MSA'
        df = df.iloc[:,locs]
        _ = utils.df_to_fasta(df, f'{outdir}/msa_vae_positions.fasta')
        
    
    
    
    def predictActivityFromEncodings(self, 
                                     X_seq, 
                                     batch_size=256,
                                     verbose=0):
        '''
        Predict the PET-hydrolase activity of categorical encodings of aligned sequences.

        Parameters
        ----------
        X_seq : 2d numpy array
            An array of integer/categorical encodings representing amino acids of 
            sequences to be predicted with shape (batch_size, seq_len).
            See utils.AMINO_DICT for integer encodings. An amino acid sequence (str) can
            be integer encoded with the function, utils.categorical_encode_sequence.
        batch_size : int (default is 256)
            Number of samples in each batch for TensorFlow/Keras model
        verbose : int (default is 0)
            Verbosity of Tensorflow/Keras model (1 > 2 > 0)

        Returns
        -------
        ypred : 1d numpy array
            An array of predicted PET-hydrolase activiity. Values range from -inf through
            0 to +inf, and higher values indicate better activity.
        '''
        
        X_seq = np.asarray(X_seq, dtype=np.int32)
        assert X_seq.shape[1] == 437, \
          'Sequences have X_seq.shape[1] positions but the model requires 437 positions'
        datagenerator = DataGenerator(X_seq=X_seq, seq_len=437, weight=None,
                                      batch_size=batch_size, shuffle=False)
        ypred = self.ensemble_model.predict(datagenerator, verbose=verbose)
        
        return ypred
    
    
    
    
    def predictActivityFromMSA(self, 
                               msa_file, 
                               out_file='./predictions.csv',
                               sort_output=True,
                               batch_size=256,
                               verbose=0):
        '''
        Predict the PET-hydrolase activity of sequences in a multiple sequence alignment.

        Parameters
        ----------
        msa_file : str 
            Path of file containing sequences aligned to VAE MSA.
        out_file: str (default is 'predictions.csv')
            Path of csv file to write predictions. If out_file is None, predictions are 
            are not written. to csv file. 
        sort_output : boolean (default is True)
            If True, sequences in output are sorted in descending order of predicted 
            activity. If False, sequences are in the same order as in msa_file.
        batch_size : int (default is 256)
            Number of samples in each batch for TensorFlow/Keras model
        verbose : int (default is 0)
            Verbosity of Tensorflow/Keras model (1 > 2 > 0)

        Returns
        -------
        df : A dataframe of predictions

        '''
        
        # Sequences to be predicted
        heads_msa, seqs_msa = utils.read_fasta(msa_file)
        accs_msa = utils.get_accession(msa_file)
        assert len(set(len(seq) for seq in seqs_msa)) == 1, \
            f'Sequences in {msa_file} are not aligned to the same length'

        # Integer encoding of sequences to be predicted
        X_seq = [utils.categorical_encode_sequence(seq) for seq in seqs_msa]
        X_seq = np.array(X_seq, dtype=np.int32)
        
        # Predict activity with model
        ypred = self.predictActivityFromEncodings(X_seq, batch_size=batch_size, 
                                                  verbose=verbose)
        
        # Write predictions to csv

        df = pd.DataFrame(ypred, columns=['prediction'], index=accs_msa)
        df['description'] = heads_msa
        if sort_output:
            df = df.sort_values(by='prediction', ascending=False)
        if out_file is not None:            
            df.to_csv(out_file)
            
        return df
            

    
    
    
    def trainVAE(self, 
                 seq_file,
                 outdir='./vae_outdir', 
                 mafft_exe='/usr/local/bin/mafft'):
        '''Train a VAE to generate variants for a sequence.
        
        Parameters
        ----------
        msa_file : str 
            Path of file containing sequences aligned to VAE MSA.
        ref_seq : int (default is 0), str (sequence path or sequence name)
            The reference sequence to which the predicted activity of sequences in msa_file
            is compared.
           
            The reference sequence can be specified as 
            1. An integer denoting the index of the reference sequence in the msa_file. 
            The default value of 0 means that the first sequence in msa_file will be used
            as reference.
            2. A path to a separate file (fasta) containing a single sequence aligned to 
            the VAE MSA.
            3. The name of a canonical high-performing PETase sequence. This may be one of
            'IsPETase', 'HotPETase', 'LCC', or 'LCC_ICCG'.
        out_file: str (default is 'predictions.csv')
            Path of csv file to write predictions. Predictions are not written if out_file
            is None.
        sort_output : boolean (default is True)
            If True, sequences in output are sorted in descending order of predicted 
            activity. If False, sequences are in the same order as in msa_file.

        Returns
        -------
        df : A dataframe of predictions'''
        
        # Read sequence data
        heads_ref, seqs_ref = utils.read_fasta(seq_file)
        heads_raw, seqs_raw = utils.read_fasta(self.vae_raw_file)
        heads_all, seqs_all = heads_ref + heads_raw, seqs_ref + seqs_raw
        utils.write_fasta(heads_all, seqs_all, f'{outdir}/all_raw.fasta')
        
        # Align sequences with mafft
        command = f'{mafft_exe} {outdir}/all_raw.fasta {outdir}/all_msa.fasta'
        _ = subprocess.check_output(command, shell=True)
        
        # Exclude positions that are gaps in reference sequence
        dfseq = utils.fasta_to_df(f'{outdir}/all_msa.fasta')
        isnotgap = [dfseq.iloc[0,i] != '-' for i in range(dfseq.shape[1])]
        dfseq = dfseq.iloc[:,isnotgap]
        utils.df_to_fasta(dfseq, f'{outdir}/all_msa_ref.fasta')

        # Align sequences with reference sequence
        #dfseq = utils.fasta_to_df()
        
        # Remove positions that are gaps in reference sequence
        
        # Split sequences to training/testing datasets
        
        # Build VAE model 
        
        # Train vae model
        
        # Save vae model
        
        
        
        return
    
    
    
    
    def generateVariants(ref_seq_file, ref_vae_path):
        pass
    
    
    
    
    
        
    

