"""
PETML: predict PET hydrolase activity with machine learning
"""




import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from Bio.Align import substitution_matrices

import subprocess
import os
import joblib
import sys
import  builtins
import argparse

sys.path.insert(1, './petml')
sys.path.insert(1, './')
from petml import helper

import warnings
warnings.filterwarnings('ignore')






def parse_arguments():
    '''Parse command-line  arguments'''
    
    parser = argparse.ArgumentParser(description="Predict PET hydrolase activity with ML")
    
    parser.add_argument('--seqfile', type=str, 
                        help='Path to fasta file of sequences')
    parser.add_argument('--outdir', type=str, default='./',
                        help='Directory where output files will be written to')
    parser.add_argument('--verbose', type=int, default=1, 
                        help="Whether to print out progress: verbose (1) or silent (0)")
    parser.add_argument('--delete_temp_files', type=int, default=0,
                        help="Whether to delete temporary files: Yes (1) or No (0)")
    args = parser.parse_args()

    return args






def calculate_blosum_score(seq1, seq2, matrix, gap_penalty=-4):
    '''Return the blosum similarity score between aligned seq1 and seq2'''
    
    assert len(seq1) == len(seq2)
    score = 0
    for a, b in zip(seq1, seq2):
        if a == '-' or b == '-':
            score += gap_penalty
        else:
            score += matrix.get((a, b), matrix.get((b, a), 0))

    return score  






def standardize(array, mean_std):
    '''Standardize values to a mean of 0 and standard deviation of 1'''

    mean, std = mean_std
    array = np.asarray(array, dtype=float)
    
    return (array - mean) / (std + 1e-8)






def print(*args, **kwargs):
    '''Custom print function to always flush output when verbose'''

    builtins.print(*args, **kwargs, flush=True)
    





def main():
    '''Run supervised and unsupervised predictions'''

    # Parse command line arguments
    args = parse_arguments()
    
    
    # Manage files and directories
    assert os.path.exists(args.seqfile), f"{args.seqfile} not found"
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if args.verbose:
        print(f"Prediction of PET hydrolase activity for sequences in {args.seqfile}")
        
    
    # Get model files
    this_dir, this_filename = os.path.split(__file__)
    b04_hmm = os.path.join(this_dir, 'data', 'unsupervised', 'jackhmmer_hmm_b04.txt')
    b04_fasta = os.path.join(this_dir, 'data', 'unsupervised', 'jackhmmer_seqs_b04_small.fasta')
    petase_hmm = os.path.join(this_dir, 'data', 'unsupervised', 'petase_hmm.txt')
    activesite_hmm = os.path.join(this_dir, 'data', 'unsupervised', 'active_site_hmm.txt')
    activesite_positions = os.path.join(this_dir, 'data', 'unsupervised', 'active_site_positions.csv')
    consensus_fasta = os.path.join(this_dir, 'data', 'unsupervised', 'jackhmmer_consensus_b04.fasta')
    aln_positions_469 = os.path.join(this_dir, 'data', 'supervised', 'aln_positions_469.csv')
    mean_std = os.path.join(this_dir, 'data', 'supervised', 'onehot_mean_std.csv')
    score_mean_std = os.path.join(this_dir, 'data', 'supervised', 'score_mean_std.csv')
    model = os.path.join(this_dir, 'data', 'supervised', 'onehot_log_reg.pkl')
    
    
    # Get executables
    hmmsearch_exe = subprocess.check_output("which hmmsearch", shell=True, text=True).strip()
    mafft_exe = subprocess.check_output("which mafft", shell=True, text=True).strip()
    
    
    # Get sequence data
    seq_accessions = helper.get_accession(args.seqfile)
    seq_accessions = [acc.split('/')[0] for acc in seq_accessions]
    
    
    # Search sequences with evolutionary HMM (jackhmmer b04) to exclude flanking domains
    if args.verbose:
        print("Scoring sequences with homologs HMM.")
    helper.search_with_HMM(seq_file=args.seqfile, 
                           hmm_file=b04_hmm, 
                           threshold=0, 
                           outdir=f"{args.outdir}/homologs_hmmout", 
                           hmmsearch_exe=hmmsearch_exe)
    
    
    # Rename searched sequence file
    copyfrom = f'{args.outdir}/homologs_hmmout/aln_no_gaps.fasta'
    hmmsearch_fasta = f'{args.outdir}/sequences_hmmsearch.fasta'
    subprocess.check_output(f'cp {copyfrom} {hmmsearch_fasta}', shell=True) 
    
    
    # METHOD 1:  Evolutionary HMM
    tabout = f"{args.outdir}/homologs_hmmout/tab_output.txt"
    df = helper.parse_hmm_tabout(tabout)
    y_homologs_hmm = df.loc[:,5]
    y_homologs_hmm.index = [item.split()[0].split('/')[0] for item in df.loc[:,0]]
    y_homologs_hmm = y_homologs_hmm[~y_homologs_hmm.index.duplicated(keep='first')]
    y_homologs_hmm = y_homologs_hmm.reindex(seq_accessions).astype(float)
    
    
    # Search sequences with PETase HMM
    if args.verbose:
        print("Scoring sequences with PET HMM.")
    helper.search_with_HMM(seq_file=args.seqfile, 
                           hmm_file=petase_hmm, 
                           threshold=0, 
                           outdir=f"{args.outdir}/petase_hmmout", 
                           hmmsearch_exe=hmmsearch_exe)
    
    
    # METHOD 2: PETase HMM
    tabout = f"{args.outdir}/petase_hmmout/tab_output.txt"
    df = helper.parse_hmm_tabout(tabout)
    y_petase_hmm = df.loc[:,5]
    y_petase_hmm.index = df.loc[:,0]
    y_petase_hmm.index = [item.split()[0].split('/')[0] for item in df.loc[:,0]]
    y_petase_hmm = y_petase_hmm[~y_petase_hmm.index.duplicated(keep='first')]    
    y_petase_hmm = y_petase_hmm.reindex(seq_accessions).astype(float)
    
    
    # Align sequences by adding to evolutionary alignment (jackhmmer b04) 
    if args.verbose:
        print("Adding sequences to homologs alignment. This may take a while.")    
    alnfile = f'{args.outdir}/sequences_aligned.fasta'
    helper.align_with_MSA(seq_file=hmmsearch_fasta,
                          msa_file=b04_fasta, 
                          out_file=alnfile, 
                          mafft_exe=mafft_exe, 
                          verbose=0)
    
    
    # Trim MSA to select IsPETase positions (for active site HMM)
    df = helper.fasta_to_df(b04_fasta)
    isnotgap = [item != '-' for item in df.iloc[0,:]]
    df = helper.fasta_to_df(alnfile)
    df = df.iloc[:,isnotgap]
    assert df.shape[1] == 290 
    df.columns = np.arange(1,291)  # Name columns 1 to 290 as in IsPETase
    positions = np.asarray(pd.read_csv(activesite_positions, index_col=0).index)
    assert len(positions) == 83
    df = df.loc[:,positions]
    active_site_fasta = f"{args.outdir}/sequences_active_site.fasta"
    helper.df_to_fasta(df, active_site_fasta)
    helper.remove_gaps(active_site_fasta, active_site_fasta)
    
    
    
    # Search selected positions with Active site HMM
    if args.verbose:
        print("Scoring sequences with active site HMM.")        
    helper.search_with_HMM(seq_file=active_site_fasta, 
                           hmm_file=activesite_hmm, 
                           threshold=0, 
                           outdir=f"{args.outdir}/active_site_hmmout", 
                           hmmsearch_exe=hmmsearch_exe)
    
    
    # METHOD 3: Active site HMM
    tabout = f"{args.outdir}/active_site_hmmout/tab_output.txt"
    df = helper.parse_hmm_tabout(tabout)
    y_active_site_hmm = df.loc[:,5]
    y_active_site_hmm.index = [item.split()[0].split('/')[0] for item in df.loc[:,0]]
    y_active_site_hmm = y_active_site_hmm[~y_active_site_hmm.index.duplicated(keep='first')]        
    y_active_site_hmm = y_active_site_hmm.reindex(seq_accessions).astype(float)
    
    
    # METHOD 4: Blosum similarity with consensus sequence
    if args.verbose:
        print("Scoring sequences with Blosum similarity to consensus.")        
    _, [consensus_seq] = helper.read_fasta(consensus_fasta)
    headers, sequences = helper.read_fasta(alnfile)
    accessions = helper.get_accession(alnfile)
    blosum62 = substitution_matrices.load('BLOSUM62')
    y_blosum = [calculate_blosum_score(seq, consensus_seq, matrix=blosum62, 
                                       gap_penalty=-4) for seq in sequences]
    y_blosum = pd.Series(y_blosum, index=accessions)
    y_blosum.index = [item.split()[0].split('/')[0] for item in y_blosum.index]
    y_blosum = y_blosum[~y_blosum.index.duplicated(keep='first')]            
    y_blosum = y_blosum.reindex(seq_accessions)
    
    
    # Trim MSA to select 469 positions used in training supervised model
    if args.verbose:
        print("Scoring sequences with pairwise supervised ranking model (logistic regression.")
    df = helper.fasta_to_df(alnfile)
    assert df.shape[1] == 1813
    locs = pd.read_csv(aln_positions_469, index_col=None, header=None).values.flatten()
    assert len(locs) == 469
    df = df.iloc[:,locs]
    trimseqfile = f"{args.outdir}/sequences_trimmed.fasta"
    helper.df_to_fasta(df, trimseqfile)
    
    
    # One-hot encode trimmed aligned sequences
    headers, sequences = helper.read_fasta(trimseqfile)
    sequences = [helper.canonize(seq) for seq in sequences]
    accessions = helper.get_accession(trimseqfile)
    df = pd.DataFrame(sequences)
    ohe = helper.OneHotEncoder()
    df = pd.DataFrame(ohe.encode_from_df(df, col=0), index=accessions)
    
    
    # Standardize one-hot representation
    mean_std_values = pd.read_csv(mean_std, index_col=0)
    X = (df - mean_std_values['means'].values) / (mean_std_values['stds'].values + 1e-8)
    
    
    # Use IsPETase as reference sequence
    ispetase = helper.fasta_to_df(b04_fasta).iloc[[0],:]
    ispetase = ispetase.iloc[:,locs]
    ispetase = ''.join(ispetase.values.flatten())
    df = pd.DataFrame([ispetase])
    df = pd.DataFrame(ohe.encode_from_df(df, col=0), index=['IsPETase'])
    X2 = (df - mean_std_values['means'].values) / (mean_std_values['stds'].values + 1e-8)
    
    
    # Predict pairwise activity ranking with the logistic regression model
    model = joblib.load(model)
    Z = X.values - X2.values
    y_logreg = pd.Series(model.predict_proba(Z)[:,-1], index=accessions)
    y_logreg.index = [item.split()[0].split('/')[0] for item in y_logreg.index]
    y_logreg = y_logreg[~y_logreg.index.duplicated(keep='first')]                
    y_logreg = y_logreg.reindex(seq_accessions).astype(float)
    
    
    # Write results
    if args.verbose:
        print("Writing predicted scores.")
    results = pd.DataFrame(index=seq_accessions)
    results['Supervised'] = y_logreg.values
    results['Active site HMM'] = y_active_site_hmm.values
    results['PET HMM'] = y_petase_hmm.values
    results['Homologs HMM'] = y_homologs_hmm.values
    results['Blosum'] = y_blosum.values
    results = results.fillna(0)
    results.to_csv(f'{args.outdir}/raw_scores.csv')
    if args.verbose:
        print(f"Raw scores are in {args.outdir}/raw_scores.csv")
    
    
    
    # Combine (average) supervised and unsupervised standardized scores
    scores = pd.DataFrame(index=seq_accessions)
    score_mean_std = pd.read_csv(score_mean_std, index_col=0)
    scores['Supervised'] = standardize(y_logreg.values, score_mean_std.loc['Supervised'])
    scores['PET HMM'] = standardize(y_petase_hmm.values, score_mean_std.loc['PET HMM'])
    scores['Active site HMM'] = standardize(y_active_site_hmm.values, score_mean_std.loc['Active site HMM'])
    scores['Homologs HMM'] = standardize(y_homologs_hmm.values, score_mean_std.loc['Homologs HMM'])
    scores['Blosum'] = standardize(y_blosum.values, score_mean_std.loc['Blosum'])
    unsupervised =  scores.loc[:,['PET HMM', 'Active site HMM', 'Homologs HMM',
                                  'Blosum']].mean(axis=1)
    scores['unsupervised'] = unsupervised
    scores['petml_scores'] = 0.5 * scores['Supervised'] + 0.5 * unsupervised
    scores.to_csv(f'{args.outdir}/final_scores.csv')
    if args.verbose:
        print(f"Final average standardized scores are in {args.outdir}/final_scores.csv")
        print("Higher scores imply higher predicted activity.")
    
    
    # Delete temp. files
    if args.delete_temp_files:
        if args.verbose:
            print(f"Deleting temporary files from {args.outdir}")    
        _ = subprocess.check_output(f"rm -rf {args.outdir}/*_hmmout", shell=True)
        _ = subprocess.check_output(f"rm -rf {args.outdir}/sequences_*.fasta", shell=True)
    else:
        if os.path.exists(f"{args.outdir}/tmp"):
            _ = subprocess.check_output(f"rm -rf {args.outdir}/tmp/*", shell=True)
        else:
            os.makedirs(f"{args.outdir}/tmp")
        _ = subprocess.check_output(f"mv -f {args.outdir}/*_hmmout {args.outdir}/tmp/",
                                    shell=True)
        _ = subprocess.check_output(f"mv -f {args.outdir}/sequences_*.fasta {args.outdir}/tmp/",
                                    shell=True)


    if args.verbose:
        print("DONE!")    
    
    
    
    
    
    
if __name__ == '__main__':

    main()
    