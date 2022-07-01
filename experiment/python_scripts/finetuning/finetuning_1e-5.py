"""
End-to-end finetuning of deepPETase (encoder + top linear/rank model)
"""




#==============#
# Imports
#==============#
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import random
import itertools

import os
import sys
sys.path.insert(1, './')

from module import utils, models

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
tf.get_logger().setLevel('ERROR')

from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.model_selection import KFold




LEARNING_RATE = 1e-5




#========================================#
# VAE model for semisupervised learning
#========================================#

def buildDeepPETase(i=0, learning_rate=1e-5):
    '''Return DeepPETase model. The model input is a list of sequence one-hot encodings 
    for a pair of PETases [X1, X2]. The model output is the probability (sigmoid output) 
    that X2 has higher PET-hydrolase activity than X1.'''
    
    # Build VAE with optimal weights
    vae = models.VAE(seq_len=437,
                     amino_dim=21,
                     dense_units=[512,256],
                     latent_dim=96,
                     activation='elu',
                     use_batch_norm=False,
                     dropout=0.2,
                     regfactor=1e-6,
                     learning_rate=1e-3,
                     mean=0.0,
                     stddev=1.0,
                     beta=2.0,
                     clipnorm=10.0)
    vaepath = './experiment/data/training/models/petvae.h5'
    vae.VaeModel.load_weights(vaepath)
    
    # Build top linear model with optimal weights
    top_model = models.buildLinearTopModel(input_shape=96,
                                           dropout=0.0,
                                           regfactor=1e-6)
    toppath = f'./experiment/data/training/models/top_model_{i}.h5'
    top_model.load_weights(toppath)
    
    # Combine VAE encoder and top model into a single model
    combined_model = models.combineEncoderAndTopModel(input_shape=(437,21), 
                                                      encoder_model=vae.EncoderModel, 
                                                      top_model=top_model)
    
    # Build pairwise rank model from combined model
    deepPETase = models.buildRankModel(score_model=combined_model,
                                       input_shape=(437,21),
                                       learning_rate=learning_rate,
                                       name=f'deepPETase{i}')
    return deepPETase
    










#========================#
# Prepare labeled data 
#========================#

# Sequence data
datapath = './experiment/data/preprocessing'
heads, seqs = utils.read_fasta(f'{datapath}/label_msa_gapless.fasta') # Sequences aligned to VAE MSA
Xlabel = np.array([utils.one_hot_encode_sequence(seq) for seq in seqs],
                  dtype=np.int32) # Shape is (batch_size, L, 21)
Xlabel = Xlabel[:,:,:-1]


# Activity (labeled) datasets
#dflabel = pd.read_csv('experiment/data/labels/datasets.csv', index_col=0)
dflabel = pd.read_excel('experiment/data/labels/datasets.xlsx', index_col=0)
dflabel.index = [item.replace(',','').replace(' ','_') for item in dflabel.index]


# Empty dictionary/lists for storing 5fold CV data
rawdata = {}  # (X, y, ylog) data for all 379 sequences from 23 studies
pairdata = {} # Pairwise data for rank prediction, all n(n-1)/2 pairs 
X1s = np.zeros((0, 437, 21))   # 1st sequence in pair
X2s = np.zeros((0, 437, 21))   # 2nd sequence in pair
weights = np.zeros(0,)         # Sample weights for each pair
pair_names = np.array([], dtype=object)  # Unique names for each pair
yints = np.zeros(0,)                     # Binary relative activity (a(X2) > a(X1))
dataset_names = np.array([], dtype=object) # Names of study (23) to which pair belongs



# Generate ML data for each dataset/study
for dataset in dflabel.index:
   
    # Activity data
    activity_path = f'experiment/data/labels/activity_data/{dataset}.csv'
    dfact = pd.read_csv(activity_path, index_col=0)
    dfact = dfact.loc[:,['Activity', 'logActivity']].dropna()
    y  = dfact['Activity'].values       # Use raw activity values for Spearman R
    ylog = dfact['logActivity'].values  # Use log(activity) values for NDCG
    
    # Sequence data
    locs = [heads.index(item) for item in dfact.index] # Location of variants in list of all data
    X = Xlabel[locs]  
    
    # Unique names of variants with mutations in alphabetical order
    names = np.array(dfact.index)
    names = np.array(['/'.join(sorted(item.split('/'))) for item in names]).astype(object)
    
    # Shuffle data to ensure balance between True/False class
    assert len(X) == len(y) == len(ylog)
    locs = np.random.choice(range(len(y)), len(y), replace=False)
    X, y, ylog, names = X[locs], y[locs], ylog[locs], names[locs]
    
    # Rename identical sequences with different names to ensure consistency throughout
    names[names=='IsPETase'] = 'IsPETase_WT'
    names[names=='PETcan401'] = 'PET2_WT'
    names[names=='LCC'] = 'LCC_WT'
    names[names=='TfCut2'] = 'TfCut2_WT'
    names[names=='TfCut'] = 'TfCut_WT'
    
    
    # Store raw sequence/activity data
    rawdata[dataset] = (X, y, ylog, names)  # X: seq one-hot, y: activity values
    
    
    # Generate all possible pairs from sequence data, i.e. n(n-1)/2 pairs
    pairs = np.array(list(itertools.combinations(np.arange(len(y)), 2)))
    
    # Sequence pairs
    X1 = X[pairs[:,0]]
    X2 = X[pairs[:,1]]
    
    # Unique names for each pair
    pair_name = [{str(names[pairs[i,0]]), str(names[pairs[i,1]])} \
                  for i in range(len(pairs))]  
        
    # Relative activity (binary)
    ydiff = y[pairs[:,1]] - y[pairs[:,0]]
    yint = (ydiff > 0).astype(np.int32)
    
    # Sample weights to pay more attention to larger activity difference
    weight = 1 + (np.abs(ydiff) / np.std(ydiff))
    
    
    # Store pairwise data 
    X1s = np.append(X1s, X1, axis=0)
    X2s = np.append(X2s, X2, axis=0)
    weights = np.append(weights, weight, axis=0)
    yints = np.append(yints, yint, axis=0)
    pair_names = np.append(pair_names, pair_name, axis=0)
    pairdata[dataset] = (X1, X2, yint, ydiff, pair_name)
    dataset_names = np.append(dataset_names, [dataset]*len(yint))
    

# Check data shape
assert len(X1s) == len(X2s) == len(weights) == len(yints) == len(pair_names) == \
        len(dataset_names)


    


    
    



#=====================================================================#
# Evaluate performance of fine-tuning with repeated cross-validation
#=====================================================================#


# Empty dictionaries to store model performance/results
mccs_val, accuracies_val = {}, {}  # 5-fold CV performance (classification/rank)
mccs_heldout, accuracies_heldout = {}, {}  # Held-out performance
rhos_heldout, ndcgs_heldout = {}, {}



# Repeated five-fold cross validation (repeat for each dataset)
'''
For each study/dataset, sequences from that study are excluded as a separate held-out
set. The remaining data is randomly split into 5 folds for cross validation. Five models
are trained with the train/test plits obtained from the folds, and the performance of
an ensemble of the 5 models is evaluated on the separate held-out set. Hence, the 
evaluation routine is a 23 x 5-fold cross-validation.
'''

for dataset in set(dataset_names):

    # Dataset is the separate held-out test set
    print(f'\nEvaluating performance on {dataset}\n') 
    dataset_locs = np.argwhere(dataset_names==dataset).reshape(-1)


    # Train combined model end-to-end with cross-validation
    kf = KFold(n_splits=3, random_state=0, shuffle=True)
    model_list = []  # List of cross-validation models for ensemble
    mccs, accuracies = [], []  # Cross validation performance
    
    
    for icv,(trainlocs, testlocs) in enumerate(kf.split(X1s)):
        
        # Exclude selected dataset as seperate test data
        trainlocs = list(set(trainlocs) - set(dataset_locs))
        testlocs = list(set(testlocs) - set(dataset_locs))
        
        
        # Remove sequence pairs in train set that are also in test set
        test_pair_names = pair_names[testlocs]
        trainlocs = [i for i in trainlocs if pair_names[i] not in test_pair_names]
        
        
        # Train/test data
        xtrain = [X1s[trainlocs], X2s[trainlocs]]
        ytrain = yints[trainlocs]          
        trainweights = weights[trainlocs]
        xtest = [X1s[testlocs], X2s[testlocs]]
        ytest = yints[testlocs]          
        testweights = weights[testlocs]

        
        # Build combined model
        deepPETase = buildDeepPETase(i=icv, learning_rate=LEARNING_RATE)
        
        
        # Training callbacks
        path = 'experiment/data/finetuning/checkpoints/{:.0e}'.format(LEARNING_RATE)
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = f'{path}/model_checkpoint'
        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     monitor='val_loss', 
                                     mode='min',
                                     verbose=0)
        reducelr = ReduceLROnPlateau(factor=0.5, 
                                     patience=10, 
                                     verbose=2, 
                                     min_delta=1e-4, 
                                     min_lr=1e-7)
        earlystopping = EarlyStopping(monitor='val_loss', 
                                      min_delta=1e-4,
                                      patience=20, 
                                      verbose=0)
        callbacks = [checkpoint, reducelr, earlystopping]
        
        
        # Finetune deepPETase model
        history = deepPETase.fit(x=xtrain,
                                 y=ytrain,
                                 sample_weight=trainweights,
                                 validation_data=(xtest, ytest, testweights),
                                 verbose=2,
                                 epochs=1000,
                                 batch_size=256,
                                 callbacks=callbacks)     
            
        # Evaluate performance of model on validation set
        deepPETase.load_weights(checkpoint_path)
        model_list.append(deepPETase)
        ypred = deepPETase.predict(xtest).reshape(-1)
        mcc = matthews_corrcoef(ytest, (ypred>0.5).astype(int))
        mccs.append(mcc)
        accuracy = accuracy_score(ytest, (ypred>0.5).astype(int))
        accuracies.append(accuracy)

    
    # Apply ensemble of trained models to predict activity (regression) 
    # of held-out testing set
    X, y, ylog, names = rawdata[dataset]
    ypred = np.array([model.layers[2].predict(X).reshape(-1) for model in model_list])
    
    
    # Average activity prediction (ensemble) of all models from each cross-validation fold
    ypred_mean = np.mean(ypred, axis=0)
    rho = spearmanr(ypred_mean, y)[0]
    rho_std = np.std([spearmanr(ypred[i,:], y)[0] for i in range(len(ypred))])
    ndcg = utils.calculate_NDCG(ypred_mean, ylog)
    ndcg_std = np.std([utils.calculate_NDCG(ypred[i,:], ylog) for i in range(len(ypred))])
    print()
    print(f'{dataset}: rho={rho}')
    
    
    # Store model predictions
    dfpred = pd.DataFrame([ypred_mean, y, ylog], index=['ypred', 'ytrue', 'ylog'], 
                          columns=names).transpose()
    path = 'experiment/data/finetuning/performance/{:.0e}/predictions'.format(LEARNING_RATE)
    if not os.path.exists(path):
        os.makedirs(path)
    csvstore = f'{path}/{dataset}.csv'
    dfpred.to_csv(csvstore)
    
    
    # Apply ensemble of trained models to predict pairwise rank (classification) 
    # of held-out testing set
    xpair = [X1s[dataset_locs], X2s[dataset_locs]]
    ypair = yints[dataset_locs]
    ypred_pair = np.array([model.predict(xpair).reshape(-1) for model in model_list])
    
    
    # Average rank prediction (ensemble) of all models from each cross-validation fold
    ypred_pair_mean = np.mean(ypred_pair, axis=0)
    mcc2 = matthews_corrcoef(ypair, (ypred_pair_mean > 0.5).astype(int))
    mcc2_std = np.std([matthews_corrcoef(ypair, (ypred_pair[i,:] > 0.5).astype(int)) \
                      for i in range(len(ypred_pair))])
    accuracy2 = accuracy_score(ypair, (ypred_pair_mean > 0.5).astype(int))
    accuracy2_std = np.std([accuracy_score(ypair, (ypred_pair[i,:] > 0.5).astype(int)) \
                           for i in range(len(ypred_pair))])
     
        
    # Store validation performance
    mccs_val[dataset] = mccs
    accuracies_val[dataset] = accuracies
    
    # Store testing performance
    mccs_heldout[dataset] = [mcc2, mcc2_std]
    accuracies_heldout[dataset] = [accuracy2, accuracy2_std]
    rhos_heldout[dataset] = [rho, rho_std]
    ndcgs_heldout[dataset] = [ndcg, ndcg_std]
    
    
    
# Save performance as file
csvpath = './experiment/data/finetuning/performance/{:.0e}'.format(LEARNING_RATE)
dfrho = pd.DataFrame(rhos_heldout)
dfmccs_val, dfmccs_heldout, dfaccs_val, dfaccs_heldout, dfrhos_heldout, dfndcgs_heldout = \
    [pd.DataFrame(item).transpose() for item in \
     [mccs_val, mccs_heldout, accuracies_val, accuracies_heldout, 
      rhos_heldout, ndcgs_heldout]]
dfmccs_val.to_csv(f'{csvpath}/mcc_validation.csv')
dfmccs_heldout.to_csv(f'{csvpath}/mcc_heldout.csv')
dfaccs_val.to_csv(f'{csvpath}/accuracy_validation.csv')
dfaccs_heldout.to_csv(f'{csvpath}/accuracy_heldout.csv')
dfrhos_heldout.to_csv(f'{csvpath}/rho_heldout.csv')
dfndcgs_heldout.to_csv(f'{csvpath}/ndcg_heldout.csv')

print('\n\nDONE\n\n')