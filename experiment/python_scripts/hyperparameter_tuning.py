"""
Optimize hyperparameters for PETase VAE and top (rank) model
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

from sklearn import metrics
from sklearn.model_selection import KFold








#==================================#
# Prepare train/test data for VAE
#==================================#

datapath = './experiment/data/preprocessing'

# Retrieve sequence data
heads_train, seqs_train = utils.read_fasta(f'{datapath}/train_msa_gapless.fasta') #6,463
heads_test, seqs_test = utils.read_fasta(f'{datapath}/test_msa_gapless.fasta') #1,618


# One-hot encode sequence data
Xtrain = np.array([utils.one_hot_encode_sequence(seq) for seq in seqs_train],
                  dtype=np.int32)
Xtest = np.array([utils.one_hot_encode_sequence(seq) for seq in seqs_test], 
                 dtype=np.int32)


# Select 21 amino acids characters (20 canonical and gap, exclude non-canonical char [X])
Xtrain = Xtrain[:,:,:-1]  # Shape = (6463, 437, 21)
Xtest = Xtest[:,:,:-1]    # Shape = (1618, 437, 21)


# Sample (hamming) weights
train_hamming = pd.read_csv(f'{datapath}/train_hamming_weights.csv', 
                            index_col=0).values.reshape(-1)
test_hamming = pd.read_csv(f'{datapath}/test_hamming_weights.csv', 
                           index_col=0).values.reshape(-1)
train_hamming = train_hamming / np.mean(train_hamming)
test_hamming = test_hamming / np.mean(test_hamming)


# Check data size
assert len(Xtrain) == len(train_hamming)
assert len(Xtest) == len(test_hamming)






#===========================================================#
# Prepare labeled data for supervised learning (top model
#===========================================================#

# Sequence data (aligned to VAE MSA with positions haveing >95% gaps dropped)
heads, seqs = utils.read_fasta(f'{datapath}/label_msa_gapless.fasta') 
Xlabel = np.array([utils.one_hot_encode_sequence(seq) for seq in seqs],
                  dtype=np.int32) # Shape is (batch_size, num_res, 21)
Xlabel = Xlabel[:,:,:-1]  # Exclude 'X' character at -1 index


# Activity (labeled) datasets
dflabel = pd.read_excel('experiment/data/labels/datasets.xlsx', index_col=0)
dflabel.index = [item.replace(',','').replace(' ','_') for item in dflabel.index]


# Empty dictionary/arrays for storing cross-validation (CV) data
rawdata = {}  # (X, y) data for all 379 sequences from 23 studies
pairdata = {} # Pairwise data for rank prediction, all n(n-1)/2 pairs 
X1s = np.zeros((0, 437, 21))   # 1st sequence in pair
X2s = np.zeros((0, 437, 21))   # 2nd sequence in pair
weights = np.zeros(0,)         # Sample weights for each pair
pair_names = np.array([], dtype=object)  # Unique names for each pair
yints = np.zeros(0,)                     # Indicator function values, i.e I(a(X2) > a(X1))
dataset_names = np.array([], dtype=object) # Names of study (23) to which each pair belongs



# Generate pairwise ML data (X1, X2) from each dataset/study
for dataset in dflabel.index:
   
    # Activity data
    activity_path = f'experiment/data/labels/activity_data/{dataset}.csv'
    dfact = pd.read_csv(activity_path, index_col=0)['Activity'].dropna()
    y  = dfact.values
    
    # Sequence data
    locs = [heads.index(item) for item in dfact.index] # Location of variants in list of all data
    X = Xlabel[locs]  
    
    # Unique names of variants with mutations in alphabetical order
    names = np.array(dfact.index)
    names = np.array(['/'.join(sorted(item.split('/'))) for item in names]).astype(object)
    
    # Shuffle data to ensure balance between True/False class
    assert len(X) == len(y)
    locs = np.random.choice(range(len(y)), len(y), replace=False)
    X, y = X[locs], y[locs]
    
    # Rename identical sequences with different names to ensure consistency throughout
    names = names[locs]
    names[names=='IsPETase'] = 'IsPETase_WT'
    names[names=='PETcan401'] = 'PET2_WT'
    names[names=='LCC'] = 'LCC_WT'
    names[names=='TfCut2'] = 'TfCut2_WT'
    names[names=='TfCut'] = 'TfCut_WT'
    
    
    # Store raw sequence/activity data
    rawdata[dataset] = (X, y)  # X: seq one-hot, y: activity values
    
    
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

    


    
    



#==================================================#
# Optimize hyperparameters with greedy search
#==================================================#


# Fixed hyperparameters
amino_dim = 21
seq_len = Xtrain.shape[1]  # 437 positions
clipnorm = 10.0
mean = 0.0
stddev = 1.0




# Hyperperameter space (optimize with greedy search in this order)
learning_rates = [1e-2, 1e-3, 1e-4]                             # best is 1e-3
batch_sizes = [32, 64, 128, 256, 512]                           # best is 256
activations = ['elu', 'relu', 'leaky_relu', 'selu']             # best is 'elu'
use_batch_norms = [True, False]                                 # best is False
dense_units_list = [[256,128], [512,256], [1024,512], [2048,1024],  # best is [512,256]
                    [512,256,128], [1024,512,256], [2048,1024,512]]  
dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]                         # best is 0.2
regfactors = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]       # best is 1e-6
latent_dims = [8, 16, 24, 32, 48, 64, 96, 128]                  # best is 96
betas = [1, 2, 5, 10, 20, 50, 100]                              # best is 2
top_learning_rates = [1e-2, 1e-3, 1e-4]                         # best is 1e-3
top_batch_sizes = [32, 64, 128, 256, 512]                       # best is 256
top_regfactors = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]   # best is 1e-6
top_dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5]                     # best is 0




# Optimize hyperparameters with greedy search to maximize MCC on pairwise classification
allperf = pd.DataFrame()  # Dataframe to store model performance results
path = './experiment/data/hyperparameter_tuning/performance/'
csvpath = path + 'hyperparameter_tuning.csv' # Write results to CSV file

#for i_, learning_rate in enumerate(learning_rates):
#for i_, batch_size in enumerate(batch_sizes):
#for i_, activation in enumerate(activations):
#for i_, use_batch_norm in enumerate(use_batch_norms):
#for i_, dense_units in enumerate(dense_units_list):    
#for i_, dropout in enumerate(dropouts):        
#for i_, regfactor in enumerate(regfactors):        
#for i_, latent_dim in enumerate(latent_dims):            
#for i_, top_learning_rate in enumerate(top_learning_rates):            
#for i_, top_batch_size in enumerate(top_batch_sizes):
#for i_, top_regfactor in enumerate(top_regfactors):  
for i_, top_dropout in enumerate(dropouts):        
    
    if top_dropout == 0:
        # Skip if hyperparameter has been evaluated
        # Change as necessary
        continue
    
    KEY = 'top_dropout' # Change as necessary
    perfstore = {}
    
    # Selected hyperparameter (Change as necessary)
    learning_rate = 1e-3
    batch_size = 256
    activation = 'elu'
    use_batch_norm = False
    dense_units = [512,256]
    dropout = 0.2    
    regfactor = 1e-6
    latent_dim = 96
    beta = 2.0
    top_learning_rate = 1e-2
    top_batch_size = 256
    top_regfactor = 1e-6
    top_dropout = top_dropout
    
    
    # Reproducibility
    seed = 0
    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
    # Build VAE model
    vae = models.VAE(seq_len=seq_len,
                     amino_dim=amino_dim, 
                     dense_units=dense_units,
                     latent_dim=latent_dim,
                     activation=activation,
                     use_batch_norm=use_batch_norm,
                     dropout=dropout, 
                     regfactor=regfactor,
                     learning_rate=learning_rate,
                     mean=mean,
                     stddev=stddev,
                     beta=beta,
                     clipnorm=clipnorm)
    

    # View model architecture
    vae.VaeModel.summary()
    vae.EncoderModel.summary()
    vae.DecoderModel.summary()
    
    
    # Prepare directories for saving VAE model checkpoints
    storepath = './experiment/data/hyperparameter_tuning'
    checkpoint_path = f'{storepath}/checkpoints/vae_tuning'
    if not os.path.exists(storepath):
        os.makedirs(storepath)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    vae_checkpoint_path = f'{checkpoint_path}/vae_checkpoint'
    
    
    # Training callbacks for VAE model
    vae_checkpoint = ModelCheckpoint(filepath=vae_checkpoint_path,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     monitor='val_loss', 
                                     mode='min',
                                     verbose=1)
    reducelr = ReduceLROnPlateau(factor=0.5, 
                                 patience=10, 
                                 verbose=1, 
                                 min_delta=1e-4, 
                                 min_lr=1e-7)
    earlystopping = EarlyStopping(monitor='val_loss', 
                                  min_delta=1e-4,
                                  patience=20, 
                                  verbose=1)
    
    # Train VAE model 
    callbacks = [vae_checkpoint, earlystopping, reducelr]
    history = vae.VaeModel.fit(x=Xtrain, 
                               y=Xtrain, 
                               sample_weight=train_hamming, 
                               validation_data=(Xtest, Xtest, test_hamming), 
                               batch_size=batch_size, 
                               verbose=2,
                               epochs=1000, 
                               callbacks=callbacks)
    
    # Load optimal VAE model weights
    vae.VaeModel.load_weights(vae_checkpoint_path)
        
    
    
    # Evaluate VAE on testing set (weighted by sample weights)
    Xtestpred = vae.VaeModel.predict(Xtest, batch_size=batch_size)
    amino_accuracy = models.aminoAccuracy(Xtest, Xtestpred).numpy()
    entropy_loss = models.entropyLoss(Xtest, Xtestpred).numpy()
    entropy_loss = np.sum(entropy_loss * test_hamming) / np.sum(test_hamming)
    elbo_loss = vae.elboLoss(Xtest, Xtestpred).numpy()
    elbo_loss = np.sum(elbo_loss * test_hamming) / np.sum(test_hamming)
    kl_loss = vae.kldLoss(Xtest, Xtestpred).numpy()
    kl_loss = np.sum(kl_loss * test_hamming) / np.sum(test_hamming)
    
    
    # Store performance
    perfstore['elbo'] = elbo_loss
    perfstore['entropy'] = entropy_loss
    perfstore['kld'] = kl_loss
    perfstore['amino_accuracy'] = amino_accuracy

    
    # Derive latent representation from VAE for sequence data
    Z1 = vae.EncoderModel.predict(X1s)[0]
    Z2 = vae.EncoderModel.predict(X2s)[0]
    
    
    # Empty list/dictionary to store supervised top model performance
    mcc_store, accuracy_store = [], []
    rhos = {}
    
    
    # Leave-group-out cross-validation (LOGOCV) to train/test top model on VAE latent space
    for dataset in set(dataset_names):
        
        print()
        print(dataset) # Dataset is the separate held-out test set
        
        
        ''' LOGOCV
        To save time, use the largest 5 datasets for hyperparameter tuning, rather than all
        23 datasets. These datasets are
        (1) Cui et al, 2021: 65 seqsuences
        (2) Erickson et al, 2022: 43 sequences
        (3) Zeng et al, 2022: 34 sequences
        (4) Nakamura et al, 2021: 24 sequences
        (5) Chen et al, 2021: 22 sequences
        In each LOGOCV repetition, one of these 5 datasets as held-out test data. Randomly
        split the remaining data to 3 folds. Train 3 models with the train/test 
        splits obtained from the folds, and evaluate the performance of an ensemble of the
        3 models on the held-out data. 
        '''
        
        # Skip if dataset/study has less than 200 pairs (< 21 sequences)
        # All but 5 datasets are skipped
        dataset_locs = np.argwhere(dataset_names==dataset).reshape(-1)
        if len(dataset_locs) < 200:
            continue
    
    
        # Train top model with cross-validation
        kf = KFold(n_splits=3, random_state=0, shuffle=True)  
        rank_model_list = []
        
        for icv,(trainlocs, testlocs) in enumerate(kf.split(Z1)):
            
            # Exclude selected dataset as seperate/held-out data
            trainlocs = list(set(trainlocs) - set(dataset_locs))
            testlocs = list(set(testlocs) - set(dataset_locs))
            
            
            # Remove sequence pairs from training set that are in testing set by names
            test_pair_names = pair_names[testlocs]
            trainlocs = [i for i in trainlocs if pair_names[i] not in test_pair_names]
            
            
            # Train/test data
            xtrain = [Z1[trainlocs], Z2[trainlocs]]
            ytrain = yints[trainlocs]          
            trainweights = weights[trainlocs]
            xtest = [Z1[testlocs], Z2[testlocs]]
            ytest = yints[testlocs]          
            testweights = weights[testlocs]
    
            
            # Reproducibility
            K.clear_session()
            tf.random.set_seed(0)
            np.random.seed(0)
            random.seed(0)
            
            # Build top (rank) model
            # Score model is a linear model on VAE latent space that returns a(X)
            score_model = models.buildLinearTopModel(input_shape=latent_dim,
                                                    dropout=top_dropout, 
                                                    regfactor=top_regfactor)
            
            # Rank model is derived from score model and returns sigmoid(a(X2) - a(X1))
            rank_model = models.buildRankModel(score_model=score_model, 
                                               input_shape=latent_dim,
                                               learning_rate=top_learning_rate)
            
            # Training callbacks for top (rank) model
            top_checkpoint_path = f'{storepath}/checkpoints/top_tuning/top_checkpoint'
            top_checkpoint = ModelCheckpoint(filepath=top_checkpoint_path,
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
            callbacks = [top_checkpoint, reducelr, earlystopping]
            
            
            # Train top (rank) model
            history = rank_model.fit(x=xtrain,
                                     y=ytrain,
                                     sample_weight=trainweights,
                                     validation_data=(xtest, ytest, testweights),
                                     verbose=2,
                                     epochs=1000,
                                     batch_size=batch_size,
                                     callbacks=callbacks)     
                
            # Evaluate performance of model on validation set
            rank_model.load_weights(top_checkpoint_path)
            rank_model_list.append(rank_model)
            ypred = rank_model.predict(xtest).reshape(-1)
            mcc = metrics.matthews_corrcoef(ytest, (ypred>0.5).astype(int))
            mcc_store.append(mcc)
            accuracy = metrics.accuracy_score(ytest, (ypred>0.5).astype(int))
            accuracy_store.append(accuracy)
        
        
        # Apply ensemble of trained top models to predict activity of held-out testing set
        X, y = rawdata[dataset]
        Z = vae.EncoderModel.predict(X)[0]
        ypred = np.array([rank_model.layers[2].predict(Z).reshape(-1) \
                          for rank_model in rank_model_list])

            
        # Average (ensemble) prediction of all models from each cross-validation fold
        ypred_mean = np.mean(ypred, axis=0)  
        
        
        # Spearman correlation between predicted and experimental activity
        rho = spearmanr(ypred_mean, y)[0]
        print()
        print(f'{dataset}: rho={rho}')
        rhos[dataset] = rho  
        
    
    # Average performance over all repetitions of 3-fold cross validation
    rhos = pd.Series(rhos)
    print(rhos)
    perfstore['rho_avg'] = rhos.mean()
    perfstore['rho_std'] = rhos.std()
    perfstore['rho_min'] = rhos.min()
    perfstore['mcc'] = np.mean(mcc_store)
    perfstore['mcc_std'] = np.std(mcc_store)
    perfstore['accuracy'] = np.mean(accuracy_store)
    perfstore['accuracy_std'] = np.std(accuracy_store)
    
    
    # Save performance of hyperparameter combination for selection
    perfstore['learning_rate'] = learning_rate
    perfstore['batch_size'] = batch_size
    perfstore['activation'] = activation
    perfstore['use_batch_norm'] = use_batch_norm 
    perfstore['dense_units'] = dense_units
    perfstore['dropout'] = dropout
    perfstore['regfactor'] = regfactor
    perfstore['latent_dim'] = latent_dim
    perfstore['beta'] = beta
    perfstore['top_regfactor'] = top_regfactor
    perfstore['top_learning_rate'] = top_learning_rate
    perfstore['top_batch_size'] = top_batch_size
    perfstore['top_dropout'] = top_dropout
    perfstore['KEY'] = KEY
    print(perfstore)
    perfstore = pd.DataFrame({0: perfstore}).transpose()
    allperf = pd.concat([allperf, perfstore], axis=0, ignore_index=True)
    allperf.to_csv(csvpath)


print('\DONE!\n')
    
