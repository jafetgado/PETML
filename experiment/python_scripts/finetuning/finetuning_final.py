"""
End-to-end finetuning of deepPETase with optimal learning rate
"""




#==============#
# Imports
#==============#
import numpy as np
import pandas as pd
import itertools

import os
import sys
sys.path.insert(1, './')

from module import utils, models

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import KFold




LEARNING_RATE = 1e-7




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
    
    # Build top model with optimal weights
    top_model = models.buildLinearTopModel(input_shape=96,
                                           dropout=0.0,
                                           regfactor=1e-6)
    toppath = f'./experiment/data/training/models/top_model_{i}.h5'
    top_model.load_weights(toppath)
    
    # Combine VAE encoder and top model into a single model
    combined_model = models.combineEncoderAndTopModel(input_shape=(437,21), 
                                                      encoder_model=vae.EncoderModel, 
                                                      top_model=top_model)
    
    # Build full pairwise ranking model from combined model
    deepPETase = models.buildRankModel(score_model=combined_model,
                                       input_shape=(437,21),
                                       learning_rate=learning_rate,
                                       name=f'deepPETase{i}')
    return deepPETase
    










#========================#
# Prepare labeled data 
#========================#

# Sequence data (aligned to VAE MSA with positions haveing >95% gaps dropped)
datapath = './experiment/data/preprocessing'
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





 
#======================================================#
# Train deepPETase models 3-fold cross validation
#======================================================#

kf = KFold(n_splits=3, random_state=0, shuffle=True)

for icv,(trainlocs, testlocs) in enumerate(kf.split(X1s)):
    
    
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

    
    # Build  model
    deepPETase = buildDeepPETase(i=icv, learning_rate=LEARNING_RATE)
    
    
    # Training callbacks
    path = 'experiment/data/finetuning/checkpoints/final'
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
        
    # Save model (rank model that takes [X1, X2] as input)
    deepPETase.load_weights(checkpoint_path)
    deepPETase.save(f'experiment/data/finetuning/models/deepPETase_{icv}.h5', 
                    save_format='h5')






#======================================================================@
# Combined trained models (3-fold CV) into a single ensemble model
#======================================================================@

model_list = []
for icv in range(3):    
    deepPETase = buildDeepPETase(i=icv, learning_rate=1e-6)
    deepPETase.load_weights(f'experiment/data/finetuning/models/deepPETase_{icv}.h5')
    model = deepPETase.layers[2]
    model._name = f'model_{icv}'
    model_list.append(model)  # Model that takes single sequence (X)


# Combine individual models
input_shape = (437,21)
X = tf.keras.layers.Input(input_shape, name='Seq1')
y_list = [model_list[i](X) for i in range(len(model_list))]
y_average = tf.keras.layers.Average(name='average')(y_list)
ensemble_model = tf.keras.Model(X, y_average, name='deepPETase_ensemble_model')


# Save deepPETase ensemble model
modelpath = 'experiment/data/finetuning/models/deepPETase_ensemble_model.h5'
ensemble_model.save(modelpath, overwrite=True, include_optimizer=False, save_format='h5')


