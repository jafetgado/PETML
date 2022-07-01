"""
Generator preprocessing sequence data in batches
"""



import numpy as np
import tensorflow as tf
import deepPETase.utils as utils






class DataGenerator(tf.keras.utils.Sequence):
    '''Data generator for preprocessing sequence data in batches '''
    
    
    def __init__(self, 
                 X_seq, 
                 X_ref=None,
                 seq_len=437,
                 weight=None,
                 batch_size=256,
                 shuffle=False):
        '''Initialize data generator'''

        # Data shape
        if weight is not None:
            assert len(X_seq) == len(weight)

        # Parameters
        self.X_seq = X_seq  # Categorical/integer encoding of aligned protein sequence
        self.X_ref = X_ref
        self.seq_len = seq_len
        self.ids = np.arange(len(X_seq))
        self.batch_size = batch_size
        self.weight = weight
        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()
        
        
        
        
    def __len__(self):
        '''Return the number of batches in one epoch'''

        num_epochs = int(np.ceil(len(self.ids) / self.batch_size))
        
        return num_epochs




    def __getitem__(self, index):
        '''Generate one batch of data for model training/prediction. Index is an integer 
        denoting the batch and is supplied by tf.keras.utils.sequence.'''

        X, sample_weights = self.preprocess_batch_data(index, vae=True)
        if self.X_ref is None:
            return (X, X, sample_weights)
        else:
            X_ref, sample_weights = self.preprocess_batch_data(index, vae=False)
            y = np.zeros((len(X)))
            return ([X_ref, X], y, sample_weights)  # Predict P(a(X) > a(Xref))
            
        

       

    def preprocess_batch_data(self, index, vae=True):
        '''Preprocess one batch of data from arrays as specified by index 
        (the batch number).'''
        
        # Ids for batch data
        start = index * self.batch_size
        stop = (index + 1) * self.batch_size
        batch_ids = self.ids[start:stop]
        
        # Fetch batch data
        if vae:
            X = self.X_seq[batch_ids,:]
        else:
            X = self.X_ref[batch_ids,:]

        # Sample weights
        if self.weight is not None:
            sample_weights = self.weight[batch_ids]
        else:
            sample_weights = None

        # One-hot encode sequence data
        X = utils.categorical_to_one_hot(X, maxlen=self.seq_len, feature_dim=22)
        X = X[:,:,:21]  # Exclude non-canonical character (X)
        
        return (X, sample_weights)
            
    
    
    
    def on_epoch_end(self):
        '''Reshufle data after each epoch'''
        
        np.random.shuffle(self.ids)
    

