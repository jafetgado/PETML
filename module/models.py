"""
Functions for building Tensorflow/Keras models
"""




#===============#
# Imports
#===============#


import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2 as L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv1D, \
                                    Add, Concatenate, BatchNormalization, ZeroPadding1D, \
                                    Lambda, GlobalAveragePooling1D, GlobalMaxPooling1D, \
                                    Reshape, Conv2DTranspose, LSTM, UpSampling1D,  \
                                    Multiply, Average, GaussianNoise, Bidirectional,  \
                                    Conv2D, GlobalMaxPooling2D, MaxPooling1D, Subtract
                                   









#=========================#
# Helper model functions
#=========================#


def aminoAccuracy(X_true, X_pred):
    '''Compute the accuracy (sequence identity) between sequence logits (X_true) and 
    the softmax output (X_pred) without considering gaps/padded characters (at index 0)'''

    xtrue_maxlocs = K.argmax(X_true, axis=-1)
    xrecon_maxlocs = K.argmax(X_pred, axis=-1)
    non_dash_mask = tf.greater(xtrue_maxlocs, 0)
    is_equal = K.equal(xtrue_maxlocs, xrecon_maxlocs)
    is_equal_masked = tf.boolean_mask(is_equal, non_dash_mask)
    accurate_counts = tf.cast(is_equal_masked, 'float32')
    accurate_counts = K.sum(accurate_counts)
    all_counts = K.sum(tf.cast(non_dash_mask, 'float32'))
    accuracy = accurate_counts / all_counts
    
    return accuracy






def entropyLoss(X_true, X_pred):
    '''Return the categorical cross entropy'''
    
    e = 1e-10
    xpred = tf.cast(X_pred, dtype=tf.float32)
    xtrue = tf.cast(X_true, dtype=tf.float32)
    xpred = K.clip(xpred, min_value=e, max_value=1.0)
    ce =  xtrue * K.log(xpred)
    ce = -K.sum(K.sum(ce, axis=-1), axis=-1)
    
    return ce  

    



def confusion_matrix(y_true, y_pred, threshold=0.5):
    '''Return the confusion matrix (tp, tn, fp, fn)'''

    y_true = tf.cast(y_true, tf.float32) 
    y_pred = tf.cast(y_pred, tf.float32)
    threshold = tf.cast(threshold, tf.float32)
    predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    true_pos = tf.cast(tf.math.count_nonzero(predicted * y_true), tf.float32)
    true_neg = tf.cast(tf.math.count_nonzero((predicted - 1) * (y_true - 1)),
                        tf.float32)
    false_pos = tf.cast(tf.math.count_nonzero(predicted * (y_true - 1)),
                        tf.float32)
    false_neg = tf.cast(tf.math.count_nonzero((predicted - 1) * y_true),
                        tf.float32)
    
    return true_pos, true_neg, false_pos, false_neg




def tfmcc(y_true, y_pred):
    '''Return the MCC computed with tensorflow'''
    true_pos, true_neg, false_pos, false_neg = confusion_matrix(y_true, y_pred)
    denominator = (true_pos + false_pos) * (true_pos + false_neg) * \
            (true_neg + false_pos) * (true_neg + false_neg)
    denominator = tf.sqrt(denominator) + 1e-8
    numerator = (true_pos * true_neg) - (false_pos * false_neg)
    score = tf.cast(numerator / denominator, tf.float32)
    
    return score




def Conv1DTranspose(input_shape, filters, kernel_size, strides=2, padding='valid', 
                    use_bias=True, kernel_regularizer=None, bias_regularizer=None, 
                    dilation_rate=1, name=None):
    '''Return a custom function to implement Conv1DTranspose from Conv2DTranspose.
    Conv1DTranspose is not available in Tensorflow 2.2'''
    
    X_input = Input(shape=input_shape)
    X = Lambda(lambda xvalue: K.expand_dims(xvalue, axis=2))(X_input)
    X = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), 
                        strides=(strides, 1), padding=padding, activation=None, 
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer, dilation_rate=dilation_rate, 
                        name=name)(X)
    X_output = Lambda(lambda xx: K.squeeze(xx, axis=2))(X)
    model = Model(X_input, X_output, name=name)
    
    return model  







#============================================#
# Variational autoencoder (VAE) models
#============================================#


class VAE():
    '''Class to build and apply VAE models.'''
    
    def __init__(self,
                 seq_len=400,
                 amino_dim=21,
                 dense_units=[1024,512],
                 latent_dim=64,
                 activation='elu',
                 use_batch_norm=False,
                 dropout=0.2,
                 regfactor=1e-4,
                 mean=0.0,
                 stddev=1.0,
                 beta=1.0,
                 learning_rate=1e-4, 
                 clipnorm=10.0):
        '''Initialize VAE'''
        
        # Hyperparameters
        self.seq_len = seq_len
        self.amino_dim = amino_dim
        self.dense_units = dense_units        
        self.latent_dim = latent_dim
        self.activation = activation if activation != 'leaky_relu' else tf.nn.leaky_relu
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout 
        self.l2reg = L2(regfactor)
        self.mean = mean
        self.stddev =  stddev
        self.beta = beta        
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm        
        
       
        # Build models
        self.buildEncoderModel()
        self.buildSamplerModel()
        self.buildDecoderModel()
        self.buildVaeModel()
        self.compileModel()
       



    def buildEncoderModel(self):
        '''Build encoder model for VAE'''
        
        seq_input = Input(shape=(self.seq_len, self.amino_dim), name='enc_seq_input')
        tensor = seq_input
        

        # Dense layers
        tensor = Flatten()(tensor)
        if self.dropout > 0:
            tensor = Dropout(self.dropout)(tensor)
        for i,units in enumerate(self.dense_units):
            tensor = Dense(units=units, activation=None, kernel_regularizer=self.l2reg,
                           bias_regularizer=self.l2reg, name=f'enc_dense_{i}')(tensor)
            if self.use_batch_norm:
                tensor = BatchNormalization()(tensor)
            tensor = Activation(self.activation)(tensor)
            if self.dropout > 0:
                tensor = Dropout(self.dropout)(tensor)
        
        # Mean, variance, and semi-supervised target
        Z_mean = Dense(units=self.latent_dim, activation=None, 
                       kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg, 
                       name='mean_dense')(tensor)
        Z_var = Dense(units=self.latent_dim, activation='softplus', 
                      kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg, 
                      name='var_dense')(tensor)
        
        # Model 
        self.EncoderModel = Model(seq_input, [Z_mean, Z_var], name='encoder')


    

    def buildDecoderModel(self):
        '''Build decoder model for VAE'''
        
        latent_input = Input(shape=self.latent_dim, name='dec_latent_input')
        tensor = latent_input
        
        # Dense layers
        if self.dropout > 0:
            tensor = Dropout(self.dropout)(tensor)
        for i,units in enumerate(self.dense_units[::-1]):
            tensor = Dense(units=units, activation=None, kernel_regularizer=self.l2reg,
                           bias_regularizer=self.l2reg, name=f'dec_dense_{i}')(tensor)
            if self.use_batch_norm:
                tensor = BatchNormalization()(tensor)
            tensor = Activation(self.activation)(tensor)
            if self.dropout > 0:
                tensor = Dropout(self.dropout)(tensor)
        
        # Upsample and reshape
        tensor = Dense(units=int(self.seq_len * self.amino_dim), activation=None,
                       kernel_regularizer=self.l2reg, bias_regularizer=self.l2reg, 
                       name='dec_upsample')(tensor)
        tensor = Reshape((self.seq_len, self.amino_dim))(tensor)
        
        # Softmax output
        seq_output = Activation('softmax')(tensor)
        
        # Model
        self.DecoderModel = Model(latent_input, seq_output, name='decoder')
        
        
        
        
    def buildSamplerModel(self):
        '''Build variational sampling model for VAE'''

        Z_mean_input = Input(shape=self.latent_dim, name='Z_mean')
        Z_var_input = Input(shape=self.latent_dim, name='Z_var')
        epsilon = K.random_normal(mean=self.mean, stddev=self.stddev, 
                                  shape=K.shape(Z_mean_input))
        Z = Z_mean_input + (K.sqrt(Z_var_input + 1e-8) * epsilon)
        self.SamplerModel = Model([Z_mean_input, Z_var_input], Z, name='sampler')
        
        
        
        
    def buildVaeModel(self):
        '''Build full VAE model'''
        
        # Encode sequence
        seq_input = Input(shape=(self.seq_len, self.amino_dim), name='seq_input')
        Z_mean, Z_var = self.EncoderModel(seq_input)
        
        # Variational sampling
        Z = self.SamplerModel([Z_mean, Z_var])
        
        # Decode sequence
        seq_output = self.DecoderModel(Z)
        
        # VAE model
        self.VaeModel = Model(seq_input, seq_output, name='vae')    




    def compileModel(self):
        '''Compile model for training'''
        
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
        self.VaeModel.compile(optimizer=optimizer, loss=self.elboLoss, 
                              weighted_metrics=[entropyLoss, self.kldLoss, aminoAccuracy,
                                                'categorical_accuracy'])
    
    
    
    
    def kldFunction(self, mean, var):
        '''Return the Kullback-leibler Divergence (KLD)'''
        
        return -0.5 * K.sum(1 + K.log(var + 1e-8) - K.square(mean) - var, axis=-1)
    
    
    
    
    def kldLoss(self, X_true, X_pred):
        '''Return the KLD loss of encoder model'''

        Z_mean, Z_var = self.EncoderModel(X_true)
        return self.kldFunction(Z_mean, Z_var)
    
    
    
    
    def elboLoss(self, X_true, X_pred):
        '''Return the Evidence Lower Bound loss (-ELBO) of VAE model'''
        
        return entropyLoss(X_true, X_pred) + self.beta * (self.kldLoss(X_true, X_pred))
    
    
    
    
    def elboIterLoss(self, X_true, batch_size=128, iters=100):
        '''Return the ELBO for X_true averaged over many iterations (iters)'''
        
        elbos = np.zeros((len(X_true), iters))
        for i in range(iters):
            Z_mean, Z_var = self.EncoderModel.predict(X_true, batch_size=batch_size)
            Z = self.SamplerModel.predict([Z_mean, Z_var], batch_size=batch_size)
            X_pred = self.DecoderModel.predict(Z, batch_size=batch_size)
            elbo = entropyLoss(X_true, X_pred) + self.kldFunction(Z_mean, Z_var)
            elbos[:,i] = elbo.numpy()
        elbo_avg = np.mean(elbos, axis=1)
        
        return elbo_avg
    
    
    
    
    def sampleFromLatent(self, mu, sigma, delta=1.0):
        '''Generate sequences from latent space with decoder by adding learned Gaussian
        noise with scaled variance'''
        
        epsilon = np.random.normal(loc=self.mean, scale=(self.stddev * delta), 
                                   size=sigma.shape)
        Z = mu + (sigma * epsilon)
        Xpred = self.DecoderModel.predict(Z)
        
        return Xpred    



    
    
    
    
    
    

#=========================#
# Top/rank models
#=========================#

def buildLinearTopModel(input_shape=64, dropout=0.0, regfactor=1e-2,
                        name='linear_top_model'):
    '''Return a linear top model'''
    
    X_input = Input(shape=input_shape, name='top_input')
    X = X_input
    reg = L2(regfactor)
    if dropout > 0.0:
        X = Dropout(dropout)(X)
    y = Dense(units=1, activation=None, kernel_regularizer=reg, 
              bias_regularizer=reg)(X)
    model = Model(X_input, y, name=name)
    
    return model
    
    
    
    


def buildRankModel(score_model, input_shape=64, learning_rate=1e-4, name='rank_model'):
    '''Return a rank model to predict the relative output of score_model as specified
    by the indicator function, I(score_model(X2) > score_model(X1))'''
    
    X1 = Input(input_shape, name='X1')
    X2 = Input(input_shape, name='X2')
    a1 = score_model(X1)
    a2 = score_model(X2)
    adiff = Subtract()([a2, a1])
    y = Activation('sigmoid', name='sigmoid')(adiff) 
    rankmodel = Model([X1, X2], y, name=name)
    rankmodel.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=10.),
                      loss='binary_crossentropy', 
                      weighted_metrics=['accuracy', tf.keras.metrics.AUC(), tfmcc])
    
    return rankmodel






def combineEncoderAndTopModel(input_shape, encoder_model, top_model, 
                              name='encoder_top_model'):
    '''Return a model (X=>y) that combines a VAE encoder model for feature 
    extraction (X=>Z) and a top model for regression (Z=>y) into a single model (X=>y)'''
    
    X = Input(input_shape, name='seq_input')
    Zmean, Zvar = encoder_model(X)
    y = top_model(Zmean)
    model = Model(X, y, name=name)
    
    return model
    
    
