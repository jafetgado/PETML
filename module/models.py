"""
Functions for building Keras models
"""




import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2 as L2
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten, Reshape, \
                                    Activation, Conv1D, Conv2DTranspose, Add, \
                                    Concatenate, ZeroPadding1D                               






def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='valid', 
                    activation='elu', kernel_regularizer=None, bias_regularizer=None):
    '''Custom implementation of Conv1DTranspose layer from Conv2DTranspose'''
    
    X = Lambda(lambda xx: K.expand_dims(xx, axis=2))(input_tensor)
    X = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), 
                        strides=(strides, 1), padding=padding, activation=activation, 
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)(X)
    X = Lambda(lambda xx: K.squeeze(xx, axis=2))(X)
    
    return X






def multiLayerPerceptron(input_dim=(674,21), cond_dim=4, output_dim=1, 
                         hidden_units=[256,256,256], residual=False, 
                         output_activation=None, dropout=0.5, regfactor=1e-4, 
                         name='multilayerperceptron'):
    '''Return a multi-layer perceptron'''
    
    # Inputs
    X_input = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = Flatten()(X_input)
    if cond_dim > 0:
        conditions_input = Input(shape=cond_dim)
        X = Concatenate()([X, conditions_input])
        
    # Dense hidden layers
    for (i,units) in enumerate(hidden_units):       
        X0 = X
        X = Dense(units=units, activation='elu', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
        X = Dropout(dropout)(X)
        if residual and (X0.shape[-1] == X.shape[-1]):
            X = Add()([X0, X])
        
    # Output layer
    if output_dim in [None, 0]:
        X_output = X
    else:
        X_output = Dense(units=output_dim, activation=output_activation, 
                         kernel_regularizer=reg, bias_regularizer=reg)(X)
    
    # Model
    if cond_dim > 0:
        mlp = Model([X_input, conditions_input], X_output, name=name)
    else:
        mlp = Model(X_input, X_output, name=name)
    
    return mlp






def denseEncoder(input_dim=(674,21), dense_units=[256,256,256], latent_dim=30, 
                 dropout=0.33, regfactor=1e-4, residual=True, outputvar=True,
                 name='encoder'):
    '''Return a dense VAE encoder model with sequence one-hot encoding as input and a 
    list of mean/var values of the encoded latent space as output'''

    # Inputs
    seq_encoding = Input(shape=input_dim)
    X = Flatten()(seq_encoding)
    reg = L2(regfactor) if regfactor > 0 else None

    # Encoder
    mlp = multiLayerPerceptron(input_dim=int(X.shape[-1]), cond_dim=0, output_dim=None, 
                               hidden_units=dense_units, residual=residual, 
                               output_activation=None, dropout=dropout, 
                               regfactor=regfactor)
    X = mlp(X)
    
    # Variational layer
    Z = Dense(units=latent_dim, activation=None, kernel_regularizer=reg,
              bias_regularizer=reg)(X)
    if outputvar:
        Z_var = Dense(units=latent_dim, activation='softplus', kernel_regularizer=reg,
                      bias_regularizer=reg)(X)
    
    # Model
    if outputvar:
        encoder = Model(seq_encoding, [Z, Z_var], name=name)
    else:
        encoder = Model(seq_encoding, Z, name=name)
    
    return encoder






def denseDecoder(input_dim=(674,21), dense_units=[256,256,256], latent_dim=30, 
                 dropout=0.33, regfactor=1e-4, residual=True, name='decoder'):
    '''Return a dense VAE decoder model with latent representation as input and decoded 
    amino acid probabilities as output'''
    
    # Inputs
    Z_input = Input(shape=latent_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = Z_input

    # Decoder
    mlp = multiLayerPerceptron(input_dim=int(X.shape[-1]), cond_dim=0, output_dim=None, 
                               hidden_units=dense_units, residual=residual, 
                               output_activation=None, dropout=dropout,
                               regfactor=regfactor)
    X = mlp(X)
   
    # Output layer
    X = Dense(units=input_dim[0] * input_dim[1], activation=None, kernel_regularizer=reg,
              bias_regularizer=reg)(X)
    X = Reshape(input_dim)(X)
    Xhat = Activation('softmax', name='Xhat')(X)
    
    # Model
    decoder = Model(Z_input, Xhat, name=name)
    
    return decoder






def residualConvBlock(input_dim=(674,21), filter_size=64, kernel=2, dropout=0.5, 
                      regfactor=1e-4, padding='causal', num_blocks=4, 
                      drates=[1,2,4,8,16,32,64,128,256], name='convnetwork'):
    '''Return a model of repeated blocks of several convolution layers with exponentially
    increasing dilation rates and residual connections between consecutive blocks'''

    # Inputs
    X_input = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = X_input
    
    # Linear embedding to filter_size
    X = Conv1D(filters=filter_size, kernel_size=1, strides=1, activation=None, 
               use_bias=False, kernel_regularizer=reg, name='embedding')(X)
    
    # Residual blocks of convolution networks with exponential dilation rates
    for block in range(num_blocks):
        for (i, drate) in enumerate(drates):
            if i==0:
                X0 = X  # Input to first layer of block for residualconnection later.
            X = Conv1D(filters=filter_size, kernel_size=kernel, strides=1, 
                       padding=padding, dilation_rate=drate, activation='elu', 
                       kernel_regularizer=reg, bias_regularizer=reg)(X)
        X = Add()([X0, X])
        X = Dropout(dropout)(X)
        
    # Model
    X_output = X
    resconvblock = Model(X_input, X_output, name=name)
    
    return resconvblock

       



        
def convEncoder(input_dim=(674,21), latent_dim=50, filter_size=64, kernel=2, dropout=0.5, 
                regfactor=1e-4, padding='causal',  num_blocks=4, outputvar=True, 
                drates=[1,2,4,8,16,32,64,128,256], name='convencoder'):
    '''Return a convolutional VAE encoder model with sequence one-hot encoding as input 
    and a list of mean/var values of the encoded latent space as output.'''

    # Inputs
    seq_encoding = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = seq_encoding
    
    # Convolutional encoder
    resconvblock = residualConvBlock(input_dim=input_dim, filter_size=filter_size, 
                                     kernel=kernel, dropout=dropout, regfactor=regfactor,
                                     padding=padding, drates=drates, 
                                     num_blocks=num_blocks)
    X = resconvblock(X)
    
    # Variational layer
    X = Flatten()(X)
    Z = Dense(units=latent_dim, activation=None, kernel_regularizer=reg,
              bias_regularizer=reg)(X)
    if outputvar:
        Z_var = Dense(units=latent_dim, activation='softplus', kernel_regularizer=reg,
                      bias_regularizer=reg)(X)
    
    # Model
    if outputvar:
        encoder = Model(seq_encoding, [Z, Z_var], name=name)
    else:
        encoder = Model(seq_encoding, Z, name=name)
    
    return encoder




    

def convDecoder(input_dim=(674,21), latent_dim=50, filter_size=64, kernel=2, dropout=0.5, 
                input_dropout=0.5, regfactor=1e-4, padding='causal', num_blocks=4, 
                drates=[1,2,4,8,16,32,64,128,256], autoregressive=True, name='decoder'):
    '''Return a convolutional VAE decoder model with latent representation as input and 
    decoded amino acid probabilities as output'''

    # Inputs
    latent_input = Input(latent_dim)
    protein_input = Input(input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = latent_input
    
    
    # Upsample and reshape embedding dimensions
    X = Dense(units=(input_dim[0] * filter_size), activation='elu', 
              kernel_regularizer=reg, bias_regularizer=reg)(X)
    X = Reshape((input_dim[0], filter_size))(X)
    X = Dropout(dropout)(X)
        
    # Concatenate with input shifted forward one time step
    if autoregressive:
        # Shift input
        X_protein = ZeroPadding1D(padding=(1,0))(protein_input) 
        X_protein = Lambda(lambda xtime: xtime[:,:-1,:])(X_protein)
        
        # Apply dropout to amino acid context
        if input_dropout > 0:
            dropout_layer = Dropout(input_dropout, noise_shape=(None, input_dim[0], 1))
            X_protein = dropout_layer(X_protein)
        
        # Embed one-hot encoding of input protein 
        X_protein = Conv1D(filters=filter_size, kernel_size=1, activation=None, 
                           use_bias=False, kernel_regularizer=reg)(X_protein)
        
        # Concatenate decoded latent space and time-shifted input sequence encoding
        X = Concatenate(axis=-1)([X, X_protein])
    
    # Decode from upsampled latent space with residual convolution blocks ()
    resconvblock = residualConvBlock(input_dim=input_dim, filter_size=filter_size, 
                                     kernel=kernel, dropout=dropout, regfactor=regfactor, 
                                     padding=padding, num_blocks=num_blocks, 
                                     drates=drates)
    X = resconvblock(X)
    
    # Output (reconstruction) layer
    Xhat = Conv1D(filters=input_dim[1], kernel_size=1, strides=1, padding='causal', 
                  dilation_rate=1, activation='softmax', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
    decoder = Model([latent_input, protein_input], Xhat,  name=name)
    
    return decoder   






def normalSampler(mean=0.0, stddev=1.0):
    '''Return a Lambda function to sample from a normal distribution'''
    
    sampler = lambda mean_var: (mean_var[0] + \
                                K.sqrt(tf.convert_to_tensor(mean_var[1] + 1e-8, 
                                                         np.float32)) * \
                                 K.random_normal(shape=K.shape(mean_var[0]), mean=mean,
                                                stddev=stddev))

    return Lambda(sampler, name='Sampler')
