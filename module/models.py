"""
MFunctions for building Keras models
"""




import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, mean_absolute_error, \
                                     mean_squared_error
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, \
                                     MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau                                     
from tensorflow.keras.regularizers import l2 as L2
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten, Reshape, \
                                    Activation, Conv1D, Conv2DTranspose, Add, \
                                    BatchNormalization, Concatenate, Multiply, \
                                    ZeroPadding1D, RepeatVector                                






def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='valid', 
                    activation='elu', kernel_regularizer=None, bias_regularizer=None):
    """Conv1DTranspose is not available in Tensorflow 2.2. This is a custom function to
    implement it from Conv2DTranspose."""
    
    X = Lambda(lambda xx: K.expand_dims(xx, axis=2))(input_tensor)
    X = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), 
                        strides=(strides, 1), padding=padding, activation=activation, 
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)(X)
    X = Lambda(lambda xx: K.squeeze(xx, axis=2))(X)
    
    return X






def multiLayerPerceptron(input_dim=(674,21), cond_dim=4, output_dim=1, 
                         hidden_units=[256,256,256], residual=True, 
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
    '''Return a dense encoder model with sequence one-hot encoding as input and a list of
    mean/var values of the encoded latent space as output'''

    # Inputs
    seq_encoding = Input(shape=input_dim)
    X = Flatten()(seq_encoding)

    # Encoder
    mlp = multiLayerPerceptron(input_dim=int(X.shape[-1]), cond_dim=0, output_dim=None, 
                               hidden_units=dense_units, residual=residual, 
                               output_activation=None, dropout=dropout, 
                               regfactor=regfactor)
    X = mlp(X)
    
    # Variational layer
    Z = Dense(units=latent_dim, activation=None)(X)
    if outputvar:
        Z_var = Dense(units=latent_dim, activation='softplus')(X)
    
    # Model
    if outputvar:
        encoder = Model(seq_encoding, [Z, Z_var], name=name)
    else:
        encoder = Model(seq_encoding, Z, name=name)
    
    return encoder






def denseDecoder(input_dim=(674,21), dense_units=[256,256,256], latent_dim=30, 
                 dropout=0.33, regfactor=1e-4, residual=True, name='decoder'):
    '''Return a dense decoder model with latent representation as input and decoded amino
    acid probabilities as output'''
    
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








def residualConvBlock(input_dim=(674,21), embedding=True, kernel=2, dropout=0.5, 
                      regfactor=1e-4, padding='causal', num_blocks=4, name='convnetwork', 
                       drates=[1,2,4,8,16,32,64,128,256], ):
    '''Return repeated blocks of convolution neural network with residual connections 
    between consecutive blocks. If 
    dilation_layer_type is 'stack', stack consequent convolution layers with increasing 
    dilation rates, else if it is 'concatenate', run parallel convolution layers with 
    increasing dilation rates and concatenate the outputs.'''

    # Inputs
    X_input = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = X_input
    
    
    # Embed one-hot encoding as continuous values
    if embedding:
        X = Conv1D(filters=input_dim[1], kernel_size=1, strides=1, activation=None, 
                   use_bias=False, kernel_regularizer=reg, name='embedding')(X)
    

    # Residual blocks of convolution networks with exponential dilation rates
    for block in range(residual_blocks):
        
        if dilation_layer_type == 'stack':
            for (i, drate) in enumerate(drates):
                if i==0:
                    X0 = X  # Input to first layer for skip connection later on
                X = Conv1D(filters=input_dim[1], kernel_size=kernel, strides=1, 
                           padding=padding, dilation_rate=drate, activation='elu', 
                           kernel_regularizer=reg, bias_regularizer=reg)(X)
            X = Add()([X0, X])
            
            #X = BatchNormalization()(X)
            X = Dropout(dropout)(X)
            
        
        elif dilation_layer_type == 'concatenate':
            X0 = X  # For skip connections
            X_drate_list = []
            for (i, drate) in enumerate(drates):
                X_drate = Conv1D(filters=input_dim[1], kernel_size=kernel, strides=1, 
                                 padding=padding, dilation_rate=drate, activation='elu', 
                                 kernel_regularizer=reg, bias_regularizer=reg)(X)
                X_drate_list.append(X_drate)
            X = Concatenate()(X_drate_list)
            if block != 0:
                X = Add()([X0, X])
    
    X_output = X
    convnet = Model(X_input, X_output, name=name)
    
    return convnet

       



        

def convRegressor(input_dim=(674,21), cond_dim=4, embedding=True, kernel=2, dropout=0.5, 
                  regfactor=1e-4, padding='causal', residual_blocks=4, repeat_cond=False,
                  dilation_layer_type='concatenate',  drates=[1,2,4,8,16,32,64,128,256],
                  dense_units=[128], output_dim=1, output_activation=None, name='convreg'):
    '''Return a dilated convolution neural network with residual connections for 
    regression. Conditional/experimental data are appended to input before the convolution
    layers. If dilation_layer_type is 'stack', stack consequent convolution layers with 
    increasing dilation rates, else if it is 'concatenate', run parallel convolution 
    layers with increasing dilation rates and concatenate the outputs.'''

    # Inputs
    X_input = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = X_input
    
    # Embed one-hot encoding as continuous values
    if embedding:
        X = Conv1D(filters=input_dim[1], kernel_size=1, strides=1, activation=None, 
                   use_bias=False, kernel_regularizer=reg, name='embedding')(X)
    
    # Append experimental conditions
    filter_size = input_dim[1]
    if cond_dim > 0:
        conditions_input = Input(shape=cond_dim)
        if repeat_cond:
            conditions_repeat = RepeatVector(input_dim[0])(conditions_input)
            X = Concatenate()([X, conditions_repeat])
            filter_size =  input_dim[1] + cond_dim
        

    # Residual blocks of convolution networks with exponential dilation rates
    convnet = residualConvBlock(input_dim=(input_dim[0], filter_size), embedding=False,
                                 kernel=kernel, dropout=dropout, regfactor=regfactor, 
                                 padding=padding, residual_blocks=residual_blocks, 
                                 dilation_layer_type=dilation_layer_type, drates=drates,
                                 name=f'{name}_res_blocks')
    X = convnet(X)
    
    # Dense Units
    if dense_units is not None:
        X = Flatten()(X)
        for dense_unit in dense_units:
            X = BatchNormalization()(X)
            X = Dropout(dropout)(X)
            X = Dense(units=input_dim[0] * input_dim[1], activation='elu',
                      kernel_regularizer=reg, bias_regularizer=reg)(X)
            
    
    # Output layer 
    if output_dim > 0:
        X = Flatten()(X)
        if not repeat_cond:
            X = Concatenate()([X, conditions_input])
        X = BatchNormalization()(X)
        X = Dropout(dropout)(X)
        X = Dense(units=output_dim, activation=output_activation, kernel_regularizer=reg,
                  bias_regularizer=reg)(X)
        
    # Model
    X_output = X
    if cond_dim == 0:
        convreg = Model(X_input, X_output, name=name)
    else:
        convreg = Model([X_input, conditions_input], X_output, name=name)
    
    return convreg



def old_convEncoder(input_dim=(674,21), latent_dim=50, embedding=True, kernel=2, dropout=0.5, 
                regfactor=1e-4, padding='causal', dilation_layer_type='concatenate', 
                residual_blocks=4, downsample=2, drates=[1,2,4,8,16,32,64,128,256], 
                name='encoder'):
    '''Return a convolution encoder model with sequence one-hot encoding as input and a 
    list of mean and var values of the encoded latent space as output.'''

    # Inputs
    seq_encoding = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = seq_encoding
    
    # Convolutional encoder
    convnet = residualConvBlock(input_dim=input_dim, embedding=embedding, kernel=kernel,
                                 dropout=dropout, regfactor=regfactor, padding=padding,
                                 dilation_layer_type=dilation_layer_type, drates=drates,
                                 residual_blocks=residual_blocks)
    X = convnet(X)
    
    # Downsample time dimension (by half each step)
    for i in range(downsample):
        X = Conv1D(filters=input_dim[1], kernel_size=kernel, strides=2, 
                   padding=padding, dilation_rate=1, activation='elu', 
                   kernel_regularizer=reg, bias_regularizer=reg)(X)
    
    # Variational layer
    X = Flatten()(X)
    X = BatchNormalization()(X)
    X = Dropout(dropout)(X)
    Z_mean = Dense(units=latent_dim, activation=None, kernel_regularizer=reg, 
                   bias_regularizer=reg)(X)
    Z_var = Dense(units=latent_dim, activation='softplus', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
    
    # Model
    encoder = Model(seq_encoding, [Z_mean, Z_var], name=name)
    
    return encoder





def convEncoder(input_dim=(674,21), latent_dim=50, embedding=True, kernel=2, dropout=0.5, 
                regfactor=1e-4, padding='causal', dilation_layer_type='concatenate', 
                residual_blocks=4, downsample=2, drates=[1,2,4,8,16,32,64,128,256], 
                outputvar=True, name='encoder'):
    '''Return a convolution encoder model with sequence one-hot encoding as input and a 
    list of mean and var values of the encoded latent space as output.'''

    # Inputs
    seq_encoding = Input(shape=input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    X = seq_encoding
    
    # Convolutional encoder
    convnet = residualConvBlock(input_dim=input_dim, embedding=embedding, kernel=kernel,
                                 dropout=dropout, regfactor=regfactor, padding=padding,
                                 dilation_layer_type=dilation_layer_type, drates=drates,
                                 residual_blocks=residual_blocks)
    X = convnet(X)
    
    # Downsample time dimension (by half each step)
    for i in range(downsample):
        X = Conv1D(filters=input_dim[1], kernel_size=kernel, strides=2, 
                   padding=padding, dilation_rate=1, activation='elu', 
                   kernel_regularizer=reg, bias_regularizer=reg)(X)
    
    # Variational layer
    X = Flatten()(X)
    #X = BatchNormalization()(X)
    #X = Dropout(dropout)(X)
    
    # Added a dense layer
    #==============#
    '''
    X = Dense(units=512, activation='elu', kernel_regularizer=reg, 
              bias_regularizer=reg)(X)
    X = BatchNormalization()(X)
    X = Dropout(dropout)(X)
    '''
    X = Dense(units=256, activation='elu', kernel_regularizer=reg, 
              bias_regularizer=reg)(X)
    X = BatchNormalization()(X)
    X = Dropout(dropout)(X)
    
    
    #==============#
    # Latent space
    Z = Dense(units=latent_dim, activation=None)(X)
    if outputvar:
        Z_var = Dense(units=latent_dim, activation='softplus')(X)
    
    # Model
    if outputvar:
        encoder = Model(seq_encoding, [Z, Z_var], name=name)
    else:
        encoder = Model(seq_encoding, Z, name=name)
    
    return encoder




    

def convDecoder(input_dim=(674,21), latent_dim=50, embedding=True, kernel=2, dropout=0.5, 
                input_dropout=0.5, regfactor=1e-4, padding='causal', isM1=True, 
                dilation_layer_type='concatenate', residual_blocks=4, upsample=0, 
                drates=[1,2,4,8,16,32,64,128,256], autoregressive=True, name='decoder'):
    '''Return a convolution decoder model with latent representation as input and decoded 
    amino acid probabilities as output'''

    
    # Inputs
    latent_input = Input(latent_dim)
    protein_input = Input(input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    if not isM1:
        yhat_input = Input(shape=(1,))
        X = Concatenate()([latent_input, yhat_input])
    else:
        X = latent_input
    
    
    # Upsample latent space (Dense if upsample==0, Dense+Conv1DTranspose if upsample > 0)
    if upsample==0:
       
        # Added an extra dense layer
        #==============#
        
        X = Dense(units=256, activation='elu', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
        
        X = BatchNormalization()(X)
        X = Dropout(dropout)(X)
        '''
        X = Dense(units=512, activation='elu', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
        '''
        
        #==============#        
        
        
        
        X = Dense(units=(input_dim[0] * input_dim[1]), activation='elu', 
                  kernel_regularizer=reg, bias_regularizer=reg)(X)
        X = Reshape(input_dim)(X)
        #X = BatchNormalization()(X)
        X = Dropout(dropout)(X)
        
        
    else:
        start_upsample_from =  input_dim[0] * (2.0**-(upsample))
        assert int(start_upsample_from) - start_upsample_from == 0, \
                'input_dim[0] is not divisible by 2^upsample, hence upsampling will not '\
                'result in the same shape'
        upsample_dims = np.array(input_dim[0] * (2.0 ** -np.arange(0, upsample+1)),
                                 dtype=int)[::-1]
        for (i, upsample_dim) in enumerate(upsample_dims):
            if i == 0:
                X = BatchNormalization()(X)
                X = Dropout(dropout)(X)
                X = Dense(units=(upsample_dim * input_dim[1]), activation='elu', 
                          kernel_regularizer=reg, bias_regularizer=reg)(X)
                X = Reshape((upsample_dim, input_dim[1]))(X)
            else:
                X = BatchNormalization()(X)
                X = Dropout(dropout)(X)
                X = Conv1DTranspose(X, filters=input_dim[1], kernel_size=kernel, 
                                    strides=2, padding='valid', activation='elu', 
                                    kernel_regularizer=reg, bias_regularizer=reg)
    
    
    if autoregressive:
        # Sequence shifted one time step backward to predict next time step
        X_protein = ZeroPadding1D(padding=(1,0))(protein_input) 
        X_protein = Lambda(lambda xtime: xtime[:,:-1,:])(X_protein)
    
        
        # Embed one-hot encoding of previous time step as continuous values
        if embedding:
            X_protein = Conv1D(filters=input_dim[1], kernel_size=1, activation=None, 
                               use_bias=False, kernel_regularizer=reg)(X_protein)
        # Apply dropout to amino acid context
        if input_dropout > 0:
            X_protein = Dropout(input_dropout, noise_shape=(None, input_dim[0], 1))(X_protein)
        
    
        # Concatenate decoded latent space and time-shifted sequence encoding
        X = Concatenate(axis=-1)([X, X_protein])
    
    
    # Decode from upsampled latent space with dilated residual convolution networks
    convnet = residualConvBlock(input_dim=list(X.shape[1:]), embedding=False,
                                 kernel=kernel, dropout=dropout, regfactor=regfactor,
                                 padding=padding, residual_blocks=residual_blocks,
                                 dilation_layer_type=dilation_layer_type, drates=drates)
    X = convnet(X)
    

    # Output (reconstruction) layer
    Xhat = Conv1D(filters=input_dim[1], kernel_size=1, strides=1, padding='causal', 
                  dilation_rate=1, activation='softmax', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
    
    # Model
    if not isM1:
        decoder = Model([latent_input, protein_input, yhat_input], Xhat, 
                        name=name)
    else:
        decoder = Model([latent_input, protein_input], Xhat,  
                        name=name)
    
    return decoder   






def old_convDecoder(input_dim=(674,21), latent_dim=50, embedding=True, kernel=2, dropout=0.5, 
                input_dropout=0.5, regfactor=1e-4, padding='causal', isM1=True, 
                dilation_layer_type='concatenate', residual_blocks=4, upsample=2, 
                drates=[1,2,4,8,16,32,64,128,256], name='decoder'):
    '''Return a convolution decoder model with latent representation as input and decoded 
    amino acid probabilities as output'''

    
    # Inputs
    latent_input = Input(latent_dim)
    protein_input = Input(input_dim)
    reg = L2(regfactor) if regfactor > 0 else None
    if not isM1:
        yhat_input = Input(shape=(1,))
        X = Concatenate()([latent_input, yhat_input])
    else:
        X = latent_input
    
    # Sequence shifted one time step backward to predict next time step
    X_protein = ZeroPadding1D(padding=(1,0))(protein_input) 
    X_protein = Lambda(lambda xtime: xtime[:,:-1,:])(X_protein)
    
    # Apply dropout to amino acid context
    if input_dropout > 0:
        X_protein = Dropout(input_dropout, noise_shape=(None, input_dim[0], 1))(X_protein)
        
    # Embed one-hot encoding of previous time step as continuous values
    if embedding:
        X_protein = Conv1D(filters=input_dim[1], kernel_size=1, activation=None, 
                           use_bias=False, kernel_regularizer=reg)(X_protein)
    
    # Upsample latent space (Dense if upsample==0, Dense+Conv1DTranspose if upsample > 0)
    if upsample==0:
        X = BatchNormalization()(X)
        X = Dropout(dropout)(X)
        X = Dense(units=(input_dim[0] * input_dim[1]), activation='elu', 
                  kernel_regularizer=reg, bias_regularizer=reg)(X)
        X = Reshape(input_dim)(X)
    else:
        start_upsample_from =  input_dim[0] * (2.0**-(upsample))
        assert int(start_upsample_from) - start_upsample_from == 0, \
                'input_dim[0] is not divisible by 2^upsample, hence upsampling will not '\
                'result in the same shape'
        upsample_dims = np.array(input_dim[0] * (2.0 ** -np.arange(0, upsample+1)),
                                 dtype=int)[::-1]
        for (i, upsample_dim) in enumerate(upsample_dims):
            if i == 0:
                X = BatchNormalization()(X)
                X = Dropout(dropout)(X)
                X = Dense(units=(upsample_dim * input_dim[1]), activation='elu', 
                          kernel_regularizer=reg, bias_regularizer=reg)(X)
                X = Reshape((upsample_dim, input_dim[1]))(X)
            else:
                X = BatchNormalization()(X)
                X = Dropout(dropout)(X)
                X = Conv1DTranspose(X, filters=input_dim[1], kernel_size=kernel, 
                                    strides=2, padding='valid', activation='elu', 
                                    kernel_regularizer=reg, bias_regularizer=reg)
    
    # Concatenate upsampled latent space and time-shifted sequence encoding
    X = Concatenate(axis=-1)([X, X_protein])
    
    # Decode with dilated residual convolution networks
    convnet = residualConvBlock(input_dim=list(X.shape[1:]), embedding=False,
                                 kernel=kernel, dropout=dropout, regfactor=regfactor,
                                 padding=padding, residual_blocks=residual_blocks,
                                 dilation_layer_type=dilation_layer_type, drates=drates)
    X = convnet(X)
    
    # Output (reconstruction) layer
    X = BatchNormalization()(X)
    X = Dropout(dropout)(X)
    Xhat = Conv1D(filters=input_dim[1], kernel_size=kernel, strides=1, padding='causal', 
                  dilation_rate=1, activation='softmax', kernel_regularizer=reg, 
                  bias_regularizer=reg)(X)
    
    # Model
    if not isM1:
        decoder = Model([latent_input, protein_input, yhat_input], Xhat, 
                        name='decoder')
    else:
        decoder = Model([latent_input, protein_input], Xhat,  
                        name='decoder')
    
    return decoder

    
          





def normalSampler(mean=0.0, stddev=1.0):
    '''Return a Tensorflow Lambda function to sample normally distributed data from an 
    array of mean and variance values.'''
    
    sampler = lambda mean_var: (mean_var[0] + \
                                K.sqrt(tf.convert_to_tensor(mean_var[1] + 1e-8, 
                                                         np.float32)) * \
                                K.random_normal(shape=K.shape(mean_var[0]), mean=mean,
                                                stddev=stddev))
    return Lambda(sampler, name='Sampler')








#==============#
# VAE classes
#==============#


class CustomLosses():
    
    @staticmethod
    def entropy_loss(X_true, X_pred, method='sum', numpy=False):
        '''Return the categorical cross entropy'''
        
        loss = categorical_crossentropy(X_true, X_pred)
        if method == 'sum':
            loss = K.sum(loss, axis=-1)
        elif method == 'mean':
            loss = K.mean(loss, axis=-1)
        if numpy:
            loss = loss.numpy()
        
        return loss
    
    
    
    
    @staticmethod
    def mae_loss(y_true, y_pred, method='sum', numpy=False):
        '''Return the mean absolute error'''
        
        loss = mean_absolute_error(y_true, y_pred)
        if method == 'sum':
            loss = K.sum(loss, axis=-1)
        elif method == 'mean':
            loss = K.mean(loss, axis=-1)
        if numpy:
            loss = loss.numpy()
        
        return loss
    
    
    
    @staticmethod
    def mse_loss(y_true, y_pred, method='mean', numpy=False):
        '''Return the mean squared error'''
        
        loss = mean_squared_error(y_true, y_pred)
        if method == 'sum':
            loss = K.sum(loss, axis=-1)
        elif method == 'mean':
            loss = K.mean(loss, axis=-1)
        if numpy:
            loss = loss.numpy()
        
        return loss
    
    
    
    
    @staticmethod
    def mmd_loss(train_z, train_xr, train_x):
        '''Compute the MMD divergence between posterior and prior'''
        
        def compute_kernel(x, y):
            x_size = K.shape(x)[0]
            y_size = K.shape(y)[0]
            dim = K.shape(x)[1]
            tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
            tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
            kernel = -K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32')
            kernel = K.exp(kernel)
            return kernel
    
    
        def compute_mmd(x, y):
            x_kernel = compute_kernel(x, x)
            y_kernel = compute_kernel(y, y)
            xy_kernel = compute_kernel(x, y)
            return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
       
        # Sample from random noise
        batch_size = K.shape(train_z)[0]
        latent_dim = K.int_shape(train_z)[1]
        true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)

        # Calculate MMD 
        loss_mmd = compute_mmd(true_samples, train_z)
    
        return loss_mmd
            
    
    
    @staticmethod
    def kl_loss(Z_mean, Z_var, numpy=False):
        '''Return the Kullback-Leibler divergence'''
    
        loss = - 0.5 * K.sum(1 + K.log(Z_var + 1e-8) - K.square(Z_mean) - Z_var, axis=-1)
        #loss = - 0.5 * K.sum(1 + K.log(Z_var + 1e-8) - K.square(Z_mean) - Z_var, axis=-1)
        if numpy:
            loss = loss.numpy()
        
        return loss
    
    
    @staticmethod
    def amino_accuracy(X_true, X_pred, numpy=False):
        '''Return the sequence identity between encodings X_true and X_pred without
        considering gaps'''
    
        xtrue_maxlocs = K.argmax(X_true, axis=-1)
        xrecon_maxlocs = K.argmax(X_pred, axis=-1)
        non_dash_mask = tf.greater(xtrue_maxlocs, 0)
        is_equal = K.equal(xtrue_maxlocs, xrecon_maxlocs)
        is_equal_masked = tf.boolean_mask(is_equal, non_dash_mask)
        accurate_counts = tf.cast(is_equal_masked, 'float32')
        accurate_counts = K.sum(accurate_counts)
        all_counts = K.sum(tf.cast(non_dash_mask, 'float32'))
        accuracy = accurate_counts / all_counts
        if numpy:
            accuracy = accuracy.numpy()
        
        return accuracy








class ProtVAE():
    '''Unsupervised/semi-supervised variational autoencoders for protein sequences'''
    
    
    def __init__(self, encoder=None, decoder=None, decoder2=None, regressor=None,
                 encoder1=None, encoder2=None, regressor1=None, mean=0, stddev=1, 
                 autoregressive=False, beta=1.0, name='m1vae'):
        
        allowed_names = ['m1vae', 'm2vae', 'mjvae', 'mbvae', 'mmdvae', 'agave']
        assert (name in allowed_names), f'name must be one of {allowed_names}'
        self.name = name
        self.mean = mean
        self.stddev = stddev
        self.encoder = encoder
        self.decoder = decoder
        self.regressor = regressor
        self.autoregressive = autoregressive
        self.beta = beta
        if name=='mbvae':
            self.encoder1 = encoder1
            self.encoder2 = encoder2
            self.regressor1 = regressor1
        if name=='agave':
            self.decoder2 = decoder2
    
    
    
    def vae_loss(self, X_true, X_pred):
        '''Return ELBO loss for VAE'''
        
        Z_mean, Z_var = self.encoder(X_true)
        kl_loss = CustomLosses.kl_loss(Z_mean, Z_var)
        entropy_loss = CustomLosses.entropy_loss(X_true, X_pred)
        
        return entropy_loss + (self.beta * kl_loss)
    
    
    
    def mmdvae_loss(self, X_true, X_pred):
        '''Return MMD loss for MMDVAE'''
        
        Z = self.encoder(X_true)
        mmd_loss = CustomLosses.mmd_loss(Z, X_pred, X_true)
        mse_loss = CustomLosses.mse_loss(X_true, X_pred)
        
        return mse_loss + (self.beta * mmd_loss)
        
        
    def _compileMMDVAE(self, input_dim, latent_dim, learning_rate, clipnorm):
        '''Build and compile and MMDVAE from encoder and decoder models'''
        
        # Build VAE
        K.clear_session()
        assert self.name == 'mmdvae'
        X_input = Input(shape=input_dim, name='seq_input')
        Z = self.encoder(X_input)
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input])
        else:
            X_pred = self.decoder(Z)
        
        # Compile VAE
        mmdvae = Model(X_input, X_pred, name=self.name)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        mmdvae.compile(optimizer=optimizer, loss=self.mmdvae_loss, 
                      metrics=[CustomLosses.amino_accuracy])
        
        return mmdvae
        


    def _compileM1VAE(self, input_dim, latent_dim, learning_rate, clipnorm):
        '''Build and compile a vanilla SSL VAE (M1) from encoder and decoder models'''
        
        # Build VAE
        K.clear_session()
        assert self.name == 'm1vae'
        X_input = Input(shape=input_dim, name='seq_input')
        [Z_mean, Z_var] = self.encoder(X_input)
        sampler = normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input])
        else:
            X_pred = self.decoder(Z)
        
        # Compile VAE
        m1vae = Model(X_input, X_pred, name=self.name)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        m1vae.compile(optimizer=optimizer, loss=self.vae_loss, 
                      metrics=[CustomLosses.amino_accuracy])
        
        return m1vae
    
    
    def _compileAGAVE(self, input_dim, latent_dim, learning_rate, clipnorm):
        '''Build and compile a AGAVE from encoder and decoder models'''
        
        # Build VAE
        K.clear_session()
        assert self.name == 'agave'
        X_input = Input(shape=input_dim, name='seq_input')
        Z_mean, Z_var = self.encoder(X_input)
        sampler = normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input])
        else:
            X_pred = self.decoder(Z)
        X_pred2 = self.decoder2(Z)
        
        # Compile VAE
        agave = Model(X_input, [X_pred, X_pred2], name=self.name)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        agave.compile(optimizer=optimizer, loss=[self.vae_loss, CustomLosses.entropy_loss], 
                      metrics={'decoder':CustomLosses.amino_accuracy},
                      loss_weights=[1, 1])

        return agave
    
    
    def _compileM2VAE(self, input_dim, latent_dim, learning_rate, clipnorm, 
                      cond_dim, alpha):
        '''Build and compile a Kingma SSL VAE (M2) from encoder, decoder, and regressor
        models'''
        
        # Build VAE
        K.clear_session()
        assert self.name == 'm2vae'
        X_input = Input(shape=input_dim, name='seq_input')
        label_mask = Input(shape=(1,), name='label_mask')
        Z_mean, Z_var = self.encoder(X_input)
        sampler = normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if cond_dim > 0:
            conditions = Input(shape=(cond_dim,), name='exp_conditions')
            y_hat = self.regressor([X_input, conditions])
        else:
            y_hat = self.regressor(X_input)
        y_masked = Multiply(name='masked')([y_hat, label_mask])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input, y_hat])
        else:
            X_pred = self.decoder([Z, y_hat])
        
        # Compile VAE
        if cond_dim > 0:
            m2vae = Model([X_input, conditions, label_mask], [X_pred, y_masked, y_hat], 
                          name=self.name)
        else:
            m2vae = Model([X_input, label_mask], [X_pred, y_masked, y_hat])
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        m2vae.compile(optimizer=optimizer, 
                      loss=[self.vae_loss, CustomLosses.mae_loss, None],
                      loss_weights=[1, alpha, 0], 
                      metrics={'decoder':CustomLosses.amino_accuracy})

        return m2vae
        
    
    def _compileMJVAE(self, input_dim, latent_dim, learning_rate, clipnorm,
                      cond_dim, alpha):
        '''Build and compile a Jurasińki SSL VAE (MJ) from encoder, decoder, and regressor
        models'''
        
        # Build VAE
        K.clear_session()
        assert self.name == 'mjvae'
        X_input = Input(shape=input_dim, name='seq_input')
        label_mask = Input(shape=(1,), name='label_mask')
        Z_mean, Z_var = self.encoder(X_input)
        sampler = normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if cond_dim > 0:
            conditions = Input(shape=(cond_dim,), name='exp_conditions')
            y_hat = self.regressor([Z_mean, conditions])
        else:
            y_hat = self.regressor(Z_mean)
        y_masked = Multiply(name='masked')([y_hat, label_mask])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input, y_hat])
        else:
            X_pred = self.decoder([Z, y_hat])
        
        # Compile VAE
        if cond_dim > 0:
            mjvae = Model([X_input, conditions, label_mask], [X_pred, y_masked, y_hat], 
                          name=self.name)
        else:
            mjvae = Model([X_input, label_mask], [X_pred, y_masked, y_hat])
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        mjvae.compile(optimizer=optimizer, loss=[self.vae_loss, CustomLosses.mae_loss, None],
                      loss_weights=[1, alpha, 0], 
                      metrics={'decoder':CustomLosses.amino_accuracy})

        return mjvae
    
        
        
    def _compileMBVAE(self, input_dim, latent_dim, learning_rate, clipnorm,
                      cond_dim, alpha):
        '''Build and compile a Berkhahn SSL VAE (MB) from encoder and decoder
        models'''
        
        K.clear_session()
        assert self.name == 'mbvae'
        
        # Build full encoder from encoder1 and encoder2
        def get_encoder():
            enc_input = Input(shape=input_dim, name='enc_input')
            encoded_rep = self.encoder1(enc_input)     
            z_mean_enc, z_var_enc = self.encoder2(encoded_rep)
            return Model(enc_input, [z_mean_enc, z_var_enc], name='encoder')
        self.encoder = get_encoder()
            
        # Build full regressor from encoder1 and regressor
        def get_regressor():
            reg1_input = Input(shape=input_dim, name='reg_input')
            encoded_rep = self.encoder1(reg1_input)
            if cond_dim > 0:
                cond_input = Input(shape=cond_dim, name='cond_input')
                reg1_output = self.regressor1([encoded_rep, cond_input])
                regressor = Model([reg1_input, cond_input], reg1_output,
                                  name='regressor')
            else:
                reg1_output = self.regressor1(encoded_rep)
                regressor = Model(reg1_input, reg1_output, name='regressor')
            return regressor
        self.regressor = get_regressor()
        
        # Build VAE from encoder, decoder, and regressor
        X_input = Input(shape=input_dim, name='seq_input')
        label_mask = Input(shape=(1,), name='label_mask')
        Z_mean, Z_var = self.encoder(X_input)
        sampler = normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if cond_dim > 0:
            conditions = Input(shape=(cond_dim,), name='exp_conditions')
            y_hat = self.regressor([X_input, conditions])
        else:
            y_hat = self.regressor(X_input)
        y_masked = Multiply(name='masked')([y_hat, label_mask])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input, y_hat])
        else:
            X_pred = self.decoder([Z, y_hat])
        
        # Compile VAE
        if cond_dim > 0:
            mbvae = Model([X_input, conditions, label_mask], [X_pred, y_masked, y_hat], 
                          name=self.name)
        else:
            mbvae = Model([X_input, label_mask], [X_pred, y_masked, y_hat])
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        mbvae.compile(optimizer=optimizer, loss=[self.vae_loss, CustomLosses.mae_loss, None],
                      loss_weights=[1, alpha, 0], 
                      metrics={'decoder':CustomLosses.amino_accuracy})

        return mbvae
    
    
    def compileVAE(self, input_dim, latent_dim, learning_rate=1e-4, clipnorm=50., 
                   num_conditions=4, alpha=1):
        '''Compile the VAE from base models'''
        
        if self.name == 'm1vae':
            self.vae = self._compileM1VAE(input_dim, latent_dim, learning_rate, 
                                          clipnorm)
        elif self.name == 'm2vae':
            self.vae = self._compileM2VAE(input_dim, latent_dim, learning_rate, 
                                          clipnorm, num_conditions, alpha)
        elif self.name == 'mjvae':
            self.vae = self._compileMJVAE(input_dim, latent_dim, learning_rate, 
                                          clipnorm, num_conditions, alpha)
        elif self.name == 'mbvae':
            self.vae = self._compileMBVAE(input_dim, latent_dim, learning_rate, 
                                          clipnorm, num_conditions, alpha)
        elif self.name == 'mmdvae':
            self.vae = self._compileMMDVAE(input_dim, latent_dim, learning_rate, 
                                           clipnorm)
        elif self.name == 'agave':
            self.vae = self._compileAGAVE(input_dim, latent_dim, learning_rate, 
                                       clipnorm)
    
    
        
        
        
        
        
class DataGenerator(tf.keras.utils.Sequence):
    '''
    Data generator for loading and preprocessing sequence data in batches for 
    semi-supervised learning (M2, MJ, MB) with appended experimental conditions..
    
    Parameters
    ------------
    X_labeled : 2D array
        Integer-encoded sequence data for labeled data with target labels     
    y_labeled : 1D array
        Target labels of labeled data
    X_unlabeled : 2D array
        Integer-encoded sequence data for unlabeled data with no target labels
    cond_labeled : 2D array
        Extra conditional/experimental data for labeled values, if available.
    maxlen : int
        Maximum length of sequence data, also 2nd dimension of generated training data
    feature_dim : int
        Number of integer-encoded features, also 3rd dimension of generated training data
    beta : float
        Ratio of unlabaled data size to labeled data size
    batch_size : int
        Number of samples (labeled and unlabeled) in a batch
    reshuffle : bool
        Whether to reshuffle the order of samples before training.
    '''
    
    
    def __init__(self, X_labeled, y_labeled, X_unlabeled, cond_labeled, maxlen=600,
                 feature_dim=22, beta=1, batch_size=512, reshuffle=True):
        '''Initialize data generator'''
        
        # Ensure labeled data is of same length
        assert len(X_labeled) == len(y_labeled)
        assert len(y_labeled) == len(cond_labeled)
        
        # Initialize data as class attributes
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled
        self.X_unlabeled = X_unlabeled
        self.cond_labeled = cond_labeled
        self.maxlen = maxlen
        self.feature_dim = feature_dim
        self.reshuffle = reshuffle
        
        # Derive unique ids for data
        self.ids_unlabeled = np.arange(len(X_unlabeled))
        self.ids_labeled = np.arange(len(X_labeled))
        
        # Randomly oversample labeled data to num_epochs * batch_size
        self.ids_labeled = np.random.choice(self.ids_labeled,
                                            size=len(self.ids_unlabeled),
                                            replace=True)
        
        # Batch data
        self.beta = beta
        self.batch_size = batch_size
        self.batch_size_labeled = int(1 / (beta + 1) * batch_size)
        self.batch_size_unlabeled = batch_size - self.batch_size_labeled
        self.num_epochs = self.__len__()
        
        return
        
        
    def __len__(self):
        '''Return the number of batches in one epoch'''

        num_epochs = len(self.ids_unlabeled) / self.batch_size_unlabeled
        num_epochs = int(np.ceil(num_epochs))
        
        return num_epochs


    def __getitem__(self, index):
        '''Generate one batch of data. Index is an integer denoting the batch, ranging
        from 0 to (self.__len__() - 1) and is supplied by tf.keras.utils.sequence.'''
        # Retrieve one unlabeled batch of unlabeled sequence data
        # Retrieve one labeled batch of labeled sequence data 
        # Convert sequence data to one-hot encoded arrays
        # Derive experimental/conditional data for unlabeled data by random sampling
        
        # Ids for labeled batch data
        start = index * self.batch_size_labeled
        stop = (index + 1) * self.batch_size_labeled
        labeled_batch_ids = self.ids_labeled[start:stop]
        
        # Convert sequence data to one-hot encoded array
        X_labeled_seq = self.X_labeled[labeled_batch_ids,:]
        X_labeled_batch = utils.categorical_to_one_hot(X_labeled_seq, maxlen=self.maxlen,
                                                       feature_dim=self.feature_dim)
        # Other labeled batch data
        cond_labeled_batch = self.cond_labeled[labeled_batch_ids, :]
        y_labeled_batch = self.y_labeled[labeled_batch_ids]
        
        # Ids for unlabeled batch data
        start = index * self.batch_size_unlabeled
        stop = (index + 1) * self.batch_size_unlabeled
        unlabeled_batch_ids = self.ids_unlabeled[start:stop]
        
        # Convert sequence data to one-hot encoded array
        X_unlabeled_seq = self.X_unlabeled[unlabeled_batch_ids, :]
        X_unlabeled_batch = utils.categorical_to_one_hot(X_unlabeled_seq, 
                                                         maxlen=self.maxlen, 
                                                         feature_dim=self.feature_dim)
        
        # Use randomly shuffled conditional labeled data for unlabeled
        cond_unlabeled_batch = np.zeros((X_unlabeled_seq.shape[0], 
                                         cond_labeled_batch.shape[1]))
        for i in range(cond_labeled_batch.shape[1]):
            cond_unlabeled_batch[:,i] = np.random.choice(cond_labeled_batch[:,i], 
                                                         replace=True, 
                                                         size=X_unlabeled_seq.shape[0])
        
        # Append labeled and unlabeled data
        X = np.append(X_labeled_batch, X_unlabeled_batch, axis=0)
        y = np.append(y_labeled_batch, np.zeros(len(X_unlabeled_batch)), axis=0)
        cond = np.append(cond_labeled_batch, cond_unlabeled_batch, axis=0)
        label_mask = np.array([1] * len(y_labeled_batch) +
                              [0] * len(X_unlabeled_batch))
        
        # Reshuffle data
        if self.reshuffle:
            shuffle = np.random.choice(range(len(y)), size=len(y), replace=False)
            X = X[shuffle]
            y = y[shuffle]
            cond = cond[shuffle]
            label_mask = label_mask[shuffle]
       
        return ([X, cond, label_mask], [X, y, y])
        
        
    def on_epoch_end(self):
        '''Reshufle data after each epoch'''
        if self.reshuffle:
            np.random.shuffle(self.ids_labeled)
            np.random.shuffle(self.ids_unlabeled)   
        
        return


    def evaluate(self, model):
        '''Apply fitted model to predict target values of data generated by the data
        generator.'''
        
        # Initialize empty lists
        Xtrue_all, mask_all, ytrue_all, Xpred_all, ypred_all = [], [], [], [], []
        
        # Predict data in batches and store results
        for index in range(self.num_epochs):
            
            # True data
            Xdata, ydata = self.__getitem__(index)
            Xtrue_all.extend(Xdata[0])
            mask_all.extend(Xdata[-1])
            ytrue_all.extend(ydata[-1])
            
            # Predict data
            Xpred, ymask, ypred = model.predict(Xdata, batch_size=self.batch_size)
            Xpred_all.extend(Xpred)
            ypred_all.extend(ypred)
        
        # Convert to arrays
        Xtrue_all = np.array(Xtrue_all).reshape(-1, self.maxlen, self.feature_dim)
        Xpred_all = np.array(Xpred_all).reshape(-1, self.maxlen, self.feature_dim)
        mask_all = np.array(mask_all).reshape(-1)
        ytrue_all = np.array(ytrue_all).reshape(-1)
        ypred_all = np.array(ypred_all).reshape(-1)
        
        
        return (Xtrue_all, mask_all, ytrue_all, Xpred_all, ypred_all)

