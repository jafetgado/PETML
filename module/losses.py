"""
Custom losses for training Tensorflow/Keras models
"""




import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error




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
    
    
    
    
def mae_loss(X_true, X_pred, method='sum', numpy=False):
    '''Return the mean absolute error'''
    
    loss = mean_absolute_error(X_true, X_pred)
    if method == 'sum':
        loss = K.sum(loss, axis=-1)
    elif method == 'mean':
        loss = K.mean(loss, axis=-1)
    if numpy:
        loss = loss.numpy()
    
    return loss
    
    
    
    
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
    
    
    
    
def compute_kernel(x, y):
    '''Compute the kernel for MMD-VAE'''
    
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    kernel = -K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32')
    kernel = K.exp(kernel)
    
    return kernel




def compute_mmd(x, y):
    '''Compute the maximum mean discrepancy (MMD)'''
    
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)    



    
def mmd_loss(train_z, train_xr, train_x):
    '''Compute the MMD divergence for MMD-VAE'''
   
    batch_size = K.shape(train_z)[0]
    latent_dim = K.int_shape(train_z)[1]
    true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    loss_mmd = compute_mmd(true_samples, train_z)

    return loss_mmd

        
    
    
def kl_loss(Z_mean, Z_var, numpy=False):
    '''Compute the Kullback-Leibler divergence'''

    loss_kl = - 0.5 * K.sum(1 + K.log(Z_var + 1e-8) - K.square(Z_mean) - Z_var, axis=-1)
    if numpy:
        loss_kl = loss_kl.numpy()
    
    return loss_kl
    
    
    
    
def amino_accuracy(X_true, X_pred, numpy=False):
    '''Compute the sequence identity between encodings X_true and X_pred without
    considering gaps (at index 0)'''

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











