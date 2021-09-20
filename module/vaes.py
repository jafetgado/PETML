"""
Variational autoencoder models
"""




from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

import module.losses as losses
import module.models as models






class ProtVAE():
    '''Variational autoencoders for generative modeling of protein sequences'''
    
    
    def __init__(self, encoder=None, decoder=None, auxdecoder=None, mean=0, stddev=1, 
                 autoregressive=False, beta=1.0, lambda_=1.0, name='m1vae'):
        
        allowed_names = ['m1vae', 'mmdvae', 'auxvae']
        assert (name in allowed_names), f'name must be one of {allowed_names}'
        self.name = name
        self.mean = mean
        self.stddev = stddev
        self.encoder = encoder
        self.decoder = decoder
        self.autoregressive = autoregressive
        self.beta = beta
        if name=='auxvae':
            self.auxdecoder = auxdecoder
            self.lambda_ = lambda_
        
    
    
    
    def vae_loss(self, X_true, X_pred):
        '''Return ELBO loss for VAE'''
        
        Z_mean, Z_var = self.encoder(X_true)
        kl_loss = losses.kl_loss(Z_mean, Z_var)
        entropy_loss = losses.entropy_loss(X_true, X_pred)
        
        return entropy_loss + (self.beta * kl_loss)
    
    
    
    
    def mmdvae_loss(self, X_true, X_pred):
        '''Return loss for MMDVAE'''
        
        Z = self.encoder(X_true)
        mmd_loss = losses.mmd_loss(Z, X_pred, X_true)
        mse_loss = losses.mse_loss(X_true, X_pred)
        
        return mse_loss + (self.beta * mmd_loss)
        
        
        
    
    def _compileM1VAE(self, input_dim, learning_rate, clipnorm):
        '''Build and compile a VAE (M1) from encoder and decoder models 
        (Kingma et al, 2014).'''
        
        # Build VAE
        assert self.name == 'm1vae'
        X_input = Input(shape=input_dim, name='seq_input')
        [Z_mean, Z_var] = self.encoder(X_input)
        sampler = models.normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input])
        else:
            X_pred = self.decoder(Z)
        
        # Compile VAE
        m1vae = Model(X_input, X_pred, name=self.name)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        m1vae.compile(optimizer=optimizer, loss=self.vae_loss, 
                      metrics=[losses.amino_accuracy])
        
        return m1vae
    
    
    
    
    def _compileMMDVAE(self, input_dim, learning_rate, clipnorm):
        '''Build and compile MMD-VAE from encoder and decoder models 
        (Zhao et al, 2019).'''
        
        # Build VAE
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
                      metrics=[losses.amino_accuracy])
        
        return mmdvae
        



    def _compileAUXVAE(self, input_dim, learning_rate, clipnorm):
        '''Build and compile an auxiliary VAE (AUXVAE) from encoder and decoder models
        (Lucas and Verbeek, 2018; Seybold et al, 2019)'''
        
        # Build VAE
        assert self.name == 'auxvae'
        X_input = Input(shape=input_dim, name='seq_input')
        Z_mean, Z_var = self.encoder(X_input)
        sampler = models.normalSampler(mean=self.mean, stddev=self.stddev)
        Z = sampler([Z_mean, Z_var])
        if self.autoregressive:
            X_pred = self.decoder([Z, X_input])
        else:
            X_pred = self.decoder(Z)
        X_pred2 = self.auxdecoder(Z)
        
        # Compile VAE
        auxvae = Model(X_input, [X_pred, X_pred2], name=self.name)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
        auxvae.compile(optimizer=optimizer, loss=[self.vae_loss, losses.entropy_loss], 
                      metrics={'decoder':losses.amino_accuracy},
                      loss_weights=[1, self.lambda_])

        return auxvae
    
    
    
    
    def compileVAE(self, input_dim, learning_rate=1e-4, clipnorm=100.):
        '''Compile the VAE from base models'''

        K.clear_session()
        
        if self.name == 'm1vae':
            self.vae = self._compileM1VAE(input_dim, learning_rate, clipnorm)
        
        elif self.name == 'mmdvae':
            self.vae = self._compileMMDVAE(input_dim, learning_rate, clipnorm)
            
        elif self.name == 'auxvae':
            self.vae = self._compileAUXVAE(input_dim, learning_rate, clipnorm)
    
    
    
    