# -*- coding: utf-8 -*-
# """
# Created on Sat Jun  5 21:47:39 2021

# @author: VIVEK OOMMEN
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense

import matplotlib.pyplot as plt

class ae(tf.keras.Model):

    def __init__(self):
        super(ae, self).__init__()
        np.random.seed(23)
        tf.random.set_seed(23)


        #Defining some model parameters
        self.latent_dim = 196

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr=10**-4

        self.encoder_ls = self.build_encoder()
        self.decoder_ls  = self.build_decoder()

    def build_encoder(self):
        ls=[]

        ls.append( Conv2D(16, (3,3), activation = 'relu', padding='same', name='conv1', input_shape=[128,128,1]) )
        ls.append( MaxPooling2D(pool_size = (2,2), strides = 2, name='pool1') ) #[64,64,16]

        ls.append( Conv2D(8, (3,3), activation = 'relu', padding='same', name='conv2') )
        ls.append( MaxPooling2D(pool_size = (2,2), strides = 2, name='pool2') ) #[32,32,8]

        ls.append( Conv2D(4, (3,3), activation = 'relu', padding='same', name='conv3') )
        ls.append( MaxPooling2D(pool_size = (2,2), strides = 2, name='pool3') ) #[16,16,4]

        ls.append( Flatten())

        ls.append( Dense(self.latent_dim, name='dense1') )

        return ls

    def build_decoder(self):
        ls=[]

        ls.append( Dense(16*16*4, activation='relu', name='dense2') )

        ls.append( Reshape((16,16,4)) )

        ls.append( Conv2DTranspose(4, kernel_size=(2,2), strides=(2,2), activation = 'relu',  name='tconv1') ) #[32,32,4]
        ls.append( Conv2DTranspose(8, kernel_size=(2,2), strides=(2,2), activation = 'relu', name='tconv2') ) #[64,64,8]
        ls.append( Conv2DTranspose(16, kernel_size=(2,2), strides=(2,2), activation = 'relu', name='tconv3') ) #[128,128,16]
        ls.append( Conv2D(1, (3,3), activation='sigmoid', padding='same', name='final_conv') ) #[128,128,1]

        return ls

    # @tf.function(jit_compile=True)
    def call(self, x):
        y=x
        for i in range(len(self.encoder_ls)):
            y = self.encoder_ls[i](y)

        for i in range(len(self.decoder_ls)):
            y = self.decoder_ls[i](y)

        return y

    # @tf.function(jit_compile=True)
    def Loss(self, y_pred, y_train):

        #-------------------------------------------------------------#
        #Total Loss
        train_loss = tf.reduce_mean(  tf.square(y_train - y_pred)  )
        #-------------------------------------------------------------#

        return([train_loss])

    # @tf.function(jit_compile=True)
    def encode(self, x):
    #x - [batch_size*nt, 128, 128, 1]
    #y - [batch_size*nt, latent_dim]
        y = x
        for i in range(len(self.encoder_ls)):
            y = self.encoder_ls[i](y)

        return y

    @tf.function(jit_compile=True)
    def decode(self, x):
        y=x
        for i in range(len(self.decoder_ls)):
            y = self.decoder_ls[i](y)

        return y[:,:,:,0]
