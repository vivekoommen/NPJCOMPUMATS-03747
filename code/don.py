# -*- coding: utf-8 -*-
# """
# Created on Sat Jun  5 21:47:39 2021

# @author: VIVEK OOMMEN
# """

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, BatchNormalization

class DeepONet_Model(tf.keras.Model):

    def __init__(self, Par):
        super(DeepONet_Model, self).__init__()
        np.random.seed(23)
        tf.random.set_seed(23)

        #Defining some model parameters
        self.latent_dim = 5
        self.m = 196  #########################################################################

        self.Par = Par

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []

        x = np.linspace(0+10**-4,1,80)[:,None]
        # y = 15*(1-x**0.2)+1
        # y = np.repeat(y, self.m, axis=1)
        # self.test_fn = tf.convert_to_tensor(y, dtype=tf.float32)


        self.lr=10**-4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.branch_net_ls = self.build_branch_net()
        self.trunk_net_ls  = self.build_trunk_net()

        self.alpha = tf.Variable(1, trainable=True)

    def build_branch_net(self):
        ls=[]

        ls.append( Conv2D(32, (3,3), name='conv1', input_shape=[14,14,self.Par['n_channels']]) ) #[12,12,32] #################################################
        ls.append( Activation(tf.math.sin)  )
        ls.append( BatchNormalization() )

        ls.append( Conv2D(16, (3,3), name='conv2') ) #[10,10,16]
        ls.append( Activation(tf.math.sin)  )
        ls.append( BatchNormalization() )

        ls.append( Conv2D(16, (3,3), name='conv3') ) #[8,8,16]
        ls.append( Activation(tf.math.sin)  )
        ls.append( BatchNormalization() )

        ls.append( Flatten() )
        ls.append(Dense(self.m*self.latent_dim))

        return ls

    def build_trunk_net(self):
        ls=[]

        ls.append( Dense(100))
        ls.append( Activation(tf.math.sin)  )

        ls.append( Dense(100))
        ls.append( Activation(tf.math.sin)  )

        ls.append(Dense(self.m*self.latent_dim))

        return ls

    # @tf.function(jit_compile=True)
    def call(self, X_func, X_loc):
    #X_func -> [BS*n_t, k*n_f]
    #X_loc  -> [n_t, 1]

        n_t = X_loc.shape[0]

        y_func = X_func
        y_func = (y_func - self.Par['mean'])/self.Par['std']



        for i in range(len(self.branch_net_ls)):
            y_func = self.branch_net_ls[i](y_func)

        y_loc = 10*(X_loc-0.5)
        for i in range(len(self.trunk_net_ls)):
            y_loc = self.trunk_net_ls[i](y_loc)

        y_func = tf.reshape(y_func, [-1, self.m, self.latent_dim])
        y_loc = tf.reshape(y_loc, [-1, self.m, self.latent_dim])

        Y = tf.einsum('ijk,pjk->ipj', y_func, y_loc)

        return(Y)

    # @tf.function(jit_compile=True)
    def Loss(self, y_pred, y_train):

        # mse = MeanSquaredError()

        #-------------------------------------------------------------#
        #Total Loss
        train_loss =  tf.reduce_mean( tf.square( y_pred - y_train ) )
        #mse(y_pred, y_train)
        #-------------------------------------------------------------#

        return([train_loss])

    # def predict(self, X_func, X_loc, n_steps):
    # #X_func -> [1,5*100]
    # #X_loc  -> [1,1]

    #     X_func_ls = [X_func]
    #     X_loc_ls = [X_loc[0]]

    #     for i in range(n_steps):
    #         y = self.call(X_func, X_loc ) #y -> [1,100 ]
    #         X_func = np.reshape(X_func, (1,5,-1))
    #         X_func = np.array([np.append(X_func[0], y, axis=0 )[1:]])
    #         X_func = np.reshape(X_func, (1,-1))
    #         X_loc = X_loc + 1

    #         X_func_ls.append(X_func)
    #         X_loc_ls.append(X_loc[0])

    #     X_func_ls = np.concatenate(X_func_ls, axis=0)
    #     X_loc_ls  = np.concatenate(X_loc_ls,  axis=0)

    #     return X_func_ls, X_loc_ls
