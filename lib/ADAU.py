# -*- coding: utf-8 -*-
"""
Title: 
------
Unsupervised transfer learning for anomaly detection: Application to complementary operating condition transfer
          (ADAU)
          
Description: 
--------------
All functions required to create an ADAU network

Please cite the corresponding paper:
          Michau, G., & Fink, O. (2021).
          Unsupervised transfer learning for anomaly detection: Application to complementary operating condition transfer. 
          Knowledge-Based Systems, 216, 106816.
          https://doi.org/10.1016/j.knosys.2021.106816
          
Version: 
--------
1.0

@author:  
--------  
Dr. Gabriel Michau,
Chair of Intelligent Maintenance Systems,
ETH Zürich
          
Created on 03.03.2022

Licence:
----------
MIT License
Copyright (c) 2022 Dr. Gabriel Michau
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from lib import flipGradientTFK

def pairwise_euclidean_distance(X, Y):
    """
    Function to compute all pairwise distances between all elements in X and Y

    Parameters
    ----------
    X : Input array of size n * d
    Y : Input array of size 

    Returns
    -------
    n * m matrix with all euclidean distances computed on the d dimensions

    """
    xy = tf.matmul(X, tf.transpose(Y))
    xx = tf.reduce_sum(tf.square(X), -1)
    yy = tf.reduce_sum(tf.square(Y), -1)
    ds = (tf.expand_dims(xx, 1) + tf.expand_dims(yy, 0) - 2 * xy)
    safe_exp = tf.sqrt(tf.where(ds > 0, ds, tf.ones_like(ds)))
    return tf.where(ds > 0, safe_exp, tf.zeros_like(ds))

def MDS_loss(X, XT, etaFixed=None):
    """
    Function to compute the multi-dimensional scaling Loss.

    Parameters
    ----------
    X : Source data: n samples in a d1-diemensional space (n*d1)
    XT : Transformed source data: n samples in a d2-dimensional sapce (n*d2)
    etaFixed : if None, the optimal eta is computed (that minimises the MDS Loss), if float use it as fixed value
        The default is None.

    Returns
    -------
    MDS Loss: TF variable with one value
    eta : Value of eta (mostly usefull if etaFixed is None)
    
    """
    
    dX  = pairwise_euclidean_distance(X, X)
    dX  = tf.linalg.set_diag(dX, tf.zeros(tf.shape(dX)[0], tf.float32))
    dXT = pairwise_euclidean_distance(XT, XT)
    dXT  = tf.linalg.set_diag(dXT, tf.zeros(tf.shape(dXT)[0], tf.float32))
    if not etaFixed:
        eta = (tf.reduce_sum(tf.multiply(dX,dXT)))/\
              (tf.reduce_sum(tf.square(dXT))+1e-8)
    else:
        eta    = tf.constant(eta)
    return tf.reduce_mean(tf.square(dX-eta*dXT)), eta


def create_encoder(inputSize , encoderStruct=[10,5], name='encoder'):
    """
    Create the encoder network of ADAU (N1 in paper)

    Parameters
    ----------
    inputSize : tuple
        Size of input data
    encoderStruct : List
        length of list is the number of layers, value of elements are the number of neurons
        The default is [10,5].
    name : STR
        name of the resulting TF model
        The default is 'encoder'.

    Returns
    -------
    TF model
    """
    input_layer = layers.Input(shape=inputSize, name='input')
    x = input_layer
    for i in range(len(encoderStruct)):
        x = layers.Dense(encoderStruct[i],activation='relu',name='d{}'.format(i))(x)
    return Model(input_layer,x,name=name)

def create_discriminator(latentSize,discrimStruct=[100,100], activation = 'sigmoid', name='discriminator'):
    """
    Create the discriminator for ADAU (N2 in paper)

    Parameters
    ----------
    latentSize : tuple
        Size of latent space.
    discrimStruct : List
        length of list is the number of layers, value of elements are the number of neurons.
        The default is [100,100].
    activation : string
        Activation function to use in last layer.
        The default is 'sigmoid'.
   name : STR
        name of the resulting TF model.
        The default is 'discriminator'.

    Returns
    -------
    TF model

    """
    input_layer = layers.Input(shape=latentSize, name='input')
    x = input_layer
    for i in range(len(discrimStruct)):
        x = layers.Dense(discrimStruct[i],activation='relu',name='fc{}'.format(i))(x)
    x = layers.Dense(1,activation=activation,name='unit')(x)
    return Model(input_layer,x,name=name)

def create_flip(latentSize = (5,), wFlip=1.0):
    input_layer = layers.Input(shape=latentSize, name='input')
    x = flipGradientTFK.GradientReversal(wFlip)(input_layer)
    return Model(input_layer, x, name='flip')

def create_ADAU(encoderStruct=[10,5], discrimStruct=[100,100], inputSize = (14,), wFlip=1.0):
    """
    Main function to create the full ADAU network (encoder + discriminator with MDS Loss)

    Parameters
    ----------
    encoderStruct : List
        length of list is the number of layers, value of elements are the number of neurons
        The default is [10,5].
    discrimStruct : List
        length of list is the number of layers, value of elements are the number of neurons.
        The default is [100,100].
    inputSize : tuple
        Size of input data.
        The default is (14,).
    wFlip : float
        Weight applied to the reversed gradient of discriminator.
        The default is 1.0.

    Returns
    -------
    model : TF model corresponding to ADAU
    
    mds_src : TF tensor of size 1 with mds loss for source data

    mds_tgt : TF tensor of size 1 with mds loss for target data
    
    eta_tgt : Value of eta minismising mdsl target loss.

    """
    ##########################################
    # Generate Network
    encoder    =  create_encoder(inputSize, encoderStruct, name='encoder')
    latentSize =  encoder.layers[-1].output_shape
    flip = create_flip(latentSize[1:],wFlip)


    ##########################################
    # Link Variables
    src      = layers.Input(shape=inputSize, name='input_src')
    tgt      = layers.Input(shape=inputSize, name='input_tgt')

    feat_src = encoder(src)
    feat_tgt = encoder(tgt)
    disc = create_discriminator(latentSize[1:], discrimStruct, activation = 'sigmoid', name='discriminator')
    lab_src = disc(flip(feat_src))
    lab_tgt = disc(flip(feat_tgt))

    mds_src, _ = MDS_loss(src, feat_src, etaFixed=1.)
    mds_tgt, eta_tgt = MDS_loss(tgt, feat_tgt, etaFixed=None)
    
    model = Model([src, tgt],[lab_src, lab_tgt])
    return model, mds_src, mds_tgt, eta_tgt


class dataGenerator(keras.utils.Sequence):
    """ generator to train ADAU"""

    def __init__(self, data_src, data_tgt, batch_size=200, batch_per_epoch = 200):
        'Initialization'
        self.data_src        = data_src
        self.data_tgt        = data_tgt
        self.batch_size      = batch_size
        self.batch_per_epoch = batch_per_epoch
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.batch_per_epoch)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexbatch = slice(index*self.batch_size,(index+1)*self.batch_size)
        # Generate data

        Y_src = np.ones((int(self.batch_size)))
        Y_tgt   = np.zeros((int(self.batch_size)))
        return [self.data_src[self.indexes_src[indexbatch],:],
                self.data_tgt[self.indexes_tgt[indexbatch],:]] , [Y_src,Y_tgt]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes_src = np.random.choice(np.arange(self.data_src.shape[0]), size = int(self.batch_per_epoch*self.batch_size), replace = False)
        self.indexes_tgt = np.random.choice(np.arange(self.data_tgt.shape[0]), size = int(self.batch_per_epoch*self.batch_size), replace = False)