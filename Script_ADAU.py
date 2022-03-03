# -*- coding: utf-8 -*-
"""
Title: 
------
Unsupervised transfer learning for anomaly detection: Application to complementary operating condition transfer
          (ADAU)

Description: 
--------------
Toy script to showcase the deep neural network ADAU.

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
# designed with TF 2.2.0
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()

from lib import ADAU as ADAU


# verbose in fit function of TF
verbose = 2
# number of training epochs
epochs  = 2000

# Number of layers and neurons in the encoder (one element in list per layer, value is the number of neurons)
encoderStruct=[10,10]
# Number of layers and neurons in the discriminator (one element in list per layer, value is the number of neurons)
discrimStruct=[5,5]
# Weight applied to the reversed gradient of discriminator
wFlip = 0.1


#######################################################
# DEFINE Training, validation, testing sets
dataSrcHealthN  = np.random.rand(1000,50) # Data Source Healthy Normalised
dataTgtHealthN  = np.random.rand(1000,50) # Data Target Healthy Normalised
dataSrcFaultN   = np.random.rand(1000,50)
dataTgtFaultN   = np.random.rand(1000,50)

# Train with 5% of target
idTrain     = np.random.choice(np.arange(dataTgtHealthN.shape[0]),int(0.05*dataTgtHealthN.shape[0]),replace=False)  # => training indices for target healthy
# Train with 80% of source
idTrainSrc  = np.random.choice(np.arange(dataSrcHealthN.shape[0]),int(0.8*dataSrcHealthN.shape[0]),replace=False)  # => training indices for source healthy
# 10% of remaining index for validation
idValid     = np.setdiff1d(np.arange(dataTgtHealthN.shape[0]),idTrain)  # => validation indices for target healthy
idValid     = np.random.choice(idValid, int(0.1*dataTgtHealthN.shape[0]),replace=False)
idValidSrc  = np.setdiff1d(np.arange(dataSrcHealthN.shape[0]),idTrainSrc)  # => validation indices for source healthy
idValidSrc  = np.random.choice(idValidSrc, int(0.1*dataSrcHealthN.shape[0]),replace=False)
# Rest for test of TN
idTest      = np.setdiff1d(np.arange(dataTgtHealthN.shape[0]),np.concatenate((idTrain,idValid)))  # => testing indices for target healthy
idTestSrc   = np.setdiff1d(np.arange(dataSrcHealthN.shape[0]),np.concatenate((idTrainSrc,idValidSrc)))  # => testing indices for source healthy

####################################################
# Create Network
K.clear_session()
net,  mds_src, mds_tgt, eta_tgt = ADAU.create_ADAU(encoderStruct=encoderStruct, discrimStruct=discrimStruct,
                                                   inputSize = (dataTgtHealthN.shape[1],), wFlip=wFlip)


####################################################
# Define Losses
def mds_src_f(x,y):
    return mds_src
def mds_tgt_f(x,y):
    return mds_tgt
def etatgtm(x,y):
    return eta_tgt

def main_loss_src(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true,y_pred) + mds_src_f(y_true,y_pred)
def main_loss_tgt(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true,y_pred) + mds_tgt_f(y_true,y_pred)
addmetric = 'binary_crossentropy'

##########################################################
# Optimiser
opt= keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
#compile network
net.compile(optimizer=opt, loss=[main_loss_src,main_loss_tgt],metrics=[addmetric,etatgtm])

####################################################
# Example of training procedure
batch_size = np.min([idTrain.__len__(),idTrainSrc.__len__()])
batch_per_epoch = 1
dataGen = ADAU.dataGenerator(dataTgtHealthN[idTrain,:], dataSrcHealthN[idTrainSrc,:], batch_size=batch_size, batch_per_epoch = batch_per_epoch)
H = net.fit(dataGen, verbose = verbose, epochs=epochs)

# Get the ADAU features to be used in subsequent anomaly detection tasks
featTrainTgt   = net.get_layer('encoder').predict(dataTgtHealthN[idTrain,:])
featTrainSrc   = net.get_layer('encoder').predict(dataSrcHealthN[idTrainSrc,:])
featValidTgt   = net.get_layer('encoder').predict(dataTgtHealthN[idValid,:])
featValidSrc   = net.get_layer('encoder').predict(dataSrcHealthN[idValidSrc,:])
featFPTgt      = net.get_layer('encoder').predict(dataTgtHealthN[idTest,:])
featFPSrc      = net.get_layer('encoder').predict(dataSrcHealthN[idTestSrc,:])
featTPTgt      = net.get_layer('encoder').predict(dataTgtFaultN[:,:])
featTPSrc      = net.get_layer('encoder').predict(dataSrcFaultN[:,:])

# Use the feature in a one-class classifer:
#    + train with concatenate(featTrainTgt,featTrainSrc)
#    + set threshold with concatenate(featValidTgt,featValidSrc)
#    + Test if you detect featTPTgt but not featFPTgt
