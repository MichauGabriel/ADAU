# -*- coding: utf-8 -*-
"""
Title: 
------
Unsupervised transfer learning for anomaly detection: Application to complementary operating condition transfer
          (ADAU)
          
Description: 
--------------
TF function to reverse the gradient.
After github repository https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py

Please cite the corresponding paper if using it as part of ADAU:
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

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    with tf.Graph().as_default() as g:
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(X)
    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def call(self, x):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
