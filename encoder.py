#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:43:56 2018

@author: matthewszhang
"""
import tensorflow as tf

class Encoder():
    def __init__(self, params):
        self.params = params
        
        cell = params["encoder_cell"]
        self.cell = cell(params["n_features"])
        
    def encode(self, t_in):
        with tf.variable_scope("encoding", reuse = tf.AUTO_REUSE):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    self.cell, self.cell, t_in, dtype = tf.float32)
            forward, backward = outputs
            t_out = forward[:,-1,:] + backward[:,-1,:]
            reduced_t_out = t_out/tf.reshape(tf.reduce_max(t_out), (-1, 1))
        return reduced_t_out
        