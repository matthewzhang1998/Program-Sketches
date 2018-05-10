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
        
        self.t_in = tf.placeholder(tf.float32, shape = (None, None, 1))
        self.t_out = self.encode(self.t_in)
        
    def encode(self, t_in):
        with tf.variable_scope("encoding", reuse = tf.AUTO_REUSE):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    self.cell, self.cell, t_in, dtype = tf.float32)
            forward, backward = outputs
            
            t_out = tf.reduce_mean((forward+backward), axis = 1)
        return t_out
    
    def test(self):
        t_in = [[[0],[1],[1],[1],[1],[1]]]
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        with self.session as sess:
            t_out = sess.run([self.t_out], feed_dict = {self.t_in:t_in})
            print(t_out)
        
if __name__ == "__main__":
    PARAMS = {'encoder_cell': tf.contrib.rnn.LSTMCell,'n_features':32}
    encoder = Encoder(PARAMS)
    encoder.test()
    