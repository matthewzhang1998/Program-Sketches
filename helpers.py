#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 12:33:21 2018

@author: matthewszhang
"""
import tensorflow as tf

INIT_SCALE = 1.43

def _linear(t_in, n_out):
    v_w = tf.get_variable(
            "w",
            shape=(t_in.get_shape()[-1], n_out),
            initializer=tf.uniform_unit_scaling_initializer(
                factor=INIT_SCALE))
    v_b = tf.get_variable(
            "b",
            shape=n_out,
            initializer=tf.constant_initializer(0))
    if len(t_in.get_shape()) == 2:
        return tf.einsum("ij,jk->ik", t_in, v_w) + v_b
    elif len(t_in.get_shape()) == 3:
        return tf.einsum("ijk,kl->ijl", t_in, v_w) + v_b
    else:
        assert False
        
def ff(t_in, widths, activations, name = None):
    assert len(t_in.get_shape()) in (2, 3)
    assert len(widths) == len(activations)
    prev_layer = t_in
    
    scope_base = ""
    if name != None:
        scope_base = name
    
    for i_layer, (width, act) in enumerate(zip(widths, activations)):
        with tf.variable_scope(scope_base + str(i_layer)):
            layer = _linear(prev_layer, width)
            if act is not None:
                layer = act(layer)
        prev_layer = layer
    return prev_layer