#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:10:27 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf
from helpers import ff

class ICM():
    def __init__(self, params, layers, encoder):
        self.params = params
        self.alpha = params["alpha_ICM"]
        
        self.forward_layers, self.backward_layers = layers
        self.forward_widths, self.forward_activations = self.forward_layers
        self.backward_widths, self.backward_activations = self.backward_layers
        
        self.t_states = tf.placeholder(dtype = tf.float32,
                                shape = (None, None, 1), name = "all_states")
        self.t_actions = tf.placeholder(dtype = tf.float32,
                                shape = (None, self.params["n_actions"]),
                                name = "all_actions")
        
        t_state_transitions = self._get_transitions(encoder.encode
                                                    (self.t_states))
        t_states_enc = encoder.encode(self.t_states)
        t_state_action_concat = tf.stop_gradient(tf.concat([t_states_enc[:-1,:],
                                           self.t_actions[:-1,:]], axis=-1))
                                                        
        t_action_scores = ff(t_state_transitions, self.backward_widths,
                             self.backward_activations, name = "backward_icm")
        t_backward_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits = t_action_scores, labels = self.t_actions[:-1])
        t_predictions = ff(t_state_action_concat, self.forward_widths, 
                          self.forward_activations, name = "forward_icm")
        t_forward_loss = 1/2 * tf.square(tf.norm(
                t_states_enc[1:] - t_predictions, axis = -1))
        
        eta = 1/self.params["n_features"]
        beta = self.params["beta_ICM"]
        lambd = self.params["lambda"]
        self.internal_reward = eta * t_forward_loss
        
        self.total_loss = 1/lambd * tf.reduce_sum((1-beta) * t_backward_loss + 
                                        beta * t_forward_loss)
        self.summary = tf.summary.scalar("ICM_Loss", self.total_loss)
        optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_op = optimizer.minimize(self.total_loss)
        
    def _get_transitions(self, t_states):
        return tf.concat([t_states[:-1], t_states[1:]], axis= -1)
        
    def _one_hot(self, t_in, depth):
        t_in = np.array(t_in)
        length = t_in.shape[0]
        t_out = np.zeros((length, depth))
        t_out[np.arange(length), t_in] = 1
        return t_out
    
    def run(self, sess, transitions):
        t_states, t_actions, _, _ = zip(*transitions)
        t_states = np.reshape(t_states, (len(t_states), -1, 1))
        t_actions = self._one_hot(t_actions, self.params["n_actions"])
        feed = {self.t_states: t_states, self.t_actions: t_actions}
        rewards = sess.run([self.internal_reward], feed)[0]
        
        intrinsic_reward = 0
        gamma = self.params["gamma"]
        for i, transition in enumerate(reversed(transitions)):
            if i != 0:    
                intrinsic_reward = intrinsic_reward * gamma + rewards[-i]
            transition[2] += intrinsic_reward
         
    def train(self, sess, transitions):
        t_states, t_actions, _, _, _ = zip(*transitions)
        t_states = np.reshape(t_states, (len(t_states), -1, 1))
        t_actions = self._one_hot(t_actions, self.params["n_actions"])
        feed = {self.t_states: t_states, self.t_actions: t_actions}
        summary, _ = sess.run([self.summary, self.train_op], feed)
        return summary