"""
Created by: Matthew Zhang
Date: 2018-03-26, 1:49PM

Learning Through Policy Sketches

critic.py: contains definition of state evaluation module
"""

import tensorflow as tf
from helpers import ff

class Critic():
    def __init__(self, params, layers):
        self.params = params
        self.name = "critic"
        
        widths, activations = layers
        
        self.t_states = tf.placeholder(name = "states",
                dtype = tf.float32, shape = (None, params["n_features"]))
        self.t_mc_returns = tf.placeholder(name = "returns",
                dtype = tf.float32, shape = (None))
        self.alpha = tf.placeholder(dtype = tf.float32, shape = [], name = "alpha")
        
        t_baselines = ff(self.t_states, widths, activations, name = self.name)
        self.t_baselines = t_baselines
        self.critic_loss = 1/2 * tf.reduce_sum(tf.square(tf.stop_gradient(self.t_mc_returns) - t_baselines))
        
        self.mc_summary = tf.summary.histogram("MC_Returns", self.t_mc_returns)
        self.baseline_summary = tf.summary.histogram("Baselines", t_baselines)
        self.loss_summary = tf.summary.scalar("Critic_Loss", self.critic_loss)
        self.merged = tf.summary.merge([self.baseline_summary, self.mc_summary, self.loss_summary])

        optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_op = optimizer.minimize(self.critic_loss)
        
    def train(self, session, t_states, t_mc_returns):
        alpha = self.params["alpha_critic"]/len(t_states)
        feed = {self.t_states: t_states, self.t_mc_returns: t_mc_returns, self.alpha: alpha}
        summary, _ = session.run([self.merged, self.train_op], feed)
        return summary
        
    def evaluate(self, session, name, t_states, t_mc_returns):
        self.name = "critic" + name
        feed = {self.t_states: t_states, self.t_mc_returns: t_mc_returns}
        t_baselines = session.run([self.t_baselines], feed)
        summary = self.train(session, t_states, t_mc_returns)
        return t_baselines[0], summary