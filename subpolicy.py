"""
Created by: Matthew Zhang
Date: 2018-03-24, 2:21PM

Learning Through Policy Sketches
"""

import tensorflow as tf
import numpy as np
from helpers import ff

class Subpolicy():
    def __init__(self, params, layers):
        '''
        initialize method for modular subpolicy class
        PARAMS
        -------------
        params: dict
            contains all hyperparameters for operation
            (to-do: replace using tf.flags)
        layers: tuple
            unpacks into widths (int) and activations (tf.nn.train functions)
        '''
        self.params = params
        self.name = "actor"
        
        widths, activations = layers
        
        # Feed-In Variables
        self.t_returns = tf.placeholder(dtype = tf.float32, shape = (None),
                                        name = "returns")
        self.t_baselines = tf.placeholder(dtype = tf.float32, shape = (None),
                                          name = "baselines")
        self.t_actions = tf.placeholder(dtype = tf.int32, 
                                        shape = (None), name = "actions")
        self.t_states = tf.placeholder(dtype = tf.float32, 
                        shape = (None, self.params["n_features"]), name = "states")
        self.alpha = tf.placeholder(dtype = tf.float32, shape = [], name = "alpha")
        
        # Obtaining Action Probs by FF network
        t_action_scores = ff(self.t_states, widths, activations, name = self.name)
        t_action_probs = tf.nn.softmax(t_action_scores)
        t_action_logprobs = tf.nn.log_softmax(t_action_scores)
        # Additional terms necessary in loss
        t_chosen_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=t_action_scores, labels=self.t_actions)
        t_entropy = -tf.reduce_mean(tf.reduce_sum(t_action_probs * t_action_logprobs))
        t_adjusted_rew = -tf.reduce_mean(t_chosen_logprobs * (self.t_returns - self.t_baselines))
        
        # Defining loss terms
        self.t_subpolicy_loss = t_adjusted_rew + params["subpolicy_entropy"] * t_entropy
        subpolicy_loss = tf.summary.scalar("Subpolicy_Loss", self.t_subpolicy_loss)
        adjusted_reward = tf.summary.scalar("Adjusted_Reward", t_adjusted_rew)
        self.merged = tf.summary.merge([subpolicy_loss, adjusted_reward])
        self.t_action_probs = t_action_probs
        
        # Optimizer with Gradient Clipping
        optimizer = tf.train.AdamOptimizer(self.alpha)
        self.train_op = optimizer.minimize(self.t_subpolicy_loss)
            
    def _get_action(self, session, state):
        '''
        obtains action based on probability outputs from FF network
        
        PARAMS
        -------------
        state: np.array (size = ((batch_size), n_features))
            contains flattened state features as input into graph
        
        RETURNS
        -------------
        int
            index of action to be chosen 
        '''
        state = np.reshape(state, (-1, self.params["n_features"]))
        feed = {self.t_states: state}
        action_probs = session.run(self.t_action_probs, feed)
                     
        action = np.random.choice(len(action_probs[0]), p = action_probs[0])
        return action
        
    def train(self, session, transitions, sketch):
        '''
        trains network using RL based on the given transitions
        
        PARAMS
        -------------
        transitions: tuple
            unpacks into current_states (np.array, (size = ((batch_size),
            n_features))), returns (int) and baseline (int)
        '''
        # Setting up learning rate to depend on transitions
        alpha = self.params["alpha_subpolicy"]/len(transitions)

        self.name = "actor" + sketch
        for curr_state, action, returns, baseline, subname in transitions:
            if subname == self.name:
                transitions = [(curr_state, action, returns, baseline)]
        
        t_states, t_actions, t_returns, t_baselines = zip(*transitions)
        feed = {self.t_returns: t_returns, self.t_states: t_states,
                self.t_actions: t_actions, self.t_baselines: t_baselines,
                self.alpha: alpha}
        summary, _ = session.run([self.merged, self.train_op], feed)
        return summary
        
    def semi_rollout(self, session, subpolicy_name, environment, temp_transitions):
        '''
        rolls out policy until stop action is selected
        
        PARAMS
        -------------
        subpolicy_name: string
            gives scope of the subpolicy for FF net
        environment: Environment class
            contains simulation details (state, act)
        
        
        RETURNS
        -------------
        float
            total reward over the subpolicy
            
        '''
        # Setting up important variables
        self.name = "actor" + subpolicy_name
        gamma = self.params["gamma"]
        curr_state = environment.state
        total_reward = 0
        
        while(1):
            # Transition
            action = self._get_action(session, curr_state)
            next_state, reward = environment.act(action)
            total_reward = total_reward + reward
            
            # Apply MC reward
            temp_transitions.append([curr_state, action, reward, self.name])
            for i, transition in enumerate(reversed(temp_transitions)):
                transition[2] += (gamma ** i) * reward
            curr_state = next_state
            
            # End if finished
            if action == (self.params['n_actions'] - 1):
                break
              
        return total_reward