"""
Created by: Matthew Zhang
Date: 2018-03-24, 1:00PM

Learning Through Policy Sketches

main.py: file containing executable code and parameter initialization
"""
import tensorflow as tf
import numpy as np
from curriculum import Curriculum
from get_tasks import get_all_tasks
import os

OLD_PATH = os.getcwd()
FILE_PATH = "./SIMULATION"

### GLOBAL PARAMETERS ###
TASKS, max_task_length, n_actions = get_all_tasks(FILE_PATH)
os.chdir(OLD_PATH)

iteration = 48

GRAPH_PATH = "tmp/build/iteration" + str(iteration)

PARAMS = {'iteration': iteration, 'alpha_subpolicy': 0.001,
          'alpha_superpolicy':0.0001, 'alpha_critic': 0.001, 'alpha_ICM':0.001,
          'gamma':0.9, 'n_hidden_size':64, 'n_hidden_layers':3, 
          'actor_activation':tf.nn.relu, 'encoder_cell': tf.contrib.rnn.LSTMCell,
          'critic_activation':tf.nn.sigmoid, 'max_plot':10000, 'beta_ICM':0.2,
          'n_ICM_size':64, 'n_ICM_layers':2,
          'forward_activation': tf.nn.sigmoid, 'backward_activation': tf.nn.sigmoid,
          'n_features':32, 'n_actions':n_actions, 'init_estimate':0.01, 
          'reward_threshold':2, 'lambda':0.5,
          'reward_memory':0.8, 'max_task_length': max_task_length, 
          'subpolicy_entropy': 0.01}

n_hidden_size = PARAMS["n_hidden_size"]
n_hidden_layers = PARAMS["n_hidden_layers"]
n_actions = PARAMS["n_actions"]

actor_widths = np.append(np.repeat(n_hidden_size, n_hidden_layers), n_actions)
critic_widths = np.append(np.repeat(n_hidden_size, n_hidden_layers), 1)
actor_activations = np.repeat(PARAMS["actor_activation"], n_hidden_layers + 1)
critic_activations = np.append(np.repeat(
        tf.nn.relu, n_hidden_layers), PARAMS["critic_activation"])

actor_layers = (actor_widths, actor_activations)
critic_layers = (critic_widths, critic_activations)

n_ICM_size = PARAMS["n_ICM_size"]
n_ICM_layers = PARAMS["n_ICM_layers"]
n_features = PARAMS["n_features"]

forward_widths = np.append(np.repeat(n_ICM_size, n_ICM_layers), n_features)
backward_widths = np.append(np.repeat(n_ICM_size, n_ICM_layers), n_actions)
forward_activations = np.repeat(PARAMS["forward_activation"], n_ICM_layers + 1)
backward_activations = np.repeat(PARAMS["backward_activation"], n_ICM_layers + 1)

forward_layers = (forward_widths, forward_activations)
backward_layers = (backward_widths, backward_activations)

LAYERS = (actor_layers, critic_layers)
ICM_LAYERS = (forward_layers, backward_layers)

if __name__ == '__main__':
    tf.reset_default_graph()
    
    curriculum = Curriculum(PARAMS, TASKS, LAYERS, ICM_LAYERS, 
                            build_graph = GRAPH_PATH)
    curriculum.run()