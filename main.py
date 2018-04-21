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
FILE_PATH = ".\SIMULATION"

### GLOBAL PARAMETERS ###
TASKS, max_task_length, n_actions = get_all_tasks(FILE_PATH)
os.chdir(OLD_PATH)

iteration = 31

GRAPH_PATH = "tmp/build/iteration" + str(iteration)

PARAMS = {'alpha_subpolicy': 0.0001, 'alpha_superpolicy':0.0001, 'alpha_critic': 0.0001,
          'gamma':0.9, 'n_hidden_size':128, 'n_hidden_layers':3, 'actor_activation':tf.nn.relu,
          'critic_activation':tf.nn.tanh, 'max_plot':10000,
          'n_features':56, 'n_actions':n_actions, 'init_estimate':0.01, 'reward_threshold':2,
          'reward_memory':0.8, 'max_task_length': max_task_length, 'subpolicy_entropy': 0.01}

n_hidden_size = PARAMS["n_hidden_size"]
n_hidden_layers = PARAMS["n_hidden_layers"]
n_actions = PARAMS["n_actions"]

actor_widths = np.append(np.repeat(n_hidden_size, n_hidden_layers), n_actions)
critic_widths = np.append(np.repeat(n_hidden_size, n_hidden_layers), 1)
actor_activations = np.repeat(PARAMS["actor_activation"], n_hidden_layers + 1)
critic_activations = np.append(np.repeat(tf.nn.relu, n_hidden_layers), PARAMS["critic_activation"])

actor_layers = (actor_widths, actor_activations)
critic_layers = (critic_widths, critic_activations)

LAYERS = (actor_layers, critic_layers)

if __name__ == '__main__':
    tf.reset_default_graph()
    
    curriculum = Curriculum(PARAMS, TASKS, LAYERS, build_graph = GRAPH_PATH)
    curriculum.run()