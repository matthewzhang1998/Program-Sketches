"""
Created by: Matthew Zhang
Date: 2018-03-26, 6:59PM

Learning Through Policy Sketches

curriculum.py: 
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import errno

from icm import ICM
from taskpolicy import Taskpolicy
from critic import Critic
from subpolicy import Subpolicy
from encoder import Encoder

GRAPH_PATH = "tmp/build/graph"

class Curriculum():
    def __init__(self, params, tasks, layers, icm_layers, build_graph = GRAPH_PATH):
        self.params = params
        self.curr_length = 1
        self.tasks = tasks
        
        actor_layers, critic_layers = layers
        
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.icm_layers = icm_layers
        
        self.encoder = Encoder(params)
        self.icm = ICM(params, self.icm_layers, self.encoder)
        self.subpolicy = Subpolicy(params, self.actor_layers, self.encoder)
        self.critic = Critic(params, self.critic_layers, self.encoder)
        self.taskpolicy = Taskpolicy(self.subpolicy, self.critic, self.icm, params)
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()
        if build_graph != None:
            self.writer = tf.summary.FileWriter(build_graph, self.session.graph)
            
            filename = 'PARAMS/' + str(self.params['iteration']) +'.txt'
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(filename, 'w') as file:
                for kw in self.params:
                    arg = self.params[kw]
                    if type(arg) == "function":
                        file.write(str(kw) + ' ' + arg.__name__)
                    else:
                        file.write(str(kw) + ' ' +  str(arg))
                    file.write("\n")
            # constructor
        
    def _sample_length(self, print_rewards = True):
        length = self.curr_length
        tasks_curr_length = self.tasks[length]
        iterator = 0
        
        curiosity = self.params["curiosity"]
        visitation = np.zeros((8, 8))
        
        if print_rewards:
            plt.ion()
            start_time = time.time()
        
        if len(tasks_curr_length) == 0:
            return
        
        def softmax_choice(x):
            x = 1 - np.array(x)
            task_probs = np.exp(x - np.max(x))
            task_probs = task_probs/task_probs.sum()
            return np.random.choice(len(task_probs), p = task_probs)
            
        expected_rewards = [self.params["init_estimate"] for _ in tasks_curr_length]
        
        while min(expected_rewards) < self.params["reward_threshold"]:
            
            index = softmax_choice(expected_rewards)
            task = tasks_curr_length[index]
            
            reward, visitation = self.taskpolicy.run(self.session, self.writer,
                                         task, curiosity = curiosity,
                                         log_vis = True, visitation = visitation)
            chi = self.params["reward_memory"]
            expected_rewards[index] = expected_rewards[index] * chi + reward * (1 - chi) 
            iterator += 1
            
            if print_rewards == True:
                now_time = time.time()
                print("Iteration: {}, Avg Rew: {}, Time: {}".
                              format(iterator, reward/self.params["rollout"],
                                     now_time - start_time))
                start_time = now_time

                if iterator == self.params["max_plot"]:
                    visitation = visitation/np.amax(visitation)
                    vis_fig = plt.gcf()
                    plt.imshow(visitation, cmap='hot', interpolation='nearest')
                    plt.show()
                    vis_fig.savefig('visitation {} {}.png'.format(self.params["iteration"],
                                    curiosity),
                                    dpi = 100)
                    break
                          
    def run(self, print_rewards = True):
        while self.curr_length <= self.params["max_task_length"]:
            self._sample_length(print_rewards)
            self.curr_length += 1

        self.writer.close()
        self.session.close()
        return