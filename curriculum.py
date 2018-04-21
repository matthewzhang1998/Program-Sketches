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

from taskpolicy import Taskpolicy
from critic import Critic
from subpolicy import Subpolicy

GRAPH_PATH = "tmp/build/graph"

class Curriculum():
    def __init__(self, params, tasks, layers, build_graph = GRAPH_PATH):
        self.params = params
        self.curr_length = 1
        self.tasks = tasks
        
        actor_layers, critic_layers = layers
        
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        
        self.subpolicy = Subpolicy(params, self.actor_layers)
        self.critic = Critic(params, self.critic_layers)
        self.taskpolicy = Taskpolicy(self.subpolicy, self.critic, params)
        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if build_graph != None:
            self.writer = tf.summary.FileWriter(build_graph, self.session.graph)
            # constructor
        
    def _sample_length(self, print_rewards = True):
        length = self.curr_length
        tasks_curr_length = self.tasks[length]
        iterator = 0
        
        if print_rewards:
            x = []
            y = []
            avg_x = []
            avg_y = []
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
            reward = self.taskpolicy.run(self.session, self.writer, task)
            chi = self.params["reward_memory"]
            expected_rewards[index] = expected_rewards[index] * chi + reward * (1 - chi) 
            iterator += 1
            
            if print_rewards == True:
                x.append(iterator)
                y.append(reward)
                if iterator % 100 == 0:
                    now_time = time.time()
                    print("Iteration: {}, Avg Rew: {}, Time: {}".
                              format(iterator, sum(y)/len(y), now_time - start_time))
                    start_time = now_time
                    avg_x.append([iterator])
                    avg_y.append(sum(y)/len(y))
                    x = []
                    y = []

                if iterator == self.params["max_plot"]:
                    plt.scatter(avg_x,avg_y)
                    break
                          
    def run(self, print_rewards = True):
        while self.curr_length <= self.params["max_task_length"]:
            self._sample_length(print_rewards)
            self.curr_length += 1

        self.writer.close()
        self.session.close()
        return