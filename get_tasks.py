"""
Created by: Matthew Zhang
Date: 2018-03-24, 4:28PM

Learning Through Policy Sketches
"""

import os
# SPECIAL ENV FOR STR MANIPULATION
class Env():
    def __init__(self, task_dir):
        self.task_dir = task_dir
        env_path = os.path.join(task_dir, "env")
        os.chdir(env_path)
        with open("sim.txt", "r") as f:
            file = {}
            for line in f:
               items = line.split()
               file[items[0]] = list(map(int, items[1:]))
        
        from task import enc, dyn, rew
        self.enc_func = enc
        self.dyn_func = dyn
        self.rew_func = rew
        
        self.raw = file
        self.state = self.enc_func(self.raw)                
            
    def act(self, action):
        state = self.raw
        new_state = self.dyn_func(state, action)
        reward = self.rew_func(new_state, state, action)
        self.raw = new_state
        self.state = self.enc_func(self.raw)
        return self.state, reward
    
    def reset(self):
        env_path = os.path.join(self.task_dir, "env")
        os.chdir(env_path)
        with open("sim.txt", "r") as f:
            file = {}
            for line in f:
               items = line.split()
               file[items[0]] = list(map(int, items[1:]))
        self.raw = file
        self.state = self.enc_func(self.raw)

class Task():
    def __init__(self, task_dir):
        os.chdir(task_dir)
        with open("task.txt", "r") as f:
            lines = f.readlines()
            self.name = lines[0]
            self.length = int(lines[1])
            self.sketch = lines[2].split()
        self.environment = Env(task_dir)
  
def get_all_tasks(path):
    os.chdir(path)
    with open("params.txt", "r") as f:
        lines = f.readlines()
        max_task_length = int(lines[0])
        n_actions = int(lines[1])
    
    with open("all_tasks.txt", "r") as f:
        task_dirs = [task_dir for task_dir in f]
    
    TASKS = [[] for _ in range(max_task_length + 1)]
    for task_dir in task_dirs:
        task = Task(task_dir)
        TASKS[task.length].append(task)
    
    return TASKS, max_task_length, n_actions