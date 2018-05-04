# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:04:33 2018

@author: Matthew
"""
import copy

def rew(new_state, state, init_state, action):
    new_list = new_state["state"]
    old_list = state["state"]
    init_list = copy.deepcopy(init_state["state"])

    reward = 0
    if action == 8 and new_list == [0,1]:
        reward = 1
        
    #print("neg")
    
    return reward