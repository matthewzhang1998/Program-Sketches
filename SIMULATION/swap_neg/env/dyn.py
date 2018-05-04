# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:04:32 2018

@author: Matthew
"""
import copy

def dyn(state_dict, action):
    new_dict = copy.deepcopy(state_dict)
    
    # left
    if action == 0:
        if new_dict["ptr"][0] > 0:
            new_dict["ptr"][0] -= 1
            new_dict["know"][0] = 0
            new_dict["flag"][0] = 0
        
    # right
    elif action == 1:
        if new_dict["ptr"][0] < len(new_dict["state"]) - 2:
            new_dict["ptr"][0] += 1
            new_dict["know"][0] = 0
            new_dict["flag"][0] = 0
        
    # compare
    elif action == 2:
        new_dict["flag"][0] = 0
        new_dict["know"][0] = 1
        if len(new_dict["stack"]) > 0:
            if (new_dict["state"][new_dict["ptr"][0]] > 
                new_dict["stack"][-1]):
                new_dict["flag"][0] = 1
        
    # push/load memory value
    elif action == 3:
        new_dict["stack"].append(new_dict["gpr"][0])
        
    # pop/store memory value
    elif action == 4:
        if len(new_dict["stack"]) > 0:
            new_dict["gpr"][0] = new_dict["stack"].pop()
            
    # push pointer value
    elif action == 5:
        new_dict["ptr_stack"].append(new_dict["ptr"][0])
    
    # pop pointer value
    elif action == 6:
        if len(new_dict["ptr_stack"]) > 0:
            new_dict["ptr"][0] = new_dict["ptr_stack"].pop()
    
    # register copy
    elif action == 7:
        new_dict["gpr"][0] = new_dict["state"][new_dict["ptr"][0]]
    
    # register write
    elif action == 8:
        new_dict["state"][new_dict["ptr"][0]] = new_dict["gpr"][0]
    
    return new_dict
        