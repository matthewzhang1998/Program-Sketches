"""
Created by: Matthew Zhang
Date: 2018-04-03, 11:59PM

Learning Through Policy Sketches NUM_SORT TASK
"""
import numpy as np
import copy

MAX_NUM = 4

def enc(state_dict):
    '''
    encodes init_state with one hots
    
    PARAMS
    -------------
    init_state: list
        contains unsorted list of n numbers
    
    RETURNS
    -------------
    list
        contains encoded array with extra features
    '''
    state = state_dict["state"]
    state_ptr = state_dict["ptr"]
    state_flag = state_dict["flag"]
    state_know = state_dict["know"]
    state_gpr_1 = state_dict["gpr_1"]
    state_gpr_2 = state_dict["gpr_2"]

    def one_hot_flatten(vector, one_hot_size):
        one_hot = np.zeros((len(vector), one_hot_size))
        one_hot[np.arange(len(vector)), vector] = 1
        return one_hot.flatten()
    
    def overlay(vector):
        vector = np.reshape(vector, (1, vector.shape[0]))
        return vector

    flatten_state = one_hot_flatten(state, MAX_NUM)
    flatten_gpr = one_hot_flatten(state_gpr_1, MAX_NUM)
    flatten_gpr = one_hot_flatten(state_gpr_2, MAX_NUM)
    flatten_ptr = one_hot_flatten(state_ptr, len(state) - 1)    
    enc = np.concatenate((flatten_state, flatten_ptr, flatten_gpr, state_flag, state_know))
    enc = overlay(enc)
    return enc

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
        
#    # push/load memory value
#    elif action == 3:
#        new_dict["stack"].append(new_dict["gpr"][0])
#        
#    # pop/store memory value
#    elif action == 4:
#        if len(new_dict["stack"]) > 0:
#            new_dict["gpr"][0] = new_dict["stack"].pop()
#            
#    # push pointer value
#    elif action == 5:
#        new_dict["ptr_stack"].append(new_dict["ptr"][0])
#    
#    # pop pointer value
#    elif action == 6:
#        if len(new_dict["ptr_stack"]) > 0:
#            new_dict["ptr"][0] = new_dict["ptr_stack"].pop()
    
    # register copy
    elif action == 3:
        new_dict["gpr_1"][0] = new_dict["state"][new_dict["ptr"][0]]
    
    # register write
    elif action == 4:
        new_dict["state"][new_dict["ptr"][0]] = new_dict["gpr_1"][0]
    
    # register copy
    elif action == 5:
        new_dict["gpr_2"][0] = new_dict["state"][new_dict["ptr"][0]]
    
    # register write
    elif action == 6:
        new_dict["state"][new_dict["ptr"][0]] = new_dict["gpr_2"][0]
    
    return new_dict
        
def rew(new_state, state, action):
    new_list = new_state["state"]
    old_list = state["state"]
    
    reward = 0
    if action == 7 and new_list == [0,1]:
        reward = 1
                
    return reward
    