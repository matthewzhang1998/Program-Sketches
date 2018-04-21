"""
Created by: Matthew Zhang
Date: 2018-04-03, 11:59PM

Learning Through Policy Sketches NUM_SORT TASK
"""
import numpy as np
import copy

MAX_NUM = 10

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
    state_ptr = [state_dict["ptr"]]
    state_flag = state_dict["flag"]
    state_know = state_dict["know"]

    def one_hot_flatten(vector, one_hot_size):
        one_hot = np.zeros((len(vector), one_hot_size))
        one_hot[np.arange(len(vector)), vector] = 1
        return one_hot.flatten()

    flatten_state = one_hot_flatten(state, MAX_NUM)
    flatten_ptr = one_hot_flatten(state_ptr, len(state) - 1)    
    enc = np.concatenate((flatten_state, flatten_ptr, state_flag, state_know))
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
        if (new_dict["state"][new_dict["ptr"][0]] > 
            new_dict["state"][new_dict["ptr"][0] + 1]):
            new_dict["flag"][0] = 1
        
    # swap
    elif action == 3:
        new_dict["state"][new_dict["ptr"][0]], new_dict["state"][new_dict["ptr"][0] + 1] = (
                new_dict["state"][new_dict["ptr"][0] + 1], new_dict["state"][new_dict["ptr"][0]])
        new_dict["know"][0] = 0
        new_dict["flag"][0] = 0
    
    return new_dict
        
def rew(new_state, state, action):
    new_list = new_state["state"]
    old_list = state["state"]
    
    length = len(old_list)
    
    value = 0
    previous = 0
    for i in range(length):
        for j in range(i + 1, length):
            if new_list[i] < new_list[j]:
                value += 2/(length ** 2 - length)
            if old_list[i] < old_list[j]:
                previous += 2/(length ** 2 - length)
    return (value - previous)
    
    
    