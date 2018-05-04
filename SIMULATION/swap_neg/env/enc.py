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
    state_gpr = state_dict["gpr"]

    def one_hot_flatten(vector, one_hot_size):
        one_hot = np.zeros((len(vector), one_hot_size))
        one_hot[np.arange(len(vector)), vector] = 1
        return one_hot.flatten()

    flatten_state = one_hot_flatten(state, MAX_NUM)
    flatten_gpr = one_hot_flatten(state_gpr, MAX_NUM)
    flatten_ptr = one_hot_flatten(state_ptr, len(state) - 1)    
    enc = np.concatenate((flatten_state, flatten_ptr, flatten_gpr, state_flag, state_know))
    return enc


    