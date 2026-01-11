from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 
    
    def reset(self):
        self.buffer.clear()

    def add(self, state, action, logits, reward, values, tbd_node_idxs): 
        self.buffer.append((state, action, logits, reward, values,tbd_node_idxs)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, logits, reward, values, tbd_node_idxs = zip(*transitions)
        return state, action, logits, reward, values, tbd_node_idxs 

    def size(self): 
        return len(self.buffer)


import torch as t
def angle(o, a):
    delta_a = a - o
    if (delta_a[0]>=0) and (delta_a[1]>=0):
        atanx = t.atanx(delta_a)
    elif (delta_a[0]<0) and (delta_a[1]>0):
        atanx = 3.1415926 - t.atanx(delta_a)
    elif (delta_a[0]<0) and (delta_a[1]<0):
        atanx = 3.1415926 + t.atanx(delta_a)
    elif (delta_a[0]>0) and (delta_a[1]<0):
        atanx = 2*3.1415926 - t.atanx(delta_a)
