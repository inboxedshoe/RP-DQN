# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:17:19 2020

@author: kirill
"""

from rlpyt.utils.collections import namedarraytuple
import torch
import random

Experience = namedarraytuple(
    'Experience',
    ('states', 'actions', 'rewards', 'next_states')
)

States = namedarraytuple(
    'States',
    ('graphs', 'last_visited', 'masks', 'Ds', 'current_depots')
)

class ReplayMemory():
    '''Taken from: https://deeplizard.com/learn/video/PyQNfsGUnQA'''
    
    def __init__(self, capacity ,problem_size, device=torch.device("cuda")):
        self.device=device
        self.capacity = capacity
        self.actions = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.rewards = torch.empty((capacity,1), device=device,dtype = torch.float32)
        self.push_count = 0
        
        #current state
        self.cStates = {}
        self.cStates["graphs"] = torch.empty((capacity,problem_size,3), device=device,dtype = torch.float32)
        self.cStates["last_visited"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.cStates["mask"] = torch.empty((capacity,1,problem_size), device=device,dtype = torch.bool)
        self.cStates["D"] = torch.empty((capacity,1), device=device,dtype = torch.float32)
        self.cStates["current_depots"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
      
        
        #next State
        self.nStates = {}
        self.nStates["graphs"] = torch.empty((capacity,problem_size,3), device=device,dtype = torch.float32)
        self.nStates["last_visited"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.nStates["mask"] = torch.empty((capacity,1,problem_size), device=device,dtype = torch.bool)
        self.nStates["D"] = torch.empty((capacity,1), device=device,dtype = torch.float32)
        self.nStates["current_depots"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        
    def push(self, experience):
        bsize = experience.rewards.size(0)
        
        #compute indices to push in
        l = self.push_count % self.capacity
        j = l+bsize
        
        #compute how much experiences fit into buffer and push them 
        split_size = min(bsize, self.capacity - l)
        self.cStates["graphs"][l:j] = experience.states.graphs[:split_size]
        self.cStates["last_visited"][l:j] = experience.states.last_visited[:split_size]
        self.cStates["mask"][l:j] = experience.states.masks[:split_size]
        self.cStates["D"][l:j] = experience.states.Ds[:split_size]
        self.cStates["current_depots"][l:j] = experience.states.current_depots[:split_size]
        
        self.nStates["graphs"][l:j] = experience.next_states.graphs[:split_size]
        self.nStates["last_visited"][l:j] = experience.next_states.last_visited[:split_size]
        self.nStates["mask"][l:j] = experience.next_states.masks[:split_size]
        self.nStates["D"][l:j] = experience.next_states.Ds[:split_size]
        self.nStates["current_depots"][l:j] = experience.next_states.current_depots[:split_size]
        
        self.actions[l:j] = experience.actions[:split_size]
        self.rewards[l:j] = experience.rewards[:split_size]

        if split_size < bsize:
            #push the rest the the beginning of buffer
            diff = bsize - split_size
            self.cStates["graphs"][:diff] = experience.states.graphs[split_size:]
            self.cStates["last_visited"][:diff] = experience.states.last_visited[split_size:]
            self.cStates["mask"][:diff] = experience.states.masks[split_size:]
            self.cStates["D"][:diff] = experience.states.Ds[split_size:]
            self.cStates["current_depots"][:diff] = experience.states.current_depots[split_size:]
            
            self.nStates["graphs"][:diff] = experience.next_states.graphs[split_size:]
            self.nStates["last_visited"][:diff] = experience.next_states.last_visited[split_size:]
            self.nStates["mask"][:diff] = experience.next_states.masks[split_size:]
            self.nStates["D"][:diff] = experience.next_states.Ds[split_size:]
            self.nStates["current_depots"][:diff] = experience.next_states.current_depots[split_size:]
            
            self.actions[:diff] = experience.actions[split_size:]
            self.rewards[:diff] = experience.rewards[split_size:]
            
        self.push_count += bsize
       
    def sample(self, batch_size):
        indices = random.sample(range(min(self.push_count,self.capacity)), batch_size)
        cstates = States(self.cStates["graphs"][indices],self.cStates["last_visited"][indices],self.cStates["mask"][indices],self.cStates["D"][indices],self.cStates["current_depots"][indices])
        nstates = States(self.nStates["graphs"][indices],self.nStates["last_visited"][indices],self.nStates["mask"][indices],self.nStates["D"][indices],self.nStates["current_depots"][indices])
    
        return cstates, self.actions[indices].view(batch_size), self.rewards[indices].view(batch_size), nstates
        
    def can_provide_sample(self,batch_size):
        return batch_size <= self.push_count
    
    def get_fill_percentage(self):
        cap = self.buffer.push_count / self.buffer.capacity
        if cap > 1:
            cap = 1
        return cap