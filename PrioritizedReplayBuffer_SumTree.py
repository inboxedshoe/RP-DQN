# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:17:19 2020

@author: kirill
"""

from rlpyt.utils.collections import namedarraytuple
import torch
import random
import SumTree

Experience = namedarraytuple(
    'Experience',
    ('states', 'actions', 'rewards', 'next_states')
)

States = namedarraytuple(
    'States',
    ('graphs', 'last_visited', 'masks', 'Ds', 'current_depots', 'uncompleted', 'i')
)

class ReplayMemory():
    '''Taken from: https://deeplizard.com/learn/video/PyQNfsGUnQA'''
    
    def __init__(self, capacity ,problem_size, alpha = 0.6, beta=0.4, num_features = 5, device=torch.device("cuda")):
        self.device=device
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.actions = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.rewards = torch.empty((capacity,1), device=device,dtype = torch.float32)
        self.push_count = 0
        self.last_indices = None
        
        #current state
        self.cStates = {}
        self.cStates["graphs"] = torch.empty((capacity,problem_size,num_features), device=device,dtype = torch.float32)
        self.cStates["last_visited"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.cStates["mask"] = torch.empty((capacity,1,problem_size), device=device,dtype = torch.bool)
        self.cStates["D"] = torch.empty((capacity,1), device=device,dtype = torch.float32)
        self.cStates["current_depots"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.cStates["i"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.cStates["uncompleted"] = torch.empty((capacity), device=device,dtype = torch.bool)
      
        
        #next State
        self.nStates = {}
        self.nStates["graphs"] = torch.empty((capacity,problem_size,num_features), device=device,dtype = torch.float32)
        self.nStates["last_visited"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.nStates["mask"] = torch.empty((capacity,1,problem_size), device=device,dtype = torch.bool)
        self.nStates["D"] = torch.empty((capacity,1), device=device,dtype = torch.float32)
        self.nStates["current_depots"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.nStates["i"] = torch.empty((capacity,1), device=device,dtype = torch.int64)
        self.nStates["uncompleted"] = torch.empty((capacity), device=device,dtype = torch.bool)
        
        self.tree = SumTree.SumTree_tn(capacity, device=device) # device=device for GPU
        
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
        self.cStates["i"][l:j] = experience.states.i[:split_size]
        self.cStates["uncompleted"][l:j] = experience.states.uncompleted[:split_size]
        
        self.nStates["graphs"][l:j] = experience.next_states.graphs[:split_size]
        self.nStates["last_visited"][l:j] = experience.next_states.last_visited[:split_size]
        self.nStates["mask"][l:j] = experience.next_states.masks[:split_size]
        self.nStates["D"][l:j] = experience.next_states.Ds[:split_size]
        self.nStates["current_depots"][l:j] = experience.next_states.current_depots[:split_size]
        self.nStates["i"][l:j] = experience.next_states.i[:split_size]
        self.nStates["uncompleted"][l:j] = experience.next_states.uncompleted[:split_size]
        
        self.actions[l:j] = experience.actions[:split_size]
        self.rewards[l:j] = experience.rewards[:split_size]
        
        # Update priorities with default priority of 1
        self.tree.update(torch.arange(start=l,end=l+split_size,device=self.device),torch.ones(split_size,device=self.device))

        if split_size < bsize:
            #push the rest the the beginning of buffer
            diff = bsize - split_size
            self.cStates["graphs"][:diff] = experience.states.graphs[split_size:]
            self.cStates["last_visited"][:diff] = experience.states.last_visited[split_size:]
            self.cStates["mask"][:diff] = experience.states.masks[split_size:]
            self.cStates["D"][:diff] = experience.states.Ds[split_size:]
            self.cStates["current_depots"][:diff] = experience.states.current_depots[split_size:]
            self.cStates["i"][:diff] = experience.states.i[split_size:]
            self.cStates["uncompleted"][:diff] = experience.states.uncompleted[split_size:]
            
            self.nStates["graphs"][:diff] = experience.next_states.graphs[split_size:]
            self.nStates["last_visited"][:diff] = experience.next_states.last_visited[split_size:]
            self.nStates["mask"][:diff] = experience.next_states.masks[split_size:]
            self.nStates["D"][:diff] = experience.next_states.Ds[split_size:]
            self.nStates["current_depots"][:diff] = experience.next_states.current_depots[split_size:]
            self.nStates["i"][:diff] = experience.next_states.i[split_size:]
            self.nStates["uncompleted"][:diff] = experience.next_states.uncompleted[split_size:]
            
            self.actions[:diff] = experience.actions[split_size:]
            self.rewards[:diff] = experience.rewards[split_size:]
            
            # Update priorities with default priority of 1
            self.tree.update(torch.arange(end=diff,device=self.device),torch.ones(diff,device=self.device))
            
        self.push_count += bsize
        
    def update_prios(self,values):
        self.tree.update(self.last_indices,values.view(-1))
    
    def get_beta(self,episode):
        return self.beta[0] + ((1-self.beta[0])/self.beta[1])*episode
        
       
    def sample(self, batch_size,episode):
        indices = self.tree.sample(batch_size)#random.sample(range(min(self.push_count,self.capacity)), batch_size)
        self.last_indices = indices
        cstates = States(self.cStates["graphs"][indices],self.cStates["last_visited"][indices],self.cStates["mask"][indices],self.cStates["D"][indices], self.cStates["current_depots"][indices], self.cStates["uncompleted"][indices], self.cStates["i"][indices])
        nstates = States(self.nStates["graphs"][indices],self.nStates["last_visited"][indices],self.nStates["mask"][indices],self.nStates["D"][indices], self.nStates["current_depots"][indices], self.nStates["uncompleted"][indices], self.nStates["i"][indices])
        
        weights = torch.pow((1/min(self.push_count,self.capacity) * (1/(self.tree.leaves[indices]/self.tree.parents[-1].item()))),self.get_beta(episode))
        weights = weights/weights.max()
        
        return cstates, self.actions[indices].view(batch_size), self.rewards[indices].view(batch_size), nstates, weights.cuda() if torch.cuda.is_available() else weights
        
    def can_provide_sample(self,batch_size):
        return batch_size <= self.push_count
    
    def get_fill_percentage(self):
        cap = self.push_count / self.capacity
        if cap > 1:
            cap = 1
        return cap