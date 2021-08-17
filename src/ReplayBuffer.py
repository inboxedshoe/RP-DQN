# -*- coding: utf-8 -*-
"""
Replay memory class and namedarraytuple
"""
import torch
from rlpyt.utils.collections import namedarraytuple
import random

Experience = namedarraytuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state','distancematrix')
)


class ReplayMemory():
    '''Taken from: https://deeplizard.com/learn/video/PyQNfsGUnQA'''
    def __init__(self, capacity,tsp_size, device=torch.device("cuda")):
        self.capacity = capacity
        self.statehis = torch.empty((0,tsp_size), device=device,dtype = torch.float32)
        self.actionhis = torch.empty((0), device=device,dtype = torch.float32)
        self.reward = torch.empty((0), device=device,dtype = torch.float32)
        self.next_state = torch.empty((0,tsp_size), device=device,dtype = torch.float32)
        self.dm = torch.empty((0), device=device,dtype = torch.float32)
        self.push_count = 0
        
    def push(self, experience):
        bsize = experience.reward.size(0)
        
        if self.reward.size(0) + bsize <= self.capacity:
            self.statehis = torch.cat((self.statehis, experience.state))
            self.actionhis = torch.cat((self.actionhis, experience.action))
            self.reward = torch.cat((self.reward, experience.reward))
            self.next_state = torch.cat((self.next_state, experience.next_state))
            self.dm = torch.cat((self.dm, experience.distancematrix))
            self.push_count += bsize
        else:
            l = int((self.push_count % self.capacity)/self.capacity * 10 *bsize)
            j = int((self.push_count % self.capacity)/self.capacity *10 * bsize + bsize)
            self.statehis.size()
            self.statehis[l: j] = experience.state
            self.actionhis[l: j] = experience.action
            self.reward[l: j]= experience.reward
            self.next_state[l: j]= experience.next_state
            self.dm[l: j] = experience.distancematrix
            self.push_count += bsize
        
    def sample(self, batch_size, tsp_size):
        indices = random.sample(range(self.reward.size(0)), batch_size)
        nexts = self.next_state[indices]
        non_final_ind = (nexts.sum(axis=1) != tsp_size).nonzero().view(-1)
        return self.statehis[indices], self.actionhis[indices].view(batch_size), self.reward[indices], nexts,non_final_ind, self.dm[indices].view(batch_size)
        
    def can_provide_sample(self, batch_size):
        return self.reward.size(0) >= batch_size