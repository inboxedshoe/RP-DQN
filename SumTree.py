# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:09:15 2020

@author: Simons PC
"""
import torch
import math 
import numpy as np

class SumTree_tn(object):
    def __init__(self, max_size,device=torch.device("cuda")):
        self.device=device
        self.max_size = max_size
        self.leaves = torch.zeros(max_size,device=device,dtype=torch.float64)
        self.levels = int(math.ceil(math.log(max_size, 2)+1))
        self.parents = [torch.zeros(math.ceil(max_size/(2**(i+1))),device=device,dtype=torch.float64) for i in range(self.levels-1)]

    def update(self,idx,value):
        value = value.double()
        change = value - self.leaves[idx]
        self.leaves[idx] = value
        idx,inverse_idx = idx.unique(return_inverse=True)
        final_change = torch.empty(len(idx),device=self.device,dtype=torch.float64)
        final_change[inverse_idx] = change
        for level in range(self.levels-1):
            idx_parent = (idx//2).long()
            self.parents[level].index_add_(0,idx_parent,final_change)
            idx = idx_parent
                   
    def search(self,value,level,idx):
        '''Recursive function to find idx from given node, value and tree_level (batch mode)'''
        # Get child indices (one level below) of nodes of current level
        childs = torch.cat((idx*2,idx*2+1),1).long()
        
        if level == 0:
            # Last layer -> check leaves
            left_vals = self.leaves[childs[:,0]] < value
            idx = childs.gather(1,left_vals.long().view(-1,1))
            return idx
        else:
            # Go one level lower
            # Check whether to go left or right (True-> go right, False -> go left)
#            print(self.parents[level-1].shape)
#            print(childs.shape)
            left_vals = self.parents[level-1][childs[:,0]] < value
            # Get index of child node to go to (next level indices)
            
            idx = childs.gather(1,left_vals.long().view(-1,1))
            # Decrement values for samples that went RIGHT by the left child value
            
            value[left_vals] = value[left_vals] - self.parents[level-1][childs[left_vals][:,0]]
        
            return self.search(value,level-1,idx)
        
    def sample(self,batchsize):
        '''Traverse the whole tree to find indices to sample given a batch-size'''
        # TODO split into *batchsize* ranges, sample one element per range uniform, then traverse
#        value = torch.tensor(np.random.uniform(high=self.parents[-1].item(),size=batchsize),device=self.device)
        if torch.cuda.is_available():
            value = torch.cuda.DoubleTensor(batchsize).uniform_(to=self.parents[-1].item())
        else:
            value = torch.DoubleTensor(batchsize).uniform_(to=self.parents[-1].item())
        idxs = self.search(value,self.levels-2,torch.zeros((batchsize,1),device=self.device))
        return idxs.view(-1).cuda() if torch.cuda.is_available() else idxs.view(-1)
    