# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:23:18 2020

@author: Simons PC
"""
import torch
import numpy as np

class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.leafs = [Node(None,None,is_leaf=True,value=0,idx=i) for i in range(max_size)]
        self.root = self.create_structure(self.leafs)
        self.cursor = 0
        self.size= 0
        
    def create_structure(self,leafs):
        nodes = leafs
        while len(nodes) > 1:
            inodes = iter(nodes)
            nodes = [Node(*pair) for pair in zip(inodes, inodes)]
        return nodes[0]
    
    def update(self,idx,value):
        for i,v in enumerate(idx):
            change = value[i] - self.leafs[v].value
            self.leafs[v].value = value[i]
            self.prop_change(change,self.leafs[v].parent)
            
    def prop_change(self,change,node):
        '''Recursively propagate the value changes along the parents'''
        node.value += change
        if node.parent is not None:
            self.prop_change(change,node.parent)
        
    def search(self,value,node):
        '''Recursive function to find idx from given node'''
        if node.is_leaf:
            return node.idx
        if node.left.value >=value:
            return self.search(value,node.left)
        else:
            return self.search(value-node.left.value,node.right)
        
    def sample(self,batchsize):
        '''Traverse the whole tree to find indices to sample given a batch-size'''
        # TODO split into *batchsize* ranges, sample one element per range uniform, then traverse
        value = np.random.uniform(high=self.root.value,size=batchsize)
        return torch.tensor([self.search(i,self.root) for i in value])
        
class Node(object):
    def __init__(self,left,right,is_leaf=False,value=0,idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.idx = idx
        self.parent = None
        
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
            right.parent = self
            left.parent = self
        else:
            self.value=value
        