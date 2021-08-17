# -*- coding: utf-8 -*-
"""
Dataset for the graphs
"""

import torch
import numpy as np

class graphsdataset():
    def __init__(self, coords):
        self.distances  = self.batch_pairwise_squared_distances(coords, coords)
        self.nodes = self.distances.shape[1]
                
    def w(self,v, u = None):
        # Return column without row u -> removes delete operation
        if u is None:
            row = self.distances[v]
            return row
        else:
            sample = self.distances[v,u]
            return sample
    
    def batch_pairwise_squared_distances(self, x, y):          
                                        
      x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
      y_t = y.permute(0,2,1).contiguous()
      y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
      dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
      dist[dist != dist] = 0 # replace nan values with 0
      return torch.clamp(dist, 0.0, np.inf)**0.5