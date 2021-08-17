# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:33:56 2020

@author: inbox
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')   
            
    
    
class gen_data(Dataset):
    def __init__(self, N_instances, N_cust = 50, N_depos = 2, capa = 100, write_file = False):
        
        self.N_instances = N_instances
        self.nodes = torch.randint(0,100,(N_instances, N_cust + N_depos, 2)) .type(torch.float32)    
        self.caps = torch.randint(0,100,(N_instances, N_cust, 1)).type(torch.float32)/capa
        zer = torch.zeros(N_instances, N_depos, 1).type(torch.float32)
        self.caps = torch.cat((zer, self.caps), dim = 1).type(torch.float32)
        
        self.nodes = torch.cat((self.nodes, self.caps), dim = 2)
    
        if write_file == True: 
            torch.save(self.nodes, "data")
                
    def __len__(self):
        return self.N_instances

    def __getitem__(self, idx):
        return self.nodes[idx]  
    

data = gen_data(10, write_file = True)

dataloader = DataLoader(data, batch_size = 2)

for i, batch in enumerate(dataloader):
    b = i, batch
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    