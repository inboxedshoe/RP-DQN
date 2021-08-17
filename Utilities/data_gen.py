# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:33:56 2020

@author: inbox
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import pickle
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')   
            
device = torch.device('cpu')
torch.set_default_tensor_type('torch.FloatTensor')    
 



        
        
        
#dataset class for creating data        
class gen_data(Dataset):
    def __init__(self,  N_instances, N_cust = 50, N_depos = 2, prob_type = "cvrp"):
        
        #kinda useless rn cause we only have cvrp
        if prob_type == "cvrp":
            
            #number of problems
            self.N_instances = N_instances
            
            #make node locations
            self.nodes = torch.rand((N_instances, N_cust + N_depos, 2))
            
            #the paper chooses the truck capacity based on problem size
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }
            #the paper generates capacities as a random int between 0 and 10, we gonna normalize and do the same thing in read
            self.caps = torch.randint(0,10, size = (N_instances, N_cust, 1)).type(torch.float32)/CAPACITIES[N_cust]    
            #create depot capacities
            zer = torch.zeros(N_instances, N_depos, 1).type(torch.float32)
            #concat to a final capacity list
            self.caps = torch.cat((zer, self.caps), dim = 1).type(torch.float32)
            
            #concat locations and capacities
            self.nodes = torch.cat((self.nodes, self.caps), dim = 2)                                   
                
    def __len__(self):
        return self.N_instances

    def __getitem__(self, idx):
        return self.nodes[idx]  
    
    
#dataset class for read data, input the list from the pickle file as is
class gen_data_read(Dataset):
    def __init__(self,  data_read, normalize = True):
        
        #unzip the list
        data_read = list(zip(*data_read))

        #read depot locations
        depots = torch.Tensor(data_read[0])
        #read node locations
        nodes = torch.Tensor(data_read[1])
        #concat to toal locations
        self.nodes = torch.cat((depots,nodes), dim = 1)

        #read capacities for nodes
        self.caps = torch.Tensor(data_read[2]).unsqueeze(2).type(torch.float32)/data_read[3][0]       
        #create zero caps for depots to match our architecture
        zer = torch.zeros(depots.size(0), depots.size(1), 1).type(torch.float32)
        #concat caps
        self.caps = torch.cat((zer, self.caps), dim = 1).type(torch.float32)
        
        #concat locations and caps
        self.nodes = torch.cat((self.nodes, self.caps), dim = 2)   
        
    def __len__(self):
        return self.N_instances

    def __getitem__(self, idx):
        return self.nodes[idx]  
    
#please rename
class our_data_shit():
    
    #check if pkl extension found or add it
    def check_extension(self, filename):
        if os.path.splitext(filename)[1] != ".pkl":
            return filename + ".pkl"
        return filename

    #save the dataset
    def save_dataset(self, dataset, filename):
    
        filedir = os.path.split(filename)[0]
    
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
    
        torch.save(dataset, self.check_extension(filename))
            
    #useless rn, didnt fix it
    def generate_tsp_data(self, dataset_size, tsp_size):
        
        return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()
    
    #create dataset, returns dataloader class
    def make_data (self, N_instances, batch, write_file = True, N_cust = 50, N_depos = 2, prob_type = "cvrp"):
    
        data = gen_data(N_instances, N_cust, N_depos, prob_type)
        
        dataloader = DataLoader(data, batch_size = batch)
        
        if write_file == True:
            self.save_dataset(dataloader, "data/"+ prob_type +"_data")
        
            return dataloader
    #read data either we wrote or from their generator, return is dataloader
    def read_data (self, filename, batch, ours = True):
        
        if ours == True:
            data = torch.load(filename)
            dataloader = DataLoader(data, batch_size = batch)
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                custom_dataset = gen_data_read(data)
                dataloader = DataLoader(custom_dataset, batch_size = batch)
                
        return dataloader


#example
sets = our_data_shit()    
#b.pkl is a cvrp i generated with their generator here:
# https://github.com/wouterkool/attention-learn-to-route/blob/master/generate_data.py
d = sets.read_data("datas/b.pkl", batch = 5, ours = False)    
k = sets.make_data(100, 5, write_file = True)
k = sets.read_data("data/cvrp_data.pkl", 5)    
