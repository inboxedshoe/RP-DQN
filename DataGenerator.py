# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:33:56 2020

@author: inbox
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pickle
import itertools

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')   
            
device = torch.device('cpu')
torch.set_default_tensor_type('torch.FloatTensor')    

        
def read_test_set(filename, problem_type):
    f = open(filename, 'rb')
    data = pickle.load(f)
    
    data = list(zip(*data))
    #read depot locations
    depots = torch.Tensor(data[0])
    if problem_type == 'cvrp':
        depots = depots.unsqueeze(1)
    
    N_depots = depots.size(0)
    #read node locations
    nodes = torch.Tensor(data[1])
    #concat to toal locations
    nodes = torch.cat((depots ,nodes), dim = 1)
    
    #read demands for nodes
    demands = torch.Tensor(data[2]).unsqueeze(2).type(torch.float32)/data[3][0]       
    #create zero demands for depots to match our architecture
    zer = torch.zeros(depots.size(0), depots.size(1), 1).type(torch.float32)
    #concat demands
    demands = torch.cat((zer, demands), dim = 1).type(torch.float32)
    
    #concat locations and demands
    nodes = torch.cat((nodes, demands), dim = 2) 
    return nodes
    
    
    
class gen_data(Dataset):
    def __init__(self,  N_instances, N_cust = 50, N_depots = 1, prob_type = "cvrp", special_demands=False):
        
        #kinda useless rn cause we only have cvrp
        if prob_type == "cvrp":
            
            #number of problems
            self.N_instances = N_instances
            
            #number of depots
            self.N_depots = N_depots
            
            #make node locations
            self.nodes = torch.rand((N_instances, N_cust + N_depots, 2))
            
            if N_cust<=10:
                cap = 20.
            elif N_cust<=20:
                cap=30.
            elif N_cust<=50:
                cap=40.
            else:
                cap=50.
            
            #the paper chooses the truck capacity based on problem size
            # CAPACITIES = {
            #     10: 20.,
            #     20: 30.,
            #     50: 40.,
            #     100: 50.
            # }
            
            if special_demands:
                min_sampler = torch.distributions.uniform.Uniform(0.003, 0.04)
                mins = min_sampler.rsample(sample_shape=torch.Size([N_instances]))
                max_sampler = torch.distributions.uniform.Uniform(0.1, 0.6)
                maxs = max_sampler.rsample(sample_shape=torch.Size([N_instances]))
                
                demands_sampler = torch.distributions.uniform.Uniform(mins, maxs)
                self.demands = demands_sampler.rsample(sample_shape=torch.Size([N_cust])).permute(1,0).unsqueeze(2)
            else:
                #the paper generates demands as a random int between 0 and 10, we gonna normalize and do the same thing in read
                self.demands = torch.randint(1,10, size = (N_instances, N_cust, 1)).type(torch.float32)/cap#CAPACITIES[N_cust]    
            #create depot capacities
            zer = torch.zeros(N_instances, N_depots, 1).type(torch.float32)
            #concat to a final capacity list
            self.demands = torch.cat((zer, self.demands), dim = 1).type(torch.float32)
            
            #concat locations and capacities
            self.nodes = torch.cat((self.nodes, self.demands), dim = 2)                                   
                
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
        self.N_depots = depots.size(1)
        #read node locations
        nodes = torch.Tensor(data_read[1])
        #concat to toal locations
        self.nodes = torch.cat((depots,nodes), dim = 1)
        
        #read demands for nodes
        self.demands = torch.Tensor(data_read[2]).unsqueeze(2).type(torch.float32)/data_read[3][0]       
        #create zero demands for depots to match our architecture
        zer = torch.zeros(depots.size(0), depots.size(1), 1).type(torch.float32)
        #concat demands
        self.demands = torch.cat((zer, self.demands), dim = 1).type(torch.float32)
        
        #concat locations and demands
        self.nodes = torch.cat((self.nodes, self.demands), dim = 2)   
        
    def __len__(self):
        return self.nodes.size(0)

    def __getitem__(self, idx):
        return self.nodes[idx]  
    
    
#please rename
class data_generator():
    
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
        
    
    #create dataset, returns dataloader class
    def make_data (self, N_instances, seed = '', write_file = True, filename = "", N_cust = 50, N_depots = 1, prob_type = "cvrp", special_demands=False):
        
        if not seed == '':
            torch.manual_seed(seed)
        
        data = gen_data(N_instances, N_cust, N_depots, prob_type, special_demands)

        if write_file == True:
            
            if filename == "":
                print("Please enter filename")
                return
            
            self.save_dataset(data, filename+"_"+str(seed))
        else:
            return data
        
    #read data either we wrote or from their generator, return is dataloader
    def read_data (self, filename,ours = True, in_tensors=True):
        
        if ours == True:
            data = torch.load(filename)
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                if in_tensors:
                    data = gen_data_read(data)
        
        return data
    
def dataset_to_googleOR(dataset, idx, scale = 1, v_per_d = 10):
    data = {}
    ndepots = dataset.N_depots
    coord = dataset.nodes[idx] * scale
    nvehicles = v_per_d * ndepots
    data['depot'] = list(range(ndepots))
    data['starts'] = data['ends'] = list(itertools.chain(*itertools.repeat(data['depot'], v_per_d) ))
    data['num_vehicles']= nvehicles
    data['demands'] = coord[:,2].int().tolist()
    #data["starts"] = [0, 2, 15, 16]
    #data["ends"] = [0, 2, 15, 16]
    data['vehicle_capacities'] = [1 * scale for i in range(nvehicles)] 
    data['locations'] = list(map(tuple, coord[:,:2].int().numpy()))
    data['distance_matrix'] = np.round(squareform(pdist(coord[:,:2],"euclidean"))).astype(int).tolist()
    return data
#%%
''' @kirill du kannst mit den beiden zeilen die mdvrp test sets laden die wir auch bei kool et al verwenden.
    Jede instanz besteht aus einem tuple(depot coords, node coords, node demands, vehicle capacity)
    Da die Anzahl der vehicle nicht begrenzt ist haben sie alle die gleiche capacatiy
 '''               

#sets = data_generator()
#d1 = sets.read_data("data/MDVRP/mdvrp20_test_seed1234.pkl", ours = False, in_tensors=True)  

#example
sets = data_generator()    
# b.pkl is a cvrp i generated with their generator here:
# https://github.com/wouterkool/attention-learn-to-route/blob/master/generate_data.py
# d = sets.read_data("datas/b.pkl", ours = False)  
  
# k = sets.make_data(100, seed = 6, filename = "datab/cvrp_data.pkl", write_file = True)
k = sets.make_data(100, seed = 6, write_file = False)
# k = sets.read_data("data/cvrp_data.pkl", ours = True)    
