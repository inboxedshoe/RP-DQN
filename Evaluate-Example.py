# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 00:13:36 2020

@author: inbox#
"""
from AttentionModel import attention_model
import torch
import os
from DataGenerator import data_generator, read_test_set
import matplotlib.pyplot as plt
import time
from TestInstanceReader import read_test_instance
from plot_cvrp import plot_vehicle_routes
import tqdm
from Utilities.model_loader import load_model
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')

import cProfile

# validation
# #choose the model_sizes that we will be testing
# model_size=[20,50,100]
# #choose the number of depots used for each model
# num_depots=[1, 2, 4]

# for i, s in enumerate(model_size):
#     #load the model from the corrosponding file, can replace with mdvrp model
#     model = load_model(model_filename='saved_models/cvrp_' + str(s) + '_final/episode_250000_final.pt',
#                        embedding_dimension=128,
#                        problem_type='cvrp',
#                        normalization='no',
#                        inner_masking=False,
#                        num_depots=num_depots[i],
#                        valid_size=100,
#                        num_cust_node = s,
#                        device=device,
#                        intermediate_dim = 512)

#     # cvrp100 = 'data/MDVRP/mdvrp' + str(size) + '_test_seed1234.pkl'
#     # data_cvrp100 = read_test_set(cvrp100, 'mdvrp')

#     start = time.time()
#     a,b = model.validate(batch_size = 8)
#     end = time.time() - start
#     print("time: " + str(end))
#     print("mean: " + str(torch.mean(b).item()) + " for: model " + str(s))


#%%

#choose the test size, model size and number of depots for each modelsize
test_size = [1]
model_size=[100]
num_depots=[4]

for s in model_size:
    for i,size in enumerate(test_size):

        model = load_model(model_filename='saved_models/mdvrp_' + str(s) + '_final/episode_250000_final.pt',
                            embedding_dimension=128,
                            problem_type='mdvrp',
                            normalization='no',
                            inner_masking=False,
                            num_depots=num_depots[i],
                            valid_size=100,
                            num_cust_node=size,
                            device=device,
                            intermediate_dim=512)
        
        #set 1 or multiple different temperatures to try
        temperatures = [0.0005,0.001,0.005,0.01,0.05]
        #set number of samples to use
        samples = [1024]
        
        #change whether mdvrp or cvrp datasets here
        data_file = 'data/MDVRP/mdvrp' + str(size) + '_test_seed1234.pkl'
        data_content = read_test_set(data_file, 'mdvrp')
        
        for temp in temperatures:
            for sample in samples:
                start = time.time()
                a,b = model.sampling(data = data_content, softmax_temperature = temperatures[i], num_solutions = sample)
                a,b = model.sampling(softmax_temperature = temp, num_solutions = sample)
                print("time: " + str(time.time() - start))
                print("mean: " + str(torch.mean(b).item()) + " for: model " + str(s) + " and test " + str(size) + " and temp: " + str(temperatures[i]))

    