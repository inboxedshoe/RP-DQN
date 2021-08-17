# -*- coding: utf-8 -*-
"""
Main Training script
"""

import torch
import os

import torch.utils.bottleneck as bottleneck
from torch.utils.tensorboard import SummaryWriter
import datetime
from ModelTraining import ModelTraining
#%%


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
    
# default `tensorboard save directory
log_dir = "runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

#%%
size = 20
folder_path= os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/TSP/20/val'))

epochs = 10
eps_start = 0.9
eps_end = 0.01
eps_decay = (epochs*0.9)/(4.5-(eps_start - eps_end))

model = ModelTraining(640,[eps_decay,eps_start,eps_end],64,0.999,3,size, target_network_params=('polyak', 0.999, True),validation_path=folder_path,writer=writer,device=device)

# In[101]
# I ran the code with cProfile to see all calls
#import cProfile

#cProfile.run("model.training(epochs,64)")  
model.training(epochs,64)
#%% Testing on instances
'''test_path = 'data/TSP/20/test'
test_graphs = []
test_size = 100
for i in range(test_size):
    test_graphs.append(model.generator.read_random_instance(test_path))
test_graphs = graphsdataset(torch.tensor(test_graphs, device = device))
#model.validate(None,mode="test",graphs=test_graphs)'''