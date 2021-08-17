# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 00:13:36 2020
@author: inbox#
"""
from AttentionModel import attention_model
from utils import flatten


from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import torch
import os
from itertools import product
import pprint
pp = pprint.PrettyPrinter(indent=4)
# Profiler
import cProfile

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')
# grid search loop; define parameters; all combinations will be tested; if specific combinations are undesired you have to manually insert an if check and continue to the next loop iteration


logging_options = {'log_mean_val': True,
                   'log_mean_train': True,
                   'log_val_plots': False,
                   'log_epsilon': True,
                   'log_buffer_cap' : True,
                   'log_episode_frequency': 50,
                   'log_val_frequency': 2000,

                   'log_qs': True,
                   'log_q_loss': True,
                   'log_target': True,
                   'log_step_frequency': 10000,

                   'log_step_debug_frequency': 10000,
                   'debug': False,

                   'log_h_params': False,

                   'verbose': False,
                   
                   'tqdm_disable': False
    }
pp.pprint(logging_options)
print()

params = {  'embedding_dimension' :     [128],
            'intermediate_dimension' :  [512],              
            'num_cust_node' :           [50],
            'episodes' :                [250000],
            'discount_factor' :         [1],
            'target_network_update' :   [0.999],
            'n' :                       [1],                         
            'exploration' :             [{'strategy':'boltzman',                 # Options: 'epsilon','boltzman'
                                          'start': 1.,
                                          'end': 0.02,
                                          'type': 'lin',                        # 'lin', 'exp', 'const'
                                          'warmup_episodes': 0,
                                          'linear_percentage': 0.5}],  
            'buffer_capacity' :         [2**14],                                # Please use a "2**x" as buffer size to make sure the sum tree is even -> faster.
            'lr' :                      [10**(-4)],
            'weight_decay' :            [0],
            'learn_iter_per_episode' :  [4],
            'batchsize' :               [1024],
            'batchsize_collect' :       [64],
            'reward_scaling' :          [False],
            'inner_masking' :           [False],
            'normalization' :           ['no'],                                 #Options: 'batch', 'inst', other strings will default to no normalization at all
            'beta_start':               [0.4],
            'buffer_min_samples':       [2**13],
            'num_depots':               [1],
            'problem_type':             ["cvrp"],
            'step_lr':                  [False],
            'special_demands':          [False],
            'validation_path':          ['data/CVRP/vrp50_validation_seed4321.pkl']                                  
            
            # mdvrp: 'data/MDVRP/mdvrp[size]_validation_seed4321.pkl'
            # cvrp:  'data/CVRP/vrp[size]_validation_seed4321.pkl'
        }
validsize = 100
profiler = False

#%%
#####begin runs######
keys, values = zip(*params.items())

for bundle in product(*values):
    dic = dict(zip(keys, bundle))
    pp.pprint(dic)
    print()
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #instantiate
    comment = ' {}{}_explr_{}__bs_{}__bsc_{}'.format(dic['problem_type'],dic['num_cust_node'],dic['exploration']['strategy'],dic['batchsize'],dic['batchsize_collect'])
    log_dir = os.path.join('runs', time + ' ' + comment)
    writer = SummaryWriter(log_dir)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(dir_path,'saved_models',time)
    dic['directory'] = directory
    
    model= attention_model(embed_dim=dic['embedding_dimension'],
                           intermediate_dim=dic['intermediate_dimension'],
                           num_depots=dic['num_depots'],
                           num_cust_node = dic['num_cust_node'],
                           num_heads=8,
                           problem_type=dic['problem_type'],
                           device=device,
                           validsize=validsize,
                           validation_path=dic['validation_path'],
                           inner_masking=dic['inner_masking'],
                           normalization = dic['normalization'],
                           )
    
    filename_config = 'config.json'
    filename_config  = os.path.join(directory,filename_config)
    os.makedirs(directory,exist_ok=True)
    
    with open(filename_config, 'w') as fp:
        json.dump(dic, fp, sort_keys=True, indent=4)
        
    # check if run should include profiler   
    if profiler:
        cProfile.run("model.train_model(dic,writer,logging_options)")
    else:
        model.train_model(dic, writer, logging_options)

    if logging_options['log_h_params']:
        val_tours,val_costs = model.validate()
        writer.add_hparams(flatten(dic),
                      {'Mean Validation Tour Cost': torch.mean(val_costs)})

    #save model
    filename_model = 'episode_' + str(dic['episodes']) + '_final.pt'
    filename_model  = os.path.join(directory,filename_model)
    torch.save(model.state_dict(), filename_model)
    
        
    writer.close()

#%%