#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:16:37 2020

@author: jukebox
"""

from AttentionModel import attention_model
import torch
import json

def load_model(model_filename, embedding_dimension, problem_type, normalization, inner_masking, device, num_depots=1, valid_size = 1000,num_cust_node=20, intermediate_dim=128):
    PATH = model_filename
    # with open('data.json', 'r') as fp:
    # data = json.load(fp)
    
    model= attention_model(embed_dim=embedding_dimension,
                       intermediate_dim=intermediate_dim,
                       num_depots=num_depots,
                       num_cust_node = num_cust_node,
                       num_heads=8,
                       problem_type=problem_type,
                       device=device,
                       validsize=valid_size,
                       validation_path=None,
                       inner_masking=inner_masking,
                       normalization = normalization
                       )
    
    model.load_state_dict(torch.load(PATH, map_location=device))
    return model

def load_model_config(model_filename, config_file, device, num_depots = 0, num_cust_node = 0):
    PATH = model_filename
    
    with open(config_file, 'r') as fp:
        dic = json.load(fp)
    
    if num_cust_node == 0:
        num_cust_node = dic['num_cust_node']
    if num_depots == 0:
        num_depots = dic["num_depots"]
        
    model= attention_model(embed_dim=dic['embedding_dimension'],
                       intermediate_dim=dic['intermediate_dimension'],
                       num_depots=num_depots,
                       num_cust_node = num_cust_node,
                       num_heads=8,
                       problem_type=dic['problem_type'],
                       device = device,
                       validsize = 1000,
                       validation_path=None,
                       inner_masking=dic['inner_masking'],
                       normalization = dic['normalization']
                       )
                           
    model.load_state_dict(torch.load(PATH, map_location=device))
    return model