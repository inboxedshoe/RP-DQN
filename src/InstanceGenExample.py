#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import InstanceGenerator as IG #import python file
import numpy as np


Gen = IG.Instance_Generator() #construct object

folder_path='data/TSP/100/train'

# generate instances and store them to folder
#Gen.create_data(problem_type='tsp', num_instances=10000, problem_size=100, folder_path=folder_path)

#read specific instance in instance folder
#data = Gen.read_specific_instance(folder_path + '/' +'00ace492-0177-459e-ab16-7fb0f242749f.pkl')
#print(data)

#read all instances in instance folder
#datas = Gen.read_all_instance(folder_path)
#print(datas[0])

#read random instance in instance folder
#data = Gen.read_random_instance(folder_path)
#print(data)

sampled_graphs = Gen.generate_tsp_data(dataset_size=10, tsp_size=20)
print(np.array(sampled_graphs).shape)
