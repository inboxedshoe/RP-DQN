import numpy as np
import uuid
import os
import random
from Utilities import data_utils

class Instance_Generator(object):
    """Generates instances of routing problems.
    TSP: standard TSP - uniform sampling in unit square

    CVRP: standard CVRP - uniform sampling in unit square
    """

    def __init__(self):
        pass
    def generate_tsp_data(self, dataset_size, tsp_size):
        return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()
    
    def generate_vrp_data(self, dataset_size, vrp_size):
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }
        return list(zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
            np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
            np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
        ))
    
    def create_data(self, problem_type, num_instances, problem_size, folder_path, seed=None):
        np.random.seed(seed)

        if problem_type == 'tsp':
            dataset = self.generate_tsp_data(num_instances, problem_size)
        elif problem_type == 'cvrp':
            dataset = self.generate_vrp_data(num_instances, problem_size)
        else:
            print('unknown problem type')

        for i in dataset:
            unique = str(uuid.uuid4())
            path = os.path.join(folder_path, unique + '.pkl')
            data_utils.save_dataset(i, path)


    def read_specific_instance(self, file_path):

        return data_utils.load_dataset(file_path)
    
    def read_all_instance(self, file_path):
        
        files = os.listdir(file_path)
        datas=[]
        for file in files:
            data = data_utils.load_dataset(os.path.join(file_path, file))
            datas.append(data)
        return datas
    
    def read_random_instance(self, file_path):

        file = random.choice (os.listdir(file_path))
        data = data_utils.load_dataset(os.path.join(file_path, file))
        return data