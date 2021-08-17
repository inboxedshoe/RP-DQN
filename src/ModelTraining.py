# -*- coding: utf-8 -*-
"""
Model training class
"""
from Epsilon import Epsilon
from ReplayBuffer import ReplayMemory
from ReplayBuffer import Experience
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from Model import s2v_DQN
from Graphs import graphsdataset
import InstanceGenerator as IG
import math

class ModelTraining():
    def __init__(self, N,epsilon,pe,gamma,n,tsp_size, target_network_params = ('polyak', 0.999, True),validation_path=None,validsize=100,writer=None,device=torch.device('cuda')):
        self.device= device
        self.replay_memory = ReplayMemory(N, tsp_size,device=self.device)
        
        self.p = torch.nn.Parameter(torch.tensor([pe], device= self.device), requires_grad= False)
        self.gamma = torch.nn.Parameter(torch.tensor([gamma], device= self.device, dtype = torch.float32))
        self.n = torch.nn.Parameter(torch.tensor([n], device= self.device), requires_grad= False)
        
        self.epsilon = Epsilon(epsilon[0],epsilon[1],epsilon[2])
        self.routes = []
        self.rewards = torch.tensor([],device= self.device, dtype = torch.float32)
        self.loss_hist= torch.tensor([],device= self.device, dtype = torch.float32)
        self.steps_done = torch.tensor([0], device= self.device, dtype = torch.float32)
        self.epochs_done = torch.tensor([0], device= self.device, dtype = torch.float32)
        
        self.generator = IG.Instance_Generator()
        self.writer = writer
            
        if validation_path is not None:
            self.path = validation_path
            self.valid_graphs = self.drawvalid(validsize)
            self.validsize = validsize
        else:
            self.path=None
        # Init network
        self.tsp_size = tsp_size
        self.network = s2v_DQN(self.p,self.tsp_size,device=self.device).to(device= self.device)
        self.number_s2v_layers = 4
        
        # Init target network
        self.target_network = s2v_DQN(self.p,self.tsp_size).to(device= self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        self.target_network_update_mode = target_network_params[0] #polyak or classic
        self.target_network_update_value = target_network_params[1]
        self.target_network_include_s2v = target_network_params[2]
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(),lr=0.001,weight_decay=0.1)
        self.model_parameter_validation()
    
    def model_parameter_validation(self):
        if self.target_network_update_mode != 'polyak' and self.target_network_update_mode != 'classic':
            raise Exception('Unknown mode: ', self.target_network_update_mode)
        
        if self.target_network_update_mode == 'polyak' and (self.target_network_update_value                                                            >= 1 or self.target_network_update_value <=0):
            raise Exception('Polyak rate must be between 1 and 0 (exclusive) but is:  ' + str(self.target_network_update_value))
            
        if self.target_network_update_mode == 'classic' and (self.target_network_update_value                                                            <= 0 or not(isinstance(self.target_network_update_value,int))):
            raise Exception('Update rate must be a positiv intiger:  ' + str(self.target_network_update_value))
            
        if not(isinstance(self.target_network_include_s2v,bool)):
            raise Exception('Target net s2v param must be boolean but is: ' + str(self.target_network_include_s2v))   
    

    
    def drawvalid(self,size):
        '''Draw validation graphs'''
        val = []
        for i in range(size):
            val.append(self.generator.read_random_instance(self.path))
        val_graphs = graphsdataset(torch.tensor(val, device=self.device))
        return val_graphs
    
    # Calculate this on GPU        
    def cumreward(self,t,hist):
        stacked = torch.cat(hist,axis=1)[:,t:t+(self.n.item())] #flag
        gammav = torch.torch.logspace(0,(self.n-1).item(),base=self.gamma.data.item(),steps = self.n.item(),device=self.device)
        stacked = (gammav * stacked).sum(axis=1)
        return stacked

        
    def calc_distances(self,orig,actions,graph):
        #save the number of column of original
        n = orig.size(1)
        m = orig.size(0)
        #cop = orig.clone()
        #repeat the rows to create scatterable version

        #orig = orig.repeat_interleave(n, dim= 0).view(orig.size(0),-1,orig.size(1))
        orig = orig.reshape(m,1,n).expand(m, n, n)

        #create identity matrix without last row
        mask = torch.eye(n ,n+1,device=self.device).bool().unsqueeze(0)
        mask = mask.expand(orig.size(0),n,n+1)
        #mask = mask.repeat_interleave(orig.size(0), dim = 0)

        #create placeholder tensor
        a = torch.zeros(orig.size(0),orig.size(1),orig.size(2)+1,device=self.device).float() 
        #create scatterable action matrix
        #actions = actions.repeat_interleave(n, dim= 0)
        actions = actions.expand(m,n).reshape(m*n,1)

        #scatter actions in diagonal, and the values of original everywhere else
        a.masked_scatter_(mask, actions.float())
        a.masked_scatter_(mask == False, orig)
        tours = a
        #copy first column and append it to the end so we have a loop
        a = torch.cat((a, a[:,:,0].view(a.size(0), a.size(1), 1)), dim = 2)
        #unfold into 2 element chunks
        a = a.unfold(2, 2, 1).long()
        
        distmat = graph.distances.unsqueeze(1).expand(-1, a.size(1), -1, -1)
        # Find distances, sum up up the tourcosts and find indices to insert at with argmin
        costs,mins  = distmat[torch.arange(distmat.size(0),device= self.device)[:, None, None, None],
                      torch.arange(distmat.size(1),device= self.device)[:, None, None],
                      a[:, :, :, 0:1], a[:, :, :, 1:2]].sum(axis=2).min(axis=1)
        # Check device
                
        return(mins,tours,-1*costs)
        

    
    def choose_action(self, q_values, features,step, epsi):
        # Masking
        q_values.masked_fill_(features.bool(),-np.inf)
        
        # Epsilon strategy
        rate = epsi
        #if ((self.steps_done%100)==0):
         #   writer.add_scalar('Exploration rate', rate, self.epochs_done)
        if rate > np.random.rand():
            
            idx = torch.arange(features.shape[1],device=self.device).expand(features.shape[0],self.tsp_size)
            #idx = torch.arange(features.shape[1],device=device).repeat(features.shape[0],1)
            
            idx = idx[features==0].view(features.shape[0],-1)
            
            mask = torch.randint(0, idx.shape[1], (idx.shape[0],1), device= self.device).view(-1)
            
            #return torch.gather(idx, 1, mask)           #try and optimize also
            return idx[torch.arange(idx.size(0), device= self.device), mask].view(-1,1)
        
        else:
            with torch.no_grad():
                _,action = torch.max(q_values,1,keepdim= True)   
                return(action)
                           
                
    def choose_greedy_action(self,qs,features):
        qs.masked_fill_(features.bool(),-np.inf)
        _,action = torch.max(qs,1,keepdim= True)   
        return(action)
    
    def transition(self,state,action):
        nstate = state.scatter(1,action,1)
        return nstate.to(device=self.device) # is this on GPU already?
            
    
    def insertnode(self,S,action,oldcost,graph):
        
        if len(S)==0:
            S = torch.cat((action.float(),S.float()),1).float()
            reward = torch.zeros((S.shape[0]),device=self.device,dtype=torch.float32).view(-1,1) # Zeros pro batch
            tourcost = torch.zeros((S.shape[0]),device=self.device,dtype=torch.float32).view(-1,1) # Zeros pro batch
                        
        else:
            indices,tours,tourcost = self.calc_distances(S,action,graph)
            S = tours[torch.arange(tours.size(0)), indices.view(-1)]             #should be about 10x faster than the gather function
            
            reward = tourcost - oldcost # Check device here

        return S, reward.float(), tourcost.float() # Check device and dtype
    
        
    def extract_tensors(self, experiences):
        # Convert batch of Experiences to Experience of batches
        #print(experiences)
        batch = list(zip(*experiences))
        t1 = torch.stack(batch[0])
        t2 = torch.cat(batch[1])
        t3 = torch.stack(batch[2])
        t4 = torch.stack(batch[3])
        non_final_ind = (t4.sum(axis=1) != self.tsp_size).nonzero().view(-1)
        t5 = torch.cat(batch[4])
        return (t1,t2,t3,t4,non_final_ind,t5)
    
    def learn(self,batch_size):
        ''' Method to apply backpropagation. A batch is taken from the replay memory, and the cost function is applied to it'''
        states,actions,rewards,next_states,mask,dist_indices = self.replay_memory.sample(batch_size, self.tsp_size)
  
        #states.size()
        #actions.size()
        #rewards.size()
        #next_states.size()
        #dist_indices.size() 
        
        nmask = np.delete(np.arange(len(states)),mask.cpu())
        state_qs = self.network.forward(states,self.graph,dist_indices)
        
        state_qs = state_qs.gather(1, actions.long().view(-1,1)).squeeze()
        state_qs = torch.cat((state_qs[mask],state_qs[nmask]))
        rewards = torch.cat((rewards[mask],rewards[nmask]))
        
        # HERE IS STILL A SMALL ERROR. WHEN THE BATCH SIZE IS SMALL, THEN IT MIGHT HAPPEN THAT THERE IS ONLY ONE NON FINAL STATE.
        # THEN, THE torch.max(x,1)[0] FUNCTION WILL NOT WORK, AS THE INPUT IS ONLY 1-DIMENSIONAL!!!
        with torch.no_grad():
            # Non-terminal states
            next_state_qs = self.target_network.forward(next_states[mask],self.graph,dist_indices[mask])
            next_state_qs = torch.max(next_state_qs,1)[0]    
            next_none_qs = torch.zeros(len(nmask),device=self.device)
            next_state_qs = torch.cat((next_state_qs,next_none_qs))
            y = (self.gamma ** self.n) * next_state_qs + rewards
            
        loss = self.criterion(state_qs.float(),y.float()).unsqueeze(0)
        
        
        # Track loss hist
        if ((self.steps_done%100)==0):
            if self.path is not None:
                self.validate(epoch = self.epochs_done)
                self.writer.add_scalar('Q-learning loss', loss, self.steps_done)
            self.loss_hist = torch.cat((self.loss_hist,loss),0) #flag
            #self.loss_hist.append(loss)''
       
        loss.backward()
        self.optimizer.step()
        self.target_network_controller()
        
        
    def validate(self, epoch,mode="val",graphs = None):
        '''Test on validation or test'''
        if mode == "test":
            graph = graphs
            size = graph.distances.shape[0]
        else:
            graph = self.valid_graphs
            size = self.validsize
        
        val_state = torch.zeros((size,self.tsp_size),device=self.device,dtype=torch.float32)
        val_S = torch.tensor([],device=self.device).float()
        val_reward_sum = torch.zeros((size,1),device=self.device,dtype=torch.float32)
        # Forward pass until episode finished
        with torch.no_grad():        
            for step in range(self.tsp_size):
                val_qs = self.network.forward(val_state,graph,torch.arange(size, device= self.device))
                val_actions = self.choose_greedy_action(val_qs,val_state)
                val_S,val_reward,val_reward_sum = self.insertnode(val_S,val_actions,val_reward_sum,graph)
                if (step+1)<self.tsp_size:    
                    val_state = self.transition(val_state,val_actions)
        # Calculate average cost
        res = torch.mean(val_reward_sum)
        
        if mode != "test":
            self.writer.add_scalar('Mean Tour Length', res, epoch)
        
        return res
            
    def training(self,episodes,batch_size):
        for e in tqdm(range(episodes)):
            epsi = torch.tensor(self.epsilon.get_exploration_rate(self.epochs_done), device=self.device)
            train_data = torch.rand(size=(batch_size, self.tsp_size, 2), device= self.device)
            self.graph = graphsdataset(train_data)
            
            state = torch.zeros((batch_size,self.tsp_size),device=self.device,dtype=torch.float32)
            S = torch.tensor([],device=self.device, dtype= torch.float32)
            
            # TODO Maybe these as tensors? I dont think its worth though
            statehist  = []
            actionhist  = []
            rewardhist  = []
            
            reward_sum = torch.zeros((batch_size,1),device=self.device,dtype=torch.float32)
            
            for step in range(self.tsp_size):
                self.optimizer.zero_grad()
                with torch.no_grad():
                    # Compute q-values and action with epsilon-greedy strategy
                    qs = self.network.forward(state,self.graph,torch.arange(batch_size, device= self.device))
                    action = self.choose_action(qs,state,self.steps_done, epsi)
                    S,reward,reward_sum = self.insertnode(S,action,reward_sum,self.graph)
                    self.steps_done+=1

                    next_state = self.transition(state,action)
                    
                    statehist.append(state)
                    actionhist.append(action)
                    rewardhist.append(reward)
                    state = next_state
                    
                if(step >= self.n):
                    cumrewards = self.cumreward(step-self.n,rewardhist)
                    
                    exp_test = Experience(statehist[step-self.n],actionhist[step-self.n].float(),cumrewards.view(-1),state.float(),torch.arange(batch_size,device=self.device).view(-1,1).float())
                    #exp_test = list(zip(statehist[step-self.n],actionhist[step-self.n],cumrewards.view(-1),state,torch.arange(batch_size).view(-1,1)))
                    self.replay_memory.push(exp_test)

                # Start backpropagation 
                if(self.replay_memory.can_provide_sample(batch_size)):
                    self.learn(batch_size)

            self.rewards = torch.cat((self.rewards,reward_sum.float()),0)
            self.routes.append(S)
            self.epochs_done += 1
            
        self.rewards = self.rewards.view(-1,batch_size)
        
            
    def target_network_controller(self):
        net_params_dict = self.network.state_dict()
        target_net_params_dict = self.target_network.state_dict()
        
        if self.target_network_include_s2v:
            relevant_parameters =  list(net_params_dict)
        else:
            relevant_parameters = list(net_params_dict)[self.number_s2v_layers*2:]
            for key in  list(net_params_dict)[:self.number_s2v_layers*2]:
                target_net_params_dict[key] = net_params_dict[key].clone().detach()
        
        for key in relevant_parameters:
            if self.target_network_update_mode == 'polyak':
                target_net_params_dict[key]  = target_net_params_dict[key] * (self.target_network_update_value) + net_params_dict[key].clone().detach() * (1 -self.target_network_update_value)
                
            elif self.target_network_update_mode == 'classic' and self.steps_done % self.target_network_update_value == 0:
                target_net_params_dict[key] = net_params_dict[key].clone().detach()
        
        
        self.target_network.load_state_dict(target_net_params_dict)
