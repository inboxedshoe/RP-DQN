import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
#import pytorch_warmup as warmup
from torch.optim.lr_scheduler import StepLR

import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm
import itertools

from PrioritizedReplayBuffer_SumTree import ReplayMemory
from PrioritizedReplayBuffer_SumTree import States
from PrioritizedReplayBuffer_SumTree import Experience

from problem import Problem
from DataGenerator import data_generator, read_test_set
from utils import Scheduler, Logging
from plot_cvrp import plot_vehicle_routes
from model_components import node_encoder, decoder_module


class attention_model(nn.Module):
    def __init__(self, embed_dim=128, intermediate_dim=512, num_depots=1, num_cust_node=20, num_features_node=3, num_heads=8, problem_type='cvrp', device=torch.device("cuda"), validsize=1000, validation_path=None, inner_masking=True, normalization='None', step_lr= False):
        super(attention_model, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_depots = num_depots
        
        if problem_type=="cvrp":
            self.num_features_node = 7
        elif problem_type=="mdvrp":
            self.num_features_node = 8
            
#        self.num_features_node = num_features_node
        self.num_heads = num_heads
        self.num_cust_node = num_cust_node
        self.graph_size = num_cust_node + num_depots
        self.problem = Problem(problem_type, self.num_depots, self.device)
        self.encoder = node_encoder(num_depots=num_depots, embed_dim=embed_dim, intermediate_dim=intermediate_dim, num_heads=num_heads, normalization=normalization, num_features_node=self.num_features_node).to(device=self.device)
        self.decoder = decoder_module(embed_dim=embed_dim,num_depots = num_depots, num_heads=num_heads, inner_masking=inner_masking).to(device=self.device)

        #target network
        self.encoder_target = node_encoder(num_depots=num_depots, embed_dim=embed_dim, intermediate_dim=intermediate_dim, num_heads=num_heads, normalization=normalization, num_features_node=self.num_features_node).to(device=self.device)
        self.decoder_target = decoder_module(embed_dim=embed_dim, num_depots = num_depots, num_heads=num_heads, inner_masking=inner_masking).to(device=self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.decoder_target.load_state_dict(self.decoder.state_dict())
        
        self.step_lr = step_lr
        
        # Initialize validation data
        self.gen = data_generator()
        if validation_path is not None:
            self.valid_path = validation_path
            self.validation_data = read_test_set(validation_path, problem_type)
        else:
            self.valid_path = None
            self.validation_data = self.gen.make_data(N_instances=validsize, write_file=False, N_cust=self.num_cust_node,  N_depots = self.num_depots).nodes
      
    def compute_static_components(self, batch):
        """ Some components of the model are fixed per batch and do not need to be recomputed at every decoding step. Also some other initialization is done here."""
        #initial values
        batch_size = batch.shape[0]
        tour, current_depot = self.problem.create_initial_tour_plan(batch_size)
        """tours can have variable length
        we index all uncompleted tours and keep track of them after every decoding step to avoid computing values for tours that don't have actions left.
        for those tours in the tour plan a -1 will be appended as the action"""
        uncompleted_tours = torch.ones(batch_size, dtype=torch.bool,device=self.device)
        capacities = self.problem.create_initial_capacities(batch_size)
        cost = self.problem.create_initial_cost(batch_size)
        features = self.problem.create_initial_features(batch)        
        return tour, uncompleted_tours, capacities, cost, current_depot, features

    def choose_action(self, qs, exploration_strategy, eps=None, softmax_temperature=1.):

        if exploration_strategy == 'epsilon':
            if eps > np.random.rand():
                nq = qs.detach().clone()
                nq [nq !=-math.inf] = 1
                nq  = nq .view(qs.shape[0],-1)
                action_odds = nn.functional.softmax(nq ,dim=1)

                acts = torch.multinomial(action_odds,1)
                return acts

            else:
                with torch.no_grad():
                    action = self.choose_greedy_action(qs)
                    return(action)

        elif exploration_strategy == 'boltzman':
            action_odds = nn.functional.softmax(qs/softmax_temperature ,dim=2).squeeze(1)
            acts = torch.multinomial(action_odds,1)
            return acts

    def choose_greedy_action(self, qs):
        _,action = torch.max(qs,2,keepdim= False)
        return(action)

    def train_model(self, opts, writer, log_opts):

        self.gamma = opts['discount_factor'] # parameter for scaling long-term rewards
        self.n = opts['n'] # N for n-step q-learning
        self.explorer = Scheduler(opts['exploration']['start'],opts['exploration']['end'], opts['episodes'], opts['exploration']['type'], opts['exploration']['warmup_episodes'], opts['exploration']['linear_percentage'])
        self.max_grad_norm = 1
        self.epsilon_td = 1e-4
        self.logging = Logging(writer, log_opts, self)
        self.buffer = ReplayMemory(opts['buffer_capacity'], self.graph_size, alpha=0.6, beta=[opts['beta_start'],opts['episodes']],num_features=self.num_features_node, device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(itertools.chain(*[self.encoder.parameters(), self.decoder.parameters()]),lr=opts['lr'], weight_decay=opts['weight_decay'])

        """Main Training loop"""
        self.steps_done = 0
        self.episodes_done=0
        cost = None # avoid undeclared variable in first logging step
        
        scheduler = StepLR(self.optimizer, step_size=40000, gamma=0.1)
        
        for episode in tqdm(range(opts['episodes']), disable=log_opts['tqdm_disable']):
            #logging
            self.logging.log_episode(cost, self.buffer.get_fill_percentage(), self.episodes_done, self.explorer.get_exploration_rate(self.episodes_done))

            # TODO Initialize history tensors of the current episode
            statehist = []
            actionhist = []
            rewardhist = []

            #generate data
            batch = self.gen.make_data(N_instances=opts['batchsize_collect'], write_file=False, N_cust=self.num_cust_node,  N_depots = self.num_depots, special_demands=opts['special_demands']).nodes

            with torch.no_grad():
                tours, uncompleted_tours, capacities, cost, current_depot,features = self.compute_static_components(batch)
                tours_remaining = True
    
                step = 0
                while tours_remaining:
                    ''' pass with experience storing'''
                    #obtain the mask (need to do first now to mask out feasibility feature)
                    mask = self.problem.create_mask(tours,
                                                    self.graph_size,
                                                    capacities,
                                                    batch,
                                                    current_depot)
                    if step > 0:
                        features = self.problem.update_features(features, mask, batch[:,:,2],  tours, capacities, current_depot)
    
    
                    Q_vals = self.forward(features[uncompleted_tours], tours[uncompleted_tours], capacities[uncompleted_tours], mask[uncompleted_tours], current_depot[uncompleted_tours], target = False)
    
                    num_step = torch.ones((batch.size(0),1))*step #only debugging reasons
                    current_state = States(features.detach().clone(), tours[:,-1].view(-1,1) if tours.size(1) != 0 else torch.ones_like(current_depot) * -1, mask.detach().clone(), capacities.detach().clone(), current_depot.detach().clone(), uncompleted_tours.detach().clone(),  num_step.detach().clone())
                    statehist.append(current_state)
    
                    # Exploration
                    action = self.choose_action(Q_vals, opts['exploration']['strategy'], eps=self.explorer.get_exploration_rate(self.episodes_done), softmax_temperature=self.explorer.get_exploration_rate(self.episodes_done))
                    temp_act = torch.zeros(opts['batchsize_collect'],1,dtype=torch.long)
                    temp_act[uncompleted_tours] = action
                    actionhist.append(temp_act)
    
                    #update tour
                    tours, current_depot = self.problem.update_tour_plan(tours,
                                                              action,
                                                              uncompleted_tours,
                                                              current_depot)
    
                    #update capacities
                    capacities[uncompleted_tours] = self.problem.update_capacities(batch[uncompleted_tours],
                                                             action,
                                                             capacities[uncompleted_tours])
    
                    #update the costs
                    cost[uncompleted_tours], tour_cost = self.problem.update_cost(cost[uncompleted_tours],
                                                                                 batch[uncompleted_tours],
                                                                                 tours[uncompleted_tours],
                                                                                 opts['reward_scaling'])
                    temp_cost = torch.zeros(opts['batchsize_collect'],1)
                    temp_cost[uncompleted_tours] = -1*tour_cost
                    rewardhist.append(temp_cost)
    
    
                    #update incomplete tour indices
                    uncompleted_tours = self.problem.update_unfinished_tour_indices(tours,
                                                                                         uncompleted_tours,
                                                                                         self.graph_size)
                    tours_remaining = uncompleted_tours.any()
    
                    step+=1

            final_state = States(features.detach().clone(), tours[:,-1].view(-1,1) if tours.size(1) != 0 else torch.ones_like(current_depot) * -1, mask.detach().clone(), capacities.detach().clone(), current_depot.detach().clone(), uncompleted_tours.detach().clone(),  num_step.detach().clone())
            # Store experience
            assert len(actionhist) == len(rewardhist) == len(statehist)
            for i in range(len(actionhist)):
                rewards = self.reward_cumulation(rewardhist,i) # returns the discounted cumulated reward for the target value
                ind = statehist[i].uncompleted

                if i + self.n < len(actionhist):
                    experience = Experience(statehist[i][ind],actionhist[i][ind],rewards[ind].view(-1,1), statehist[i+self.n][ind])
                else:
                    experience = Experience(statehist[i][ind],actionhist[i][ind],rewards[ind].view(-1,1), final_state[ind])

                self.buffer.push(experience)

            # Learn
            if self.buffer.can_provide_sample(opts['buffer_min_samples']) and self.buffer.can_provide_sample(opts['batchsize']):
                for j in range(opts['learn_iter_per_episode']):
                    self.learn(opts['batchsize'], opts['target_network_update'])
                    self.steps_done+=1

            self.episodes_done+=1
            
            if (self.step_lr == True):
                scheduler.step()
            if self.episodes_done % 10000 == 0:
                
                #save model
                filename_model = 'episode_' + str(self.episodes_done) + '.pt'
                filename_model  = os.path.join(opts['directory'],filename_model)
                os.makedirs(opts['directory'],exist_ok=True)
                torch.save(self.state_dict(), filename_model)
                

        #return whether any incomplete tours exist
        #return ( tour, cost, mask)
    def reward_cumulation(self,rewards,t):
        '''Cumulates the rewards of the next n-1 steps at timestep t and multiplies them with the
            discount factor sum_over_n-1(gamma^n * r_n)'''

        stacked = torch.cat(rewards,axis=1)[:,t:t+(self.n)] #flag
        gammav = torch.torch.logspace(0,stacked.size(1)-1,base=self.gamma,steps = stacked.size(1),device=self.device) # Gamma vector
        cumulated = (gammav * stacked).sum(axis=1)
        return cumulated

    # Forward for learning
    def forward(self, features, last_visited, Ds, masks, current_depot, target = False):
        """Forwarding experience through encoder and decoder for learning"""
        if target == False:
            encoder = self.encoder
            decoder = self.decoder
        else:
            encoder = self.encoder_target
            decoder = self.decoder_target

        node_embeddings = encoder(features)

        uncompleted_tours = torch.ones(features.shape[0], dtype=torch.bool, device=self.device)
        #average embedding
        graph_embedding = torch.mean(node_embeddings, dim=1)

        #create the context
        context = self.problem.create_context(node_embeddings,
                                          graph_embedding,
                                          last_visited,
                                          Ds,
                                          features.shape[0],
                                          uncompleted_tours,
                                          current_depot) #second pre - last one is the current batch_size after not counting the tours that are already complete as we don't want to compute for those

        #run decoder
        Q_vals = decoder(context,
                         node_embeddings,
                         masks)

        return Q_vals

    def learn(self,batchsize, polyak_factor):
        '''Sample batch of experiences and learn on them'''
        self.optimizer.zero_grad()
        states,actions,rewards,next_states,weights = self.buffer.sample(batchsize,self.episodes_done)

        state_Qs = self.forward(states.graphs,states.last_visited,states.Ds,states.masks, states.current_depots)
        state_Qs = state_Qs.squeeze(1).gather(1, actions.long().view(-1,1)).squeeze()

        # Next state Qs for non final next states
        with torch.no_grad():

            ind = next_states.uncompleted
            next_state_Qs = self.forward(next_states.graphs[ind], next_states.last_visited[ind], next_states.Ds[ind], next_states.masks[ind], next_states.current_depots[ind], target = False)
            next_state_Qs_actions = torch.max(next_state_Qs.squeeze(1),1)[1]

            next_state_Qs = self.forward(next_states.graphs[ind], next_states.last_visited[ind], next_states.Ds[ind], next_states.masks[ind], next_states.current_depots[ind], target = True)
            next_state_Qs = next_state_Qs.squeeze(1).gather(1, next_state_Qs_actions.long().view(-1,1))

            Qs = torch.zeros((batchsize,1), device=self.device)
            Qs[ind] = next_state_Qs
            Qs = Qs.squeeze()

            y = (self.gamma ** self.n) * Qs + rewards


        #loss = self.criterion(state_Qs.float(),y.float()).unsqueeze(0)
        loss = (torch.pow(state_Qs.float()- y.float(),2) * weights).mean()
        loss.backward()

        # Store updated priorities into the buffer
        with torch.no_grad():
            td_error_up =  torch.pow((torch.abs(state_Qs.float()- y.float()) + self.epsilon_td),self.buffer.alpha)
            self.buffer.update_prios(td_error_up)

        #clip gradients
        nn.utils.clip_grad_norm_(itertools.chain(*[self.encoder.parameters(), self.decoder.parameters()]), self.max_grad_norm)
        self.optimizer.step()
        #self.warmup_scheduler.dampen()

        #update target network
        self.target_network_controller(polyak_factor)

        # log q loss, qs and targets
        self.logging.log_learn(loss, y, state_Qs, self.steps_done)
        self.logging.log_target_and_reward_distributions(self.steps_done, states, rewards, y)

    def validate(self, data = None, return_rewards=False,log_hist=False, batch_size=1024, sampling=(False,1)):
        '''Solve all validation instances with current model.'''
        with torch.no_grad():
            self.eval()
            if log_hist:
                log_dir = os.path.join('Q_value_analysis',datetime.datetime.now().strftime("%Y%m%d-%H%M%S_"+str(self.n)+"_"+str(self.gamma)))
                newwriter = SummaryWriter(log_dir)

            if not data is None:
                loader = DataLoader(data, batch_size)
            else:
                loader = DataLoader(self.validation_data, batch_size)

            q_vals, rewards = [], []  #only used with return_rewards True; Please only use it with single samples in the visualize_q_values_and_reward function
            all_tours, all_costs  = [], []

            max_tour_length=0
            for batch_ndx, sample in enumerate(loader):
                tours, uncompleted_tours, capacities, cost, current_depot,features = self.compute_static_components(sample)
                tours_remaining = True

                step = 0
                while tours_remaining:
                    #obtain the mask (need to do first now to mask out feasibility feature)
                    mask = self.problem.create_mask(tours,
                                                    self.graph_size,
                                                    capacities,
                                                    sample,
                                                    current_depot)
                    if step > 0:
                        features = self.problem.update_features(features, mask, sample[:,:,2],  tours, capacities, current_depot)
    
    
                    Q_vals = self.forward(features[uncompleted_tours], tours[uncompleted_tours], capacities[uncompleted_tours], mask[uncompleted_tours], current_depot[uncompleted_tours], target = False)
                    if log_hist:
                        if self.writer is not None:
                            newwriter.add_histogram("Q-value distribution",Q_vals[Q_vals!=-math.inf].view(-1),step)

                    if sampling[0]:
                        action = self.choose_action(Q_vals, 'boltzman', softmax_temperature=sampling[1])
                    else:
                        action = self.choose_greedy_action(Q_vals) # Choose !!Greedy!! action instead of normal choose_action


                     # Update last position feature in feature variable
                    last_pos = torch.zeros(uncompleted_tours.sum(),self.graph_size).scatter_(1,action,1)
                    features[:,:,3][uncompleted_tours] = last_pos

                    # Update demand feature
                    temp_act = torch.zeros(sample.shape[0],1,dtype=torch.long)
                    temp_act[uncompleted_tours] = action
                    features[:,:,2].scatter_(1,temp_act,0)

                    #update tour
                    tours, current_depot = self.problem.update_tour_plan(tours,
                                                              action,
                                                              uncompleted_tours,
                                                              current_depot)

                    #update capacities
                    capacities[uncompleted_tours] = self.problem.update_capacities(sample[uncompleted_tours],
                                                             action,
                                                             capacities[uncompleted_tours])

                    #update the costs
                    cost[uncompleted_tours], tour_cost = self.problem.update_cost(cost[uncompleted_tours],
                                                                                 sample[uncompleted_tours],
                                                                                 tours[uncompleted_tours])

                    if return_rewards:
                        q_val, action = torch.max(Q_vals ,2 ,keepdim= False)
                        q_vals.append(q_val)
                        rewards.append(tour_cost)

                    #update incomplete tour indices
                    uncompleted_tours = self.problem.update_unfinished_tour_indices(tours,
                                                                                         uncompleted_tours,
                                                                                         self.graph_size)
                    tours_remaining = uncompleted_tours.any()
                    step+=1
                if tours.size(1) > max_tour_length:
                    max_tour_length = tours.size(1)

                all_tours.append(tours)
                all_costs.append(cost)

            all_tours = [torch.cat((tour, torch.ones(tour.size(0), max_tour_length - tour.size(1),dtype=torch.long)*-1), dim=1) for tour in all_tours]

            all_tours = torch.cat(all_tours)
            all_costs = torch.cat(all_costs)
            self.train()
            
            if return_rewards:
                return all_tours, all_costs, q_vals, rewards
            else:
                return all_tours, all_costs
            
    def sampling(self, data = None, softmax_temperature=1., num_solutions=1024):
        if not data is None:
            loader = DataLoader(data, 1)
        else:
            loader = DataLoader(self.validation_data, 1)
        
        max_tour_length=0
        all_tours, all_costs  = [], []
        for idx, problem in enumerate(loader):
            problem = problem.repeat(num_solutions, 1, 1)
            tours, costs = self.validate(data=problem, sampling=(True, softmax_temperature))
            min_cost, ind = torch.min(costs, dim=0)
            # min_tour = tours[ind]
            # all_tours.append(min_tour)
            all_costs.append(min_cost)
            
            if tours.size(1) > max_tour_length:
                max_tour_length = tours.size(1)
        
        # all_tours = [torch.cat((tour, torch.ones(tour.size(0), max_tour_length - tour.size(1),dtype=torch.long)*-1), dim=1) for tour in all_tours]

        # all_tours = torch.cat(all_tours)
        all_costs = torch.cat(all_costs)
        return all_tours, all_costs
    
    def target_network_controller(self,target_network_update):
        for target_param, param in zip([*self.encoder_target.parameters(), *self.decoder_target.parameters()], [*self.encoder.parameters(), *self.decoder.parameters()]):
            target_param.data.copy_(target_network_update*target_param.data + param.data*(1.0 - target_network_update))

    def plot_solution(self,idx=None ,ax = None, show = True):
        '''Function to plot random validation solution for demonstration purposes'''
        if idx is None:
            idx = np.random.randint(0,self.validation_data.shape[0],1)
        else:
            idx = np.array([idx])
        data = self.validation_data[idx]
        tour,cost = self.validate(data=data)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        plot_vehicle_routes(data.squeeze(),tour.squeeze(),ax,visualize_demands=False, depot_num = self.num_depots, demand_scale=50, round_demand=True,plot_legend = False,epoch=self.episodes_done)
        if show:
            plt.show()

    def plot_n_solutions(self,n= 10,plots_per_row = 2,show = True):
        n = min(self.validation_data.shape[0],n)
        rows = int(math.ceil(n/plots_per_row))
        fig, ax = plt.subplots(rows,plots_per_row,figsize= (plots_per_row*10,rows*10))
        row = 0
        col = 0
        for plot in range(n):
            self.plot_solution(idx = plot, ax = ax[row][col], show = False)
            if col == plots_per_row -1:
                col = 0
                row = row +1
            else:
                col = col + 1
        if show:
            plt.show()
        else:
            return fig

    def write_n_plots(self, n = 10,tboard=True,file=False):
        # TODO plot few validation graphs plotimage()
        n = min(n, self.validation_data.shape[0])
        for plot in range(n):
            fig, ax = plt.subplots(figsize=(10, 10))
            self.plot_solution(idx = plot, ax = ax, show = False)
            if tboard:
                self.writer.add_figure('Validation tours/Tour'+ str(plot),fig,self.episodes_done)
            if file:
                fig.savefig("ValidationTours/Tour"+str(plot)+" "+str(self.episodes_done)+".png")

    def visualize_q_values_and_reward(self, idx=1, return_rewards=True):
        self.plot_solution(idx=idx)

        data = self.validation_data[torch.tensor([idx])]
        tours, cost, q_vals, rewards = self.validate(data=data,return_rewards=True)
        rewards = torch.stack(rewards).squeeze(1).squeeze(1)
        q_vals = torch.stack(q_vals).squeeze(1).squeeze(1)*-1
        tours = tours.squeeze(0)

        true_reward_til_end = torch.cumsum(rewards.flip(0), dim=0).flip(0)
        immediate_reward = rewards

        for i,val in enumerate(tours[1:]):
            if val == 0:
                c='wheat'
            else:
                c='ivory'
            plt.axvspan(i, i+1, facecolor=c, alpha=0.5)

        plt.plot(q_vals.cpu().detach().numpy(), label='Q-values')
        plt.plot(immediate_reward.cpu().detach().numpy(), label='immediate_reward')
        plt.plot(true_reward_til_end.cpu().detach().numpy(), label='true cumulative reward til end of episode')
        plt.legend()
        plt.title('q vals and true rewards for single instance')
        plt.xlabel('episode step')
        plt.show()
