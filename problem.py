
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:59:16 2020
@author: timde
"""
import torch


class Problem(object):
    """This class contains mostly problem dependent methods like the masking procedure or the decision when a tour is finalized and the corresponding indexing. Potentially also the reward function.
    
    Currently supported:
        Capacitated Vehicle Routing Problem with 'cvrp' as problem type string
        Beta Multi-Depot Routing 'mdvrp'
    """

    def __init__(self, problem_type, num_depots, device=torch.device("cuda")):
        self.problem_type = problem_type
        self.device=device
        self.number_depots = num_depots
    
    def create_initial_tour_plan(self, batch_size):
        if self.problem_type == 'cvrp':
            #tour starts out with an initial tensor with zeroes of dimension (batchsize, 1) as the first node is the depot and then a vector with the new node indices will be appended at each timestep
            current_depot = torch.zeros((batch_size, 1), dtype = torch.int64, device=self.device)
            return current_depot, current_depot
        
        if self.problem_type == 'mdvrp':    #added a tracker to track the current origin depots
            
            #generate a tour start depot tensor to keep track of current origin depot
            current_depot = torch.ones((batch_size, 1), dtype = torch.int64, device=self.device)  * -1
            tour = torch.empty((batch_size, 0), dtype = torch.int64, device=self.device)
            #generate dummy embedding for depot
            #self.current_dEmbedding_index = torch.zeros((batch_size, 1), dtype = torch.int64)
            
            #tour starts out with an initial tensor with zeroes of dimension (batchsize, 1) as the first node is the depot and then a vector with the new node indices will be appended at each timestep
            return tour, current_depot
        
    def update_tour_plan(self, tour, action, uncompleted_tours, current_depot):
        if self.problem_type=='cvrp':
            # default action is -1 as this will be inserted for all already complete tours
            default_action = torch.ones((uncompleted_tours.size()[0], 1), dtype=torch.int64,device=self.device) * -1
            #for all incomplete tours replace the action with the correct one
            default_action[uncompleted_tours] = action
            #append actions
            return torch.cat((tour, default_action), dim = 1), current_depot
        
        if self.problem_type=='mdvrp': #included an updater to current depot tracker
            # default action is -1 as this will be inserted for all already complete tours
            default_action = torch.ones((uncompleted_tours.size()[0], 1), dtype=torch.int64, device=self.device) * -1
            #for all incomplete tours replace the action with the correct one
            default_action[uncompleted_tours] = action
        
            current_depot_uncompleted = current_depot[uncompleted_tours]
            
            #check if the action is a depot and if this is the same depot that we have now
            #update_depot_mask_sameDepot = torch.logical_and(current_depot_uncompleted == action , action < self.number_depots)
            update_depot_mask = action < self.number_depots
            same_depot_mask = current_depot_uncompleted == action
            
            #update all depot actions
            current_depot_uncompleted[update_depot_mask] = action[update_depot_mask]
            
            #update all returns to current depot
            current_depot_uncompleted[same_depot_mask] = -1
            
            current_depot[uncompleted_tours] = current_depot_uncompleted
            #assign -1 to the same depot actions, meaning we returned and no longer have an origin
            #current_depot[uncompleted_tours][update_depot_mask_sameDepot] = -1
            
            #if its not the same depot, replace it with the current new depot , replace the embedding index pointer
            #current_depot[uncompleted_tours] = current_depot[uncompleted_tours].masked_scatter_(~update_depot_mask_sameDepot, action)
            #current_dEmbedding_index[uncompleted_tours].masked_scatter_(~update_depot_mask_sameDepot, action)

            #append actions
            return torch.cat((tour, default_action), dim = 1), current_depot

        
    def update_unfinished_tour_indices(self, tour, uncompleted_tours, graph_size):
        if self.problem_type == 'cvrp':
            """This method relies on the sum of all indices being the gaussian sum for n= graphsize-1. 
            At each timestep we check for all yet uncompleted tours whether they have been completed by checking whether the sum of the tour is equal to the sum of the natural numbers up to the maximum possible index.
            The second condition is to check whether the last node is the depot. If both conditions are fulfilled we add the tour to the completed tours and always only append -1 as the action in the tour plan."""
            
            gaussian_sum = ((graph_size-1)**2 + (graph_size-1))/2
            bool_index = torch.logical_and(tour.sum(1) == gaussian_sum, tour[:,-1]==0)
            #set all uncompleted tours which have completed this timestep to False
            uncompleted_tours[bool_index] = False
            return uncompleted_tours

        
        if self.problem_type == 'mdvrp': #making sure it accounts for all depots and not just the 0 depot
            """This method relies on the sum of all indices being the gaussian sum for n= graphsize-1. 
            At each timestep we check for all yet uncompleted tours whether they have been completed by checking whether the sum of the tour is equal to the sum of the natural numbers up to the maximum possible index.
            The second condition is to check whether the last node is the depot. If both conditions are fulfilled we add the tour to the completed tours and always only append -1 as the action in the tour plan."""
            
            
            #developed version of below commented gaussian sum formula for n = graphsize -1 and m = number depots
            
            gaussian_sum = ((graph_size- self.number_depots)*(graph_size - 1 + self.number_depots))/2
            #gaussian_sum = ((graph_size - 1)*(graph_size -1 +1) - (self.number_depots-1)*self.number_depots)/2

            temp = torch.masked_fill(tour, tour < self.number_depots, 0)
            
            bool_index = torch.logical_and(temp.sum(1) == gaussian_sum, tour[:,-1] < self.number_depots)
            #set all uncompleted tours which have completed this timestep to False
            uncompleted_tours[bool_index] = False
            
            #self.unfinished_current_depot = self.current_depot[uncompleted_tours]
            
            return uncompleted_tours

    
    def create_initial_capacities(self, batch_size):
        if self.problem_type == 'cvrp':
            #all vehicles have initial capacity one
            return torch.ones((batch_size, 1),device=self.device)
        
        if self.problem_type == 'mdvrp': #nothing changed
            #all vehicles have initial capacity one
            return torch.ones((batch_size, 1), device=self.device)
        
        
    def update_capacities(self, batch, action, old_vehicle_capacities):
        if self.problem_type == 'cvrp':
            #get list of capacities only
            all_node_capacities = batch[:,:,2]
            #get the capacity of the selected node in each batch element
            selected_node_capacities = all_node_capacities.gather(1, action)
            #new = old - diff
            new_vehicle_capacities = old_vehicle_capacities - selected_node_capacities
            #set capacities of vehicles that just have returned to the depot to 1.
            new_vehicle_capacities[action==0] = 1.
            return new_vehicle_capacities
        
        
        if self.problem_type == 'mdvrp': #making sure it accounts for all depots and not just the 0 depot
            #get list of capacities only
            all_node_capacities = batch[:,:,2]
            #get the capacity of the selected node in each batch element
            selected_node_capacities = all_node_capacities.gather(1, action)
            #new = old - diff
            new_vehicle_capacities = old_vehicle_capacities - selected_node_capacities
            #set capacities of vehicles that just have returned to the depot to 1.
            new_vehicle_capacities[action < self.number_depots] = 1.
            return new_vehicle_capacities        

    
    def create_initial_cost(self, batch_size):
        if self.problem_type == 'cvrp':
            return torch.zeros((batch_size, 1),device=self.device)

        if self.problem_type == 'mdvrp': #nothing changed
            return torch.zeros((batch_size, 1),device=self.device)
        
    def update_cost(self, cost, batch, tour, reward_scaling=False):
        if self.problem_type == 'cvrp':
            #get the indices of the last two nodes of the tour to calculate the distance between them
            index = tour[:,-2:].unsqueeze(2)
            #retrieve the x and y coordinates seperately and concatenate them; I found no better way
            x_coordinates = batch[:,:,0:1].gather(1, index)
            y_coordinates = batch[:,:,1:2].gather(1, index)
            node_pairs = torch.cat((x_coordinates,y_coordinates), dim=2)
            #calculate the distance of the corresponding pairs
            distances = torch.norm(node_pairs[:,0:1,:] - node_pairs[:,1:2,:], p=2, dim=2)
            #add the distances to the current cost and return
            cost = cost + distances
            if reward_scaling:
                distances = distances / batch.size(1)
            return cost, distances

        if self.problem_type == 'mdvrp': #if the origin and destination are depots, set the delta cost to 0
            if tour.size(1) > 1:
                #get the indices of the last two nodes of the tour to calculate the distance between them
                index = tour[:,-2:].unsqueeze(2)
                #retrieve the x and y coordinates seperately and concatenate them; I found no better way
                x_coordinates = batch[:,:,0:1].gather(1, index)
                y_coordinates = batch[:,:,1:2].gather(1, index)
                node_pairs = torch.cat((x_coordinates,y_coordinates), dim=2)
                #calculate the distance of the corresponding pairs
                distances = torch.norm(node_pairs[:,0:1,:] - node_pairs[:,1:2,:], p=2, dim=2)
                #add the distances to the current cost and return
                
                #if the origin and destination are depots, set the delta cost to 0
                mask_both_depots = torch.logical_and(index[:,0] < self.number_depots, index[:,1] < self.number_depots)
                distances[mask_both_depots] = 0
                #add the distances to the current cost and return
                cost = cost + distances
                if reward_scaling:
                    distances = distances / batch.size(1)
            else:
                distances = torch.zeros_like(cost)
                
            return cost, distances
    
    def create_mask(self, tour, graph_size, capacities, batch, current_depot):
        if self.problem_type == 'cvrp':
            #initialize mask
            mask = torch.zeros(batch.size(0), graph_size, dtype=torch.bool,device=self.device) 
            
            #need to remove duplicate 0 indices to prevent that tour has more values than mask along dimension 1 which breaks scatter; torch.unique does not work here
            sorted_tensor, indices = tour.sort(dim=1, descending=True)
            sorted_tensor = sorted_tensor[:,0:graph_size]
            
            #mask all positions that have been visited
            mask.scatter_(1,sorted_tensor,True)
            #mask all positions where the capacity conditions are not met
            mask[batch[:,:,2] >= capacities] = True 
            #unmask all depots where the last node was not the depot
            mask[:,0] = tour[:,-1] == False 
            return mask.unsqueeze(1)

        if self.problem_type == 'mdvrp': #added masking logic for mdvrp cases
            #initialize mask
            mask = torch.zeros(batch.size(0), graph_size, dtype=torch.bool) 
            
            #need to remove duplicate 0 indices to prevent that tour has more values than mask along dimension 1 which breaks scatter; torch.unique does not work here
            sorted_tensor, indices = tour.sort(dim=1, descending=True)
            sorted_tensor = sorted_tensor[:,0:graph_size]
            
            #mask all positions that have been visited
            mask.scatter_(1,sorted_tensor,True)
        
            #mask all positions where the capacity conditions are not met
            mask[batch[:,:,2] >= capacities] = True 
            
            if tour.size(1) != 0:
                #mask all depots 
                mask[:, 0:self.number_depots] = True
                
                #if we're starting a new tour, unmask all depot, we can pick any
                mask_resetDepot = current_depot.reshape(-1) == -1
                mask_indices = mask_resetDepot.nonzero().squeeze(1)
                mask[mask_indices, 0:self.number_depots] = False
                mask[mask_indices, self.number_depots:] = True
                #if tour (-1) same as current depot, mask all depots
            
                mask_maskAll = tour[:, -1].T == current_depot.reshape(-1)
                mask_indices = mask_maskAll.nonzero().squeeze(1)
                mask[mask_indices, 0:self.number_depots] = True


            
                #else unmask current depot
                mask_notReset_notMaskAll = torch.logical_and(~mask_maskAll, ~mask_resetDepot)
                #inplace doesnt work here for some reason
                mask[mask_notReset_notMaskAll] = mask[mask_notReset_notMaskAll].scatter_(1, current_depot[mask_notReset_notMaskAll], False)
            else:
                mask[:, self.number_depots:] = True
            
            return mask.unsqueeze(1)
        
    def create_context(self, node_embeddings, graph_embedding, tour, rem_caps, batch_size, uncompleted_tours, current_depot):
        if self.problem_type == 'cvrp':
            #get last node embedding for each batch element
            last_embed = node_embeddings[torch.arange(0, batch_size), tour[:,-1], :]
            #
            # tour2 = tour.clone()
            # tour2[tour>0] = 1
            # percentage_of_customers_still_left = 1 - (tour2.sum(1,keepdim=True).float() / (node_embeddings.size(1) -1))
            #concatanate and return
            
            # con = torch.cat((graph_embedding, rem_caps, percentage_of_customers_still_left), dim = 1)# removed last emedding
            con = torch.cat((last_embed, graph_embedding, rem_caps), dim = 1)
            
            return con
    
        if self.problem_type == 'mdvrp': #nothing changed
            #get last node embedding for each batch element
            if tour.size(1)!=0:
                last_embed = node_embeddings[torch.arange(0, batch_size), tour[:,-1], :]
                tour2 = tour.clone()
                no_depot_index = tour2[:,-1] == -1
                if tour.size(1) > 1:
                    tour2[no_depot_index, -1] = tour2[no_depot_index, -2]
                depot_embed = node_embeddings[torch.arange(0, batch_size), current_depot.squeeze(), :]
                #depot_embed = node_embeddings[torch.arange(0, batch_size), self.current_dEmbedding_index[uncompleted_tours][:,0]]
                
                #depot_embed = torch.gather(node_embeddings,1 , self.current_depot)
                #depot_embed.shape = torch.index_select(node_embeddings,1 , self.current_depot.reshape(-1))
                            
                #concatanate and return    
                #con =torch.cat((graph_embedding, rem_caps, last_embed, depot_embed), dim = 1)
                # tour2 = tour.clone()
                # tour2[tour>0] = 1
                # percentage_of_customers_still_left = 1 - (tour2.sum(1,keepdim=True).float() / (node_embeddings.size(1) -1))
                #temporarily using the original context to check for bugs
                
                # con = torch.cat((last_embed, graph_embedding, rem_caps, depot_embed, percentage_of_customers_still_left), dim = 1)
                con = torch.cat((graph_embedding, rem_caps, last_embed, depot_embed), dim = 1)
            else:
                con = torch.cat((graph_embedding, rem_caps), dim = 1)
            return con
    
    def create_initial_features(self, batch):
        if self.problem_type == 'cvrp':
            # Create batch copy with new features and updated demands
            features = batch.detach().clone()
            # add last position (which is the depot in CVRP case and nothing in MDVRP case)
            last_pos = torch.zeros(1,batch.shape[1],1)
            last_pos[0,0,0] = 1
            features = torch.cat((features,last_pos.repeat(batch.shape[0],1,1)),axis=2)
            
            
            # add feature for feasibility - is this node feasible this iteration?
            feasible = torch.ones(last_pos.size())
            feasible[0,0,0] = 0
            features = torch.cat((features,feasible.repeat(batch.shape[0],1,1)),axis=2)
            
            #capacity infeasible
            features = torch.cat((features,torch.ones(last_pos.size()).repeat(batch.shape[0],1,1)),axis=2)
            #already in tour infeasible
            features = torch.cat((features,torch.ones(last_pos.size()).repeat(batch.shape[0],1,1)),axis=2)
            
            
        if self.problem_type == 'mdvrp':
            # Create batch copy with new features and updated demands
            features = batch.detach().clone()
            # add last position (which is the depot in CVRP case and nothing in MDVRP case)
            last_pos = torch.zeros(1,batch.shape[1],1)
            features = torch.cat((features,last_pos.repeat(batch.shape[0],1,1)),axis=2)
            features = torch.cat((features,last_pos.repeat(batch.shape[0],1,1)),axis=2)
            
            
            # add feature for feasibility - is this node feasible this iteration?
            feasible = torch.zeros(last_pos.size())
            feasible[0,0:self.number_depots,0] = 1
            features = torch.cat((features,feasible.repeat(batch.shape[0],1,1)),axis=2)
            
            #capacity infeasible
            features = torch.cat((features,torch.ones(last_pos.size()).repeat(batch.shape[0],1,1)),axis=2)
            #already in tour infeasible
            features = torch.cat((features,torch.ones(last_pos.size()).repeat(batch.shape[0],1,1)),axis=2)
            
            #closest depot index
            # min_dist, depot_index = torch.min(torch.norm(features[:,0:self.number_depots,[0,1]].unsqueeze(1) - features[:,:,[0,1]].unsqueeze(2), p=2, dim=3), dim=2)
            # features = torch.cat((features,depot_index.unsqueeze(2).float()),axis=2)
                
        return features
    
    def update_features(self, features, mask, demands, tours, capacities, current_depot):
        if self.problem_type == 'cvrp':
            current_position = tours[:,-1]
            current_position[current_position < 0] = 0
            current_position = current_position.unsqueeze(1)
            # use mask to create last feature
            features[:,:,4] = (~mask).squeeze(1).int()
            
                            # Update last position feature in feature variable
            last_pos = torch.zeros(features.size(0),features.size(1)).scatter(1,current_position,1)
            features[:,:,3] = last_pos

            # Update demand feature
            features[:,:,2].scatter_(1,current_position,0)
            
            #update infeasible because already inserted
            temp_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool,device=self.device)
            sorted_tensor, indices = tours.sort(dim=1, descending=True)
            sorted_tensor = sorted_tensor[:,0:features.size(1)]
            temp_mask.scatter_(1,sorted_tensor,True)
            temp_mask[:,0]= False
            temp_mask = ~temp_mask
            features[:,:,5] = temp_mask.float()
            
            #update infeasible because capacity
            features[:,:,6] = ~(demands > capacities)
            
        if self.problem_type == 'mdvrp':
            current_position = tours[:,-1]
            current_position[current_position < 0] = 0
            current_position = current_position.unsqueeze(1)
            # use mask to create last feature
            features[:,:,5] = (~mask).squeeze(1).int()
            
            # Update last position feature in feature variable
            last_pos = torch.zeros(features.size(0),features.size(1)).scatter(1,current_position,1)
            features[:,:,3] = last_pos
            
            #update current_depot
            binary_depot = torch.zeros(features.size(0),features.size(1))
            depot_inds = (current_depot >=0).squeeze(1)
            binary_depot[depot_inds] = binary_depot[depot_inds].scatter(1,current_depot[depot_inds],1)
            features[:,:,4] = binary_depot

            # Update demand feature
            features[:,:,2].scatter_(1,current_position,0)
            
            #update infeasible because already inserted
            temp_mask = torch.zeros(features.size(0), features.size(1), dtype=torch.bool,device=self.device)
            sorted_tensor, indices = tours.sort(dim=1, descending=True)
            sorted_tensor = sorted_tensor[:,0:features.size(1)]
            temp_mask.scatter_(1,sorted_tensor,True)
            temp_mask[:,0:self.number_depots]= False
            temp_mask = ~temp_mask
            features[:,:,6] = temp_mask.float()
            
            #update infeasible because capacity
            features[:,:,7] = ~(demands > capacities)
            
        return features
    

# test example for mask creation
# graph_size=8
# batch_size=2
# tour = torch.tensor([[0,7,5,0,3,0,1,0,2],[0,3,4,6,0,1,0,2,0]])
# batch = torch.rand((batch_size,graph_size,3))
# qvalues = torch.rand((batch_size,1,graph_size))
# capacities = torch.tensor([[0.6],[0.7]])
# current_cost = torch.zeros((batch_size, 1))

# problem = Problem('cvrp')
# mask = problem.create_mask(tour, graph_size, capacities, batch)

# import math
# qvalues[mask] = -math.inf

# _, action = torch.max(qvalues, dim = 2)
