# -*- coding: utf-8 -*-
"""
Model Class with Structure2Vec and Deep Q-Net component
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class s2v_DQN(nn.Module):
    def __init__(self,p,size,device=torch.device("cuda")):
        super(s2v_DQN, self).__init__()
        self.p = p.data.item()
        self.size = size
        self.device=device
        
        # structure2vec layers
        self.lin1 = nn.Linear(1,out_features = self.p)
        self.lin2 = nn.Linear(self.p,self.p)
        self.lin3 = nn.Linear(self.p,self.p)
        self.lin4 = nn.Linear(1,self.p)
        
        # q-learning layers
        self.lin5 = nn.Linear(2*self.p,1)
        self.lin6 = nn.Linear(self.p,self.p)
        self.lin7 = nn.Linear(self.p,self.p)
        
    def q_net(self, embeddings):
        
        # Q-learning
        comp7 = self.lin7(embeddings)
        comp6 = self.lin6(torch.sum(embeddings,1)).view(embeddings.shape[0],-1,self.p)*torch.ones(embeddings.shape[0],self.graph.nodes,self.p, device = self.device)
        comp5 = self.lin5(F.relu(torch.cat((comp6,comp7),2)))
        return comp5.squeeze()
        
    def s2v(self, features,T,dist_indices):
  
        embeddings = torch.zeros((features.shape[0],self.size,self.p),device=self.device)
        features = features.view(-1,self.size,1)

        for t in range(T):
            comp1 = self.lin1(features)        
            comp2 = self.lin2((torch.sum(embeddings,1).unsqueeze(1).expand(-1,self.size,-1)-embeddings))
            comp4 = self.lin3(torch.sum(F.relu(self.lin4(self.graph.distances[dist_indices].unsqueeze(-1))),2))
            embeddings = F.relu(comp1+comp2+comp4)
        return embeddings
        
    def forward(self,state,graph,dist_indices,T=4):
        self.graph = graph
        embeddings = self.s2v(state,T,dist_indices.long())
        mean = torch.mean(embeddings, 1, keepdim=True).expand(-1,self.size,-1)
        std = torch.std(embeddings, 1, keepdim=True).expand(-1,self.size,-1)
        mask = std.nonzero(as_tuple=True)
        embeddings_copy = torch.zeros_like(embeddings, device = self.device)
        embeddings_copy[mask] = (embeddings[mask] - mean[mask])/std[mask]
        qs = self.q_net(embeddings_copy)
        return qs
