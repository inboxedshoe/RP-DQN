import torch
from torch import nn
import math

class Normalization(nn.Module):
    """Supported normalization: batch and instance. All other strings default to no normalization"""
    def __init__(self, embed_dim, normalization):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'inst': nn.InstanceNorm1d
        }.get(normalization, None)
        
        if normalizer_class is not None:
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def forward(self, input):
        if hasattr(self, 'normalizer'):
            if isinstance(self.normalizer, nn.BatchNorm1d):
                #batch norm expects (batch dim, input dim, sequence length)
                return self.normalizer(input)
            elif isinstance(self.normalizer, nn.InstanceNorm1d):
                #instance norm expects (batch dim, input dim, sequence length)
                return self.normalizer(input)
        else:
            return input

class self_attention_block(nn.Module):
    def __init__(self,  embed_dim, intermediate_dim, num_heads, normalization):
        super(self_attention_block, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.normalization1 = Normalization(embed_dim, normalization)
        self.elementwise_linear = nn.Linear(embed_dim, intermediate_dim)
        self.elementwise_linear2 = nn.Linear(intermediate_dim, embed_dim)
        self.activation = nn.ReLU()
        self.normalization2 = Normalization(embed_dim, normalization)

    def forward(self, x):
        #MultiHead Attention expects(sequence length, batch dim, input dim)
        out = self.multihead_attn(x, x, x, need_weights=False)[0] + x #addition is the residual connection

        #batch norm expects (batch dim, input dim, sequence length)
        out = self.normalization1(out.permute(1,2,0))

        #elementwise linear expects (N,∗,H_in)
        out = out.transpose(1,2)
        out = self.elementwise_linear2(self.activation(self.elementwise_linear(out))) + out

        #batch norm expects (batch dim, input dim, sequence length)
        out = self.normalization2(out.transpose(2,1))

        return out.permute(2,0,1)

class node_encoder(nn.Module):
    def __init__(self, num_depots, embed_dim, intermediate_dim, num_heads, normalization, num_features_node = 3):
        super(node_encoder, self).__init__()
        self.num_depots = num_depots

        #initial linear embedding
        self.lin_customer = nn.Linear(num_features_node ,embed_dim)
        self.lin_depot = nn.Linear(num_features_node ,embed_dim)

        self.sa_block1 = self_attention_block(embed_dim, intermediate_dim, num_heads, normalization)
        self.sa_block2 = self_attention_block(embed_dim, intermediate_dim, num_heads, normalization)
        self.sa_block3 = self_attention_block(embed_dim, intermediate_dim, num_heads, normalization)

    def forward(self, x):
        #elementwise linear expects (N,∗,H_in)
        linear_depot_embedding = self.lin_depot(x[:,0:self.num_depots,:])
        linear_customer_embedding = self.lin_customer(x[:,self.num_depots:,:])
        #concat again
        out = torch.cat((linear_depot_embedding, linear_customer_embedding),1)

        #SA blocks expect (sequence length, batch dim, input dim)
        out = out.transpose(1,0)
        out = self.sa_block3(self.sa_block2(self.sa_block1(out)))

        #change shape again to (batch dim, sequence length, embedding dim)
        return out.transpose(1,0)

class decoder_module(nn.Module):
    def __init__(self,  num_depots, embed_dim, num_heads, inner_masking=True):
        super(decoder_module, self).__init__()
        #set inner masking
        self.inner_masking = inner_masking
        self.num_depots=num_depots

        #mapping the context features to the embed_dim required
        if num_depots == 1:
            self.context_projection = nn.Linear(embed_dim*2 + 1, embed_dim, bias=False) #try change
        else: 
            self.context_projection = nn.Linear(embed_dim*3 + 1, embed_dim, bias=False) #try change
            self.W_placeholder = nn.Parameter(nn.init.xavier_normal_(torch.empty((embed_dim*2, 1), requires_grad=True)))   #placeholder for last node embedding in case 

        #giving number of heads and expected dimension
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        #setting up compatibility
        self.WQ = nn.Parameter(nn.init.xavier_normal_(torch.empty((embed_dim, embed_dim), requires_grad=True)))
        self.Wk = nn.Parameter(nn.init.xavier_normal_(torch.empty((embed_dim, embed_dim), requires_grad=True)))

    def forward(self, context, node_embeddings, mask):
        if self.num_depots > 1 and context.size(1)==129:
            context = torch.cat((context, self.W_placeholder.repeat(1,context.shape[0]).transpose(1,0)), axis=1)
        
        #MultiHead Attention expects(sequence length, batch dim, input dim)
        context = self.context_projection(context)

        #fixing dimensions for MHA
        context = context.unsqueeze(0)

        node_embeddings = node_embeddings.permute(1,0,2)
        #mask = mask.permute(1,0,2)

        # MHA with query = context, and k,v = encoder output
        if self.inner_masking:
            context = self.multihead_attn(context, node_embeddings, node_embeddings, key_padding_mask=mask.squeeze(1), need_weights=False)[0]
        else:
            context = self.multihead_attn(context, node_embeddings, node_embeddings, need_weights=False)[0]
            
        
        Key = torch.matmul(self.Wk, node_embeddings.transpose(2,1)).transpose(0,2)
        Query = torch.matmul(context.transpose(0,1), self.WQ)

        Q_values = torch.matmul(Query,Key)
        Q_values[mask] = -math.inf
        return Q_values
