import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, use_layernorm=False, max_action=1):
        """
        """
        super(Actor, self).__init__()
        self.use_layernorm = use_layernorm
        self.layer_1 = nn.Linear(state_dim, 300)
        self.layer_2 = nn.Linear(300, 200)
        self.layer_3 = nn.Linear(200, action_dim)
        self.max_action = max_action    
        self.layer_norm1 = nn.LayerNorm(300)
        self.layer_norm2 = nn.LayerNorm(200)
    def forward(self, obs):        
        x = F.relu(self.layer_1(obs))
        if self.use_layernorm:
            x = self.layer_norm1(x)
        x = F.relu(self.layer_2(x))
        if self.use_layernorm:
            x = self.layer_norm2(x)
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x




class CNNCritic(nn.Module):
    def __init__(self, D_obs, state_dim, action_dim, args, D_out=200,conv_channels=[16, 32], kernel_sizes=[8, 4], strides=[4,2], use_layernorm=False):
        super(CNNCritic, self).__init__()
        # Defining the first Critic neural network
        channels = args.history_length
        self.use_layernorm = use_layernorm
        self.conv_1 =  torch.nn.Conv2d(channels, conv_channels[0], kernel_sizes[0], strides[0])
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 =  torch.nn.Conv2d(conv_channels[0], conv_channels[1], kernel_sizes[1], strides[1])
        self.relu_2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.Linear  = torch.nn.Linear(2592, D_out)
        self.relu_3 = torch.nn.ReLU()

        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)
        self.layer_norm1 = nn.LayerNorm(400)
        self.layer_norm2 = nn.LayerNorm(300)

    def forward(self, obs, u):
        xu = torch.cat([obs, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        if self.use_layernorm:
            x1 = self.layer_norm1(x1)
        x1 = F.relu(self.layer_2(x1))
        if self.use_layernorm:
            x1 = self.layer_norm2(x1)
        x1 = self.layer_3(x1)
        
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        if self.use_layernorm:
            x2 = self.layer_norm1(x2)
        x2 = F.relu(self.layer_5(x2))
        if self.use_layernorm:
            x2 = self.layer_norm2(x2)
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, obs, u):
        xu = torch.cat([obs, u], 1)
        x1 = F.relu(self.layer_1(xu))
        if self.use_layernorm:
            x1 = self.layer_norm1(x1)
        x1 = F.relu(self.layer_2(x1))
        if self.use_layernorm:
            x1 = self.layer_norm2(x1)
        x1 = self.layer_3(x1)
        return x1

    def create_vector(self, obs):
        obs_shape = obs.size()
        if_high_dim = (len(obs_shape) == 5)
        if if_high_dim: # case of RNN input
            obs = obs.view(-1, *obs_shape[2:])  
        x = self.conv_1(obs)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.flatten(x)
        obs = self.relu_3(self.Linear(x)) 

        if if_high_dim:
            obs = obs.view(obs_shape[0], obs_shape[1], -1)
        return obs

