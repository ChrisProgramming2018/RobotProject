import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module, Linear
from torch.distributions import Distribution, Normal
from torch.nn.functional import relu, logsigmoid


class Actor(Module):
    def __init__(self, args, state_dim, action_dim):
        super().__init__()
        self.device = args.device
        self.log_std_min_max = (-20, 2)
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [256, 256], 2 * action_dim)
    
    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*self.log_std_min_max)
        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std, self.device)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob
    
    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)[None, :]
        action, _ = self.forward(obs)
        action = action[0].cpu().detach().numpy()
        return action



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

class TanhNormal(Distribution, device):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device), torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)
    
    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + logsigmoid(2 * pre_tanh) + logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result
    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh

