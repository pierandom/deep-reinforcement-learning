import torch as th
import torch.nn as nn
from torch.distributions import Categorical, Normal

def mlp(sizes, activ, output_activ=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        a = [activ()] if i < len(sizes)-2 else [output_activ()]
        layers.extend([nn.Linear(sizes[i], sizes[i+1])] + a)
    return nn.Sequential(*layers)


class MLPCategoricalActor(nn.Module):
    """Stochastic Categorical Policy"""
    def __init__(self, obs_dim, act_dim, hid_dims, activ):
        super().__init__()
        self.net = mlp([obs_dim] + list(hid_dims) + [act_dim], activ)

    def forward(self, obs):
        policy = Categorical(logits=self.net(obs))
        return policy


class MLPGaussianActor(nn.Module):
    """Stochastic Gaussian Policy"""
    def __init__(self, obs_dim, act_dim, hid_dims, activ, log_std_limits=[-20, 2]):
        super().__init__()
        self.net = mlp([obs_dim] + list(hid_dims), activ, activ)
        self.mean_layer = nn.Linear(hid_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hid_dims[-1], act_dim)
        self.log_std_min, self.log_std_max = log_std_limits

    def forward(self, obs):
        hid = self.net(obs)
        mean = self.mean_layer(hid)
        log_std = th.clamp(self.log_std_layer(hid), self.log_std_min, self.log_std_max)
        std = th.exp(log_std)
        policy = Normal(mean, std)
        return policy


class MLPDeterministicActor(nn.Module):
    """Deterministic Policy"""
    def __init__(self, obs_dim, act_dim, hid_dims, activ, output_activ=nn.Identity):
        super().__init__()
        self.net = mlp([obs_dim] + list(hid_dims) + [act_dim], activ, output_activ)
    
    def forward(self, obs):
        return self.net(obs)


class MLPValueCritic(nn.Module):
    """Value V(s) Function"""
    def __init__(self, obs_dim, hid_dims, activ):
        super().__init__()
        self.net = mlp([obs_dim] + list(hid_dims) + [1], activ)
    
    def forward(self, obs):
        return self.net(obs).squeeze(1)


class MLPQualityCritic(nn.Module):
    """Quality Q(s,a) Function"""
    def __init__(self, obs_dim, act_dim, hid_dim, activ):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hid_dim) + [1], activ)
    
    def forward(self, obs, act):
        return self.net(th.cat([obs, act], dim=1)).squeeze(1)
