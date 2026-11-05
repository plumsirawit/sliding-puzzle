from board import BoardState
import torch
from torch import nn
from torch.distributions.categorical import Categorical
import numpy as np

class MLP(nn.Module):
    def __init__(self, nInput, nOutput, nLayer, nHidden, act_fn):
        super(MLP, self).__init__()
        layers = [] 
        
        layers.append(nn.Linear(nInput, nHidden))
        for l in range(1, nLayer):
            layers.append(act_fn)
            layers.append(nn.Linear(nHidden, nHidden))
        layers.append(act_fn)
        layers.append(nn.Linear(nHidden, nOutput))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Policy:
    def __init__(self):
        self.mlp = MLP(16, 4, 3, 20, nn.ReLU())

    def run_policy(self, state_tensor, allowed=None):
        logits = self.mlp(state_tensor)
        if allowed is not None:
            for i in range(4):
                if i not in allowed:
                    logits[i] = -torch.inf
        return Categorical(logits=logits)

    def get_loss(self, states, actions, weights):
        states_tensor = torch.as_tensor(np.array([state.board.flatten() for state in states]), dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
        logp = self.run_policy(states_tensor).log_prob(actions_tensor)
        return -(logp * weights_tensor).mean()
