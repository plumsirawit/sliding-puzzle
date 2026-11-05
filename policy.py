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

    def run_policy(self, state_tensor, mask=None):
        logits = self.mlp(state_tensor)
        probs = torch.softmax(logits, -1)
        if mask is not None:
            probs = probs * mask
        return Categorical(probs)

    def get_loss(self, states, actions, weights, allowed_actions_masks):
        states_tensor = torch.as_tensor(np.array([state.board.flatten() for state in states]), dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32)
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
        masks_tensor = torch.as_tensor(np.array(allowed_actions_masks), dtype=torch.float32)
        logp = self.run_policy(states_tensor, masks_tensor).log_prob(actions_tensor)
        return -(logp * weights_tensor).mean()
