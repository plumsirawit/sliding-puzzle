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
    def __init__(self, lr, device):
        self.device = torch.device(device)
        self.mlp = MLP(256, 4, 3, 500, nn.ReLU()).to(self.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr, foreach=False)

    def run_policy(self, state_tensor, mask=None):
        logits = self.mlp(state_tensor)
        probs = torch.softmax(logits, -1)
        if mask is not None:
            probs = probs * mask
        return Categorical(probs)

    @torch.inference_mode()
    def sample_action(self, state_tensor, mask):
        logits = self.mlp(state_tensor).masked_fill(mask == 0, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def get_loss(self, states, actions, advantages, allowed_actions_masks):
        states_tensor = torch.stack([state.to_tensor() for state in states]).to(self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        masks_tensor = torch.as_tensor(
            np.array(allowed_actions_masks), dtype=torch.float32, device=self.device
        )
        logp = self.run_policy(states_tensor, masks_tensor).log_prob(actions_tensor)
        return -(logp * advantages_tensor).mean()


class Critic:
    def __init__(self, device):
        self.device = torch.device(device)
        self.mlp = MLP(256, 1, 3, 500, nn.ReLU()).to(self.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=3e-4, foreach=False)
    
    def run(self, state_tensor):
        out = self.mlp(state_tensor)
        return out