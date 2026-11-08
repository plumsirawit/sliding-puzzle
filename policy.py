from board import BoardState
import torch
from torch import nn
from torch.distributions.categorical import Categorical

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
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return Categorical(logits=logits)

    @torch.inference_mode()
    def sample_action(self, state_tensor, mask):
        logits = self.mlp(state_tensor).masked_fill(mask == 0, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def evaluate_actions(self, states, actions, masks):
        distribution = self.run_policy(states, masks)
        return distribution.log_prob(actions), distribution.entropy()


class Critic:
    def __init__(self, device):
        self.device = torch.device(device)
        self.mlp = MLP(256, 1, 3, 500, nn.ReLU()).to(self.device)
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=3e-4, foreach=False)
    
    def run(self, state_tensor):
        out = self.mlp(state_tensor)
        return out