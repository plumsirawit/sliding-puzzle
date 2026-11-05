from policy import Policy
from board import BoardState
import numpy as np
import torch

class Trainer:
    def __init__(self, lr):
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.mlp.parameters(), lr=lr)

    def sample_initial_state(self):
        return BoardState(np.array([
            [1, 2, 8, 3],
            [5, 7, 10, 4],
            [9, 6, 11, 0],
            [13, 14, 15, 12]
        ]))

    def train_one_epoch(self, batch_size, episode_limit, verbose=False):
        total_length = 0
        batch_states = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lengths = []
        while total_length < batch_size:
            episode_states = [self.sample_initial_state()]
            episode_actions = []
            episode_rewards = []
            if verbose:
                print('[LOG] Starting Episode', len(batch_returns), 'with state', episode_states[-1])
            while not episode_states[-1].is_solved() and len(episode_states) < episode_limit:
                cat = self.policy.run_policy(torch.as_tensor(episode_states[-1].board.flatten(), dtype=torch.float32))
                action = cat.sample().item()
                if verbose:
                    print('[LOG] Current state is')
                    print(episode_states[-1])
                    print('[LOG] Sampled action is', 'ULDR'[action])
                state = episode_states[-1].copy()
                reward = state.perform_action(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
            episode_total_rewards = sum(episode_rewards)
            batch_returns.append(episode_total_rewards)
            episode_length = len(episode_rewards)
            batch_lengths.append(episode_length)
            batch_weights += [episode_total_rewards] * episode_length
            total_length += episode_length
            batch_states += episode_states[1:]
            batch_actions += episode_actions
        self.optimizer.zero_grad()
        batch_loss = self.policy.get_loss(batch_states, batch_actions, batch_weights)
        batch_loss.backward()
        self.optimizer.step()
        if verbose:
            print('[LOG] Batch end; loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                (batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))
        return batch_loss, batch_returns, batch_lengths
            