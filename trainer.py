from policy import Policy
from board import BoardState
import numpy as np
import torch
import random

class Trainer:
    def __init__(self, lr):
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.mlp.parameters(), lr=lr)
        random.seed(42)

    def sample_initial_state(self):
        MAX_SCAFFOLD_DIST = 10
        state = BoardState(np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ]))
        chosen_dist = random.randint(0, MAX_SCAFFOLD_DIST)
        for i in range(chosen_dist):
            k = random.choice(state.actions())
            state.perform_action(k)
        while state.is_solved():
            k = random.choice(state.actions())
            state.perform_action(k)
        return state

    def reward_to_go(self, rewards):
        n = len(rewards)
        reverse_quicksum = np.zeros_like(rewards)
        for i in reversed(range(n)):
            reverse_quicksum[i] = rewards[i] + (reverse_quicksum[i+1] if i+1 < n else 0)
        return reverse_quicksum

    def train_one_epoch(self, batch_size, episode_limit, verbose=False):
        total_length = 0
        batch_states = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lengths = []
        batch_allowed_actions_masks = []
        batch_success_count = 0
        while total_length < batch_size:
            episode_states = [self.sample_initial_state()]
            episode_actions = []
            episode_rewards = []
            episode_allowed_actions_masks = []
            if verbose:
                print('[LOG] Starting Episode', len(batch_returns), 'with state', episode_states[-1])
            while not episode_states[-1].is_solved() and len(episode_states) < episode_limit:
                allowed = episode_states[-1].actions()
                mask = [0 if i not in allowed else 1 for i in range(4)]
                cat = self.policy.run_policy(torch.as_tensor(episode_states[-1].board.flatten(), dtype=torch.float32), torch.as_tensor(mask, dtype=torch.float32))
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
                episode_allowed_actions_masks.append(mask)
            if episode_states[-1].is_solved():
                batch_success_count += 1
            episode_total_rewards = sum(episode_rewards)
            batch_returns.append(episode_total_rewards)
            episode_length = len(episode_rewards)
            batch_lengths.append(episode_length)
            batch_weights += list(self.reward_to_go(episode_rewards))
            total_length += episode_length
            batch_states += episode_states[:-1]
            batch_actions += episode_actions
            batch_allowed_actions_masks += episode_allowed_actions_masks
        self.optimizer.zero_grad()
        batch_loss = self.policy.get_loss(batch_states, batch_actions, batch_weights, batch_allowed_actions_masks)
        batch_loss.backward()
        self.optimizer.step()
        print('[LOG] Batch end; loss: %.3f \t return: %.3f \t ep_len: %.3f \t success rate: %.3f' %
            (batch_loss, np.mean(batch_returns), np.mean(batch_lengths), batch_success_count / len(batch_returns)), flush=True)
        return batch_loss, batch_returns, batch_lengths
            