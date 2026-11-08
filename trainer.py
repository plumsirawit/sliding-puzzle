from policy import Policy, Critic
from gen import sample_board
import numpy as np
import torch
import random

class Trainer:
    def __init__(self, lr, max_scaffold_dist=1):
        self.policy = Policy(lr)
        self.critic = Critic()
        self.MAX_SCAFFOLD_DIST = max_scaffold_dist
        random.seed(42)

    def sample_initial_state(self):
        return sample_board(self.MAX_SCAFFOLD_DIST)

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
            while not episode_states[-1].is_solved() and len(episode_states) <= episode_limit:
                allowed = episode_states[-1].actions()
                mask = [0 if i not in allowed else 1 for i in range(4)]
                cat = self.policy.run_policy(episode_states[-1].to_tensor(), torch.as_tensor(mask, dtype=torch.float32))
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
        states = torch.stack([s.to_tensor() for s in batch_states])
        returns = torch.as_tensor(batch_weights, dtype=torch.float32)

        # Compute advantages BEFORE fitting the critic on this batch.
        # no_grad prevents actor training from changing the critic.
        with torch.no_grad():
            predicted_returns = self.critic.run(states).squeeze(-1)  # [N]
            advantages = returns - predicted_returns

            # Optional, commonly useful for keeping update scales manageable.
            advantages = (
                (advantages - advantages.mean())
                / (advantages.std(unbiased=False) + 1e-8)
            )
        
        self.policy.optimizer.zero_grad()
        actor_loss = self.policy.get_loss(batch_states, batch_actions, advantages.tolist(), batch_allowed_actions_masks)
        actor_loss.backward()
        self.policy.optimizer.step()
        for _ in range(10):
            values = self.critic.run(states).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(values, returns)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

        masks = torch.as_tensor(np.array(batch_allowed_actions_masks), dtype=torch.float32)
        with torch.no_grad():
            values = self.critic.run(states).squeeze(-1)
            critic_mse = torch.nn.functional.mse_loss(values, returns)
            policy_entropy = self.policy.run_policy(states, masks).entropy().mean()

        print('[LOG] Batch end; loss: %.3f \t return: %.3f \t ep_len: %.3f \t success rate: %.3f'
              ' \t scaffold_dist: %d \t critic_mse: %.3f \t policy_entropy: %.3f' %
            (actor_loss.item(), np.mean(batch_returns), np.mean(batch_lengths),
             batch_success_count / len(batch_returns), self.MAX_SCAFFOLD_DIST,
             critic_mse.item(), policy_entropy.item()), flush=True)
        return actor_loss, batch_returns, batch_lengths
            