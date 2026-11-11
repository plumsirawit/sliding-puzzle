from policy import Policy, Critic
from gen import bfs_generator, random_walk_generator
import numpy as np
import torch
import random
from pathlib import Path

class Trainer:
    def __init__(
        self,
        lr,
        max_scaffold_dist=1,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=4,
        minibatch_size=256,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1")
        if not 0.0 < clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be between 0 and 1")
        if ppo_epochs <= 0 or minibatch_size <= 0:
            raise ValueError("ppo_epochs and minibatch_size must be positive")
        if entropy_coef < 0.0 or value_coef < 0.0 or max_grad_norm <= 0.0:
            raise ValueError("loss coefficients must be nonnegative and max_grad_norm positive")
        self.device = torch.device(device)
        self.policy = Policy(lr, self.device)
        self.critic = Critic(self.device)
        self.MAX_SCAFFOLD_DIST = max_scaffold_dist
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        random.seed(42)

    def sample_initial_state(self):
        if self.MAX_SCAFFOLD_DIST <= 10:
            return bfs_generator.sample_board(self.MAX_SCAFFOLD_DIST)
        else:
            return random_walk_generator.sample_board(2*self.MAX_SCAFFOLD_DIST)

    def save_checkpoint(self, path, epoch):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

        numpy_rng = np.random.get_state()
        checkpoint = {
            "format_version": 1,
            "epoch": epoch,
            "max_scaffold_dist": self.MAX_SCAFFOLD_DIST,
            "trainer_config": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "ppo_epochs": self.ppo_epochs,
                "minibatch_size": self.minibatch_size,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "max_grad_norm": self.max_grad_norm,
            },
            "actor_state_dict": self.policy.mlp.state_dict(),
            "critic_state_dict": self.critic.mlp.state_dict(),
            "actor_optimizer_state_dict": self.policy.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic.optimizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": (
                    numpy_rng[0],
                    numpy_rng[1].tolist(),
                    numpy_rng[2],
                    numpy_rng[3],
                    numpy_rng[4],
                ),
                "torch_cpu": torch.get_rng_state(),
                "torch_xpu": (
                    torch.xpu.get_rng_state_all()
                    if self.device.type == "xpu"
                    else None
                ),
            },
        }

        try:
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError(f"Unsupported checkpoint format in {path}")

        self.policy.mlp.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.mlp.load_state_dict(checkpoint["critic_state_dict"])
        self.policy.optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic.optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.MAX_SCAFFOLD_DIST = checkpoint["max_scaffold_dist"]

        for name, value in checkpoint["trainer_config"].items():
            setattr(self, name, value)

        rng_state = checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        numpy_rng = rng_state["numpy"]
        np.random.set_state(
            (
                numpy_rng[0],
                np.asarray(numpy_rng[1], dtype=np.uint32),
                numpy_rng[2],
                numpy_rng[3],
                numpy_rng[4],
            )
        )
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        if self.device.type == "xpu" and rng_state["torch_xpu"] is not None:
            torch.xpu.set_rng_state_all(
                [state.cpu() for state in rng_state["torch_xpu"]]
            )

        return checkpoint["epoch"]

    def train_one_epoch(self, batch_size, episode_limit, verbose=False):
        if batch_size <= 0 or episode_limit <= 0:
            raise ValueError("batch_size and episode_limit must be positive")
        total_length = 0
        batch_states = []
        batch_next_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_returns = []
        batch_lengths = []
        batch_allowed_actions_masks = []
        episode_ranges = []
        batch_success_count = 0
        while total_length < batch_size:
            episode_states = [self.sample_initial_state()]
            episode_actions = []
            episode_rewards = []
            episode_dones = []
            episode_allowed_actions_masks = []
            if verbose:
                print('[LOG] Starting Episode', len(batch_returns), 'with state', episode_states[-1])
            while not episode_states[-1].is_solved() and len(episode_actions) < episode_limit:
                allowed = episode_states[-1].actions()
                mask = [0 if i not in allowed else 1 for i in range(4)]
                action = self.policy.sample_action(
                    episode_states[-1].to_tensor().to(self.device),
                    torch.as_tensor(mask, dtype=torch.float32, device=self.device),
                )
                if verbose:
                    print('[LOG] Current state is')
                    print(episode_states[-1])
                    print('[LOG] Sampled action is', 'ULDR'[action])
                state = episode_states[-1].copy()
                reward = state.perform_action(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(state.is_solved())
                episode_allowed_actions_masks.append(mask)
            if episode_states[-1].is_solved():
                batch_success_count += 1
            episode_total_rewards = sum(episode_rewards)
            batch_returns.append(episode_total_rewards)
            episode_length = len(episode_rewards)
            batch_lengths.append(episode_length)
            start = len(batch_actions)
            total_length += episode_length
            batch_states += episode_states[:-1]
            batch_next_states += episode_states[1:]
            batch_actions += episode_actions
            batch_rewards += episode_rewards
            batch_dones += episode_dones
            batch_allowed_actions_masks += episode_allowed_actions_masks
            episode_ranges.append((start, len(batch_actions)))

        states = torch.stack([s.to_tensor() for s in batch_states]).to(self.device)
        next_states = torch.stack([s.to_tensor() for s in batch_next_states]).to(self.device)
        actions = torch.as_tensor(batch_actions, dtype=torch.long, device=self.device)
        masks = torch.as_tensor(
            np.asarray(batch_allowed_actions_masks), dtype=torch.float32, device=self.device
        )
        rewards = np.asarray(batch_rewards, dtype=np.float32)

        with torch.no_grad():
            old_log_probs, _ = self.policy.evaluate_actions(
                states, actions, masks
            )
            values = self.critic.run(states).squeeze(-1)
            next_values = self.critic.run(next_states).squeeze(-1)

        values_cpu = values.cpu().numpy()
        next_values_cpu = next_values.cpu().numpy()
        advantages_array = np.zeros_like(rewards)
        returns_array = np.zeros_like(rewards)
        for start, end in episode_ranges:
            gae = 0.0
            for index in reversed(range(start, end)):
                continuation = 0.0 if batch_dones[index] else 1.0
                delta = (
                    rewards[index]
                    + self.gamma * continuation * next_values_cpu[index]
                    - values_cpu[index]
                )
                gae = delta + self.gamma * self.gae_lambda * continuation * gae
                advantages_array[index] = gae
                returns_array[index] = gae + values_cpu[index]

        advantages = torch.as_tensor(advantages_array, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_array, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_log_probs = old_log_probs.detach()

        actor_losses = []
        critic_losses = []
        clip_fractions = []
        approximate_kls = []
        sample_count = len(batch_actions)
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, self.minibatch_size):
                indices = permutation[start:start + self.minibatch_size].to(self.device)
                log_probs, entropy = self.policy.evaluate_actions(
                    states[indices], actions[indices], masks[indices]
                )
                ratio = torch.exp(log_probs - old_log_probs[indices])
                unclipped_objective = ratio * advantages[indices]
                clipped_objective = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * advantages[indices]
                actor_loss = -torch.minimum(
                    unclipped_objective, clipped_objective
                ).mean()
                entropy_mean = entropy.mean()

                self.policy.optimizer.zero_grad(set_to_none=True)
                (actor_loss - self.entropy_coef * entropy_mean).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.mlp.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                predicted_values = self.critic.run(states[indices]).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(
                    predicted_values, returns[indices]
                )
                self.critic.optimizer.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.mlp.parameters(), self.max_grad_norm
                )
                self.critic.optimizer.step()

                with torch.no_grad():
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    )
                    approximate_kls.append(
                        ((ratio - 1.0) - torch.log(ratio)).mean().item()
                    )

        with torch.no_grad():
            values = self.critic.run(states).squeeze(-1)
            critic_mse = torch.nn.functional.mse_loss(values, returns)
            policy_entropy = self.policy.run_policy(states, masks).entropy().mean()

        mean_actor_loss = float(np.mean(actor_losses))
        mean_critic_loss = float(np.mean(critic_losses))
        print('[LOG] Batch end; loss: %.3f \t return: %.3f \t ep_len: %.3f \t success rate: %.3f'
              ' \t scaffold_dist: %d \t critic_mse: %.3f \t policy_entropy: %.3f'
              ' \t clip_fraction: %.3f \t approx_kl: %.5f' %
            (mean_actor_loss, np.mean(batch_returns), np.mean(batch_lengths),
             batch_success_count / len(batch_returns), self.MAX_SCAFFOLD_DIST,
             critic_mse.item(), policy_entropy.item(),
             np.mean(clip_fractions), np.mean(approximate_kls)), flush=True)
        return mean_actor_loss, batch_returns, batch_lengths
            