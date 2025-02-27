"""
Deep learning agent with advanced capabilities for Tello drone control.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random

from .base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """
    Actor network for the deep agent that outputs action mean and standard deviation.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std layers
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        """Forward pass through the network."""
        x = self.shared(state)
        
        # Calculate mean and log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """Sample actions from the policy given a state."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean), None
        
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Sample from the normal distribution
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Calculate log probability of the action
        log_prob = normal.log_prob(x_t)
        
        # Apply the change of variables formula for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network for the deep agent that estimates Q-values.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture (for double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """Forward pass through both Q-networks."""
        # Concatenate state and action
        sa = torch.cat([state, action], 1)
        
        # Get Q-values from both networks
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        
        return q1, q2


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch]).reshape(-1, 1)
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch]).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DeepAgent(BaseAgent):
    """
    Advanced deep reinforcement learning agent for drone control.
    
    This agent implements Soft Actor-Critic (SAC) with automatic entropy tuning,
    which is particularly well-suited for continuous control tasks like drone navigation.
    """
    
    def __init__(self, observation_space, action_space, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(action_space)
        
        self.device = device
        
        # Get dimensions from spaces
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        
        # Copy critic parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=1000000, state_dim=self.state_dim, action_dim=self.action_dim)
        
        # SAC parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Target network update rate
        self.alpha = 0.2  # Temperature parameter for entropy
        
        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Training parameters
        self.batch_size = 256
        self.learning_starts = 10000
        self.update_every = 1
        self.updates_per_step = 1
        
        # Evaluation parameters
        self.is_training = True
        
        # Step counter
        self.total_steps = 0
    
    def get_action(self, observation):
        """Get action from the actor network."""
        # Convert observation to tensor
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Get action from actor network
        with torch.no_grad():
            if self.is_training:
                action, _ = self.actor.sample(state, deterministic=False)
            else:
                action, _ = self.actor.sample(state, deterministic=True)
        
        # Convert to numpy array and reshape
        action = action.cpu().numpy().reshape(-1)
        
        return action
    
    def update_parameters(self, batch_size):
        """Update the parameters of the networks."""
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current alpha
        alpha = self.log_alpha.exp()
        
        # --- Update Critic ---
        with torch.no_grad():
            # Get next actions and log probs from actor
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Get Q-values from target critic
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Calculate target Q-value
            target_q = rewards + (1 - dones) * self.gamma * (next_q - alpha * next_log_probs)
        
        # Get current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Update Actor ---
        # Get actions and log probs from actor
        new_actions, log_probs = self.actor.sample(states)
        
        # Get Q-values for new actions
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        # Compute actor loss
        actor_loss = (alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- Update Alpha ---
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # --- Update Target Networks ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def train(self, env, total_timesteps=1000000, eval_freq=10000, eval_episodes=5):
        """Train the agent."""
        # Reset environment
        state, _ = env.reset()
        
        # Training loop
        episode_reward = 0
        episode_steps = 0
        episode_num = 0
        
        for step in range(1, total_timesteps + 1):
            self.total_steps = step
            
            # Select action
            if step < self.learning_starts:
                # Random exploration at the beginning
                action = env.action_space.sample()
            else:
                action = self.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            episode_reward += reward
            
            # Check if episode is done
            done = terminated or truncated
            
            # Store transition in replay buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Update state
            state = next_state
            
            # Update networks
            if step >= self.learning_starts and step % self.update_every == 0:
                for _ in range(self.updates_per_step):
                    self.update_parameters(self.batch_size)
            
            # End of episode handling
            if done:
                # Log episode stats
                print(f"Episode {episode_num + 1} - Steps: {episode_steps}, Reward: {episode_reward:.2f}")
                
                # Reset environment for next episode
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_num += 1
            
            # Evaluation
            if step % eval_freq == 0:
                self.eval()
                self._evaluate(env, eval_episodes)
                self.train()
    
    def _evaluate(self, env, eval_episodes=5):
        """Evaluate the agent."""
        print(f"\nEvaluating agent after {self.total_steps} steps")
        
        avg_reward = 0.0
        
        for episode in range(eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                state = next_state
            
            avg_reward += episode_reward
            print(f"Eval Episode {episode + 1} - Reward: {episode_reward:.2f}")
        
        avg_reward /= eval_episodes
        print(f"Average Evaluation Reward: {avg_reward:.2f}\n")
    
    def train(self):
        """Set agent to training mode."""
        super().train()
        self.actor.train()
        self.critic.train()
    
    def eval(self):
        """Set agent to evaluation mode."""
        super().eval()
        self.actor.eval()
        self.critic.eval()
    
    def save(self, path):
        """Save the agent to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load the agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
