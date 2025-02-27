"""
Reinforcement Learning agents for Tello drone control.
"""

import os
import time
import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from .base_agent import BaseAgent


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model when the training reward improves.
    """
    
    def __init__(self, check_freq=1000, log_dir="./logs/", verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
    
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Get mean episode reward
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            
            # New best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                # Save the model
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                
                self.model.save(os.path.join(self.save_path, "model"))
        
        return True


class RLAgent(BaseAgent):
    """
    Base class for reinforcement learning agents.
    
    This wraps Stable Baselines 3 RL algorithms.
    """
    
    def __init__(self, observation_space, action_space, algorithm="ppo", model_kwargs=None):
        """
        Initialize the RL agent.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            algorithm: RL algorithm to use ('ppo', 'sac', or 'td3')
            model_kwargs: Additional keyword arguments for the algorithm
        """
        super().__init__(action_space)
        
        self.observation_space = observation_space
        self.algorithm_name = algorithm.lower()
        
        # Set default model kwargs
        self.model_kwargs = {
            "verbose": 1,
            "tensorboard_log": "./logs/tensorboard/"
        }
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)
        
        # Initialize the RL model
        self.model = self._create_model()
    
    def _create_model(self):
        """Create the RL model based on the specified algorithm."""
        if self.algorithm_name == "ppo":
            return PPO(
                "MlpPolicy",
                self.observation_space,
                self.action_space,
                **self.model_kwargs
            )
        
        elif self.algorithm_name == "sac":
            return SAC(
                "MlpPolicy",
                self.observation_space,
                self.action_space,
                **self.model_kwargs
            )
        
        elif self.algorithm_name == "td3":
            # Add action noise for TD3
            n_actions = self.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
            
            return TD3(
                "MlpPolicy",
                self.observation_space,
                self.action_space,
                action_noise=action_noise,
                **self.model_kwargs
            )
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")
    
    def get_action(self, observation):
        """
        Get an action from the RL model.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            action: An action to be taken in the environment.
        """
        if self.is_training:
            # Let the model handle exploration during training
            action, _ = self.model.predict(observation, deterministic=False)
        else:
            # Use deterministic actions for evaluation
            action, _ = self.model.predict(observation, deterministic=True)
        
        return action
    
    def train(self, env, total_timesteps=100000, callback=None, log_dir="./logs/"):
        """
        Train the RL agent.
        
        Args:
            env: The environment to train in
            total_timesteps: Number of timesteps to train for
            callback: Optional callback function
            log_dir: Directory to save logs and models
        """
        super().train()
        
        # Create callback for saving best model
        if callback is None:
            callback = SaveOnBestTrainingRewardCallback(
                check_freq=10000,
                log_dir=log_dir
            )
        
        # Train the model
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        self.model.save(os.path.join(final_model_path, "model"))
        
        # Switch to evaluation mode
        self.eval()
    
    def save(self, path):
        """Save the RL model to disk."""
        self.model.save(path)
    
    def load(self, path):
        """Load the RL model from disk."""
        if self.algorithm_name == "ppo":
            self.model = PPO.load(path, self.observation_space, self.action_space)
        elif self.algorithm_name == "sac":
            self.model = SAC.load(path, self.observation_space, self.action_space)
        elif self.algorithm_name == "td3":
            self.model = TD3.load(path, self.observation_space, self.action_space)


class PPOAgent(RLAgent):
    """PPO agent for drone control."""
    
    def __init__(self, observation_space, action_space, model_kwargs=None):
        """
        Initialize a PPO agent.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            model_kwargs: Additional keyword arguments for PPO
        """
        # Default PPO parameters tuned for drone control
        ppo_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "policy_kwargs": {
                "net_arch": [dict(pi=[128, 128], vf=[128, 128])],
                "activation_fn": th.nn.ReLU
            }
        }
        
        # Update with user-provided kwargs
        if model_kwargs is not None:
            ppo_kwargs.update(model_kwargs)
        
        super().__init__(observation_space, action_space, "ppo", ppo_kwargs)


class SACAgent(RLAgent):
    """SAC agent for drone control."""
    
    def __init__(self, observation_space, action_space, model_kwargs=None):
        """
        Initialize a SAC agent.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            model_kwargs: Additional keyword arguments for SAC
        """
        # Default SAC parameters tuned for drone control
        sac_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": th.nn.ReLU
            }
        }
        
        # Update with user-provided kwargs
        if model_kwargs is not None:
            sac_kwargs.update(model_kwargs)
        
        super().__init__(observation_space, action_space, "sac", sac_kwargs)


class TD3Agent(RLAgent):
    """TD3 agent for drone control."""
    
    def __init__(self, observation_space, action_space, model_kwargs=None):
        """
        Initialize a TD3 agent.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            model_kwargs: Additional keyword arguments for TD3
        """
        # Default TD3 parameters tuned for drone control
        td3_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 100,
            "tau": 0.005,
            "gamma": 0.99,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "policy_kwargs": {
                "net_arch": [400, 300],
                "activation_fn": th.nn.ReLU
            }
        }
        
        # Update with user-provided kwargs
        if model_kwargs is not None:
            td3_kwargs.update(model_kwargs)
        
        super().__init__(observation_space, action_space, "td3", td3_kwargs)
