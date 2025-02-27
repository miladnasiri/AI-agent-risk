"""
Training script for reinforcement learning agents.
"""

import os
import argparse
import numpy as np
from datetime import datetime

from ..simulation.environment import TelloEnv, TelloGymEnv
from .rl_agent import PPOAgent, SACAgent, TD3Agent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent for Tello drone control")
    
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac", "td3"],
                       help="RL algorithm to use (default: ppo)")
    
    parser.add_argument("--task", type=str, default="hover",
                       choices=["hover", "takeoff", "land", "waypoint", "square", "circle", "flip"],
                       help="Task to train on (default: hover)")
    
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total timesteps for training (default: 500000)")
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: None)")
    
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during training")
    
    parser.add_argument("--eval", action="store_true",
                       help="Evaluate the agent after training")
    
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of episodes for evaluation (default: 10)")
    
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to load a pre-trained model (default: None)")
    
    parser.add_argument("--save-dir", type=str, default="./models",
                       help="Directory to save models (default: ./models)")
    
    return parser.parse_args()


def create_agent(algorithm, observation_space, action_space):
    """Create an agent based on the specified algorithm."""
    if algorithm == "ppo":
        return PPOAgent(observation_space, action_space)
    elif algorithm == "sac":
        return SACAgent(observation_space, action_space)
    elif algorithm == "td3":
        return TD3Agent(observation_space, action_space)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_env(task, render_mode=None, seed=None):
    """Create a Tello environment for the specified task."""
    # Create the environment
    if task == "default":
        env = TelloEnv(render_mode=render_mode)
    else:
        env = TelloGymEnv(task=task, render_mode=render_mode)
    
    # Set random seed if specified
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def evaluate(agent, env, num_episodes=10):
    """Evaluate an agent in the environment."""
    rewards = []
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.get_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {i+1}: Reward = {total_reward:.2f}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward


def main():
    """Main training function."""
    args = parse_args()
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f"{args.algorithm}_{args.task}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create log directory
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    render_mode = "human" if args.render else None
    env = create_env(args.task, render_mode=render_mode, seed=args.seed)
    
    # Create agent
    agent = create_agent(args.algorithm, env.observation_space, env.action_space)
    
    # Load pre-trained model if specified
    if args.model_path is not None:
        print(f"Loading model from {args.model_path}")
        agent.load(args.model_path)
    
    # Train the agent
    print(f"Training {args.algorithm.upper()} agent on {args.task} task for {args.timesteps} timesteps")
    agent.train(env, total_timesteps=args.timesteps, log_dir=log_dir)
    
    # Save the final model
    model_path = os.path.join(save_dir, "final_model.zip")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate the agent if requested
    if args.eval:
        print("Evaluating agent...")
        # Close training environment
        env.close()
        
        # Create new environment for evaluation
        eval_env = create_env(args.task, render_mode="human", seed=args.seed)
        agent.eval()  # Set agent to evaluation mode
        evaluate(agent, eval_env, num_episodes=args.eval_episodes)
        eval_env.close()


if __name__ == "__main__":
    main()
