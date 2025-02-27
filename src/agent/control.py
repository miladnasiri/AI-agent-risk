"""
Main control module for running trained agents.
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym

from ..simulation.environment import TelloEnv, TelloGymEnv
from .base_agent import RandomAgent, HoverAgent, WaypointAgent
from .rl_agent import PPOAgent, SACAgent, TD3Agent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a Tello drone agent in simulation")
    
    parser.add_argument("--agent", type=str, default="hover",
                       choices=["random", "hover", "waypoint", "ppo", "sac", "td3"],
                       help="Agent type to use (default: hover)")
    
    parser.add_argument("--task", type=str, default="hover",
                       choices=["hover", "takeoff", "land", "waypoint", "square", "circle", "flip"],
                       help="Task to run (default: hover)")
    
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model (for RL agents)")
    
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run (default: 5)")
    
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode (default: 1000)")
    
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: None)")
    
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering")
    
    parser.add_argument("--record", action="store_true",
                       help="Record video of the agent")
    
    parser.add_argument("--record-dir", type=str, default="./videos",
                       help="Directory to save recorded videos (default: ./videos)")
    
    return parser.parse_args()


def create_agent(agent_type, env):
    """Create agent based on specified type."""
    observation_space = env.observation_space
    action_space = env.action_space
    
    if agent_type == "random":
        return RandomAgent(action_space)
    
    elif agent_type == "hover":
        # Target position for hover depends on the task
        if hasattr(env, "task") and env.task == "hover":
            target_position = env.target_position
        else:
            target_position = np.array([0.0, 0.0, 1.5])  # Default hover position
        
        return HoverAgent(action_space, target_position)
    
    elif agent_type == "waypoint":
        # Use task-specific waypoints if available
        if hasattr(env, "task"):
            if env.task == "waypoint" and hasattr(env, "waypoints"):
                waypoints = env.waypoints
            elif env.task == "square" and hasattr(env, "square_corners"):
                waypoints = env.square_corners
            elif env.task == "circle":
                # Generate waypoints around a circle
                n_points = 8
                radius = env.circle_radius
                height = env.circle_height
                center = env.circle_center
                
                waypoints = []
                for i in range(n_points + 1):  # +1 to close the circle
                    angle = 2 * np.pi * i / n_points
                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    waypoints.append(np.array([x, y, height]))
            else:
                # Default square pattern
                waypoints = [
                    np.array([0.0, 0.0, 1.5]),
                    np.array([2.0, 0.0, 1.5]),
                    np.array([2.0, 2.0, 1.5]),
                    np.array([0.0, 2.0, 1.5]),
                    np.array([0.0, 0.0, 1.5])
                ]
        else:
            # Default square pattern
            waypoints = [
                np.array([0.0, 0.0, 1.5]),
                np.array([2.0, 0.0, 1.5]),
                np.array([2.0, 2.0, 1.5]),
                np.array([0.0, 2.0, 1.5]),
                np.array([0.0, 0.0, 1.5])
            ]
        
        return WaypointAgent(action_space, waypoints)
    
    elif agent_type == "ppo":
        return PPOAgent(observation_space, action_space)
    
    elif agent_type == "sac":
        return SACAgent(observation_space, action_space)
    
    elif agent_type == "td3":
        return TD3Agent(observation_space, action_space)
    
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def run_agent(env, agent, num_episodes=5, max_steps=1000, render=True):
    """Run the agent for a specified number of episodes."""
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()
        
        total_reward = 0
        step_count = 0
        
        while step_count < max_steps:
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update total reward
            total_reward += reward
            step_count += 1
            
            # Print step info
            if step_count % 100 == 0 or terminated or truncated:
                pos = info['drone_state']['position']
                print(f"Step {step_count}: Pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], Reward={reward:.2f}")
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Episode summary
        print(f"Episode {episode + 1} completed - Steps: {step_count}, Total Reward: {total_reward:.2f}")
        episode_rewards.append(total_reward)
    
    # Print overall statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Average reward over {num_episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return episode_rewards


def main():
    """Main function for running agents."""
    args = parse_args()
    
    # Determine render mode
    render_mode = None if args.no_render else "human"
    
    # Create environment
    if args.task == "default":
        env = TelloEnv(render_mode=render_mode, max_episode_steps=args.max_steps)
    else:
        env = TelloGymEnv(task=args.task, render_mode=render_mode, max_episode_steps=args.max_steps)
    
    # Set seed if specified
    if args.seed is not None:
        env.reset(seed=args.seed)
    
    # Create agent
    agent = create_agent(args.agent, env)
    
    # Load model for RL agents
    if args.agent in ["ppo", "sac", "td3"]:
        if args.model_path is None:
            print(f"Warning: No model path provided for {args.agent.upper()} agent. Using untrained model.")
        else:
            print(f"Loading {args.agent.upper()} model from {args.model_path}")
            agent.load(args.model_path)
        
        # Set to evaluation mode
        agent.eval()
    
    # Set up video recording if requested
    if args.record:
        os.makedirs(args.record_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(args.record_dir, f"{args.agent}_{args.task}_{timestamp}.mp4")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.record_dir,
            episode_trigger=lambda x: True,  # Record all episodes
            name_prefix=f"{args.agent}_{args.task}"
        )
        print(f"Recording videos to {args.record_dir}")
    
    # Run the agent
    print(f"Running {args.agent} agent on {args.task} task for {args.episodes} episodes")
    run_agent(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, render=not args.no_render)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
