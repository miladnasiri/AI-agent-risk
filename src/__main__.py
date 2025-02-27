"""
Main entry point for the DJTello Digital Twin.
"""

import argparse
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DJTello Digital Twin - A digital twin simulation for DJI Tello drones with AI agent")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulation", help="Run the simulation")
    sim_parser.add_argument("--render", action="store_true", help="Enable rendering")
    sim_parser.add_argument("--task", type=str, default="hover", 
                           choices=["hover", "takeoff", "land", "waypoint", "square", "circle", "flip"],
                           help="Task to run (default: hover)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument("--algorithm", type=str, default="ppo", 
                             choices=["ppo", "sac", "td3", "deep"],
                             help="RL algorithm to use (default: ppo)")
    train_parser.add_argument("--task", type=str, default="hover",
                             choices=["hover", "takeoff", "land", "waypoint", "square", "circle", "flip"],
                             help="Task to train on (default: hover)")
    train_parser.add_argument("--timesteps", type=int, default=500000,
                             help="Total timesteps for training (default: 500000)")
    train_parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    
    # Control command
    control_parser = subparsers.add_parser("control", help="Run a trained agent")
    control_parser.add_argument("--agent", type=str, default="hover",
                               choices=["random", "hover", "waypoint", "ppo", "sac", "td3", "deep"],
                               help="Agent type to use (default: hover)")
    control_parser.add_argument("--task", type=str, default="hover",
                               choices=["hover", "takeoff", "land", "waypoint", "square", "circle", "flip"],
                               help="Task to run (default: hover)")
    control_parser.add_argument("--model-path", type=str, default=None,
                               help="Path to trained model (for RL agents)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "simulation":
        from .simulation.environment import TelloGymEnv
        
        # Create environment
        render_mode = "human" if args.render else None
        env = TelloGymEnv(task=args.task, render_mode=render_mode)
        
        # Run simulation with random actions
        print(f"Running simulation with task: {args.task}")
        
        for episode in range(3):
            obs, _ = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            
            print(f"Episode {episode + 1} completed - Steps: {steps}")
        
        env.close()
    
    elif args.command == "train":
        # Run the training script
        import sys
        from .agent.train import main as train_main
        
        # Build args for the training script
        train_args = ["--algorithm", args.algorithm, 
                      "--task", args.task, 
                      "--timesteps", str(args.timesteps)]
        
        if args.render:
            train_args.append("--render")
        
        # Call the training main function
        sys.argv = ["train.py"] + train_args
        train_main()
    
    elif args.command == "control":
        # Run the control script
        import sys
        from .agent.control import main as control_main
        
        # Build args for the control script
        control_args = ["--agent", args.agent, 
                        "--task", args.task]
        
        if args.model_path:
            control_args.extend(["--model-path", args.model_path])
        
        # Call the control main function
        sys.argv = ["control.py"] + control_args
        control_main()
    
    else:
        print("Please specify a command to run. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
