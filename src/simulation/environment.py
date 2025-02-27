"""
Simulation environment for the Tello drone digital twin.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .drone_model import TelloDroneModel


class TelloEnv(gym.Env):
    """
    A Gymnasium environment for simulating the DJI Tello drone.
    
    This environment simulates the Tello drone with realistic physics and
    provides an interface for reinforcement learning agents to train on.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }
    
    def __init__(self, render_mode=None, max_episode_steps=1000):
        super().__init__()
        
        # Initialize drone model
        self.drone = TelloDroneModel()
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.render_mode = render_mode
        self.renderer = None
        
        # Define action space: [thrust, roll, pitch, yaw]
        # All values in range [-1, 1] except thrust which is [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, omega_x, omega_y, omega_z, battery]
        self.observation_space = spaces.Box(
            low=np.array([
                -10.0, -10.0, 0.0,     # position
                -5.0, -5.0, -5.0,      # velocity
                -np.pi, -np.pi, -np.pi, # orientation
                -10.0, -10.0, -10.0,   # angular velocity
                0.0                     # battery
            ]),
            high=np.array([
                10.0, 10.0, 10.0,      # position
                5.0, 5.0, 5.0,         # velocity
                np.pi, np.pi, np.pi,   # orientation
                10.0, 10.0, 10.0,      # angular velocity
                100.0                  # battery
            ]),
            dtype=np.float32
        )
        
        # Set up environment bounds
        self.bounds = {
            'x': (-10.0, 10.0),
            'y': (-10.0, 10.0),
            'z': (0.0, 10.0)
        }
        
        # Initialize visualization if needed
        if self.render_mode is not None:
            self._setup_rendering()
    
    def _setup_rendering(self):
        """Set up the rendering engine."""
        try:
            from .visualization import DroneRenderer
            self.renderer = DroneRenderer(render_mode=self.render_mode)
        except ImportError:
            print("Warning: Could not import visualization module. Rendering will be disabled.")
            self.render_mode = None
    
    def _get_observation(self):
        """Convert drone state to observation vector."""
        return np.concatenate([
            self.drone.position,
            self.drone.velocity,
            self.drone.orientation,
            self.drone.angular_velocity,
            [self.drone.battery_percentage]
        ]).astype(np.float32)
    
    def _is_done(self):
        """Check if episode is done."""
        # Check if drone has crashed (hit the ground)
        crashed = self.drone.position[2] <= 0.05 and self.drone.velocity[2] < 0
        
        # Check if drone is out of bounds
        out_of_bounds = (
            self.drone.position[0] < self.bounds['x'][0] or
            self.drone.position[0] > self.bounds['x'][1] or
            self.drone.position[1] < self.bounds['y'][0] or
            self.drone.position[1] > self.bounds['y'][1] or
            self.drone.position[2] < self.bounds['z'][0] or
            self.drone.position[2] > self.bounds['z'][1]
        )
        
        # Check if battery is depleted
        battery_depleted = self.drone.battery_percentage < 1.0
        
        # Check if maximum steps reached
        max_steps_reached = self.steps >= self.max_episode_steps
        
        return crashed or out_of_bounds or battery_depleted or max_steps_reached
    
    def _compute_reward(self, action):
        """
        Compute reward based on drone state and action.
        
        This is a simple reward function that penalizes crashes and encourages stability.
        Can be customized depending on the specific task.
        """
        reward = 0.0
        
        # Base reward for staying alive
        reward += 0.1
        
        # Penalty for crashing
        if self.drone.position[2] <= 0.05 and self.drone.velocity[2] < 0:
            reward -= 100.0
        
        # Penalty for large angular velocities (encourage stability)
        reward -= 0.01 * np.sum(np.square(self.drone.angular_velocity))
        
        # Penalty for high-speed motion (encourage gentle movements)
        reward -= 0.01 * np.sum(np.square(self.drone.velocity))
        
        # Penalty for large control inputs (encourage efficiency)
        reward -= 0.01 * np.sum(np.square(action))
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.steps = 0
        
        # Reset drone to a random position (but always above ground)
        if options is not None and 'position' in options:
            initial_position = options['position']
        else:
            initial_position = np.array([
                self.np_random.uniform(*self.bounds['x']),
                self.np_random.uniform(*self.bounds['y']),
                self.np_random.uniform(0.5, 2.0)  # Starting height
            ])
        
        # Reset drone model
        self.drone.reset(position=initial_position)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initialize renderer if needed
        if self.render_mode == 'human' and self.renderer is not None:
            self.renderer.reset(self.drone)
        
        info = {}
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Ensure action is in correct format and range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Set drone commands
        self.drone.set_commands(
            thrust=action[0],
            roll=action[1],
            pitch=action[2],
            yaw=action[3]
        )
        
        # Step the drone simulation
        self.drone.step()
        
        # Get new observation
        observation = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False
        
        # Additional info
        info = {
            'drone_state': {
                'position': self.drone.position.copy(),
                'velocity': self.drone.velocity.copy(),
                'orientation': self.drone.orientation.copy(),
                'angular_velocity': self.drone.angular_velocity.copy(),
                'battery': self.drone.battery_percentage
            }
        }
        
        # Render if needed
        if self.render_mode == 'human' and self.renderer is not None:
            self.renderer.render(self.drone)
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state of the environment."""
        if self.renderer is None or self.render_mode is None:
            return None
        
        return self.renderer.render(self.drone)
    
    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
        
        super().close()


class TelloGymEnv(TelloEnv):
    """
    A wrapper around TelloEnv that provides specific tasks like takeoff,
    landing, waypoint navigation, etc.
    """
    
    def __init__(self, task='hover', render_mode=None, max_episode_steps=1000):
        super().__init__(render_mode=render_mode, max_episode_steps=max_episode_steps)
        
        # Available tasks
        self.available_tasks = [
            'hover',
            'takeoff',
            'land',
            'waypoint',
            'square',
            'circle',
            'flip'
        ]
        
        # Set task
        self.set_task(task)
    
    def set_task(self, task):
        """Set the current task for the environment."""
        if task not in self.available_tasks:
            raise ValueError(f"Task '{task}' not supported. Available tasks: {self.available_tasks}")
        
        self.task = task
        self._setup_task()
    
    def _setup_task(self):
        """Setup the environment for the current task."""
        # Task-specific parameters
        if self.task == 'hover':
            self.target_position = np.array([0.0, 0.0, 1.5])
            self.target_orientation = np.zeros(3)
            self.hover_time = 5.0  # seconds
        
        elif self.task == 'takeoff':
            self.target_height = 1.5
        
        elif self.task == 'land':
            self.safe_landing_velocity = -0.5  # m/s
        
        elif self.task == 'waypoint':
            self.waypoints = [
                np.array([0.0, 0.0, 1.5]),
                np.array([2.0, 0.0, 1.5]),
                np.array([2.0, 2.0, 1.5]),
                np.array([0.0, 2.0, 1.5]),
                np.array([0.0, 0.0, 1.5])
            ]
            self.current_waypoint_idx = 0
            self.waypoint_threshold = 0.3  # meters
        
        elif self.task == 'square':
            self.square_size = 2.0  # meters
            self.square_height = 1.5  # meters
            self.square_corners = [
                np.array([0.0, 0.0, self.square_height]),
                np.array([self.square_size, 0.0, self.square_height]),
                np.array([self.square_size, self.square_size, self.square_height]),
                np.array([0.0, self.square_size, self.square_height]),
                np.array([0.0, 0.0, self.square_height])
            ]
            self.current_corner_idx = 0
            self.corner_threshold = 0.3  # meters
        
        elif self.task == 'circle':
            self.circle_radius = 2.0  # meters
            self.circle_height = 1.5  # meters
            self.circle_center = np.array([0.0, 0.0, self.circle_height])
            self.circle_period = 10.0  # seconds for one full circle
        
        elif self.task == 'flip':
            self.flip_direction = 'front'  # 'front', 'back', 'left', 'right'
            self.flip_phases = ['prepare', 'flip', 'recover']
            self.current_flip_phase = 'prepare'
            self.flip_phase_time = {
                'prepare': 1.0,  # seconds
                'flip': 0.5,  # seconds
                'recover': 1.0  # seconds
            }
            self.flip_timer = 0.0
    
    def _compute_task_reward(self):
        """Compute task-specific reward."""
        reward = 0.0
        
        if self.task == 'hover':
            # Reward for staying close to target position
            pos_error = np.linalg.norm(self.drone.position - self.target_position)
            reward -= pos_error
            
            # Bonus for very stable hovering
            if pos_error < 0.1:
                reward += 0.5
            
            # Penalty for large orientation errors
            orient_error = np.linalg.norm(self.drone.orientation - self.target_orientation)
            reward -= orient_error
        
        elif self.task == 'takeoff':
            # Reward proportional to height up to target height
            reward += min(self.drone.position[2], self.target_height) / self.target_height
            
            # Bonus for reaching target height
            if self.drone.position[2] >= self.target_height:
                reward += 1.0
            
            # Penalty for horizontal movement during takeoff
            horizontal_movement = np.linalg.norm(self.drone.velocity[:2])
            reward -= 0.1 * horizontal_movement
        
        elif self.task == 'land':
            # Reward inversely proportional to height
            reward += 1.0 - (self.drone.position[2] / 2.0)
            
            # Bonus for smooth landing (slow descent)
            if self.drone.position[2] < 0.5 and self.drone.velocity[2] > self.safe_landing_velocity:
                reward += 1.0
            
            # Penalty for horizontal movement during landing
            horizontal_movement = np.linalg.norm(self.drone.velocity[:2])
            reward -= 0.1 * horizontal_movement
        
        elif self.task == 'waypoint':
            # Current target waypoint
            target = self.waypoints[self.current_waypoint_idx]
            
            # Distance to current waypoint
            distance = np.linalg.norm(self.drone.position - target)
            
            # Reward inversely proportional to distance
            reward += 1.0 / (1.0 + distance)
            
            # Check if waypoint reached
            if distance < self.waypoint_threshold:
                reward += 10.0  # Bonus for reaching waypoint
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
        
        elif self.task == 'square':
            # Similar to waypoint, but with square corners
            target = self.square_corners[self.current_corner_idx]
            distance = np.linalg.norm(self.drone.position - target)
            reward += 1.0 / (1.0 + distance)
            
            if distance < self.corner_threshold:
                reward += 5.0
                self.current_corner_idx = (self.current_corner_idx + 1) % len(self.square_corners)
        
        elif self.task == 'circle':
            # Target position based on current time
            angle = (2 * np.pi * self.drone.time) / self.circle_period
            target_x = self.circle_center[0] + self.circle_radius * np.cos(angle)
            target_y = self.circle_center[1] + self.circle_radius * np.sin(angle)
            target = np.array([target_x, target_y, self.circle_height])
            
            # Distance to target position on circle
            distance = np.linalg.norm(self.drone.position - target)
            reward += 1.0 / (1.0 + distance)
            
            # Bonus for being close to the circle path
            radial_distance = abs(np.linalg.norm(self.drone.position[:2] - self.circle_center[:2]) - self.circle_radius)
            if radial_distance < 0.2:
                reward += 0.5
        
        elif self.task == 'flip':
            # Update flip timer
            self.flip_timer += self.drone.dt
            
            # Handle flip phases
            if self.current_flip_phase == 'prepare':
                # Reward for stable hovering in preparation
                target = np.array([0.0, 0.0, 1.5])
                distance = np.linalg.norm(self.drone.position - target)
                reward += 1.0 / (1.0 + distance)
                
                # Move to flip phase once preparation time is up
                if self.flip_timer >= self.flip_phase_time['prepare']:
                    self.current_flip_phase = 'flip'
                    self.flip_timer = 0.0
            
            elif self.current_flip_phase == 'flip':
                # Reward for rotation speed in appropriate axis
                if self.flip_direction == 'front':
                    reward += abs(self.drone.angular_velocity[0])
                elif self.flip_direction == 'back':
                    reward += abs(self.drone.angular_velocity[0])
                elif self.flip_direction == 'left':
                    reward += abs(self.drone.angular_velocity[1])
                elif self.flip_direction == 'right':
                    reward += abs(self.drone.angular_velocity[1])
                
                # Check for completed rotation
                if self.flip_timer >= self.flip_phase_time['flip']:
                    self.current_flip_phase = 'recover'
                    self.flip_timer = 0.0
                    reward += 10.0  # Bonus for completing flip
            
            elif self.current_flip_phase == 'recover':
                # Reward for stabilizing after flip
                target_orientation = np.zeros(3)
                orient_error = np.linalg.norm(self.drone.orientation - target_orientation)
                reward += 1.0 / (1.0 + orient_error)
                
                # End recovery phase
                if self.flip_timer >= self.flip_phase_time['recover']:
                    self.current_flip_phase = 'prepare'
                    self.flip_timer = 0.0
                    reward += 5.0  # Bonus for successful recovery
        
        return reward
    
    def _compute_reward(self, action):
        """Compute combined reward based on drone state, action, and task."""
        # Get basic safety/stability reward from parent class
        basic_reward = super()._compute_reward(action)
        
        # Get task-specific reward
        task_reward = self._compute_task_reward()
        
        return basic_reward + task_reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment for the current task."""
        # Reset timer for task phases
        self.flip_timer = 0.0
        
        # Reset task-specific variables
        if self.task == 'waypoint':
            self.current_waypoint_idx = 0
        elif self.task == 'square':
            self.current_corner_idx = 0
        elif self.task == 'flip':
            self.current_flip_phase = 'prepare'
        
        # Reset the environment
        return super().reset(seed=seed, options=options)
