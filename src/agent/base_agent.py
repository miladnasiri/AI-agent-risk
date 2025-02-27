"""
Base class for Tello drone agents.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all drone control agents.
    
    This provides a common interface for different types of agents
    (rule-based, learning-based, etc.) to control the drone.
    """
    
    def __init__(self, action_space=None):
        """
        Initialize the agent.
        
        Args:
            action_space: The action space of the environment.
        """
        self.action_space = action_space
        self.is_training = False
    
    @abstractmethod
    def get_action(self, observation):
        """
        Get an action based on the current observation.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            action: An action to be taken in the environment.
        """
        pass
    
    def train(self):
        """Set agent to training mode."""
        self.is_training = True
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.is_training = False
    
    def save(self, path):
        """
        Save agent to disk.
        
        Args:
            path: Path to save the agent.
        """
        pass
    
    def load(self, path):
        """
        Load agent from disk.
        
        Args:
            path: Path to load the agent from.
        """
        pass


class RandomAgent(BaseAgent):
    """
    A simple agent that takes random actions.
    
    Useful for testing and as a baseline.
    """
    
    def __init__(self, action_space):
        super().__init__(action_space)
    
    def get_action(self, observation):
        """Return a random action from the action space."""
        return self.action_space.sample()


class HoverAgent(BaseAgent):
    """
    A simple rule-based agent that attempts to hover at a fixed position.
    
    Uses a PID controller to maintain position.
    """
    
    def __init__(self, action_space, target_position=None):
        super().__init__(action_space)
        
        # Default hover position
        self.target_position = np.array([0.0, 0.0, 1.5]) if target_position is None else np.array(target_position)
        
        # PID controller gains
        self.p_gains = np.array([0.5, 0.5, 0.8])  # For x, y, z
        self.i_gains = np.array([0.01, 0.01, 0.02])
        self.d_gains = np.array([0.2, 0.2, 0.4])
        
        # PID controller state
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
    
    def get_action(self, observation):
        """
        Compute hover action using PID controller.
        
        Args:
            observation: [x, y, z, vx, vy, vz, roll, pitch, yaw, omega_x, omega_y, omega_z, battery]
            
        Returns:
            action: [thrust, roll, pitch, yaw]
        """
        # Extract position from observation
        position = observation[:3]
        
        # Compute position error
        error = self.target_position - position
        
        # Derivative of error (use velocity if available, otherwise approximate)
        velocity = observation[3:6]
        d_error = -velocity
        
        # Integral of error
        self.integral += error * 0.01  # Assuming dt = 0.01
        
        # PID control
        p_term = self.p_gains * error
        i_term = self.i_gains * self.integral
        d_term = self.d_gains * d_error
        
        # Compute control signals (position adjustment)
        control = p_term + i_term + d_term
        
        # Convert to drone commands
        thrust = 0.5 + control[2]  # Base thrust + Z adjustment
        roll = -control[1]         # Roll to adjust Y position (inverted)
        pitch = control[0]         # Pitch to adjust X position
        yaw = 0.0                  # No yaw control for hovering
        
        # Clip values to action space
        thrust = np.clip(thrust, 0.0, 1.0)
        roll = np.clip(roll, -1.0, 1.0)
        pitch = np.clip(pitch, -1.0, 1.0)
        
        # Update PID state
        self.prev_error = error.copy()
        
        return np.array([thrust, roll, pitch, yaw])


class WaypointAgent(BaseAgent):
    """
    A rule-based agent that navigates through a series of waypoints.
    
    Uses the HoverAgent as a building block for each waypoint.
    """
    
    def __init__(self, action_space, waypoints=None):
        super().__init__(action_space)
        
        # Default waypoints: takeoff, hover, square pattern, land
        if waypoints is None:
            self.waypoints = [
                np.array([0.0, 0.0, 1.5]),    # Takeoff
                np.array([2.0, 0.0, 1.5]),    # Forward
                np.array([2.0, 2.0, 1.5]),    # Right
                np.array([0.0, 2.0, 1.5]),    # Back
                np.array([0.0, 0.0, 1.5]),    # Left
                np.array([0.0, 0.0, 0.5]),    # Land
            ]
        else:
            self.waypoints = [np.array(w) for w in waypoints]
        
        # Current waypoint index
        self.current_waypoint_idx = 0
        
        # Threshold for reaching waypoint
        self.waypoint_threshold = 0.2  # meters
        
        # Create hover agent for PID control
        self.hover_agent = HoverAgent(action_space, self.waypoints[0])
    
    def get_action(self, observation):
        """
        Navigate to the current waypoint using the hover agent.
        
        Args:
            observation: The current observation from the environment.
            
        Returns:
            action: An action to be taken in the environment.
        """
        # Extract position from observation
        position = observation[:3]
        
        # Check if we've reached the current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(position - current_waypoint)
        
        if distance < self.waypoint_threshold:
            # Advance to next waypoint
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            self.hover_agent.target_position = self.waypoints[self.current_waypoint_idx]
            self.hover_agent.integral = np.zeros(3)  # Reset integral term
        
        # Get action from hover agent
        return self.hover_agent.get_action(observation)
    
    def reset(self):
        """Reset the agent to the first waypoint."""
        self.current_waypoint_idx = 0
        self.hover_agent.target_position = self.waypoints[self.current_waypoint_idx]
        self.hover_agent.integral = np.zeros(3)
        self.hover_agent.prev_error = np.zeros(3)
