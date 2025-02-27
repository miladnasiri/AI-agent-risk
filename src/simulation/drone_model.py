"""
Physics-based model of a DJI Tello drone.
"""

import numpy as np
from scipy.integrate import solve_ivp


class TelloDroneModel:
    """
    A physics-based model of the DJI Tello drone that simulates its dynamics.
    
    This model includes:
    - 6 degrees of freedom (position: x, y, z and orientation: roll, pitch, yaw)
    - Realistic physics including gravity, drag, and motor forces
    - Battery simulation with discharge based on motor usage
    - Sensor simulation (IMU, barometer, camera)
    """
    
    # Tello drone physical specifications
    MASS = 0.080  # kg
    MAX_THRUST = 2.0 * MASS * 9.81  # N (2x gravity to allow for maneuvers)
    MAX_TORQUE = 0.10  # N·m
    DRAG_COEFFICIENT = 0.3
    MOMENT_OF_INERTIA = np.array([
        [1.43e-5, 0, 0],
        [0, 1.43e-5, 0],
        [0, 0, 2.89e-5]
    ])  # kg·m²
    
    # Motor specifications
    NUM_MOTORS = 4
    MOTOR_DISTANCE = 0.07  # m (distance from center to motor)
    MAX_MOTOR_SPEED = 30000  # RPM
    MOTOR_THRUST_COEFFICIENT = MAX_THRUST / (NUM_MOTORS * MAX_MOTOR_SPEED**2)
    MOTOR_TORQUE_COEFFICIENT = MAX_TORQUE / (NUM_MOTORS * MAX_MOTOR_SPEED**2)
    
    # Battery specifications
    BATTERY_CAPACITY = 1.1 * 3600  # Wh (1.1 Wh converted to Joules)
    MOTOR_POWER_COEFFICIENT = 0.05  # W/(RPM^2)
    
    def __init__(self):
        # State variables
        self.position = np.zeros(3)  # x, y, z in world frame
        self.velocity = np.zeros(3)  # vx, vy, vz in world frame
        self.orientation = np.zeros(3)  # roll, pitch, yaw (euler angles)
        self.angular_velocity = np.zeros(3)  # omega_x, omega_y, omega_z
        
        # Motor states
        self.motor_speeds = np.zeros(4)  # in RPM
        
        # Battery state
        self.battery_charge = self.BATTERY_CAPACITY  # in Joules
        self.battery_percentage = 100.0  # in percentage
        
        # Sensor readings
        self.acceleration = np.zeros(3)  # ax, ay, az
        self.gyro = np.zeros(3)  # gyro_x, gyro_y, gyro_z
        self.barometer = 0.0  # altitude in meters
        
        # Simulation parameters
        self.time = 0.0  # simulation time in seconds
        self.dt = 0.01  # simulation time step in seconds
        
        # Camera parameters
        self.camera_resolution = (960, 720)  # pixels
        self.camera_fov = 82.6  # degrees
        self.camera_fps = 30  # frames per second
        
        # Command inputs
        self.cmd_thrust = 0.0  # normalized thrust command [0, 1]
        self.cmd_roll = 0.0  # normalized roll command [-1, 1]
        self.cmd_pitch = 0.0  # normalized pitch command [-1, 1]
        self.cmd_yaw = 0.0  # normalized yaw command [-1, 1]
    
    def set_commands(self, thrust, roll, pitch, yaw):
        """Set control commands for the drone."""
        self.cmd_thrust = np.clip(thrust, 0.0, 1.0)
        self.cmd_roll = np.clip(roll, -1.0, 1.0)
        self.cmd_pitch = np.clip(pitch, -1.0, 1.0)
        self.cmd_yaw = np.clip(yaw, -1.0, 1.0)
        
        # Calculate desired motor speeds based on commands
        # This is a simplified mixing logic
        motor_1 = self.cmd_thrust + self.cmd_pitch + self.cmd_yaw
        motor_2 = self.cmd_thrust + self.cmd_roll - self.cmd_yaw
        motor_3 = self.cmd_thrust - self.cmd_pitch + self.cmd_yaw
        motor_4 = self.cmd_thrust - self.cmd_roll - self.cmd_yaw
        
        # Scale to RPM range and clip
        motor_commands = np.array([motor_1, motor_2, motor_3, motor_4])
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        self.motor_speeds = motor_commands * self.MAX_MOTOR_SPEED
    
    def _compute_forces_and_torques(self):
        """Compute forces and torques based on current motor speeds."""
        gravity = np.array([0, 0, -9.81 * self.MASS])
        
        # Compute rotation matrix from body to world frame
        roll, pitch, yaw = self.orientation
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x  # Body to world rotation matrix
        
        # Compute individual motor thrusts
        motor_thrusts = self.MOTOR_THRUST_COEFFICIENT * self.motor_speeds**2
        
        # Total thrust in body frame (aligned with z-axis)
        thrust_body = np.array([0, 0, np.sum(motor_thrusts)])
        
        # Convert thrust to world frame
        thrust_world = R @ thrust_body
        
        # Compute drag force (proportional to velocity squared)
        velocity_squared = self.velocity**2 * np.sign(self.velocity)
        drag = -self.DRAG_COEFFICIENT * velocity_squared
        
        # Total force in world frame
        total_force = gravity + thrust_world + drag
        
        # Compute torques in body frame
        # Layout:     m1(front)
        #         m4          m2
        #              m3
        # Where +x is forward, +y is right
        
        # Motor distance vectors from CoG in body frame
        r1 = np.array([self.MOTOR_DISTANCE, 0, 0])
        r2 = np.array([0, self.MOTOR_DISTANCE, 0])
        r3 = np.array([-self.MOTOR_DISTANCE, 0, 0])
        r4 = np.array([0, -self.MOTOR_DISTANCE, 0])
        
        # Motor thrust vectors in body frame
        f1 = np.array([0, 0, motor_thrusts[0]])
        f2 = np.array([0, 0, motor_thrusts[1]])
        f3 = np.array([0, 0, motor_thrusts[2]])
        f4 = np.array([0, 0, motor_thrusts[3]])
        
        # Torques from motor forces
        tau1 = np.cross(r1, f1)
        tau2 = np.cross(r2, f2)
        tau3 = np.cross(r3, f3)
        tau4 = np.cross(r4, f4)
        
        # Motor reaction torques (due to propeller rotation)
        # Motors 1 and 3 rotate CW, Motors 2 and 4 rotate CCW
        reaction1 = np.array([0, 0, -self.MOTOR_TORQUE_COEFFICIENT * self.motor_speeds[0]**2])
        reaction2 = np.array([0, 0, self.MOTOR_TORQUE_COEFFICIENT * self.motor_speeds[1]**2])
        reaction3 = np.array([0, 0, -self.MOTOR_TORQUE_COEFFICIENT * self.motor_speeds[2]**2])
        reaction4 = np.array([0, 0, self.MOTOR_TORQUE_COEFFICIENT * self.motor_speeds[3]**2])
        
        # Total torque
        total_torque = tau1 + tau2 + tau3 + tau4 + reaction1 + reaction2 + reaction3 + reaction4
        
        return total_force, total_torque
    
    def _compute_battery_consumption(self, dt):
        """Compute battery consumption based on motor usage."""
        # Power consumption of each motor
        motor_power = self.MOTOR_POWER_COEFFICIENT * self.motor_speeds**2
        total_power = np.sum(motor_power)
        
        # Energy consumption in this time step
        energy_consumed = total_power * dt
        
        # Update battery charge
        self.battery_charge -= energy_consumed
        self.battery_charge = max(0.0, self.battery_charge)
        
        # Update battery percentage
        self.battery_percentage = (self.battery_charge / self.BATTERY_CAPACITY) * 100.0
    
    def _update_sensors(self):
        """Update simulated sensor readings."""
        # Acceleration in world frame
        self.acceleration = self.velocity - self._prev_velocity
        
        # Gyroscope readings (angular velocity in body frame)
        self.gyro = self.angular_velocity
        
        # Barometer (altitude)
        self.barometer = self.position[2]
    
    def _state_derivative(self, t, state):
        """
        Compute the derivative of the state vector for numerical integration.
        
        State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, omega_x, omega_y, omega_z]
        """
        # Unpack state vector
        pos = state[0:3]
        vel = state[3:6]
        orient = state[6:9]
        omega = state[9:12]
        
        # Compute forces and torques
        total_force, total_torque = self._compute_forces_and_torques()
        
        # Linear acceleration
        acc = total_force / self.MASS
        
        # Angular acceleration
        omega_cross = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        inertia_omega = self.MOMENT_OF_INERTIA @ omega
        alpha = np.linalg.inv(self.MOMENT_OF_INERTIA) @ (total_torque - omega_cross @ inertia_omega)
        
        # Euler angle derivatives
        # This is a simplified model and might exhibit gimbal lock
        roll, pitch, yaw = orient
        roll_dot = omega[0] + omega[1] * np.sin(roll) * np.tan(pitch) + omega[2] * np.cos(roll) * np.tan(pitch)
        pitch_dot = omega[1] * np.cos(roll) - omega[2] * np.sin(roll)
        yaw_dot = omega[1] * np.sin(roll) / np.cos(pitch) + omega[2] * np.cos(roll) / np.cos(pitch)
        
        # Build state derivative
        deriv = np.zeros_like(state)
        deriv[0:3] = vel
        deriv[3:6] = acc
        deriv[6:9] = [roll_dot, pitch_dot, yaw_dot]
        deriv[9:12] = alpha
        
        return deriv
    
    def step(self, dt=None):
        """
        Advance the simulation by one time step.
        
        Args:
            dt: Time step in seconds. If None, use the default time step.
        """
        if dt is not None:
            self.dt = dt
        
        # Store previous velocity for acceleration calculation
        self._prev_velocity = self.velocity.copy()
        
        # Pack state vector
        state = np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity
        ])
        
        # Integrate state using scipy's ODE solver
        sol = solve_ivp(
            self._state_derivative,
            [self.time, self.time + self.dt],
            state,
            method='RK45',
            t_eval=[self.time + self.dt]
        )
        
        # Update state
        new_state = sol.y[:, 0]
        self.position = new_state[0:3]
        self.velocity = new_state[3:6]
        self.orientation = new_state[6:9]
        self.angular_velocity = new_state[9:12]
        
        # Update time
        self.time += self.dt
        
        # Update battery
        self._compute_battery_consumption(self.dt)
        
        # Update sensors
        self._update_sensors()
        
        # Return current state as a dictionary
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'battery': self.battery_percentage,
            'time': self.time
        }
    
    def reset(self, position=None, orientation=None):
        """Reset the drone to initial state."""
        # Reset state variables
        self.position = np.zeros(3) if position is None else np.array(position)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3) if orientation is None else np.array(orientation)
        self.angular_velocity = np.zeros(3)
        
        # Reset motor states
        self.motor_speeds = np.zeros(4)
        
        # Reset battery
        self.battery_charge = self.BATTERY_CAPACITY
        self.battery_percentage = 100.0
        
        # Reset sensor readings
        self.acceleration = np.zeros(3)
        self.gyro = np.zeros(3)
        self.barometer = self.position[2]
        
        # Reset simulation parameters
        self.time = 0.0
        
        # Reset command inputs
        self.cmd_thrust = 0.0
        self.cmd_roll = 0.0
        self.cmd_pitch = 0.0
        self.cmd_yaw = 0.0
        
        # Initialize previous velocity for acceleration computation
        self._prev_velocity = np.zeros(3)
