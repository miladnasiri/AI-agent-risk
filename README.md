# DJTello Digital Twin

A sophisticated digital twin simulation of a DJI Tello drone with advanced AI agent integration for autonomous control, navigation, and complex task execution.

![DJTello Digital Twin](Screenshot%20from%202025-02-28%2001-07-27.png)

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Architecture](#technical-architecture)
- [System Components](#system-components)
- [AI Agent Framework](#ai-agent-framework)
- [Ai-Agents-Overview](#ai-agents-overview.md)
- [Physics Simulation](#physics-simulation)
- [Control Systems](#control-systems)
- [Methodologies](#methodologies)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Development Workflow](#development-workflow)
- [Future Roadmap](#future-roadmap)
- [Technical Documentation](#technical-documentation)
- [License](#license)

## Project Overview

DJTello Digital Twin is an advanced simulation platform that creates a high-fidelity virtual replica of a DJI Tello drone, complete with realistic physics, sensor simulations, and a suite of AI agents for autonomous control. This digital twin allows for rapid prototyping, training, and testing of drone control algorithms in a safe, virtual environment before deployment to physical hardware.

### Goals

1. **High-Fidelity Simulation**: Create a realistic physics-based model of the DJI Tello drone that accurately simulates flight dynamics, motor characteristics, battery discharge, and sensor responses.

2. **AI Agent Development**: Implement multiple AI agent architectures ranging from rule-based systems to advanced deep reinforcement learning models that can control the drone autonomously.

3. **Reinforcement Learning Platform**: Provide a standardized environment compatible with the Gymnasium interface for training and evaluating reinforcement learning algorithms.

4. **Visualization & Analysis**: Deliver 3D visualization of drone behavior along with data collection and analysis tools to understand and improve agent performance.

5. **Real-to-Virtual Transfer**: Enable algorithms developed in the simulation to transfer effectively to real DJI Tello drones with minimal adaptation.

6. **Expandable Framework**: Create a modular architecture that can be expanded to include additional sensors, environmental conditions, and multi-drone scenarios.

## Technical Architecture

The DJTello Digital Twin follows a modular, layered architecture that separates concerns while enabling tight integration between components.

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      DJTello Digital Twin                     │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                         Core Framework                        │
├──────────────┬──────────────┬───────────────┬───────────────┤
│ Drone Model  │ Environment  │ Visualization │ Task Manager  │
└──────────────┴──────────────┴───────────────┴───────────────┘
                                │
           ┌──────────────────────────────────────┐
           ▼                                      ▼
┌────────────────────┐                 ┌────────────────────┐
│   Physics Engine   │                 │    Agent Layer     │
├────────────────────┤                 ├────────────────────┤
│ - Rigid Body       │                 │ - Base Agent       │
│ - Aerodynamics     │                 │ - Rule-based       │
│ - Motor Dynamics   │                 │ - RL Agents        │
│ - Collision        │                 │ - Deep Agents      │
└────────────────────┘                 └────────────────────┘
                                              │
                                              ▼
                                     ┌────────────────────┐
                                     │   Training System  │
                                     ├────────────────────┤
                                     │ - Replay Buffer    │
                                     │ - Policy Training  │
                                     │ - Evaluation       │
                                     │ - Model Management │
                                     └────────────────────┘
```

### Component Interaction Flow

```
┌────────────┐    Commands     ┌────────────┐    Observations    ┌────────────┐
│            │ ──────────────> │            │ ───────────────>   │            │
│   Agent    │                 │  Digital   │                    │  Observer/ │
│            │ <────────────── │   Twin     │ <─────────────────┤  Renderer  │
└────────────┘    Feedback     └────────────┘    Visualization   └────────────┘
       │                               │                              │
       │                               │                              │
       ▼                               ▼                              ▼
┌────────────┐                ┌────────────┐                  ┌────────────┐
│   Policy   │                │  Physics   │                  │ Monitoring │
│   Model    │                │  Engine    │                  │ & Analysis │
└────────────┘                └────────────┘                  └────────────┘
```

## System Components

### 1. Drone Simulation Module

The core simulation module implements a high-fidelity model of the DJI Tello drone with the following components:

#### 1.1 Physical Specifications

- **Mass**: 80 grams
- **Dimensions**: 98×92.5×41 mm
- **Motor Configuration**: 4 brushed motors
- **Maximum Thrust**: 2.0 * Mass * 9.81 N
- **Battery**: 1.1 Wh LiPo battery with realistic discharge model
- **Moments of Inertia**: Calculated based on drone weight distribution

#### 1.2 Dynamic Model

- **Degrees of Freedom**: 6-DOF model (position: x, y, z and orientation: roll, pitch, yaw)
- **State Vector**: [x, y, z, vx, vy, vz, roll, pitch, yaw, ω_roll, ω_pitch, ω_yaw]
- **Euler Integration**: Numerical integration of equations of motion using RK45
- **Aerodynamic Effects**: Drag coefficients, propeller airflow, ground effect

#### 1.3 Sensor Simulation

- **IMU**: Accelerometer and gyroscope with realistic noise models
- **Barometer**: Pressure-based altitude estimation
- **Downward Vision System**: Optical flow for position holding
- **Camera**: Forward-facing camera with configurable resolution and FPS

#### 1.4 Environment Integration

- **Gymnasium Interface**: Standard RL environment compliant with Gymnasium API
- **Observation Space**: Customizable state observations
- **Action Space**: Direct control of [thrust, roll, pitch, yaw] commands
- **Reward Functions**: Task-specific reward structures for RL training

### 2. AI Agent Framework

The agent framework provides multiple implementations for controlling the drone:

#### 2.1 Base Agent

Abstract class defining the common interface for all agent types:
- `get_action(observation)`: Core method to produce actions
- `train()` / `eval()`: Mode switching
- `save()` / `load()`: Model persistence

#### 2.2 Rule-Based Agents

- **RandomAgent**: Baseline agent producing random actions
- **HoverAgent**: PID controller maintaining position at a set point
- **WaypointAgent**: Navigation between specified waypoints

#### 2.3 Reinforcement Learning Agents

- **PPOAgent**: Proximal Policy Optimization implementation
- **SACAgent**: Soft Actor-Critic implementation
- **TD3Agent**: Twin Delayed DDPG implementation

#### 2.4 Deep Learning Agent

Advanced implementation with:
- Actor-Critic architecture with separate policy and value networks
- Automatic entropy tuning for optimal exploration
- Experience replay with prioritized sampling
- Policy gradient training with value bootstrapping

### 3. Visualization System

- **OpenGL Renderer**: 3D visualization of the drone and environment
- **Camera Controls**: Configurable viewpoint and following modes
- **Telemetry Display**: Real-time display of drone state
- **Recording**: Video capture of simulation runs

## AI Agent Framework

The AI agent framework is designed to accommodate multiple approaches to drone control, from simple rule-based systems to advanced deep learning techniques.

### Agent Hierarchy

```
                        ┌───────────────┐
                        │  BaseAgent    │
                        └───────────────┘
                             ▲     ▲
               ┌─────────────┘     └───────────┐
               │                               │
      ┌────────────────┐              ┌────────────────┐
      │  Rule-Based    │              │    Learning    │
      │    Agents      │              │     Agents     │
      └────────────────┘              └────────────────┘
               ▲                               ▲
       ┌───────┴───────┐                ┌──────┴───────┐
       │               │                │              │
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ RandomAgent │ │ HoverAgent  │ │  RLAgent    │ │ DeepAgent   │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                                        ▲
                         ┌──────────────┼──────────────┐
                         │              │              │
                  ┌─────────────┐┌─────────────┐┌─────────────┐
                  │  PPOAgent   ││  SACAgent   ││  TD3Agent   │
                  └─────────────┘└─────────────┘└─────────────┘
```

### Learning Agent Architecture

```
                      ┌───────────────────┐
                      │   Environment     │
                      └─────────┬─────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                  Agent Processing Pipeline                 │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ Observation │    │   Policy    │    │   Action    │   │
│  │ Processing  │───>│   Network   │───>│  Execution  │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
│          ▲                │                  │           │
│          │                ▼                  │           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │  Replay     │<───│    Value    │<───│   Reward    │   │
│  │  Buffer     │    │   Network   │    │ Calculation │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
└───────────────────────────────────────────────────────────┘
                │                 │
                ▼                 ▼
      ┌─────────────────┐ ┌─────────────────┐
      │  Policy Update  │ │  Value Update   │
      └─────────────────┘ └─────────────────┘
```

### Neural Network Architectures

For deep learning agents, the following network architectures are implemented:

#### Actor Network (Policy)
```
Input Layer (State Dimension) → Dense(256) → ReLU
                               → Dense(256) → ReLU
                               → [Mean Layer, Log Std Layer]
                               → Tanh Output (Action)
```

#### Critic Network (Value)
```
Input Layer (State+Action) → Dense(256) → ReLU
                           → Dense(256) → ReLU
                           → Dense(1) → Q-Value
```

## Physics Simulation

The physics simulation is at the core of the digital twin, providing high-fidelity modeling of drone behavior.

### Physics Update Cycle

```
┌────────────────┐
│  Input Forces  │
│  and Torques   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ State Dynamics │
│  Computation   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Numerical    │
│  Integration   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ State Update & │
│ Sensor Models  │
└────────────────┘
```

### Force and Torque Models

- **Gravity**: Applied in world frame Z-axis
- **Motor Thrust**: Individual motor thrust based on motor speed
- **Drag**: Aerodynamic drag proportional to velocity squared
- **Motor Torques**: Torque produced by each motor's rotation
- **Reaction Torques**: Counter-torque from motor rotation

### Integration Method

The simulation uses `scipy.integrate.solve_ivp` with RK45 method for numerical integration of the system's ordinary differential equations:

1. **State Vector**: x = [position, velocity, orientation, angular_velocity]
2. **Derivative Function**: dx/dt = f(x, u), where u are the control inputs
3. **Update Equation**: x(t+dt) = x(t) + ∫f(x,u)dt from t to t+dt

## Control Systems

### PID Control Implementation

For rule-based agents like the HoverAgent, a PID control system is implemented:

```
Error = Target Position - Current Position

P_term = Kp * Error
I_term = Ki * ∫Error dt
D_term = Kd * d(Error)/dt

Control Signal = P_term + I_term + D_term
```

With the following default gain values:
- Position: Kp = [0.5, 0.5, 0.8], Ki = [0.01, 0.01, 0.02], Kd = [0.2, 0.2, 0.4]
- Attitude: Kp = [2.0, 2.0, 1.0], Ki = [0.1, 0.1, 0.1], Kd = [0.1, 0.1, 0.5]

### Waypoint Navigation

The WaypointAgent uses this control system with a sequence of waypoints:

1. Set target position to current waypoint
2. Use PID controller to navigate to target
3. When within threshold distance (default: 0.2m), advance to next waypoint
4. Repeat until all waypoints are visited

## Methodologies

### Development Methodology

The DJTello Digital Twin was developed using the following methodology:

1. **Requirements Analysis**: Identifying key features and capabilities needed
2. **System Design**: Creating modular architecture and component interfaces
3. **Implementation**: Building system components in order of dependency
   - Physics model → Environment → Visualization → Agents → Training
4. **Testing**: Unit tests for components and integration tests for the system
5. **Validation**: Comparing simulation behavior to real drone specifications
6. **Iteration**: Refining components based on test results and performance

### Agent Training Methodology

The reinforcement learning agents follow this training methodology:

1. **Environment Setup**: Configure task, reward function, and episode parameters
2. **Exploration Phase**: Initial random actions to populate replay buffer
3. **Policy Learning Phase**: Update policy based on collected experiences
4. **Evaluation**: Periodically evaluate performance on test episodes
5. **Model Selection**: Save best models based on evaluation metrics
6. **Hyperparameter Tuning**: Grid search over key parameters

### Task-Specific Reward Functions

Each task has a specialized reward function:

#### Hover Task
```
reward = 0.1                   # Base survival reward
reward -= |position - target|  # Position error penalty
reward -= |orientation|        # Orientation error penalty
reward += 0.5 if position_error < 0.1  # Precision bonus
```

#### Waypoint Task
```
reward = 1.0 / (1.0 + distance)  # Inverse distance to waypoint
reward += 10.0 if waypoint_reached  # Waypoint completion bonus
reward -= 0.1 * |horizontal_velocity|  # Smooth movement bonus
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- OpenGL compatible graphics card
- 4GB RAM minimum (8GB recommended)
- 2GB disk space

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/djtello-digital-twin.git
   cd djtello-digital-twin
   ```

2. Run the setup script:
   ```bash
   chmod +x startup.sh
   ./startup.sh
   ```

3. Verify installation:
   ```bash
   source venv/bin/activate
   python -m src simulation --task hover --render
   ```

### Dependencies

| Package               | Version    | Purpose                          |
|-----------------------|------------|----------------------------------|
| numpy                 | >=1.20.0   | Numerical computation            |
| scipy                 | >=1.7.0    | Scientific calculations          |
| pygame                | >=2.0.0    | Window management                |
| gymnasium             | >=0.26.0   | RL environment interface         |
| stable-baselines3     | >=1.5.0    | RL algorithm implementations     |
| torch                 | >=1.10.0   | Deep learning framework          |
| pybullet              | >=3.2.1    | Collision detection              |
| opencv-python         | >=4.5.0    | Image processing                 |
| pillow                | >=8.0.0    | Image manipulation               |
| tqdm                  | >=4.60.0   | Progress display                 |
| pytest                | >=6.2.5    | Testing framework                |

## Usage Guide

### Basic Usage

To run the digital twin with a specific task:

```bash
python -m src simulation --task hover --render
```

Available tasks:
- `hover`: Maintain position at a fixed point
- `takeoff`: Ascend to a target height
- `land`: Descend safely to the ground
- `waypoint`: Navigate through a series of waypoints
- `square`: Fly in a square pattern
- `circle`: Fly in a circular pattern
- `flip`: Perform flip maneuvers

### Training an AI Agent

To train a reinforcement learning agent:

```bash
python -m src train --algorithm ppo --task hover --timesteps 500000 --render
```

Available algorithms:
- `ppo`: Proximal Policy Optimization
- `sac`: Soft Actor-Critic
- `td3`: Twin Delayed DDPG
- `deep`: Custom deep agent implementation

### Using a Trained Agent

To run a simulation with a trained agent:

```bash
python -m src control --agent ppo --task waypoint --model-path models/ppo_waypoint_latest/final_model.zip
```

Available agents:
- `random`: Random action agent (baseline)
- `hover`: PID-based hover controller
- `waypoint`: Waypoint navigation with PID control
- `ppo`, `sac`, `td3`, `deep`: Trained reinforcement learning agents

### Recording Simulations

To record videos of the simulation:

```bash
python -m src control --agent ppo --task circle --model-path models/ppo_circle/model.zip --record
```

### Advanced Configuration

Create a custom configuration file (`config.yaml`):

```yaml
environment:
  max_steps: 2000
  bounds:
    x: [-10, 10]
    y: [-10, 10]
    z: [0, 10]
  
drone:
  mass: 0.08  # kg
  motor_distance: 0.07  # m
  battery_capacity: 1.1  # Wh
  
visualization:
  width: 1280
  height: 720
  camera_distance: 5.0
  
agent:
  type: "sac"
  model_path: "./models/sac_waypoint/model.zip"
  
task:
  name: "waypoint"
  waypoints:
    - [0.0, 0.0, 1.5]
    - [2.0, 0.0, 1.5]
    - [2.0, 2.0, 1.5]
    - [0.0, 2.0, 1.5]
```

Then run with the config file:

```bash
python -m src run --config config.yaml
```

## Development Workflow

### Project Structure

```
djtello-digital-twin/
├── src/                    # Source code
│   ├── simulation/         # Drone physics simulation
│   │   ├── __init__.py
│   │   ├── drone_model.py  # Physical drone model
│   │   └── environment.py  # Gymnasium environment
│   ├── agent/              # AI agent implementation
│   │   ├── __init__.py
│   │   ├── base_agent.py   # Agent interface
│   │   ├── rl_agent.py     # RL agents
│   │   ├── deep_agent.py   # Advanced deep agent
│   │   ├── train.py        # Training script
│   │   └── control.py      # Agent control script
│   ├── visualization/      # 3D visualization
│   │   ├── __init__.py
│   │   └── drone_renderer.py
│   ├── __init__.py
│   └── __main__.py         # Main entry point
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── data/                   # Training data and models
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup file
├── startup.sh              # Setup script
└── README.md               # This file
```

### Contributing Guidelines

1. **Branching Strategy**:
   - `main`: Stable release branch
   - `develop`: Development branch
   - `feature/*`: Feature branches
   - `bugfix/*`: Bug fix branches

2. **Commit Message Format**:
   ```
   [Component] Brief description
   
   Detailed explanation if needed
   ```

3. **Code Style**:
   - Follow PEP 8 guidelines
   - Use docstrings for all classes and functions
   - Maintain test coverage above 80%

4. **Pull Request Process**:
   - Create PR against `develop` branch
   - Ensure all tests pass
   - Obtain at least one code review
   - Squash and merge

## Future Roadmap

### Phase 1: Core Enhancement
- [ ] Improved collision detection with environment objects
- [ ] Enhanced aerodynamic models including wind effects
- [ ] Support for multiple drone types beyond Tello

### Phase 2: Advanced Features
- [ ] Swarm simulation with multiple coordinated drones
- [ ] Computer vision-based navigation tasks
- [ ] Integration with real Tello SDK for hardware deployment

### Phase 3: Ecosystem Development
- [ ] Web-based visualization and remote control
- [ ] Community model sharing platform
- [ ] Integration with popular robotics frameworks (ROS, etc.)

## Technical Documentation

### Agent API Reference

```python
class BaseAgent:
    def get_action(self, observation):
        """
        Get action based on current observation.
        
        Args:
            observation: numpy.ndarray - Current observation
        
        Returns:
            numpy.ndarray - Action to take
        """
        pass
    
    def train(self):
        """Set agent to training mode."""
        pass
    
    def eval(self):
        """Set agent to evaluation mode."""
        pass
    
    def save(self, path):
        """Save agent to disk."""
        pass
    
    def load(self, path):
        """Load agent from disk."""
        pass
```

### Environment Interface

```python
class TelloEnv(gym.Env):
    def __init__(self, render_mode=None, max_episode_steps=1000):
        """
        Initialize environment.
        
        Args:
            render_mode: str - 'human' or 'rgb_array'
            max_episode_steps: int - Maximum steps per episode
        """
        pass
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Returns:
            tuple - (observation, info)
        """
        pass
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: numpy.ndarray - Action to take
        
        Returns:
            tuple - (observation, reward, terminated, truncated, info)
        """
        pass
    
    def render(self):
        """Render the environment."""
        pass
    
    def close(self):
        """Close the environment."""
        pass
```

### Camera Projection Model

The camera projection model follows the standard pinhole camera model:

```
x_image = fx * X / Z + cx
y_image = fy * Y / Z + cy
```

Where:
- `(X, Y, Z)` are 3D world coordinates
- `(x_image, y_image)` are image coordinates
- `fx, fy` are focal lengths
- `cx, cy` are principal point offsets

Default parameters for the Tello camera:
- Resolution: 960x720 pixels
- Field of View: 82.6 degrees
- Frame Rate: 30 FPS

## License

MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
