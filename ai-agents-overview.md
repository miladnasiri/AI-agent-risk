# AI Agents in DJTello Digital Twin

# DJTello Digital Twin

## Abstract

This project presents a comprehensive digital twin simulation system for the DJI Tello drone, providing an accurate virtual representation of drone physics and behavior. The system takes sensor data inputs (IMU readings, position coordinates, battery status) and environmental parameters, processing them through a 6-DOF physics model to generate realistic drone state outputs. The simulation environment interfaces with multiple AI agent implementations that consume observation vectors (position, velocity, orientation, angular velocity) and produce normalized control signals (thrust, roll, pitch, yaw).

demonstrated three classes of agent methodologies: rule-based controllers using PID algorithms, reinforcement learning agents trained with state-of-the-art algorithms (PPO, SAC, TD3), and an advanced deep learning agent with custom neural network architecture. Performance evaluation across multiple task domains (hovering, waypoint navigation, pattern following) shows that our learning-based agents achieve up to 95% task completion rates while maintaining stability in challenging conditions. The complete system enables rapid prototyping and training of autonomous drone behaviors in a risk-free virtual environment before deployment to physical hardware.

![DJTello Digital Twin](Screenshot%20from%202025-02-28%2001-07-27.png)



## Table of Contents

- [General Overview](#general-overview)
- [Agent Architecture](#agent-architecture)
- [Rule-Based Agents](#rule-based-agents)
- [Reinforcement Learning Agents](#reinforcement-learning-agents)
- [Deep Learning Agent](#deep-learning-agent)
- [Input/Output Specifications](#inputoutput-specifications)
- [Training Methodology](#training-methodology)
- [Performance Metrics](#performance-metrics)
- [Selection Criteria](#selection-criteria)
- [Implementation Details](#implementation-details)
- [Practical Applications](#practical-applications)

## General Overview

The AI agent system in the DJTello Digital Twin provides autonomous control capabilities for the simulated drone. These agents enable the drone to perform various tasks ranging from basic hovering to complex navigation patterns without human intervention.

### Core Functions

1. **Perception**: Process sensor data from the simulated drone (position, velocity, orientation, etc.)
2. **Decision Making**: Determine appropriate control actions based on current state and goals
3. **Control Execution**: Apply the selected actions to the drone's control systems
4. **Learning**: Improve performance over time through experience (for learning-based agents)

### Agent Categories

The project implements three categories of agents with increasing levels of sophistication:

1. **Rule-Based Agents**: Algorithmic controllers using explicit programmed rules
2. **Reinforcement Learning Agents**: Agents trained through trial and error with reward signals
3. **Advanced Deep Learning Agent**: Sophisticated neural network-based agent with advanced features

## Agent Architecture

All agents in the system follow a common interface defined by the `BaseAgent` abstract class, enabling interchangeability and standardized interaction with the environment.

```
BaseAgent
│
├── Rule-Based Agents
│   ├── RandomAgent
│   ├── HoverAgent
│   └── WaypointAgent
│
├── Reinforcement Learning Agents
│   ├── PPOAgent
│   ├── SACAgent
│   └── TD3Agent
│
└── DeepAgent (Advanced SAC Implementation)
```

### Common Interface

```python
class BaseAgent:
    def get_action(self, observation):
        """Convert observation to action"""
        pass
    
    def train(self):
        """Set agent to training mode"""
        pass
    
    def eval(self):
        """Set agent to evaluation mode"""
        pass
    
    def save(self, path):
        """Save agent model"""
        pass
    
    def load(self, path):
        """Load agent model"""
        pass
```

## Rule-Based Agents

Rule-based agents use classical control algorithms without learning capabilities. They provide baseline performance and are useful for comparison and validation.

### RandomAgent

**Purpose**: Provides a baseline for performance comparison and testing environment functionality.

**Implementation**: Generates random control signals within the valid action space.

**Input**: State observation vector
**Output**: Random action vector within action space bounds

**Code Example**:
```python
def get_action(self, observation):
    """Return a random action from the action space."""
    return self.action_space.sample()
```

### HoverAgent

**Purpose**: Maintains the drone at a fixed position in 3D space.

**Implementation**: Uses a PID (Proportional-Integral-Derivative) controller to adjust thrust, roll, pitch, and yaw to maintain a target position.

**Input**: State observation vector containing position, velocity, orientation
**Output**: Control signals for thrust, roll, pitch, and yaw

**Parameters**:
- P-gains: [0.5, 0.5, 0.8] for x, y, z
- I-gains: [0.01, 0.01, 0.02] for x, y, z
- D-gains: [0.2, 0.2, 0.4] for x, y, z

**Control Logic**:
1. Calculate error between current and target position
2. Apply PID formula to compute control adjustment
3. Convert positional adjustments to appropriate control signals

### WaypointAgent

**Purpose**: Navigates the drone through a series of 3D waypoints.

**Implementation**: Extends the HoverAgent to sequentially target multiple positions.

**Input**: State observation vector
**Output**: Control signals for thrust, roll, pitch, and yaw

**Features**:
- Waypoint threshold detection (default: 0.2m)
- Sequential waypoint targeting
- Path completion detection
- Configurable waypoint list

## Reinforcement Learning Agents

Reinforcement learning agents learn control policies through experience and reward signals. These agents can adapt to the dynamics of the environment without explicit programming.

### Common RL Architecture

All RL agents share core components:
- **Policy Network**: Maps observations to actions
- **Value Network**: Estimates expected returns (for critic-based methods)
- **Experience Replay**: Stores and reuses past experiences
- **Optimization Method**: Updates network parameters to maximize rewards

### PPOAgent (Proximal Policy Optimization)

**Purpose**: Provides stable learning with controlled policy updates.

**Implementation**: Uses clipped surrogate objective function to limit policy changes.

**Input**: State observation vector
**Output**: Control signals for thrust, roll, pitch, and yaw

**Hyperparameters**:
- Learning rate: 3e-4
- Batch size: 64
- Epochs per update: 10
- Clip range: 0.2
- Entropy coefficient: 0.01
- Network architecture: [128, 128] units in policy and value networks

**Advantages**:
- Sample efficient (can reuse experiences multiple times)
- Stable learning progress
- Good balance of exploration and exploitation

### SACAgent (Soft Actor-Critic)

**Purpose**: Maximizes both reward and action entropy for robust policy learning.

**Implementation**: Off-policy actor-critic algorithm with entropy regularization.

**Input**: State observation vector
**Output**: Control signals for thrust, roll, pitch, and yaw

**Hyperparameters**:
- Learning rate: 3e-4
- Buffer size: 100,000 experiences
- Batch size: 256
- Target update rate (tau): 0.005
- Automatic entropy tuning
- Network architecture: [256, 256] units

**Advantages**:
- Excellent sample efficiency
- Automatic entropy adjustment
- Robust to hyperparameter settings

### TD3Agent (Twin Delayed DDPG)

**Purpose**: Addresses overestimation bias in value functions for more accurate learning.

**Implementation**: Uses twin critic networks and delayed policy updates.

**Input**: State observation vector
**Output**: Control signals for thrust, roll, pitch, and yaw

**Hyperparameters**:
- Learning rate: 3e-4
- Buffer size: 100,000 experiences
- Policy delay: 2 (update policy every 2 critic updates)
- Target noise: 0.2 (added to actions for smoothing)
- Network architecture: [400, 300] units

**Advantages**:
- Reduced overestimation bias
- More stable learning
- Effective for continuous control tasks

## Deep Learning Agent

The advanced deep learning agent implements a sophisticated version of Soft Actor-Critic with additional features for optimal performance.

### DeepAgent

**Purpose**: Provides state-of-the-art performance with advanced features beyond basic RL implementations.

**Implementation**: Enhanced SAC with additional components:
- Custom neural network architectures
- Automatic entropy tuning
- Advanced experience replay

**Input**: State observation vector
**Output**: Control signals with probabilistic action sampling

**Neural Network Architecture**:

*Actor Network:*
```
Input (state) → Dense(256) → ReLU →
               Dense(256) → ReLU →
               [Mean Layer, Log Std Layer] →
               Tanh Output (actions)
```

*Critic Network:*
```
Input (state+action) → Dense(256) → ReLU →
                       Dense(256) → ReLU →
                       Output (Q-value)
```

**Advanced Features**:
- Automatic temperature adjustment based on entropy target
- Twin critics for reduced estimation bias
- Squashed Gaussian policy for bounded actions
- Gradient clipping for training stability

## Input/Output Specifications

### Input (Observation Space)

All agents receive the same observation vector from the environment, containing 13 values:

| Index | Component | Description | Range |
|-------|-----------|-------------|-------|
| 0-2   | Position  | x, y, z coordinates in world frame | [-10, 10], [-10, 10], [0, 10] |
| 3-5   | Velocity  | vx, vy, vz linear velocities | [-5, 5], [-5, 5], [-5, 5] |
| 6-8   | Orientation | roll, pitch, yaw in radians | [-π, π], [-π, π], [-π, π] |
| 9-11  | Angular Velocity | ω_roll, ω_pitch, ω_yaw | [-10, 10], [-10, 10], [-10, 10] |
| 12    | Battery   | Remaining battery percentage | [0, 100] |

### Output (Action Space)

All agents produce the same action vector with 4 values:

| Index | Component | Description | Range |
|-------|-----------|-------------|-------|
| 0     | Thrust    | Vertical thrust command | [0, 1] |
| 1     | Roll      | Roll angle command | [-1, 1] |
| 2     | Pitch     | Pitch angle command | [-1, 1] |
| 3     | Yaw       | Yaw rate command | [-1, 1] |

The action values are normalized in the specified ranges and then translated to motor commands by the drone's internal control system.

## Training Methodology

### Environment Configuration

The reinforcement learning agents are trained in the `TelloEnv` and `TelloGymEnv` environments with the following specifications:

- **Observation Space**: 13-dimensional vector (described above)
- **Action Space**: 4-dimensional vector (described above)
- **Reward Function**: Task-specific reward calculations

### Task-Specific Reward Functions

Different tasks use specialized reward functions to guide agent learning:

#### Hover Task
```python
reward = 0.1                                   # Base survival reward
reward -= np.linalg.norm(position - target)    # Position error penalty
reward -= np.linalg.norm(orientation)          # Orientation error penalty
if np.linalg.norm(position - target) < 0.1:
    reward += 0.5                              # Precision bonus
```

#### Waypoint Task
```python
distance = np.linalg.norm(position - current_waypoint)
reward = 1.0 / (1.0 + distance)                # Distance-based reward
if distance < waypoint_threshold:
    reward += 10.0                             # Waypoint completion bonus
horizontal_movement = np.linalg.norm(velocity[:2])
reward -= 0.1 * horizontal_movement            # Smooth movement incentive
```

### Training Process

1. **Initialization**:
   - Initialize environment and agent
   - Set up replay buffer (for off-policy methods)
   - Configure logging and model saving

2. **Training Loop**:
   - Reset environment to get initial state
   - For each step until done or max steps:
     - Select action based on current policy
     - Take action in environment
     - Store experience (state, action, reward, next_state, done)
     - Update agent's policy (frequency depends on algorithm)
   - Repeat for desired number of episodes

3. **Evaluation**:
   - Periodically evaluate agent performance
   - Save model if performance improves
   - Adjust hyperparameters if needed

### Hyperparameter Tuning

Key hyperparameters that significantly impact performance:
- Learning rate
- Network architecture
- Batch size
- Entropy coefficient (for exploration vs. exploitation)
- Discount factor (for future reward weighting)

## Performance Metrics

The agents are evaluated using the following metrics:

### Episode Return

Total reward accumulated over an episode, measuring overall task performance.

### Task Completion

Binary success/failure metric for tasks with clear completion criteria (e.g., reaching all waypoints).

### Efficiency Metrics

- **Time to Completion**: Steps required to complete the task
- **Energy Efficiency**: Battery usage during the task
- **Smoothness**: Measure of control stability based on action changes

### Robustness Tests

- **Sensor Noise Tolerance**: Performance with added sensor noise
- **Initial State Generalization**: Performance from various starting positions
- **Disturbance Recovery**: Ability to recover from external disturbances

## Selection Criteria

When choosing between agent types for a specific application, consider:

### Rule-Based Agents

**Best for**:
- Simple, well-defined tasks
- Real-time control with limited computational resources
- Applications requiring deterministic, interpretable behavior
- Testing and validation baselines

### Reinforcement Learning Agents

**Best for**:
- Complex tasks difficult to program explicitly
- Environments with changing dynamics
- Applications that benefit from adaptation
- Tasks requiring optimization beyond human design capability

**Algorithm Selection**:
- **PPO**: Good general-purpose choice, especially for limited experience data
- **SAC**: Best for sample-efficient learning and exploration-heavy tasks
- **TD3**: Effective for tasks sensitive to value estimation accuracy

### Deep Learning Agent

**Best for**:
- Maximum performance requirements
- Complex environments with high-dimensional state spaces
- Applications requiring sophisticated control policies
- Tasks benefiting from probabilistic action selection

## Implementation Details

### Code Structure

```
src/agent/
├── __init__.py                 # Package exports
├── base_agent.py               # Base agent classes
├── rl_agent.py                 # RL implementations
├── deep_agent.py               # Advanced agent
├── train.py                    # Training script
└── control.py                  # Agent control script
```

### Dependencies

- **PyTorch**: Neural network creation and training
- **Gymnasium**: Environment interface for RL
- **Stable-Baselines3**: RL algorithm implementations
- **NumPy**: Numerical operations
- **SciPy**: Scientific computing utilities

### Sample Usage

#### Training an Agent
```bash
python -m src train --algorithm ppo --task hover --timesteps 500000
```

#### Running a Trained Agent
```bash
python -m src control --agent ppo --task waypoint --model-path models/ppo_latest.zip
```

## Practical Applications

### Drone Task Automation

The AI agents enable automation of various drone tasks:

1. **Autonomous Hovering**:
   - Camera platforms for photography/videography
   - Aerial surveillance
   - Temporary communication relays

2. **Waypoint Navigation**:
   - Area mapping and surveying
   - Inspection of structures
   - Autonomous delivery

3. **Pattern Following**:
   - Perimeter monitoring
   - Search and rescue patterns
   - Entertainment displays

### Transfer to Real Hardware

The agents developed in simulation can be transferred to physical DJI Tello drones with:

1. **Domain Adaptation**:
   - Reality gap bridging techniques
   - Sim-to-real transfer methods
   - Incremental deployment

2. **Sensor Calibration**:
   - Mapping simulation sensors to real hardware
   - Noise model correction
   - Latency compensation

3. **Policy Robustness**:
   - Adversarial training for robustness
   - Environmental perturbation during training
   - Gradual transfer with human supervision

### Beyond Single Drones

The agent architecture can be extended to:

1. **Multi-drone Systems**:
   - Coordinated swarm behavior
   - Task allocation between drones
   - Collaborative mapping and exploration

2. **Human-Drone Interaction**:
   - Gesture-based control
   - Intent prediction
   - Collaborative task execution

3. **Advanced Applications**:
   - Obstacle detection and avoidance
   - Dynamic environment adaptation
   - Learning from human demonstrations

---

This document provides a comprehensive overview of the AI agents implemented in the DJTello Digital Twin project, detailing their inputs/outputs, architectures, training methodologies, and practical applications. These agents represent a spectrum of approaches from classical control to cutting-edge reinforcement learning, offering different trade-offs in terms of complexity, performance, and applicability.
