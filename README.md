# AI agent DJTello Digital Twin 🚁

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26.0-orange.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<div align="center">
  <img src="Screenshot%20from%202025-02-28%2001-07-27.png" alt="DJTello Digital Twin" width="800">
</div>

## 📝 Abstract

This project presents a comprehensive digital twin simulation system for the DJI Tello drone, providing an accurate virtual representation of drone physics and behavior. The system implements a physics-based simulation that generates all necessary data internally without requiring external datasets. Using established aerodynamic principles and the DJI Tello's published specifications, the simulation creates realistic flight dynamics, sensor readings, and environmental interactions in real-time.

The AI agents interact with this self-contained simulation environment, which provides observation data (position, velocity, orientation, angular velocity) generated by the physics model itself. No pre-existing flight data or external datasets are required to run the system. This approach allows users to immediately deploy and test the digital twin after installation, with the simulation generating all necessary training data for the reinforcement learning agents through their interaction with the virtual environment.

<div align="center">
  <h3>🧠 AI Agents &nbsp;•&nbsp; 🔄 Digital Twin &nbsp;•&nbsp; 🧪 Physics Simulation &nbsp;•&nbsp; 📊 Data Generation</h3>
</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Technical Architecture](#-technical-architecture)
- [System Components](#-system-components)
- [AI Agent Framework](#-ai-agent-framework)
- [Ai-Agents-Overview](#-ai-agents-overview)
- [Physics Simulation](#-physics-simulation)
- [Control Systems](#-control-systems)
- [Methodologies](#-methodologies)
- [Data Flow & Requirements](#-data-flow--requirements)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Development Workflow](#-development-workflow)
- [Future Roadmap](#-future-roadmap)
- [Technical Documentation](#-technical-documentation)
- [License](#-license)

---

## 🔍 Project Overview

<div align="center">
  <img src="https://via.placeholder.com/800x300.png?text=DJTello+Digital+Twin+Architecture" alt="Project Architecture Overview" width="800">
</div>

DJTello Digital Twin is an advanced simulation platform that creates a high-fidelity virtual replica of a DJI Tello drone, complete with realistic physics, sensor simulations, and a suite of AI agents for autonomous control. This digital twin allows for rapid prototyping, training, and testing of drone control algorithms in a safe, virtual environment before deployment to physical hardware.

### 🎯 Goals

<table>
  <tr>
    <td width="60px" align="center"><b>🧪</b></td>
    <td><b>High-Fidelity Simulation</b>: Create a realistic physics-based model of the DJI Tello drone that accurately simulates flight dynamics, motor characteristics, battery discharge, and sensor responses.</td>
  </tr>
  <tr>
    <td width="60px" align="center"><b>🤖</b></td>
    <td><b>AI Agent Development</b>: Implement multiple AI agent architectures ranging from rule-based systems to advanced deep reinforcement learning models that can control the drone autonomously.</td>
  </tr>
  <tr>
    <td width="60px" align="center"><b>🏋️</b></td>
    <td><b>Reinforcement Learning Platform</b>: Provide a standardized environment compatible with the Gymnasium interface for training and evaluating reinforcement learning algorithms.</td>
  </tr>
  <tr>
    <td width="60px" align="center"><b>📊</b></td>
    <td><b>Visualization & Analysis</b>: Deliver 3D visualization of drone behavior along with data collection and analysis tools to understand and improve agent performance.</td>
  </tr>
  <tr>
    <td width="60px" align="center"><b>🔄</b></td>
    <td><b>Real-to-Virtual Transfer</b>: Enable algorithms developed in the simulation to transfer effectively to real DJI Tello drones with minimal adaptation.</td>
  </tr>
  <tr>
    <td width="60px" align="center"><b>🔌</b></td>
    <td><b>Expandable Framework</b>: Create a modular architecture that can be expanded to include additional sensors, environmental conditions, and multi-drone scenarios.</td>
  </tr>
</table>

---

## 🏗️ Technical Architecture

The DJTello Digital Twin follows a modular, layered architecture that separates concerns while enabling tight integration between components.

<details>
<summary><b>🔍 System Architecture Diagram (Click to expand)</b></summary>

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
</details>

<details>
<summary><b>🔄 Component Interaction Flow (Click to expand)</b></summary>

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
</details>

---

## 🧩 System Components

### 1. Drone Simulation Module 🚁

The core simulation module implements a high-fidelity model of the DJI Tello drone with the following components:

<details>
<summary><b>Physical Specifications</b></summary>

- **Mass**: 80 grams
- **Dimensions**: 98×92.5×41 mm
- **Motor Configuration**: 4 brushed motors
- **Maximum Thrust**: 2.0 * Mass * 9.81 N
- **Battery**: 1.1 Wh LiPo battery with realistic discharge model
- **Moments of Inertia**: Calculated based on drone weight distribution
</details>

<details>
<summary><b>Dynamic Model</b></summary>

- **Degrees of Freedom**: 6-DOF model (position: x, y, z and orientation: roll, pitch, yaw)
- **State Vector**: [x, y, z, vx, vy, vz, roll, pitch, yaw, ω_roll, ω_pitch, ω_yaw]
- **Euler Integration**: Numerical integration of equations of motion using RK45
- **Aerodynamic Effects**: Drag coefficients, propeller airflow, ground effect
</details>

<details>
<summary><b>Sensor Simulation</b></summary>

- **IMU**: Accelerometer and gyroscope with realistic noise models
- **Barometer**: Pressure-based altitude estimation
- **Downward Vision System**: Optical flow for position holding
- **Camera**: Forward-facing camera with configurable resolution and FPS
</details>

<details>
<summary><b>Environment Integration</b></summary>

- **Gymnasium Interface**: Standard RL environment compliant with Gymnasium API
- **Observation Space**: Customizable state observations
- **Action Space**: Direct control of [thrust, roll, pitch, yaw] commands
- **Reward Functions**: Task-specific reward structures for RL training
</details>

### 2. AI Agent Framework 🤖

The agent framework provides multiple implementations for controlling the drone:

<div class="agent-types" style="display: flex; gap: 16px; margin-bottom: 20px;">
  <div style="flex: 1; padding: 16px; background-color: #f0f8ff; border-radius: 8px; border-left: 4px solid #1e90ff;">
    <h4>Rule-Based Agents</h4>
    <ul>
      <li>RandomAgent</li>
      <li>HoverAgent (PID)</li>
      <li>WaypointAgent</li>
    </ul>
  </div>
  
  <div style="flex: 1; padding: 16px; background-color: #f0fff0; border-radius: 8px; border-left: 4px solid #32cd32;">
    <h4>RL Agents</h4>
    <ul>
      <li>PPOAgent</li>
      <li>SACAgent</li>
      <li>TD3Agent</li>
    </ul>
  </div>
  
  <div style="flex: 1; padding: 16px; background-color: #fff0f5; border-radius: 8px; border-left: 4px solid #ff69b4;">
    <h4>Deep Learning Agent</h4>
    <ul>
      <li>Actor-Critic Networks</li>
      <li>Automatic Entropy Tuning</li>
      <li>Prioritized Experience Replay</li>
    </ul>
  </div>
</div>

### 3. Visualization System 🎮

The visualization system provides real-time 3D rendering of the drone and its environment:

- **OpenGL Renderer**: 3D visualization of the drone and environment
- **Camera Controls**: Configurable viewpoint and following modes
- **Telemetry Display**: Real-time display of drone state
- **Recording**: Video capture of simulation runs

---

## 🧠 AI Agent Framework

<div align="center">
  <img src="https://via.placeholder.com/800x300.png?text=AI+Agent+Framework" alt="AI Agent Framework" width="800">
</div>

The AI agent framework is designed to accommodate multiple approaches to drone control, from simple rule-based systems to advanced deep learning techniques.

<details>
<summary><b>🌳 Agent Hierarchy (Click to expand)</b></summary>

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
</details>

<details>
<summary><b>🔄 Learning Agent Architecture (Click to expand)</b></summary>

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
</details>

<details>
<summary><b>🧮 Neural Network Architectures (Click to expand)</b></summary>

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
</details>

To see the complete details of all AI agents in this system, please refer to the [AI-Agents-Overview](AI_AGENTS_OVERVIEW.md) document.

---

## 🧪 Physics Simulation

<div align="center">
  <img src="https://via.placeholder.com/800x250.png?text=Physics+Simulation" alt="Physics Simulation" width="800">
</div>

The physics simulation is at the core of the digital twin, providing high-fidelity modeling of drone behavior.

### Physics Update Cycle

<div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 4px solid #007bff; margin-bottom: 20px;">
  <code>
  Input Forces/Torques → State Dynamics Computation → Numerical Integration → State Update & Sensor Models
  </code>
</div>

### Force and Torque Models

<table>
  <tr>
    <th>Force/Torque</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>Gravity</b></td>
    <td>Applied in world frame Z-axis</td>
  </tr>
  <tr>
    <td><b>Motor Thrust</b></td>
    <td>Individual motor thrust based on motor speed</td>
  </tr>
  <tr>
    <td><b>Drag</b></td>
    <td>Aerodynamic drag proportional to velocity squared</td>
  </tr>
  <tr>
    <td><b>Motor Torques</b></td>
    <td>Torque produced by each motor's rotation</td>
  </tr>
  <tr>
    <td><b>Reaction Torques</b></td>
    <td>Counter-torque from motor rotation</td>
  </tr>
</table>

### Integration Method

```python
# The simulation uses scipy.integrate.solve_ivp with RK45 method
def _state_derivative(self, t, state):
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
```

---

## 🎛️ Control Systems

The project implements various control systems for drone navigation and stability.

### PID Control Implementation

<div style="background-color: #f0f8ff; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
  <h4>PID Control Formula</h4>
  <p><code>Error = Target Position - Current Position</code></p>
  <p><code>P_term = Kp * Error</code></p>
  <p><code>I_term = Ki * ∫Error dt</code></p>
  <p><code>D_term = Kd * d(Error)/dt</code></p>
  <p><code>Control Signal = P_term + I_term + D_term</code></p>
</div>

#### Default PID Gain Values

<table>
  <tr>
    <th>Control Domain</th>
    <th>Kp</th>
    <th>Ki</th>
    <th>Kd</th>
  </tr>
  <tr>
    <td><b>Position</b></td>
    <td>[0.5, 0.5, 0.8]</td>
    <td>[0.01, 0.01, 0.02]</td>
    <td>[0.2, 0.2, 0.4]</td>
  </tr>
  <tr>
    <td><b>Attitude</b></td>
    <td>[2.0, 2.0, 1.0]</td>
    <td>[0.1, 0.1, 0.1]</td>
    <td>[0.1, 0.1, 0.5]</td>
  </tr>
</table>

### Waypoint Navigation

<div align="center">
  <img src="https://via.placeholder.com/600x200.png?text=Waypoint+Navigation" alt="Waypoint Navigation" width="600">
</div>

The WaypointAgent uses a PID control system with sequential waypoint targeting:

1. Set target position to current waypoint
2. Use PID controller to navigate to target
3. When within threshold distance (default: 0.2m), advance to next waypoint
4. Repeat until all waypoints are visited

---

## 📊 Data Flow & Requirements

<div class="highlight-box" style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #28a745; margin-bottom: 20px;">
  <h3>No External Data Required! ✅</h3>
  <p>This project is completely self-contained and does not require any external datasets, pre-recorded flight data, or real-world drone measurements. All data is generated programmatically by the physics simulation based on the drone's known specifications.</p>
</div>

### Simulation Data Generation

The system generates several types of data during operation:

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px;">
  <div style="background-color: #e3f2fd; padding: 16px; border-radius: 8px;">
    <h4>📊 State Data</h4>
    <p>The physics simulation continuously calculates the drone's position, velocity, orientation, and angular velocity based on applied forces, aerodynamics, and gravity.</p>
  </div>
  
  <div style="background-color: #e8f5e9; padding: 16px; border-radius: 8px;">
    <h4>📡 Sensor Data</h4>
    <p>Virtual sensors simulate realistic IMU readings (accelerometer, gyroscope), barometer readings, and camera images with appropriate noise characteristics.</p>
  </div>
  
  <div style="background-color: #fff3e0; padding: 16px; border-radius: 8px;">
    <h4>🌍 Environmental Data</h4>
    <p>The simulation tracks environmental interactions including collisions, ground effect, and battery discharge based on motor usage.</p>
  </div>
  
  <div style="background-color: #e0f7fa; padding: 16px; border-radius: 8px;">
    <h4>🎮 Control Data</h4>
    <p>The AI agents generate control signals (thrust, roll, pitch, yaw) which are fed back into the physics simulation.</p>
  </div>
</div>

### Training Data for AI Agents

The reinforcement learning agents are trained using data collected during interaction with the simulation:

1. **Observations**: State vectors containing the drone's position, velocity, orientation, and other relevant parameters.
2. **Actions**: Control commands issued by the agent to the drone.
3. **Rewards**: Calculated values based on how well the agent is performing its assigned task.
4. **Transitions**: Sequences of (state, action, reward, next_state) tuples stored in replay buffers during training.

This training data is generated automatically as the agents interact with the environment - there's no need to provide external training datasets.

---

## 💻 Installation & Setup

### Prerequisites

<table>
  <tr>
    <td width="40px" align="center">🐍</td>
    <td><b>Python 3.8+</b></td>
  </tr>
  <tr>
    <td width="40px" align="center">🖥️</td>
    <td><b>OpenGL compatible graphics card</b></td>
  </tr>
  <tr>
    <td width="40px" align="center">🧠</td>
    <td><b>4GB RAM minimum (8GB recommended)</b></td>
  </tr>
  <tr>
    <td width="40px" align="center">💾</td>
    <td><b>2GB disk space</b></td>
  </tr>
</table>

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/djtello-digital-twin.git
cd djtello-digital-twin

# Run the setup script
chmod +x startup.sh
./startup.sh

# Verify installation
source venv/bin/activate
python -m src simulation --task hover --render
```

<details>
<summary><b>📦 Dependencies (Click to expand)</b></summary>

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
</details>

---

## 📘 Usage Guide

### Quick Start Commands

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px;">
  <div style="background-color: #e3f2fd; padding: 16px; border-radius: 8px;">
    <h4>🚀 Run Simulation</h4>
    <code>python -m src simulation --task hover --render</code>
  </div>
  
  <div style="background-color: #e8f5e9; padding: 16px; border-radius: 8px;">
    <h4>🧠 Train Agent</h4>
    <code>python -m src train --algorithm ppo --task hover --timesteps 500000</code>
  </div>
  
  <div style="background-color: #fff3e0; padding: 16px; border-radius: 8px;">
    <h4>🎮 Run Trained Agent</h4>
    <code>python -m src control --agent ppo --task waypoint --model-path models/ppo_latest.zip</code>
  </div>
</div>

### Available Tasks

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px;">
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>hover</code>: Maintain position at a fixed point
  </div>
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>takeoff</code>: Ascend to a target height
  </div>
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>land</code>: Descend safely to the ground
  </div>
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>waypoint</code>: Navigate through waypoints
  </div>
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>square</code>: Fly in a square pattern
  </div>
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>circle</code>: Fly in a circular pattern
  </div>
  <div style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
    <code>flip</code>: Perform flip maneuvers
  </div>
</div>



### Advanced Configuration

<details>
<summary><b>📝 Configuration File Example (Click to expand)</b></summary>

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
</details>

### Recording Simulations

<div style="background-color: #f0f0f0; padding: 16px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ff6b6b;">
  <h4>📹 Video Recording</h4>
  <p>Capture your simulation runs for later analysis or demonstration:</p>
  <code>python -m src control --agent ppo --task circle --record</code>
</div>

Videos are saved in the `videos/` directory in MP4 format with timestamp filenames.

---

## 🔧 Development Workflow

### Project Structure

<div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 14px; line-height: 1.5; overflow-x: auto; margin-bottom: 20px;">
<pre>
djtello-digital-twin/
├── src/                    <span style="color: #28a745;">🗂️ Source code</span>
│   ├── simulation/         <span style="color: #007bff;">🚁 Drone physics simulation</span>
│   │   ├── __init__.py
│   │   ├── drone_model.py  <span style="color: #6f42c1;">⚙️ Physical drone model</span>
│   │   └── environment.py  <span style="color: #6f42c1;">🌍 Gymnasium environment</span>
│   ├── agent/              <span style="color: #007bff;">🤖 AI agent implementation</span>
│   │   ├── __init__.py
│   │   ├── base_agent.py   <span style="color: #6f42c1;">🧩 Agent interface</span>
│   │   ├── rl_agent.py     <span style="color: #6f42c1;">🧠 RL agents</span>
│   │   ├── deep_agent.py   <span style="color: #6f42c1;">🔬 Advanced deep agent</span>
│   │   ├── train.py        <span style="color: #6f42c1;">📊 Training script</span>
│   │   └── control.py      <span style="color: #6f42c1;">🎮 Agent control script</span>
│   ├── visualization/      <span style="color: #007bff;">🎨 3D visualization</span>
│   │   ├── __init__.py
│   │   └── drone_renderer.py
│   ├── __init__.py
│   └── __main__.py         <span style="color: #6f42c1;">🚀 Main entry point</span>
├── tests/                  <span style="color: #28a745;">🧪 Unit and integration tests</span>
├── docs/                   <span style="color: #28a745;">📚 Documentation</span>
├── data/                   <span style="color: #28a745;">💾 Training data and models</span>
├── requirements.txt        <span style="color: #28a745;">📋 Python dependencies</span>
├── setup.py                <span style="color: #28a745;">⚙️ Package setup file</span>
├── startup.sh              <span style="color: #28a745;">🔌 Setup script</span>
└── README.md               <span style="color: #28a745;">📘 This file</span>
</pre>
</div>

### Contributing Guidelines

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px;">
  <div style="background-color: #f0f8ff; padding: 16px; border-radius: 8px;">
    <h4>🌿 Branching Strategy</h4>
    <ul>
      <li><code>main</code>: Stable release branch</li>
      <li><code>develop</code>: Development branch</li>
      <li><code>feature/*</code>: Feature branches</li>
      <li><code>bugfix/*</code>: Bug fix branches</li>
    </ul>
  </div>
  
  <div style="background-color: #f0fff0; padding: 16px; border-radius: 8px;">
    <h4>✍️ Commit Message Format</h4>
    <pre style="background-color: #f8f9fa; padding: 8px; border-radius: 4px;">[Component] Brief description

Detailed explanation if needed</pre>
  </div>
  
  <div style="background-color: #fff0f5; padding: 16px; border-radius: 8px;">
    <h4>🧹 Code Style</h4>
    <ul>
      <li>Follow PEP 8 guidelines</li>
      <li>Use docstrings for all classes and functions</li>
      <li>Maintain test coverage above 80%</li>
    </ul>
  </div>
  
  <div style="background-color: #f0f0ff; padding: 16px; border-radius: 8px;">
    <h4>🔄 Pull Request Process</h4>
    <ul>
      <li>Create PR against <code>develop</code> branch</li>
      <li>Ensure all tests pass</li>
      <li>Obtain at least one code review</li>
      <li>Squash and merge</li>
    </ul>
  </div>
</div>

---

## 🚀 Future Roadmap

<div style="display: flex; gap: 16px; margin-bottom: 20px; overflow-x: auto; padding-bottom: 10px;">
  <div style="flex: 1; min-width: 250px; padding: 16px; background-color: #e3f2fd; border-radius: 8px;">
    <h3>Phase 1: Core Enhancement</h3>
    <ul>
      <li>[ ] Improved collision detection with environment objects</li>
      <li>[ ] Enhanced aerodynamic models including wind effects</li>
      <li>[ ] Support for multiple drone types beyond Tello</li>
    </ul>
  </div>
  
  <div style="flex: 1; min-width: 250px; padding: 16px; background-color: #e8f5e9; border-radius: 8px;">
    <h3>Phase 2: Advanced Features</h3>
    <ul>
      <li>[ ] Swarm simulation with multiple coordinated drones</li>
      <li>[ ] Computer vision-based navigation tasks</li>
      <li>[ ] Integration with real Tello SDK for hardware deployment</li>
    </ul>
  </div>
  
  <div style="flex: 1; min-width: 250px; padding: 16px; background-color: #fff3e0; border-radius: 8px;">
    <h3>Phase 3: Ecosystem Development</h3>
    <ul>
      <li>[ ] Web-based visualization and remote control</li>
      <li>[ ] Community model sharing platform</li>
      <li>[ ] Integration with popular robotics frameworks (ROS, etc.)</li>
    </ul>
  </div>
</div>

---

## 📓 Technical Documentation

### Agent API Reference

<div style="background-color: #f6f8fa; padding: 16px; border-radius: 8px; margin-bottom: 20px; font-family: monospace; overflow-x: auto;">
<pre style="margin: 0;">
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
</pre>
</div>

### Environment Interface

<div style="background-color: #f6f8fa; padding: 16px; border-radius: 8px; margin-bottom: 20px; font-family: monospace; overflow-x: auto;">
<pre style="margin: 0;">
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
</pre>
</div>

### Camera Projection Model

<div style="background-color: #f0f8ff; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
  <h4>📷 Pinhole Camera Model</h4>
  <p><code>x_image = fx * X / Z + cx</code></p>
  <p><code>y_image = fy * Y / Z + cy</code></p>
  
  <p>Where:</p>
  <ul>
    <li><code>(X, Y, Z)</code> are 3D world coordinates</li>
    <li><code>(x_image, y_image)</code> are image coordinates</li>
    <li><code>fx, fy</code> are focal lengths</li>
    <li><code>cx, cy</code> are principal point offsets</li>
  </ul>
  
  <p>Default parameters for the Tello camera:</p>
  <ul>
    <li>Resolution: 960x720 pixels</li>
    <li>Field of View: 82.6 degrees</li>
    <li>Frame Rate: 30 FPS</li>
  </ul>
</div>

---

## 📜 License

<div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
  <h3>MIT License</h3>
  <
</div>

---

<div align="center">
  <p><i>Milad Nasiri</p>
</div>
