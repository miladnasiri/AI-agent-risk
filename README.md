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
  <p>Copyright (c) 2025 Your Name</p>
  <p>Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:</p>
  <p>The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.</p>
  <p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.</p>
</div>

---

<div align="center">
  <p><i>Developed with ❤️ using advanced AI techniques</i></p>
</div>
