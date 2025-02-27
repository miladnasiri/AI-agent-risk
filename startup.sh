#!/bin/bash

# Startup script for DJTello Digital Twin

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing DJTello Digital Twin in development mode..."
pip install -e .

# Run a quick test to make sure everything is working
echo "Running a quick test..."
python -m src simulation --task hover

# Print success message
echo ""
echo "DJTello Digital Twin has been set up successfully!"
echo ""
echo "To run the simulation, use:"
echo "python -m src simulation --task hover --render"
echo ""
echo "To train an agent, use:"
echo "python -m src train --algorithm ppo --task hover --render"
echo ""
echo "To run a trained agent, use:"
echo "python -m src control --agent ppo --task hover --model-path /path/to/model"
echo ""
echo "For more information, use:"
echo "python -m src --help"

# Deactivate virtual environment
deactivate
