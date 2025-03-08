#!/bin/bash
# Simple script to set up a UV virtual environment and install requirements

# Create virtual environment with UV
echo "Creating virtual environment with UV..."
uv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete! Virtual environment is activated and dependencies are installed."
echo "To manually activate this environment in the future, run: source .venv/bin/activate" 