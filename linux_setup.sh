#!/bin/bash
# Script to set up environment for GRPO training on Linux
# Assumes PyTorch 2.4 is already installed

set -e  # Exit on error

echo "Setting up environment for GRPO training on Linux..."

# Create virtual environment if it doesn't exist
if [ ! -d "linux_env" ]; then
    echo "Creating virtual environment..."
    python -m venv linux_env
    source linux_env/bin/activate
else
    echo "Using existing virtual environment..."
    source linux_env/bin/activate
fi

# Install uv (faster package installer)
echo "Installing uv package installer..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to path for this session
export PATH="$HOME/.cargo/bin:$PATH"

# Verify uv installation
which uv || { 
    echo "Error: uv installation failed. Try installing manually: curl -LsSf https://astral.sh/uv/install.sh | sh"; 
    exit 1; 
}

# Install required packages
echo "Installing required packages..."
uv pip install setuptools wheel

# First ensure PyTorch is properly installed in the virtual environment
echo "Installing PyTorch 2.4..."
uv pip install torch>=2.4.0

# Verify PyTorch 2.4 installation
python -c "import torch; assert torch.__version__.startswith('2.4'), 'PyTorch 2.4+ required, found '+torch.__version__; print('✓ PyTorch', torch.__version__)" || { 
    echo "Error: PyTorch 2.4+ installation failed. Try installing manually: pip install torch>=2.4.0"; 
    exit 1; 
}

# Install dependencies one by one
echo "Installing dependencies..."
uv pip install transformers accelerate peft trl datasets bitsandbytes sentencepiece scipy einops

# Install flash-attn separately without --no-deps
echo "Installing flash-attn (this may take a while)..."
uv pip install flash-attn

# Install unsloth
echo "Installing unsloth..."
uv pip install unsloth

# Clone unsloth if needed (for GRPO)
if [ ! -d "unsloth" ]; then
    echo "Cloning unsloth for GRPO..."
    git clone https://github.com/unslothai/unsloth.git
fi

# Install any remaining requirements
echo "Installing required packages from requirements.txt if it exists..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
fi

echo "✓ Environment setup complete!"
echo ""
echo "To activate this environment, run:"
echo "    source linux_env/bin/activate"
echo ""
echo "To train a GRPO model, run:"
echo "    python qwen_notebook_clone.py"
echo ""
echo "To convert the model to GGUF, run:"
echo "    ./create_gguf.sh"

# Optional fallback instructions if flash-attn installation fails
echo ""
echo "NOTE: If you experience issues with flash-attn installation, you can try manually installing it after activating the environment:"
echo "    pip install flash-attn --no-build-isolation" 