#!/bin/bash
# Script to set up GRPO training on Linux using system PyTorch
# Uses the existing system PyTorch 2.4+ installation

set -e  # Exit on error

echo "Setting up environment for GRPO training on Linux..."

# Verify system PyTorch installation
echo "Verifying system PyTorch installation..."
python -c "import torch, numpy; import pkg_resources; torch_ver = pkg_resources.parse_version(torch.__version__.split('+')[0]); req_ver = pkg_resources.parse_version('2.4.0'); assert torch_ver >= req_ver, f'PyTorch 2.4+ required, found {torch.__version__}'; print(f'✓ System PyTorch {torch.__version__} detected')" || { 
    echo "Error: System PyTorch 2.4+ not found. Please check your RunPod image."; 
    exit 1; 
}

# Install uv (faster package installer)
echo "Installing uv package installer..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to path for this session
export PATH="$HOME/.cargo/bin:$PATH"

# Verify uv installation
which uv || { 
    echo "Error: uv installation failed. Falling back to pip..."; 
    USE_PIP=true
}

# Install function that works with either uv or pip
install_pkg() {
    if [ "$USE_PIP" = true ]; then
        pip install $@
    else
        uv pip install --system $@
    fi
}

# Install core dependencies
echo "Installing core dependencies..."
install_pkg setuptools wheel ninja pyyaml

# Install ML dependencies 
echo "Installing ML dependencies..."
install_pkg transformers accelerate peft trl datasets bitsandbytes sentencepiece scipy einops

# Install vLLM (required for fast_inference)
echo "Installing vLLM (required for fast_inference)..."
install_pkg vllm xformers

# Skip flash-attn by default as it's complex to build
echo "Note: Skipping flash-attn installation for stability"
echo "The model will work without it, just slightly slower"

# Install unsloth and dependencies
echo "Installing unsloth and related packages..."
install_pkg packaging protobuf tiktoken
install_pkg unsloth

# Clone unsloth if needed (for GRPO)
if [ ! -d "unsloth" ]; then
    echo "Cloning unsloth for GRPO..."
    git clone https://github.com/unslothai/unsloth.git
fi

# Install any remaining requirements
echo "Installing required packages from requirements.txt if it exists..."
if [ -f "requirements.txt" ]; then
    install_pkg -r requirements.txt
fi

echo "✓ Setup complete using system PyTorch!"
echo ""
echo "To train a GRPO model, run:"
echo "    python qwen_notebook_clone.py"
echo ""
echo "To convert the model to GGUF, run:"
echo "    ./create_gguf.sh"

# Optional flash-attn installation instructions
echo ""
echo "NOTE: If you want to install flash-attn for faster training:"
echo "    1. SKIP_CUDA_BUILD=1 pip install flash-attn"
echo "    2. pip install flash-attn --no-build-isolation" 