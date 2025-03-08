#!/bin/bash
# Script to create GGUF file using llama.cpp with safeguards

set -e  # Exit on error

# Step 1: Create merged model if needed
if [ ! -d "final_model" ]; then
  echo "Merging LoRA weights with base model..."
  python just_make_it_work.py
fi

# Step 2: Clone llama.cpp if not already done
if [ ! -d "llama.cpp" ]; then
  echo "Cloning llama.cpp..."
  git clone https://github.com/ggerganov/llama.cpp.git
fi

# Step 3: Build llama.cpp with CMake
echo "Building llama.cpp with CMake..."
cd llama.cpp
mkdir -p build
cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release -j
cd ../..

# Step 4: Check if the conversion script exists, if not download it
SCRIPT_PATH="llama.cpp/convert_hf_to_gguf.py"
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Conversion script not found. Checking alternate name..."
  if [ -f "llama.cpp/convert-hf-to-gguf.py" ]; then
    SCRIPT_PATH="llama.cpp/convert-hf-to-gguf.py"
    echo "Found script with hyphenated name."
  else
    echo "Downloading conversion script directly..."
    curl -o convert_hf_to_gguf.py https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py
    SCRIPT_PATH="./convert_hf_to_gguf.py"
  fi
fi

# Get absolute paths for clarity
SCRIPT_PATH=$(realpath "$SCRIPT_PATH")
MODEL_PATH=$(realpath "final_model")
# For output file that doesn't exist yet, construct the path manually
OUTPUT_FILE="qwen-grpo-unquantized.gguf"
OUTPUT_DIR=$(realpath ".")
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

# Step 5: Run the conversion - create unquantized model first
echo "Converting to unquantized GGUF format..."
echo "Using script: $SCRIPT_PATH"
echo "Source model: $MODEL_PATH" 
echo "Output file: $OUTPUT_PATH"

# Create unquantized GGUF first (F16)
python "$SCRIPT_PATH" --outtype f16 --outfile "$OUTPUT_PATH" "$MODEL_PATH"

# Step 6: Quantize the model with a format suitable for LM Studio
QUANTIZED_OUTPUT="$OUTPUT_DIR/qwen-grpo-q5_k_m.gguf"
echo "Quantizing model to Q5_K_M format for improved LM Studio compatibility..."
echo "Input: $OUTPUT_PATH"
echo "Output: $QUANTIZED_OUTPUT"

cd llama.cpp
./build/bin/llama-quantize "$OUTPUT_DIR/$OUTPUT_FILE" "$QUANTIZED_OUTPUT" q5_k_m

echo "Quantized GGUF file created at: $QUANTIZED_OUTPUT"
echo "Try loading this file in LM Studio"
echo "If that doesn't work, the unquantized file is available at: $OUTPUT_PATH" 