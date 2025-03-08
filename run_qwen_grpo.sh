#!/bin/bash
# Run Qwen2.5 GRPO training with compatible settings that avoid the math constraint error

# For this to work properly, need to make sure that:
# batch_size * grad_accum_steps is divisible by num_generations
# Setting batch_size=8, grad_accum_steps=1, num_generations=8 satisfies this

echo "Starting Qwen2.5 GRPO training with compatible settings..."
echo "Note: Using batch_size=8 and num_generations=8 to satisfy GRPO constraint"

python qwen2_5_ada6000_grpo.py \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --num_generations 8 \
  --gpu_memory_utilization 0.92

# If you want to use the exact settings from the notebook but fix the constraint:
# python qwen2_5_ada6000_grpo.py \
#   --batch_size 1 \
#   --grad_accum_steps 8 \
#   --num_generations 8 \
#   --max_prompt_length 256 \
#   --max_completion_length 200

echo "Training complete!" 