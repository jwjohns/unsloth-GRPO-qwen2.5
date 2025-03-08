#!/usr/bin/env python3
"""
Dead simple script to merge LoRA weights and save the model for LM Studio
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True
)

print("Loading and merging LoRA weights...")
model = PeftModel.from_pretrained(model, "./grpo_saved_lora")
model = model.merge_and_unload()

# Set this flag explicitly
model.config.trust_remote_code = True

print("Saving model...")
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

print("\nDONE!")
print("Open LM Studio")
print("Click 'Add Model' -> 'Browse Local Files'")
print("Select the 'final_model' folder")
print("That's it.") 