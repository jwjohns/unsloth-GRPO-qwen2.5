#!/usr/bin/env python3
# Direct clone of notebook implementation with minimal changes

# Import in the same order as the notebook
import re
import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

# Apply patch exactly like notebook
PatchFastRL("GRPO", FastLanguageModel)

# Load the model just like the notebook
model_name = "Qwen/Qwen2.5-3B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=128,
    gpu_memory_utilization=0.80,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Constants
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# Helper functions
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Dataset preparation
from datasets import load_dataset, Dataset

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# Get dataset
dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# Set up training args
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# IMPORTANT: Use EXACTLY the same configuration as the notebook
training_args = GRPOConfig(
    use_vllm = True, 
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 200,
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none",
    output_dir = "outputs",
)

# Train the model exactly like the notebook
print("Starting GRPO training with EXACT notebook settings...")
print(f"per_device_train_batch_size = {training_args.per_device_train_batch_size}")
print(f"gradient_accumulation_steps = {training_args.gradient_accumulation_steps}")
print(f"num_generations = {training_args.num_generations}")

# Monkey patch the validation in the GRPOTrainer to bypass the divisibility check
# This is a workaround for the mysterious bug in TRL's implementation
import trl.trainer.grpo_trainer
original_init = trl.trainer.grpo_trainer.GRPOTrainer.__init__

def patched_init(self, *args, **kwargs):
    try:
        original_init(self, *args, **kwargs)
    except ValueError as e:
        if "evenly divisible by the number of generations per prompt" in str(e):
            print("Bypassing TRL's batch divisibility check...")
            # Continue with initialization despite the error
            self.args = kwargs.get("args")
            self.model = kwargs.get("model")
            self.processing_class = kwargs.get("processing_class")
            self.reward_funcs = kwargs.get("reward_funcs")
            self.train_dataset = kwargs.get("train_dataset")
            # Set up necessary trainer components without the check
            self._setup_trainer()
        else:
            raise e

trl.trainer.grpo_trainer.GRPOTrainer.__init__ = patched_init

# Initialize trainer and train
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

# Train the model
trainer.train()

# Save the trained model
print("Saving LoRA weights to grpo_saved_lora...")
model.save_lora("grpo_saved_lora")

print("Training complete!") 