#!/usr/bin/env python3
# Qwen2.5 (3B) GRPO Training Script
# Originally based on Unsloth's GRPO notebook
# Modified for RunPod with RTX Ada 6000 (48GB VRAM, 64GB RAM)

import re
import argparse
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

# Apply GRPO patch to FastLanguageModel
PatchFastRL("GRPO", FastLanguageModel)

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

def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML format"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """Extract the answer from hash format (GSM8K style)"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    """Load and prepare GSM8K dataset"""
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Check if the extracted answer matches the correct answer"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Check if the extracted answer is an integer"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Check if the completion follows the exact format"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Check if the completion has XML tags in any format"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """Count XML tags and calculate a score based on format correctness"""
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
    """Calculate XML format score for each completion"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def load_model(args):
    """Load and configure the model with LoRA"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer

def get_training_args(args):
    """Configure GRPO training arguments"""
    return GRPOConfig(
        use_vllm=True,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=args.output_dir,
    )

def train_model(model, tokenizer, dataset, training_args):
    """Train the model using GRPO"""
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    return trainer

def test_model(model, tokenizer, text_prompt, lora_path=None):
    """Test the model with a given prompt"""
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text_prompt},
    ], tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    
    if lora_path:
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_path),
        )[0].outputs[0].text
    else:
        output = model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0].outputs[0].text
    
    return output

def save_model(model, tokenizer, args):
    """Save the model in the specified format"""
    # Save LoRA weights
    model.save_lora(args.lora_output_path)
    
    if args.save_full_model:
        if args.save_format == "merged_16bit":
            model.save_pretrained_merged(args.model_output_path, tokenizer, save_method="merged_16bit")
        elif args.save_format == "merged_4bit":
            model.save_pretrained_merged(args.model_output_path, tokenizer, save_method="merged_4bit")
        elif args.save_format == "lora":
            model.save_pretrained_merged(args.model_output_path, tokenizer, save_method="lora")
        elif args.save_format == "gguf":
            model.save_pretrained_gguf(args.model_output_path, tokenizer, quantization_method=args.gguf_quant_method)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a model with GRPO")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", 
                        help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=128, 
                        help="LoRA rank for fine-tuning")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU memory utilization (0-1)")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Number of generations for GRPO")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=300,
                        help="Maximum completion length")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints")
    
    # Saving configuration
    parser.add_argument("--lora_output_path", type=str, default="grpo_saved_lora",
                        help="Output path for LoRA weights")
    parser.add_argument("--save_full_model", action="store_true",
                        help="Whether to save the full model")
    parser.add_argument("--model_output_path", type=str, default="model",
                        help="Output path for the full model")
    parser.add_argument("--save_format", type=str, default="lora",
                        choices=["merged_16bit", "merged_4bit", "lora", "gguf"],
                        help="Format to save the model in")
    parser.add_argument("--gguf_quant_method", type=str, default="q4_k_m",
                        help="Quantization method for GGUF format")
    
    # Testing configuration
    parser.add_argument("--test_model", action="store_true",
                        help="Whether to test the model after training")
    parser.add_argument("--test_prompt", type=str, default="How many r's are in strawberry?",
                        help="Prompt to test the model with")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load model
    print(f"Loading model {args.model_name}...")
    model, tokenizer = load_model(args)
    
    # Get dataset
    print("Loading GSM8K dataset...")
    dataset = get_gsm8k_questions()
    
    # Configure training
    print("Configuring training arguments...")
    training_args = get_training_args(args)
    
    # Train model
    print("Starting GRPO training...")
    trainer = train_model(model, tokenizer, dataset, training_args)
    
    # Save model
    print(f"Saving LoRA weights to {args.lora_output_path}...")
    model.save_lora(args.lora_output_path)
    
    if args.save_full_model:
        print(f"Saving full model to {args.model_output_path} in {args.save_format} format...")
        save_model(model, tokenizer, args)
    
    # Test model
    if args.test_model:
        print("Testing base model...")
        base_output = test_model(model, tokenizer, args.test_prompt)
        print(f"Base model output:\n{base_output}\n")
        
        print("Testing fine-tuned model...")
        ft_output = test_model(model, tokenizer, args.test_prompt, args.lora_output_path)
        print(f"Fine-tuned model output:\n{ft_output}")

if __name__ == "__main__":
    main()