"""
Main entry point for the Qwen2.5 GRPO Fine-tuning package.
This module simply imports and runs the original script.
"""

import sys
import os
from pathlib import Path

def main():
    """
    Run the Qwen2.5 GRPO fine-tuning script.
    This is a simple wrapper around the original script.
    """
    # Get the path to the script
    script_path = Path(__file__).parent.parent / "qwen2_5_(3b)_grpo.py"
    
    # Check if the script exists
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Run the script with the current arguments
    sys.argv[0] = str(script_path)
    exec(open(script_path).read())

if __name__ == "__main__":
    main() 