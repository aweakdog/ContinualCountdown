#!/usr/bin/env python3
"""
Generate dummy Tinyzero dataset by sampling from the original Tinyzero dataset
and dividing it into 4 groups for continual learning experiments.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
TRAIN_SAMPLES_PER_GROUP = 100_000  # Same as continual learning setup
TEST_SAMPLES_PER_GROUP = 1_000     # Same as continual learning setup
GROUPS = ['0', '1', '2', '3']  # Numerical groups for random sampling
SEED = 42

def make_prefix(dp, operators, template_type='qwen-instruct'):
    """Format the prompt for each data point."""
    target = dp['target']
    numbers = dp['nums']
    
    if template_type == 'qwen-instruct':
        prefix = f"""Assistant
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. 
User
 Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant
Let me solve this step by step.
<think>"""
    return prefix

def create_group_data(df, group_name, train_size, test_size, output_dir):
    """Sample data for a specific group and save train/test splits."""
    # Sample data for this group
    group_data = df.sample(n=train_size + test_size, random_state=SEED)
    
    # Add prompts to the data
    operators = ['+', '-', '*', '/']  # Using all operators for dummy data
    
    def add_prompt(row):
        row['prompt'] = make_prefix({'target': row['target'], 'nums': row['nums']}, operators)
        return row
    
    group_data = group_data.apply(add_prompt, axis=1)
    
    # Split into train and test
    train_data = group_data.iloc[:train_size]
    test_data = group_data.iloc[train_size:train_size + test_size]
    
    # Create group directory
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    train_data.to_parquet(group_dir / 'train.parquet')
    test_data.to_parquet(group_dir / 'test.parquet')
    
    print(f"Created {group_name} dataset:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

def main():
    # Setup paths
    input_file = Path('/data/tinyzero/data/train.parquet')
    output_dir = Path('/data/dummy_tinyzero')
    
    print(f"Reading input data from {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Total samples in input: {len(df)}")
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data for each group
    for group in GROUPS:
        create_group_data(
            df=df,
            group_name=group,
            train_size=TRAIN_SAMPLES_PER_GROUP,
            test_size=TEST_SAMPLES_PER_GROUP,
            output_dir=output_dir
        )
    
    print("\nDummy Tinyzero dataset generation complete!")

if __name__ == '__main__':
    main()
