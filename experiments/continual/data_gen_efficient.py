"""Generate Countdown task data with different operator groups and train/test splits."""

import os
import random
from typing import List, Dict, Tuple
from datasets import Dataset
import sys
sys.path.append('.')
from examples.data_preprocess.countdown_directly import CountDownDirectly
from examples.data_preprocess.countdown_reverse import CountDownReverse
from tqdm import tqdm
from rich import print as rprint


def make_prefix(dp, operators, template_type='base'):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""Assistant\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. \nUser\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant\nLet me solve this step by step.\n<think>"""
    return prefix


class DataGenerator:
    def __init__(self, base_dir: str = "/app/data/continual"):  
        self.base_dir = base_dir
        self.operator_groups = [
            [['+', '-', '*'], ['+', '-', '*']],  # plus_minus_mul
            [['+', '-', '/'], ['+', '-', '/']],  # plus_minus_div
            [['-', '*', '/'], ['-', '*', '/']],  # minus_mul_div
            [['+', '*', '/'], ['+', '*', '/']]   # plus_mul_div
        ]
        self.distinct = True
        #self.operator_groups = [
        #    [['*','/','+'], ['/']],  # plus_minus_mul
        #    [['*', '/','-'], ['*']],  # plus_minus_div
        #    [['+'], ['+']],  # minus_mul_div
        #    [['-'], ['-']]   # plus_mul_div
        #]
        #self.group_names = [
        #    "plus_minus_mul",
        #    "plus_minus_div",
        #    "minus_mul_div",
        #    "plus_mul_div"
        #]
        self.group_names = [
            "0",
            "1",
            "2",
            "3"
        ]
        os.makedirs(base_dir, exist_ok=True)
        os.system(f"chmod -R 777 {base_dir}")


    def generate_group_data(self, group_idx: int, train_size: int = 512000, test_size: int = 512) -> Tuple[Dataset, Dataset]:
        """Generate train and test data for a specific operator group"""
        candidate_operators = self.operator_groups[group_idx][0]
        neccessary_operators = self.operator_groups[group_idx][1]
        group_name = self.group_names[group_idx]
        group_dir = os.path.join(self.base_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        def generate_samples(num_samples: int, seed_offset: int = 0):
            random.seed(42 + group_idx + seed_offset)
            samples = []
            
            for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples for {group_name}"):
                # Initialize countdown with random start size between 4 and 6
                start_size = random.randint(4, 4)
                cd = CountDownReverse(min_target=3, max_target=100, start_size=start_size, 
                                   max_internal_value=100, 
                                   candidate_operators=candidate_operators, 
                                   neccessary_operators=neccessary_operators,
                                   distinct=self.distinct)
                target, nums, solution = cd.generate()
                rating = 1.0
                samples.append({
                    "target": target,
                    "nums": nums,
                    "solution": solution,
                    "rating": rating,
                })
            
            return samples
        
        rprint(f"[yellow]Generating training samples...[/yellow]")
        train_samples = generate_samples(train_size)
        
        rprint(f"[yellow]Generating test samples...[/yellow]")
        test_samples = generate_samples(test_size, seed_offset=100)  # Different seed for test set
        #test_samples = train_samples
        
        # Convert to dataset format
        def create_dataset(samples, split: str) -> Dataset:
            # Convert samples to proper dataset format
            data = {
                "target": [s["target"] for s in samples],
                "nums": [s["nums"] for s in samples],
                "solution": [s["solution"] for s in samples],
                "rating": [s["rating"] for s in samples],
            }
            dataset = Dataset.from_dict(data)
            
            def process_fn(example, idx):
                # Create prompt template
                question = make_prefix(example, operators=["+", "-", "*", "/"])

                # Add solution and metadata
                data = {
                    "data_source": "countdown_continual",
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": {
                            "target": example['target'],
                            "numbers": example['nums'],
                            "solution": example['solution'],
                            "rating": example['rating'],
                        }
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'operator_group': self.group_names[group_idx],
                    }
                }
                return data
            
            # Map the processing function over the dataset
            dataset = dataset.map(function=process_fn, with_indices=True)
            
            # Save dataset
            output_path = os.path.join(group_dir, f"{split}.parquet")
            dataset.to_parquet(output_path)
            rprint(f"[green]Saved {split} dataset to {output_path}[/green]")
            
            return dataset
        
        train_dataset = create_dataset(train_samples, "train")
        test_dataset = create_dataset(test_samples, "test")
        
        return train_dataset, test_dataset


if __name__ == "__main__":
    rprint("[bold blue]Countdown Task Data Generator - New Groups[/bold blue]")
    generator = DataGenerator()
    
    # Generate data for each operator group
    for group_idx in range(len(generator.group_names)):
        rprint(f"\n[bold cyan]Generating data for group {generator.group_names[group_idx]}[/bold cyan]")
        generator.generate_group_data(group_idx)
