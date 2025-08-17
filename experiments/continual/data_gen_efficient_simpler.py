"""Generate Countdown task data with simplified operator groups and train/test splits."""

import os
import random
from typing import List, Dict, Tuple
import sys

# Add project root to Python path - more robust approach
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
sys.path.insert(0, project_root)

# Also add the current directory to path
sys.path.insert(0, os.getcwd())

try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not found. Will save as JSON instead of parquet.")

# Try different import approaches
try:
    from examples.data_preprocess.countdown_directly import CountDownDirectly
    from examples.data_preprocess.countdown_reverse import CountDownReverse
except ImportError:
    # Try relative import
    import sys
    sys.path.append(os.path.join(project_root, 'examples', 'data_preprocess'))
    from countdown_directly import CountDownDirectly
    from countdown_reverse import CountDownReverse
from tqdm import tqdm
from rich import print as rprint


def make_prefix(dp, operators, template_type='base'):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""Assistant\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. \nUser\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant\nLet me solve this step by step.\n<think>"""
    elif template_type == 'llama':
        prefix = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. For example, <answer> (1 + 2) / 3 </answer>.
Let me see if I can solve this step by step.
<think>"""
    return prefix


class DataGenerator:
    def __init__(self, base_dir: str = "/app/data/continual"):  
        self.base_dir = base_dir
        # Simplified operator groups - only 2 groups now
        self.operator_groups = [
            # Group 0: multiplication-based operations
            {
                'name': '0',
                'distributions': [
                    {'weight': 0.5, 'candidate': ['+', '-', '*'], 'necessary': ['+','-','*'], 'start_size': 4},
                    {'weight': 0.25, 'candidate': ['+', '*'], 'necessary': ['+', '*'], 'start_size': 3},
                    {'weight': 0.25, 'candidate': ['-', '*'], 'necessary': ['-', '*'], 'start_size': 3}
                ]
            },
            # Group 1: division-based operations  
            {
                'name': '1',
                'distributions': [
                    {'weight': 0.5, 'candidate': ['+', '-', '/'], 'necessary': ['+', '-', '/'], 'start_size': 4},
                    {'weight': 0.25, 'candidate': ['+', '/'], 'necessary': ['+', '/'], 'start_size': 3},
                    {'weight': 0.25, 'candidate': ['-', '/'], 'necessary': ['-', '/'], 'start_size': 3}
                ]
            },
            # Group 2: mod-based operations  
            {
                'name': '2',
                'distributions': [
                    {'weight': 0.5, 'candidate': ['+', '-', '%'], 'necessary': ['+', '-', '%'], 'start_size': 4},
                    {'weight': 0.25, 'candidate': ['+', '%'], 'necessary': ['+', '%'], 'start_size': 3},
                    {'weight': 0.25, 'candidate': ['-', '%'], 'necessary': ['-', '%'], 'start_size': 3}
                ]
            },
<<<<<<< HEAD
            # {
            #     'name': '3',
            #     'distributions': [
            #         {'weight': 1, 'candidate': ['+', '-', '*', '/'], 'necessary': [], 'start_size': 4},
            #     ]
            # }
=======
            # Group 2: division-based operations  
            #{
            #    'name': '3',
            #    'distributions': [
            #        {'weight': 0.25, 'candidate': ['+', '*', '/'], 'necessary': ['+', '*', '/'], 'start_size': 4},
            #        {'weight': 0.25, 'candidate': ['-', '*', '/'], 'necessary': ['-', '*', '/'], 'start_size': 4},
            #        {'weight': 0.25, 'candidate': ['*', '/'], 'necessary': ['*', '/'], 'start_size': 3},
            #        {'weight': 0.25, 'candidate': ['*', '/'], 'necessary': ['*', '/'], 'start_size': 3}
            #    ]
            #}
>>>>>>> ae2b96219bada739e6da7968b873df1ed06cf616

        ]
        self.distinct = True
        os.makedirs(base_dir, exist_ok=True)
        os.system(f"chmod -R 777 {base_dir}")

    def generate_group_data(self, group_idx: int, train_size: int = 512000, test_size: int = 512) -> Tuple[Dataset, Dataset]:
        """Generate train and test data for a specific operator group"""
        group = self.operator_groups[group_idx]
        group_name = group['name']
        group_dir = os.path.join(self.base_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        def generate_samples(num_samples: int, seed_offset: int = 0):
            random.seed(42 + group_idx + seed_offset)
            samples = []
            
            # Calculate exact number of samples for each configuration
            group = self.operator_groups[group_idx]
            distributions = group['distributions']
            
            config_samples = []
            for dist in distributions:
                count = int(num_samples * dist['weight'])
                config_samples.append((dist, count))
            
            # Handle any rounding errors by adding remaining samples to the first config
            total_assigned = sum(count for _, count in config_samples)
            if total_assigned < num_samples:
                config_samples[0] = (config_samples[0][0], config_samples[0][1] + (num_samples - total_assigned))
            
            # Generate samples for each configuration
            for idx, (config, count) in enumerate(config_samples):
                candidate_operators = config['candidate']
                neccessary_operators = config['necessary']
                start_size = config['start_size']
<<<<<<< HEAD
                other_groups = []
                for id, operator_group in enumerate(self.operator_groups):
                    if group_idx == id:
                        other_groups += operator_group['distributions'][:idx] + operator_group['distributions'][idx+1:]
                    else:
                        other_groups += operator_group['distributions']
                # print(other_groups)
=======
                other_groups = config_samples[:idx] + config_samples[idx+1:]

>>>>>>> ae2b96219bada739e6da7968b873df1ed06cf616
                for _ in tqdm(range(count), desc=f"Generating {count} samples for {config['candidate']} config"):
                
                    cd = CountDownReverse(min_target=3, max_target=100, start_size=start_size, 
                                       max_internal_value=100, 
                                       candidate_operators=candidate_operators, 
                                       neccessary_operators=neccessary_operators,
                                       other_groups=other_groups,
                                       distinct=self.distinct)
                    target, nums, solution, full_expr = cd.generate()
                    rating = 1.0
                    samples.append({
                        "target": target,
                        "nums": nums,
                        "solution": solution,
                        "rating": rating,
                        "config": config  # Store config for debugging
                    })
            
            # Shuffle the samples to mix different configurations
            random.shuffle(samples)
            return samples
        
        rprint(f"[yellow]Generating training samples for group {group_name}...[/yellow]")
        train_samples = generate_samples(train_size)
        
        rprint(f"[yellow]Generating test samples for group {group_name}...[/yellow]")
        test_samples = generate_samples(test_size, seed_offset=100)  # Different seed for test set
        
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
                question = make_prefix(example, operators=["+", "-", "*", "/", "%"])

                # Add solution and metadata
                data = {
                    "data_source": "countdown_continual_simpler",
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
                        'operator_group': group_name,
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

    def print_group_info(self):
        """Print information about the operator groups and their distributions"""
        rprint("[bold blue]Operator Group Configuration:[/bold blue]")
        for i, group in enumerate(self.operator_groups):
            rprint(f"\n[bold cyan]Group {group['name']}:[/bold cyan]")
            for j, dist in enumerate(group['distributions']):
                rprint(f"  {dist['weight']*100:4.0f}% - candidate: {dist['candidate']}, necessary: {dist['necessary']}, nums: {dist['start_size']}")


if __name__ == "__main__":
    rprint("[bold blue]Countdown Task Data Generator - Simplified Groups[/bold blue]")
    generator = DataGenerator()
    
    # Print group configuration
    generator.print_group_info()
    
    # Generate data for each operator group
    for group_idx in range(len(generator.operator_groups)):
        group_name = generator.operator_groups[group_idx]['name']
        rprint(f"\n[bold cyan]Generating data for group {group_name}[/bold cyan]")
        generator.generate_group_data(group_idx)
