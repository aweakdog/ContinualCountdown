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
    user_message = f"Using the numbers {numbers}, create an equation that equals {target}. Use the basic arithmetic operations ({', '.join(operators)}). These operators follow Python's rules of execution (e.g., / performs precise division, % performs a modulo operation). Each number must be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {user_message}
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen2.5-Instruct Models"""
        # Qwen2.5-Instruct format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        prefix = f"""<|im_start|>system
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    elif template_type == 'llama-instruct':
        """This works for Llama3.2-Instruct Models"""
        # Llama3.2-Instruct format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...<|eot_id|><|start_header_id|>user<|end_header_id|>\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step.
<think>"""
    return prefix

'''
python experiments/continual/data_gen_efficient_simpler.py \
    --model_type all \
    --train_size 512000 \
    --test_size 512
'''

class DataGenerator:
    def __init__(self, base_dir: str = "/app/data/continual", model_type: str = "base"):  
        self.base_dir = base_dir
        self.model_type = model_type
        # Map model_type to template_type for backward compatibility
        template_mapping = {
            'llama': 'llama-instruct',
            'qwen': 'qwen-instruct', 
            'base': 'base'
        }
        self.template_type = template_mapping.get(model_type, 'base')
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
            # Group 2: division-based operations  
            {
                'name': '2',
                'distributions': [
                    {'weight': 0.5, 'candidate': ['+', '-', '%'], 'necessary': ['+', '-', '%'], 'start_size': 4},
                    {'weight': 0.25, 'candidate': ['+', '%'], 'necessary': ['+', '%'], 'start_size': 3},
                    {'weight': 0.25, 'candidate': ['-', '%'], 'necessary': ['-', '%'], 'start_size': 3}
                ]
            }
 
            ## Group 2: division-based operations  
            #{
            #    'name': '2',
            #    'distributions': [
            #        {'weight': 0.25, 'candidate': ['+', '*', '/'], 'necessary': ['+', '*', '/'], 'start_size': 4},
            #        {'weight': 0.25, 'candidate': ['-', '*', '/'], 'necessary': ['-', '*', '/'], 'start_size': 4},
            #        {'weight': 0.5, 'candidate': ['*', '/'], 'necessary': ['*', '/'], 'start_size': 3},
            #    ]
            #}
            
            #{
            #    'name': '3',
            #    'distributions': [
            #        {'weight': 1, 'candidate': ['+', '-', '*', '/'], 'necessary': [], 'start_size': 4},
            #    ]
            #}
        ]
        # Simplified operator groups - only 2 groups now
        #self.operator_groups = [
        #    # Group 0: + and % operations
        #    {
        #        'name': '0',
        #        'distributions': [
        #            {'weight': 0.2, 'candidate': ['+', '%'], 'necessary': ['+', '%'], 'start_size': 3},
        #            {'weight': 0.1, 'candidate': ['%'], 'necessary': ['%', '%'], 'start_size': 3},
        #            {'weight': 0.1, 'candidate': ['+'], 'necessary': ['+', '+'], 'start_size': 3},
        #            {'weight': 0.2, 'candidate': ['+', '%'], 'necessary': ['+', '%', '%'], 'start_size': 4},
        #            {'weight': 0.1, 'candidate': ['%'], 'necessary': ['%', '%', '%'], 'start_size': 4},
        #            {'weight': 0.2, 'candidate': ['+', '%'], 'necessary': ['+', '+', '%'], 'start_size': 4},
        #            {'weight': 0.1, 'candidate': ['+'], 'necessary': ['+', '+', '+'], 'start_size': 4}
        #        ]
        #    },
        #    # Group 1: - and / operations
        #    {
        #        'name': '1',
        #        'distributions': [
        #            {'weight': 0.2, 'candidate': ['-', '/'], 'necessary': ['-', '/'], 'start_size': 3},
        #            {'weight': 0.1, 'candidate': ['/'], 'necessary': ['/', '/'], 'start_size': 3},
        #            {'weight': 0.1, 'candidate': ['-'], 'necessary': ['-', '-'], 'start_size': 3},
        #            {'weight': 0.2, 'candidate': ['-', '/'], 'necessary': ['-', '/', '/'], 'start_size': 4},
        #            {'weight': 0.1, 'candidate': ['/'], 'necessary': ['/', '/', '/'], 'start_size': 4},
        #            {'weight': 0.2, 'candidate': ['-', '/'], 'necessary': ['-', '-', '/'], 'start_size': 4},
        #            {'weight': 0.1, 'candidate': ['-'], 'necessary': ['-', '-', '-'], 'start_size': 4}
        #        ]
        #    },
        #    # Group 2: * and @ operations
        #    {
        #        'name': '2',
        #        'distributions': [
        #            {'weight': 0.2, 'candidate': ['*', '@'], 'necessary': ['*', '@'], 'start_size': 3},
        #            {'weight': 0.1, 'candidate': ['@'], 'necessary': ['@', '@'], 'start_size': 3},
        #            {'weight': 0.1, 'candidate': ['*'], 'necessary': ['*', '*'], 'start_size': 3},
        #            {'weight': 0.2, 'candidate': ['*', '@'], 'necessary': ['*', '@', '@'], 'start_size': 4},
        #            {'weight': 0.1, 'candidate': ['@'], 'necessary': ['@', '@', '@'], 'start_size': 4},
        #            {'weight': 0.2, 'candidate': ['*', '@'], 'necessary': ['*', '*', '@'], 'start_size': 4},
        #            {'weight': 0.1, 'candidate': ['*'], 'necessary': ['*', '*', '*'], 'start_size': 4}
        #        ]
        #    }
        #]
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
                other_groups = []
                for id, operator_group in enumerate(self.operator_groups):
                    if group_idx == id:
                        other_groups += operator_group['distributions'][:idx] + operator_group['distributions'][idx+1:]
                    else:
                        other_groups += operator_group['distributions']
                # print(other_groups)
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
            # Convert samples to proper dataset format with all template types
            data = {
                "target": [s["target"] for s in samples],
                "nums": [s["nums"] for s in samples],
                "solution": [s["solution"] for s in samples],
                "rating": [s["rating"] for s in samples],
                "prompt": [],  # Unified field for default template
                "response": [],  # Unified field for response
                "prompt_base": [],
                "prompt_qwen_instruct": [],
                "prompt_llama_instruct": [],
            }
            
            # Generate prompts for all template types for each sample
            for i, sample in enumerate(samples):
                template_types = ['base', 'qwen-instruct', 'llama-instruct']
                prompts = {}
                for template_type in template_types:
                    question = make_prefix(sample, operators=["+", "-", "*", "/", "%"], template_type=template_type)
                    prompts[template_type] = question
                    data[f"prompt_{template_type.replace('-', '_')}"].append(question)
                
                # Set default prompt based on model_type
                data["prompt"].append(prompts[self.template_type])
                
                # Generate unified response (same for all templates)
                response = f"Looking at this problem, I need to use the numbers {sample['nums']} to create an equation that equals {sample['target']}.\n\nAfter working through the possibilities, I found: {sample['solution']}\n\n<answer>{sample['solution']}</answer>"
                data["response"].append(response)
            
            dataset = Dataset.from_dict(data)
            
            def process_fn(example, idx):
                # Add solution and metadata (keeping original structure for compatibility)
                data = {
                    "data_source": "countdown_continual_simpler",
                    "prompt": [{
                        "role": "user", 
                        "content": example['prompt'],  # Use unified prompt field
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
                        'model_type': self.model_type,
                        'template_type': self.template_type,
                    },
                    # Keep unified fields
                    "prompt": example['prompt'],
                    "response": example['response'],
                    # Keep all template prompts for compatibility
                    "prompt_base": example['prompt_base'],
                    "prompt_qwen_instruct": example['prompt_qwen_instruct'],
                    "prompt_llama_instruct": example['prompt_llama_instruct'],
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

    def generate_samples(self, group_idx, num_samples, seed_offset=0):
        """Generate raw samples for a specific group"""
        group = self.operator_groups[group_idx]
        
        def generate_samples_internal(count, seed_offset=0):
            """Internal function to generate samples"""
            samples = []
            
            # Set random seed for reproducibility
            random.seed(42 + seed_offset)
            
            for idx, config in enumerate(group['distributions']):
                weight = config['weight']
                sample_count = int(count * weight)
                
                # Adjust for rounding errors
                if idx == len(group['distributions']) - 1:
                    sample_count = count - len(samples)
                
                candidate_operators = config['candidate']
                neccessary_operators = config['necessary']
                start_size = config['start_size']
                other_groups = []
                for id, operator_group in enumerate(self.operator_groups):
                    if group_idx == id:
                        other_groups += operator_group['distributions'][:idx] + operator_group['distributions'][idx+1:]
                    else:
                        other_groups += operator_group['distributions']
                
                for _ in tqdm(range(sample_count), desc=f"Generating {sample_count} samples for {config['candidate']} config"):
                    cd = CountDownReverse(min_target=3, max_target=100, start_size=start_size, 
                                       max_internal_value=100, 
                                       candidate_operators=candidate_operators, 
                                       neccessary_operators=neccessary_operators,
                                       other_groups=other_groups,
                                       distinct=self.distinct)
                    target, nums, solution, full_expr = cd.generate()
                    
                    # Create question and answer using the same format as SFT data generator
                    sample_data = {"target": target, "nums": nums}
                    question = make_prefix(sample_data, operators=["+", "-", "*", "/", "%"], template_type='base')
                    answer = solution
                    
                    samples.append({
                        "target": target,
                        "numbers": nums,
                        "operators": candidate_operators,
                        "question": question,
                        "answer": answer,
                        "solution": solution,
                        "config": config
                    })
            
            # Shuffle the samples to mix different configurations
            random.shuffle(samples)
            return samples
        
        return generate_samples_internal(num_samples, seed_offset)

    def format_prompt(self, question, template_type):
        """Format prompt with different template types"""
        instruction_following = "Please solve this step by step and provide the final answer."
        
        if template_type == 'qwen' or template_type == 'qwen-instruct':
            # Qwen instruct format
            user_message = f"{question}\n\n{instruction_following}"
            return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        elif template_type == 'llama' or template_type == 'llama-instruct':
            # Llama instruct format
            user_message = f"{question}\n\n{instruction_following}"
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Base format
            return f"{question}"

    def save_samples_with_template(self, group_idx, train_samples, test_samples):
        """Save pre-generated samples with the current template"""
        group = self.operator_groups[group_idx]
        group_name = group['name']
        group_dir = os.path.join(self.base_dir, f"group_{group_name}")
        os.makedirs(group_dir, exist_ok=True)
        
        def create_dataset_from_samples(samples, split):
            """Convert raw samples to dataset with current template"""
            def process_fn(example, idx):
                # Apply current template to the raw sample
                if self.model_type == 'all':
                    # Generate all three templates
                    prompt_base = self.format_prompt(example['question'], 'base')
                    prompt_qwen = self.format_prompt(example['question'], 'qwen')
                    prompt_llama = self.format_prompt(example['question'], 'llama')
                    
                    data = {
                        "data_source": "countdown",
                        "ability": "arithmetic",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": example['answer']
                        },
                        "extra_info": {
                            'split': split,
                            'index': idx,
                            'group': group_name,
                            'operators': example['operators'],
                            'numbers': example['numbers'],
                            'target': example['target'],
                            'question': example['question'],
                            'answer': example['answer']
                        },
                        "prompt_base": prompt_base,
                        "prompt_qwen_instruct": prompt_qwen,
                        "prompt_llama_instruct": prompt_llama,
                    }
                else:
                    # Generate single template
                    prompt = self.format_prompt(example['question'], self.model_type)
                    data = {
                        "data_source": "countdown",
                        "ability": "arithmetic", 
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": example['answer']
                        },
                        "extra_info": {
                            'split': split,
                            'index': idx,
                            'group': group_name,
                            'operators': example['operators'],
                            'numbers': example['numbers'],
                            'target': example['target'],
                            'question': example['question'],
                            'answer': example['answer']
                        },
                        "prompt": prompt,
                    }
                return data
            
            # Convert samples to dataset
            dataset = Dataset.from_list(samples)
            dataset = dataset.map(function=process_fn, with_indices=True)
            
            # Save dataset
            output_path = os.path.join(group_dir, f"{split}.parquet")
            dataset.to_parquet(output_path)
            rprint(f"[green]Saved {split} dataset to {output_path}[/green]")
            
            return dataset
        
        train_dataset = create_dataset_from_samples(train_samples, "train")
        test_dataset = create_dataset_from_samples(test_samples, "test")
        
        return train_dataset, test_dataset

    def print_group_info(self):
        """Print information about the operator groups and their distributions"""
        rprint("[bold blue]Operator Group Configuration:[/bold blue]")
        for i, group in enumerate(self.operator_groups):
            rprint(f"\n[bold cyan]Group {group['name']}:[/bold cyan]")
            for j, dist in enumerate(group['distributions']):
                rprint(f"  {dist['weight']*100:4.0f}% - candidate: {dist['candidate']}, necessary: {dist['necessary']}, nums: {dist['start_size']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Countdown task data with unified templates")
    parser.add_argument("--model_type", default="all", choices=['llama', 'qwen', 'base', 'all'], 
                       help="Model type for template selection. Use 'all' to generate all 3 templates (default: all)")
    parser.add_argument("--base_dir", default="/app/data/continual", 
                       help="Base directory for output data")
    parser.add_argument("--group", type=int, default=None,
                       help="Generate data for specific group only (0, 1, 2)")
    parser.add_argument("--train_size", type=int, default=512000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=512,
                       help="Number of test samples")
    
    args = parser.parse_args()
    
    if args.model_type == 'all':
        # Generate data ONCE and apply to all 3 templates
        model_types = ['base', 'llama', 'qwen']
        base_dirs = {
            'base': './data/base',
            'llama': './data/llama_instruct', 
            'qwen': './data/qwen_instruct'
        }
        
        rprint(f"[bold blue]Countdown Task Data Generator - Generating ALL Templates from Same Data[/bold blue]")
        
        # Create a single generator to generate the base data once
        temp_generator = DataGenerator(base_dir="./temp", model_type='base')
        
        # Generate data for specific group or all groups
        if args.group is not None:
            if args.group < len(temp_generator.operator_groups):
                group_name = temp_generator.operator_groups[args.group]['name']
                rprint(f"\n[bold cyan]Generating base data for group {group_name}[/bold cyan]")
                
                # Generate the raw data once
                train_samples = temp_generator.generate_samples(args.group, args.train_size)
                test_samples = temp_generator.generate_samples(args.group, args.test_size)
                
                # Apply each template to the same data
                for model_type in model_types:
                    model_base_dir = base_dirs[model_type]
                    rprint(f"\n[bold magenta]=== Applying {model_type.upper()} template to same data ===[/bold magenta]")
                    rprint(f"[yellow]Output directory: {model_base_dir}[/yellow]")
                    
                    generator = DataGenerator(base_dir=model_base_dir, model_type=model_type)
                    generator.save_samples_with_template(args.group, train_samples, test_samples)
            else:
                rprint(f"[red]Error: Group {args.group} not found. Available groups: 0, 1, 2[/red]")
        else:
            # Generate data for all operator groups
            for group_idx in range(len(temp_generator.operator_groups)):
                group_name = temp_generator.operator_groups[group_idx]['name']
                rprint(f"\n[bold cyan]Generating base data for group {group_name}[/bold cyan]")
                
                # Generate the raw data once for this group
                train_samples = temp_generator.generate_samples(group_idx, args.train_size)
                test_samples = temp_generator.generate_samples(group_idx, args.test_size)
                
                # Apply each template to the same data
                for model_type in model_types:
                    model_base_dir = base_dirs[model_type]
                    rprint(f"  Applying {model_type.upper()} template to group {group_name}")
                    
                    generator = DataGenerator(base_dir=model_base_dir, model_type=model_type)
                    generator.save_samples_with_template(group_idx, train_samples, test_samples)
        
        rprint(f"\n[bold green]All templates generated successfully![/bold green]")
        rprint(f"[yellow]Data locations:[/yellow]")
        for model_type, base_dir in base_dirs.items():
            rprint(f"  {model_type.upper()}: {base_dir}")
    else:
        # Generate data for single model type
        rprint(f"[bold blue]Countdown Task Data Generator - Model Type: {args.model_type}[/bold blue]")
        generator = DataGenerator(base_dir=args.base_dir, model_type=args.model_type)
        
        # Print group configuration
        generator.print_group_info()
        
        # Generate data for specific group or all groups
        if args.group is not None:
            if args.group < len(generator.operator_groups):
                group_name = generator.operator_groups[args.group]['name']
                rprint(f"\n[bold cyan]Generating data for group {group_name} with {args.model_type} template[/bold cyan]")
                generator.generate_group_data(args.group, train_size=args.train_size, test_size=args.test_size)
            else:
                rprint(f"[red]Error: Group {args.group} not found. Available groups: 0, 1, 2[/red]")
        else:
            # Generate data for all operator groups
            for group_idx in range(len(generator.operator_groups)):
                group_name = generator.operator_groups[group_idx]['name']
                rprint(f"\n[bold cyan]Generating data for group {group_name} with {args.model_type} template[/bold cyan]")
                generator.generate_group_data(group_idx, train_size=args.train_size, test_size=args.test_size)
