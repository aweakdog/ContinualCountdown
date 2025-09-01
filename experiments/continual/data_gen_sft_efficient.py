"""Generate Countdown task data for SFT, only for group 0, with train/test splits."""

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
import itertools


def evaluate_step_by_step(expression_str):
    """Evaluate an expression step by step and return the steps."""
    import re
    
    def find_innermost_parentheses(expr):
        """Find the innermost parentheses and their content."""
        # Find all parentheses pairs
        stack = []
        for i, char in enumerate(expr):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    start = stack.pop()
                    if not stack:  # This is the innermost
                        return start, i, expr[start+1:i]
        return None, None, None
    
    def evaluate_simple_expression(expr):
        """Evaluate a simple expression without parentheses."""
        expr = expr.strip()
        
        # Handle multiplication first (order of operations)
        while '*' in expr:
            match = re.search(r'(\d+)\s*\*\s*(\d+)', expr)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                result = a * b
                expr = expr.replace(match.group(0), str(result), 1)
            else:
                break
        
        # Handle addition and subtraction from left to right
        while '+' in expr or '-' in expr:
            # Find first + or - operation (but not negative numbers)
            match = re.search(r'(\d+)\s*([+\-])\s*(\d+)', expr)
            if match:
                a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                if op == '+':
                    result = a + b
                else:
                    result = a - b
                expr = expr.replace(match.group(0), str(result), 1)
            else:
                break
        
        return int(expr) if expr.isdigit() or (expr.startswith('-') and expr[1:].isdigit()) else eval(expr)
    
    try:
        steps = []
        current_expr = expression_str.strip()
        original_expr = current_expr
        
        # Handle parentheses first
        while '(' in current_expr:
            start, end, inner_expr = find_innermost_parentheses(current_expr)
            if start is not None:
                # Evaluate the inner expression
                inner_result = evaluate_simple_expression(inner_expr)
                # Replace the parentheses with the result
                new_expr = current_expr[:start] + str(inner_result) + current_expr[end+1:]
                steps.append(f"{original_expr} = {new_expr}")
                current_expr = new_expr
                original_expr = new_expr
            else:
                break
        
        # Now evaluate the remaining expression without parentheses
        if '*' in current_expr or '+' in current_expr or '-' in current_expr:
            # Handle multiplication first
            # Handle multiplication (including negative numbers)
            while '*' in current_expr:
                match = re.search(r'(-?\d+)\s*\*\s*(-?\d+)', current_expr)
                if match:
                    a, b = int(match.group(1)), int(match.group(2))
                    result = a * b
                    new_expr = current_expr.replace(match.group(0), str(result), 1)
                    steps.append(f"{original_expr} = {new_expr}")
                    current_expr = new_expr
                    original_expr = new_expr
                else:
                    break
            
            # Handle addition and subtraction (including negative numbers)
            while '+' in current_expr or '-' in current_expr:
                # Match patterns like: number + number, number - number, -number + number, -number - number
                match = re.search(r'(-?\d+)\s*([+\-])\s*(-?\d+)', current_expr)
                if match:
                    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
                    if op == '+':
                        result = a + b
                    else:
                        result = a - b
                    new_expr = current_expr.replace(match.group(0), str(result), 1)
                    steps.append(f"{original_expr} = {new_expr}")
                    current_expr = new_expr
                    original_expr = new_expr
                else:
                    break
        
        final_result = int(current_expr) if current_expr.isdigit() or (current_expr.startswith('-') and current_expr[1:].isdigit()) else eval(current_expr)
        
        # If no steps were generated, add at least one step
        if not steps:
            steps = [f"{expression_str} = {final_result}"]
        
        return steps, final_result
    
    except Exception as e:
        # Fallback to simple evaluation
        try:
            result = eval(expression_str)
            return [f"{expression_str} = {result}"], result
        except:
            return [f"{expression_str} = Error"], 0


def safe_evaluate_expression(expr_str):
    """Safely evaluate expression, handling division by zero and other errors."""
    try:
        # Replace division by zero with a safe alternative
        import re
        # Check for potential division by zero patterns
        if '/ 0' in expr_str or '% 0' in expr_str:
            return None, None
        
        result = eval(expr_str)
        # Check for invalid results
        if not isinstance(result, (int, float)) or not (-1000 < result < 1000):
            return None, None
        if isinstance(result, float) and (result != result or abs(result) == float('inf')):  # NaN or inf
            return None, None
        
        return evaluate_step_by_step(expr_str)
    except (ZeroDivisionError, ValueError, OverflowError, TypeError):
        return None, None

def generate_incorrect_solutions(source_numbers, target_num, num_incorrect=3):
    """Generate incorrect solutions with complex expressions similar to correct solutions."""
    incorrect_solutions = []
    operators = ['+', '-', '*', '/', '%']
    
    # Try to generate different incorrect solutions with complex expressions
    attempts = 0
    max_attempts = 50
    
    while len(incorrect_solutions) < num_incorrect and attempts < max_attempts:
        attempts += 1
        
        # Create a random permutation of source numbers
        nums = source_numbers.copy()
        random.shuffle(nums)
        
        if len(nums) >= 4:
            # Generate complex expressions similar to correct solutions
            # Pattern 1: (a op b) op (c op d)
            # Pattern 2: ((a op b) op c) op d
            # Pattern 3: a op ((b op c) op d)
            
            pattern = random.choice([1, 2, 3])
            ops = [random.choice(operators) for _ in range(3)]
            
            if pattern == 1:
                # (a op b) op (c op d)
                expr_str = f"({nums[0]} {ops[0]} {nums[1]}) {ops[1]} ({nums[2]} {ops[2]} {nums[3]})"
            elif pattern == 2:
                # ((a op b) op c) op d
                expr_str = f"(({nums[0]} {ops[0]} {nums[1]}) {ops[1]} {nums[2]}) {ops[2]} {nums[3]}"
            else:
                # a op ((b op c) op d)
                expr_str = f"{nums[0]} {ops[0]} (({nums[1]} {ops[1]} {nums[2]}) {ops[2]} {nums[3]})"
        
        elif len(nums) >= 3:
            # For 3 numbers: a op (b op c) or (a op b) op c
            ops = [random.choice(operators) for _ in range(2)]
            if random.choice([True, False]):
                expr_str = f"{nums[0]} {ops[0]} ({nums[1]} {ops[1]} {nums[2]})"
            else:
                expr_str = f"({nums[0]} {ops[0]} {nums[1]}) {ops[1]} {nums[2]}"
        
        else:
            # Fallback for 2 numbers
            op = random.choice(operators)
            expr_str = f"{nums[0]} {op} {nums[1]}"
        
        # Use safe evaluation
        steps, result = safe_evaluate_expression(expr_str)
        
        # Only add if evaluation was successful and result is different from target
        if steps is not None and result is not None and result != target_num:
            # Create step-by-step solution string
            if len(steps) > 1:
                # Extract intermediate steps, avoiding trivial final steps
                intermediate_steps = []
                for step in steps:
                    parts = step.split(' = ')
                    if len(parts) >= 2:
                        intermediate_steps.append(parts[1])
                
                # Remove the last step if it's trivial (same as result)
                if intermediate_steps and intermediate_steps[-1] == str(result):
                    intermediate_steps = intermediate_steps[:-1]
                
                if intermediate_steps:
                    step_by_step = ' = '.join(intermediate_steps)
                    bad_solution = f"{expr_str} = {step_by_step} = {result}"
                else:
                    bad_solution = f"{expr_str} = {result}"
            else:
                bad_solution = f"{expr_str} = {result}"
            
            if bad_solution not in incorrect_solutions:
                incorrect_solutions.append(bad_solution)
    
    # If we couldn't generate enough, pad with simple incorrect ones
    while len(incorrect_solutions) < num_incorrect:
        if len(source_numbers) >= 2:
            a, b = random.sample(source_numbers, 2)
            op = random.choice(operators)
            try:
                result = eval(f"{a} {op} {b}")
                if result != target_num and isinstance(result, (int, float)):
                    bad_solution = f"{a} {op} {b} = {result}"
                    if bad_solution not in incorrect_solutions:
                        incorrect_solutions.append(bad_solution)
            except:
                pass
        
        # Fallback: create a simple incorrect solution
        if len(incorrect_solutions) < num_incorrect:
            fallback_result = target_num + random.randint(1, 10) * random.choice([-1, 1])
            if len(source_numbers) >= 2:
                a, b = source_numbers[0], source_numbers[1]
                incorrect_solutions.append(f"{a} + {b} = {fallback_result}")
        
        break  # Prevent infinite loop
    
    return incorrect_solutions[:num_incorrect]


def make_prefix(dp, operators, template_type='base'):
    target = dp['target']
    numbers = dp['nums']
    user_message = f"Using the numbers {numbers}, create an equation that equals {target}. Use the basic arithmetic operations ({', '.join(operators)}). These operators follow Python's rules of execution (e.g., / performs precise division, % performs a modulo operation). Each number must be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: {user_message}\nAssistant: Let me solve this step by step.\n<think>"""
    elif template_type == 'qwen-instruct':
        # Qwen2.5-Instruct format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        prefix = f"""<|im_start|>system
You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""
    elif template_type == 'llama-instruct':
        # Llama3.2-Instruct format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...<|eot_id|><|start_header_id|>user<|end_header_id|>\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let me solve this step by step.
<think>"""
    return prefix


class SFTDataGenerator:
    def __init__(self, base_dir: str = "./data/continual/sft/0", model_type: str = "base"):
        self.base_dir = base_dir
        self.model_type = model_type
        # Map model_type to template_type
        template_mapping = {
            'llama': 'llama-instruct',
            'qwen': 'qwen-instruct', 
            'base': 'base'
        }
        self.template_type = template_mapping.get(model_type, 'base')
        # Group 0 operator configurations with exact proportions
        self.operator_configs = [
            {
                'candidate_operators': ['+', '-', '*'],
                'necessary_operators': ['+', '-', '*'],
                'start_size': 4,
                'proportion': 0.5,  # 50%
                'name': 'plus_minus_mul_4nums'
            },
            {
                'candidate_operators': ['+', '*'],
                'necessary_operators': ['+', '*'],
                'start_size': 3,
                'proportion': 0.25,  # 25%
                'name': 'plus_mul_3nums'
            },
            {
                'candidate_operators': ['-', '*'],
                'necessary_operators': ['-', '*'],
                'start_size': 3,
                'proportion': 0.25,  # 25%
                'name': 'minus_mul_3nums'
            }
        ]
        self.distinct = True
        os.makedirs(base_dir, exist_ok=True)
        os.system(f"chmod -R 777 {base_dir}")

    def generate_group_data(self, train_size: int = 2048, test_size: int = 512) -> Tuple[Dataset, Dataset]:
        group_dir = self.base_dir
        os.makedirs(group_dir, exist_ok=True)
        
        # Print configuration info
        rprint(f"[bold blue]SFT Data Generation Configuration:[/bold blue]")
        for i, config in enumerate(self.operator_configs):
            samples_count = int(train_size * config['proportion'])
            rprint(f"  Config {i}: {config['name']} - {config['proportion']*100}% ({samples_count} samples)")
            rprint(f"    Operators: {config['candidate_operators']}, Start size: {config['start_size']}")

        def generate_samples_for_config(config, num_samples: int, seed_offset: int = 0):
            """Generate samples for a specific operator configuration."""
            random.seed(44 + seed_offset + hash(config['name']) % 1000)  # Different seed per config
            samples = []
            
            candidate_operators = config['candidate_operators']
            necessary_operators = config['necessary_operators']
            start_size = config['start_size']
            config_name = config['name']
            
            for i in tqdm(range(num_samples), desc=f"Generating {config_name} samples"):
                cd = CountDownReverse(min_target=3, max_target=100, start_size=start_size,
                                      max_internal_value=100,
                                      candidate_operators=candidate_operators,
                                      neccessary_operators=necessary_operators,
                                      distinct=self.distinct)
                target, nums, solution, full_expr = cd.generate()
                if i == 0:
                    rprint(f"[green]Sample {i}: target={target}, nums={nums}, solution={solution}, full_expr={full_expr}[/green]")
                samples.append({
                    "target": target,
                    "nums": nums,
                    "solution": solution,
                    "full_expr": full_expr,
                    "config_name": config_name  # Track which config generated this sample
                })
            return samples

        def generate_samples(total_samples: int, seed_offset: int = 0):
            """Generate samples with exact proportions for each operator configuration."""
            all_samples = []
            
            for config in self.operator_configs:
                # Calculate exact number of samples for this config
                config_samples = int(total_samples * config['proportion'])
                
                rprint(f"[yellow]Generating {config_samples} samples for {config['name']}...[/yellow]")
                config_sample_list = generate_samples_for_config(config, config_samples, seed_offset)
                all_samples.extend(config_sample_list)
            
            # Shuffle all samples to mix different configurations
            random.seed(42 + seed_offset)
            random.shuffle(all_samples)
            
            rprint(f"[green]Generated {len(all_samples)} total samples[/green]")
            return all_samples

        rprint(f"[yellow]Generating training samples...[/yellow]")
        train_samples = generate_samples(train_size)

        rprint(f"[yellow]Generating test samples...[/yellow]")
        test_samples = generate_samples(test_size, seed_offset=100)

        def create_dataset(samples, split: str) -> Dataset:
            # Create datasets for all template types
            data = {
                "prompt": [],  # Unified field for default template
                "response": [],  # Unified field for response
                "prompt_base": [],
                "response_base": [],
                "prompt_qwen_instruct": [],
                "response_qwen_instruct": [],
                "prompt_llama_instruct": [],
                "response_llama_instruct": [],
                "target": [],
                "nums": [],
                "full_expr": []
            }
            for idx, s in enumerate(samples):
                # Generate prompts for all template types
                template_types = ['base', 'qwen-instruct', 'llama-instruct']
                questions = {}
                operators = ["+","-","*","/","%"]
                for template_type in template_types:
                    questions[template_type] = make_prefix(s, operators=operators, template_type=template_type)
                
                steps = s["solution"]
                full_expr = s.get("full_expr", None)
                target_num = s.get("target", None)
                source_number = s.get("nums", None)
                
                # Generate response (same for all templates)
                response = f"Looking at this problem, I need to use the numbers {source_number} each exactly once to create an equation that equals {target_num}.\n\n"
                
                # Randomly decide number of incorrect attempts (0-2)
                import random
                num_incorrect = random.randint(0, 1)
                incorrect_solutions = generate_incorrect_solutions(source_number, target_num, num_incorrect=num_incorrect) if num_incorrect > 0 else []
                
                #response += "Let me try some approaches:\n\n"
                
                # Add incorrect attempts
                if incorrect_solutions:
                    for i, incorrect in enumerate(incorrect_solutions):
                        response += f"{i+1}, {incorrect}, Not correct, Let's try another one.\n\n"
                
                # Add correct solution as the final attempt
                try:
                    steps_result, final_result = evaluate_step_by_step(full_expr)
                    # Format the correct solution in the same style as incorrect ones
                    thinking_process = " = ".join([step.split(" = ")[1] for step in steps_result if " = " in step])
                    attempt_num = len(incorrect_solutions) + 1
                    if thinking_process:
                        response += f"{attempt_num}, {full_expr} = {thinking_process} Correct!\n"
                    else:
                        response += f"{attempt_num}, {full_expr} = {final_result} Correct!\n"
                except:
                    attempt_num = len(incorrect_solutions) + 1
                    response += f"{attempt_num}, {full_expr} Correct!\n"
                
                response += f"</think>\n\n<answer>{full_expr}</answer>"
                
                # Debug: print response for verification
                
                
                # Set unified fields based on model_type
                data["prompt"].append(str(questions[self.template_type]))
                data["response"].append(str(response))
                if idx < 10:  # Only print first 2 samples to avoid spam
                    print(str(questions[self.template_type]))
                    print(str(response))
                    print("=" * 60)
                
                # Store all template versions for compatibility
                data["prompt_base"].append(str(questions['base']))
                data["response_base"].append(str(response))
                data["prompt_qwen_instruct"].append(str(questions['qwen-instruct']))
                data["response_qwen_instruct"].append(str(response))
                data["prompt_llama_instruct"].append(str(questions['llama-instruct']))
                data["response_llama_instruct"].append(str(response))
                
                # Store metadata
                data["target"].append(target_num)
                data["nums"].append(source_number)
                data["full_expr"].append(full_expr)
            # Debug: print the first few samples for each template
            print(f"\n[DEBUG] First few {split} samples for each template:")
            for i in range(min(3, len(data["prompt_base"]))):
                print(f"Sample {i} (Base):\n  Prompt: {data['prompt_base'][i][:1000]}{'...' if len(data['prompt_base'][i])>1000 else ''}\n")
                print(f"Sample {i} (Qwen):\n  Prompt: {data['prompt_qwen_instruct'][i][:1000]}{'...' if len(data['prompt_qwen_instruct'][i])>1000 else ''}\n")
                print(f"Sample {i} (Llama):\n  Prompt: {data['prompt_llama_instruct'][i][:1000]}{'...' if len(data['prompt_llama_instruct'][i])>1000 else ''}\n")
            dataset = Dataset.from_dict(data)
            output_path = os.path.join(group_dir, f"{split}.parquet")
            dataset.to_parquet(output_path)
            rprint(f"[green]Saved {split} dataset to {output_path}[/green]")
            return dataset

        train_dataset = create_dataset(train_samples, "train")
        test_dataset = create_dataset(test_samples, "test")
        return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SFT data for Countdown task")
    parser.add_argument("--model_type", default="base", choices=['llama', 'qwen', 'base'], 
                       help="Model type for template selection (default: base)")
    parser.add_argument("--base_dir", default="./data/continual/sft/0", 
                       help="Base directory for output data")
    parser.add_argument("--train_size", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=1000,
                       help="Number of test samples")
    
    args = parser.parse_args()
    
    # Update base_dir to include model_type
    model_base_dir = f"{args.base_dir.rstrip('/')}/{args.model_type}"
    
    rprint(f"[bold blue]Countdown SFT Data Generator - Model Type: {args.model_type}[/bold blue]")
    rprint(f"[yellow]Output directory: {model_base_dir}[/yellow]")
    
    generator = SFTDataGenerator(base_dir=model_base_dir, model_type=args.model_type)
    generator.generate_group_data(train_size=args.train_size, test_size=args.test_size)
