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


def generate_incorrect_solutions(source_numbers, target_num, num_incorrect=3):
    """Generate incorrect solutions with step-by-step reasoning using source numbers and random operators."""
    incorrect_solutions = []
    operators = ['+', '-', '*']
    
    # Try to generate different incorrect solutions
    attempts = 0
    max_attempts = 50
    
    while len(incorrect_solutions) < num_incorrect and attempts < max_attempts:
        attempts += 1
        
        # Create a random permutation of source numbers
        nums = source_numbers.copy()
        random.shuffle(nums)
        
        # Generate random operators for the expression
        if len(nums) >= 2:
            # For simplicity, create expressions with 2-4 numbers
            expr_length = min(len(nums), random.randint(4, 4))
            selected_nums = nums[:expr_length]
            
            # Generate random operators between numbers
            selected_ops = [random.choice(operators) for _ in range(expr_length - 1)]
            
            # Build expression string
            expr_parts = []
            for i, num in enumerate(selected_nums):
                expr_parts.append(str(num))
                if i < len(selected_ops):
                    expr_parts.append(selected_ops[i])
            
            expr_str = ' '.join(expr_parts)
            
            try:
                # Get step-by-step evaluation
                steps, result = evaluate_step_by_step(expr_str)
                
                # Only add if result is different from target and is a reasonable number
                if result != target_num and isinstance(result, (int, float)) and -1000 < result < 1000:
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
            except:
                # Skip invalid expressions
                continue
    
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
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""Assistant\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. \nUser\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant\nLet me solve this step by step.\n<think>"""
    return prefix


class SFTDataGenerator:
    def __init__(self, base_dir: str = "./data/continual/sft/0"):
        self.base_dir = base_dir
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
            # Only keep prompt/response in the final dataset
            data = {
                "prompt": [],
                "response": []
            }
            for s in samples:
                # Determine operators based on config_name
                config_name = s.get("config_name", "plus_minus_mul_4nums")
                if "plus_minus_mul" in config_name:
                    operators = ["+", "-", "*"]
                elif "plus_mul" in config_name:
                    operators = ["+", "*"]
                elif "minus_mul" in config_name:
                    operators = ["-", "*"]
                else:
                    operators = ["+", "-", "*"]  # fallback
                
                # Compose prompt/response as in process_fn, but ensure string type
                question = make_prefix(s, operators=operators)
                steps = s["solution"]
                full_expr = s.get("full_expr", None)
                target_num = s.get("target", None)
                source_number = s.get("nums", None)
                # Generate incorrect solutions
                num_incorrect = random.randint(0, 1)
                incorrect_solutions = generate_incorrect_solutions(source_number, target_num, num_incorrect)
                
                # Generate step-by-step reasoning for the correct solution
                try:
                    if full_expr:
                        correct_steps, correct_result = evaluate_step_by_step(full_expr)
                        if len(correct_steps) > 1:
                            # Extract intermediate steps, avoiding trivial final steps
                            intermediate_steps = []
                            for step in correct_steps:
                                parts = step.split(' = ')
                                if len(parts) >= 2:
                                    intermediate_steps.append(parts[1])
                            
                            # Remove the last step if it's trivial (same as target)
                            if intermediate_steps and intermediate_steps[-1] == str(target_num):
                                intermediate_steps = intermediate_steps[:-1]
                            
                            if intermediate_steps:
                                step_by_step_correct = ' = '.join(intermediate_steps)
                                correct_solution_text = f"{full_expr} = {step_by_step_correct} = {target_num}"
                            else:
                                correct_solution_text = f"{full_expr} = {target_num}"
                        else:
                            correct_solution_text = f"{full_expr} = {target_num}"
                    else:
                        correct_solution_text = f"No valid solution found"
                except:
                    correct_solution_text = f"{full_expr} = {target_num}"
                
                if isinstance(steps, list):
                    st = "\n".join(steps)
                    # Build response with incorrect solutions first, then correct one
                    solution_text = "One possible solution is:\n"
                    for bad_sol in incorrect_solutions:
                        solution_text += f"{bad_sol}, Incorrect! So Let's try next one.\n"
                    solution_text += f"{correct_solution_text}, Correct!"
                    
                    response = f"Our source number is: {source_number}, and our target is {target_num}.\n{solution_text}\n</think>"
                else:
                    # Build response with incorrect solutions first, then correct one
                    solution_text = "One possible solution is:\n"
                    for bad_sol in incorrect_solutions:
                        solution_text += f"{bad_sol}, Incorrect!\n"
                    solution_text += f"{correct_solution_text}, Correct!"
                    
                    response = f"{solution_text}\n</think>"
                if not full_expr:
                    response += "\n<answer>None</answer>"
                else:
                    response += f"\n<answer>{full_expr}</answer>"
                data["prompt"].append(str(question))
                data["response"].append(str(response))
            # Debug: print the first 100 prompt/response pairs
            print(f"\n[DEBUG] First 100 {split} samples:")
            for i in range(min(10, len(data["prompt"]))):
                prompt = data["prompt"][i]
                response = data["response"][i]
                print(f"Sample {i}:\n  Prompt: {prompt[:1000]}{'...' if len(prompt)>1000 else ''}\n  Response: {response[:1000]}{'...' if len(response)>1000 else ''}\n")
            dataset = Dataset.from_dict(data)
            output_path = os.path.join(group_dir, f"{split}.parquet")
            dataset.to_parquet(output_path)
            rprint(f"[green]Saved {split} dataset to {output_path}[/green]")
            return dataset

        train_dataset = create_dataset(train_samples, "train")
        test_dataset = create_dataset(test_samples, "test")
        return train_dataset, test_dataset


if __name__ == "__main__":
    rprint("[bold blue]Countdown SFT Data Generator - Group 0 Only[/bold blue]")
    generator = SFTDataGenerator()
    generator.generate_group_data()
