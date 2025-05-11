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


def make_prefix(dp, operators, template_type='base'):
    target = dp['target']
    numbers = dp['nums']
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""Assistant\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. \nUser\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant\nLet me solve this step by step.\n<think>"""
    return prefix


class SFTDataGenerator:
    def __init__(self, base_dir: str = "./data/continual/sft/0"):
        self.base_dir = base_dir
        # Only one operator group for SFT (group 0)
        self.operator_groups = [
            [['+', '-', '*'], ['+', '-', '*']],  # plus_minus_mul
        ]
        self.group_names = ["0"]
        self.distinct = True
        os.makedirs(base_dir, exist_ok=True)
        os.system(f"chmod -R 777 {base_dir}")

    def generate_group_data(self, train_size: int = 512000, test_size: int = 512) -> Tuple[Dataset, Dataset]:
        group_idx = 0
        candidate_operators = self.operator_groups[group_idx][0]
        neccessary_operators = self.operator_groups[group_idx][1]
        group_name = self.group_names[group_idx]
        group_dir = self.base_dir
        os.makedirs(group_dir, exist_ok=True)

        def generate_samples(num_samples: int, seed_offset: int = 0):
            random.seed(44 + group_idx + seed_offset) # sft have different seed to the RL
            samples = []
            for i in tqdm(range(num_samples), desc=f"Generating {num_samples} samples for {group_name}"):
                start_size = random.randint(4, 4)
                cd = CountDownReverse(min_target=3, max_target=100, start_size=start_size,
                                      max_internal_value=100,
                                      candidate_operators=candidate_operators,
                                      neccessary_operators=neccessary_operators,
                                      distinct=self.distinct)
                target, nums, solution, full_expr = cd.generate()
                if i == 0:
                    print(f"[DEBUG] Example raw solution: {solution}")
                    print(f"[DEBUG] Example full expression: {full_expr}")
                rating = 1.0
                samples.append({
                    "target": target,
                    "nums": nums,
                    "solution": solution,
                    "full_expr": full_expr,
                    "rating": rating,
                })
            return samples

        rprint(f"[yellow]Generating training samples...[/yellow]")
        train_samples = generate_samples(train_size)

        rprint(f"[yellow]Generating test samples...[/yellow]")
        test_samples = generate_samples(test_size, seed_offset=100)

        def create_dataset(samples, split: str) -> Dataset:
            data = {
                "target": [s["target"] for s in samples],
                "nums": [s["nums"] for s in samples],
                "solution": [s["solution"] for s in samples],
                "full_expr": [s["full_expr"] for s in samples],
                "rating": [s["rating"] for s in samples],
            }
            dataset = Dataset.from_dict(data)

            def process_fn(example, idx):
                question = make_prefix(example, operators=["+"])
                # Compose the response: reasoning steps and final answer
                steps = example["solution"]
                import re
                from collections import namedtuple
                # Define a simple expression tree
                class ExprNode:
                    def __init__(self, value, left=None, right=None, op=None):
                        self.value = value  # Either a number (str) or result (str)
                        self.left = left
                        self.right = right
                        self.op = op  # '+', '-', '*', '/'
                    def __str__(self):
                        if self.op is None:
                            return str(self.value)
                        # Add parentheses for clarity
                        return f'({self.left}{self.op}{self.right})'
                def parse_step(step):
                    # e.g., '63-60=3' => lhs: '63-60', rhs: '3'
                    if '=' not in step:
                        return None, None, None
                    lhs, rhs = step.split('=')
                    lhs = lhs.strip()
                    rhs = rhs.strip()
                    # Find the operator and operands
                    for op in ['+', '-', '*', '/']:
                        # Only split on the first occurrence
                        if op in lhs:
                            parts = lhs.split(op, 1)
                            left = parts[0].strip()
                full_expr = example.get("full_expr", None)
                target_num = example.get("target", None)
                source_number = example.get("nums", None)
                if isinstance(steps, list):
                    response = "<think>\n" + "\n".join(steps) + "\n</think>"
                    st = "\n".join(steps)
                    response = f"<think>\nOur source number is: {source_number}, and our target is {target_num}.\nOne possible solution is: \n{st}, Correct!\nSo the answer should be {full_expr}\n</think>"
                else:
                    response = f"<think>\nOne possible solution is {str(steps)}, Correct! So the answer should be {full_expr}\n</think>"
                if not full_expr:
                    response += "\n<answer>None</answer>"
                else:
                    response += f"\n<answer>{full_expr}</answer><|endoftext|>"
                if idx <= 100:
                    print(f"[DEBUG] Source numbers: {example.get('nums', 'N/A')}")
                    print(f"[DEBUG] Example generated response: {response}")
                data = {
                    "prompt": question,
                    "response": response
                }
                return data

            dataset = dataset.map(function=process_fn, with_indices=True)
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
