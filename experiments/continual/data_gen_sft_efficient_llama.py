"""Generate Countdown task data for SFT for Llama, with variable train/test splits."""

import os
import random
import argparse
from typing import List, Dict, Tuple
from datasets import Dataset
import sys
# Get the directory of the current script and add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)
from examples.data_preprocess.countdown_directly import CountDownDirectly
from examples.data_preprocess.countdown_reverse import CountDownReverse
from tqdm import tqdm
from rich import print as rprint


def make_prefix(dp, operators, template_type='llama'):
    target = dp['target']
    numbers = dp['nums']
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""Assistant\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer. \nUser\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations ({', '.join(operators)}) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant\nLet me solve this step by step.\n<think>"""
    elif template_type == 'llama':
        prefix = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. For example, <answer> (1 + 2) / 3 </answer>.
Let me see if I can solve this step by step.
<think>"""
    return prefix


class SFTDataGenerator:
    def __init__(self, train_size: int, test_size: int):
        self.base_dir = f"./data/continual/sft/{train_size}"
        self.train_size = train_size
        self.test_size = test_size
        # Only one operator group for SFT (group 0)
        self.operator_groups = [
            [['+', '-', '*'], ['+', '-', '*']],  # plus_minus_mul
        ]
        self.group_names = ["0"]
        self.distinct = True
        os.makedirs(self.base_dir, exist_ok=True)
        os.system(f"chmod -R 777 {self.base_dir}")

    def generate_group_data(self) -> Tuple[Dataset, Dataset]:
        group_idx = 0
        candidate_operators = self.operator_groups[group_idx][0]
        neccessary_operators = self.operator_groups[group_idx][1]
        group_name = self.group_names[group_idx]
        group_dir = self.base_dir

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
                rating = 1.0
                samples.append({
                    "target": target,
                    "nums": nums,
                    "solution": solution,
                    "full_expr": full_expr,
                    "rating": rating,
                })
            return samples

        rprint(f"[yellow]Generating {self.train_size} training samples...[/yellow]")
        train_samples = generate_samples(self.train_size)

        rprint(f"[yellow]Generating {self.test_size} test samples...[/yellow]")
        test_samples = generate_samples(self.test_size, seed_offset=100)

        def create_dataset(samples, split: str) -> Dataset:
            data = {
                "prompt": [],
                "response": []
            }
            for s in samples:
                question = make_prefix(s, operators=self.operator_groups[0][0], template_type='llama')
                steps = s["solution"]
                full_expr = s.get("full_expr", None)
                target_num = s.get("target", None)
                source_number = s.get("nums", None)
                if isinstance(steps, list):
                    st = "\n".join(steps)
                    response = f"<think>\nOur source number is: {source_number}, and our target is {target_num}.\nOne possible solution is: \n{st}, Correct!\nSo the answer should be {full_expr}\n</think>"
                else:
                    response = f"<think>\nOne possible solution is {str(steps)}, Correct! So the answer should be {full_expr}\n</think>"
                if not full_expr:
                    response += "\n<answer>None</answer>"
                else:
                    response += f"\n<answer>{full_expr}</answer>"
                
                # The response for Llama should end with the end-of-turn token
                response += "<|eot_id|>"

                data["prompt"].append(str(question))
                data["response"].append(str(response))

            dataset = Dataset.from_dict(data)
            output_path = os.path.join(group_dir, f"{split}.parquet")
            dataset.to_parquet(output_path)
            rprint(f"[green]Saved {split} dataset to {output_path}[/green]")
            return dataset

        train_dataset = create_dataset(train_samples, "train")
        test_dataset = create_dataset(test_samples, "test")
        return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Countdown SFT Data Generator for Llama.")
    parser.add_argument("--train_size", type=int, required=True, help="Number of training samples to generate.")
    parser.add_argument("--test_size", type=int, default=512, help="Number of test samples to generate.")
    args = parser.parse_args()

    rprint(f"[bold blue]Countdown SFT Data Generator for Llama - Train Size: {args.train_size}[/bold blue]")
    generator = SFTDataGenerator(train_size=args.train_size, test_size=args.test_size)
    generator.generate_group_data()
