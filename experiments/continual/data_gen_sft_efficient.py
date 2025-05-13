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

    def generate_group_data(self, train_size: int = 320, test_size: int = 512) -> Tuple[Dataset, Dataset]:
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
            # Only keep prompt/response in the final dataset
            data = {
                "prompt": [],
                "response": []
            }
            for s in samples:
                # Compose prompt/response as in process_fn, but ensure string type
                question = make_prefix(s, operators=["+"])
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
                data["prompt"].append(str(question))
                data["response"].append(str(response))
            # Debug: print the first 100 prompt/response pairs
            print(f"\n[DEBUG] First 100 {split} samples:")
            for i in range(min(100, len(data["prompt"]))):
                prompt = data["prompt"][i]
                response = data["response"][i]
                print(f"Sample {i}:\n  Prompt: {prompt[:120]}{'...' if len(prompt)>120 else ''}\n  Response: {response[:120]}{'...' if len(response)>120 else ''}\n")
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
