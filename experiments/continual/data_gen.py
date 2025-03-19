"""Generate Countdown task data with different operator groups and train/test splits."""

import os
from typing import List, Dict, Tuple
from datasets import Dataset
from examples.data_preprocess.countdown import gen_dataset, make_prefix
from rich.progress import track
from rich import print as rprint

class DataGenerator:
    def __init__(self, base_dir: str = "/data/continual"):
        self.base_dir = base_dir
        self.operator_groups = [
            ["+"],
            ["+", "-"],
            ["+", "-", "*"],
            ["+", "-", "*", "/"]
        ]
        self.group_names = [
            "plus",
            "plus_minus",
            "plus_minus_mul",
            "plus_minus_mul_div"
        ]
        os.makedirs(base_dir, exist_ok=True)

    def generate_group_data(self, group_idx: int, train_size: int = 100000, test_size: int = 1000) -> Tuple[Dataset, Dataset]:
        """Generate train and test data for a specific operator group"""
        operators = self.operator_groups[group_idx]
        group_name = self.group_names[group_idx]
        group_dir = os.path.join(self.base_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        rprint(f"[yellow]Generating training samples...[/yellow]")
        # Generate raw samples for train and test
        train_samples = gen_dataset(
            num_samples=train_size,
            num_operands=6,
            max_target=1000,
            min_number=1,
            max_number=100,
            operations=operators,
            seed_value=42 + group_idx  # Different seed for each group
        )
        
        rprint(f"[yellow]Generating test samples...[/yellow]")
        test_samples = gen_dataset(
            num_samples=test_size,
            num_operands=6,
            max_target=1000,
            min_number=1,
            max_number=100,
            operations=operators,
            seed_value=42 + group_idx + 100  # Different seed for test set
        )
        
        # Convert to dataset format
        def create_dataset(samples, split: str) -> Dataset:
            data = {
                "target": [s[0] for s in samples],
                "nums": [s[1] for s in samples]
            }
            dataset = Dataset.from_dict(data)
            
            # Add prompts using the base template
            def add_prompts(example):
                example["prompt"] = make_prefix(example, template_type="base")
                return example
            
            dataset = dataset.map(add_prompts)
            
            # Save dataset
            output_path = os.path.join(group_dir, f"{split}.parquet")
            dataset.to_parquet(output_path)
            rprint(f"[green]Saved {split} dataset to {output_path}[/green]")
            
            return dataset
        
        train_dataset = create_dataset(train_samples, "train")
        test_dataset = create_dataset(test_samples, "test")
        
        return train_dataset, test_dataset

if __name__ == "__main__":
    rprint("[bold blue]Countdown Task Data Generator[/bold blue]")
    generator = DataGenerator()
    
    # Generate data for each operator group
    for group_idx in track(range(len(generator.operator_groups)), description="Generating datasets"):
        rprint(f"\n[bold cyan]Processing operator group: {generator.operator_groups[group_idx]}[/bold cyan]")
        train_dataset, test_dataset = generator.generate_group_data(group_idx)
        rprint(f"[green]✓ Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples[/green]")
        rprint(f"[blue]  Saved in /data/continual/{generator.group_names[group_idx]}/[/blue]")
