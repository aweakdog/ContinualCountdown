"""
Script to show input data from the Continual Countdown task directories.
"""
import os
import pandas as pd
from rich import print
from rich.table import Table
from rich.console import Console

OPERATOR_GROUPS = ['plus_minus_mul', 'plus_minus_div', 'minus_mul_div', 'plus_mul_div']
DATA_DIR = './data/continual'
NUM_EXAMPLES = 1280  # Number of examples to show from each split

def load_dataset(group, split='train'):
    """Load a specific dataset group and split."""
    file_path = os.path.join(DATA_DIR, group, f'{split}.parquet')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_parquet(file_path)

def show_examples(dataset, num_examples, console, group_name, split):
    """Display examples in a table format."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim")
    table.add_column("Target Number")
    table.add_column("Available Numbers")
    table.add_column("Solution", style="green")
    for i in range(min(num_examples, len(dataset))):
        row = dataset.iloc[i]
        table.add_row(
            str(i),
            str(row['target']),
            str(row['nums']),
            str(row['solution'])
        )

    console.print(f"\n[bold blue]{group_name} - {split} split[/bold blue] (showing {min(num_examples, len(dataset))} of {len(dataset)} examples)")
    console.print(table)

def main():
    console = Console()
    console.print("[bold blue]Continual Countdown Data Viewer[/bold blue]")
    
    for group in OPERATOR_GROUPS:
        try:
            # Show train split
            train_data = load_dataset(group, 'train')
            show_examples(train_data, NUM_EXAMPLES, console, group, 'train')
            
            # Show test split
            test_data = load_dataset(group, 'test')
            show_examples(test_data, NUM_EXAMPLES, console, group, 'test')
            
            console.print("\n" + "-" * 80 + "\n")  # Separator between groups
            
        except FileNotFoundError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}\n")

if __name__ == "__main__":
    main()
