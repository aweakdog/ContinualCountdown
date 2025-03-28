"""
Script to show input data from the Continual Countdown task directories.
"""
import os
import pandas as pd
from rich import print
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt, IntPrompt

OPERATOR_GROUPS = ['plus', 'plus_minus', 'plus_minus_mul', 'plus_minus_mul_div']
DATA_DIR = '/data/continual'  # Updated path for Docker environment

def load_dataset(group, split='train'):
    """Load a specific dataset group and split."""
    file_path = os.path.join(DATA_DIR, group, f'{split}.parquet')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_parquet(file_path)

def show_examples(dataset, start_idx, num_examples, console):
    """Display examples in a table format."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim")
    table.add_column("Target Number")
    table.add_column("Available Numbers")
    table.add_column("Solution", style="green")
    
    end_idx = min(start_idx + num_examples, len(dataset))
    for i in range(start_idx, end_idx):
        row = dataset.iloc[i]
        table.add_row(
            str(i),
            str(row['target']),
            str(row['nums']),
            str(row['solution'])
        )
    
    console.print(table)

def main():
    console = Console()
    console.print("[bold blue]Continual Countdown Data Viewer[/bold blue]")
    
    # Select operator group
    console.print("\n[yellow]Available operator groups:[/yellow]")
    for i, group in enumerate(OPERATOR_GROUPS, 1):
        console.print(f"{i}. {group}")
    
    group_idx = int(Prompt.ask("Select operator group", choices=[str(i) for i in range(1, len(OPERATOR_GROUPS) + 1)])) - 1
    group = OPERATOR_GROUPS[group_idx]
    
    # Select split
    split = Prompt.ask("Select split", choices=["train", "test"], default="train")
    
    try:
        dataset = load_dataset(group, split)
        console.print(f"\n[bold green]Loaded {split} dataset for {group}[/bold green]")
        console.print(f"Total examples: {len(dataset)}")
        
        while True:
            start_idx = IntPrompt.ask("Enter starting index", default=0)
            num_examples = IntPrompt.ask("How many examples to show", default=5)
            
            show_examples(dataset, start_idx, num_examples, console)
            
            if not Prompt.ask("Show more examples?", choices=["y", "n"], default="y") == "y":
                break
                
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
