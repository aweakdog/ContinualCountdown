"""
Script to show input data of the Countdown task.
"""
from datasets import load_dataset
from rich import print
from rich.table import Table
from rich.console import Console

def main():
    # Load the dataset
    console = Console()
    console.print("[bold blue]Loading Countdown Task Dataset...[/bold blue]")
    
    dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    
    # Create a table to display samples
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim")
    table.add_column("Target Number")
    table.add_column("Available Numbers")
    
    # Show first 10 examples
    for i in range(10):
        example = dataset[i]
        table.add_row(
            str(i),
            str(example['target']),
            str(example['nums'])
        )
    
    console.print("\n[bold green]First 10 examples from the dataset:[/bold green]")
    console.print(table)
    
    # Print dataset statistics
    console.print("\n[bold yellow]Dataset Statistics:[/bold yellow]")
    console.print(f"Total number of examples: {len(dataset)}")
    
if __name__ == "__main__":
    main()
