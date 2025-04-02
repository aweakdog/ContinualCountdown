#!/usr/bin/env python3
"""
Script to test the difficulty of countdown problems using DeepSeek API.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), 'log')
os.makedirs(log_dir, exist_ok=True)

# Clear existing log file
log_file = os.path.join(log_dir, 'difficulty.log')
if os.path.exists(log_file):
    open(log_file, 'w').close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class DifficultyTester:
    def __init__(self, api_key: str):
        """Initialize the difficulty tester with DeepSeek API key."""
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def create_prompt(self, numbers: List[int], target: int, operators: List[str]) -> str:
        """Create a prompt for difficulty classification."""
        base_prompt = """You are a math problem difficulty classifier. Given a math problem, analyze its difficulty based on the criteria below and output your analysis in a structured format.

CRITERIA:
1. Operations Required:
   - Easy: Only addition/subtraction
   - Medium: Includes multiplication/division
   - Hard: Requires parentheses/nested operations

2. Solution Steps:
   - Easy: Three-step solution
   - Medium: Four-step solution
   - Hard: Five+ steps or non-integer intermediates

3. Proximity to Target:
   - Easy: Direct combination (e.g., 3+5=8)
   - Medium: Indirect but logical path (e.g., (6-2)*3=12)
   - Hard: Requires creative grouping (e.g., 8/(3-(8/3))=24)

PROBLEM:
Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations [+,-,*,/] and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.

OUTPUT FORMAT:
<reasoning>Brief explanation of the classification based on the above criteria.</reasoning>
<difficulty>easy|medium|hard</difficulty>
"""
        return base_prompt.format(
            numbers=numbers,
            target=target,
            operators=operators
        )

    def classify_difficulty(self, numbers: List[int], target: int, operators: List[str]) -> Dict[str, Any]:
        """Classify the difficulty of a problem using DeepSeek API."""
        prompt = self.create_prompt(numbers, target, operators)
        
        logging.info(f"\nAnalyzing problem - Numbers: {numbers}, Target: {target}, Operators: {operators}")
        
        try:
            # Log the request details
            logging.info("Sending request to DeepSeek API...")
            
            # Send the request using OpenAI SDK
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=8192,
                stream=False
            )
            # Log the response
            logging.info(f"Raw API Response: {response}")
            
            answer = response.choices[0].message.content
            logging.info(f"\nDeepSeek Response Content:\n{answer}")
            
            # Extract difficulty from <difficulty> tags
            import re
            difficulty = re.search(r"<difficulty>(.*?)</difficulty>", answer)
            thinking = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
            
            difficulty_level = difficulty.group(1) if difficulty else "unknown"
            reasoning = thinking.group(1).strip() if thinking else ""
            
            logging.info(f"Extracted Difficulty: {difficulty_level}")
            if reasoning:
                logging.info(f"Reasoning:\n{reasoning}")
            
            return {
                "difficulty": difficulty_level,
                "reasoning": reasoning,
                "full_response": answer
            }
            
        except Exception as e:
            error_msg = f"Error calling DeepSeek API: {str(e)}"
            logging.error(error_msg)
            if hasattr(e, 'response'):
                logging.error(f"Response status: {e.response.status_code}")
                logging.error(f"Response content: {e.response.text}")
            return {
                "difficulty": "error",
                "reasoning": str(e),
                "full_response": str(e)
            }

    def analyze_dataset(self, parquet_file: str, output_file: str = None, max_samples: int = 1000, dataset_name: str = None):
        """Analyze all problems in a parquet dataset.
        
        Args:
            parquet_file (str): Path to the parquet file containing problems
            output_file (str, optional): Path to save results. Defaults to None.
            max_samples (int, optional): Maximum number of samples to analyze. Defaults to 1000.
        """
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        
        # Sample if dataset is larger than max_samples
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            
        results = []
        
        # Add dataset name to track source in combined results
        if dataset_name is None:
            dataset_name = os.path.basename(os.path.dirname(parquet_file))
        
        # Calculate time estimates
        total_problems = len(df)
        avg_time_per_problem = 5  # Estimate 5 seconds per API call
        estimated_total_time = total_problems * avg_time_per_problem
        estimated_finish_time = datetime.now() + timedelta(seconds=estimated_total_time)
        
        print(f"\nDataset: {dataset_name}")
        print(f"Total problems: {total_problems}")
        print(f"Estimated time: {estimated_total_time//60} minutes {estimated_total_time%60} seconds")
        print(f"Estimated finish time: {estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        processed_count = 0
        
        for _, row in df.iterrows():
            # Handle both JSON strings and numpy arrays
            numbers = row['nums'].tolist() if hasattr(row['nums'], 'tolist') else json.loads(row['nums'])
            target = int(row['target'])
            
            # Use all operators since they're filtered at solution generation time
            operators = ['+', '-', '*', '/']
            
            result = self.classify_difficulty(numbers, target, operators)
            result.update({
                "dataset": dataset_name,
                "dataset_name": dataset_name,  # Add this for crosstab
                "dataset_path": os.path.dirname(parquet_file),  # Add this for crosstab
                "problem_id": len(results),
                "numbers": numbers,
                "target": target,
                "operators": operators,
                "solution": row.get('solution', None)
            })
            results.append(result)
            
            # Update progress and ETA
            processed_count += 1
            if processed_count % 10 == 0:  # Update every 10 problems
                elapsed_time = time.time() - start_time
                avg_time = elapsed_time / processed_count
                remaining_time = (total_problems - processed_count) * avg_time
                new_eta = datetime.now() + timedelta(seconds=remaining_time)
                
                print(f"\nProgress: {processed_count}/{total_problems} problems")
                print(f"Average time per problem: {avg_time:.1f} seconds")
                print(f"Estimated time remaining: {remaining_time//60:.0f} minutes {remaining_time%60:.0f} seconds")
                print(f"New ETA: {new_eta.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add a small delay to respect API rate limits
            time.sleep(0.5)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if output file is specified
        if output_file:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_parquet(output_file)
        
        return results_df

def plot_difficulty_distribution(results_df: pd.DataFrame, output_dir: str, dataset_name: str = None):
    """Create visualizations for difficulty distribution."""
    # Treat 'unknown' as 'hard'
    results_df = results_df.copy()
    results_df['difficulty'] = results_df['difficulty'].replace('unknown', 'hard')
    plt.figure(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # Create the main bar plot
    difficulty_counts = results_df["difficulty"].value_counts()
    ax = sns.barplot(
        x=difficulty_counts.index,
        y=difficulty_counts.values,
        hue=difficulty_counts.index,
        palette={"easy": "#2ecc71", "medium": "#f1c40f", "hard": "#e74c3c"},
        legend=False
    )
    
    # Add value labels on top of each bar
    for i, v in enumerate(difficulty_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    # Customize the plot
    title = f"Difficulty Distribution{' for ' + dataset_name if dataset_name else ''}"
    plt.title(title, pad=20, fontsize=14, fontweight="bold")
    plt.xlabel("Difficulty Level", labelpad=10)
    plt.ylabel("Number of Problems", labelpad=10)
    
    # Add percentage labels
    total = len(results_df)
    percentages = (difficulty_counts / total * 100).round(1)
    for i, (count, pct) in enumerate(zip(difficulty_counts.values, percentages)):
        ax.text(i, count/2, f"{pct}%", ha='center', va='center', color='white', fontweight='bold')
    
    # Create plot directory and save the plot
    if dataset_name:
        # Remove any file extension from dataset_name
        base_name = os.path.splitext(dataset_name)[0]
        plot_dir = os.path.join(output_dir, base_name)
    else:
        plot_dir = output_dir
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_name = "difficulty_distribution.png"
    plt.savefig(os.path.join(plot_dir, plot_name), bbox_inches='tight', dpi=300)
    plt.close()

def plot_difficulty_heatmap(results_df: pd.DataFrame, output_dir: str):
    """Create a heatmap showing difficulty distribution across datasets."""
    # Treat 'unknown' as 'hard'
    results_df = results_df.copy()
    results_df['difficulty'] = results_df['difficulty'].replace('unknown', 'hard')
    plt.figure(figsize=(12, 8))
    
    # Create cross-tabulation
    heatmap_data = pd.crosstab(results_df["dataset"], results_df["difficulty"], normalize="index") * 100
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap="YlOrRd",
        cbar_kws={'label': 'Percentage of Problems'},
    )
    
    plt.title("Difficulty Distribution Across Datasets", pad=20, fontsize=14, fontweight="bold")
    plt.xlabel("Difficulty Level", labelpad=10)
    plt.ylabel("Dataset", labelpad=10)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "difficulty_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()

def find_parquet_files(base_dir: str, pattern: str = "train.parquet") -> List[Dict[str, str]]:
    """Recursively find all parquet files matching the pattern.
    Returns a list of dicts with input_path and relative_path.
    """
    matches = []
    base_dir = os.path.abspath(base_dir)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == pattern:
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_dir)
                matches.append({
                    'input_path': input_path,
                    'relative_path': relative_path,
                    'filename': file
                })
    return sorted(matches, key=lambda x: x['input_path'])  # Sort for consistent ordering

def read_api_key(api_key_file: str = "deepseek_api.txt") -> str:
    """Read DeepSeek API key from file."""
    try:
        with open(api_key_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key file '{api_key_file}' not found. "
            "Please create this file with your DeepSeek API key."
        )

def plot_dataset_comparison(continual_results: pd.DataFrame, tinyzero_results: pd.DataFrame, output_dir: str):
    """Create comparison plots between continual and tinyzero datasets."""
    plt.figure(figsize=(12, 6))
    
    # Calculate percentages for both datasets
    continual_counts = continual_results['difficulty'].value_counts(normalize=True) * 100
    tinyzero_counts = tinyzero_results['difficulty'].value_counts(normalize=True) * 100
    
    # Set up bar positions
    bar_width = 0.35
    r1 = np.arange(len(continual_counts))
    r2 = [x + bar_width for x in r1]
    
    # Create grouped bar chart
    plt.bar(r1, continual_counts, color='skyblue', width=bar_width, label='Continual')
    plt.bar(r2, tinyzero_counts, color='lightgreen', width=bar_width, label='TinyZero')
    
    # Customize the plot
    plt.xlabel('Difficulty Level')
    plt.ylabel('Percentage of Problems')
    plt.title('Difficulty Distribution: Continual vs TinyZero')
    plt.xticks([r + bar_width/2 for r in range(len(continual_counts))], continual_counts.index)
    
    # Add percentage labels on bars
    for i, v in enumerate(continual_counts):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(tinyzero_counts):
        plt.text(i + bar_width, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test difficulty of countdown problems using DeepSeek API")
    parser.add_argument("--data-dir", nargs="+", help="One or more base directories to recursively search for train.parquet files")
    parser.add_argument("--input", nargs="*", help="Specific parquet files to analyze (optional)")
    parser.add_argument("--pattern", default="train.parquet", help="Pattern to match parquet files (default: train.parquet)")
    parser.add_argument("--output-dir", default="difficulty_results", help="Output directory for results")
    parser.add_argument("--api-key-file", default="deepseek_api.txt", help="File containing DeepSeek API key")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum number of samples to analyze per dataset (default: 1000)")
    args = parser.parse_args()
    
    # Read API key from file
    api_key = read_api_key(args.api_key_file)
    
    # Collect input files from all data directories
    input_files = []
    if args.data_dir:
        for data_dir in args.data_dir:
            # Get the dataset name from the first directory component after /app/data/
            # This ensures files from /app/data/continual and /app/data/tinyzero are treated correctly
            path_parts = os.path.normpath(data_dir).split(os.sep)
            try:
                data_idx = path_parts.index('data')
                dataset_name = path_parts[data_idx + 1] if data_idx + 1 < len(path_parts) else os.path.basename(data_dir)
            except ValueError:
                dataset_name = os.path.basename(data_dir)
            
            files = find_parquet_files(data_dir, args.pattern)
            for file_info in files:
                file_info['dataset_name'] = dataset_name
            input_files.extend(files)
    
    if args.input:
        # For specific files, create similar structure
        for file_path in args.input:
            abs_path = os.path.abspath(file_path)
            # Determine dataset name from path
            if 'continual' in abs_path:
                dataset_name = 'continual'
            elif 'tinyzero' in abs_path:
                dataset_name = 'tinyzero'
            else:
                dataset_name = 'other'
            
            input_files.append({
                'input_path': abs_path,
                'relative_path': os.path.dirname(os.path.relpath(abs_path, args.data_dir[0] if args.data_dir else os.path.dirname(abs_path))),
                'filename': os.path.basename(abs_path),
                'dataset_name': dataset_name
            })
    
    if not input_files:
        raise ValueError(
            "No input files specified. Either provide --data-dir for recursive search "
            "or --input for specific files."
        )
    
    tester = DifficultyTester(api_key)
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Keep track of processed files to avoid duplicates
    processed_files = set()
    all_results = []
    for file_info in input_files:
        input_path = file_info['input_path']
        # Skip if we've already processed this file
        if input_path in processed_files:
            continue
        processed_files.add(input_path)
        relative_path = file_info['relative_path']
        
        print(f"\nAnalyzing {input_path}...")
        
        # Create output directory mirroring input structure
        output_subdir = os.path.join(args.output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Generate output filenames
        base_name = file_info['filename']
        output_base = base_name.replace('.parquet', '')
        
        # Data output
        output_file = os.path.join(output_subdir, f"{output_base}_difficulty.parquet")
        
        # Analyze dataset
        results = tester.analyze_dataset(
            parquet_file=input_path,
            output_file=output_file,
            max_samples=args.max_samples,
            dataset_name=file_info['dataset_name']  # Pass dataset_name
        )
        results['dataset_path'] = relative_path  # Add path info for combined analysis
        all_results.append(results)
        
        # Generate visualizations for this dataset
        plot_difficulty_distribution(
            results,
            output_subdir,
            dataset_name=os.path.join(relative_path, output_base)
        )
        
        # Print per-file statistics
        print(f"\nDifficulty Distribution for {os.path.join(relative_path, base_name)}:")
        print(results["difficulty"].value_counts())
    
    # Combine all results and save overall statistics
    if len(all_results) > 1:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results in the root output directory
        combined_output = os.path.join(args.output_dir, "combined_difficulty.parquet")
        combined_results.to_parquet(combined_output)
        
        print("\nOverall Difficulty Distribution:")
        print(combined_results["difficulty"].value_counts())
        
        # Generate difficulty distribution by dataset path and type
        print("\nDifficulty Distribution by Dataset:")
        dataset_stats = pd.crosstab([combined_results["dataset_name"], combined_results["dataset_path"]], 
                                   combined_results["difficulty"])
        print(dataset_stats)
        
        # Create overall visualizations in root directory
        plot_difficulty_distribution(combined_results, args.output_dir, "overall")
        plot_difficulty_heatmap(combined_results, args.output_dir)
        
        # Create dataset-type level visualizations
        for dataset_name in combined_results["dataset_name"].unique():
            dataset_results = combined_results[combined_results["dataset_name"] == dataset_name]
            dataset_dir = os.path.join(args.output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            plot_difficulty_distribution(dataset_results, dataset_dir, f"{dataset_name}_overall")
            
            # Create group-level visualizations within each dataset
            for path in dataset_results["dataset_path"].unique():
                group_results = dataset_results[dataset_results["dataset_path"] == path]
                group_dir = os.path.join(dataset_dir, path)
                plot_difficulty_distribution(group_results, group_dir, f"group_{os.path.basename(path)}")
        
        # Create comparison between continual and tinyzero
        if 'continual' in combined_results['dataset_name'].unique() and \
           'tinyzero' in combined_results['dataset_name'].unique():
            continual_results = combined_results[combined_results['dataset_name'] == 'continual']
            tinyzero_results = combined_results[combined_results['dataset_name'] == 'tinyzero']
            plot_dataset_comparison(continual_results, tinyzero_results, args.output_dir)

if __name__ == "__main__":
    main()
