# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DeepMath-103K dataset to parquet format
"""

import os
import re
import json
import argparse
import pandas as pd
import glob
import json
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    """Extract the final answer from boxed format like \\boxed{answer}"""
    if not solution_str:
        return None
        
    # Try to find boxed answer first - handle nested braces
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    boxed_match = re.search(boxed_pattern, solution_str)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Fallback: try to find the last mathematical expression
    patterns = [
        r"-?\\d?frac\{[^}]+\}\{[^}]+\}",  # LaTeX fractions
        r"[a-zA-Z]\^?\d*\s*[+\-=]\s*\d+",  # simple equations
        r"-?\d+(?:\.\d+)?",  # decimals
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, solution_str)
        if matches:
            return matches[-1].strip()
    
    return None


if __name__ == '__main__':
    '''
    python examples/data_preprocess/deepmath.py \
  --from_local \
  --local_dir ./data/deepmath/0
    '''

    '''
    python examples/data_preprocess/deepmath.py \
  --from_local \
  --local_dir ./data/deepmath/0 \
  --prepend_cot_examples \
  --preview_n 3
    '''
    '''
    python examples/data_preprocess/deepmath.py --from_local --local_dir ./data/DeepMath-103K/data --max_prompt_length 1000 --prepend_cot_examples --output_dir ./data/deepmath/0
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/deepmath')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--from_local', action='store_true',
                        help='Load DeepMath from local parquet files instead of the Hugging Face Hub')
    parser.add_argument('--local_parquet_dir', default=None,
                        help='Directory containing train-*.parquet files. '
                             'Defaults to data/DeepMath-103K/data/ when --from_local is set')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for processed parquet files. If not set, saves to local_parquet_dir')
    # CoT multi-shot prefix options
    parser.add_argument('--prepend_cot_examples', action='store_true',
                        help='Prepend a fixed 4-shot CoT Q/A block before each question')
    parser.add_argument('--cot_examples_file', default=None,
                        help='Path to a text file containing the CoT examples block. If not set, a built-in 4-shot block is used')
    # Preview options
    parser.add_argument('--preview_n', type=int, default=0,
                        help='Print N complete processed training samples to stdout for inspection')
    # Data sampling options
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (useful for creating smaller datasets)')
    parser.add_argument('--max_prompt_length', type=int, default=800,
                        help='Maximum prompt length in tokens. Samples exceeding this will be filtered out. Should be lower than model max_length to account for chat template overhead.')

    args = parser.parse_args()

    data_source = 'zwhe99/DeepMath-103K'
    
    # Initialize tokenizer for token-based length filtering
    # Use the local Qwen tokenizer from SFT model
    tokenizer_path = "/nas/shared/sys2/yuanhangli/tmp/qwen_sft_model/global_step_0"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Local tokenizer not found at {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Using local Qwen tokenizer from: {tokenizer_path}")

    # Load dataset: from local parquet if requested; otherwise from HF Hub
    if args.from_local:
        data_dir = args.local_parquet_dir or './data/DeepMath-103K/data/'
        
        # Find all parquet training files
        parquet_files = glob.glob(os.path.join(data_dir, 'train-*.parquet'))
        
        if len(parquet_files) > 0:
            print(f"Found {len(parquet_files)} parquet files in {data_dir}")
            dataset = datasets.load_dataset('parquet', data_files={'train': parquet_files})
        else:
            raise FileNotFoundError(f"No train-*.parquet files found in {data_dir}")
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    
    # Sample dataset if max_samples is specified
    if args.max_samples and len(train_dataset) > args.max_samples:
        print(f"Sampling {args.max_samples} from {len(train_dataset)} total samples")
        train_dataset = train_dataset.select(range(args.max_samples))

    # Instruction to guide the model's answer formatting
    instruction_following = "Solve this step by step and put your final answer in \\boxed{}."

    # Prepare CoT examples block if requested
    cot_examples = None
    if args.prepend_cot_examples:
        if args.cot_examples_file is not None:
            with open(args.cot_examples_file, 'r', encoding='utf-8') as f:
                cot_examples = f.read().strip()
        else:
            # Default 4-shot CoT block using real DeepMath-103K style examples
            #cot_examples = (
            #    "Below are 4 worked examples. Follow the same reasoning style and answer format.\n\n"
            #    "Q: Find the limit: $\\lim_{x \\to 0} \\frac{\\sin(3x)}{x}$.\n"
            #    "A: I need to evaluate this limit. Since we have the indeterminate form $\\frac{0}{0}$, I can use L'Hôpital's rule or the standard limit $\\lim_{u \\to 0} \\frac{\\sin u}{u} = 1$. Using the substitution $u = 3x$, as $x \\to 0$, we have $u \\to 0$. So $\\frac{\\sin(3x)}{x} = \\frac{\\sin(3x)}{3x} \\cdot 3 = 3 \\cdot \\frac{\\sin(3x)}{3x} \\to 3 \\cdot 1 = 3$. The answer is 3.\n\\boxed{3}\n\n"
            #    "Q: Solve the quadratic equation $x^2 - 5x + 6 = 0$.\n"
            #    "A: I need to find the roots of $x^2 - 5x + 6 = 0$. I can factor this quadratic. Looking for two numbers that multiply to 6 and add to -5, I get -2 and -3. So $x^2 - 5x + 6 = (x - 2)(x - 3) = 0$. This gives us $x = 2$ or $x = 3$. The answer is $x = 2, 3$.\n\\boxed{x = 2, 3}\n\n"
            #    "Q: Find the derivative of $f(x) = x^3 + 2x^2 - 4x + 1$.\n"
            #    "A: I need to find $f'(x)$ using the power rule. For each term $ax^n$, the derivative is $nax^{n-1}$. So $f'(x) = 3x^2 + 2 \\cdot 2x - 4 + 0 = 3x^2 + 4x - 4$. The answer is $3x^2 + 4x - 4$.\n\\boxed{3x^2 + 4x - 4}\n\n"
            #    "Q: Evaluate $\\int_0^2 (3x^2 + 2x) dx$.\n"
            #    "A: I need to compute this definite integral. First, I find the antiderivative: $\\int (3x^2 + 2x) dx = x^3 + x^2 + C$. Now I evaluate from 0 to 2: $[x^3 + x^2]_0^2 = (2^3 + 2^2) - (0^3 + 0^2) = 8 + 4 - 0 = 12$. The answer is 12.\n\\boxed{12}\n"
            #)
            cot_examples = (
                "Below are a worked example. Follow the same reasoning style and answer format.\n\n"
                "Q: Solve the quadratic equation $x^2 - 5x + 6 = 0$.\n"
                "A: I need to find the roots of $x^2 - 5x + 6 = 0$. I can factor this quadratic. Looking for two numbers that multiply to 6 and add to -5, I get -2 and -3. So $x^2 - 5x + 6 = (x - 2)(x - 3) = 0$. This gives us $x = 2$ or $x = 3$. The answer is $x = 2, 3$.\n\\boxed{x = 2, 3}\n\n"
            )

    # Process the dataset
    def process_fn(example, idx):
        question_raw = example['question']
        final_answer_raw = example['final_answer']
        # Use the first solution as the reference solution
        solution_raw = example['r1_solution_1']

        # Assemble prompt with optional CoT examples block in Q/A format
        if cot_examples is not None:
            prompt_body = (
                cot_examples
                + "\n\nNow solve the following question by following the above style.\n\n"
                + f"Q: {question_raw}\nA: "
                + instruction_following
            )
        else:
            prompt_body = question_raw + ' ' + instruction_following

        # Check prompt length using chat template (same as training) and filter if too long
        messages = [{"role": "user", "content": prompt_body}]
        prompt_with_chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt_with_chat_template, add_special_tokens=True)
        prompt_token_length = len(tokens)
        
        if prompt_token_length > args.max_prompt_length:
            return None

        # Extract ground truth answer from final_answer field
        ground_truth = example['final_answer']
        if not ground_truth:
            return None
        
        # Clean up ground truth - remove quotes only
        ground_truth = ground_truth.strip('"\'').strip()

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt_body,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'final_answer': final_answer_raw,
                'solution': solution_raw,
                "question": question_raw,
                'difficulty': example.get('difficulty', None),
                'topic': example.get('topic', None),
            }
        }
        return data

    # Get original dataset size for statistics
    original_size = len(train_dataset)
    
    train_dataset = train_dataset.map(function=process_fn, with_indices=True)
    
    # Filter out None values (filtered samples)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    
    # Calculate filtering statistics
    retained_samples = len(train_dataset)
    filtered_samples = original_size - retained_samples
    
    print(f"\n===== Filtering Statistics =====")
    print(f"Total samples processed: {original_size}")
    print(f"Samples filtered out (prompt > {args.max_prompt_length} tokens): {filtered_samples}")
    print(f"Samples retained: {retained_samples}")
    print(f"Filter rate: {filtered_samples/original_size*100:.2f}%")

    # Create test dataset by splitting from train (DeepMath doesn't have separate test set)
    # Use fixed 1025 samples for test, rest for train
    total_samples_after_filter = len(train_dataset)
    test_size = min(1025, total_samples_after_filter)  # At most 1025 samples for test
    train_size = total_samples_after_filter - test_size
    
    test_dataset = train_dataset.select(range(test_size))
    train_dataset = train_dataset.select(range(test_size, total_samples_after_filter))
    
    print(f"Split dataset: {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    # Preview a few processed samples
    if args.preview_n and args.preview_n > 0:
        n = min(args.preview_n, len(train_dataset))
        print(f"\n===== Preview {n} processed training samples =====")
        preview_ds = train_dataset.select(range(n))
        for i, ex in enumerate(preview_ds):
            print("--------------------------------")
            print(f"[Train Sample {i}]")
            try:
                print(json.dumps(ex, ensure_ascii=False, indent=2))
            except Exception:
                print(ex)

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(output_dir, 'test.parquet'))
    print(f"Saved {len(train_dataset)} samples to {os.path.join(output_dir, 'train.parquet')}")
    print(f"Saved {len(test_dataset)} samples to {os.path.join(output_dir, 'test.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
