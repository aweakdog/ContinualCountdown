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


def make_prefix(question, instruction_following, cot_examples=None, template_type='base'):
    """Generate prompt with different template formats"""
    if cot_examples is not None:
        user_message = (
            cot_examples
            + "\n\nNow solve the following question by following the above style.\n\n"
            + f"Q: {question}\nA: "
            + instruction_following
        )
    else:
        user_message = question + ' ' + instruction_following
    
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step.
User: {user_message}
Assistant: """
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant that solves advanced math problems step by step.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    elif template_type == 'llama-instruct':
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that solves advanced math problems step by step.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prefix


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
    '''
    python examples/data_preprocess/deepmath.py --from_local --local_parquet_dir ./data/DeepMath-103K/data --output_dir ./data/deepmath --max_prompt_length 1000 --prepend_cot_examples --template_type all --model_type all
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
    # Template support
    parser.add_argument('--template_type', default='all', choices=['base', 'qwen-instruct', 'llama-instruct', 'all'],
                        help='Template type to generate. "all" generates all templates in the same dataset')
    parser.add_argument('--model_type', default='all', choices=['base', 'qwen', 'llama', 'all', 'single'],
                        help='Model type for directory structure. "all" generates separate datasets for each template')
    parser.add_argument('--tokenizer_path', default=None,
                        help='Path to tokenizer for length filtering. If not set, uses default paths based on template')

    args = parser.parse_args()

    data_source = 'zwhe99/DeepMath-103K'

def process_single_model_type(args, data_source):
    # Initialize tokenizer for token-based length filtering
    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        # Use default tokenizer paths
        if args.template_type == 'qwen-instruct' or args.template_type == 'all':
            tokenizer_path = "/cpfs04/user/liyuanhang.p/model/qwen_instruct3b"
        elif args.template_type == 'llama-instruct':
            tokenizer_path = "/cpfs04/user/liyuanhang.p/model/llama_instruct3b"
        else:
            tokenizer_path = "/cpfs04/user/liyuanhang.p/model/qwen_instruct3b"  # default
    
    if not os.path.exists(tokenizer_path):
        print(f"Warning: Tokenizer not found at {tokenizer_path}, skipping length filtering")
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Using tokenizer from: {tokenizer_path}")

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

        # Extract ground truth answer from final_answer field
        ground_truth = example['final_answer']
        if not ground_truth:
            return None
        
        # Clean up ground truth - remove quotes only
        ground_truth = ground_truth.strip('"\'').strip()
        
        # Generate prompts for all templates if template_type is 'all'
        templates_to_generate = ['base', 'qwen-instruct', 'llama-instruct'] if args.template_type == 'all' else [args.template_type]
        
        data = {
            "data_source": data_source,
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
        
        # Generate prompts for each template
        valid_templates = []
        for template in templates_to_generate:
            prompt_prefix = make_prefix(question_raw, instruction_following, cot_examples, template)
            
            # Length filtering if tokenizer is available
            if tokenizer is not None:
                tokens = tokenizer.encode(prompt_prefix, add_special_tokens=False)
                if len(tokens) > args.max_prompt_length:
                    continue  # Skip this template if too long
            
            valid_templates.append(template)
            
            # Store prompt and response for this template
            if template == 'base':
                data["prompt_base"] = prompt_prefix
                data["response_base"] = solution_raw
            elif template == 'qwen-instruct':
                data["prompt_qwen_instruct"] = prompt_prefix
                data["response_qwen_instruct"] = solution_raw
            elif template == 'llama-instruct':
                data["prompt_llama_instruct"] = prompt_prefix
                data["response_llama_instruct"] = solution_raw
        
        # Return None if no valid templates (all filtered out)
        if not valid_templates:
            return None
        
        # Set unified prompt field for consistency with other datasets
        if args.template_type != 'all':
            # For single template mode, use the template-specific prompt as string
            template_key = f"prompt_{args.template_type.replace('-', '_')}"
            if template_key in data:
                data["prompt"] = data[template_key]  # Use string directly, not dict format
        else:
            # For 'all' mode, set default prompt to llama_instruct for consistency
            if "prompt_llama_instruct" in data:
                data["prompt"] = data["prompt_llama_instruct"]
        
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
    # Use fixed 1024 samples for test, rest for train
    total_samples_after_filter = len(train_dataset)
    test_size = min(1024, total_samples_after_filter)  # At most 1025 samples for test
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
        copy(src=output_dir, dst=hdfs_dir)

if __name__ == '__main__':
    # Handle model_type all - generate separate datasets
    if args.model_type == 'all':
        model_types = ['base', 'qwen', 'llama']
        base_dirs = {
            'base': './data/base/deepmath',
            'qwen': './data/qwen_instruct/deepmath',
            'llama': './data/llama_instruct/deepmath'
        }
        template_mapping = {
            'base': 'base',
            'qwen': 'qwen-instruct', 
            'llama': 'llama-instruct'
        }
        
        print(f"[DeepMath] Generating datasets for all model types")
        
        for model_type in model_types:
            print(f"\n=== Generating {model_type.upper()} template data ===")
            model_output_dir = base_dirs[model_type]
            model_template_type = template_mapping[model_type]
            
            # Create a copy of args for this model type
            import copy
            model_args = copy.deepcopy(args)
            model_args.output_dir = model_output_dir
            model_args.template_type = model_template_type
            
            print(f"Output directory: {model_output_dir}")
            print(f"Template type: {model_template_type}")
            
            # Process this model type
            process_single_model_type(model_args, data_source)
        
        print(f"\nAll model types generated successfully!")
        exit(0)
    
    # Single model type processing
    process_single_model_type(args, data_source)
