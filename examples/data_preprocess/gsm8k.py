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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import glob
import json
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


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
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it step by step.
User: {user_message}
Assistant: """
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant that solves math problems step by step.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    elif template_type == 'llama-instruct':
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that solves math problems step by step.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prefix


if __name__ == '__main__':
    '''
    python examples/data_preprocess/gsm8k.py \
  --from_local \
  --source main \
  --local_json_dir ./data/gsm8k/main \
  --local_dir ./data/gsm8k/0
    '''

    '''
    python examples/data_preprocess/gsm8k.py \
  --from_local \
  --source main \
  --local_json_dir ./data/gsm8k/main \
  --local_dir ./data/gsm8k/0 \
  --prepend_cot_examples \
  --preview_n 3
    '''
    '''
    python examples/data_preprocess/gsm8k.py \
  --from_local \
  --source main \
  --local_json_dir ./data/gsm8k/main \
  --local_dir ./data/gsm8k/1 \
  --prepend_cot_examples \
  --preview_n 3
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--source', default='main', choices=['main', 'socratic'])
    parser.add_argument('--from_local', action='store_true',
                        help='Load GSM8K from local JSONL files instead of the Hugging Face Hub')
    parser.add_argument('--local_json_dir', default=None,
                        help='Directory containing train-*.jsonl and test-*.jsonl. '
                             'Defaults to <local_dir>/<source> when --from_local is set')
    # CoT multi-shot prefix options
    parser.add_argument('--prepend_cot_examples', action='store_true',
                        help='Prepend a fixed 8-shot CoT Q/A block before each question')
    parser.add_argument('--cot_examples_file', default=None,
                        help='Path to a text file containing the CoT examples block. If not set, a built-in 8-shot block is used')
    # Preview options
    parser.add_argument('--preview_n', type=int, default=0,
                        help='Print N complete processed training samples to stdout for inspection')
    # Template support
    parser.add_argument('--template_type', default='all', choices=['base', 'qwen-instruct', 'llama-instruct', 'all'],
                        help='Template type to generate. "all" generates all templates in the same dataset')
    parser.add_argument('--model_type', default='single', choices=['base', 'qwen', 'llama', 'all', 'single'],
                        help='Model type for directory structure. "all" generates separate datasets for each template')
    parser.add_argument('--tokenizer_path', default=None,
                        help='Path to tokenizer for length filtering. If not set, uses default paths based on template')
    parser.add_argument('--max_prompt_length', type=int, default=800,
                        help='Maximum prompt length in tokens for filtering')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'openai/gsm8k'

def process_single_model_type(args, data_source):
    
    # Initialize tokenizer for length filtering
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

    # Load dataset: from local files if requested; otherwise from HF Hub
    if args.from_local:
        data_dir = args.local_json_dir
        
        # Check for parquet files first
        train_parquet = glob.glob(os.path.join(data_dir, 'train-*.parquet'))
        test_parquet = glob.glob(os.path.join(data_dir, 'test-*.parquet'))
        
        if train_parquet and test_parquet:
            print(f"Loading from local parquet files: {len(train_parquet)} train, {len(test_parquet)} test")
            dataset = datasets.load_dataset('parquet', data_files={
                'train': train_parquet,
                'test': test_parquet
            })
        else:
            # Fall back to JSONL files
            train_file = os.path.join(data_dir, 'train.jsonl')
            test_file = os.path.join(data_dir, 'test.jsonl')
            
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Neither parquet nor JSONL files found. Train file not found: {train_file}")
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test file not found: {test_file}")
            
            print(f"Loading from local JSONL files: {train_file}, {test_file}")
            dataset = datasets.load_dataset('json', data_files={'train': train_file, 'test': test_file})
    else:
        dataset = datasets.load_dataset(data_source, 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Sample dataset if max_samples is specified
    if args.max_samples and len(train_dataset) > args.max_samples:
        print(f"Sampling {args.max_samples} from {len(train_dataset)} total samples")
        train_dataset = train_dataset.select(range(args.max_samples))

    # Process the dataset
    def process_fn(example, idx):
        question_raw = example['question']
        answer_raw = example['answer']

        # Extract ground truth answer from answer field
        ground_truth = extract_solution(answer_raw)
        if not ground_truth:
            return None
        
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
                'split': 'train' if idx < len(train_dataset) else 'test',
                'index': idx,
                'answer': answer_raw,
                "question": question_raw,
            }
        }
        
        # Generate prompts for each template
        valid_templates = []
        instruction_following = "Let's think step by step and output the final answer after \"####\"."
        for template in templates_to_generate:
            prompt_prefix = make_prefix(question_raw, instruction_following, cot_examples=None, template_type=template)
            
            # Length filtering if tokenizer is available
            if tokenizer is not None:
                tokens = tokenizer.encode(prompt_prefix, add_special_tokens=False)
                if len(tokens) > args.max_prompt_length:
                    continue  # Skip this template if too long
            
            valid_templates.append(template)
            
            # Store prompt and response for this template
            if template == 'base':
                data["prompt_base"] = prompt_prefix
                data["response_base"] = answer_raw
            elif template == 'qwen-instruct':
                data["prompt_qwen_instruct"] = prompt_prefix
                data["response_qwen_instruct"] = answer_raw
            elif template == 'llama-instruct':
                data["prompt_llama_instruct"] = prompt_prefix
                data["response_llama_instruct"] = answer_raw
        
        # Return None if no valid templates (all filtered out)
        if not valid_templates:
            return None
        
        # Keep backward compatibility with single prompt field for non-all modes
        if args.template_type != 'all':
            if f"prompt_{args.template_type.replace('-', '_')}" in data:
                data["prompt"] = [{
                    "role": "user",
                    "content": data[f"prompt_{args.template_type.replace('-', '_')}"],
                }]
                data["response"] = data[f"response_{args.template_type.replace('-', '_')}"]
        
        return data

    # Get original dataset size for statistics
    original_train_size = len(train_dataset)
    original_test_size = len(test_dataset)
    
    train_dataset = train_dataset.map(function=process_fn, with_indices=True)
    test_dataset = test_dataset.map(function=process_fn, with_indices=True)
    
    # Filter out None values (filtered samples)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.filter(lambda x: x is not None)
    
    # Calculate filtering statistics
    retained_train_samples = len(train_dataset)
    retained_test_samples = len(test_dataset)
    filtered_train_samples = original_train_size - retained_train_samples
    filtered_test_samples = original_test_size - retained_test_samples
    
    print(f"\n===== Filtering Statistics =====")
    print(f"Train samples processed: {original_train_size}")
    print(f"Train samples filtered out (prompt > {args.max_prompt_length} tokens): {filtered_train_samples}")
    print(f"Train samples retained: {retained_train_samples}")
    print(f"Train filter rate: {filtered_train_samples/original_train_size*100:.2f}%")
    print(f"Test samples processed: {original_test_size}")
    print(f"Test samples filtered out (prompt > {args.max_prompt_length} tokens): {filtered_test_samples}")
    print(f"Test samples retained: {retained_test_samples}")
    print(f"Test filter rate: {filtered_test_samples/original_test_size*100:.2f}%")

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
    output_dir = args.local_dir
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
            'base': './data/base/gsm8k',
            'qwen': './data/qwen_instruct/gsm8k',
            'llama': './data/llama_instruct/gsm8k'
        }
        template_mapping = {
            'base': 'base',
            'qwen': 'qwen-instruct', 
            'llama': 'llama-instruct'
        }
        
        print(f"[GSM8K] Generating datasets for all model types")
        
        for model_type in model_types:
            print(f"\n=== Generating {model_type.upper()} template data ===")
            model_output_dir = base_dirs[model_type]
            model_template_type = template_mapping[model_type]
            
            # Create a copy of args for this model type
            import copy
            model_args = copy.deepcopy(args)
            model_args.local_dir = model_output_dir
            model_args.template_type = model_template_type
            
            print(f"Output directory: {model_output_dir}")
            print(f"Template type: {model_template_type}")
            
            # Process this model type
            process_single_model_type(model_args, data_source)
        
        print(f"\nAll model types generated successfully!")
        exit(0)
    
    # Single model type processing
    process_single_model_type(args, data_source)
