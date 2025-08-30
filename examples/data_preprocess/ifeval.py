#!/usr/bin/env python3

import argparse
import json
import os
import re
import glob
import datasets
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import pandas as pd

def extract_constraint_info(ground_truth_str):
    """Extract constraint validation function from ground_truth JSON string"""
    try:
        gt_data = json.loads(ground_truth_str)
        # Return the full JSON string for the scorer to parse
        return ground_truth_str
    except:
        # Return a default JSON structure if parsing fails
        return '{"func_name": "unknown", "N": null}'


def make_prefix(user_message, template_type='base'):
    """Generate prompt with different template formats"""
    instruction_following = "Follow the given constraints carefully and provide a helpful response."
    prompt_body = f"{user_message}\n\n{instruction_following}"
    
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user gives instructions with constraints, and the Assistant follows them carefully.
User: {prompt_body}
Assistant: """
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant that follows instructions and constraints carefully.<|im_end|>
<|im_start|>user
{prompt_body}<|im_end|>
<|im_start|>assistant
"""
    elif template_type == 'llama-instruct':
        prefix = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that follows instructions and constraints carefully.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt_body}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prefix

def process_fn(example, idx, tokenizer, args, data_source, template_type='all'):
    """Process a single IFeval example into RLHF format"""
    
    # Extract the user message content
    messages = example['messages']
    if not messages or len(messages) == 0:
        return None
    
    user_message = messages[0]['content']
    constraint = example.get('constraint', '')
    constraint_type = example.get('constraint_type', '')
    
    # Generate prompts for all templates if template_type is 'all'
    templates_to_generate = ['base', 'qwen-instruct', 'llama-instruct'] if template_type == 'all' else [template_type]
    
    # Extract ground truth constraint validation function
    ground_truth = extract_constraint_info(example['ground_truth'])
    
    data = {
        "data_source": data_source,
        "ability": "instruction_following",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            'split': 'train',
            'index': idx,
            'constraint_type': constraint_type,
            'constraint': constraint,
            'dataset': example.get('dataset', 'ifeval'),
        }
    }
    
    # Generate prompts for each template
    valid_templates = []
    for template in templates_to_generate:
        prompt_prefix = make_prefix(user_message, template)
        
        # Length filtering if tokenizer is available
        if tokenizer is not None:
            tokens = tokenizer.encode(prompt_prefix, add_special_tokens=False)
            if len(tokens) > args.max_prompt_length:
                continue  # Skip this template if too long
        
        valid_templates.append(template)
        
        # Store prompt for this template
        if template == 'base':
            data["prompt_base"] = prompt_prefix
            data["response_base"] = ""  # IFEval doesn't have reference responses
        elif template == 'qwen-instruct':
            data["prompt_qwen_instruct"] = prompt_prefix
            data["response_qwen_instruct"] = ""
        elif template == 'llama-instruct':
            data["prompt_llama_instruct"] = prompt_prefix
            data["response_llama_instruct"] = ""
    
    # Return None if no valid templates (all filtered out)
    if not valid_templates:
        return None
    
    # Keep backward compatibility with single prompt field for non-all modes
    if template_type != 'all':
        if f"prompt_{template_type.replace('-', '_')}" in data:
            data["prompt"] = [{
                "role": "user",
                "content": data[f"prompt_{template_type.replace('-', '_')}"],
            }]
    
    return data


if __name__ == '__main__':
    '''
    python examples/data_preprocess/ifeval.py --from_local --local_dir ./data/RLVR-IFeval/data --max_prompt_length 1000 --output_dir ./data/ifeval/0
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/ifeval')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--from_local', action='store_true',
                        help='Load IFeval from local parquet files instead of the Hugging Face Hub')
    parser.add_argument('--local_parquet_dir', default=None,
                        help='Directory containing train-*.parquet files. '
                             'Defaults to data/RLVR-IFeval/data/ when --from_local is set')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for processed parquet files. If not set, saves to local_dir')
    
    # Preview options
    parser.add_argument('--preview_n', type=int, default=0,
                        help='Print N complete processed training samples to stdout for inspection')
    # Data sampling options
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (useful for creating smaller datasets)')
    parser.add_argument('--max_prompt_length', type=int, default=700,
                        help='Maximum prompt length in tokens. Samples exceeding this will be filtered out. Should be lower than model max_length to account for chat template overhead.')
    # Template support
    parser.add_argument('--template_type', default='all', choices=['base', 'qwen-instruct', 'llama-instruct', 'all'],
                        help='Template type to generate. "all" generates all templates in the same dataset')
    parser.add_argument('--model_type', default='single', choices=['base', 'qwen', 'llama', 'all', 'single'],
                        help='Model type for directory structure. "all" generates separate datasets for each template')
    parser.add_argument('--tokenizer_path', default=None,
                        help='Path to tokenizer for length filtering. If not set, uses default paths based on template')

    args = parser.parse_args()

    data_source = 'RLVR-IFeval'

def process_single_model_type(args, data_source):
    
    # Initialize tokenizer for token-based length filtering
    if args.tokenizer_path:
        tokenizer_path = args.tokenizer_path
    else:
        # Use default tokenizer paths
        if args.template_type == 'qwen-instruct' or args.template_type == 'all':
            tokenizer_path = "./models/qwen_instruct3b"
        elif args.template_type == 'llama-instruct':
            tokenizer_path = "./models/llama_instruct3b"
        else:
            tokenizer_path = "./models/qwen_instruct3b"  # default
    
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
        data_dir = args.local_parquet_dir or './data/RLVR-IFeval/data/'
        
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

    # Process the dataset
    def process_fn(example, idx):
        # Extract prompt from messages field
        messages = example['messages']
        if isinstance(messages, list) and len(messages) > 0:
            prompt_raw = messages[0]['content']
        else:
            prompt_raw = str(messages)
        
        # IFeval doesn't have response field - we'll generate empty response for training
        response_raw = ""
        
        # Extract constraint information
        constraint_type = example.get('constraint_type', '')
        constraint = example.get('constraint', '')
        ground_truth = example.get('ground_truth', '')

        # Generate prompts for all templates if template_type is 'all'
        templates_to_generate = ['base', 'qwen-instruct', 'llama-instruct'] if args.template_type == 'all' else [args.template_type]
        
        data = {
            "data_source": data_source,
            "ability": "instruction_following",
            "reward_model": {
                "style": "rule",
                "ground_truth": None  # IFEval uses rule-based evaluation
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'constraint_type': constraint_type,
                'constraint': constraint,
                'ground_truth': ground_truth,
                "prompt": prompt_raw,
                "response": response_raw,
            }
        }
        
        # Generate prompts for each template
        valid_templates = []
        for template in templates_to_generate:
            prompt_prefix = make_prefix(prompt_raw, template)
            
            # Length filtering if tokenizer is available
            if tokenizer is not None:
                tokens = tokenizer.encode(prompt_prefix, add_special_tokens=False)
                if len(tokens) > args.max_prompt_length:
                    continue  # Skip this template if too long
            
            valid_templates.append(template)
            
            # Store prompt and response for this template
            if template == 'base':
                data["prompt_base"] = prompt_prefix
                data["response_base"] = response_raw
            elif template == 'qwen-instruct':
                data["prompt_qwen_instruct"] = prompt_prefix
                data["response_qwen_instruct"] = response_raw
            elif template == 'llama-instruct':
                data["prompt_llama_instruct"] = prompt_prefix
                data["response_llama_instruct"] = response_raw
        
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

    # Create test dataset by splitting from train (IFEval doesn't have separate test set)
    # Use fixed 1024 samples for test, rest for train
    total_samples_after_filter = len(train_dataset)
    test_size = min(1024, total_samples_after_filter // 10)  # Use 10% for test, max 1024
    test_size = max(1, test_size)  # At least 1 sample for test
    train_size = total_samples_after_filter - test_size
    
    if train_size <= 0:
        # If dataset is too small, use all for train and duplicate first sample for test
        test_dataset = train_dataset.select([0])
        # Keep original train_dataset as is
    else:
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
            'base': './data/base/ifeval',
            'qwen': './data/qwen_instruct/ifeval',
            'llama': './data/llama_instruct/ifeval'
        }
        template_mapping = {
            'base': 'base',
            'qwen': 'qwen-instruct', 
            'llama': 'llama-instruct'
        }
        
        print(f"[IFEval] Generating datasets for all model types")
        
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
