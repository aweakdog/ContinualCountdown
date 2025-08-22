#!/usr/bin/env python3

import argparse
import json
import os
import re
import glob
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

def process_fn(example, idx, tokenizer, args, data_source):
    """Process a single IFeval example into RLHF format"""
    
    # Extract the user message content
    messages = example['messages']
    if not messages or len(messages) == 0:
        return None
    
    user_message = messages[0]['content']
    constraint = example.get('constraint', '')
    constraint_type = example.get('constraint_type', '')
    
    # Create instruction following prompt
    instruction_following = "Follow the given constraints carefully and provide a helpful response."
    
    # Combine user message with instruction
    prompt_body = f"{user_message}\n\n{instruction_following}"
    
    # Check prompt length using chat template (same as training) and filter if too long
    messages_for_template = [{"role": "user", "content": prompt_body}]
    prompt_with_chat_template = tokenizer.apply_chat_template(messages_for_template, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(prompt_with_chat_template, add_special_tokens=True)
    prompt_token_length = len(tokens)
    
    if prompt_token_length > args.max_prompt_length:
        return None

    # Extract ground truth constraint validation function
    ground_truth = extract_constraint_info(example['ground_truth'])
    
    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": prompt_body,
        }],
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
    parser.add_argument('--max_prompt_length', type=int, default=800,
                        help='Maximum prompt length in tokens. Samples exceeding this will be filtered out. Should be lower than model max_length to account for chat template overhead.')

    args = parser.parse_args()

    data_source = 'RLVR-IFeval'
    
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
        data_dir = args.local_parquet_dir or './data/RLVR-IFeval/data/'
        
        # Find all parquet training files
        parquet_files = glob.glob(os.path.join(data_dir, 'train-*.parquet'))
        
        if len(parquet_files) > 0:
            print(f"Found {len(parquet_files)} parquet files in {data_dir}")
            # Load all parquet files
            dfs = []
            for file in parquet_files:
                df = pd.read_parquet(file)
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            train_dataset = Dataset.from_pandas(combined_df)
        else:
            raise FileNotFoundError(f"No train-*.parquet files found in {data_dir}")
    else:
        # Load from Hugging Face Hub (if available)
        print("Loading from Hugging Face Hub not implemented yet. Use --from_local flag.")
        exit(1)

    original_size = len(train_dataset)
    print(f"Loaded {original_size} samples from IFeval dataset")

    # Sample if requested
    if args.max_samples is not None and args.max_samples < len(train_dataset):
        print(f"Sampling {args.max_samples} from {len(train_dataset)} total samples")
        train_dataset = train_dataset.select(range(args.max_samples))
    
    train_dataset = train_dataset.map(function=process_fn, with_indices=True, 
                                     fn_kwargs={'tokenizer': tokenizer, 'args': args, 'data_source': data_source})
    
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

    # Create test dataset by splitting from train
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
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)
        copy(src=output_dir, dst=hdfs_dir)
