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

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


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

    args = parser.parse_args()

    num_few_shot = 5
    data_source = 'openai/gsm8k'

    # Load dataset: from local JSONL if requested; otherwise from HF Hub
    if args.from_local:
        data_dir = args.local_json_dir or os.path.join(os.path.expanduser(args.local_dir), args.source)

        # Prefer local parquet shards if present, else fall back to JSONL
        parquet_train = glob.glob(os.path.join(data_dir, 'train.parquet')) + \
                        glob.glob(os.path.join(data_dir, 'train-*.parquet'))
        parquet_test = glob.glob(os.path.join(data_dir, 'test.parquet')) + \
                       glob.glob(os.path.join(data_dir, 'test-*.parquet'))

        if len(parquet_train) > 0 and len(parquet_test) > 0:
            data_files = {
                'train': parquet_train,
                'test': parquet_test,
            }
            dataset = datasets.load_dataset('parquet', data_files=data_files)
        else:
            data_files = {
                'train': [
                    os.path.join(data_dir, 'train.jsonl'),
                    os.path.join(data_dir, 'train-*.jsonl'),
                ],
                'test': [
                    os.path.join(data_dir, 'test.jsonl'),
                    os.path.join(data_dir, 'test-*.jsonl'),
                ],
            }
            dataset = datasets.load_dataset('json', data_files=data_files)
    else:
        dataset = datasets.load_dataset(data_source, args.source)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Instruction to guide the model's answer formatting
    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # Prepare CoT examples block if requested
    cot_examples = None
    if args.prepend_cot_examples:
        if args.cot_examples_file is not None:
            with open(args.cot_examples_file, 'r', encoding='utf-8') as f:
                cot_examples = f.read().strip()
        else:
            # Default 8-shot CoT block (as provided)
            #cot_examples = (
            #    "Below are 8 worked examples. Follow the same reasoning style and answer format.\n\n"
            #    "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
            #    "A: There been 21 are 15 15 trees originally. Then there were 21 trees after some more were planted. So there must have = 6. The answer is 6.\n#### 6\n"
            #    "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
            #    "A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n#### 5\n"
            #    "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
            #    "A: Originally, had 74 35 = Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they 39. The answer is 39.\n#### 39\n"
            #    "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
            #    "A: Jason started The answer is 8. with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 12 = 8.\n#### 8\n"
            #    "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
            #    "A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n#### 9\n"
            #    "Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n"
            #    "A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n#### 29\n"
            #    "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n"
            #    "A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 23 = 35. After losing 2 more, he had 35 2 = 33 golf balls. The answer is 33.\n#### 33\n"
            #    "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
            #    "A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 15 dollars left. 23 15 is 8. The answer is 8.\n#### 8\n"
            #)
            cot_examples = (
                "Below are 4 worked examples. Follow the same reasoning style and answer format.\n\n"
                "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
                "A: There been 21 are 15 15 trees originally. Then there were 21 trees after some more were planted. So there must have = 6. The answer is 6.\n#### 6\n"
                "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
                "A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n#### 5\n"
                "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
                "A: Originally, had 74 35 = Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they 39. The answer is 39.\n#### 39\n"
                "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
                "A: Jason started The answer is 8. with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 12 = 8.\n#### 8\n"
            )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

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

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt_body,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Preview a few processed samples (train split)
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

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
