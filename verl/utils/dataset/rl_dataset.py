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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 config=None,
                 operator_group=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.operator_group = operator_group

        # Filter files by operator group if specified
        if operator_group:
            self.parquet_files = [f for f in self.parquet_files if operator_group in f]
            if not self.parquet_files:
                raise ValueError(f'No parquet files found for operator group: {operator_group}')

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.config = config

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            # Add source file as a column to track curriculum progression
            dataframe['_source_file'] = parquet_file
            dataframes.append(dataframe)

        # Combine all dataframes
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        # Get unique operator groups from file paths
        self.operator_groups = sorted(list(set(
            os.path.basename(os.path.dirname(f)) for f in self.parquet_files
        )))

        # Add group and round information
        self.dataframe['_group'] = self.dataframe['_source_file'].apply(
            lambda x: os.path.basename(os.path.dirname(x))
        )
        
        # Add epoch information if provided in config
        self.epochs_per_group = None
        self.total_rounds = None
        if hasattr(self, 'config') and hasattr(self.config, 'data'):
            self.epochs_per_group = self.config.data.get('epochs_per_group')
            self.total_rounds = self.config.data.get('total_rounds')
            
            if self.epochs_per_group and self.total_rounds:
                # Calculate total samples needed for each group
                samples_per_group = len(self.dataframe) // (len(self.operator_groups) * self.total_rounds)
                
                # Replicate data for each round and epoch
                new_dataframes = []
                for round_num in range(self.total_rounds):
                    for group in self.operator_groups:
                        group_data = self.dataframe[self.dataframe['_group'] == group]
                        for epoch in range(self.epochs_per_group):
                            epoch_data = group_data.copy()
                            epoch_data['_round'] = round_num
                            epoch_data['_epoch'] = epoch
                            new_dataframes.append(epoch_data)
                
                self.dataframe = pd.concat(new_dataframes, ignore_index=True)
            dataframe['_source_file'] = parquet_file
            dataframes.append(dataframe)
        # Reset index to maintain order of files
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        # Handle both string prompts and chat format
        #prompt_with_chat_template = chat[0]['content'] if isinstance(chat, list) else chat
        #prompt_with_chat_template = chat
        prompt_with_chat_template = chat[0]['content']

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
