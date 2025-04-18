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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.curriculum_sampler import CurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        
        if self.config.data.get('curriculum_learning', False):
            # Create separate datasets and dataloaders for each operator group
            self.train_datasets = {}
            self.train_dataloaders = {}
            
            # Extract groups from training file paths, preserving input order
            import os
            from collections import OrderedDict
            group_files = OrderedDict()
            ordered_groups = []
            for file_path in self.config.data.train_files:
                if 'train' not in file_path:
                    continue
                # Extract group name from path (assumes structure like path/to/group_name/train.parquet)
                group = os.path.basename(os.path.dirname(file_path))
                if group not in group_files:
                    group_files[group] = []
                    ordered_groups.append(group)
                group_files[group].append(file_path)
            
            # Store ordered groups for use in fit()
            self.ordered_groups = ordered_groups
            print(f"Initialized ordered_groups from training files: {self.ordered_groups}")
            
            # Create validation dataloaders first to ensure val_dataloaders is initialized
            self._create_validation_dataloaders()
            
            # Use groups in order from input files
            for group in ordered_groups:
                self.train_datasets[group] = RLHFDataset(
                    parquet_files=group_files[group],
                    tokenizer=self.tokenizer,
                    prompt_key=self.config.data.prompt_key,
                    max_prompt_length=self.config.data.max_prompt_length,
                    filter_prompts=True,
                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                    truncation='error',
                    config=self.config)
                # Create dataloader for this group
                self.train_dataloaders[group] = DataLoader(
                    dataset=self.train_datasets[group],
                    batch_size=self.config.data.train_batch_size,
                    shuffle=True,  
                    drop_last=True,
                    collate_fn=collate_fn)
            
            # For validation purposes, set train_dataset to last group's dataset
            last_group = self.ordered_groups[-1]
            self.train_dataset = self.train_datasets[last_group]
            # For length checks, use last group's dataloader
            self.train_dataloader = self.train_dataloaders[last_group]
            
            # Calculate total training steps after both train and validation dataloaders are created
            self._calculate_total_training_steps()
        else:
            # Regular training without curriculum
            self.train_dataset = RLHFDataset(
                parquet_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error',
                config=self.config)
            
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.data.train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn)
            
            # Calculate total training steps after both train and validation dataloaders are created
            self._calculate_total_training_steps()
    def _create_limited_dataloader(self, group, sample_size, epoch=None):
        """Create a new dataloader with a limited sample size from the original dataset
        
        Args:
            group: The group to create a limited dataloader for
            sample_size: The maximum number of samples to use
            epoch: If provided, use a deterministic seed based on the epoch number
                  to ensure different samples for each epoch without recreating the dataset
        """
        from torch.utils.data import DataLoader, Subset
        import random
        import numpy as np
        
        # Get the original dataset for this group
        dataset = self.train_datasets[group]
        
        # Determine the number of samples to use
        total_samples = len(dataset)
        samples_to_use = min(sample_size, total_samples)
        
        # If epoch is provided, use a deterministic seed based on the group and epoch
        if epoch is not None:
            # Create a deterministic seed based on the group and epoch
            # This ensures different samples for each epoch without recreating the dataset
            seed = hash(f"{group}_{epoch}") % 10000
            rng = random.Random(seed)
            indices = rng.sample(range(total_samples), samples_to_use)
            print(f"Using deterministic sampling for group {group}, epoch {epoch} with seed {seed}")
        else:
            # Use standard random sampling
            indices = random.sample(range(total_samples), samples_to_use)
        
        # Create a subset dataset with the sampled indices
        subset_dataset = Subset(dataset, indices)
        
        # Create and return a new dataloader with the subset
        return DataLoader(
            dataset=subset_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn)
            
    def _create_validation_dataloaders(self):
        """Create validation datasets and dataloaders for each group"""
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        import os
        from collections import OrderedDict
        
        self.val_datasets = {}
        self.val_dataloaders = {}
        val_group_files = OrderedDict()
        
        # Extract groups from validation file paths
        for file_path in self.config.data.val_files:
            if 'test' not in file_path:
                continue
            # Extract group name from path (assumes structure like path/to/group_name/test.parquet)
            group = os.path.basename(os.path.dirname(file_path))
            if group not in val_group_files:
                val_group_files[group] = []
            val_group_files[group].append(file_path)
        
        # Initialize ordered_groups if it doesn't exist yet
        if not hasattr(self, 'ordered_groups'):
            self.ordered_groups = list(val_group_files.keys())
            print(f"Initializing ordered_groups from validation files: {self.ordered_groups}")
            
        # Create validation datasets and loaders using groups from validation files
        for group in val_group_files.keys():
            self.val_datasets[group] = RLHFDataset(
                parquet_files=val_group_files[group],
                tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error',
                config=self.config)
            self.val_dataloaders[group] = DataLoader(
                dataset=self.val_datasets[group],
                batch_size=self.config.data.val_batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       config=self.config,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        # Only check the val_dataloader length since train_dataloader may not exist yet
        assert len(self.val_dataloader) >= 1
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

    def _calculate_total_training_steps(self):
        """Calculate total training steps and inject into config
        
        For curriculum learning, this only sets a placeholder value that will be updated
        by _reset_learning_rates for each group. For regular training, this calculates
        the actual total_training_steps.
        """
        # For curriculum learning, just set a placeholder - real calculation happens in _reset_learning_rates
        if self.config.data.get('curriculum_learning', False):
            # Just set a placeholder value that will be updated by _reset_learning_rates for each group
            total_training_steps = 1  # Placeholder, will be updated per group in _reset_learning_rates
            print(f'Curriculum learning: Setting placeholder total_training_steps that will be updated per group')
        else:
            # Regular training without curriculum - calculate the actual value
            if hasattr(self, 'train_dataloader'):
                total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
                print(f'Regular training: Calculated total_training_steps = {total_training_steps}')
            else:
                print("Warning: Cannot calculate total_training_steps because train_dataloader is not initialized yet.")
                # Defer calculation until dataloaders are available
                print("Deferring total_training_steps calculation until dataloaders are available")
                return

        # No need to override with config value as it doesn't exist

        # Store the value
        self.total_training_steps = total_training_steps

        # Update the optimizer configs directly - this is hacky but necessary
        from omegaconf import OmegaConf, open_dict
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if hasattr(self.config, 'critic') and hasattr(self.config.critic, 'optim'):
                self.config.critic.optim.total_training_steps = total_training_steps
            
    def _analyze_generation_probabilities(self, input_batch, output_batch, group, round_num=None, epoch=None, global_step=None):
        """Analyze token probabilities from the actual generated output
        
        Args:
            input_batch: The input batch with prompts
            output_batch: The output batch with generated sequences
            group: The current group being validated
            round_num: The current round number (optional)
            epoch: The current epoch number (optional)
            global_step: The current global step (optional)
        """
        # Only run this analysis on the main process (rank 0) to avoid duplicate outputs
        if hasattr(self, 'rank') and self.rank != 0:
            return
            
        try:
            import math
            import os
            
            # Get current round, epoch, and step information if not provided
            current_round = round_num if round_num is not None else getattr(self, 'current_round', 0)
            current_epoch = epoch if epoch is not None else getattr(self, 'current_epoch', 0)
            current_step = global_step if global_step is not None else getattr(self, 'global_steps', 0)
            
            # Get the input prompt from the non_tensor_batch
            if hasattr(input_batch, 'non_tensor_batch') and 'target' in input_batch.non_tensor_batch:
                target = input_batch.non_tensor_batch['target'][0] if len(input_batch.non_tensor_batch['target']) > 0 else "[No target available]"
            else:
                target = "[No target available]"
                
            if hasattr(input_batch, 'non_tensor_batch') and 'nums' in input_batch.non_tensor_batch:
                nums = input_batch.non_tensor_batch['nums'][0] if len(input_batch.non_tensor_batch['nums']) > 0 else "[No nums available]"
            else:
                nums = "[No nums available]"
            
            # Get the input_ids and generated sequences
            if not hasattr(output_batch, 'batch'):
                print("Output batch does not have the expected batch attribute.")
                return
                
            if 'input_ids' not in output_batch.batch:
                print("Could not find 'input_ids' in the output batch.")
                return
                
            # Get the full sequences (input + output)
            sequences = output_batch.batch['input_ids'][0]
            
            # Get the original input_ids to determine where the generated part starts
            prompt_length = None
            
            # Try several methods to determine the prompt length
            if hasattr(input_batch, 'batch') and 'input_ids' in input_batch.batch:
                # Method 1: Get from input batch directly
                input_ids = input_batch.batch['input_ids'][0]
                prompt_length = input_ids.shape[0]
            elif hasattr(output_batch, 'batch') and 'prompts' in output_batch.batch:
                # Method 2: Get from prompts in output batch
                prompt_length = output_batch.batch['prompts'].shape[1]
            elif hasattr(input_batch, 'non_tensor_batch') and 'prompt' in input_batch.non_tensor_batch:
                # Method 3: Try to get from non-tensor batch
                prompt_text = input_batch.non_tensor_batch['prompt'][0]
                tokenized = self.tokenizer(prompt_text, return_tensors="pt")
                prompt_length = tokenized.input_ids.shape[1]
            else:
                # Method 4: Use a default prompt based on the problem
                prompt_text = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant:"
                print(prompt_text)
                tokenized = self.tokenizer(prompt_text, return_tensors="pt")
                prompt_length = tokenized.input_ids.shape[1]
                
            if prompt_length is None:
                print("Could not determine prompt length using any method.")
                return
            
            # Check if we have token probabilities from vLLM
            token_log_probs = None
            
            # First check if they're in the batch
            if 'token_log_probs' in output_batch.batch:
                token_log_probs = output_batch.batch['token_log_probs'][0]
            # If not, check if they're stored as an instance variable in the rollout worker
            elif 'has_token_probs' in output_batch.batch and output_batch.batch['has_token_probs'].any():
                try:
                    # Try to access the token log probabilities from the rollout worker
                    token_log_probs = self.actor_rollout_wg.rollout._token_log_probs[0]
                except (AttributeError, IndexError) as e:
                    print(f"Could not access token log probabilities from rollout worker: {e}")
                    
            if token_log_probs is not None:
                
                # Get the generated part (exclude the prompt)
                generated_only = sequences[prompt_length:]
                
                # Get the full text for display
                full_text = self.tokenizer.decode(sequences, skip_special_tokens=True)
                generated_text = self.tokenizer.decode(generated_only, skip_special_tokens=True)
                
                # Print header with round, group, epoch, and step information
                print(f"\n{'='*80}")
                print(f"CASE STUDY - TOKEN ANALYSIS - Round: {current_round}, Group: {group}, Epoch: {current_epoch}, Step: {current_step}")
                print(f"{'='*80}")
                print(f"Target: {target}")
                print(f"Numbers: {nums}")
                print(f"Full text: {full_text}")
                print(f"Generated text: {generated_text}")
                print("\nToken-by-token probability analysis:")
                
                # Get the maximum number of tokens to analyze (limit to 1024 for efficiency)
                max_tokens_to_analyze = min(len(generated_only), 1024)
                
                # For each token, print its probability along with round, group, epoch info
                for i in range(max_tokens_to_analyze):
                    if i >= len(token_log_probs):
                        break
                        
                    token_id = generated_only[i].item()
                    token_text = self.tokenizer.decode([token_id])
                    # Replace any line breaks with spaces for cleaner output
                    token_text = token_text.replace('\n', ' ').replace('\r', ' ')
                    log_prob = token_log_probs[i].item()
                    
                    # Safely convert log probability to probability with error handling
                    try:
                        # For very negative log probs, math.exp can cause overflow
                        if log_prob < -100:
                            prob = 0.0  # Effectively zero probability
                        else:
                            prob = math.exp(log_prob)
                    except (OverflowError, ValueError):
                        # Handle math range error
                        prob = 0.0
                    
                    # Include round, group, epoch in the output for easier data extraction
                    print(f"Round: {current_round}, Group: {group}, Epoch: {current_epoch}, Step: {current_step}, Token {i+1}: '{token_text}' - Probability: {prob:.4f} (log prob: {log_prob:.4f})")
            else:
                # We don't have token probabilities from vLLM
                print("Token probabilities not available from vLLM output.")
                print("Make sure 'validate' is set to True in the meta_info.")
                
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error in token probability analysis: {e}")
            import traceback
            traceback.print_exc()
    

    
    def _validate_small(self, group=None):
        """Quick validation using a single batch from the dataloader."""
        if not group:
            return {}
            
        # Check if val_dataloaders exists and contains the group
        if not hasattr(self, 'val_dataloaders') or group not in self.val_dataloaders:
            print(f"Warning: No validation dataloader found for group {group}")
            return {}
            
        # Get the next batch from the validation dataloader
        if not hasattr(self, '_val_iterators'):
            self._val_iterators = {}
        if group not in self._val_iterators:
            self._val_iterators[group] = iter(self.val_dataloaders[group])
            
        try:
            test_data = next(self._val_iterators[group])
        except StopIteration:
            # Reset iterator when we've gone through all batches
            self._val_iterators[group] = iter(self.val_dataloaders[group])
            test_data = next(self._val_iterators[group])
        
        test_batch = DataProto.from_single_dict(test_data)
        
        if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
            return {}

        test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
        test_gen_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': True,  # Enable log probability computation for token analysis
            'do_sample': False,
            'validate': True,
            'return_dict_in_generate': True,  # Return detailed generation info
            'output_scores': True,  # Return scores for token probability analysis
        }

        # Generate and evaluate
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
        test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        
        # We'll do token probability analysis in the full _validate method instead
        
        test_batch = test_batch.union(test_output_gen_batch)
        
        # Get rewards
        reward_tensor = self.val_reward_fn(test_batch)
        data_source = test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
        
        # Calculate metrics
        reward_mean = reward_tensor.sum(-1).mean().item()
        return {f'val/test_score_quick/{data_source[0]}': reward_mean}
        
    def _validate(self, group=None):
        """Full validation on all test data. Used less frequently for thorough evaluation."""
        reward_tensor_lst = []
        data_source_lst = []
        
        # Only validate on specified group
        if not group:
            return {}
            
        # Check if val_dataloaders exists and contains the group
        if not hasattr(self, 'val_dataloaders') or group not in self.val_dataloaders:
            print(f"Warning: No validation dataloader found for group {group}")
            return {}
            
        # Run validation on specified group
        cnt = 1
        for test_data in self.val_dataloaders[group]:
                print('test_data_tmp:',cnt)
                cnt += 1
                test_batch = DataProto.from_single_dict(test_data)
                # test_batch = test_batch.to('cuda')

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': True,  # Enable log probability computation for token analysis
                    'do_sample': False,
                    'validate': True,
                    'return_dict_in_generate': True,  # Return detailed generation info
                    'output_scores': True,  # Return scores for token probability analysis
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                # Case study: Analyze token probabilities for one sample (only for the first batch)
                if cnt == 2:  # This is the first batch (cnt starts at 1 and is incremented before processing)
                    # Use the actual generated output for token probability analysis
                    # Pass current round, epoch, and global step for easier plotting
                    current_round = getattr(self, 'current_round', 0)
                    current_epoch = getattr(self, 'current_epoch', 0)
                    self._analyze_generation_probabilities(test_batch, test_output_gen_batch, group, 
                                                         round_num=current_round, 
                                                         epoch=current_epoch, 
                                                         global_step=self.global_steps)
                
                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            # The methods are already bound without prefix by spawn()
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _reset_learning_rates(self, group=None, current_dataloader=None, steps_per_epoch=None, epochs=None):
        """Reset learning rates of actor and critic to their default values
        
        Args:
            group: The group to reset learning rates for
            current_dataloader: If provided, calculate steps based on this dataloader
            steps_per_epoch: If provided, use this value directly for steps per epoch
            epochs: If provided, use this value for number of epochs, otherwise use config value
        """
        # If using curriculum learning and a group is specified, update total_training_steps for this group
        if self.config.data.get('curriculum_learning', False) and group is not None:
            # Determine how to calculate total steps
            if steps_per_epoch is not None:
                # Use the provided steps_per_epoch directly
                epochs_to_use = epochs if epochs is not None else self.config.data.epochs_per_group
                steps_for_group = steps_per_epoch * epochs_to_use
                print(f'Using provided steps_per_epoch: {steps_per_epoch} for {epochs_to_use} epochs')
            elif current_dataloader is not None:
                # Calculate based on dataloader length
                dataloader_to_use = current_dataloader
                steps_for_group = len(dataloader_to_use) * self.config.data.epochs_per_group
                print(f'Calculating from dataloader: {len(dataloader_to_use)} steps per epoch')
            elif hasattr(self, 'train_dataloaders') and group in self.train_dataloaders:
                # Use the default dataloader for this group
                dataloader_to_use = self.train_dataloaders[group]
                steps_for_group = len(dataloader_to_use) * self.config.data.epochs_per_group
                print(f'Using default dataloader: {len(dataloader_to_use)} steps per epoch')
            else:
                print(f'Warning: No dataloader or steps_per_epoch available for group {group}')
                return
                
            print(f'Updating total_training_steps for group {group}: {steps_for_group} steps')
            
            # Update the optimizer configs directly - this is hacky but necessary
            from omegaconf import OmegaConf, open_dict
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = steps_for_group
                if hasattr(self.config, 'critic') and hasattr(self.config.critic, 'optim'):
                    self.config.critic.optim.total_training_steps = steps_for_group
            
            # Store the updated value
            self.total_training_steps = steps_for_group
        
        # Reset critic learning rate if using critic
        if self.use_critic:
            self.critic_wg.critic_reset_optimizer_learning_rate()
            
        # Reset actor learning rate
        self.actor_rollout_wg.reset_optimizer_learning_rate()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        #if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
        #    # Check if ordered_groups is initialized
        #    if hasattr(self, 'ordered_groups') and self.ordered_groups:
        #        val_metrics = self._validate(self.ordered_groups[0]) # for debug
        #        pprint(f'Initial validation metrics: {val_metrics}')
        #        logger.log(data=val_metrics, step=self.global_steps)
        #        if self.config.trainer.get('val_only', False):
        #            return
        #    else:
        #        print("Warning: Cannot perform validation before training because ordered_groups is not initialized yet.")

        # we start from step 1
        self.global_steps += 1

        if self.config.data.get('curriculum_learning', False):
            # Debug: Print the train_sample_size parameter
            print("\n" + "="*50)
            print("DEBUG: Checking train_sample_size parameter")
            print(f"Has train_sample_size attribute: {hasattr(self.config.data, 'train_sample_size')}")
            if hasattr(self.config.data, 'train_sample_size'):
                print(f"train_sample_size value: {self.config.data.train_sample_size}")
                print(f"train_sample_size type: {type(self.config.data.train_sample_size)}")
            print("="*50 + "\n")
            
            # Get sample sizes for each group if provided
            print("\n" + "="*50)
            print("DEBUG: Checking train_sample_size parameter")
            print(f"train_sample_size: {self.config.data.train_sample_size}")
            print(f"train_sample_size type: {type(self.config.data.train_sample_size)}")
            
            # Initialize sample_sizes dictionary
            sample_sizes = {}
            
            # Convert train_sample_size to list if it's a string
            if isinstance(self.config.data.train_sample_size, str):
                try:
                    train_sample_size = eval(self.config.data.train_sample_size)
                    print(f"Converted train_sample_size to: {train_sample_size}")
                except Exception as e:
                    print(f"Error converting train_sample_size: {e}")
            
            # Handle ListConfig or list
            if hasattr(self.config.data.train_sample_size, '__iter__') and not isinstance(self.config.data.train_sample_size, str):
                print(f"Processing iterable train_sample_size: {self.config.data.train_sample_size}")
                print(f"Ordered groups: {self.ordered_groups}")
                
                for i, group in enumerate(self.ordered_groups):
                    if i < len(self.config.data.train_sample_size):
                        sample_size = self.config.data.train_sample_size[i]
                        if sample_size > 0:
                            # Use the group directly as the key (don't convert to string)
                            # This ensures we use the same key type when accessing the dictionary later
                            sample_sizes[group] = sample_size
                            print(f"Set sample_size for group {group} to {sample_size}")
            
            print(f"Final sample_sizes dictionary: {sample_sizes}")
            print("="*50 + "\n")
        
            for round_num in range(self.config.data.total_rounds):
                # Store the current round as an instance variable for access in validation
                self.current_round = round_num
                
                # Use groups in order from input files
                for group in self.ordered_groups:
                    print(f'Starting training on group: {group}')
                    
                    # First, get the sample size for this group if specified
                    sample_size = sample_sizes.get(group, 0)
                    print("sample_size:", sample_size)
                    print("sample_sizes:", sample_sizes)
                    print("group:", group, "type:", type(group))
                    
                    # Calculate the number of steps per epoch based on sample size and batch size
                    if sample_size > 0:
                        batch_size = self.config.data.train_batch_size
                        steps_per_epoch = max(1, sample_size // batch_size)
                        print(f"Using original dataloader but limiting to {steps_per_epoch} steps per epoch")
                        print(f"Sample size: {sample_size}, Batch size: {batch_size}")
                        
                        # Reset learning rates by directly passing the steps_per_epoch
                        # This is much more efficient than creating a dummy dataloader
                        self._reset_learning_rates(group=group, steps_per_epoch=steps_per_epoch)
                    else:
                        # Use the full dataloader if no sample size is specified
                        steps_per_epoch = len(self.train_dataloaders[group])
                        self._reset_learning_rates(group=group, current_dataloader=self.train_dataloaders[group])
                    
                    # Always use the original dataloader - no need to create limited ones
                    current_dataloader = self.train_dataloaders[group]
                    dataloader_iter = iter(current_dataloader)
                    
                    # Track total steps for this group
                    total_steps_for_group = 0
                    
                    for epoch in range(self.config.data.epochs_per_group):
                        # Store the current epoch as an instance variable for access in validation
                        self.current_epoch = epoch
                        print(f"Starting epoch {epoch} for group {group} - will run {steps_per_epoch} steps")
                        
                        # Run only the specified number of steps for this epoch
                        for step in range(steps_per_epoch):
                            try:
                                # Try to get the next batch
                                batch_dict = next(dataloader_iter)
                            except StopIteration:
                                # If we've reached the end of the dataloader, create a new iterator
                                print(f"Reached end of dataloader, creating new iterator for group {group}")
                                dataloader_iter = iter(current_dataloader)
                                batch_dict = next(dataloader_iter)
                            
                            # Increment step counter
                            total_steps_for_group += 1
                            
                            # Process the batch

                            
                            print(f'Intotal we have Round num: {self.config.data.total_rounds}, Group num {len(self.ordered_groups)}, Epoch num {self.config.data.epochs_per_group}, Step num {steps_per_epoch}')
                            print(f'Round {round_num}, Group {group}, Epoch {epoch}, Step {self.global_steps}')
                            metrics = {}
                            timing_raw = {}
                            
                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                            
                            # pop those keys for generation
                            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                            with _timer('step', timing_raw):
                                # generate a batch
                                with _timer('gen', timing_raw):
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                         dtype=object)
                                # repeat to align with repeated responses in rollout
                                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                batch = batch.union(gen_batch_output)

                                # balance the number of valid tokens on each dp rank.
                                # Note that this breaks the order of data inside the batch.
                                # Please take care when you implement group based adv computation such as GRPO and rloo
                                self._balance_batch(batch, metrics=metrics)

                                # compute global_valid tokens
                                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                            if self.use_reference_policy:
                                # compute reference log_prob
                                with _timer('ref', timing_raw):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                            # compute values
                            if self.use_critic:
                                with _timer('values', timing_raw):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)

                            with _timer('adv', timing_raw):
                                # compute scores. Support both model and function-based.
                                # We first compute the scores using reward model. Then, we call reward_fn to combine
                                # the results from reward model and rule-based results.
                                if self.use_rm:
                                    # we first compute reward model score
                                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                                    batch = batch.union(reward_tensor)

                                # we combine with rule-based rm
                                reward_tensor = self.reward_fn(batch)
                                batch.batch['token_level_scores'] = reward_tensor

                                # compute rewards. apply_kl_penalty if available
                                if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                    batch, kl_metrics = apply_kl_penalty(batch,
                                                                         kl_ctrl=self.kl_ctrl,
                                                                         kl_penalty=self.config.algorithm.kl_penalty)
                                    metrics.update(kl_metrics)
                                else:
                                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                                # compute advantages, executed on the driver process
                                batch = compute_advantage(batch,
                                                          adv_estimator=self.config.algorithm.adv_estimator,
                                                          gamma=self.config.algorithm.gamma,
                                                          lam=self.config.algorithm.lam,
                                                          num_repeat=self.config.actor_rollout_ref.rollout.n)

                            # update critic
                            if self.use_critic:
                                with _timer('update_critic', timing_raw):
                                    critic_output = self.critic_wg.update_critic(batch)
                                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                                metrics.update(critic_output_metrics)

                            # implement critic warmup
                            if self.config.trainer.critic_warmup <= self.global_steps:
                                # update actor
                                with _timer('update_actor', timing_raw):
                                    actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                                metrics.update(actor_output_metrics)

                            # Quick validation every step
                            #if self.val_reward_fn is not None:
                            #    with _timer('group_small_testing', timing_raw):
                            #        val_metrics: dict = self._validate_small(group=group)
                            #    metrics.update({f'{group}/test_small': val_metrics})
                                
                            # Full validation less frequently
                            if self.val_reward_fn is not None and self.global_steps % self.config.trainer.test_freq == 0:
                                #with _timer('group_full_testing', timing_raw):
                                #    val_metrics: dict = self._validate(group=group)
                                
                                val_metrics: dict = self._validate(group=group)

                                pprint(f'Step validation metrics: {val_metrics}') # debug
                                metrics.update(val_metrics)

                            if self.config.trainer.save_freq > 0 and \
                                    self.global_steps % self.config.trainer.save_freq == 0:
                                with _timer('save_checkpoint', timing_raw):
                                    self._save_checkpoint()

                            # collect metrics
                            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                            ## Run validation after each step
                            #if self.val_reward_fn is not None:
                            #    # Only validate current group to save memory
                            #    val_metrics = self._validate(group=group)
                            #    metrics.update({f'{group}/test': val_metrics})

                            # Log all metrics
                            logger.log(data=metrics, step=self.global_steps)
                            self.global_steps += 1

                            # update kl control
                            if self.use_reference_policy and 'kl_mean' in metrics:
                                self.kl_ctrl.update(metrics['kl_mean'], n_steps=1)
        else:
            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    print(f'epoch {epoch}, step {self.global_steps}')
                    metrics = {}
                    timing_raw = {}
                    
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    # pop those keys for generation
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                    with _timer('step', timing_raw):
                        # generate a batch
                        with _timer('gen', timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                 dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
