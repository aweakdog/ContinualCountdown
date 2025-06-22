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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import ray

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean, entropy_from_logits
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.single_controller.base.decorator import register, Dispatch
import torch.nn as nn
import torch.nn.functional as F
from verl.utils.redo_utils.fsdp_flat_utils import analyze_all_fsdp_dormant_neurons, analyze_all_fsdp_zero_grad_space
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        original_param_shapes: dict = None, 
        grad_analyzer: "ray.actor.ActorHandle" = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.global_steps = 0  
        self.fsdp_grad_metric_enabled = True  
        print("[DEBUG][Actor] Config keys at init:", list(config.keys()) if hasattr(config, 'keys') else type(config))
        print("[DEBUG][Actor] fsdp_grad_metric_enabled in config:", getattr(config, "fsdp_grad_metric_enabled", None))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.original_param_shapes = original_param_shapes 
        self.grad_analyzer = grad_analyzer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        
        self._redo_step = 0
        self.redo_enabled = getattr(self.config, 'redo_enabled', False)
        self.redo_metric_freq = getattr(self.config, 'redo_metric_freq', 1)
        self.redo_reset_freq = getattr(self.config, 'redo_reset_freq', 1000)
        self.redo_mode = getattr(self.config, 'redo_mode', 'threshold')
        self.redo_tau = getattr(self.config, 'redo_tau', 0.3)
        print(f'[DEBUG][Actor] ReDo config: enabled={self.redo_enabled}, metric_freq={self.redo_metric_freq}, '
              f'reset_freq={self.redo_reset_freq}, mode={self.redo_mode}, tau={self.redo_tau}')

        self.optim_config = None
        if hasattr(self.config, 'optim'):
            self.optim_config = self.config.optim

        self.lr_scheduler = None
        if self.actor_optimizer is not None and self.optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            total_steps = self.optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = self.optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.actor_optimizer,
                num_warmup_steps=num_warmup_steps
            )

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer_learning_rate(self):
        """Reset learning rates to initial values while keeping optimizer state"""
        if self.actor_optimizer is not None and self.lr_scheduler is not None:
            print("Before reset - Learning rates:", [group['lr'] for group in self.actor_optimizer.param_groups])
            
            self.lr_scheduler.last_epoch = -1
            self.lr_scheduler.step()
            
            print("After reset - Learning rates:", [group['lr'] for group in self.actor_optimizer.param_groups])

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  

                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  

                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  
                logits_rmpad = output.logits.squeeze(0)  

                logits_rmpad.div_(temperature)

                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  

                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  

            else:  
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        """
        Update the policy with the given data. Accepts global_steps from trainer for correct frequency control.
        If global_steps is None, increments internal counter. Otherwise, uses the provided value.
        """
        if 'global_steps' in data.meta_info:
            self.global_steps = data.meta_info['global_steps']
        else:
            self.global_steps += 1
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_fsdp = isinstance(self.actor_module, FSDP)
        zero_grad_stats = None

        with torch.no_grad():
            if is_fsdp and self.grad_analyzer is not None and self.global_steps % self.redo_metric_freq == 0:
                if rank == 0: print(f"[INFO][Actor][Step {self.global_steps}] Analyzing gradients via remote analyzer...")
                
                grad_state_dict = {}
                with self.actor_module.summon_full_params(self.actor_module, writeback=False, rank0_only=False):
                    grad_state_dict = {
                        name: param.grad.cpu() for name, param in self.actor_module.named_parameters() if param.grad is not None
                    }
                
                if grad_state_dict:
                    analysis_future = self.grad_analyzer.analyze_gradients.remote(
                        component_name="actor",
                        grad_state_dict=grad_state_dict,
                        tau=self.redo_tau,
                        original_shapes_map=self.original_param_shapes
                    )
                    
                    try:
                        zero_grad_stats = ray.get(analysis_future, timeout=60)
                        if rank == 0 and zero_grad_stats:
                            print(f"[INFO][Actor][Step {self.global_steps}] Remote analysis complete. Stats: {zero_grad_stats.get('__global__')}")
                    except Exception as e:
                        if rank == 0: print(f"[ERROR] Failed to get gradient analysis results: {e}")
                elif rank == 0:
                    print("[WARN][Actor] No gradients found to analyze.")

            if zero_grad_stats and '__global__' in zero_grad_stats:
                zero_gradspace_ratio_avg = zero_grad_stats['__global__'].get('aggregated_ratio', 0.0)
                if rank == 0:
                    print(f"[INFO][Actor][Step {self.global_steps}] Aggregated Zero Grad Space Ratio from analyzer: {zero_gradspace_ratio_avg:.4f}")
            else:
                if rank == 0 and self.grad_analyzer is not None:
                    print(f"[DEBUG][Actor][Step {self.global_steps}] zero_grad_stats not available or does not contain '__global__' key. Stats: {zero_grad_stats}")
                zero_gradspace_ratio_avg = 0.0

            if rank == 0:
                metrics['actor/zero_gradspace_ratio'] = zero_gradspace_ratio_avg
                print(f"[ZeroGradV2-Metrics][After Optim Step][Step {self.global_steps}] Aggregated Zero Grad Space Ratio: {zero_gradspace_ratio_avg:.4f}")
        # --- END FSDP analysis/reset ---

        self.actor_optimizer.zero_grad()
        return metrics
