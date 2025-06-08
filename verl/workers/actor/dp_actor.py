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

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.single_controller.base.decorator import register, Dispatch
import torch
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
        original_param_shapes: dict = None, # <<< Cascade: Added original_param_shapes
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.global_steps = 0  # Track global steps for redo/reset logic
        # self.fsdp_grad_metric_enabled = getattr(config, "fsdp_grad_metric_enabled", False)  # Uncomment for config-driven
        self.fsdp_grad_metric_enabled = True  # DEBUG: Hard-coded to True for debugging FSDP gradient metrics
        print("[DEBUG][Actor] Config keys at init:", list(config.keys()) if hasattr(config, 'keys') else type(config))
        print("[DEBUG][Actor] fsdp_grad_metric_enabled in config:", getattr(config, "fsdp_grad_metric_enabled", None))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.original_param_shapes = original_param_shapes # <<< Cascade: Store original_param_shapes
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        
        # Initialize ReDo-related attributes
        self._redo_step = 0
        self.redo_enabled = getattr(self.config, 'redo_enabled', False)
        self.redo_metric_freq = getattr(self.config, 'redo_metric_freq', 1)
        self.redo_reset_freq = getattr(self.config, 'redo_reset_freq', 1000)
        self.redo_mode = getattr(self.config, 'redo_mode', 'threshold')
        self.redo_tau = getattr(self.config, 'redo_tau', 0.3)
        print(f'[DEBUG][Actor] ReDo config: enabled={self.redo_enabled}, metric_freq={self.redo_metric_freq}, '
              f'reset_freq={self.redo_reset_freq}, mode={self.redo_mode}, tau={self.redo_tau}')

        # Store initial optimizer config
        self.optim_config = None
        if hasattr(self.config, 'optim'):
            self.optim_config = self.config.optim

        # Create learning rate scheduler
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
            # Print current learning rates
            print("Before reset - Learning rates:", [group['lr'] for group in self.actor_optimizer.param_groups])
            
            # Reset scheduler's internal state
            self.lr_scheduler.last_epoch = -1
            # Update learning rate
            self.lr_scheduler.step()
            
            # Print new learning rates
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
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

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
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
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
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
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

        # --- FSDP dormant neuron and zero grad space analysis/reset (do only ONCE after optimizer step, BEFORE zero_grad) ---
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from verl.utils.redo_utils.fsdp_flat_utils import analyze_all_fsdp_dormant_neurons, analyze_all_fsdp_zero_grad_space, fsdp_dormant_neuron_mask_and_reset
        import torch.distributed as dist
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        is_fsdp = isinstance(self.actor_module, FSDP)
        with torch.no_grad():
            if is_fsdp and getattr(self, 'redo_enabled', False):
                # Calculate metrics at the specified frequency
                zero_gradspace_ratio = 0.0
                dormant_stats = None
                zero_grad_stats = None
                if self.global_steps % self.redo_metric_freq == 0:
                    # Ensure imports are at the top of the file or within the class if not already.
                    # from verl.utils.redo_utils.fsdp_flat_utils import analyze_all_fsdp_zero_grad_space, analyze_all_fsdp_dormant_neurons
                    
                    if rank == 0: print(f"[INFO][Actor][Step {self.global_steps}] Analyzing FSDP gradient metrics...")
                    zero_grad_stats = analyze_all_fsdp_zero_grad_space(self.actor_module, tau=self.redo_tau, verbose=(rank==0), original_shapes_map=self.original_param_shapes, skip_mlp=True) # <<< Cascade: Pass original_param_shapes and skip MLP layers
                    #dormant_stats = analyze_all_fsdp_dormant_neurons(self.actor_module, mode=self.redo_mode, tau=self.redo_tau, verbose=(rank==0))
                    
                    # if dormant_stats and '__global__' in dormant_stats: # analyze_all_fsdp_dormant_neurons also returns a __global__ key
                    #     total_dormant = dormant_stats['__global__'].get('dormant', 0)
                    #     total_neurons = dormant_stats['__global__'].get('total_neurons', 0)
                    #     if rank == 0 and total_neurons > 0:
                    #         print(f"[INFO][Actor][Step {self.global_steps}] Dormant Neuron Stats (Global): {total_dormant}/{total_neurons} ({total_dormant/total_neurons:.2%})")
                #     if rank == 0:
                #         if mask is not None:
                #             print(f"[FSDP-ReDo][Actor][Boot] Step {self.global_steps}: reset {mask.sum().item()} dormant neurons.")
                #             from verl.utils.redo_utils.fsdp_flat_utils import map_dormant_neurons_to_layers
                #             # Get both dormant neuron locations and statistics
                #             dormant_info, stats = map_dormant_neurons_to_layers(self.actor_module, mask, return_stats=True)
                #             # Removed: dormant_info, stats = None, None # hacky
                #             print(f"[DormantNeuron][Boot][Step {self.global_steps}] Locations (sample): {str(dormant_info[:10]) if dormant_info else 'N/A'}...")
                #             # Stats are expected to be printed by map_dormant_neurons_to_layers itself
                #         else:
                #             print(f"[FSDP-ReDo][Actor][Boot] Step {self.global_steps}: reset None.")
                # Perform neuron reset at the specified frequency after boot period
                # elif self.global_steps % self.redo_reset_freq == 0 and self.global_steps > 0:
                #     mask = fsdp_dormant_neuron_mask_and_reset(self.actor_module, mode=self.redo_mode, tau=self.redo_tau, optimizer=self.actor_optimizer)
                #     if rank == 0:
                #         if mask is not None:
                #             print(f"[FSDP-ReDo][Actor] Step {self.global_steps}: Performed neuron reset, {mask.sum().item()} dormant neurons reset.")
                #             # Optionally, you can also print detailed stats here if needed, similar to the boot phase
                #             # from verl.utils.redo_utils.fsdp_flat_utils import map_dormant_neurons_to_layers
                #             # dormant_info, stats = map_dormant_neurons_to_layers(self.actor_module, mask, return_stats=True)
                #             # print(f"[DormantNeuron][Regular][Step {self.global_steps}] Locations (sample): {str(dormant_info[:10]) if dormant_info else 'N/A'}...")
                #         else:
                #             print(f"[FSDP-ReDo][Actor] Step {self.global_steps}: fsdp_dormant_neuron_mask_and_reset returned None, no reset performed.")

                # Use the aggregated zero gradient ratio from fsdp_flat_utils.py
                # This ensures consistency with the fixed method in fsdp_flat_utils.py
                if zero_grad_stats and '__global__' in zero_grad_stats:
                    # Use the ratio directly from the stats if available
                    if 'aggregated_ratio' in zero_grad_stats['__global__']:
                        # Use the aggregated ratio (from per-layer aggregation) as source of truth
                        zero_gradspace_ratio_avg = zero_grad_stats['__global__']['aggregated_ratio']
                        if rank == 0:
                            print(f"[DEBUG] Using aggregated_ratio from zero_grad_stats: {zero_gradspace_ratio_avg:.4f}")
                    else:
                        # Fall back to the ratio in __global__ if aggregated_ratio is not available
                        # (This should not happen with the updated fsdp_flat_utils.py)
                        zero_gradspace_ratio_avg = zero_grad_stats['__global__']['ratio']
                        if rank == 0:
                            print(f"[DEBUG] Using ratio from zero_grad_stats: {zero_gradspace_ratio_avg:.4f} (aggregated_ratio not found)")
                else:
                    # Fall back to the old method if __global__ is not available
                    local_zero = 0.0
                    local_total = 0.0
                    if zero_grad_stats:
                        for key, stats in zero_grad_stats.items():
                            if key != '__global__' and stats and 'zero' in stats and 'total' in stats:
                                local_zero += stats['zero']
                                local_total += stats['total']
                    
                    zero_grad_tensor = torch.tensor([local_zero, local_total], dtype=torch.float32, device=next(self.actor_module.parameters()).device)
                    dist.all_reduce(zero_grad_tensor, op=dist.ReduceOp.SUM)
                    global_zero_grad, global_total_grad = zero_grad_tensor.tolist()
                    zero_gradspace_ratio_avg = global_zero_grad / (global_total_grad + 1e-8) if global_total_grad > 0 else 0.0
                    if rank == 0:
                        print(f"[DEBUG] Calculated zero_gradspace_ratio_avg manually: {zero_gradspace_ratio_avg:.4f} (no __global__ stats found)")
                if rank == 0:
                    metrics['actor/zero_gradspace_ratio'] = zero_gradspace_ratio_avg
                    print(f"[ZeroGradV2-Metrics][After Optim Step][Step {self.global_steps}] Aggregated Zero Grad Space Ratio: {zero_gradspace_ratio_avg:.4f}")
        # --- END FSDP analysis/reset ---

        self.actor_optimizer.zero_grad()
        return metrics
