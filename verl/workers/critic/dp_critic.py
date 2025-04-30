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
Implement a multiprocess PPOCritic
"""
import itertools
from typing import Iterable

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.single_controller.base.decorator import register, Dispatch

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOCritic']


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.fsdp_grad_metric_enabled = getattr(config, "fsdp_grad_metric_enabled", False)
        print("[DEBUG][Critic] Config keys at init:", list(config.keys()) if hasattr(config, 'keys') else type(config))
        print("[DEBUG][Critic] fsdp_grad_metric_enabled in config:", getattr(config, "fsdp_grad_metric_enabled", None))
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Critic use_remove_padding={self.use_remove_padding}')

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        
        # Store initial optimizer config
        self.optim_config = None
        if hasattr(self.config, 'optim'):
            self.optim_config = self.config.optim

        # Create learning rate scheduler
        self.lr_scheduler = None
        if self.critic_optimizer is not None and self.optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            total_steps = self.optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = self.optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.critic_optimizer,
                num_warmup_steps=num_warmup_steps
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer_learning_rate(self):
        """Reset learning rates to initial values while keeping optimizer state"""
        if self.critic_optimizer is not None and self.lr_scheduler is not None:
            # Reset scheduler's internal state
            self.lr_scheduler.last_epoch = -1
            # Update learning rate
            self.lr_scheduler.step()

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1:-1]
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits
                values = values[:, -response_length - 1:-1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)
        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length - 1:-1]

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        return values

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.critic_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # critic device is cpu when using offload
                input_ids = data['input_ids']
                responses = data['responses']
                attention_mask = data['attention_mask']
                position_ids = data['position_ids']
                values = data['values']
                returns = data['returns']
                response_length = responses.size(1)

                eos_mask = attention_mask[:, -response_length - 1:-1]

                vpreds = self._forward_micro_batch(data)

                # assert not torch.any(torch.isnan(vpreds)).item()

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                loss = vf_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'critic/vf_loss': vf_loss.detach().item(),
                    'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                    'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                }

                append_to_dict(metrics, data)

            # FSDP gradient analysis after backward, before optimizer step
            print(f"[DEBUG][Critic] fsdp_grad_metric_enabled: {getattr(self, 'fsdp_grad_metric_enabled', False)}", flush=True)
            if getattr(self, 'fsdp_grad_metric_enabled', False):
                from verl.utils.redo_utils.fsdp_grad_gather import gather_full_grad
                import torch.distributed as dist
                print("[DEBUG][Critic] Checking parameter gradients before gather_full_grad...")
                for name, p in self.critic_module.named_parameters():
                    print(f"[DEBUG][Critic] {name}: grad is None? {p.grad is None}, shape: {p.grad.shape if p.grad is not None else 'N/A'}")
                full_grad = gather_full_grad(self.critic_module)
                if full_grad is not None:
                    print(f"[DEBUG][Critic] full_grad shape: {full_grad.shape}, numel: {full_grad.numel()}")
                    if dist.get_rank() == 0:
                        zero_grad_count = (full_grad == 0).sum().item()
                        total = full_grad.numel()
                        zero_grad_ratio = zero_grad_count / (total + 1e-8)
                        append_to_dict(metrics, {'critic/zero_grad_ratio': zero_grad_ratio})
                        print(f"[FSDP][Rank 0][Critic] Zero grad ratio: {zero_grad_ratio:.6f} ({zero_grad_count}/{total})")
                    nullspace_count = 0
                    total_params = 0
                    tau = getattr(self, 'critic_redo_tau', 0.0)
                    for p in self.critic_module.parameters():
                        if p.requires_grad:
                            total_params += 1
                            if p.grad is not None and p.grad.abs().max().item() < tau:
                                nullspace_count += 1
                    nullspace_count_tensor = torch.tensor([nullspace_count], device='cuda')
                    total_params_tensor = torch.tensor([total_params], device='cuda')
                    dist.all_reduce(nullspace_count_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_params_tensor, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0 and total_params_tensor.item() > 0:
                        nullspace_ratio = nullspace_count_tensor.item() / (total_params_tensor.item() + 1e-8)
                        append_to_dict(metrics, {'critic/nullspace_ratio': nullspace_ratio})
                        print(f"[FSDP][Rank 0][Critic] nullspace ratio: {nullspace_ratio:.6f} ({nullspace_count_tensor.item()}/{total_params_tensor.item()})")
                else:
                    print(f"[DEBUG][Critic][Rank {dist.get_rank()}] full_grad is None (not rank 0, expected in FSDP)")
                append_to_dict(metrics, {'critic/fsdp_grad_metric_ran': True})
            grad_norm = self._optimizer_step()
            data = {'critic/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics
