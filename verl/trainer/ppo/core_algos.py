# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_srpo_advantage(token_level_rewards: torch.Tensor,
                           eos_mask: torch.Tensor,
                           index: torch.Tensor,
                           old_log_prob: torch.Tensor,
                           ref_log_prob: torch.Tensor,
                           beta: float = 0.1,
                           epsilon: float = 1e-6,
                           baseline_type: str = 'mean',
                           debug: bool = False,
                           use_whitening: bool = True):
    """
    Compute advantage for SRPO (Step-wise Relative Policy Optimization).
    Combines outcome advantage (like GRPO) with step-wise process advantage (MSA).
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length) - rewards for each token
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length) - mask for valid tokens
        index: `(torch.Tensor)`
            shape: (bs,) - prompt group indices for grouping responses
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length) - log probabilities from current policy
        ref_log_prob: `(torch.Tensor)`
            shape: (bs, response_length) - log probabilities from reference policy
        beta: `(float)`
            coefficient for cumulative probability ratio computation
        epsilon: `(float)`
            small value to avoid division by zero
        baseline_type: `(str)`
            type of baseline for MSA computation ('mean', 'max', 'median')
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length) - combined advantages (outcome + process)
        returns: `(torch.Tensor)`
            shape: (bs, response_length) - same as advantages for compatibility
    """
    device = token_level_rewards.device
    bsz, response_length = token_level_rewards.shape
    
    # Step 1: Compute outcome advantages (same as GRPO)
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # (bs,)
    
    id2rows = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        # Build mapping from group id to row indices; support tensor or string/list indices
        for i in range(bsz):
            key = index[i].item() if torch.is_tensor(index) else index[i]
            id2rows[key].append(i)

        # Compute group statistics for outcome advantages
        for key, rows in id2rows.items():
            if len(rows) == 0:
                raise ValueError(f"No rows for prompt index: {key}")
            group_scores = scores[torch.tensor(rows, device=device, dtype=torch.long)]
            if len(rows) == 1:
                id2mean[key] = torch.tensor(0.0, device=device)
                id2std[key] = torch.tensor(1.0, device=device)
            else:
                id2mean[key] = torch.mean(group_scores)
                # Clamp std to avoid divide-by-zero when variance is zero
                std = torch.std(group_scores)
                id2std[key] = torch.clamp(std, min=epsilon)

        # Compute standardized outcome advantages
        outcome_advantages = torch.zeros_like(scores)
        for i in range(bsz):
            key = index[i].item() if torch.is_tensor(index) else index[i]
            outcome_advantages[i] = (scores[i] - id2mean[key]) / (id2std[key] + epsilon)
        
        # Step 2: Compute Cumulative Probability Ratios (CPR)
        log_ratio = beta * (old_log_prob - ref_log_prob)  # (bs, response_length)
        cpr = torch.cumsum(log_ratio * eos_mask, dim=-1)  # (bs, response_length)
        
        # Step 3: Compute Masked Step Advantages (MSA)
        msa = torch.zeros_like(cpr)
        
        # Group responses by prompt index for step-wise comparison using id2rows
        for key, rows in id2rows.items():
            group_indices = torch.tensor(rows, device=device, dtype=torch.long)
            
            if len(group_indices) <= 1:
                # Single response: MSA = 0 (no comparison possible)
                continue
            
            # Get CPR for this group across all time steps
            group_cpr = cpr[group_indices]  # (group_size, response_length)
            group_mask = eos_mask[group_indices]  # (group_size, response_length)
            
            # For each time step, compute MSA
            for t in range(response_length):
                # Find valid responses at time step t
                valid_mask = group_mask[:, t] > 0  # which responses are valid at step t
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) <= 1:
                    continue  # Need at least 2 responses for comparison
                
                # Get CPR values for valid responses at step t
                group_cpr_t = group_cpr[valid_indices, t]  # (n_valid,)
                
                # Compute baseline with better numerical stability
                if baseline_type == 'mean':
                    baseline_t = torch.mean(group_cpr_t)
                elif baseline_type == 'max':
                    baseline_t = torch.max(group_cpr_t)
                elif baseline_type == 'median':
                    baseline_t = torch.median(group_cpr_t)
                else:
                    baseline_t = torch.mean(group_cpr_t)  # default to mean
                
                # Compute MSA for valid responses at step t
                msa_t = group_cpr_t - baseline_t
                
                # Map back to original indices
                original_indices = group_indices[valid_indices]
                msa[original_indices, t] = msa_t
        
        # Step 4: Combine outcome and process advantages
        # Broadcast outcome advantages to all time steps
        outcome_adv_broadcast = outcome_advantages.unsqueeze(-1).expand(-1, response_length)  # (bs, response_length)
        
        if use_whitening:
            # Whitening strategy: outcome centering (like DrGRPO) + MSA full whitening
            # This allows 1:1 weighting without manual hyperparameter tuning
            print(f"SRPO-WHITENING: Applying DrGRPO-style centering for outcome + full whitening for MSA")
            
            # Outcome: Only subtract mean (DrGRPO style centering, no std normalization)
            outcome_per_token = outcome_adv_broadcast * eos_mask
            valid_outcome_mask = eos_mask > 0
            if valid_outcome_mask.sum() > 1:  # Need at least 2 valid tokens
                valid_outcome_values = outcome_per_token[valid_outcome_mask]
                outcome_mean = torch.mean(valid_outcome_values)
                outcome_whitened = outcome_per_token - outcome_mean  # Only subtract mean
                print(f"SRPO-WHITENING: Outcome centering - mean={outcome_mean:.4f}, valid_tokens={valid_outcome_mask.sum()}")
            else:
                outcome_whitened = outcome_per_token
                print(f"SRPO-WHITENING: Outcome - insufficient tokens for centering ({valid_outcome_mask.sum()})")
            
            # MSA: Full whitening (subtract mean + divide by std)
            valid_msa_mask = eos_mask > 0
            if valid_msa_mask.sum() > 1:  # Need at least 2 valid tokens
                valid_msa_values = msa[valid_msa_mask]
                msa_mean = torch.mean(valid_msa_values)
                msa_std = torch.std(valid_msa_values)
                msa_whitened = (msa - msa_mean) / (msa_std + epsilon)
                print(f"SRPO-WHITENING: MSA full whitening - mean={msa_mean:.4f}, std={msa_std:.4f}, valid_tokens={valid_msa_mask.sum()}")
            else:
                msa_whitened = msa
                print(f"SRPO-WHITENING: MSA - insufficient tokens for whitening ({valid_msa_mask.sum()})")
            
            # Combine with 1:1 weighting after processing
            total_advantages = (outcome_whitened + msa_whitened) * eos_mask
            print(f"SRPO-WHITENING: Combined advantages with 1:1 weighting (outcome centered + MSA whitened)")
        else:
            # Original combination without whitening
            total_advantages = (outcome_adv_broadcast + msa) * eos_mask
        
        if debug:
            # Expand outcome advantage to per-token for debugging/printing
            if use_whitening:
                # Include both original and whitened values for comparison
                outcome_per_token_orig = outcome_adv_broadcast * eos_mask
                debug_info = {
                    'msa': msa.detach(),
                    'msa_whitened': msa_whitened.detach() if use_whitening else msa.detach(),
                    'outcome_per_token': outcome_per_token_orig.detach(),
                    'outcome_per_token_whitened': outcome_whitened.detach() if use_whitening else outcome_per_token_orig.detach(),
                    'eos_mask': eos_mask.detach(),
                    'use_whitening': use_whitening,
                }
            else:
                outcome_per_token = outcome_adv_broadcast * eos_mask
                debug_info = {
                    'msa': msa.detach(),
                    'outcome_per_token': outcome_per_token.detach(),
                    'eos_mask': eos_mask.detach(),
                    'use_whitening': use_whitening,
                }
            return total_advantages, total_advantages, debug_info
    
    return total_advantages, total_advantages


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
