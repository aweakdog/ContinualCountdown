#!/usr/bin/env python3

import json
from typing import List, Dict, Any
from examples.data_preprocess.ifeval_scorer import IFEvalScorer

class IFEvalRewardModel:
    """Reward model adapter for IFeval constraints in RLHF training"""
    
    def __init__(self):
        self.scorer = IFEvalScorer()
    
    def get_reward(self, responses: List[str], prompts: List[str], ground_truths: List[str]) -> List[float]:
        """
        Calculate rewards for a batch of responses
        
        Args:
            responses: List of model responses
            prompts: List of input prompts (not used for IFeval)
            ground_truths: List of ground truth constraint specifications
            
        Returns:
            List of reward scores (0.0 to 1.0)
        """
        rewards = []
        for response, gt in zip(responses, ground_truths):
            try:
                reward = self.scorer.score(response, gt)
                rewards.append(reward)
            except Exception as e:
                print(f"Error calculating reward: {e}")
                rewards.append(0.0)
        
        return rewards
    
    def get_reward_single(self, response: str, prompt: str, ground_truth: str) -> float:
        """Calculate reward for a single response"""
        return self.scorer.score(response, ground_truth)


def test_reward_model():
    """Test the reward model with sample data"""
    reward_model = IFEvalRewardModel()
    
    # Test data matching your dataset format
    responses = [
        "this is a lowercase response without any capital letters",
        "This Response Has Capital Letters",
        "First paragraph * * * Second paragraph * * * Third paragraph",
        "Only two paragraphs * * * Second paragraph"
    ]
    
    ground_truths = [
        '{"func_name": "validate_lowercase", "N": null}',
        '{"func_name": "validate_lowercase", "N": null}',
        '{"func_name": "verify_paragraph_count", "N": 3}',
        '{"func_name": "verify_paragraph_count", "N": 3}'
    ]
    
    prompts = [""] * len(responses)  # Not used for IFeval
    
    rewards = reward_model.get_reward(responses, prompts, ground_truths)
    
    print("IFeval Reward Model Test Results:")
    for i, (response, gt, reward) in enumerate(zip(responses, ground_truths, rewards)):
        gt_data = json.loads(gt)
        constraint_type = gt_data['func_name']
        print(f"Sample {i+1}:")
        print(f"  Constraint: {constraint_type}")
        print(f"  Response: {response[:50]}...")
        print(f"  Reward: {reward}")
        print()


if __name__ == '__main__':
    test_reward_model()
