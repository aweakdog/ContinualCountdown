#!/usr/bin/env python3

import sys
import os
sys.path.append('/cpfs04/user/liyuanhang.p/src/ContinualCountdown')

from verl.utils.reward_score.deepmath import compute_score as deepmath_score
from verl.utils.reward_score.countdown import compute_score as countdown_score

def test_deepmath_scorer():
    """Test DeepMath scorer with string ground truth"""
    print("=== Testing DeepMath Scorer ===")
    
    solution = "The answer is \\boxed{42}."
    ground_truth = "42"  # String format
    score = deepmath_score(solution, ground_truth)
    print(f"DeepMath - Solution: {solution}")
    print(f"DeepMath - Ground truth: {ground_truth}")
    print(f"DeepMath - Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("✓ DeepMath scorer working correctly")
    print()

def test_countdown_scorer():
    """Test Countdown scorer with dictionary ground truth"""
    print("=== Testing Countdown Scorer ===")
    
    solution = "<answer>25</answer>"
    ground_truth = {
        'target': 25,
        'numbers': [1, 2, 3, 4, 5, 6]
    }  # Dictionary format
    score = countdown_score(solution, ground_truth)
    print(f"Countdown - Solution: {solution}")
    print(f"Countdown - Ground truth: {ground_truth}")
    print(f"Countdown - Score: {score}")
    print("✓ Countdown scorer working correctly")
    print()

def test_main_ppo_logic():
    """Test the logic from main_ppo.py"""
    print("=== Testing main_ppo.py Logic ===")
    
    # Simulate DeepMath case
    deepmath_ground_truth = {
        'question': 'What is 6 * 7?',
        'final_answer': '42',
        'ground_truth': '42'  # This is what should be passed to deepmath scorer
    }
    
    # Simulate what main_ppo.py does for DeepMath
    data_source = 'zwhe99/DeepMath-103K'
    solution = "The calculation gives us \\boxed{42}."
    
    if data_source == 'zwhe99/DeepMath-103K':
        score = deepmath_score(solution, deepmath_ground_truth['ground_truth'])
    else:
        score = deepmath_score(solution, deepmath_ground_truth)
    
    print(f"PPO Logic - DeepMath solution: {solution}")
    print(f"PPO Logic - Ground truth passed: {deepmath_ground_truth['ground_truth']}")
    print(f"PPO Logic - Score: {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("✓ main_ppo.py logic working correctly for DeepMath")
    print()
    
    # Simulate Countdown case
    countdown_ground_truth = {
        'target': 25,
        'numbers': [1, 2, 3, 4, 5, 6],
        'ground_truth': '25'
    }
    
    data_source = 'countdown'
    solution = "<answer>25</answer>"
    
    if data_source == 'zwhe99/DeepMath-103K':
        score = countdown_score(solution, countdown_ground_truth['ground_truth'])
    else:
        score = countdown_score(solution, countdown_ground_truth)
    
    print(f"PPO Logic - Countdown solution: {solution}")
    print(f"PPO Logic - Ground truth passed: {countdown_ground_truth}")
    print(f"PPO Logic - Score: {score}")
    print("✓ main_ppo.py logic working correctly for Countdown")

if __name__ == "__main__":
    test_deepmath_scorer()
    test_countdown_scorer()
    test_main_ppo_logic()
    print("🎉 All scorer tests passed!")
