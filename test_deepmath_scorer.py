#!/usr/bin/env python3

import sys
import os
sys.path.append('/cpfs04/user/liyuanhang.p/src/ContinualCountdown')

from verl.utils.reward_score.deepmath import compute_score

def test_deepmath_scorer():
    """Test the DeepMath scorer with various examples"""
    
    # Test case 1: Simple boxed answer
    solution1 = "The answer is \\boxed{2}."
    ground_truth1 = "2"
    score1 = compute_score(solution1, ground_truth1)
    print(f"Test 1 - Solution: {solution1}")
    print(f"Ground truth: {ground_truth1}")
    print(f"Score: {score1}")
    print("---")
    
    # Test case 2: Boxed answer with different format
    solution2 = "After solving, we get \\boxed{0}."
    ground_truth2 = "0"
    score2 = compute_score(solution2, ground_truth2)
    print(f"Test 2 - Solution: {solution2}")
    print(f"Ground truth: {ground_truth2}")
    print(f"Score: {score2}")
    print("---")
    
    # Test case 3: No boxed answer, fallback to number extraction
    solution3 = "The final answer is 42."
    ground_truth3 = "42"
    score3 = compute_score(solution3, ground_truth3)
    print(f"Test 3 - Solution: {solution3}")
    print(f"Ground truth: {ground_truth3}")
    print(f"Score: {score3}")
    print("---")
    
    # Test case 4: Mismatch case
    solution4 = "The answer is \\boxed{5}."
    ground_truth4 = "3"
    score4 = compute_score(solution4, ground_truth4)
    print(f"Test 4 - Solution: {solution4}")
    print(f"Ground truth: {ground_truth4}")
    print(f"Score: {score4}")
    print("---")
    
    # Test case 5: Complex LaTeX fraction
    solution5 = "The result is \\boxed{\\frac{1}{2}}."
    ground_truth5 = "\\frac{1}{2}"
    score5 = compute_score(solution5, ground_truth5)
    print(f"Test 5 - Solution: {solution5}")
    print(f"Ground truth: {ground_truth5}")
    print(f"Score: {score5}")
    print("---")

if __name__ == "__main__":
    test_deepmath_scorer()
