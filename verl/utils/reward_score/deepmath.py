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

import re
import random


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    # Strip any prompt/example block by starting from the last marker
    # commonly present in our prompts (Q:, A:, or the bridge sentence)
    text = solution_str if solution_str is not None else ""
    markers = ["\nQ:", "\nA:", "Now solve the following question"]
    last_idx = -1
    for m in markers:
        idx = text.rfind(m)
        if idx > last_idx:
            last_idx = idx
    if last_idx != -1:
        text = text[last_idx:]

    if method == 'strict':
        # Look for boxed answers first (DeepMath format) - handle nested braces
        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        boxed_matches = re.findall(boxed_pattern, text)
        if len(boxed_matches) > 0:
            final_answer = boxed_matches[-1].strip()
        else:
            # Enhanced fallback patterns for LaTeX expressions
            patterns = [
                r"-?\\d?frac\{[^}]+\}\{[^}]+\}",  # LaTeX fractions (complete)
                r"[a-zA-Z]\^?\d*\s*[+\-=]\s*\d+", # Simple equations
                r"(\-?[0-9\.,]+)",               # Numbers
            ]
            
            final_answer = None
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    final_answer = matches[-1].replace(',', '').replace('$', '')
                    break
                    
    elif method == 'flexible':
        # First try boxed format with nested braces
        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        boxed_matches = re.findall(boxed_pattern, text)
        if len(boxed_matches) > 0:
            final_answer = boxed_matches[-1].strip()
        else:
            # Enhanced flexible extraction with LaTeX support
            patterns = [
                r"-?\\d?frac\{[^}]+\}\{[^}]+\}",  # LaTeX fractions (complete)
                r"[a-zA-Z]\^?\d*\s*[+\-=]\s*\d+", # Simple equations
                r"(\-?[0-9\.\,]+)",               # Numbers
            ]
            
            final_answer = None
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    final_answer = matches[-1].replace(',', '').replace('$', '')
                    break
                    
    return final_answer


def normalize_answer(answer):
    """Minimal normalization for answer comparison"""
    if answer is None:
        return None
    
    # Convert to string and strip whitespace
    answer = str(answer).strip()
    
    # Only handle the most essential normalizations:
    # 1. Remove spaces around operators for consistency
    answer = answer.replace(' + ', '+').replace(' - ', '-').replace(' = ', '=')
    # 2. Normalize LaTeX fraction commands
    answer = answer.replace('\\dfrac', '\\frac')
    
    return answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for DeepMath-103K.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        # Normalize both answers for comparison
        normalized_answer = normalize_answer(answer)
        normalized_ground_truth = normalize_answer(ground_truth)
        
        if normalized_answer == normalized_ground_truth:
            return score
        else:
            return format_score
