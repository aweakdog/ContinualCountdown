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

    # Heuristic: strip any prompt/example block by starting from the last marker
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
        # Use the LAST occurrence of a formatted answer within the stripped text
        matches = re.findall(r"#### (\-?[0-9\.,]+)", text)
        if len(matches) == 0:
            final_answer = None
        else:
            final_answer = matches[-1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", text)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

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
        if answer == ground_truth:
            return score
        else:
            return format_score