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
import json
import random
from typing import Dict, Any


def validate_lowercase(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate that entire response is in lowercase"""
    return response == response.lower()


def verify_paragraph_count(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify response has exactly N paragraphs separated by * * *"""
    expected_count = gt_data.get('N', 0)
    if expected_count <= 0:
        return False
    
    # Split by markdown divider * * *
    paragraphs = re.split(r'\s*\*\s*\*\s*\*\s*', response.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return len(paragraphs) == expected_count


def validate_no_commas(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate that response contains no commas"""
    return ',' not in response


def validate_word_count(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response has exactly N words"""
    expected_count = gt_data.get('N', 0)
    if expected_count <= 0:
        return False
    
    words = response.split()
    return len(words) == expected_count


def validate_keyword_frequency(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate keyword appears exactly N times"""
    keyword = gt_data.get('word', '')
    expected_count = gt_data.get('N', 0)
    
    if not keyword:
        return False
    
    # Case-insensitive count
    actual_count = response.lower().count(keyword.lower())
    return actual_count == expected_count


def validate_forbidden_words(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response doesn't contain forbidden words"""
    forbidden_words = gt_data.get('forbidden_words', [])
    if not forbidden_words:
        return True
    
    response_lower = response.lower()
    for word in forbidden_words:
        if word.lower() in response_lower:
            return False
    return True


def validate_end_phrase(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response ends with specific phrase"""
    end_phrase = gt_data.get('end_phrase', '')
    if not end_phrase:
        return False
    
    return response.strip().endswith(end_phrase.strip())


def validate_first_word(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response starts with specific word"""
    first_word = gt_data.get('first_word', '')
    if not first_word:
        return False
    
    words = response.strip().split()
    if not words:
        return False
    
    return words[0].lower() == first_word.lower()


def validate_letter_frequency(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate specific letter appears exactly N times"""
    letter = gt_data.get('letter', '')
    expected_count = gt_data.get('N', 0)
    
    if not letter:
        return False
    
    actual_count = response.lower().count(letter.lower())
    return actual_count == expected_count


def validate_quotation(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response is wrapped in quotation marks"""
    response = response.strip()
    return (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'"))


def validate_json_format(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response is valid JSON"""
    try:
        json.loads(response.strip())
        return True
    except:
        return False


def validate_title(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains a title (line starting with #)"""
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith('#'):
            return True
    return False


def validate_postscript(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains a postscript (P.S. or PS:)"""
    response_lower = response.lower()
    return 'p.s.' in response_lower or 'ps:' in response_lower or 'postscript' in response_lower


# Constraint function mapping
CONSTRAINT_FUNCTIONS = {
    'validate_lowercase': validate_lowercase,
    'verify_paragraph_count': verify_paragraph_count,
    'validate_no_commas': validate_no_commas,
    'validate_word_count': validate_word_count,
    'validate_keyword_frequency': validate_keyword_frequency,
    'validate_forbidden_words': validate_forbidden_words,
    'validate_end_phrase': validate_end_phrase,
    'validate_first_word': validate_first_word,
    'validate_letter_frequency': validate_letter_frequency,
    'validate_quotation': validate_quotation,
    'validate_json_format': validate_json_format,
    'validate_title': validate_title,
    'validate_postscript': validate_postscript,
}


def compute_score(solution_str, ground_truth, score=1.):
    """The scoring function for IFeval instruction following constraints.

    Args:
        solution_str: the solution text (model response)
        ground_truth: the ground truth constraint specification (JSON string)
        score: the score for satisfying the constraint
    """
    try:
        gt_data = json.loads(ground_truth)
        func_name = gt_data.get('func_name', 'unknown')
        
        do_print = random.randint(1, 64) == 1
        if do_print:
            print(f"--------------------------------")
            print(f"Constraint function: {func_name}")
            print(f"Ground truth: {ground_truth}")
            print(f"Solution string: {solution_str[:200]}...")
        
        if func_name not in CONSTRAINT_FUNCTIONS:
            if do_print:
                print(f"Unknown constraint function: {func_name}")
            return 0.0
        
        # Call the appropriate validation function
        constraint_satisfied = CONSTRAINT_FUNCTIONS[func_name](solution_str, gt_data)
        
        if do_print:
            print(f"Constraint satisfied: {constraint_satisfied}")
        
        return score if constraint_satisfied else 0.0
        
    except Exception as e:
        do_print = random.randint(1, 16) == 1
        if do_print:
            print(f"Error scoring IFeval response: {e}")
        return 0.0
