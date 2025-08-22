# coding=utf-8
"""
IFeval scorer based on official Google Research implementation.
This module provides accurate instruction-following evaluation using the official IFeval library.
"""

import json
import random
import re
import string
import collections
from typing import Dict, Any, Optional, Union, Sequence

# Import official IFeval library components
import sys
import os
ifeval_lib_path = os.path.join(os.path.dirname(__file__), 'ifeval_lib')
if ifeval_lib_path not in sys.path:
    sys.path.insert(0, ifeval_lib_path)

import instructions_registry
import instructions_util


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute IFeval score using official Google Research implementation.
    
    Args:
        solution_str: The model's response text
        ground_truth: JSON string containing constraint information
        
    Returns:
        Float score between 0.0 and 1.0
    """
    try:
        if not ground_truth or ground_truth.strip() == "":
            print(f"Error scoring IFeval response: Empty ground truth")
            return 0.0
            
        gt_data = json.loads(ground_truth)
        func_name = gt_data.get('func_name', '')
        
        if not func_name:
            print(f"Error scoring IFeval response: No func_name in ground truth")
            return 0.0
        
        # Use official library
        score = _compute_score_official(solution_str, func_name, gt_data)
        
        # Debug output (random sampling like DeepMath scorer)
        do_print = random.randint(1, 64) == 1
        if do_print:
            print(f"--------------------------------")
            print(f"IFeval Constraint: {func_name}")
            print(f"Ground truth: {ground_truth}")
            print(f"Response (first 300 chars): {solution_str[:300]}...")
            print(f"Score: {score}")
            print(f"--------------------------------")
        
        return score
        
    except json.JSONDecodeError as e:
        print(f"Error scoring IFeval response: JSON decode error - {e}")
        print(f"Ground truth content: '{ground_truth}'")
        return 0.0
    except Exception as e:
        print(f"Error scoring IFeval response: {e}")
        print(f"Ground truth content: '{ground_truth}'")
        return 0.0


def _compute_score_official(response: str, instruction_id: str, kwargs: Dict[str, Any]) -> float:
    """
    Compute score using official IFeval library.
    
    Args:
        response: Model response
        instruction_id: Instruction identifier (e.g., 'punctuation:no_comma')
        kwargs: Instruction parameters
        
    Returns:
        Score (1.0 if instruction followed, 0.0 otherwise)
    """
    try:
        # Map func_name to official instruction_id format
        instruction_id_mapped = _map_func_name_to_instruction_id(instruction_id)
        
        if instruction_id_mapped not in instructions_registry.INSTRUCTION_DICT:
            print(f"Unknown instruction ID: {instruction_id_mapped}")
            return 0.0
        
        # Get instruction class and create instance
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id_mapped]
        instruction = instruction_cls(instruction_id_mapped)
        
        # Build instruction with parameters
        try:
            # Filter kwargs to only include relevant parameters
            filtered_kwargs = _filter_kwargs_for_instruction(kwargs, instruction)
            instruction.build_description(**filtered_kwargs)
        except Exception as e:
            print(f"Error building instruction {instruction_id_mapped}: {e}")
            return 0.0
        
        # Check if response follows instruction
        is_following = instruction.check_following(response.strip())
        return 1.0 if is_following else 0.0
        
    except Exception as e:
        print(f"Error in official scorer for {instruction_id}: {e}")
        return 0.0


def _map_func_name_to_instruction_id(func_name: str) -> str:
    """Map our func_name to official instruction ID format."""
    mapping = {
        'validate_lowercase': 'change_case:english_lowercase',
        'validate_uppercase': 'change_case:english_capital', 
        'validate_no_commas': 'punctuation:no_comma',
        'validate_word_count': 'length_constraints:number_words',
        'verify_paragraph_count': 'length_constraints:number_paragraphs',
        'validate_keyword_frequency': 'keywords:frequency',
        'validate_forbidden_words': 'keywords:forbidden_words',
        'validate_end_phrase': 'startend:end_checker',
        'validate_first_word': 'length_constraints:nth_paragraph_first_word',
        'validate_letter_frequency': 'keywords:letter_frequency',
        'validate_quotation': 'startend:quotation',
        'validate_json_format': 'detectable_format:json_format',
        'validate_title': 'detectable_format:title',
        'validate_postscript': 'detectable_content:postscript',
        'verify_bullet_points': 'detectable_format:number_bullet_lists',
        'validate_end': 'startend:end_checker',
        'verify_keyword_frequency': 'keywords:frequency',
        'validate_placeholders': 'detectable_content:number_placeholders',
        'validate_choice': 'detectable_format:constrained_response',
        'verify_letter_frequency': 'keywords:letter_frequency',
        'validate_repeat_prompt': 'combination:repeat_prompt',
        'verify_postscript': 'detectable_content:postscript',
        'validate_highlighted_sections': 'detectable_format:number_highlighted_sections',
        'validate_word_constraint': 'keywords:existence',
        'validate_sections': 'detectable_format:multiple_sections',
        'validate_paragraphs': 'length_constraints:number_paragraphs',
        'verify_sentence_constraint': 'length_constraints:number_sentences',
        'validate_frequency_capital_words': 'change_case:capital_word_frequency',
        'validate_two_responses': 'combination:two_responses',
        'verify_keywords': 'keywords:existence',
    }
    return mapping.get(func_name, func_name)


def _filter_kwargs_for_instruction(kwargs: Dict[str, Any], instruction) -> Dict[str, Any]:
    """Filter kwargs to only include parameters relevant to the instruction."""
    try:
        valid_keys = instruction.get_instruction_args_keys()
    except:
        valid_keys = []
    
    filtered = {}
    
    # Map our parameter names to official ones
    param_mapping = {
        'N': 'num_paragraphs',
        'frequency': 'frequency', 
        'letter': 'letter',
        'keyword': 'keyword',
        'end_phrase': 'end_phrase',
        'first_word': 'first_word',
        'postscript_marker': 'postscript_marker',
        'options': 'constrained_responses',
        'forbidden_words': 'forbidden_words',
        'keyword_list': 'keywords',
        'original_prompt': 'prompt_to_repeat',
        'num_words': 'num_words',
        'num_sentences': 'num_sentences',
        'num_paragraphs': 'num_paragraphs',
        'num_bullets': 'num_bullets',
        'num_highlights': 'num_highlights',
        'num_sections': 'num_sections',
        'section_splitter': 'section_spliter',  # Note: official lib has typo
        'capital_frequency': 'capital_frequency',
        'let_frequency': 'let_frequency',
        'let_relation': 'let_relation',
        'capital_relation': 'capital_relation',
        'relation': 'relation',
    }
    
    for key, value in kwargs.items():
        if value is not None:
            mapped_key = param_mapping.get(key, key)
            if mapped_key in valid_keys:
                filtered[mapped_key] = value
    
    return filtered


