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

import json
import re
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
    """Validate response contains a title (markdown # or angular brackets <<title>>)"""
    import re
    # Check for markdown-style titles
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith('#'):
            return True
    
    # Check for angular bracket titles like <<title>>
    angular_pattern = r'<<.*?>>'
    if re.search(angular_pattern, response):
        return True
    
    return False


def validate_postscript(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains a postscript (P.S. or PS:)"""
    response_lower = response.lower()
    return 'p.s.' in response_lower or 'ps:' in response_lower or 'postscript' in response_lower


# Additional constraint validation functions
def verify_bullet_points(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify response contains bullet points"""
    lines = response.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('•') or stripped.startswith('*') or stripped.startswith('-') or stripped.startswith('◦'):
            return True
    return False


def validate_end(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response ends with specific phrase"""
    end_phrase = gt_data.get('end_phrase', '')
    if not end_phrase:
        return True
    return response.strip().endswith(end_phrase)


def verify_keyword_frequency(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify keyword appears with specified frequency"""
    keyword = gt_data.get('keyword', '')
    frequency = gt_data.get('frequency', 1)
    if not keyword:
        return True
    return response.lower().count(keyword.lower()) >= frequency


def validate_placeholders(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains placeholders like [placeholder]"""
    import re
    placeholders = re.findall(r'\[.*?\]', response)
    return len(placeholders) > 0


def validate_choice(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains one of the specified choices"""
    options = gt_data.get('options', [])
    if not options:
        return True
    response_lower = response.lower()
    return any(option.lower() in response_lower for option in options)


def verify_letter_frequency(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify specific letter appears with required frequency"""
    letter = gt_data.get('letter', '')
    frequency = gt_data.get('frequency', 1)
    if not letter:
        return True
    return response.lower().count(letter.lower()) >= frequency


def validate_repeat_prompt(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response repeats the original prompt"""
    original_prompt = gt_data.get('original_prompt', '')
    if not original_prompt:
        return True
    return original_prompt.lower() in response.lower()


def verify_postscript(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify response contains postscript with specific marker"""
    postscript_marker = gt_data.get('postscript_marker', 'P.S.')
    return postscript_marker.lower() in response.lower()


def validate_highlighted_sections(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains highlighted sections (markdown or HTML)"""
    import re
    # Check for markdown bold/italic or HTML tags
    markdown_pattern = r'\*\*.*?\*\*|\*.*?\*|__.*?__|_.*?_'
    html_pattern = r'<[^>]+>.*?</[^>]+>'
    return bool(re.search(markdown_pattern, response) or re.search(html_pattern, response))


def validate_word_constraint(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response meets word-related constraints"""
    word = gt_data.get('word', '')
    if not word:
        return True
    return word.lower() in response.lower()


def validate_sections(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response is divided into sections"""
    section_splitter = gt_data.get('section_splitter', '\n\n')
    sections = response.split(section_splitter)
    return len(sections) >= 2


def validate_paragraphs(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response has required number of paragraphs"""
    N = gt_data.get('N', 1)
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    return len(paragraphs) >= N


def verify_sentence_constraint(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify sentence-level constraints"""
    import re
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    N = gt_data.get('N', 1)
    return len(sentences) >= N


def validate_uppercase(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response is in uppercase"""
    return response.isupper()


def validate_frequency_capital_words(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate frequency of capitalized words"""
    import re
    words = re.findall(r'\b[A-Z][a-z]*\b', response)
    frequency = gt_data.get('frequency', 1)
    return len(words) >= frequency


def validate_two_responses(response: str, gt_data: Dict[str, Any]) -> bool:
    """Validate response contains two separate responses"""
    # Look for common separators or numbered responses
    import re
    separators = ['\n\n', '---', '***', 'Response 1:', 'Response 2:', '1.', '2.']
    for sep in separators:
        if sep in response:
            parts = response.split(sep)
            if len([p for p in parts if p.strip()]) >= 2:
                return True
    return False


def verify_keywords(response: str, gt_data: Dict[str, Any]) -> bool:
    """Verify response contains required keywords"""
    keyword_list = gt_data.get('keyword_list', [])
    if not keyword_list:
        return True
    response_lower = response.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


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
    # Additional constraint functions
    'verify_bullet_points': verify_bullet_points,
    'validate_end': validate_end,
    'verify_keyword_frequency': verify_keyword_frequency,
    'validate_placeholders': validate_placeholders,
    'validate_choice': validate_choice,
    'verify_letter_frequency': verify_letter_frequency,
    'validate_repeat_prompt': validate_repeat_prompt,
    'verify_postscript': verify_postscript,
    'validate_highlighted_sections': validate_highlighted_sections,
    'validate_word_constraint': validate_word_constraint,
    'validate_sections': validate_sections,
    'validate_paragraphs': validate_paragraphs,
    'verify_sentence_constraint': verify_sentence_constraint,
    'validate_uppercase': validate_uppercase,
    'validate_frequency_capital_words': validate_frequency_capital_words,
    'validate_two_responses': validate_two_responses,
    'verify_keywords': verify_keywords,
}


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute IFeval score for a response given ground truth constraints.
    
    Args:
        solution_str: The model's response text
        ground_truth: JSON string containing constraint information
        
    Returns:
        float: 1.0 if all constraints are satisfied, 0.0 otherwise
    """
    try:
        # Parse the ground truth JSON
        if not ground_truth or ground_truth.strip() == "":
            print(f"Error scoring IFeval response: Empty ground truth")
            return 0.0
            
        gt_data = json.loads(ground_truth)
        func_name = gt_data.get('func_name', '')
        
        # Get the validation function
        validation_func = CONSTRAINT_FUNCTIONS.get(func_name)
        if validation_func is None:
            print(f"Error scoring IFeval response: Unknown constraint function: {func_name}")
            return 0.0
        
        # Validate the response
        is_valid = validation_func(solution_str, gt_data)
        score = 1.0 if is_valid else 0.0
        
        # Debug output (like DeepMath scorer)
        do_print = random.randint(1, 64) == 1
        if do_print:
            print(f"--------------------------------")
            print(f"IFeval Constraint: {func_name}")
            print(f"Ground truth: {ground_truth}")
            
            # Special handling for validate_repeat_prompt to show prompt vs response
            if func_name == 'validate_repeat_prompt':
                original_prompt = gt_data.get('original_prompt', '')
                print(f"ORIGINAL PROMPT (should be repeated): {original_prompt}")
                print(f"FULL MODEL RESPONSE: {solution_str}")
                if original_prompt and solution_str.startswith(original_prompt):
                    remaining = solution_str[len(original_prompt):]
                    print(f"✓ REPEATED PART: {original_prompt}")
                    print(f"✓ ADDITIONAL CONTENT: {remaining.strip()}")
                    print(f"Validation logic: Check if original_prompt is contained in response")
                else:
                    print(f"Response does not start with original prompt")
            else:
                print(f"Full Response: {solution_str}")
            
            print(f"Validation result: {is_valid}")
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
