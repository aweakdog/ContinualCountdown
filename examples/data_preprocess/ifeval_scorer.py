#!/usr/bin/env python3

import re
import json
from typing import Dict, Any, Optional

class IFEvalScorer:
    """Scorer for IFeval instruction following constraints"""
    
    def __init__(self):
        self.constraint_functions = {
            'validate_lowercase': self.validate_lowercase,
            'verify_paragraph_count': self.verify_paragraph_count,
            'validate_no_commas': self.validate_no_commas,
            'validate_word_count': self.validate_word_count,
            'validate_keyword_frequency': self.validate_keyword_frequency,
            'validate_forbidden_words': self.validate_forbidden_words,
            'validate_end_phrase': self.validate_end_phrase,
            'validate_first_word': self.validate_first_word,
            'validate_letter_frequency': self.validate_letter_frequency,
            'validate_quotation': self.validate_quotation,
            'validate_json_format': self.validate_json_format,
            'validate_title': self.validate_title,
            'validate_postscript': self.validate_postscript,
        }
    
    def score(self, response: str, ground_truth: str) -> float:
        """
        Score a response against IFeval constraints
        
        Args:
            response: Model's response text
            ground_truth: JSON string containing constraint validation info
            
        Returns:
            Score between 0.0 and 1.0 (1.0 = constraint satisfied)
        """
        try:
            gt_data = json.loads(ground_truth)
            func_name = gt_data.get('func_name', 'unknown')
            
            if func_name not in self.constraint_functions:
                return 0.0
            
            # Call the appropriate validation function
            return float(self.constraint_functions[func_name](response, gt_data))
            
        except Exception as e:
            print(f"Error scoring response: {e}")
            return 0.0
    
    def validate_lowercase(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate that entire response is in lowercase"""
        return response == response.lower()
    
    def verify_paragraph_count(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Verify response has exactly N paragraphs separated by * * *"""
        expected_count = gt_data.get('N', 0)
        if expected_count <= 0:
            return False
        
        # Split by markdown divider * * *
        paragraphs = re.split(r'\s*\*\s*\*\s*\*\s*', response.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return len(paragraphs) == expected_count
    
    def validate_no_commas(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate that response contains no commas"""
        return ',' not in response
    
    def validate_word_count(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response has exactly N words"""
        expected_count = gt_data.get('N', 0)
        if expected_count <= 0:
            return False
        
        words = response.split()
        return len(words) == expected_count
    
    def validate_keyword_frequency(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate keyword appears exactly N times"""
        keyword = gt_data.get('word', '')
        expected_count = gt_data.get('N', 0)
        
        if not keyword:
            return False
        
        # Case-insensitive count
        actual_count = response.lower().count(keyword.lower())
        return actual_count == expected_count
    
    def validate_forbidden_words(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response doesn't contain forbidden words"""
        forbidden_words = gt_data.get('forbidden_words', [])
        if not forbidden_words:
            return True
        
        response_lower = response.lower()
        for word in forbidden_words:
            if word.lower() in response_lower:
                return False
        return True
    
    def validate_end_phrase(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response ends with specific phrase"""
        end_phrase = gt_data.get('end_phrase', '')
        if not end_phrase:
            return False
        
        return response.strip().endswith(end_phrase.strip())
    
    def validate_first_word(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response starts with specific word"""
        first_word = gt_data.get('first_word', '')
        if not first_word:
            return False
        
        words = response.strip().split()
        if not words:
            return False
        
        return words[0].lower() == first_word.lower()
    
    def validate_letter_frequency(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate specific letter appears exactly N times"""
        letter = gt_data.get('letter', '')
        expected_count = gt_data.get('N', 0)
        
        if not letter:
            return False
        
        actual_count = response.lower().count(letter.lower())
        return actual_count == expected_count
    
    def validate_quotation(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response is wrapped in quotation marks"""
        response = response.strip()
        return (response.startswith('"') and response.endswith('"')) or \
               (response.startswith("'") and response.endswith("'"))
    
    def validate_json_format(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response is valid JSON"""
        try:
            json.loads(response.strip())
            return True
        except:
            return False
    
    def validate_title(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response contains a title (line starting with #)"""
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                return True
        return False
    
    def validate_postscript(self, response: str, gt_data: Dict[str, Any]) -> bool:
        """Validate response contains a postscript (P.S. or PS:)"""
        response_lower = response.lower()
        return 'p.s.' in response_lower or 'ps:' in response_lower or 'postscript' in response_lower


def test_scorer():
    """Test the scorer with sample constraints"""
    scorer = IFEvalScorer()
    
    # Test lowercase constraint
    gt_lowercase = '{"func_name": "validate_lowercase", "N": null}'
    print("Lowercase test:")
    print(f"  'hello world' -> {scorer.score('hello world', gt_lowercase)}")
    print(f"  'Hello World' -> {scorer.score('Hello World', gt_lowercase)}")
    
    # Test paragraph count
    gt_paragraphs = '{"func_name": "verify_paragraph_count", "N": 3}'
    response_3_para = "First paragraph * * * Second paragraph * * * Third paragraph"
    response_2_para = "First paragraph * * * Second paragraph"
    print("\nParagraph count test:")
    print(f"  3 paragraphs -> {scorer.score(response_3_para, gt_paragraphs)}")
    print(f"  2 paragraphs -> {scorer.score(response_2_para, gt_paragraphs)}")
    
    # Test no commas
    gt_no_commas = '{"func_name": "validate_no_commas", "N": null}'
    print("\nNo commas test:")
    print(f"  'Hello world' -> {scorer.score('Hello world', gt_no_commas)}")
    print(f"  'Hello, world' -> {scorer.score('Hello, world', gt_no_commas)}")


if __name__ == '__main__':
    test_scorer()
