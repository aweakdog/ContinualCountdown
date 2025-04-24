import re
import random
import ast
import operator
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError

from pyparsing import (Word, nums, oneOf, Forward, Group, Suppress, 
                      ZeroOrMore, ParseException, infixNotation, opAssoc,
                      ParseSyntaxException, White, Regex, StringEnd)



def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None

import re
def extract_complex_expressions(s, number_of_numbers):
    allowed_chars = set('0123456789+-*/(). ')
    start_chars = set('0123456789(')
    digits = set('0123456789')
    operators = set("+-*/")
    n = len(s)
    valid_exprs = []
    i = 0
    while i < n:
        if s[i] not in start_chars or s[i-1] in digits or s[i-1]==".":
            i += 1
            continue
        max_vaild_j = i-1
        j = i
        numbers = 0
        stack = [0]  # 0 empty, 1 digit, 2 operator, 3 (digit+operator)digit
        while j < n:
            # print(i,j,s[i:j], "checking", s[j], max_vaild_j)
            if s[j] not in allowed_chars:
                break
            # print("112")
            if s[j] =='.':
                break
            if s[j] ==' ':
                j += 1
                continue
            if s[j] == '(':
                stack.append(0)
                j += 1
                continue

            if s[j] == ')':
                if stack[-1] == 0 or stack[-1] == 1 or stack[-1] == 2:
                    # i = j + 1
                    break
                stack.pop()
                if len(stack) == 0:
                    break

                if stack[-1] == 1 or stack[-1] == 3:
                    break
                if stack[-1] == 0:
                    stack[-1] = 1
                else:
                    stack[-1] = 3
                if len(stack) == 1 and stack[-1] == 3:
                    max_vaild_j = j
                    max_vaild_j_numbers = numbers
                j += 1
                continue
            

            if s[j] in operators:
                if len(stack) == 0:
                    break
                if stack[-1] == 0 or stack[-1] == 2:
                    break
                stack[-1] = 2
                j += 1
                continue
            
            if s[j] in digits:
                if stack[-1] == 1 or stack[-1] == 3:
                    break
                if stack[-1] == 0:
                    stack[-1] = 1
                else:
                    stack[-1] = 3
                if s[j]=='0' and j < n - 1 and s[j+1] in digits:
                    break
                num_point = 0
                while j < n :
                    if s[j] in digits or s[j] ==".":
                        j += 1
                        if s[j] == ".":
                            num_point += 1
                    else:
                        break
                if num_point>=2:
                    break
                numbers += 1
                j -= 1
                if len(stack) == 1 and stack[-1] == 3:
                    max_vaild_j = j
                    max_vaild_j_numbers = numbers
                j += 1
                continue
        if max_vaild_j > i:
            if max_vaild_j_numbers == number_of_numbers:
                valid_exprs.append(s[i:max_vaild_j+1])
            i = max_vaild_j + 1
        else:
            i = i + 1
    return valid_exprs

def extract_think_contents(text):
    """提取所有think标签内容"""
    return re.findall(
        r'<think>(.*?)</think>', 
        text, 
        flags=re.DOTALL | re.IGNORECASE
    )

def extract_thought(solution_str, number_of_numbers=4):
    """主提取函数"""
    if not isinstance(solution_str, str):
        return []
    
    think_texts = extract_think_contents(solution_str)
    if not think_texts:
        return []
    
    results = []
    for think_text in think_texts:
        results.extend(extract_complex_expressions(think_text, number_of_numbers))
    
    return results


def estimate_thought_reward(thoughts, available_numbers, do_print=False):
    """
    Calculate the reward for the thought: 0.01 per unique, valid result, up to a maximum of 0.1.
    Only count if the expression uses all available numbers exactly once and produces a new result.
    """
    seen_results = dict()
    for expr in set(thoughts):
        if validate_equation(expr, available_numbers):
            result = evaluate_equation(expr)
            if result is not None and result not in seen_results:
                seen_results[result] = expr
    if do_print and seen_results:
        print("[estimate_thought_reward] Seen results and corresponding expressions:")  # DEBUG
        for res, expr in seen_results.items():
            print(f"  Result: {res} | Expression: {expr}")
    reward = 0.01 * len(seen_results)
    return min(reward, 0.1)

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    # Handle both simple and extenged ground truth formats
    if isinstance(ground_truth, dict) and 'ground_truth' in ground_truth:
        ground_truth = ground_truth['ground_truth']
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    if isinstance(numbers, list):
        numbers = list(numbers)
    elif hasattr(numbers, 'tolist'):
        numbers = numbers.tolist()
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    thoughts = extract_thought(solution_str=solution_str)
    if do_print:
        print('extracted thoughts:',thoughts)
    format_score = estimate_thought_reward(thoughts, numbers, do_print)

    #if equation is None:
    #    if do_print:
    #        print(f"No equation found")
    #    return 0

    if equation is None:
        if do_print:
            print(f"No equation found")
        return format_score 
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 