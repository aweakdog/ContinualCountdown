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

def extract_complex_expressions(text):
    """
    准确提取含括号和空格的数学表达式（至少3个运算符）
    """
    # 定义基础元素（完全支持空格）
    integer = Regex(r'\d+').setWhitespaceChars(' \t')
    lpar = Suppress(White().leaveWhitespace() + '(' + White().leaveWhitespace())
    rpar = Suppress(White().leaveWhitespace() + ')' + White().leaveWhitespace())
    operator = oneOf("+ - * /").setWhitespaceChars(" \t")
    
    # 构建表达式解析器
    expr = Forward()
    atom = Group(lpar + expr + rpar) | integer
    expr <<= infixNotation(atom, [
        (oneOf("* /"), 2, opAssoc.LEFT),
        (oneOf("+ -"), 2, opAssoc.LEFT),
    ]) + StringEnd()
    
    # 改进的正则匹配候选表达式
    candidate_pattern = r'''
        (?:                     
            \( \s*              # 开括号和空格
            (?:                 
                [^()]          # 非括号字符
                |             
                \( [^()]* \)    # 一层嵌套括号
            )*                 
            \s* \)             # 闭括号和空格
            |                   # 或
            \d+                 # 数字
        )
        (?:                     # 运算符和操作数组
            \s*                 # 空格
            [+\-*/]             # 运算符
            \s*                 # 空格
            (?:                 
                \( [^()]+ \)    # 括号表达式
                |             
                \d+             # 数字
            )                 
        ){2,}                  # 至少2个（总共至少3个运算符）
    '''
    
    valid_exprs = []
    
    # 提取候选表达式
    for match in re.finditer(candidate_pattern, text, re.VERBOSE):
        expr_str = match.group(0).strip()
        
        # 基础验证
        if expr_str.count('(') != expr_str.count(')'):
            continue
            
        # 统计运算符数量
        op_count = len(re.findall(r'[+\-*/]', expr_str))
        if op_count < 3:
            continue
            
        # 保留原始格式（合并多余空格）
        normalized = ' '.join(expr_str.split())
        valid_exprs.append(normalized)
    
    return valid_exprs

def extract_think_contents(text):
    """提取所有think标签内容"""
    return re.findall(
        r'<think>(.*?)</think>', 
        text, 
        flags=re.DOTALL | re.IGNORECASE
    )

def extract_thought(solution_str):
    """主提取函数"""
    if not isinstance(solution_str, str):
        return []
    
    think_texts = extract_think_contents(solution_str)
    if not think_texts:
        return []
    
    results = []
    for think_text in think_texts:
        results.extend(extract_complex_expressions(think_text))
    
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

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if equation is None:
        if do_print:
            print(f"No equation found")
        return format_score 
    
    ## Validate equation uses correct numbers
    #if not validate_equation(equation, numbers):
    #    if do_print:
    #        print(f"Invalid equation")
    #    return format_score
        
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