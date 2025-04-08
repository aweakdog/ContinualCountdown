import random
from copy import deepcopy
from .countdown_utils import combine_nums, sum_heuristic, mult_heuristic, great_prune, mult_prune
import math
class Node:
    def __init__(self, operator, left=None, right=None, value=None):   
        self.operator = operator
        self.left = left
        self.right = right
        self.value = value
        self.parent_id = -1
    def is_lowest_operator(self):
        if self.left is None or self.right is None:
            return 
        return self.left.operator is None and self.right.operator is None
            


class CountDownReverse(object):
    def __init__(self, max_target=24, start_size=4, min_target=10, max_internal_value=1000, candidate_operators=None, neccessary_operators=None, distinct=False, one_limit_prob=0.1):
        self.max_target = max_target
        self.min_target = min_target
        self.max_internal_value = max_internal_value
        self.start_size = start_size
        self.candidate_operators = candidate_operators if candidate_operators is not None else ["+", "-", "*", "/"]
        self.neccessary_operators = neccessary_operators if neccessary_operators is not None else []
        self.distinct = distinct
        # if len(self.operators) == 1 :
        #     self.max_cnt_limit = 10
        # else:
        #     self.max_cnt_limit = 10000
        self.max_cnt_limit = 10000
        self.one_limit_prob = one_limit_prob  # Probability to accept samples containing 1
    
    def check_distinct_numbers(self, nums):
        """Check if all numbers in the list are distinct."""
        return len(nums) == len(set(nums))

    def generate(self):
        while True:
            all_nodes = self.generate_operation_tree(self.start_size)
            cnt = 0
            while cnt <= self.max_cnt_limit:
                res = self.fill_values(all_nodes[-1], [self.min_target, self.max_target], 1)
                cnt += 1
                assert self.min_target!=0 or res or self.max_target!=self.max_internal_value
                if res:
                    target, nums, solution = self.encode(all_nodes)
                    # Check if numbers should be distinct and if they are
                    # Check if numbers should be distinct
                    if not self.distinct or self.check_distinct_numbers(nums):
                        # Check if sample contains 1
                        has_one = 1 in nums
                        # Accept sample based on one_limit_prob if it has 1, or always accept if it doesn't have 1
                        # this is hacky
                        if not has_one or random.random() < self.one_limit_prob:
                            return target, nums, solution
            print("Failed to generate a valid value after", cnt, "attempts.")
    def encode(self, all_nodes):
        target = all_nodes[-1].value
        assert self.min_target<=target and target<=self.max_target
        nodes = deepcopy(all_nodes)
        lowest_operator_nodes = [node for node in nodes 
             if node.is_lowest_operator()]
        nums = [node.value for node in nodes if node.operator is None]
        operations = []
        # print("nums:", nums)
        while (len(lowest_operator_nodes)>0):
            # print(len(lowest_operator_nodes))
            node = random.sample(lowest_operator_nodes, 1)[0]
            # print(node.left.value, node.right.value, [node.operator])
            value, operation = combine_nums(node.left.value, node.right.value, [node.operator])[0]
            operations.append(operation)
            assert node.value == value
            lowest_operator_nodes.remove(node)
            node.operator = None
            if node.parent_id!=-1:
                parent = nodes[node.parent_id]
                if parent.is_lowest_operator():
                    lowest_operator_nodes.append(parent)
        return target, nums, operations
                
    def generate_operation_tree(self, start_size):
        pass_check = False
        while not pass_check:
            operators = random.choices(self.candidate_operators, k=start_size-1)
            operators_set = set(operators)
            pass_check = True
            for i in self.neccessary_operators:
                if i not in operators_set:
                    pass_check = False
                    break
            
        all_nodes: List[Node] = [Node(None) for _ in range(start_size)]
        nodes = all_nodes[:]
        num = 0
        while len(nodes) > 1:
            # Randomly select two nodes
            node1, node2 = random.sample(nodes, 2)
            
            # Remove selected nodes from the list
            nodes.remove(node1)
            nodes.remove(node2)
            
            # Create a new parent node with a random operator
            #operator = random.choice(self.operators)
            parent = Node(operators[num], node1, node2)
            num += 1

            # Add the new parent node back to the list
            nodes.append(parent)
            all_nodes.append(parent)
            node1.parent_id = len(all_nodes)-1
            node2.parent_id = len(all_nodes)-1
        all_nodes[-1].parent_id = -1
        return all_nodes
    def get_all_factors(self, num):
        if num == 0:
            return [0]
        factors = [i for i in range(1, int(math.sqrt(num))+1) if num % i == 0]
        l = len(factors)
        for i in range(l):
            if i*i == num:
                continue
            factors.append(num//factors[i])
        return factors
    def sample_int_in_range(self, value_range, base):
        seed_range = [(value_range[0]+(base-1))//base, value_range[1]//base]
        assert seed_range[0] <= seed_range[1]
        seed = random.randint(seed_range[0], seed_range[1])
        return seed * base
    def fill_values(self, tree_node, value_range, base):
        # print("fill_value1:", value_range, base, tree_node.operator)
        if value_range[0] > value_range[1]:
            return False
        if tree_node.operator == None:
            intersection_range = [max(value_range[0], self.min_target), min(value_range[1], self.max_target)]
            # print("range", intersection_range)
            if intersection_range[0]>intersection_range[1]:
                return False
            tree_node.value = self.sample_int_in_range(intersection_range, base)
            # print("fill_value2:", value_range, base, tree_node.value)
            return True
        if tree_node.operator == "+":
            left_value_range = [0, value_range[1]]
            if not self.fill_values(tree_node.left, left_value_range, base):
                return False
            left_value = tree_node.left.value
            right_value_range = [max(value_range[0]-left_value, 0), value_range[1]-left_value]
            if not self.fill_values(tree_node.right, right_value_range, base):
                return False
            right_value = tree_node.right.value
            tree_node.value = left_value + right_value
        elif tree_node.operator == "-":
            left_value_range = [value_range[0], self.max_internal_value]
            if not self.fill_values(tree_node.left, left_value_range, base):
                return False
            left_value = tree_node.left.value
            right_value_range = [max(left_value - value_range[1], 0), left_value - value_range[0]]
            # print("right_value_range - :", right_value_range, left_value, value_range, tree_node.operator)
            if not self.fill_values(tree_node.right, right_value_range, base):
                return False
            right_value = tree_node.right.value
            tree_node.value = left_value - right_value
        elif tree_node.operator == "*":
            product = self.sample_int_in_range(value_range, base)
            # print("product", product)
            left_value_sample = random.sample(self.get_all_factors(product), 1)[0]
            left_value_range = [left_value_sample, left_value_sample]
            if not self.fill_values(tree_node.left, left_value_range, 1):
                return False
            left_value = tree_node.left.value
            if (left_value == 0):
                assert(product == 0)
                right_value_range = [0, 0]
            else:
                right_value_range = [product//left_value, product//left_value]
            # print("right_value_range * :", right_value_range, left_value, value_range, tree_node.operator)
            if not self.fill_values(tree_node.right, right_value_range, 1):
                return False
            right_value = tree_node.right.value
            tree_node.value = left_value * right_value
        elif tree_node.operator == "/":
            quotient = self.sample_int_in_range(value_range, base)
            if (quotient == 0):
                left_value_range = [1, self.max_internal_value]
            else:
                left_value_range = [1, self.max_internal_value//quotient]
            # print("left_value_range / :", left_value_range, value_range, tree_node.operator, base, quotient)
            if not self.fill_values(tree_node.left, left_value_range, 1):
                return False
            left_value = tree_node.left.value
            if (value_range[1] == 0):
                right_value_range = [0, 0]
            else:
                right_value_range = [left_value*quotient, left_value*quotient]
            # print("right_value_range / :", right_value_range, left_value, value_range, tree_node.operator)
            if not self.fill_values(tree_node.right, right_value_range, 1):
                return False
            right_value = tree_node.right.value
            assert right_value % left_value == 0
            tree_node.value = right_value // left_value
        else:
            print("Operator Error.")
        return True
    def random_countdown(self, nums):
        remain_nums = deepcopy(nums)
        operations = []
        while (len(remain_nums)>1):
            i, j = random.sample(range(len(remain_nums)), 2)
            possible = combine_nums(remain_nums[i], remain_nums[j], self.operators)
            choice = random.sample(possible, 1)[0]
            remain_nums = [remain_nums[k] for k in range(len(remain_nums)) if k != i and k != j] + [choice[0]]
            operations+=[choice[1]]
        return remain_nums[0], operations

    
    def convert_to_path(self, target, nums, operations):
        # convert solution to readable path

        operations_explored = []
        search_trace = ""
        search_trace += f"Current State: {target}:{nums}, Operations: {operations_explored}\n"
        node_index = 1
        for operation in operations:
            # split at operation +, -, *, /
            if "+" in operation:
                i, j = operation.split("=")[0].split("+")
                i, j = int(i), int(j)
                result = i + j
            elif "-" in operation:
                i, j = operation.split("=")[0].split("-")
                i, j = int(i), int(j)
                result = i - j
            elif "*" in operation:
                i, j = operation.split("=")[0].split("*")
                i, j = int(i), int(j)
                result = i * j
            elif "/" in operation:
                i, j = operation.split("=")[0].split("/")
                i, j = int(i), int(j)
                result = i / j

            result = int(result)
            new_nums = [int(nums[k]) for k in range(len(nums)) if nums[k] != i and nums[k] != j] + [result]
            nums = new_nums
            search_trace += f"Exploring Operation: {operation}, Resulting Numbers: {nums}\n"
            if len(nums) == 1:
                search_trace += f"{nums[0]},{target} equal: Goal Reached\n"
            else:
                node_index += 1
                search_trace += f"Generated Node #{node_index}: {new_nums} from Operation: {operation}\n"
                operations_explored.append(operation)
                search_trace += f"Current State: {target}:{nums}, Operations: {operations_explored}\n"
        return search_trace
def evaluate_tree(root):
    if not root:
        return None
    
    if not root.is_operator:  # Leaf node
        return root.value
    
    left_val = evaluate_tree(root.left)
    right_val = evaluate_tree(root.right)
    
    if root.value == '+':
        return left_val + right_val
    elif root.value == '-':
        return left_val - right_val
    elif root.value == '*':
        return left_val * right_val
    elif root.value == '/':
        if right_val == 0:  # Avoid division by zero
            return float('inf')
        return left_val / right_val

def find_valid_values(root, target_range=(1, 1000)):
    def is_valid(val):
        return target_range[0] <= val <= target_range[1]
    
    def dfs(node):
        if not node:
            return set()
        
        if not node.is_operator:  # Leaf node
            # Return all possible values for this leaf
            return {i for i in range(target_range[0], target_range[1] + 1)}
        
        left_vals = dfs(node.left)
        right_vals = dfs(node.right)
        
        valid_values = set()
        for left_val in left_vals:
            for right_val in right_vals:
                if node.value == '+':
                    result = left_val + right_val
                elif node.value == '-':
                    result = left_val - right_val
                elif node.value == '*':
                    result = left_val * right_val
                elif node.value == '/':
                    if right_val == 0:
                        continue
                    result = left_val / right_val
                    if not result.is_integer():
                        continue
                    result = int(result)
                
                if is_valid(result):
                    valid_values.add(result)
        
        return valid_values

    return dfs(root)

def assign_values(root, target_range=(1, 1000)):
    """
    Assigns valid values to leaf nodes to make the expression evaluate within target range
    """
    import random
    
    def is_valid(val):
        return target_range[0] <= val <= target_range[1]
    
    def try_assign(node):
        if not node:
            return True, None
        
        if not node.is_operator:  # Leaf node
            # Try a random value in range
            val = random.randint(target_range[0], target_range[1])
            node.value = val
            return True, val
        
        # For operator nodes, try different combinations
        for _ in range(100):  # Limit attempts
            success_left, left_val = try_assign(node.left)
            success_right, right_val = try_assign(node.right)
            
            if not (success_left and success_right):
                continue
                
            result = None
            if node.value == '+':
                result = left_val + right_val
            elif node.value == '-':
                result = left_val - right_val
            elif node.value == '*':
                result = left_val * right_val
            elif node.value == '/':
                if right_val == 0:
                    continue
                if left_val % right_val != 0:  # Ensure integer division
                    continue
                result = left_val // right_val
            
            if is_valid(result):
                return True, result
        
        return False, None
    
    success, final_value = try_assign(root)
    return success, final_value

# Example usage
if __name__ == "__main__":
    # Create a sample expression tree: (a + b) * (c - d)
    #       *
    #      / \
    #     +   -
    #    / \ / \
    #   a  b c  d
    
    tree = Node('*', is_operator=True)
    tree.left = Node('+', is_operator=True)
    tree.right = Node('-', is_operator=True)
    tree.left.left = Node(None)
    tree.left.right = Node(None)
    tree.right.left = Node(None)
    tree.right.right = Node(None)
    
    # Try to assign valid values
    success, result = assign_values(tree)
    if success:
        print("Found valid assignment!")
        print(f"Left side: {tree.left.left.value} + {tree.left.right.value}")
        print(f"Right side: {tree.right.left.value} - {tree.right.right.value}")
        print(f"Final result: {result}")
    else:
        print("Could not find valid assignment")
