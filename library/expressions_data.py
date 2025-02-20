import random
import numpy as np
import sympy as sp
import tree_pde as tree
import tree_grammar_pde as tree_grammar
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count

class Grammar:
    def __init__(self, case="default"):
        self.base_alphabet = {
            '+' : 2, 
            '-' : 2,
            '*' : 2, 
            '/' : 2, 
            'sin' : 1, 
            'exp' : 1, 
            'log' : 1, 
            'cos' : 1, 
            'x' : 0, 
            '(-x)': 0, 
            '1': 0, 
            'x^2' : 0, 
            't^3' : 0, 
            't^2': 0, 
            't': 0, 
            '(-t)': 0
        }
        self.base_rules = { 'S' : [
            ('+', ['S', 'S']),
            ('-', ['S', 'S']), 
            ('*', ['S', 'S']), 
            ('/', ['S', 'S']), 
            ('sin', ['S']), 
            ('exp', ['S']), 
            ('log', ['S']), 
            ('cos', ['S']), 
            ('x', []),
            ('t', []), 
            ('(-x)', []),  
            ('(-t)', []), 
            ('1', []), 
            ('x^2', []), 
            ('t^3', []), 
            ('t^2', [])
        ]}
        self.set_grammar(case)

    def set_grammar(self, case):
        self.alphabet = self.base_alphabet.copy()
        self.rules = self.base_rules.copy()
        self.grammar = tree_grammar.TreeGrammar(self.alphabet, list(self.rules.keys()), 'S', self.rules)

grammar = Grammar()

def sample_tree(grammar, depth=0, max_depth=3):
    if depth >= max_depth:
        return _sample_literal().to_list_format()
    
    choice = random.choices(
        ['binary', 'unary', 'literal', 'chain_binary'],
        weights=[0.3, 0.3, 0.3, 0.1],
        k=1
    )[0]
    
    if choice == 'binary':
        return _sample_binary(depth + 1, max_depth).to_list_format()
    elif choice == 'unary':
        return _sample_unary(depth + 1, max_depth).to_list_format()
    elif choice == 'literal':
        return _sample_literal().to_list_format()
    else:
        return _sample_chain_binary(depth + 1, max_depth).to_list_format()

def _sample_combination(depth, max_depth):
    choice = random.choices(
        ['binary', 'unary', 'chain_binary'],
        weights=[0.4, 0.4, 0.2],
        k=1
    )[0]
    
    if choice == 'binary':
        return _sample_binary(depth + 1, max_depth)
    elif choice == 'unary':
        return _sample_unary(depth + 1, max_depth)
    else:
        return _sample_chain_binary(depth + 1, max_depth)

def _sample_binary(depth, max_depth):
    r = random.choice(['+', '*', '-', '/'])
    left = _sample_literal() if depth >= max_depth else _sample_combination(depth, max_depth)
    right = _sample_literal() if depth >= max_depth else _sample_combination(depth, max_depth)
    return tree.Tree(r, [left, right])

def _sample_chain_binary(depth, max_depth):
    r = random.choice(['+', '-', '*', '/'])
    left = _sample_literal() if depth >= max_depth else _sample_combination(depth, max_depth)
    right = _sample_chain_binary(depth + 1, max_depth) if depth + 1 < max_depth and random.random() > 0.5 else _sample_literal()
    return tree.Tree(r, [left, right])

def _sample_unary(depth, max_depth):
    r = random.choice(['exp', 'sin', 'log', 'cos'])
    child = _sample_literal() if depth >= max_depth else _sample_combination(depth, max_depth)
    return tree.Tree(r, [child])

def _sample_literal():
    literals = ['1', 'x', 'x^2', 't^2', 't^3', 't', '(-t)', '(-x)']
    r = random.choice(literals)
    return tree.Tree(r)

def to_algebraic_string(nodes, adj, i=0):
    if nodes[i] in ['+', '*', '/', '-']:
        return to_algebraic_string(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] in ['sin', 'exp', 'log', 'cos']:
        return nodes[i] + '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ')'
    else:
        return nodes[i]

def remove_outer_parentheses(expr):
    if expr.startswith('(') and expr.endswith(')'):
        balance = 0
        for i, char in enumerate(expr):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            if balance == 0 and i != len(expr) - 1:
                return expr
        return expr[1:-1]
    return expr

def preprocess_expression(expr):
    expr = expr.replace('(-x)', '-x').replace('(-t)', '-t')
    return expr

def is_valid_simplified_expr(expr):
    x, t = sp.symbols('x t')
    def check_powers(expression, x_max_power=2, t_max_power=3):
        if expression.is_Pow:
            base, exp = expression.args
            if base == x and exp > x_max_power:
                return False
            if base == t and exp > t_max_power:
                return False
        for arg in expression.args:
            if not check_powers(arg, x_max_power, t_max_power):
                return False
        return True
    return check_powers(expr)

def contains_x_and_t(expr):
    x, t = sp.symbols('x t')
    return expr.has(x) and expr.has(t)

def generate_expression(grammar, max_depth):
    while True:
        nodes, adj = sample_tree(grammar, max_depth=max_depth)
        expr_string = to_algebraic_string(nodes, adj)
        expr_string = remove_outer_parentheses(expr_string)

        x, t = sp.symbols('x t')
        expr_string_symbolic = preprocess_expression(expr_string)
        simplified_expr = sp.sympify(expr_string_symbolic).simplify()

        if contains_x_and_t(simplified_expr) and is_valid_simplified_expr(simplified_expr):
            return expr_string, simplified_expr

def collect_expression(args):
    grammar, expressions, simplified_expressions, unique_exprs, lock, max_depth = args
    while True:
        expr_string, simplified_expr = generate_expression(grammar, max_depth)
        with lock:
            if str(simplified_expr) not in unique_exprs:
                expressions.append(expr_string)
                simplified_expressions.append(simplified_expr)
                unique_exprs.append(str(simplified_expr))
                break

def save_expressions_to_file(filename, num_expressions, case="default"):
    grammar = Grammar(case)
    manager = Manager()
    expressions = manager.list()
    simplified_expressions = manager.list()
    unique_exprs = manager.list()  # Changed to list
    lock = manager.Lock()
    
    half_num_expressions = num_expressions // 2
    tasks = [(grammar, expressions, simplified_expressions, unique_exprs, lock, 2) for _ in range(half_num_expressions)]
    tasks += [(grammar, expressions, simplified_expressions, unique_exprs, lock, 3) for _ in range(num_expressions - half_num_expressions)]
    
    with tqdm(total=num_expressions, desc="Generating Expressions") as pbar:
        with Pool(processes=min(6, cpu_count() // 2)) as pool:
            for _ in pool.imap_unordered(collect_expression, tasks):
                pbar.update(1)
    
    with open(filename, 'w') as f:
        for expr in tqdm(expressions, desc="Saving Expressions"):
            f.write(expr + '\n')

