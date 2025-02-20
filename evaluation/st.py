from contextlib import redirect_stdout
import symengine as sm
import sympy as sp
import numpy as np
import concurrent
import cupy as cp
import gc
from memory_profiler import profile
import tracemalloc
# from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pdf")
import random
import re
import os
import logging
import psutil
logging.basicConfig(filename='new-test-o.17-lesser-burger-6-accel-2-different-sign-sigma-equal to accel-3.log', level=logging.INFO, format='%(asctime)s %(message)s')
# Set seeds for reproducibility
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)

import symengine as sm
import sympy as sp
import numpy as np

class MeshGenerator:
    @staticmethod
    def generate(lbound, ubound, num):
        '''non-uniform'''
        uniform_grid = np.linspace(0, 1, num)
        a = lbound
        b = ubound
        X = a + (b - a) * (0.5 * (1 - np.cos(np.pi * uniform_grid))) 
        T = a + (b - a) * (0.5 * (1 - np.cos(np.pi * uniform_grid)))
        # X = uniform_grid
        # T = X

        X, T = np.meshgrid(X, T)  # Create 2D mesh grid
        dx = X[1, 0] - X[0, 0]
        dt = T[0, 1] - T[0, 0]
        return X, T, dx, dt

# class SystemProcessor:
#     def __init__(self, equations, u_expression):
#         self.equations = equations.splitlines()
#         self.u_expression = sm.sympify(u_expression)
#         self.x, self.t = sm.symbols('x t')
    
#     def apply_operator(self, operator, expr):
#         operator_map = {
#             'd3/dx3': sm.diff(sm.diff(sm.diff(expr, self.x), self.x), self.x),
#             'd2/dx2': sm.diff(sm.diff(expr, self.x), self.x),
#             'd/dx': sm.diff(expr, self.x),
#             'd2/dt2': sm.diff(sm.diff(expr, self.t), self.t),
#             'd/dt': sm.diff(expr, self.t),
#             'u': expr,
#             'u^2': expr^2
#         }
#         return operator_map.get(operator, expr)
    

class SystemProcessor:
    def __init__(self, equation, u_expression):
        self.equations = equation
        self.u_expression = sm.sympify(u_expression)
        self.x, self.t = sm.symbols('x t')

    # Regular expression to detect invalid combinations of 'x' and 't'
    invalid_pattern = re.compile(r'([xt]{2,})|(\d+[xt]+)|([xt]+\d+)')

    def contains_invalid_symbols(self, expr):
        # Check if any invalid combination of 'x' and 't' exists
        if self.invalid_pattern.search(str(expr)):
            return True  # Invalid combination found
        return False  # No invalid combinations

    
    # def apply_operator(self, operator, expr):
    #     operator_map = {
    #         'd3/dx3': sm.diff(sm.diff(sm.diff(expr, self.x), self.x), self.x),
    #         'd2/dx2': sm.diff(sm.diff(expr, self.x), self.x),
    #         'd/dx': sm.diff(expr, self.x),
    #         'd2/dt2': sm.diff(sm.diff(expr, self.t), self.t),
    #         'd/dt': sm.diff(expr, self.t),
    #         'u': f"({expr})", 
    #         'u^2': f"({expr})**2", 
    #         'u^3': f"({expr})**3" 
    #     }
    #     return operator_map.get(operator, expr)
    def apply_operator(self, operator, expr):
        if 'd3/dx3' in operator:
            return sm.diff(sm.diff(sm.diff(expr, self.x), self.x), self.x)
        elif 'd2/dx2' in operator:
            return sm.diff(sm.diff(expr, self.x), self.x)
        elif 'd/dx' in operator:
            if 'u^2' in operator:
                return sm.diff(expr**2, self.x)
            elif 'u^3' in operator:
                return sm.diff(expr**3, self.x)
            return sm.diff(expr, self.x)
        elif 'd2/dt2' in operator:
            return sm.diff(sm.diff(expr, self.t), self.t)
        elif 'd/dt' in operator:
            return sm.diff(expr, self.t)
        elif 'u^2' in operator:
            return expr**2
        elif 'u^3' in operator:
            return expr**3
        elif 'u' in operator:
            return expr
        else:
            return sm.sympify(operator)

    
    # def substitute_and_simplify(self, timeout=25):
    #     def perform_simplification():
    #         equation = self.equations.replace(" ", "")
    #         logging.info(f'equation :{equation}')
            
    #         if '-(' in equation:
    #             lhs, rhs = equation.split('-(')
    #             rhs = f"-({rhs.strip()[:-1]})"  # Remove the last closing parenthesis
    #         elif '+(' in equation:
    #             lhs, rhs = equation.split('+(')
    #             rhs = f"+({rhs.strip()[:-1]})"  # Remove the last closing parenthesis
    #         else:
    #             lhs = equation
    #             rhs = '0'

    #         simplified_expr = sm.sympify(0)
    #         lhs_terms = re.split(r'(\+|\-)', lhs)
    #         current_sign = 1  # Assume positive sign initially
    #         for term in lhs_terms:
    #             if term == '+':
    #                 current_sign = 1
    #                 continue
    #             elif term == '-':
    #                 current_sign = -1
    #                 continue
    #             logging.info(f'Processing term: {term}')
    #             if '*' in term:
    #                 parts = term.split('*')
    #                 logging.info(f'Processing term: {parts}')
    #                 if len(parts) == 2:
    #                     first, second = parts
    #                     if 'dx' in first or 'dt' in first or 'u^' in first:
    #                         operator, coefficient = first.strip(), second.strip()
    #                     else:
    #                         coefficient, operator = first.strip(), second.strip()

    #                     if 'u' not in coefficient:
    #                         coefficient = sm.sympify(coefficient)
    #                     else:
    #                         coefficient = sm.sympify(self.apply_operator(operator, sm.sympify(self.u_expression)))

    #                     if ('dx' in operator or 'dt' in operator) and ('u^' not in operator):
    #                         operator = operator.split('(')[0].strip()
    #                         simplified_expr += coefficient * self.apply_operator(operator, sm.sympify(self.u_expression))
    #                     elif ('dx' in operator or 'dt' in operator) and ('u^' in operator):
    #                         powered_u = operator[operator.find('(') + 1:operator.find(')')]
    #                         operator = operator.split('(')[0].strip()
    #                         power_op = self.apply_operator(powered_u, sm.sympify(self.u_expression))
    #                         simplified_expr += coefficient * self.apply_operator(operator, sm.sympify(power_op))
    #                     else:
    #                         simplified_expr += coefficient * sm.sympify(self.apply_operator(operator, sm.sympify(self.u_expression)))
    #             else:
    #                 if ('dx' in term or 'dt' in term) and ('u^' not in term):
    #                     operator = term.strip().split('(')[0].strip()
    #                     simplified_expr += self.apply_operator(operator, sm.sympify(self.u_expression))
    #                 elif ('dx' in term or 'dt' in term) and ('u^' in term):
    #                     powered_u = term[term.find('(') + 1:term.find(')')]
    #                     operator = term.strip().split('(')[0].strip()
    #                     power_op = self.apply_operator(powered_u, sm.sympify(self.u_expression))
    #                     simplified_expr += self.apply_operator(operator, sm.sympify(power_op))
    #                 else:
    #                     if 'u' in term:
    #                         constant = sm.sympify(self.apply_operator(term, sm.sympify(self.u_expression)))
    #                         simplified_expr += constant
    #                     else:
    #                         constant = sm.sympify(term.strip())
    #                         simplified_expr += constant
    #             logging.info(f'simplified_expr: {simplified_expr}')

    #         if 'u' in rhs.strip():
    #             rhs_expr = sm.sympify(rhs.strip().replace('u', f'({self.u_expression})'))
    #         else:
    #             rhs_expr = sm.sympify(rhs.strip())

    #         full_expr = simplified_expr + rhs_expr
    #         logging.info(f' full_expr :{full_expr}')
            
    #         # Attempt to simplify with a timeout
    #         return sp.simplify(full_expr)

    #     try:
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             future = executor.submit(perform_simplification)
    #             results = future.result(timeout=timeout)
    #     except concurrent.futures.TimeoutError:
    #         logging.warning(f"Simplification exceeded the timeout of {timeout} seconds.")
    #         results = None  # Return None or some fallback result
    #     except Exception as e:
    #         logging.error(f"An error occurred during simplification: {e}")
    #         results = None  # Return None or some fallback result
        
    #     logging.info(f'Final results: {results}')
    #     return results


    def substitute_and_simplify(self, timeout=25):
        def perform_simplification():
            equation = self.equations.replace(" ", "")
            logging.info(f'equation :{equation}')
            # Skip processing if the expression contains invalid symbols
            # if self.contains_invalid_symbols(equation):
            #     logging.warning(f"Skipping invalid equation: {equation}")
            #     return None  # Skip processing and return None

            
            # if '-(' in equation:
            #     lhs, rhs = equation.split('-(')
            #     rhs = f"-({rhs.strip()[:-1]})"  # Remove the last closing parenthesis
            # elif '+(' in equation:
            #     lhs, rhs = equation.split('+(')
            #     rhs = f"+({rhs.strip()[:-1]})"  # Remove the last closing parenthesis
            # else:
            #     lhs, rhs = equation.split('-', 1) if '-' in equation else (equation, '0')

            # if not rhs.startswith('('):
            #     rhs = f"({rhs.strip()})"  # Wrap RHS in parentheses if it's not already
            if '-(' in equation:
                lhs, rhs = equation.split('-(', 1)
                rhs = f"-({rhs}"  # Keep the remaining RHS as a whole
            elif '+(' in equation:
                lhs, rhs = equation.split('+(', 1)
                rhs = f"+({rhs}"  # Keep the remaining RHS as a whole
            else:
                lhs = equation
                rhs = '0'


            simplified_expr = sm.sympify(0)
            lhs_terms = re.split(r'(\+|\-)', lhs)
            # logging.info(f'lhs_terms: {lhs_terms}')
            current_sign = 1  # Assume positive sign initially
            
            for term in lhs_terms:
                term_expr=1
                if term == '+':
                    current_sign = 1
                    continue
                elif term == '-':
                    current_sign = -1
                    continue
                
                # logging.info(f'Processing term: {term}')
                
                # Split the term by '*' and process it as needed
                if 'd/dx' in term or 'd/dt' in term or 'd2/dx2' in term or 'd2/dt2' in term  or 'u^' in term or 'u' in term:
                    parts = term.split('*')
                    term_expr = current_sign * sm.sympify(1)
                    for part in parts:
                        # logging.info(f'part: {part}')
                        
                        part = part.strip()
                        
                        # logging.info(f'applied operator: {self.apply_operator(part, sm.sympify(self.u_expression))}')
                        term_expr *= self.apply_operator(part, sm.sympify(self.u_expression))
                else:
                    # logging.info(f'just coef: {sm.sympify(term)}')
                    term_expr *= sm.sympify(term)

                simplified_expr += term_expr
                logging.info(f'simplified_expr: {simplified_expr}')

            if 'u' in rhs.strip():
                rhs_expr = sm.sympify(rhs.strip().replace('u', f'({self.u_expression})'))
            else:
                # logging.info(f'rhs: {rhs}')
                rhs_expr = sm.sympify(rhs.strip())

            full_expr = simplified_expr + rhs_expr
            # logging.info(f' full_expr :{full_expr}')
            
            # Attempt to simplify with a timeout
            return sp.simplify(full_expr)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(perform_simplification)
                results = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logging.warning(f"Simplification exceeded the timeout of {timeout} seconds.")
            results = None  # Return None or some fallback result
        except Exception as e:
            logging.error(f"An error occurred during simplification: {e}")
            results = None  # Return None or some fallback result
        
        # logging.info(f'Final results: {results}')
        return results


    def evaluate_on_grid(self, simplified_expressions, X, T, timeout=15):
        def evaluate():
            f = sp.lambdify((self.x, self.t), simplified_expressions, 'numpy')
            return f(X, T)
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(evaluate)
                evaluated_results = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logging.warning(f"Evaluation exceeded the timeout of {timeout} seconds.")
            evaluated_results = np.full(X.shape, np.nan)  # Return NaN array as a fallback
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            evaluated_results = np.full(X.shape, np.nan)  # Return NaN array in case of any error
        
        return evaluated_results

    # def calculate_rmse(self, evaluated_results):
    #     rmses = []
    #     for result in evaluated_results:
    #         rmse = np.sqrt(np.mean(result**2))
    #         rmses.append(rmse)
    #     return rmses
    
    def calculate_rmse(self, evaluated_results):
        # rmses = []
        # for result in evaluated_results:
        rmse = np.sqrt(np.mean(evaluated_results**2))
       
        return rmse

# Example Usage:

# # List of equations in string form
# equations_str = """
# 3.0*d3/dx3(u)+2.0*d/dx(u)+1.0*d2/dt2(u)-(4.0*x)

# 7.0*d3/dx3(u)+2.0*d2/dt2(u)-(7.0*sin(x-2.4))
# 6.0*d3/dx3(u)+3.0*d/dt(u)-(3.0)
# 4.0*d3/dx3(u)+3.0*d/dx(u)+2.0*d/dt(u)-(5.0*t*cos(2.4*t)+2.0*sin(2.4*t))
# 3.0*d3/dx3(u)+4.0*d2/dx2(u)+2.0*d2/dt2(u)+(2.0*cos(t-2.4))
# """  # Continue with other equations as needed

# The expression for u
# equations_str = '5.0*d2/dx2(u)+3.0*d/dt(u)-(7.0)'
# u_expr = "2.2*t + x + 1.7"
equations_str="d2/dt2(u) + d2/dx2(u) + 4 . 9 9 * u^2 - ( 3 . 4 3 * ( t - x + 0 . 4 1 ) ^ 2 )"
u_expr=" t - x + 0.41"
# Instantiate the processor
processor = SystemProcessor(equations_str, u_expr)

# Substitute and simplify equations
simplified_expressions = processor.substitute_and_simplify()

# Generate the non-uniform mesh grid
X, T, dx, dt = MeshGenerator.generate(0, 1, 200)

# Evaluate the simplified expressions on the grid
evaluated_results = processor.evaluate_on_grid(simplified_expressions, X, T)

# Calculate RMSE for each evaluated result
rmses = processor.calculate_rmse(evaluated_results)
logging.info(f"Simplified Equation {simplified_expressions}")
logging.info(f"RMSE for Equation  {rmses}")

