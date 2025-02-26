#!/usr/bin/env python3
"""
Generate triplets (solution, operator, forcing term) for differential equations
based on configurations in a YAML file.
"""

import sys
import csv
import argparse
import random
import sympy as sp
from tqdm import tqdm

# Import configuration helper functions
from config_dt_helpers import (
    load_config,
    get_spatial_vars,
    get_valid_dimensions,
    get_coefficient,
    get_random_solution_option,
    get_all_operator_types,
    get_all_solution_types,
    get_solution_params
)

def generate_manufactured_solution_string(operator_type, dims, solution_type=None, parameters=None, config=None):
    """
    Returns a string representing the manufactured solution u.
    
    Args:
        operator_type (str): Type of differential operator
        dims (int): Spatial dimension
        solution_type (str, optional): Type of solution
        parameters (dict, optional): Solution parameters
        config (dict, optional): Configuration dictionary
        
    Returns:
        str: Manufactured solution string
    """
    # If parameters not provided, initialize as empty dict
    if parameters is None:
        parameters = {}
    
    # If solution_type is not specified, choose a default based on operator
    if solution_type is None:
        if config is not None:
            # Get a random solution option from config
            solution_option = get_random_solution_option(operator_type, config)
            solution_type = solution_option.get("type")
            # Update parameters with the option settings
            for key, value in solution_option.items():
                if key != "type":
                    parameters[key] = value
        else:
            # Use defaults from the original code
            if operator_type in ['wave', 'navier-stokes']:
                solution_type = 'sine_cosine'
            elif operator_type in ['diffusion', 'reaction-diffusion']:
                solution_type = 'exp_decay'
            elif operator_type == 'burgers':
                solution_type = 'tanh'
            elif operator_type == 'advection':
                solution_type = 'traveling_wave'
            elif operator_type in ['helmholtz', 'laplacian']:
                solution_type = 'sinusoidal'
            elif operator_type == 'biharmonic':
                solution_type = 'polynomial'
            else:
                solution_type = random.choice(['sine_cosine', 'exp_decay', 'polynomial'])
    
    # Generate solution based on type and dimension
    spatial_vars = get_spatial_vars(dims)
    
    if solution_type == 'sine_cosine':
        if dims == 1:
            # For wave equation, use sqrt(c2) in the time component if provided
            if operator_type == 'wave' and 'c2' in parameters and parameters.get('use_sqrt_c2', True):
                c2 = parameters['c2']
                return f"sin(x)*cos(sqrt({c2})*t)"
            else:
                return "sin(x)*cos(t)"
        else:
            space = "*".join(f"sin({var})" for var in spatial_vars)
            # For wave equation, use sqrt(c2) in the time component if provided
            if operator_type == 'wave' and 'c2' in parameters and parameters.get('use_sqrt_c2', True):
                c2 = parameters['c2']
                return f"{space}*cos(sqrt({c2})*t)"
            else:
                return f"{space}*cos(t)"
    
    elif solution_type == 'exp_decay':
        # For diffusion, use the diffusion coefficient if provided
        decay_rate = parameters.get('decay_rate', 1)
        if dims == 1:
            return f"exp(-{decay_rate}*t)*sin(x)"
        else:
            space = "*".join(f"sin({var})" for var in spatial_vars)
            return f"exp(-{decay_rate}*t)*{space}"
    
    elif solution_type == 'tanh':
        # For Burgers equation, use the advection velocity if provided
        velocity = parameters.get('velocity', 1)
        if dims == 1:
            return f"tanh(x-{velocity}*t)"
        else:
            # For higher dims, use tanh of a linear combination
            vars_str = "+".join(var for var in spatial_vars)
            return f"tanh({vars_str}-{velocity}*t)"
    
    elif solution_type == 'traveling_wave':
        # For advection, use the advection velocity if provided
        velocity = parameters.get('velocity', 1)
        if dims == 1:
            return f"sin(x-{velocity}*t)"
        else:
            # Simple traveling wave
            vars_str = "+".join(var for var in spatial_vars)
            return f"sin({vars_str}-{velocity}*t)"
    
    elif solution_type == 'sinusoidal':
        if dims == 1:
            return "sin(2*x)"
        else:
            return "*".join(f"sin(2*{var})" for var in spatial_vars)
    
    elif solution_type == 'polynomial':
        if dims == 1:
            return "x**2*(1-x)**2"
        else:
            # Product of quadratics for each dimension
            return "*".join(f"{var}**2*(1-{var})**2" for var in spatial_vars)
    
    elif solution_type == 'gaussian':
        sigma = parameters.get('sigma', 0.1)
        if dims == 1:
            return f"exp(-(x-0.5)**2/{sigma})"
        else:
            terms = [f"(({var}-0.5)**2)" for var in spatial_vars]
            sum_terms = "+".join(terms)
            return f"exp(-({sum_terms})/{sigma})"
    
    else:
        raise ValueError(f"Unknown solution type: {solution_type}")

def generate_operator_string(operator_type, dims, parameters, formatted=True):
    """
    Generate the differential operator string.
    
    Args:
        operator_type (str): Type of differential operator
        dims (int): Spatial dimension
        parameters (dict): Operator parameters
        formatted (bool): Whether to return grammar-friendly format (True) or sympy-compatible format (False)
        
    Returns:
        str: Operator string
    """
    spatial_vars = get_spatial_vars(dims)
    
    # Common symbols for both formats
    separator = "·" if formatted else "*"
    
    if operator_type == 'diffusion':
        D_val = parameters.get('D', 0.5)
        if dims == 1:
            if formatted:
                return f"dt(u) - {D_val}·dxx(u)"
            else:
                return f"diff(u, t) - {D_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {D_val}·({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {D_val}*({laplace})"
    
    elif operator_type == 'wave':
        c2_val = parameters.get('c2', 1.0)
        if dims == 1:
            if formatted:
                return f"dtt(u) - {c2_val}·dxx(u)"
            else:
                return f"diff(u, t, 2) - {c2_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dtt(u) - {c2_val}·({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t, 2) - {c2_val}*({laplace})"
    
    elif operator_type == 'burgers':
        nu_val = parameters.get('nu', 0.05)
        if formatted:
            return f"dt(u) + u·dx(u) - {nu_val}·dxx(u)"
        else:
            return f"diff(u, t) + u*diff(u, x) - {nu_val}*diff(u, x, 2)"
    
    elif operator_type == 'reaction-diffusion':
        D_val = parameters.get('D', 0.5)
        k_val = parameters.get('k', 0.5)
        if dims == 1:
            if formatted:
                return f"dt(u) - {D_val}·dxx(u) + {k_val}·u"
            else:
                return f"diff(u, t) - {D_val}*diff(u, x, 2) + {k_val}*u"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {D_val}·({laplace}) + {k_val}·u"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {D_val}*({laplace}) + {k_val}*u"
    
    elif operator_type == 'navier-stokes':
        nu_val = parameters.get('nu', 0.05)
        if dims == 1:
            if formatted:
                return f"dt(u) + u·dx(u) - {nu_val}·dxx(u)"
            else:
                return f"diff(u, t) + u*diff(u, x) - {nu_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) + u·dx(u) - {nu_val}·({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) + u*diff(u, x) - {nu_val}*({laplace})"
    
    elif operator_type == 'advection':
        v_val = parameters.get('velocity', 1.0)
        if dims == 1:
            if formatted:
                return f"dt(u) + {v_val}·dx(u)"
            else:
                return f"diff(u, t) + {v_val}*diff(u, x)"
        else:
            if formatted:
                advection = " + ".join(f"{v_val}·d{var}(u)" for var in spatial_vars)
                return f"dt(u) + {advection}"
            else:
                advection = " + ".join(f"{v_val}*diff(u, {var})" for var in spatial_vars)
                return f"diff(u, t) + {advection}"
    
    elif operator_type == 'helmholtz':
        k_val = parameters.get('k', 1.0)
        if dims == 1:
            if formatted:
                return f"-dxx(u) - {k_val}·u"
            else:
                return f"-diff(u, x, 2) - {k_val}*u"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"-({laplace}) - {k_val}·u"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"-({laplace}) - {k_val}*u"
    
    elif operator_type == 'biharmonic':
        alpha_val = parameters.get('alpha', 0.05)
        if dims == 1:
            if formatted:
                return f"dt(u) + {alpha_val}·dxxxx(u)"
            else:
                return f"diff(u, t) + {alpha_val}*diff(u, x, 4)"
        else:
            if formatted:
                biharmonic = " + ".join(f"d{var}{var}{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) + {alpha_val}·({biharmonic})"
            else:
                biharmonic = " + ".join(f"diff(u, {var}, 4)" for var in spatial_vars)
                return f"diff(u, t) + {alpha_val}*({biharmonic})"
    
    elif operator_type == 'laplacian':
        coef_val = parameters.get('coef', 1.0)
        if dims == 1:
            if formatted:
                return f"{coef_val}·dxx(u)"
            else:
                return f"{coef_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"{coef_val}·({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"{coef_val}*({laplace})"
    
    # Default fallback
    return f"L[u] for {operator_type} in {dims}D"

def compute_forcing_term(u_str, L_sympy_str, dims):
    """
    Compute the forcing term f = L[u] using SymPy with more aggressive simplification.
    """
    try:
        # Set up symbols based on dimension
        t = sp.symbols('t')
        if dims == 1:
            x = sp.symbols('x')
            local_dict = {'x': x, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tanh': sp.tanh}
            vars_list = [x, t]
        elif dims == 2:
            x, y = sp.symbols('x y')
            local_dict = {'x': x, 'y': y, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tanh': sp.tanh}
            vars_list = [x, y, t]
        elif dims == 3:
            x, y, z = sp.symbols('x y z')
            local_dict = {'x': x, 'y': y, 'z': z, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'tanh': sp.tanh}
            vars_list = [x, y, z, t]
        
        # Parse the manufactured solution
        u_expr = sp.sympify(u_str, locals=local_dict)
        
        # For the operator, treat 'u' as a function of the variables
        u = sp.Function('u')(*vars_list)
        
        # Parse the operator string with u as a function
        local_dict_L = dict(local_dict)
        local_dict_L.update({'diff': sp.diff, 'u': u})
        L_expr = sp.sympify(L_sympy_str, locals=local_dict_L)
        
        # Substitute u_expr for u in L_expr
        f_expr = L_expr.subs({u: u_expr})
        
        # More aggressive simplification
        try:
            # Expand and collect terms
            f_expr = sp.expand(f_expr)
            f_expr = sp.collect(f_expr, t)
            
            # Additional simplification techniques
            f_expr = sp.simplify(f_expr)
            f_expr = sp.trigsimp(f_expr, deep=True)
            f_expr = sp.factor(f_expr)
        except Exception as e:
            print(f"Warning: Advanced simplification failed: {e}", file=sys.stderr)
        
        # Convert the forcing term to a string
        f_formatted = str(f_expr)
        
        return f_formatted
    
    except Exception as e:
        print(f"Error computing forcing term: {e}", file=sys.stderr)
        return "Failed to compute forcing term"

def generate_triplet(operator_type, dims, solution_type=None, solution_params=None, config=None):
    """
    Generate a triplet (u_str, L_str, f_str) for the given operator and dimension.
    
    Args:
        operator_type (str): Type of differential operator
        dims (int): Spatial dimension (1, 2, or 3)
        solution_type (str, optional): Type of solution
        solution_params (dict, optional): Solution parameters
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: (solution string, operator string, forcing term string)
    """
    if config is None:
        config = load_config()
    
    # Check if dimension is valid
    valid_dims = get_valid_dimensions(operator_type, config)
    if dims not in valid_dims:
        raise ValueError(f"Dimension {dims} not valid for operator type {operator_type}. Valid dimensions: {valid_dims}")
    
    # Get a random solution option if not specified
    if solution_type is None:
        solution_option = get_random_solution_option(operator_type, config)
        solution_type = solution_option.get("type")
        solution_params = solution_option
    elif solution_params is None:
        # Find the matching solution option in config
        solution_params = get_solution_params(operator_type, solution_type, config)
    
    # Initialize parameters dictionary
    parameters = {}
    
    # Get coefficient(s) for this operator
    if operator_type == 'reaction-diffusion':
        D_val, k_val = get_coefficient(operator_type, config)
        parameters['D'] = D_val
        parameters['k'] = k_val
        # Use reaction rate as decay rate for exp_decay solutions
        if solution_type == 'exp_decay':
            parameters['decay_rate'] = k_val * solution_params.get("decay_rate_factor", 1.0)
    else:
        coef = get_coefficient(operator_type, config)
        
        # Set parameters based on operator type
        if operator_type == 'diffusion':
            parameters['D'] = coef
            # For exp_decay solutions, use diffusion coefficient as decay rate
            if solution_type == 'exp_decay':
                parameters['decay_rate'] = coef * solution_params.get("decay_rate_factor", 1.0)
        elif operator_type == 'wave':
            parameters['c2'] = coef
            # Pass the use_sqrt_c2 parameter for sine_cosine solutions
            parameters['use_sqrt_c2'] = solution_params.get("use_sqrt_c2", True)
        elif operator_type == 'burgers':
            parameters['nu'] = coef
            parameters['velocity'] = solution_params.get("velocity", 1.0)
        elif operator_type == 'navier-stokes':
            parameters['nu'] = coef
        elif operator_type == 'advection':
            parameters['velocity'] = coef * solution_params.get("velocity_factor", 1.0)
        elif operator_type == 'helmholtz':
            parameters['k'] = coef
        elif operator_type == 'biharmonic':
            parameters['alpha'] = coef
        elif operator_type == 'laplacian':
            parameters['coef'] = coef
    
    # Apply any special solution parameters
    if solution_type == 'gaussian':
        parameters['sigma'] = solution_params.get("sigma", 0.1)
    
    # Generate the manufactured solution
    u_str = generate_manufactured_solution_string(operator_type, dims, solution_type, parameters)
    
    # Generate the operator string (formatted for readability)
    L_formatted = generate_operator_string(operator_type, dims, parameters, True)
    
    # Generate the sympy-compatible operator string for computation
    L_sympy_str = generate_operator_string(operator_type, dims, parameters, False)
    
    # Compute the forcing term
    f_str = compute_forcing_term(u_str, L_sympy_str, dims)
    
    return u_str, L_formatted, f_str

def generate_comprehensive_dataset(output_csv='comprehensive_triplets.csv', dims_set=None, config=None):
    """
    Generates a comprehensive dataset with all operators, dimensions, and solution types.
    
    Args:
        output_csv (str): Path to output CSV file
        dims_set (list, optional): List of dimensions to include
        config (dict, optional): Configuration dictionary
        
    Returns:
        int: Number of triplets generated
    """
    if config is None:
        config = load_config()
    
    # Get all operator types from config
    operator_types = get_all_operator_types(config)
    
    rows = []
    headers = ['operator_type', 'spatial_dimension', 'solution_type', 'manufactured_solution_u', 'operator_L', 'forcing_term_f']
    
    # Calculate total iterations for progress bar
    total_iterations = 0
    for op_type in operator_types:
        valid_dims = get_valid_dimensions(op_type, config)
        if dims_set is not None:
            valid_dims = list(set(dims_set).intersection(valid_dims))
        solution_types = get_all_solution_types(op_type, config)
        total_iterations += len(valid_dims) * len(solution_types)
    
    # Create progress bar
    with tqdm(total=total_iterations, desc="Generating comprehensive dataset") as pbar:
        for op_type in operator_types:
            valid_dims = get_valid_dimensions(op_type, config)
            if dims_set is not None:
                valid_dims = list(set(dims_set).intersection(valid_dims))
            
            if not valid_dims:
                continue  # Skip if no valid dimensions
            
            solution_types = get_all_solution_types(op_type, config)
            
            for dims in valid_dims:
                for sol_type in solution_types:
                    try:
                        # Update progress bar description
                        pbar.set_description(f"Processing {op_type}, dims={dims}, sol={sol_type}")
                        
                        # Get solution parameters
                        sol_params = get_solution_params(op_type, sol_type, config)
                        
                        # Generate the triplet
                        u_str, L_formatted, f_str = generate_triplet(op_type, dims, sol_type, sol_params, config)
                        
                        # Add to results
                        rows.append([op_type, dims, sol_type, u_str, L_formatted, f_str])
                    except Exception as e:
                        print(f"Error generating {op_type}, dims={dims}, sol={sol_type}: {e}", file=sys.stderr)
                    finally:
                        # Update progress bar
                        pbar.update(1)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Comprehensive dataset saved to {output_csv}")
    return len(rows)

def generate_random_dataset(output_csv='random_triplets.csv', num_per_operator=1, operators=None, 
                           dimensions=None, solution_types=None, config=None):
    """
    Generates a random dataset with specified number of examples per operator.
    
    Args:
        output_csv (str): Path to output CSV file
        num_per_operator (int): Number of examples per operator
        operators (list, optional): List of operators to include
        dimensions (list, optional): List of dimensions to consider
        solution_types (list, optional): List of solution types to consider
        config (dict, optional): Configuration dictionary
        
    Returns:
        int: Number of triplets generated
    """
    if config is None:
        config = load_config()
    
    # Get operators to include
    if operators is None:
        operators = get_all_operator_types(config)
    
    rows = []
    headers = ['operator_type', 'spatial_dimension', 'solution_type', 'manufactured_solution_u', 'operator_L', 'forcing_term_f']
    
    # Calculate total iterations for progress bar
    total_iterations = len(operators) * num_per_operator
    
    # Create progress bar
    with tqdm(total=total_iterations, desc="Generating random dataset") as pbar:
        for op_type in operators:
            # Get valid dimensions for this operator
            valid_dims = get_valid_dimensions(op_type, config)
            if dimensions is not None:
                valid_dims = list(set(dimensions).intersection(valid_dims))
            
            if not valid_dims:
                # Skip this operator if no valid dimensions
                pbar.update(num_per_operator)
                continue
            
            # Get valid solution types for this operator
            valid_sol_types = get_all_solution_types(op_type, config)
            if solution_types is not None:
                valid_sol_types = list(set(solution_types).intersection(valid_sol_types))
            
            if not valid_sol_types:
                # Skip this operator if no valid solution types
                pbar.update(num_per_operator)
                continue
            
            for i in range(num_per_operator):
                try:
                    # Randomly select dimension
                    dims = random.choice(valid_dims)
                    
                    # Randomly select solution type
                    sol_type = random.choice(valid_sol_types)
                    
                    # Get solution parameters
                    sol_params = get_solution_params(op_type, sol_type, config)
                    
                    # Update progress bar description
                    pbar.set_description(f"Processing {op_type} ({i+1}/{num_per_operator}), dims={dims}, sol={sol_type}")
                    
                    # Generate the triplet
                    u_str, L_formatted, f_str = generate_triplet(op_type, dims, sol_type, sol_params, config)
                    
                    # Add to results
                    rows.append([op_type, dims, sol_type, u_str, L_formatted, f_str])
                except Exception as e:
                    print(f"Error generating random triplet for {op_type}: {e}", file=sys.stderr)
                finally:
                    # Update progress bar
                    pbar.update(1)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Random dataset saved to {output_csv}")
    return len(rows)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate differential equation triplets')
    
    parser.add_argument('--config', type=str, default='config_dataset.yaml',
                        help='Path to configuration YAML file')
    
    parser.add_argument('--output', type=str, default='datasets/generated_triplets.csv',
                        help='Output CSV file path')
    
    parser.add_argument('--mode', type=str, choices=['comprehensive', 'random'], 
                        default='random',
                        help='Dataset generation mode')
    
    parser.add_argument('--num-per-operator', type=int, default=1,
                        help='Number of examples per operator (for random mode)')
    
    parser.add_argument('--operators', type=str, nargs='+',
                        help='Specific operator types to include')
    
    parser.add_argument('--dimensions', type=int, nargs='+', choices=[1, 2, 3],
                        help='Specific dimensions to include')
    
    parser.add_argument('--solutions', type=str, nargs='+',
                        help='Specific solution types to include')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    # Generate dataset based on mode
    try:
        if args.mode == 'comprehensive':
            count = generate_comprehensive_dataset(
                output_csv=args.output,
                dims_set=args.dimensions,
                config=config
            )
        else:  # random mode
            count = generate_random_dataset(
                output_csv=args.output,
                num_per_operator=args.num_per_operator,
                operators=args.operators,
                dimensions=args.dimensions,
                solution_types=args.solutions,
                config=config
            )
        
        print(f"Successfully generated {count} triplets")
        return 0
    
    except Exception as e:
        print(f"Error generating dataset: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())