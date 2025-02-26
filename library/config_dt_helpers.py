#!/usr/bin/env python3
"""
Helper functions for working with differential equation configuration.
"""

import random
import sys
import yaml

def load_config(config_file='config_dataset.yaml'):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.", file=sys.stderr)
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}", file=sys.stderr)
        raise

def get_spatial_vars(dims):
    """
    Returns a list of spatial variable names based on the dimension.
    For dims=1: ['x'], dims=2: ['x','y'], dims=3: ['x','y','z'].
    
    Args:
        dims (int): Spatial dimension (1, 2, or 3)
        
    Returns:
        list: List of variable names
        
    Raises:
        ValueError: If the dimension is not supported
    """
    if dims == 1:
        return ['x']
    elif dims == 2:
        return ['x', 'y']
    elif dims == 3:
        return ['x', 'y', 'z']
    else:
        raise ValueError(f"Unsupported dimension: {dims}")

def get_valid_dimensions(operator_type, config=None):
    """
    Returns valid dimensions for an operator from config.
    
    Args:
        operator_type (str): Type of the differential operator
        config (dict, optional): Configuration dictionary. If None, loads from file.
        
    Returns:
        list: List of valid dimensions
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    operator_info = operators.get(operator_type, {})
    
    # Default to all dimensions if not specified in config
    return operator_info.get("valid_dims", [1, 2, 3])

def get_coefficient(operator_type, config=None):
    """
    Returns coefficient(s) for an operator based on config.
    
    Args:
        operator_type (str): Type of the differential operator
        config (dict, optional): Configuration dictionary. If None, loads from file.
        
    Returns:
        float or tuple: The coefficient value(s)
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    operator_info = operators.get(operator_type, {})
    
    # Special case for reaction-diffusion with separate coefficient ranges
    if operator_type == 'reaction-diffusion':
        d_range = operator_info.get("d_coefficient_range", [0.1, 1.0])
        k_range = operator_info.get("k_coefficient_range", [0.1, 1.0])
        D_val = round(random.uniform(d_range[0], d_range[1]), 3)
        k_val = round(random.uniform(k_range[0], k_range[1]), 3)
        return D_val, k_val
    
    # Standard case for other operators
    coef_range = operator_info.get("coefficient_range", [0.1, 1.0])
    coef = round(random.uniform(coef_range[0], coef_range[1]), 3)
    
    # Apply any transform specified
    transform = operator_info.get("coefficient_transform")
    if transform == "square":
        # For wave equation, c^2
        return round(coef**2, 3)
    
    return coef

def get_random_solution_option(operator_type, config=None):
    """
    Returns a randomly selected solution option for an operator.
    
    Args:
        operator_type (str): Type of the differential operator
        config (dict, optional): Configuration dictionary. If None, loads from file.
        
    Returns:
        dict: Solution option dictionary with type and parameters
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    operator_info = operators.get(operator_type, {})
    solution_options = operator_info.get("solution_options", [{"type": "sine_cosine"}])
    
    return random.choice(solution_options)

def get_all_operator_types(config=None):
    """
    Returns a list of all operator types defined in the configuration.
    
    Args:
        config (dict, optional): Configuration dictionary. If None, loads from file.
        
    Returns:
        list: List of operator type strings
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    return list(operators.keys())

def get_all_solution_types(operator_type, config=None):
    """
    Returns a list of all solution types defined for an operator.
    
    Args:
        operator_type (str): Type of the differential operator
        config (dict, optional): Configuration dictionary. If None, loads from file.
        
    Returns:
        list: List of solution type strings
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    operator_info = operators.get(operator_type, {})
    solution_options = operator_info.get("solution_options", [])
    
    return [option.get("type") for option in solution_options]

def get_solution_params(operator_type, solution_type, config=None):
    """
    Gets parameters for a specific solution type of an operator.
    
    Args:
        operator_type (str): Type of the differential operator
        solution_type (str): Type of the solution
        config (dict, optional): Configuration dictionary. If None, loads from file.
        
    Returns:
        dict: Solution parameters dictionary
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    operator_info = operators.get(operator_type, {})
    solution_options = operator_info.get("solution_options", [])
    
    # Find the solution option with matching type
    for option in solution_options:
        if option.get("type") == solution_type:
            return option
    
    # If not found, return a default option with just the type
    return {"type": solution_type}

# Test function to verify configuration
def test_config(config_file='config_dataset.yaml'):
    """
    Test loading and accessing the configuration.
    
    Args:
        config_file (str): Path to the YAML configuration file
    """
    try:
        config = load_config(config_file)
        
        # Print all operator types
        operator_types = get_all_operator_types(config)
        print(f"Found operators: {', '.join(operator_types)}")
        
        # Test each operator
        for op_type in operator_types:
            print(f"\nOperator: {op_type}")
            
            # Get valid dimensions
            valid_dims = get_valid_dimensions(op_type, config)
            print(f"  Valid dimensions: {valid_dims}")
            
            # Get solution types
            sol_types = get_all_solution_types(op_type, config)
            print(f"  Solution types: {sol_types}")
            
            # Get a coefficient
            if op_type == 'reaction-diffusion':
                D_val, k_val = get_coefficient(op_type, config)
                print(f"  Sample coefficients: D={D_val}, k={k_val}")
            else:
                coef = get_coefficient(op_type, config)
                print(f"  Sample coefficient: {coef}")
            
            # Get a random solution option
            sol_option = get_random_solution_option(op_type, config)
            print(f"  Random solution option: {sol_option}")
        
        print("\nConfiguration test successful!")
        
    except Exception as e:
        print(f"Error testing configuration: {e}", file=sys.stderr)
        return False
    
    return True

if __name__ == "__main__":
    # If run directly, test the configuration
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        test_config(config_file)
    else:
        test_config()