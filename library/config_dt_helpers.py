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
    return operator_info.get("valid_dimensions", [1, 2, 3])

def get_coefficient(operator_type, config=None):
    """
    Returns coefficient(s) for an operator based on config.
    """
    if config is None:
        config = load_config()
    
    operators = config.get("operators", {})
    operator_info = operators.get(operator_type, {})
    
    # # Check for coefficient_options list
    # if "coefficient_options" in operator_info:
    #     return random.choice(operator_info["coefficient_options"])
    
    # # Check for coefficient_pairs list (for reaction-diffusion)
    # if "coefficient_pairs" in operator_info and operator_type == 'reaction-diffusion':
    #     return random.choice(operator_info["coefficient_pairs"])
    
    if operator_type == 'reaction_diffusion':
        d_range = operator_info.get("d_coefficient_range", [0.1, 1.0])
        k_range = operator_info.get("k_coefficient_range", [0.1, 1.0])
        D_val = random.uniform(float(d_range[0]), float(d_range[1]))
        k_val = random.uniform(float(k_range[0]), float(k_range[1]))
        return D_val, k_val
    elif operator_type == 'ginzburg_landau':
        print(f"DEBUG GL COEF: Starting get_coefficient for ginzburg_landau")
        try:
            # For the general case, return α and β values as a tuple
            alpha_range = operator_info.get("alpha_coefficient_range", [0.1, 2.0])
            beta_range = operator_info.get("beta_coefficient_range", [0.1, 2.0])
            
            print(f"DEBUG GL COEF: Alpha range: {alpha_range}, Beta range: {beta_range}")
            
            # Check if we should use coefficient pairs
            if "coefficient_pairs" in operator_info and random.random() < 0.7:
                pair = random.choice(operator_info["coefficient_pairs"])
                print(f"DEBUG GL COEF: Selected coefficient pair: {pair}, type: {type(pair)}")
                
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    return_val = (float(pair[0]), float(pair[1]))
                    print(f"DEBUG GL COEF: Returning tuple from pair: {return_val}, type: {type(return_val)}")
                    return return_val
                else:
                    print(f"DEBUG GL COEF: Invalid pair format, using defaults")
                    return (1.0, 1.0)
            
            # Otherwise generate random values from ranges
            alpha_val = random.uniform(float(alpha_range[0]), float(alpha_range[1]))
            beta_val = random.uniform(float(beta_range[0]), float(beta_range[1]))
            return_val = (alpha_val, beta_val)
            print(f"DEBUG GL COEF: Returning generated tuple: {return_val}, type: {type(return_val)}")
            return return_val
        except Exception as e:
            import traceback
            print(f"DEBUG GL COEF ERROR: {e}")
            print(f"DEBUG GL COEF TRACEBACK: {traceback.format_exc()}")
            return (1.0, 1.0)
    # Standard case for other operators
    coef_range = operator_info.get("coefficient_range", [0.1, 1.0])
    coef = random.uniform(float(coef_range[0]), float(coef_range[1]))
    return coef  # No squaring for wave here; handled in generate_triplet if needed
    
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
    solutions = operator_info.get("solutions", [{"type": "sine_cosine"}])
    
    return random.choice(solutions)

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
    solutions = operator_info.get("solutions", [])
    
    return [option.get("type") for option in solutions]

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
    solutions = operator_info.get("solutions", [])
    
    # Find the solution option with matching type
    for option in solutions:
        if option.get("type") == solution_type:
            return option
    
    # If not found, return a default option with just the type
    return {"type": solution_type}

def get_sample_range(param_dict, key, default_value):
    """If key exists as a range [low, high] in param_dict, return a uniformly sampled value (rounded to 3 decimals)."""
    if key in param_dict and isinstance(param_dict[key], list) and len(param_dict[key]) == 2:
        low, high = param_dict[key]
        return round(random.uniform(low, high), 3)
    return default_value


if __name__ == "__main__":

    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        test_config(config_file)
    else:
        test_config()

