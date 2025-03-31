import sys
import csv
import argparse
import random
import re
import sympy as sp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from library_utils import *
from config_dt_helpers import (
    load_config,
    get_spatial_vars,
    get_valid_dimensions,
    get_coefficient,
    get_random_solution_option,
    get_all_operator_types,
    get_all_solution_types,
    get_solution_params,
    get_sample_range
)

# =============================================================================
# INITIAL CONDITION GENERATION
# =============================================================================


# Dictionary mapping IC type to function
IC_GENERATORS = {
    'sine': generate_sine_ic,
    'gaussian': generate_gaussian_ic,
    'polynomial': generate_polynomial_ic,
    'step': generate_step_ic,
    'tanh': generate_tanh_ic,
}

def generate_initial_condition(ic_type, dims, params):
    """
    Generate initial condition based on type.
    
    Args:
        ic_type (str): Type of initial condition ('sine', 'gaussian', etc.)
        dims (int): Spatial dimension (1, 2, or 3)
        params (dict): Parameters for the initial condition
        
    Returns:
        str: Initial condition expression string
    """
    spatial_vars = get_spatial_vars(dims)
    
    # Get the generator function or default to sine
    generator = IC_GENERATORS.get(ic_type, generate_sine_ic)
    
    # Generate the initial condition
    return generator(spatial_vars, params)


# Dictionary mapping operator type to function
OPERATOR_BEHAVIORS = {
    'wave': apply_wave_operator,
    'diffusion': apply_diffusion_operator,
    'advection': apply_advection_operator,
    'reaction_diffusion': apply_reaction_diffusion_operator,
    'burgers': apply_burgers_operator,
    'inv_burgers': apply_inv_burgers_operator,
    'helmholtz': apply_helmholtz_operator,
    'laplacian': apply_laplacian_operator,
    'biharmonic': apply_biharmonic_operator,
    'allen_cahn': apply_allen_cahn_operator,
    'ginzburg_landau': apply_ginzburg_landau_operator,
    'fisher_kpp': apply_fisher_kpp_operator,
    'convection_diffusion': apply_convection_diffusion_operator,
    'fitzhugh_nagumo': apply_fitzhugh_nagumo_operator,
    'sine_gordon': apply_sine_gordon_operator,
    'telegrapher': apply_telegrapher_operator,
    'black_scholes': apply_black_scholes_operator,
    'tricomi': apply_tricomi_operator
}



# =============================================================================
# SOLUTION GENERATION
# =============================================================================


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
   
    if parameters is None:
        parameters = {}
    

    if solution_type is None:
        
        if config is not None:
            
            solution_option = get_random_solution_option(operator_type, config)
            solution_type = solution_option.get("type")
            
            for key, value in solution_option.items():
                if key != "type":
                    parameters[key] = value
        else:
           
            if operator_type in ['wave', 'navier-stokes']:
                solution_type = 'sine_cosine'
            elif operator_type in ['diffusion', 'reaction_diffusion']:
                solution_type = 'exp_decay'
            elif operator_type == 'burgers':
                solution_type = 'tanh'
            elif operator_type == 'inv_burgers':
                solution_type = 'tanh'
            elif operator_type == 'advection':
                solution_type = 'traveling_wave'
            elif operator_type in ['helmholtz', 'laplacian']:
                solution_type = 'sinusoidal'
            elif operator_type == 'biharmonic':
                solution_type = 'polynomial'
            else:
                solution_type = random.choice(['sine_cosine', 'exp_decay', 'polynomial'])
    parameters['solution_type'] = solution_type          
    # Convert the older solution_type parameter to ic_type for compatibility
    if 'ic_type' not in parameters:
        # Map solution_type to initial condition type
        solution_to_ic = {
            'sine_cosine': 'sine',
            'exp_decay': 'sine',  # Use sine as IC, exp decay as time component
            'traveling_wave': 'sine',
            'tanh': 'tanh',
            'sinusoidal': 'sine',
            'polynomial': 'polynomial',
            'gaussian': 'gaussian',
        }
        
        if solution_type:
            parameters['ic_type'] = solution_to_ic.get(solution_type, 'sine')
        else:
            parameters['ic_type'] = 'sine'  # Default to sine
    
    # Set appropriate parameters based on operator type
    if operator_type == 'diffusion':
        diffusion_coef = parameters.get('D', get_coefficient(operator_type, config))
        parameters['D'] = diffusion_coef
        parameters.setdefault('use_eigenvalue_decay', True)
        parameters.setdefault('decay_rate', diffusion_coef)
    
    elif operator_type == 'wave':
        wave_speed = parameters.get('c2', get_coefficient(operator_type, config))
        parameters['c2'] = wave_speed
        parameters['use_sqrt_c2'] = parameters.get('use_sqrt_c2', True)
        parameters.setdefault('direction', '+')
        # For wave equations, always use π-based wavenumbers by default
        parameters.setdefault('use_pi_wavenumbers', True)
    
    elif operator_type == 'advection':
        velocity = parameters.get('velocity', get_coefficient(operator_type, config))
        parameters['velocity'] = velocity * parameters.get('velocity_factor', 1.0)
    
    elif operator_type == 'reaction-diffusion':
        if isinstance(get_coefficient(operator_type, config), tuple):
            D_val, k_val = get_coefficient(operator_type, config)
            parameters['D'] = D_val
            parameters['k'] = k_val
        else:
            parameters['D'] = parameters.get('D', 0.5)
            parameters['k'] = parameters.get('k', 0.5)
        
        parameters.setdefault('use_eigenvalue_decay', True)
        parameters.setdefault('decay_rate', parameters['D'])
    
    elif operator_type == 'burgers':
        nu_val = parameters.get('nu', get_coefficient(operator_type, config))
        parameters['nu'] = nu_val
        parameters.setdefault('use_shock_solution', True)
        parameters.setdefault('shock_speed', 1.0)
        parameters.setdefault('shock_width', 2 * nu_val)

    elif operator_type == 'inv_burgers':
        nu_val = parameters.get('nu', get_coefficient(operator_type, config))
        parameters['nu'] = nu_val
        parameters.setdefault('use_shock_solution', True)
        parameters.setdefault('shock_speed', 1.0)
        parameters.setdefault('shock_width', 2 * nu_val)
    
    elif operator_type == 'navier-stokes':
        nu_val = parameters.get('nu', get_coefficient(operator_type, config))
        parameters['nu'] = nu_val
    
    elif operator_type == 'helmholtz':
        k_val = parameters.get('k', get_coefficient(operator_type, config))
        parameters['k'] = k_val
    
    elif operator_type == 'biharmonic':
        alpha_val = parameters.get('alpha', get_coefficient(operator_type, config))
        parameters['alpha'] = alpha_val
        parameters['time_dependent'] = parameters.get('time_dependent', True)
    
    elif operator_type == 'laplacian':
        coef_val = parameters.get('coef', get_coefficient(operator_type, config))
        parameters['coef'] = coef_val
    
    
    
    # 1. Generate initial condition
    ic_expr = generate_initial_condition(parameters.get('ic_type', 'sine'), dims, parameters)
    
    # 2. Apply appropriate operator to the initial condition
    operator_function = OPERATOR_BEHAVIORS.get(operator_type)
    
    if operator_function:
        return operator_function(ic_expr, dims, parameters)
    else:
        # For unknown operators, just return the initial condition
        return ic_expr

def compute_forcing_term(u_str, L_sympy_str, dims):
    """
    Compute the forcing term f = L[u] using SymPy.
    If 'tanh' is present in the manufactured solution, only the derivatives are computed.
    Otherwise, additional simplification steps are applied to yield a shorter expression.
    
    Args:
        u_str (str): Manufactured solution expression.
        L_sympy_str (str): Operator expression in SymPy format.
        dims (int): Spatial dimension.
        
    Returns:
        str: Forcing term expression with controlled precision.
    """
    try:
        import sympy as sp
        import re

        t = sp.symbols('t')
        pi = sp.pi

        # Set up symbols based on spatial dimension
        if dims == 1:
            x = sp.symbols('x')
            local_dict = {
                'x': x, 't': t, 'pi': pi, 
                'sin': sp.sin, 'cos': sp.cos, 
                'exp': sp.exp, 'tanh': sp.tanh, 
                'sech': sp.sech, 'cosh': sp.cosh, 'sqrt': sp.sqrt
            }
            vars_list = [x, t]
        elif dims == 2:
            x, y = sp.symbols('x y')
            local_dict = {
                'x': x, 'y': y, 't': t, 'pi': pi,
                'sin': sp.sin, 'cos': sp.cos, 
                'exp': sp.exp, 'tanh': sp.tanh, 
                'sech': sp.sech, 'cosh': sp.cosh, 'sqrt': sp.sqrt
            }
            vars_list = [x, y, t]
        elif dims == 3:
            x, y, z = sp.symbols('x y z')
            local_dict = {
                'x': x, 'y': y, 'z': z, 't': t, 'pi': pi,
                'sin': sp.sin, 'cos': sp.cos, 
                'exp': sp.exp, 'tanh': sp.tanh, 
                'sech': sp.sech, 'cosh': sp.cosh, 'sqrt': sp.sqrt
            }
            vars_list = [x, y, z, t]

        # Check if the manufactured solution contains 'tanh'
        contains_tanh = 'tanh' in u_str

        # Parse the manufactured solution
        u_expr = sp.sympify(u_str, locals=local_dict)

        # Define u as a function of the variables
        u = sp.Function('u')(*vars_list)

        # Parse the operator expression; add diff and u to the namespace
        local_dict_L = dict(local_dict)
        local_dict_L.update({'diff': sp.diff, 'u': u})
        L_expr = sp.sympify(L_sympy_str, locals=local_dict_L)

        # Substitute the manufactured solution into the operator expression
        f_expr = L_expr.subs({u: u_expr})

        if not contains_tanh:
            # Additional simplification to produce a shorter expression:
            f_expr = sp.cancel(f_expr)        # Cancel common factors in fractions
            f_expr = sp.together(f_expr)        # Combine terms into a single fraction where possible
            f_expr = sp.factor_terms(f_expr)    # Factor out common terms from the expression
            f_expr = sp.simplify(f_expr)          # Apply heuristic simplifications
        else:
            print("tanh detected: computing derivatives only, skipping advanced simplification.")
            f_expr = f_expr.doit()
            f_expr = sp.collect(f_expr, t)
        # Convert the expression to string and round any floating point numbers
        f_str = str(f_expr)
        def round_reals(match):
            number = float(match.group(0))
            return "0" if number == 0 else str(round(number, 3))
        f_formatted = re.sub(r'[-+]?\d*\.\d+', round_reals, f_str)
        
        return f_formatted

    except Exception as e:
        print(f"Error computing forcing term: {e}")
        return "Failed to compute forcing term"

# =============================================================================
# OPERATOR STRING GENERATION
# =============================================================================

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
    separator = "*" if formatted else "*"
    
    if operator_type == 'diffusion':
        D_val = parameters.get('D', 0.5)
        if dims == 1:
            if formatted:
                return f"dt(u) - {D_val}*dxx(u)"
            else:
                return f"diff(u, t) - {D_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {D_val}*({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {D_val}*({laplace})"
    
    elif operator_type == 'wave':
        c2_val = parameters.get('c2', 1.0)
        if dims == 1:
            if formatted:
                return f"dtt(u) - {c2_val}*dxx(u)"
            else:
                return f"diff(u, t, 2) - {c2_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dtt(u) - {c2_val}*({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t, 2) - {c2_val}*({laplace})"
    
    elif operator_type == 'burgers':
        nu_val = parameters.get('nu', 0.05)
        if formatted:
            return f"dt(u) + u*dx(u) - {nu_val}*dxx(u)"
        else:
            return f"diff(u, t) + u*diff(u, x) - {nu_val}*diff(u, x, 2)"
    
    elif operator_type == 'inv_burgers':
        nu_val = 0.0
        if formatted:
            return f"dt(u) + u*dx(u) - {nu_val}*dxx(u)"
        else:
            return f"diff(u, t) + u*diff(u, x) - {nu_val}*diff(u, x, 2)"
    elif operator_type == 'allen_cahn':
        epsilon_squared = parameters.get('epsilon_squared', 1.0)
        if dims == 1:
            if formatted:
                return f"dt(u) - {epsilon_squared}*dxx(u) - u + u**3"
            else:
                return f"diff(u, t) - {epsilon_squared}*diff(u, x, 2) - u + u**3"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {epsilon_squared}*({laplace}) - u + u**3"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {epsilon_squared}*({laplace}) - u + u**3"

    elif operator_type == 'ginzburg_landau':
        alpha = parameters.get('alpha', 1.0)
        beta = parameters.get('beta', 1.0)
        if dims == 1:
            if formatted:
                return f"dt(u) - dxx(u) - {alpha}*u + {beta}*u**3"
            else:
                return f"diff(u, t) - diff(u, x, 2) - {alpha}*u + {beta}*u**3"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - ({laplace}) - {alpha}*u + {beta}*u**3"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - ({laplace}) - {alpha}*u + {beta}*u**3"
    elif operator_type == 'fisher_kpp':
        r_val = parameters.get('r', 1.0)
        D_val = parameters.get('D', 1.0)  # Get actual D value
        
        if dims == 1:
            if formatted:
                return f"dt(u) - {D_val}*dxx(u) + {r_val}*u*(1-u)"
            else:
                return f"diff(u, t) - {D_val}*diff(u, x, 2) + {r_val}*u*(1-u)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {D_val}*({laplace}) + {r_val}*u*(1-u)"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {D_val}*({laplace}) + {r_val}*u*(1-u)"


    elif operator_type == 'convection_diffusion':
        D_val = parameters.get('D', 0.5)
        velocity = parameters.get('velocity', 1.0)
        if dims == 1:
            if formatted:
                return f"dt(u) + {velocity}*dx(u) - {D_val}*dxx(u)"
            else:
                return f"diff(u, t) + {velocity}*diff(u, x) - {D_val}*diff(u, x, 2)"
        else:
            if formatted:
                advection = f"{velocity}*d{spatial_vars[0]}(u)"
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) + {advection} - {D_val}*({laplace})"
            else:
                advection = f"{velocity}*diff(u, {spatial_vars[0]})"
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) + {velocity}*diff(u, {spatial_vars[0]}) - {D_val}*({laplace})"

    elif operator_type == 'fitzhugh_nagumo':
        D_val = parameters.get('D', 0.5)
        v_val = parameters.get('v_value', 0.1)  # Add a v value parameter
        
        if dims == 1:
            if formatted:
                return f"dt(u) - {D_val}*dxx(u) - u + u**3 + {v_val}"
            else:
                return f"diff(u, t) - {D_val}*diff(u, x, 2) - u + u**3 + {v_val}"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {D_val}*({laplace}) - u + u**3 + {v_val}"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {D_val}*({laplace}) - u + u**3 + {v_val}"

    elif operator_type == 'sine_gordon':
        if dims == 1:
            c2 = parameters.get('c2', get_sample_range(parameters, 'coefficient_range', 1.0))
            wavenumbers = parameters.get('wavenumbers', [1])
            k = wavenumbers[0]
            domain_length = parameters.get('domain_length', 1.0)
            # Incorporate c2 in the string to vary it.
            if formatted:
                return f"dtt(u) - dxx(u) + sin(u)"
            else:
                return f"diff(u, t, 2) - diff(u, x, 2) + sin(u)"
        else:
            raise ValueError("Sine–Gordon operator is only defined in 1D.")


    elif operator_type == 'telegrapher':
        # Telegrapher: u_tt + 2α*u_t - c2*u_xx = 0 (1D only)
        if dims == 1:
            # Use the helper to sample from the range if available.
            alpha = parameters.get('alpha',get_sample_range(parameters, 'alpha_range', 0.5))
            c2 = parameters.get('c2',get_sample_range(parameters, 'c2_range', 1.0))
            if formatted:
                return f"dtt(u) + 2*{alpha}*dt(u) - {c2}*dxx(u)"
            else:
                return f"diff(u, t, 2) + 2*{alpha}*diff(u, t) - {c2}*diff(u, x, 2)"
        else:
            raise ValueError("Telegrapher operator is only defined in 1D.")
            
    elif operator_type == 'black_scholes':
        sigma = parameters.get('sigma', 0.2)
        r = parameters.get('r', 0.05)
        if dims == 1:
            if formatted:
                return f"dt(u) + 0.5*({sigma}**2)*x**2*dxx(u) + {r}*x*dx(u) - {r}*u"
            else:
                return f"diff(u, t) + 0.5*({sigma}**2)*x**2*diff(u, x, 2) + {r}*x*diff(u, x) - {r}*u"
        else:
            raise ValueError("Black–Scholes operator is defined only for 1D.")
    elif operator_type == 'tricomi':
        # Tricomi equation: u_xx + x u_yy = 0.
        spatial_vars = get_spatial_vars(dims)  # Should be ['x','y'] for 2D
        if dims == 2:
            if formatted:
                return "dxx(u) + x*dyy(u)"
            else:
                return "diff(u, x, 2) + x*diff(u, y, 2)"
        else:
            raise ValueError("Tricomi operator is defined only for 2D.")






    elif operator_type == 'reaction_diffusion':
        D_val = parameters.get('D', 0.5)
        k_val = parameters.get('k', 0.5)
        
        # If a reaction term is explicitly specified in the parameters, use it.
        # Otherwise, randomly choose from the desired options.
        if "reaction_term" in parameters:
            reaction_term = parameters["reaction_term"]
        else:
            reaction_terms = [
                "(sin(u))",
                "(cos(u))",
                "(exp(u))",
                "(u - u**2)"
            ]
            reaction_term = random.choice(reaction_terms)
        
        if dims == 1:
            if formatted:
                return f"dt(u) - {D_val}*dxx(u) + {k_val}*{reaction_term}"
            else:
                return f"diff(u, t) - {D_val}*diff(u, x, 2) + {k_val}*{reaction_term}"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) - {D_val}*({laplace}) + {k_val}*{reaction_term}"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) - {D_val}*({laplace}) + {k_val}*{reaction_term}"
    
    spatial_vars = get_spatial_vars(dims)
    


    
    if operator_type == 'navier-stokes':
        nu_val = parameters.get('nu', 0.05)
        if dims == 1:
            if formatted:
                return f"dt(u) + u*dx(u) - {nu_val}*dxx(u)"
            else:
                return f"diff(u, t) + u*diff(u, x) - {nu_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"dt(u) + u*dx(u) - {nu_val}*({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"diff(u, t) + u*diff(u, x) - {nu_val}*({laplace})"
    
    elif operator_type == 'advection':
        v_val = parameters.get('velocity', 1.0)
        if dims == 1:
            if formatted:
                return f"dt(u) + {v_val}*dx(u)"
            else:
                return f"diff(u, t) + {v_val}*diff(u, x)"
        else:
            if formatted:
                advection = " + ".join(f"{v_val}*d{var}(u)" for var in spatial_vars)
                return f"dt(u) + {advection}"
            else:
                advection = " + ".join(f"{v_val}*diff(u, {var})" for var in spatial_vars)
                return f"diff(u, t) + {advection}"
    
    elif operator_type == 'helmholtz':
        k_val = parameters.get('k', 1.0)
        if dims == 1:
            if formatted:
                return f"dxx(u) + {k_val}**2*u"  
            else:
                return f"diff(u, x, 2) + {k_val}**2*u" 
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"({laplace}) + {k_val}**2*u"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"({laplace}) + {k_val}**2*u"  

    elif operator_type == 'biharmonic':
        alpha_val = parameters.get('alpha', 0.05)
        if dims == 1:
            if formatted:
                return f"dt(u) + {alpha_val}*dxxxx(u)"
            else:
                return f"diff(u, t) + {alpha_val}*diff(u, x, 4)"
        elif dims == 2:
            if formatted:
                # More accurate representation of biharmonic in 2D
                return f"dt(u) + {alpha_val}*(dxxxx(u) + 2*dxxyy(u) + dyyyy(u))"
            else:
                # More accurate representation in sympy format
                return f"diff(u, t) + {alpha_val}*(diff(u, x, 4) + 2*diff(diff(u, x, 2), y, 2) + diff(u, y, 4))"
    
    elif operator_type == 'laplacian':
        coef_val = parameters.get('coef', 1.0)
        if dims == 1:
            if formatted:
                return f"{coef_val}*dxx(u)"
            else:
                return f"{coef_val}*diff(u, x, 2)"
        else:
            if formatted:
                laplace = " + ".join(f"d{var}{var}(u)" for var in spatial_vars)
                return f"{coef_val}*({laplace})"
            else:
                laplace = " + ".join(f"diff(u, {var}, 2)" for var in spatial_vars)
                return f"{coef_val}*({laplace})"
    
    return f"L[u] for {operator_type} in {dims}D"


# =============================================================================
# TRIPLET GENERATION
# =============================================================================
# Global set to track used operators
USED_OPERATORS = set()

def generate_triplet(operator_type, dims, solution_type=None, solution_params=None, config=None):
    """
    Generate a triplet (u_str, L_str, f_str) for the given operator and dimension.
    
    This function creates a physically appropriate solution for the given operator,
    with increased parameter diversity to generate a range of different cases.
    
    Args:
        operator_type (str): Type of differential operator
        dims (int): Spatial dimension (1, 2, or 3)
        solution_type (str, optional): Type of solution
        solution_params (dict, optional): Solution parameters
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: (solution string, operator string, forcing term string)
    """
    global USED_OPERATORS
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
        solution_params = get_solution_params(operator_type, solution_type, config)
    
    # Check for parameter variation settings
    param_variation = config.get('dataset', {}).get('parameter_variation', {})
    vary_coefficients = param_variation.get('vary_coefficients', True)
    vary_wavenumbers = param_variation.get('vary_wavenumbers', True)
    vary_centers = param_variation.get('vary_centers', True)
    vary_widths = param_variation.get('vary_widths', True)
    vary_polynomial_terms = param_variation.get('vary_polynomial_terms', True)
    
    parameters = {}
    randomized_params = dict(solution_params)
    
    # Add randomization to wavenumbers, centers, widths, polynomial terms (unchanged from your code)
    if vary_wavenumbers and 'ic_type' in solution_params and solution_params['ic_type'] == 'sine':
        wavenumbers = solution_params.get('wavenumbers', [1] * dims)
        if random.random() < 0.3 and not solution_params.get('fixed_wavenumbers', False):
            randomized_wavenumbers = []
            for k in wavenumbers:
                adjusted_k = max(1, int(k * random.uniform(0.8, 1.2)))
                randomized_wavenumbers.append(adjusted_k)
            randomized_params['wavenumbers'] = randomized_wavenumbers
    
    if vary_centers and 'ic_type' in solution_params and solution_params['ic_type'] in ['gaussian', 'step', 'tanh']:
        centers = solution_params.get('centers', [0.5] * dims)
        if random.random() < 0.3 and not solution_params.get('fixed_centers', False):
            randomized_centers = [round(random.uniform(0.3, 0.7), 2) for _ in centers]
            randomized_params['centers'] = randomized_centers
    
    if vary_widths and 'ic_type' in solution_params and solution_params['ic_type'] in ['gaussian', 'step', 'tanh']:
        width = solution_params.get('width', 0.1)
        sigma = solution_params.get('sigma', 0.1)
        if 'width' in solution_params and random.random() < 0.3 and not solution_params.get('fixed_width', False):
            randomized_params['width'] = round(random.uniform(0.05, 0.2), 2)
        if 'sigma' in solution_params and random.random() < 0.3 and not solution_params.get('fixed_sigma', False):
            randomized_params['sigma'] = round(random.uniform(0.05, 0.2), 2)
    
    if vary_polynomial_terms and 'ic_type' in solution_params and solution_params['ic_type'] == 'polynomial':
        if isinstance(solution_params.get('poly_degree'), list):
            randomized_params['poly_degree'] = random.choice(solution_params['poly_degree'])
        if 'coefficient_templates' in solution_params:
            randomized_params['coefficient_template'] = random.choice(solution_params['coefficient_templates'])
            if 'coefficient_ranges' in solution_params:
                coef_values = {k: round(random.uniform(v_range[0], v_range[1]), 2) for k, v_range in solution_params['coefficient_ranges'].items()}
                randomized_params['coefficient_values'] = coef_values
    
    if operator_type == 'wave' and random.random() < 0.4:
        wave_forms = ['traveling', 'standing', 'damped', 'superposition']
        form_weights = [0.4, 0.3, 0.2, 0.1]
        randomized_params['wave_form'] = random.choices(wave_forms, weights=form_weights, k=1)[0]
        if randomized_params['wave_form'] == 'traveling':
            randomized_params['direction'] = random.choice(['+', '-'])
        if randomized_params['wave_form'] == 'damped':
            randomized_params['damping_coefficient'] = round(random.uniform(0.05, 0.2), 2)
    
    # Ensure unique operator by varying coefficients and checking against USED_OPERATORS
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        if vary_coefficients:
            if operator_type == 'reaction_diffusion':
                D_val, k_val = get_coefficient(operator_type, config)
                # Add small perturbation for uniqueness
                D_val = round(D_val + random.uniform(-0.01, 0.01), 3)
                k_val = round(k_val + random.uniform(-0.01, 0.01), 3)
                parameters['D'] = D_val
                parameters['k'] = k_val
                if random.random() < 0.3:
                    D_val = round(D_val * random.uniform(0.8, 1.2), 3)
                    k_val = round(k_val * random.uniform(0.8, 1.2), 3)
                    parameters['D'] = D_val
                    parameters['k'] = k_val
                if solution_type == 'exp_decay':
                    decay_factor = randomized_params.get("decay_rate_factor", 1.0)
                    parameters['decay_rate'] = k_val * decay_factor
            elif operator_type == 'ginzburg_landau':
                print("DEBUG: Starting ginzburg_landau handling")
                try:
                    # Get coefficient value
                    print("DEBUG: About to get coefficient")
                    coef_value = get_coefficient(operator_type, config)
                    print(f"DEBUG: Got coefficient: {coef_value} of type {type(coef_value)}")
                    
                    # Try to handle the coefficient safely
                    if isinstance(coef_value, (tuple, list)) and len(coef_value) >= 2:
                        alpha = float(coef_value[0])
                        beta = float(coef_value[1])
                        print(f"DEBUG: Extracted alpha={alpha}, beta={beta}")
                    else:
                        print(f"DEBUG: Invalid coefficient format: {coef_value}")
                        alpha = 1.0
                        beta = 1.0
                        
                    print("DEBUG: About to apply variations")
                    # Now add variations
                    if vary_coefficients:
                        try:
                            alpha = round(alpha + random.uniform(-0.01, 0.01), 3)
                            beta = round(beta + random.uniform(-0.01, 0.01), 3)
                            print(f"DEBUG: After perturbation: alpha={alpha}, beta={beta}")
                            
                            if random.random() < 0.3:
                                alpha = round(alpha * random.uniform(0.8, 1.2), 3)
                                beta = round(beta * random.uniform(0.8, 1.2), 3)
                                print(f"DEBUG: After scaling: alpha={alpha}, beta={beta}")
                        except Exception as e:
                            print(f"DEBUG: Error during perturbation: {e}")
                            
                    # Store parameters individually
                    parameters['alpha'] = alpha
                    parameters['beta'] = beta
                    parameters['diffusion_coefficient'] = 1.0
                    print("DEBUG: Parameters set successfully")
                except Exception as e:
                    import traceback
                    print(f"DEBUG: Error in ginzburg_landau handling: {e}")
                    print(f"DEBUG: Traceback: {traceback.format_exc()}")
                    # Use default values
                    parameters['alpha'] = 1.0
                    parameters['beta'] = 1.0
                    parameters['diffusion_coefficient'] = 1.0
            elif operator_type == 'allen_cahn':
                epsilon_squared = get_coefficient(operator_type, config)
                epsilon_squared = round(epsilon_squared + random.uniform(-0.01, 0.01), 3)
                if random.random() < 0.3:
                    epsilon_squared = round(epsilon_squared * random.uniform(0.8, 1.2), 3)
                parameters['epsilon_squared'] = epsilon_squared
                
                # Handle solution parameters for standard solution types
                if solution_type == 'exp_decay':
                    parameters['use_eigenvalue_decay'] = randomized_params.get('use_eigenvalue_decay', True)
                    decay_factor = randomized_params.get("decay_rate_factor", 1.0)
                    parameters['decay_rate'] = epsilon_squared * decay_factor
                elif solution_type == 'gaussian_decay':
                    parameters['sigma'] = randomized_params.get('sigma', 0.1)
                elif solution_type == 'polynomial_decay':
                    # Handle polynomial templates
                    if 'coefficient_templates' in randomized_params:
                        parameters['coefficient_template'] = random.choice(randomized_params['coefficient_templates'])
            else:
                # Handle all other operators
                coef = get_coefficient(operator_type, config)
                
                # Check type and handle accordingly for arithmetic
                if isinstance(coef, (tuple, list)):
                    # Handle tuple - extract components and operate on them individually
                    if len(coef) >= 2:
                        alpha, beta = coef
                        alpha = round(alpha + random.uniform(-0.01, 0.01), 3)
                        beta = round(beta + random.uniform(-0.01, 0.01), 3)
                        
                        if random.random() < 0.3:
                            alpha = round(alpha * random.uniform(0.8, 1.2), 3)
                            beta = round(beta * random.uniform(0.8, 1.2), 3)
                            
                        # For other operators that might return tuples
                        coef = (alpha, beta)
                    else:
                        # Just in case we get a tuple/list with only one element
                        coef = float(coef[0])
                        coef = round(coef + random.uniform(-0.01, 0.01), 3)
                        
                        if random.random() < 0.3:
                            coef = round(coef * random.uniform(0.8, 1.2), 3)
                else:
                    # It's a scalar, can do arithmetic normally
                    coef = round(coef + random.uniform(-0.01, 0.01), 3)
                    
                    if random.random() < 0.3:
                        coef = round(coef * random.uniform(0.8, 1.2), 3)
                
                # Now use the coef value for the appropriate operator
                if operator_type == 'diffusion':
                    parameters['D'] = coef
                    if solution_type == 'exp_decay':
                        decay_factor = randomized_params.get("decay_rate_factor", 1.0)
                        parameters['decay_rate'] = coef * decay_factor
                elif operator_type == 'wave':
                    parameters['c2'] = coef
                    parameters['use_sqrt_c2'] = randomized_params.get("use_sqrt_c2", True)
                    parameters['use_pi_wavenumbers'] = randomized_params.get('use_pi_wavenumbers', True)
                elif operator_type == 'burgers':
                    parameters['nu'] = coef
                    parameters['velocity'] = randomized_params.get("velocity", 1.0)
                    if 'shock_width' in randomized_params:
                        shock_width = randomized_params['shock_width']
                        shock_width = round(shock_width * random.uniform(0.8, 1.2), 2)
                        randomized_params['shock_width'] = shock_width
                elif operator_type == 'inv_burgers':
                    parameters['nu'] = coef
                    parameters['velocity'] = randomized_params.get("velocity", 1.0)
                    if 'shock_width' in randomized_params:
                        shock_width = randomized_params['shock_width']
                        shock_width = round(shock_width * random.uniform(0.8, 1.2), 2)
                        randomized_params['shock_width'] = shock_width
                elif operator_type == 'navier-stokes':
                    parameters['nu'] = coef
                elif operator_type == 'advection':
                    velocity = coef * randomized_params.get("velocity_factor", 1.0)
                    if random.random() < 0.3:
                        velocity = round(velocity * random.uniform(0.8, 1.2), 2)
                    parameters['velocity'] = velocity
                elif operator_type == 'helmholtz':
                    parameters['k'] = coef
                elif operator_type == 'biharmonic':
                    parameters['alpha'] = coef
                    parameters['time_dependent'] = randomized_params.get('time_dependent', True)
                elif operator_type == 'laplacian':
                    parameters['coef'] = coef
        # Add all solution parameters to the parameters dict
        for key, value in randomized_params.items():
            if key != "type":
                parameters[key] = value
        
        # Generate the operator string
        L_formatted = generate_operator_string(operator_type, dims, parameters, True)
        
        # Check for uniqueness
        if L_formatted not in USED_OPERATORS:
            USED_OPERATORS.add(L_formatted)
            break
        
        attempt += 1
    else:
        raise ValueError(f"Could not generate a unique operator for {operator_type} in {dims}D after {max_attempts} attempts")
    
    # Generate the manufactured solution
    u_str = generate_manufactured_solution_string(operator_type, dims, solution_type, parameters)
    

    # Generate the sympy-compatible operator string for computation
    L_sympy_str = generate_operator_string(operator_type, dims, parameters, False)
    
    # Compute the forcing term
    f_str = compute_forcing_term(u_str, L_sympy_str, dims)

    # Simplify the solution string for more compact representation
    u_str = sp.simplify(u_str)
    # Round all numerical values to 3 decimal places
    u_str = str(u_str)
    u_str = re.sub(r'(\d+\.\d{3})\d+', r'\1', u_str) 
    
    return u_str, L_formatted, f_str


def get_enhanced_polynomial_params(params, config):
    """Extract enhanced polynomial parameters from configuration."""
    result = {}
    
    # Handle polynomial degree
    if isinstance(params.get('poly_degree'), list):
        result['poly_degree'] = random.choice(params['poly_degree'])
    else:
        result['poly_degree'] = params.get('poly_degree', 2)
    
    # Handle polynomial forms
    if 'polynomial_forms' in params:
        result['polynomial_form'] = random.choice(params['polynomial_forms'])
    
    # Handle coefficient ranges
    if 'coefficient_ranges' in params:
        ranges = params['coefficient_ranges']
        result['coefficient_values'] = {
            k: random.uniform(v[0], v[1]) for k, v in ranges.items()
        }
    
    # Handle coefficient templates
    if 'coefficient_templates' in params:
        result['coefficient_template'] = random.choice(params['coefficient_templates'])
    
    # Handle mixed terms
    result['include_mixed_terms'] = params.get('include_mixed_terms', False)
    
    return result

def comprehensive_dataset_worker(item):
    """Worker function for processing comprehensive dataset items."""
    op_type, dims, sol_type, rep_idx, config = item
    try:
        # Get solution parameters
        sol_params = get_solution_params(op_type, sol_type, config)
        
        # Add some variation for different repetitions of the same combination
        if rep_idx > 0:
            # Add slight variation to parameters for different repetitions
            sol_params = dict(sol_params)  # Make a copy
            
            # Vary coefficients slightly based on repetition index
            if 'coefficient' in sol_params:
                base_coef = sol_params['coefficient']
                # Vary by up to ±10% based on repetition
                variation = 0.9 + 0.2 * (rep_idx / max(1, num_per_operator))
                sol_params['coefficient'] = base_coef * variation
        
        # Generate the triplet
        u_str, L_formatted, f_str = generate_triplet(op_type, dims, sol_type, sol_params, config)
        return [op_type, dims, sol_type, u_str, L_formatted, f_str]
    except Exception as e:
        print(f"Error generating {op_type}, dims={dims}, sol={sol_type}, rep={rep_idx}: {e}", file=sys.stderr)
        return None

def generate_comprehensive_dataset(output_csv='comprehensive_triplets.csv', dims_set=None, config=None, num_per_operator=1):

    """
    Generates a comprehensive dataset with all operators, dimensions, and solution types using parallelization.
    
    Args:
        output_csv (str): Path to output CSV file
        dims_set (list, optional): List of dimensions to include
        config (dict, optional): Configuration dictionary
        num_per_operator (int): Number of examples per operator-dimension-solution combination
        
    Returns:
        int: Number of triplets generated
    """
    print(f"DEBUG: num_per_operator in comprehensive = {num_per_operator}, type = {type(num_per_operator)}")
    global USED_OPERATORS
    USED_OPERATORS.clear()
    
    if config is None:
        config = load_config()
    
    # Get all operator types from config
    operator_types = get_all_operator_types(config)
    
    # Prepare work items
    work_items = []
    for op_type in operator_types:
        valid_dims = get_valid_dimensions(op_type, config)
        if dims_set is not None:
            valid_dims = list(set(dims_set).intersection(valid_dims))
        
        if not valid_dims:
            continue  # Skip if no valid dimensions
        
        solution_types = get_all_solution_types(op_type, config)
        
        # Create a work item for each combination and repetition
        for dims in valid_dims:
            for sol_type in solution_types:
                for i in range(num_per_operator):
                    work_items.append((op_type, dims, sol_type, i, config))
    
    # Execute in parallel
    num_cpus = min(cpu_count(), len(work_items))
    print(f"Using {num_cpus} CPU cores for parallel processing of {len(work_items)} total items")
    rows = []
    
    with Pool(processes=num_cpus) as pool:
        with tqdm(total=len(work_items), desc="Generating comprehensive dataset") as pbar:
            for result in pool.imap_unordered(comprehensive_dataset_worker, work_items):
                if result:
                    rows.append(result)
                pbar.update(1)
    
    # Save to CSV
    headers = ['operator_type', 'spatial_dimension', 'solution_type', 'manufactured_solution_u', 'operator_L', 'forcing_term_f']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Comprehensive dataset saved to {output_csv} with {len(rows)} triplets")
    return len(rows)

def generate_random_dataset(output_csv='random_triplets.csv', num_per_operator=50, operators=None, 
                           dimensions=None, solution_types=None, config=None):
    """
    Generates a random dataset with specified number of examples per operator using multiprocessing.
    """
    print(f"DEBUG: num_per_operator = {num_per_operator}, type = {type(num_per_operator)}")
    
    if config is None:
        config = load_config()
    
    if operators is None:
        operators = get_all_operator_types(config)
    
    # Prepare work items
    work_items = []
    for op_type in operators:
        valid_dims = get_valid_dimensions(op_type, config)
        if dimensions is not None:
            valid_dims = list(set(dimensions).intersection(valid_dims))
        if not valid_dims:
            continue
            
        valid_sol_types = get_all_solution_types(op_type, config)
        if solution_types is not None:
            valid_sol_types = list(set(solution_types).intersection(valid_sol_types))
        if not valid_sol_types:
            continue
        
        # Create work items for this operator
        for i in range(num_per_operator):
            work_items.append((op_type, valid_dims, valid_sol_types, config))
    
    # Define worker function
    def worker(item):
        op_type, valid_dims, valid_sol_types, config = item
        try:
            dims = random.choice(valid_dims)
            sol_type = random.choice(valid_sol_types)
            sol_params = get_solution_params(op_type, sol_type, config)
            
            u_str, L_formatted, f_str = generate_triplet(op_type, dims, sol_type, sol_params, config)
            
            return [op_type, dims, sol_type, u_str, L_formatted, f_str]
        except Exception as e:
            print(f"Error generating triplet for {op_type}: {e}")
            return None
    
    # Execute in parallel
    num_cpus = min(cpu_count(), len(work_items))
    rows = []
    
    with Pool(processes=num_cpus) as pool:
        with tqdm(total=len(work_items), desc="Generating random dataset") as pbar:
            for result in pool.imap_unordered(worker, work_items):
                if result:
                    rows.append(result)
                pbar.update(1)
    
    # Save results
    headers = ['operator_type', 'spatial_dimension', 'solution_type', 'manufactured_solution_u', 'operator_L', 'forcing_term_f']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Random dataset saved to {output_csv} with {len(rows)} triplets")
    return len(rows)
# =============================================================================
# COMMAND-LINE FUNCTIONALITY
# =============================================================================

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
    
    parser.add_argument('--num-per-operator', type=int, default=30,
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
                config=config,
                num_per_operator=args.num_per_operator
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