import random
from config_dt_helpers import get_spatial_vars
import math
from config_dt_helpers import get_sample_range
import random


# =============================================================================
# INITIAL CONDITION FUNCTIONS
# =============================================================================

def generate_sine_ic(spatial_vars, params):
    dims = len(spatial_vars)
    k_vals = params.get('wavenumbers', [1] * dims)
    if len(k_vals) < dims:
        k_vals.extend([1] * (dims - len(k_vals)))
    
    use_pi = params.get('use_pi_wavenumbers', True)  # Default to True
    domain_length = params.get('domain_length', 1.0)
    
    # Use consistent pi format
    if use_pi:
        k_terms = [f"(pi*{k}/{domain_length})" for k in k_vals]
    else:
        k_terms = [str(k) for k in k_vals]
    
    # Create sine terms
    if dims == 1:
        return f"sin({k_terms[0]}*{spatial_vars[0]})"
    else:
        return "*".join(f"sin({k_terms[i]}*{var})" for i, var in enumerate(spatial_vars))

def generate_gaussian_ic(spatial_vars, params):
    dims = len(spatial_vars)
    sigma = round(params.get('sigma', random.uniform(0.2, 0.5)), 3)  # YAML range [0.2, 0.5]
    centers = params.get('centers', [0.5] * dims)
    if len(centers) < dims:
        centers.extend([0.5] * (dims - len(centers)))
    centers = [round(c, 3) for c in centers]
    if dims == 1:
        return f"exp(-((({spatial_vars[0]}-{centers[0]}))**2)/{sigma})"
    else:
        terms = [f"(({var}-{centers[i]})**2)" for i, var in enumerate(spatial_vars)]
        sum_terms = "+".join(terms)
        return f"exp(-({sum_terms})/{sigma})"

def generate_polynomial_ic(spatial_vars, params):
    dims = len(spatial_vars)
    if isinstance(params.get('poly_degree'), list):
        degree = random.choice(params['poly_degree'])
    else:
        degree = params.get('poly_degree', 2)
    if 'polynomial_form' in params:
        poly_form = params['polynomial_form']
        coef_values = params.get('coefficient_values', {'a': 1.0, 'b': 1.0, 'c': 1.0})
        for var_idx, var in enumerate(spatial_vars):
            var_letter = 'xyz'[var_idx]
            poly_form = poly_form.replace(var_letter, var)
        for coef_name, coef_val in coef_values.items():
            poly_form = poly_form.replace(coef_name, str(round(coef_val, 3)))
        return poly_form
    if 'coefficient_templates' in params:
        template = random.choice(params['coefficient_templates'])
        coef_ranges = params.get('coefficient_ranges', {'a': [0.5, 2.0], 'b': [-1.0, 1.0]})
        coef_values = {k: round(random.uniform(v_range[0], v_range[1]), 3) for k, v_range in coef_ranges.items()}
        for var_idx, var in enumerate(spatial_vars):
            var_letter = 'xyz'[var_idx]
            template = template.replace(var_letter, var)
        for coef_name, coef_val in coef_values.items():
            template = template.replace(coef_name, str(coef_val))
        return template
    include_mixed = params.get('include_mixed_terms', False)
    if dims > 1 and include_mixed:
        if dims == 2:
            x, y = spatial_vars
            mixed_term = f"+ {round(random.uniform(-2, 2), 3)}*{x}*{y}"
            return f"{x}**{degree}*(1-{x})**{degree} + {y}**{degree}*(1-{y})**{degree} {mixed_term}"
        elif dims == 3:
            x, y, z = spatial_vars
            mixed_terms = [
                f"+ {round(random.uniform(-2, 2), 3)}*{x}*{y}",
                f"+ {round(random.uniform(-2, 2), 3)}*{y}*{z}",
                f"+ {round(random.uniform(-2, 2), 3)}*{x}*{z}"
            ]
            return f"{x}**{degree}*(1-{x})**{degree} + {y}**{degree}*(1-{y})**{degree} + {z}**{degree}*(1-{z})**{degree} {''.join(mixed_terms)}"
    if dims == 1:
        return f"{spatial_vars[0]}**{degree}*(1-{spatial_vars[0]})**{degree}"
    else:
        return "*".join(f"{var}**{degree}*(1-{var})**{degree}" for var in spatial_vars)

def generate_step_ic(spatial_vars, params):
    dims = len(spatial_vars)
    width = round(params.get('width', random.uniform(0.2, 0.5)), 3)  # YAML range [0.2, 0.5]
    centers = params.get('centers', [0.5] * dims)
    if len(centers) < dims:
        centers.extend([0.5] * (dims - len(centers)))
    centers = [round(c, 3) for c in centers]
    if dims == 1:
        return f"0.5*(1+tanh((({spatial_vars[0]}-{centers[0]}))/{width}))"
    else:
        terms = [f"(({var}-{centers[i]}))/{width}" for i, var in enumerate(spatial_vars)]
        sum_terms = "+".join(terms)
        return f"0.5*(1+tanh({sum_terms}))"

def generate_tanh_ic(spatial_vars, params):
    dims = len(spatial_vars)
    width = round(params.get('width', random.uniform(0.2, 0.5)), 3)  # YAML range [0.2, 0.5]
    centers = params.get('centers', [0.5] * dims)
    if len(centers) < dims:
        centers.extend([0.5] * (dims - len(centers)))
    centers = [round(c, 3) for c in centers]
    if dims == 1:
        return f"tanh((({spatial_vars[0]}-{centers[0]}))/{width})"
    else:
        terms = [f"(({var}-{centers[i]}))/{width}" for i, var in enumerate(spatial_vars)]
        sum_terms = "+".join(terms)
        return f"tanh({sum_terms})"

# =============================================================================
# OPERATOR BEHAVIOR FUNCTIONS
# =============================================================================
PERTURBATION_PROBABILITY = 0.3

def apply_wave_operator(ic_expr, dims, parameters):
    parameters.setdefault('use_pi_wavenumbers', True)  # Changed to True per YAML
    c2 = round(parameters.get('c2', random.uniform(0.5, 1.0)), 3)  # YAML range [0.5, 1.0]
    c = f"sqrt({c2})" if parameters.get('use_sqrt_c2', True) else f"{c2}"
    wave_form = parameters.get('wave_form', 'traveling')
    domain_length = parameters.get('domain_length', 1.0)
    spatial_vars = get_spatial_vars(dims)
    
    if wave_form == 'traveling':
        direction = parameters.get('direction', '+')
        if parameters.get('ic_type', 'sine') == 'sine':
            wavenumbers = parameters.get('wavenumbers', [1])
            if dims == 1:
                k_str = f"{wavenumbers[0]}*pi/{domain_length}"
                base_solution = f"sin({k_str}*(x{direction}{c}*t))"
            else:
                terms = []
                for i, var in enumerate(spatial_vars):
                    k_idx = min(i, len(wavenumbers)-1)
                    k_str = f"{wavenumbers[k_idx]}*pi/{domain_length}"
                    if i == 0:
                        terms.append(f"sin({k_str}*({var}{direction}{c}*t))")
                    else:
                        terms.append(f"sin({k_str}*{var})")
                base_solution = "*".join(terms)
        else:
            modified_ic = ic_expr
            for var in spatial_vars:
                if var == 'x':
                    modified_ic = modified_ic.replace(var, f"({var}{direction}{c}*t)")
            base_solution = modified_ic
    elif wave_form == 'standing':
        if parameters.get('ic_type', 'sine') == 'sine':
            wavenumbers = parameters.get('wavenumbers', [1])
            k_squared_terms = [f"({k}*pi/{domain_length})**2" for i, k in enumerate(wavenumbers[:dims])]
            k_squared = "+".join(k_squared_terms) if k_squared_terms else f"(pi/{domain_length})**2"
            omega = f"{c}*sqrt({k_squared})"
            base_solution = f"({ic_expr})*cos({omega}*t)"
        else:
            omega = f"{c}*pi/{domain_length}"
            base_solution = f"({ic_expr})*cos({omega}*t)"
    elif wave_form == 'damped':
        damping = round(parameters.get('damping_coefficient', random.uniform(0.1, 0.2)), 3)
        if parameters.get('ic_type', 'sine') == 'sine':
            wavenumbers = parameters.get('wavenumbers', [1])
            k_squared_terms = [f"({k}*pi/{domain_length})**2" for i, k in enumerate(wavenumbers[:dims])]
            k_squared = "+".join(k_squared_terms) if k_squared_terms else f"(pi/{domain_length})**2"
            omega = f"{c}*sqrt({k_squared})"
        else:
            omega = f"{c}*pi/{domain_length}"
        base_solution = f"({ic_expr})*cos({omega}*t)*exp(-{damping}*t)"
    elif wave_form == 'superposition':
        modes = parameters.get('modes', [1, 2, 3])
        amplitudes = parameters.get('amplitudes', [1.0, 0.5, 0.25])
        if len(amplitudes) < len(modes):
            amplitudes.extend([0.1] * (len(modes) - len(amplitudes)))
        amplitudes = [round(a, 3) for a in amplitudes]
        terms = []
        for i, mode in enumerate(modes):
            if i < len(amplitudes):
                amp = amplitudes[i]
                k = f"{mode}*pi/{domain_length}"
                omega = f"{c}*{k}"
                if parameters.get('superposition_type', 'standing') == 'standing':
                    terms.append(f"{amp}*sin({k}*x)*cos({omega}*t)")
                else:
                    direction = parameters.get('direction', '+')
                    terms.append(f"{amp}*sin({k}*(x{direction}{c}*t))")
        base_solution = " + ".join(terms)
    else:
        direction = parameters.get('direction', '+')
        if parameters.get('ic_type', 'sine') == 'sine':
            wavenumbers = parameters.get('wavenumbers', [1])
            k_str = f"{wavenumbers[0]}*pi/{domain_length}"
            base_solution = f"sin({k_str}*(x{direction}{c}*t))"
        else:
            modified_ic = ic_expr
            for var in spatial_vars:
                if var == 'x':
                    modified_ic = modified_ic.replace(var, f"({var}{direction}{c}*t)")
            base_solution = modified_ic
    if random.random() < PERTURBATION_PROBABILITY:
        strength = round(random.uniform(0.1, 0.5), 3)  # Adjusted per your preference
        k = random.choice([1, 2])
        omega = random.choice([1, 2])
        form = random.choice(["sin", "cos"])
        perturbation = f" + {strength}*{form}({k}*pi*x)*{form}({omega}*t)"
        return f"{base_solution}{perturbation}"
    return base_solution


def apply_diffusion_operator(ic_expr, dims, parameters):

    D = round(parameters.get('D', random.uniform(0.1, 0.5)), 3)
    
    # 50% chance to create exact or non-exact solution
    create_exact = random.random() < 0.5
    
    if parameters.get('ic_type', 'sine') == 'sine' and parameters.get('use_eigenvalue_decay', True):
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        domain_length = parameters.get('domain_length', 1.0)
        
        # Use consistent pi format
        k_squared_terms = [f"(pi*{k}/{domain_length})**2" for k in wavenumbers[:dims]]
        k_squared = "+".join(k_squared_terms) if k_squared_terms else "1"
        
        if create_exact:
            # Exact solution - uses same coefficient, will produce zero forcing
            base_solution = f"({ic_expr})*exp(-{D}*({k_squared})*t)"
        else:
            # Non-exact solution - perturb coefficient by ±30%
            perturbation = random.uniform(0.7, 1.3)
            D_perturbed = round(D * perturbation, 3)
            base_solution = f"({ic_expr})*exp(-{D_perturbed}*({k_squared})*t)"
    else:
        decay_rate = round(parameters.get('decay_rate', D), 3)
        
        if create_exact:
            # Exact solution matches the PDE
            base_solution = f"({ic_expr})*exp(-{decay_rate}*t)"
        else:
            # Non-exact solution uses different decay rate
            decay_perturbed = round(decay_rate * random.uniform(0.7, 1.3), 3)
            base_solution = f"({ic_expr})*exp(-{decay_perturbed}*t)"
    
    # Optionally add perturbation for non-exact solutions
    if random.random() < 0.3 and not create_exact:
        strength = round(random.uniform(0.1, 0.5), 3)
        k = random.choice([1, 2])
        perturbation = f" + {strength}*sin({k}*pi*x)*exp(-{D}*t)"
        return f"{base_solution}{perturbation}"
    
    return base_solution

def apply_advection_operator(ic_expr, dims, parameters):
    if parameters.get('solution_type') == 'packet':
        sigma = parameters.get('sigma', 0.1)
        centers = parameters.get('centers', [0.5] * dims)
        spatial_vars = get_spatial_vars(dims)
        # Build a Gaussian envelope over the spatial variables
        if dims == 1:
            envelope = f"exp(-(({spatial_vars[0]} - {centers[0]})**2)/{sigma})"
        else:
            terms = [f"({var} - {centers[i]})**2" for i, var in enumerate(spatial_vars)]
            envelope = f"exp(-({' + '.join(terms)})/{sigma})"
        # Apply advection: shift the first spatial variable by velocity*t
        velocity = parameters.get('velocity', 1.0)
        shifted_ic = ic_expr.replace(spatial_vars[0], f"({spatial_vars[0]} - {velocity}*t)")
        return f"{envelope}*({shifted_ic})"
    velocity = round(parameters.get('velocity', random.uniform(0.5, 1.0)), 3)  # YAML range [0.5, 1.0]
    spatial_vars = get_spatial_vars(dims)
    modified_ic = ic_expr
    for var in spatial_vars:
        modified_ic = modified_ic.replace(var, f"({var}-{velocity}*t)")
    if random.random() < PERTURBATION_PROBABILITY:
        strength = round(random.uniform(0.1, 0.5), 3)
        k = random.choice([1, 2])
        omega = random.choice([1, 2])
        form = random.choice(["sin", "cos"])
        perturbation = f" + {strength}*{form}({k}*pi*{spatial_vars[0]})*{form}({omega}*t)"
        return f"{modified_ic}{perturbation}"
    return modified_ic

def apply_reaction_diffusion_operator(ic_expr, dims, parameters):
    if parameters.get('solution_type') == 'gaussian_decay':
        D = round(parameters.get('D', random.uniform(0.1, 0.5)), 3)
        sigma = parameters.get('sigma', 0.1)
        centers = parameters.get('centers', [0.5] * dims)
        spatial_vars = get_spatial_vars(dims)
        if dims == 1:
            envelope = f"exp(-(({spatial_vars[0]} - {centers[0]})**2)/{sigma})"
        else:
            terms = [f"({var} - {centers[i]})**2" for i, var in enumerate(spatial_vars)]
            envelope = f"exp(-({' + '.join(terms)})/{sigma})"
        # Combine the envelope with a time decay from diffusion
        return f"({ic_expr})*({envelope})*exp(-{D}*t)"
    D = round(parameters.get('D', random.uniform(0.1, 0.5)), 3)  # YAML range [0.1, 0.5]
    k = round(parameters.get('k', random.uniform(-0.5, 0.5)), 3)  # YAML range [-0.5, 0.5]
    if parameters.get('ic_type', 'sine') == 'sine' and parameters.get('use_eigenvalue_decay', True):
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', False)
        domain_length = parameters.get('domain_length', 1.0)
        k_squared_terms = [f"({'pi' if use_pi else ''}{k}/{domain_length})**2" for k in wavenumbers[:dims]]
        k_squared = "+".join(k_squared_terms) if k_squared_terms else "1"
        base_solution = f"({ic_expr})*exp(-({D}*({k_squared})+{k})*t)"
    else:
        base_solution = f"({ic_expr})*exp(-({D}+{k})*t)"
    if random.random() < PERTURBATION_PROBABILITY:
        strength = round(random.uniform(0.1, 0.5), 3)
        omega = random.choice([1, 2])
        form = random.choice(["sin", "cos"])
        perturbation = f"*(1 + {strength}*{form}({omega}*t))"
        return f"{base_solution}{perturbation}"
    return base_solution

def apply_burgers_operator(ic_expr, dims, parameters):
    nu = round(parameters.get('nu', random.uniform(0.01, 0.5)), 3)  # YAML range [0.05, 0.1]
    use_shock = parameters.get('use_shock_solution', True)
    if use_shock and parameters.get('ic_type', 'tanh') in ['tanh', 'step']:
        velocity = round(parameters.get('shock_speed', random.uniform(0.01, 0.5)), 3)
        spatial_vars = get_spatial_vars(dims)
        modified_ic = ic_expr
        for var in spatial_vars:
            if var == 'x':
                modified_ic = modified_ic.replace(var, f"({var}-{velocity}*t)")
        base_solution = modified_ic
    else:
        velocity = round(parameters.get('velocity', random.uniform(0.01, 0.5)), 3)
        base_solution = apply_advection_operator(ic_expr, dims, {'velocity': velocity})
    if random.random() < PERTURBATION_PROBABILITY:
        strength = round(random.uniform(0.01, 0.5), 3)
        k = random.choice([1, 2])
        decay = round(random.uniform(0.01, 0.5), 3)
        form = random.choice(["sin", "cos"])
        perturbation = f" + {strength}*{form}({k}*pi*x)*exp(-{decay}*t)"
        return f"{base_solution}{perturbation}"
    return base_solution

def apply_inv_burgers_operator(ic_expr, dims, parameters):
    nu = 0
    use_shock = parameters.get('use_shock_solution', True)
    if use_shock and parameters.get('ic_type', 'tanh') in ['tanh', 'step']:
        velocity = 0
        spatial_vars = get_spatial_vars(dims)
        modified_ic = ic_expr
        for var in spatial_vars:
            if var == 'x':
                modified_ic = modified_ic.replace(var, f"({var}-{velocity}*t)")
        base_solution = modified_ic
    else:
        velocity = 0
        base_solution = apply_advection_operator(ic_expr, dims, {'velocity': velocity})
    if random.random() < PERTURBATION_PROBABILITY:
        strength = round(random.uniform(0.01, 0.5), 3)
        k = random.choice([1, 2])
        decay = round(random.uniform(0.01, 0.5), 3)
        form = random.choice(["sin", "cos"])
        perturbation = f" + {strength}*{form}({k}*pi*x)*exp(-{decay}*t)"
        return f"{base_solution}{perturbation}"
    return base_solution

def apply_helmholtz_operator(ic_expr, dims, parameters):
    """
    Create a manufactured solution for the Helmholtz equation: ∇²u + k²u = 0
    Designed to produce small, clean forcing terms.
    """
    k = parameters.get('k', 1.0)
    ic_type = parameters.get('ic_type', 'sine')
    solution_type = parameters.get('solution_type', 'eigenfunction')
    
    if solution_type == 'eigenfunction':
        if dims == 1:
            wavenumber = parameters.get('wavenumbers', [1])[0]
            use_pi = parameters.get('use_pi_wavenumbers', True)
            k_factor = f"{wavenumber}*pi" if use_pi else f"{wavenumber}"
            # This will give zero forcing term if k is chosen to match the wavenumber
            return f"sin({k_factor}*x)"
        elif dims == 2:
            kx = parameters.get('wavenumbers', [1, 1])[0]
            ky = parameters.get('wavenumbers', [1, 1])[1]
            use_pi = parameters.get('use_pi_wavenumbers', True)
            kx_factor = f"{kx}*pi" if use_pi else f"{kx}"
            ky_factor = f"{ky}*pi" if use_pi else f"{ky}"
            # 2D eigenfunction 
            return f"sin({kx_factor}*x)*sin({ky_factor}*y)"
    
    elif solution_type == 'simple_harmonic':
        # Simple harmonic functions that satisfy the Helmholtz equation
        if dims == 1:
            amplitude = parameters.get('amplitudes', [1.0])[0]
            return f"{amplitude}*cos({k}*x)"
        elif dims == 2:
            amplitude = parameters.get('amplitudes', [1.0])[0]
            return f"{amplitude}*cos({k}*sqrt(x^2+y^2))"
    
    # For other solution types, try to adapt them appropriately
    if ic_type == 'polynomial':
        # For polynomial, use a simple solution that satisfies Helmholtz
        if dims == 1:
            # Simple solution: Quadratic polynomial solution
            return f"1 - {k}^2*x^2/2"  # Approximate solution near x=0
        elif dims == 2:
            # 2D polynomial approximation
            return f"1 - {k}^2*(x^2+y^2)/4"
    
    elif ic_type == 'gaussian':
        # For Gaussian, use a localized approximation
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            sigma = parameters.get('sigma', 0.2)
            # Approximate solution (not exact but gives small forcing near center)
            return f"exp(-((x-{center})/{sigma})^2)"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            sigma = parameters.get('sigma', 0.2)
            return f"exp(-((x-{center_x})^2+(y-{center_y})^2)/{sigma}^2)"
    
    # Default fallback - ensures we get different solutions for different cases
    wavenumber = parameters.get('wavenumbers', [k])[0]
    use_pi = parameters.get('use_pi_wavenumbers', True)
    k_factor = f"{wavenumber}*pi" if use_pi else f"{wavenumber}"
    
    if dims == 1:
        return f"sin({k_factor}*x + 0.1*{k}*t)"  # Adding slight time dependence for variety
    else:
        return f"sin({k_factor}*x)*cos({k_factor}*y)"

def apply_laplacian_operator(ic_expr, dims, parameters):
    """
    Simplified implementation for Laplacian (∇²u = 0) and Poisson (∇²u = f) equations.
    Focuses on simpler solutions with lower-degree polynomials.
    """
    ic_type = parameters.get('ic_type', 'sine')
    equation_type = parameters.get('equation_type', 'laplacian')  # 'laplacian' or 'poisson'
    
    # Handle 1D case
    if dims == 1:
        if equation_type == 'laplacian':  # ∇²u = 0
            # Linear function (only harmonic function in 1D)
            a = parameters.get('coefficient', 1.0)
            return f"{a}*x"
        else:  # Poisson equation
            if ic_type == 'polynomial':
                # Simple quadratic with constant forcing
                a = parameters.get('coefficient', 1.0)
                return f"{a}*x^2"  # ∇²u = 2a
            elif ic_type == 'sine':
                # Sine with predictable forcing
                k = parameters.get('wavenumbers', [1])[0]
                use_pi = parameters.get('use_pi_wavenumbers', True)
                pi_factor = "pi*" if use_pi else ""
                return f"sin({k}*{pi_factor}x)"
    
    # Handle 2D case
    elif dims == 2:
        if equation_type == 'laplacian':  # ∇²u = 0
            if ic_type == 'polynomial':
                # x² - y² is the simplest non-trivial harmonic polynomial
                return "x^2 - y^2"
            elif ic_type == 'sine':
                # Simple harmonic using hyperbolic sine
                k = parameters.get('wavenumbers', [1])[0]
                return f"sin({k}*x)*sinh({k}*y)"
        else:  # Poisson equation
            if ic_type == 'polynomial':
                # Simple quadratic with constant forcing
                a = parameters.get('coefficient', 1.0)
                return f"{a}*(x^2 + y^2)"  # ∇²u = 4a
            elif ic_type == 'sine':
                # Sine product with predictable forcing
                kx = parameters.get('wavenumbers', [1, 1])[0]
                ky = parameters.get('wavenumbers', [1, 1])[1]
                use_pi = parameters.get('use_pi_wavenumbers', True)
                pi_factor = "pi*" if use_pi else ""
                return f"sin({kx}*{pi_factor}x)*sin({ky}*{pi_factor}y)"
    
    # Fallback to original IC
    return ic_expr
def apply_biharmonic_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the biharmonic equation: ∂u/∂t + α∇⁴u = 0
    With focus on small forcing terms.
    """
    alpha = parameters.get('alpha', 0.05)
    ic_type = parameters.get('ic_type', 'sine')
    
    if ic_type == 'sine':
        # Sine functions are eigenfunctions of the biharmonic
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        
        if dims == 1:
            k = wavenumbers[0]
            pi_str = "pi*" if use_pi else ""
            # The eigenvalue is k⁴
            decay_rate = alpha * k**4 * (math.pi**4 if use_pi else 1)
            return f"exp(-{decay_rate}*t)*sin({k}*{pi_str}x)"
        elif dims == 2:
            kx, ky = wavenumbers[:2]
            pi_str = "pi*" if use_pi else ""
            # The eigenvalue is (kx²+ky²)²
            k_squared = kx**2 + ky**2
            decay_rate = alpha * k_squared**2 * (math.pi**4 if use_pi else 1)
            return f"exp(-{decay_rate}*t)*sin({kx}*{pi_str}x)*sin({ky}*{pi_str}y)"
    
    elif ic_type == 'polynomial':
        # Boundary-matched polynomial with simple 4th derivative
        if dims == 1:
            # x²(1-x)² vanishes at x=0,1 with first derivatives
            decay_rate = 24 * alpha  # From the 4th derivative of x²(1-x)²
            return f"exp(-{decay_rate}*t)*x^2*(1-x)^2"
        elif dims == 2:
            decay_rate = 24 * alpha
            return f"exp(-{decay_rate}*t)*x^2*(1-x)^2*y^2*(1-y)^2"
    
    elif ic_type == 'gaussian':
        # Gaussian with appropriate time decay
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            sigma = parameters.get('sigma', 0.2)
            # Approximate decay rate for the 4th derivative of Gaussian
            decay_rate = alpha / (sigma**4)
            return f"exp(-{decay_rate}*t)*exp(-((x-{center})/{sigma})^2)"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            sigma = parameters.get('sigma', 0.2)
            decay_rate = alpha / (sigma**4)
            return f"exp(-{decay_rate}*t)*exp(-((x-{center_x})^2+(y-{center_y})^2)/{sigma}^2)"
    
   

    elif solution_type == 'gaussian_decay':
        # Gaussian solution that decays with time
        centers = parameters.get('centers', [0.5] * dims)
        sigma = parameters.get('sigma', 0.2)
        decay_rate = parameters.get('decay_rate', alpha)
        
        if dims == 1:
            center_x = centers[0]
            return f"exp(-{decay_rate}*t)*exp(-((x-{center_x})/{sigma})^2)"
        elif dims == 2:
            center_x, center_y = centers[:2]
            return f"exp(-{decay_rate}*t)*exp(-((x-{center_x})/{sigma})^2-((y-{center_y})/{sigma})^2)"
    
    elif solution_type == 'sinusoidal_decay':
        # Sinusoidal solution with time decay
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        decay_rate = parameters.get('decay_rate', alpha)
        
        if dims == 1:
            k = parameters.get('wavenumbers', [1])[0]
            # Use higher wavenumber products for biharmonic (distinctive)
            return f"exp(-{alpha}*(pi*{k})^4*t)*sin(pi*{k}*x)"
        elif dims == 2:
            kx = parameters.get('wavenumbers', [1, 1])[0]
            ky = parameters.get('wavenumbers', [1, 1])[1]
            # Use product of different modes - very distinctive for biharmonic
            return f"exp(-{alpha}*(pi^4)*({kx}^4+{ky}^4+2*{kx}^2*{ky}^2)*t)*(sin(pi*{kx}*x)*sin(pi*{ky}*y) + 0.2*sin(2*pi*{kx}*x)*sin(2*pi*{ky}*y))"
    
    # Default fallback
    return ic_expr

def apply_allen_cahn_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the Allen-Cahn equation: ∂u/∂t = ε²∇²u + u - u³
    With focus on clean, small forcing terms and varied solution types.
    """
    epsilon_squared = parameters.get('epsilon_squared', 1.0)
    ic_type = parameters.get('ic_type', 'tanh')
    solution_type = parameters.get('solution_type', 'traveling_wave')
    
    if solution_type == 'traveling_wave':
        # Traveling wave solution
        velocity = parameters.get('velocity', 0.5)
        width = parameters.get('width', 2.0 * math.sqrt(epsilon_squared))
        amplitude = parameters.get('amplitude', 1.0)
        
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            return f"{amplitude}*0.5*(1 + tanh((x-{center}-{velocity}*t)/{width}))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            # Radial traveling wave
            return f"{amplitude}*0.5*(1 + tanh((sqrt((x-{center_x})^2+(y-{center_y})^2)-{velocity}*t)/{width}))"
    
    elif solution_type == 'standing_pattern':
        # Standing pattern solution
        width = parameters.get('width', 2.0 * math.sqrt(epsilon_squared))
        amplitude = parameters.get('amplitude', 1.0)
        
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            return f"{amplitude}*0.5*(1 + tanh((x-{center})/{width}))*exp(-0.1*t)"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            # Stationary pattern with decay
            return f"{amplitude}*0.5*(1 + tanh((sqrt((x-{center_x})^2+(y-{center_y})^2))/{width}))*exp(-0.1*t)"
    
    elif solution_type == 'growth_decay':
        # Growth/decay in linearized regime
        wavenumber = parameters.get('wavenumber', 1)
        amplitude = parameters.get('amplitude', 0.1)
        
        if dims == 1:
            k_squared = wavenumber**2 * math.pi**2
            growth_rate = 1.0 - epsilon_squared * k_squared
            return f"{amplitude}*exp({growth_rate}*t)*sin({wavenumber}*pi*x)"
        elif dims == 2:
            k_squared = 2 * wavenumber**2 * math.pi**2
            growth_rate = 1.0 - epsilon_squared * k_squared
            return f"{amplitude}*exp({growth_rate}*t)*sin({wavenumber}*pi*x)*sin({wavenumber}*pi*y)"
    
    elif solution_type == 'pattern_formation':
        # Pattern formation solution
        amplitude = parameters.get('amplitude', 0.1)
        
        if dims == 1:
            return f"{amplitude}*(sin(pi*x) + 0.5*sin(2*pi*x))*exp((1-{epsilon_squared}*pi^2)*t)"
        elif dims == 2:
            return f"{amplitude}*(sin(pi*x)*sin(pi*y) + 0.5*sin(2*pi*x)*sin(2*pi*y))*exp((1-{epsilon_squared}*2*pi^2)*t)"
    
    # Default behaviors based on IC type
    if ic_type == 'tanh':
        # Tanh is a natural traveling wave solution for Allen-Cahn
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            width = parameters.get('width', 2.0 * math.sqrt(epsilon_squared))
            # Add time dependence for non-stationary solution
            return f"0.5*(1 + tanh((x-{center}-0.1*t)/{width}))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            width = parameters.get('width', 2.0 * math.sqrt(epsilon_squared))
            # For 2D, use radial distance from center with time modulation
            return f"0.5*(1 + tanh((sqrt((x-{center_x})^2+(y-{center_y})^2)-0.1*t)/{width}))"
    
    elif ic_type == 'sine':
        # Small amplitude sine solution (linearized regime)
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        amplitude = parameters.get('amplitude', 0.1)
        
        if dims == 1:
            k = wavenumbers[0]
            pi_str = "pi*" if use_pi else ""
            # Add unique time modulation
            return f"{amplitude}*exp(0.2*t)*sin({k}*{pi_str}x)"
        elif dims == 2:
            kx, ky = wavenumbers[:2]
            pi_str = "pi*" if use_pi else ""
            return f"{amplitude}*exp(0.2*t)*sin({kx}*{pi_str}x)*sin({ky}*{pi_str}y)"
    
    elif ic_type == 'polynomial':
        # Use polynomial approximation with time decay
        if dims == 1:
            # Approximate solution near a stable point u=±1
            decay_rate = random.uniform(0.05, 0.2)
            return f"1 - {decay_rate}*exp(-t)*x^2*(1-x)^2"
        elif dims == 2:
            decay_rate = random.uniform(0.05, 0.2)
            return f"1 - {decay_rate}*exp(-t)*x^2*(1-x)^2*y^2*(1-y)^2"
    
    # Fallback with custom time behavior
    time_factor = random.uniform(0.05, 0.2)
    return f"exp(-{time_factor}*t)*({ic_expr})"

def apply_ginzburg_landau_operator(ic_expr, dims, parameters):
        """
        Create manufactured solutions for the Ginzburg-Landau equation: ∂u/∂t = ∇²u + αu - βu³
        With focus on small, clean forcing terms and varied solution types.
        """
        alpha = parameters.get('alpha', 1.0)
        beta = parameters.get('beta', 1.0)
        ic_type = parameters.get('ic_type', 'sine')
        solution_type = parameters.get('solution_type', 'pattern_formation')
        
        if solution_type == 'traveling_front':
            # Traveling front solution
            if dims == 1:
                velocity = parameters.get('velocity', 0.5)
                width = parameters.get('width', 1.0/math.sqrt(alpha/2))
                amplitude_factor = parameters.get('amplitude_factor', 1.0)
                center = parameters.get('centers', [0.5])[0]
                return f"{amplitude_factor}*sqrt({alpha}/{beta})*tanh(({spatial_vars[0]}-{center}-{velocity}*t)/{width})"
            elif dims == 2:
                velocity = parameters.get('velocity', 0.5)
                width = parameters.get('width', 1.0/math.sqrt(alpha/2))
                amplitude_factor = parameters.get('amplitude_factor', 1.0)
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                return f"{amplitude_factor}*sqrt({alpha}/{beta})*tanh((x-{center_x}-{velocity}*t)/{width})*cos(y-{center_y})"
        
        elif solution_type == 'pattern_formation':
            # Pattern formation with amplitude modulation
            if dims == 1:
                k = parameters.get('wavenumbers', [1])[0]
                amplitude = parameters.get('amplitude', 0.1)
                return f"{amplitude}*exp(({alpha} - {k}^2)*t)*sin({k}*pi*x)"
            elif dims == 2:
                kx = parameters.get('wavenumbers', [1, 1])[0]
                ky = parameters.get('wavenumbers', [1, 1])[1]
                amplitude = parameters.get('amplitude', 0.1)
                return f"{amplitude}*exp(({alpha} - ({kx}^2+{ky}^2))*t)*sin({kx}*pi*x)*sin({ky}*pi*y)"
        
        elif solution_type == 'localized_structure':
            # Localized structure solution
            if dims == 1:
                sigma = parameters.get('sigma', 0.2)
                center = parameters.get('centers', [0.5])[0]
                amplitude = parameters.get('amplitude', 0.8)
                time_factor = parameters.get('time_factor', 0.2)
                return f"{amplitude}*sech((x-{center})/({sigma}*sqrt(1+{time_factor}*t)))"
            elif dims == 2:
                sigma = parameters.get('sigma', 0.2)
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                amplitude = parameters.get('amplitude', 0.8)
                time_factor = parameters.get('time_factor', 0.2)
                return f"{amplitude}*exp(-((x-{center_x})^2+(y-{center_y})^2)/({sigma}^2*(1+{time_factor}*t)))"
        
        # Solution types based on IC types
        if ic_type == 'sine':
            # For sine, use growth/decay in linearized regime
            wavenumbers = parameters.get('wavenumbers', [1] * dims)
            use_pi = parameters.get('use_pi_wavenumbers', True)
            amplitude = parameters.get('amplitude', 0.1)
            
            if dims == 1:
                k = wavenumbers[0]
                pi_str = "pi*" if use_pi else ""
                k_squared = k**2 * (math.pi**2 if use_pi else 1)
                # Use unique combination of parameters
                growth_rate = alpha - k_squared + 0.1*beta
                return f"{amplitude}*exp({growth_rate}*t)*sin({k}*{pi_str}x)"
            elif dims == 2:
                kx, ky = wavenumbers[:2]
                pi_str = "pi*" if use_pi else ""
                k_squared = (kx**2 + ky**2) * (math.pi**2 if use_pi else 1)
                growth_rate = alpha - k_squared + 0.1*beta
                return f"{amplitude}*exp({growth_rate}*t)*sin({kx}*{pi_str}x)*sin({ky}*{pi_str}y)"
        
        elif ic_type == 'tanh':
            # Tanh-based solutions for interfaces
            if dims == 1:
                center = parameters.get('centers', [0.5])[0]
                width = parameters.get('width', 1.0/math.sqrt(alpha/2))
                # Add time dependence for variety
                return f"sqrt({alpha}/{beta})*tanh((x-{center})/{width})*exp(-0.1*t)"
            elif dims == 2:
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                width = parameters.get('width', 1.0/math.sqrt(alpha/2))
                # Different pattern in 2D
                return f"sqrt({alpha}/{beta})*tanh((sqrt((x-{center_x})^2+(y-{center_y})^2))/{width})*exp(-0.1*t)"
        
        elif ic_type == 'gaussian':
            # Localized Gaussian pulse with growth/decay
            if dims == 1:
                center = parameters.get('centers', [0.5])[0]
                sigma = parameters.get('sigma', 0.2)
                growth_rate = alpha - 1/(2*sigma**2) - 0.05*beta
                amplitude = parameters.get('amplitude', 0.1)
                return f"{amplitude}*exp({growth_rate}*t)*exp(-((x-{center})/{sigma})^2)"
            elif dims == 2:
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                sigma = parameters.get('sigma', 0.2)
                growth_rate = alpha - 1/(sigma**2) - 0.05*beta
                amplitude = parameters.get('amplitude', 0.1)
                return f"{amplitude}*exp({growth_rate}*t)*exp(-((x-{center_x})^2+(y-{center_y})^2)/{sigma}^2)"
        
        # Fallback solution with unique parameterization
        time_factor = random.uniform(0.05, 0.2)
        amplitude = random.uniform(0.05, 0.15)
        k = random.choice([1, 2, 3])
        return f"{amplitude}*exp({alpha*0.8}*t)*({ic_expr})*cos({k}*t)"

def apply_fisher_kpp_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the Fisher-KPP equation: ∂u/∂t = D∇²u + ru(1-u)
    With focus on small, clean forcing terms and solution types from YAML.
    """
    D = parameters.get('D', 1.0)
    r = parameters.get('r', 1.0)
    ic_type = parameters.get('ic_type', 'tanh')
    solution_type = parameters.get('solution_type', 'logistic_front')
    spatial_vars = get_spatial_vars(dims)
    
    if solution_type == 'logistic_front':
        # Traveling front emerging from logistic growth dynamics
        wave_speed = parameters.get('wave_speed', 2.0 * math.sqrt(D * r))
        inflection_point = parameters.get('inflection_point', 0.5)
        
        if dims == 1:
            # Classic traveling wave front solution
            center = parameters.get('centers', [0.5])[0]
            width = math.sqrt(D/r)  # Natural width scale
            return f"0.5*(1 - tanh((x - {center} - {wave_speed}*t)/{width}))"
        elif dims == 2:
            # Radially expanding wave
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            width = math.sqrt(D/r)
            return f"0.5*(1 - tanh((sqrt((x-{center_x})^2+(y-{center_y})^2) - {wave_speed}*t)/{width}))"
    
    elif solution_type == 'traveling_wave':
        # Traveling wave solution with velocity from parameters
        velocity = parameters.get('velocity', 1.0)
        velocity_factor = parameters.get('velocity_factor', 1.0)
        actual_velocity = velocity * velocity_factor
        
        if dims == 1:
            if ic_type == 'sine':
                wavenumbers = parameters.get('wavenumbers', [1])
                use_pi = parameters.get('use_pi_wavenumbers', True)
                k = wavenumbers[0]
                pi_str = "pi*" if use_pi else ""
                # Moving sine wave with amplitude modulation
                return f"0.2*(1 + sin({k}*{pi_str}(x-{actual_velocity}*t)))*exp({r}*t/(1+{r}*t))"
            elif ic_type == 'gaussian':
                center = parameters.get('centers', [0.5])[0]
                sigma = parameters.get('sigma', 0.1)
                # Moving Gaussian pulse
                return f"exp(-((x-{center}-{actual_velocity}*t)/{sigma})^2)*exp({r}*t/(1+{r}*t))"
            else:
                # Default traveling wave for other ICs
                center = parameters.get('centers', [0.5])[0]
                return f"0.5*(1 - tanh((x-{center}-{actual_velocity}*t)/({math.sqrt(D/r)})))"
        elif dims == 2:
            if ic_type == 'sine':
                wavenumbers = parameters.get('wavenumbers', [1, 1])
                use_pi = parameters.get('use_pi_wavenumbers', True)
                kx, ky = wavenumbers[:2]
                pi_str = "pi*" if use_pi else ""
                # Moving sine pattern in 2D
                return f"0.2*(1 + sin({kx}*{pi_str}(x-{actual_velocity}*t))*sin({ky}*{pi_str}y))*exp({r}*t/(1+{r}*t))"
            elif ic_type == 'gaussian':
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                sigma = parameters.get('sigma', 0.1)
                # Moving Gaussian in 2D
                return f"exp(-((x-{center_x}-{actual_velocity}*t)^2+(y-{center_y})^2)/{sigma}^2)*exp({r}*t/(1+{r}*t))"
            else:
                # Default traveling front in 2D
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                width = math.sqrt(D/r)
                return f"0.5*(1 - tanh((x-{center_x}-{actual_velocity}*t)/{width}))*cos(pi*y/{center_y})"
    
    # Default behaviors based on IC type if no matching solution type
    if ic_type == 'tanh':
        # Traveling wave solution approximation
        if dims == 1:
            # Wave speed c = 2*sqrt(D*r) for the minimal speed
            wave_speed = 2.0 * math.sqrt(D * r)
            width = parameters.get('width', math.sqrt(D/r))
            center = parameters.get('centers', [0.5])[0]
            # Traveling wave moving to the right
            return f"0.5*(1-tanh((x-{center}-{wave_speed}*t)/{width}))"
        elif dims == 2:
            # Radially expanding wave (approximate)
            wave_speed = 2.0 * math.sqrt(D * r)
            width = parameters.get('width', math.sqrt(D/r))
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            return f"0.5*(1-tanh((sqrt((x-{center_x})^2+(y-{center_y})^2)-{wave_speed}*t)/{width}))"
    
    elif ic_type == 'step':
        # Step function evolving into traveling wave
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            # Simplified evolution with matching speed
            return f"0.5*(1+tanh(-(x-{center}-2*sqrt({D}*{r})*t)/sqrt({D}/{r})))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            return f"0.5*(1+tanh(-(sqrt((x-{center_x})^2+(y-{center_y})^2)-2*sqrt({D}*{r})*t)/sqrt({D}/{r})))"
    
    elif ic_type == 'sine':
        # For small u, linearize around u=0
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        amplitude = parameters.get('amplitude', 0.1)
        
        if dims == 1:
            k = wavenumbers[0]
            pi_str = "pi*" if use_pi else ""
            k_squared = k**2 * (math.pi**2 if use_pi else 1)
            # Linearized growth rate
            growth_rate = r - D * k_squared
            return f"{amplitude}*exp({growth_rate}*t)*sin({k}*{pi_str}x)"
        elif dims == 2:
            kx, ky = wavenumbers[:2]
            pi_str = "pi*" if use_pi else ""
            k_squared = (kx**2 + ky**2) * (math.pi**2 if use_pi else 1)
            growth_rate = r - D * k_squared
            return f"{amplitude}*exp({growth_rate}*t)*sin({kx}*{pi_str}x)*sin({ky}*{pi_str}y)"
    
    # Fallback solution with logistic-like growth
    return f"(0.5+0.1*({ic_expr}))*exp({r}*t/(1+exp({r}*t)))"

def apply_convection_diffusion_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the convection-diffusion equation: ∂u/∂t + v·∇u = D∇²u
    With focus on solution types from YAML.
    """
    D = parameters.get('D', 0.5)
    velocity = parameters.get('velocity', 1.0)
    ic_type = parameters.get('ic_type', 'gaussian')
    solution_type = parameters.get('solution_type', 'traveling_wave')
    spatial_vars = get_spatial_vars(dims)
    
    if solution_type == 'traveling_wave':
        # Traveling wave with advection-diffusion dynamics
        velocity_factor = parameters.get('velocity_factor', 1.0)
        actual_velocity = velocity * velocity_factor
        
        if ic_type == 'sine':
            # Sine wave with decay and translation
            wavenumbers = parameters.get('wavenumbers', [1] * dims)
            use_pi = parameters.get('use_pi_wavenumbers', True)
            
            if dims == 1:
                k = wavenumbers[0]
                pi_str = "pi*" if use_pi else ""
                k_val = k * (math.pi if use_pi else 1)
                # Exact solution for sine initial condition
                decay_rate = D * k_val**2
                return f"exp(-{decay_rate}*t)*sin({k}*{pi_str}(x-{actual_velocity}*t))"
            elif dims == 2:
                kx, ky = wavenumbers[:2]
                pi_str = "pi*" if use_pi else ""
                kx_val = kx * (math.pi if use_pi else 1)
                ky_val = ky * (math.pi if use_pi else 1)
                decay_rate = D * (kx_val**2 + ky_val**2)
                return f"exp(-{decay_rate}*t)*sin({kx}*{pi_str}(x-{actual_velocity}*t))*sin({ky}*{pi_str}y)"
        
        elif ic_type == 'gaussian':
            # Gaussian packet solution - exact for convection-diffusion
            if dims == 1:
                center = parameters.get('centers', [0.5])[0]
                sigma_0 = parameters.get('sigma', 0.1)
                # For a Gaussian, we can derive the exact solution
                return f"(1/sqrt(1+4*{D}*t/{sigma_0}^2))*exp(-((x-{center}-{actual_velocity}*t)^2)/(4*{D}*t+{sigma_0}^2))"
            elif dims == 2:
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                sigma_0 = parameters.get('sigma', 0.1)
                # 2D Gaussian with convection in x-direction
                return f"(1/sqrt(1+4*{D}*t/{sigma_0}^2))*exp(-((x-{center_x}-{actual_velocity}*t)^2+(y-{center_y})^2)/(4*{D}*t+{sigma_0}^2))"
        
        elif ic_type in ['step', 'tanh']:
            # Smoothed step/tanh function solution
            if dims == 1:
                center = parameters.get('centers', [0.5])[0]
                width = parameters.get('width', 0.1)
                # Error function is the exact solution for step initial condition
                return f"0.5*(1+erf((x-{center}-{actual_velocity}*t)/(2*sqrt({D}*t+{width}^2))))"
            elif dims == 2:
                center_x = parameters.get('centers', [0.5, 0.5])[0]
                center_y = parameters.get('centers', [0.5, 0.5])[1]
                width = parameters.get('width', 0.1)
                # 2D with convection in x-direction
                return f"0.5*(1+erf((x-{center_x}-{actual_velocity}*t)/(2*sqrt({D}*t+{width}^2))))"
        
        else:  # polynomial or other
            # Apply simple advection-diffusion to the IC
            modified_ic = ic_expr
            if dims == 1:
                modified_ic = modified_ic.replace('x', f'(x-{actual_velocity}*t)')
            elif dims >= 2:
                modified_ic = modified_ic.replace(spatial_vars[0], f'({spatial_vars[0]}-{actual_velocity}*t)')
            
            return f"({modified_ic})*exp(-{D}*t)"
    
    elif solution_type == 'decaying_packet':
        # Specifically for Gaussian packet with custom decay
        sigma = parameters.get('sigma', 0.1)
        decay_rate = parameters.get('decay_rate', 0.5)
        
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            # Gaussian packet solution with additional exponential decay
            return f"exp(-{decay_rate}*t)*(1/sqrt(1+4*{D}*t/{sigma}^2))*exp(-((x-{center}-{velocity}*t)^2)/(4*{D}*t+{sigma}^2))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            # 2D Gaussian packet with decay
            return f"exp(-{decay_rate}*t)*(1/sqrt(1+4*{D}*t/{sigma}^2))*exp(-((x-{center_x}-{velocity}*t)^2+(y-{center_y})^2)/(4*{D}*t+{sigma}^2))"
    
    # Default fallback based on IC type if no matching solution type
    if ic_type == 'gaussian':
        # Gaussian packet solution - exact for convection-diffusion
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            sigma_0 = parameters.get('sigma', 0.1)
            # For a Gaussian, we can derive the exact solution
            return f"(1/sqrt(1+4*{D}*t/{sigma_0}^2))*exp(-((x-{center}-{velocity}*t)^2)/(4*{D}*t+{sigma_0}^2))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            sigma_0 = parameters.get('sigma', 0.1)
            # 2D Gaussian with convection in x-direction
            return f"(1/sqrt(1+4*{D}*t/{sigma_0}^2))*exp(-((x-{center_x}-{velocity}*t)^2+(y-{center_y})^2)/(4*{D}*t+{sigma_0}^2))"
    
    elif ic_type == 'sine':
        # Sine wave with decay and translation
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        
        if dims == 1:
            k = wavenumbers[0]
            pi_str = "pi*" if use_pi else ""
            k_val = k * (math.pi if use_pi else 1)
            # Exact solution for sine initial condition
            decay_rate = D * k_val**2
            return f"exp(-{decay_rate}*t)*sin({k}*{pi_str}(x-{velocity}*t))"
        elif dims == 2:
            kx, ky = wavenumbers[:2]
            pi_str = "pi*" if use_pi else ""
            kx_val = kx * (math.pi if use_pi else 1)
            ky_val = ky * (math.pi if use_pi else 1)
            decay_rate = D * (kx_val**2 + ky_val**2)
            return f"exp(-{decay_rate}*t)*sin({kx}*{pi_str}(x-{velocity}*t))*sin({ky}*{pi_str}y)"
    
    # Fallback solution - simple traveling wave with diffusion
    modified_ic = ic_expr
    for var in spatial_vars:
        if var == spatial_vars[0]:  # Only apply to first variable (usually x)
            modified_ic = modified_ic.replace(var, f"({var}-{velocity}*t)")
    
    return f"({modified_ic})*exp(-{D}*t)"


def apply_fitzhugh_nagumo_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the FitzHugh-Nagumo equation: ∂u/∂t = D∇²u - u³ + u + v
    This is a simplified version focusing on the u-component with varied solution types.
    """
    D = parameters.get('D', 0.5)
    v_val = parameters.get('v_value', 0.1)
    ic_type = parameters.get('ic_type', 'pulse')
    solution_type = parameters.get('solution_type', 'pulse')
    
    if solution_type == 'excitability_wave':
        # Wave of excitation solution
        threshold = parameters.get('threshold', 0.5)
        recovery_rate = parameters.get('recovery_rate', 0.1)
        wave_speed = parameters.get('wave_speed', 1.0)
        
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            # Traveling pulse with recovery
            return f"exp(-{recovery_rate}*t)*tanh((x-{center}-{wave_speed}*t)/{threshold})"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            # Radial wave with recovery
            return f"exp(-{recovery_rate}*t)*tanh((sqrt((x-{center_x})^2+(y-{center_y})^2)-{wave_speed}*t)/{threshold})"
    
    elif solution_type == 'pulse':
        # Pulse-like solution
        sigma = parameters.get('sigma', 0.1)
        amplitude = parameters.get('amplitude', 1.0)
        
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            # Pulse with unique amplitude and wave speed
            return f"{amplitude}*exp(-((x-{center}-0.2*t)/{sigma})^2)"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            # 2D pulse with movement in x direction
            return f"{amplitude}*exp(-((x-{center_x}-0.2*t)^2+(y-{center_y})^2)/{sigma}^2)"
    
    # Default behaviors based on IC type
    if ic_type == 'pulse':
        # Pulse-like solution approximation with time modulation
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            width = parameters.get('width', 0.1)
            # Add oscillatory time component
            return f"exp(-((x-{center})/{width})^2)*(1+0.1*sin(t))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            width = parameters.get('width', 0.1)
            # 2D pulse with time oscillation
            return f"exp(-((x-{center_x})^2+(y-{center_y})^2)/{width}^2)*(1+0.1*sin(t))"
    
    elif ic_type == 'sine':
        # Small amplitude sine wave with growth/decay
        wavenumbers = parameters.get('wavenumbers', [1] * dims)
        use_pi = parameters.get('use_pi_wavenumbers', True)
        amplitude = parameters.get('amplitude', 0.1)
        
        if dims == 1:
            k = wavenumbers[0]
            pi_str = "pi*" if use_pi else ""
            k_squared = k**2 * (math.pi**2 if use_pi else 1)
            # Linearized behavior with v_val influence
            growth_rate = 1 - D * k_squared - v_val
            return f"{amplitude}*exp({growth_rate}*t)*sin({k}*{pi_str}x)"
        elif dims == 2:
            kx, ky = wavenumbers[:2]
            pi_str = "pi*" if use_pi else ""
            k_squared = (kx**2 + ky**2) * (math.pi**2 if use_pi else 1)
            growth_rate = 1 - D * k_squared - v_val
            return f"{amplitude}*exp({growth_rate}*t)*sin({kx}*{pi_str}x)*sin({ky}*{pi_str}y)"
    
    elif ic_type == 'tanh':
        # Interface solution with recovery dynamics
        if dims == 1:
            center = parameters.get('centers', [0.5])[0]
            width = parameters.get('width', 0.1)
            # Traveling front with oscillation
            return f"tanh((x-{center}-0.2*t)/{width})*(1-0.1*sin(2*t))"
        elif dims == 2:
            center_x = parameters.get('centers', [0.5, 0.5])[0]
            center_y = parameters.get('centers', [0.5, 0.5])[1]
            width = parameters.get('width', 0.1)
            # 2D traveling interface
            return f"tanh((sqrt((x-{center_x})^2+(y-{center_y})^2)-0.2*t)/{width})*(1-0.1*sin(2*t))"
    
    # Fallback solution with unique temporal behavior
    growth_decay = random.choice([-1, 1]) * random.uniform(0.05, 0.2)
    k = random.choice([1, 2, 3])
    return f"{0.1}*exp({growth_decay}*t)*({ic_expr})*(1+0.1*sin({k}*t))"

def apply_sine_gordon_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the Sine-Gordon equation: ∂²u/∂t² - ∂²u/∂x² + sin(u) = 0
    With focus on small, clean forcing terms.
    """
    if dims != 1:
        # Sine-Gordon typically used in 1D
        return ic_expr
    
    ic_type = parameters.get('ic_type', 'sine')
    c2 = parameters.get('c2', 1.0)  # Wave speed parameter
    
    if ic_type == 'sine':
        # Small amplitude sine wave
        wavenumber = parameters.get('wavenumbers', [1])[0]
        use_pi = parameters.get('use_pi_wavenumbers', True)
        amplitude = parameters.get('amplitude', 0.1)  # Small for linearization
        
        k = wavenumber * (math.pi if use_pi else 1)
        omega = math.sqrt(k**2 + 1)  # Dispersion relation for linearized Sine-Gordon
        
        # Standing wave solution for small amplitude (linearized sin(u) ≈ u)
        return f"{amplitude}*cos({omega}*t)*sin({wavenumber}*{'pi*' if use_pi else ''}x)"
    
    elif ic_type == 'kink':
        # Kink soliton solution
        width = parameters.get('width', 1.0)
        velocity = parameters.get('velocity', 0.0)
        gamma = 1.0 / math.sqrt(1.0 - velocity**2)
        
        # Standard kink solution
        return f"4*atan(exp((x-{velocity}*t)/({width}*{gamma})))"
    
    elif ic_type == 'breather':
        # Breather soliton (simplified)
        omega = parameters.get('frequency', 0.5)
        width = 1.0 / math.sqrt(1.0 - omega**2)
        
        # Simplified breather solution
        return f"4*atan({width}*sin({omega}*t)/cosh({width}*x))"
    
    # Fallback with simple standing wave
    return f"0.1*sin(x)*cos(sqrt(2)*t)"

def apply_telegrapher_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the Telegrapher equation: ∂²u/∂t² + 2α∂u/∂t - c²∂²u/∂x² = 0
    With focus on small, clean forcing terms.
    """
    if dims != 1:
        # Telegrapher typically used in 1D
        return ic_expr
    
    alpha = parameters.get('alpha', 0.5)  # Damping parameter
    c2 = parameters.get('c2', 1.0)  # Wave speed squared
    ic_type = parameters.get('ic_type', 'sine')
    
    if ic_type == 'sine':
        # Damped wave solution
        wavenumber = parameters.get('wavenumbers', [1])[0]
        use_pi = parameters.get('use_pi_wavenumbers', True)
        
        k = wavenumber * (math.pi if use_pi else 1)
        # Compute the dispersion relation
        omega_squared = c2 * k**2 - alpha**2
        
        if omega_squared > 0:
            # Underdamped case: oscillatory with decay
            omega = math.sqrt(omega_squared)
            return f"exp(-{alpha}*t)*sin({wavenumber}*{'pi*' if use_pi else ''}x)*cos({omega}*t)"
        else:
            # Overdamped case: pure decay
            return f"exp(-{alpha}*t)*sin({wavenumber}*{'pi*' if use_pi else ''}x)"
    
    elif ic_type == 'gaussian':
        # Gaussian pulse solution (approximate)
        center = parameters.get('centers', [0.5])[0]
        sigma = parameters.get('sigma', 0.1)
        
        # Simple damped Gaussian
        return f"exp(-{alpha}*t)*exp(-((x-{center})/{sigma})^2)"
    
    # Fallback with simple damped wave
    return f"exp(-{alpha}*t)*({ic_expr})"

def apply_black_scholes_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the Black-Scholes equation:
    ∂u/∂t + 0.5*σ²*x²*∂²u/∂x² + r*x*∂u/∂x - r*u = 0
    Designed to match solution types from YAML.
    """
    if dims != 1:
        # Black-Scholes typically used in 1D
        return ic_expr
    
    sigma = parameters.get('sigma', 0.2)  # Volatility
    r = parameters.get('r', 0.05)  # Risk-free rate
    ic_type = parameters.get('ic_type', 'polynomial')
    solution_type = parameters.get('solution_type', 'power_exponential')
    
    if solution_type == 'power_exponential':
        # Using power function x^p which has clean derivatives
        p = parameters.get('p', 1.0)
        
        # For certain values of p, we get exact solutions
        if abs(p - 1.0) < 1e-6:  # p ≈ 1: linear solution
            return f"exp({r}*t)*x"
        else:
            # Power solution with appropriate time scaling
            return f"exp({r}*(1-{p})*t)*x^{p}"
    
    # Default behaviors based on IC type
    if ic_type == 'polynomial':
        # Simple polynomial solution
        p = random.uniform(0.5, 2.0)  # Exponent between 0.5 and 2.0
        
        # Power solution with appropriate time scaling
        return f"exp({r}*(1-{p})*t)*x^{p}"
    
    elif ic_type == 'sine':
        # Simple oscillatory solution (approximate)
        k = parameters.get('wavenumbers', [1])[0]
        use_pi = parameters.get('use_pi_wavenumbers', True)
        
        # Use a solution that mixes exponential growth with oscillations
        if use_pi:
            return f"exp({r}*t)*x*sin({k}*pi*log(x))"
        else:
            return f"exp({r}*t)*x*sin({k}*log(x))"
    
    elif ic_type == 'gaussian':
        # Approximate solution with log-normal characteristics
        center = parameters.get('centers', [1.0])[0]  # Center around 1.0 for log-normality
        sigma_param = parameters.get('sigma', 0.2)
        
        # Log-normal-like solution
        return f"exp({r}*t)*exp(-(log(x/{center}))^2/(2*{sigma_param}^2))"
    
    # Fallback with simple exponential growth
    return f"exp({r}*t)*x"


def apply_tricomi_operator(ic_expr, dims, parameters):
    """
    Create manufactured solutions for the Tricomi equation: ∂²u/∂x² + x*∂²u/∂y² = 0
    Designed to match solution types from YAML.
    """
    if dims != 2:
        # Tricomi is a 2D operator
        return ic_expr
    
    ic_type = parameters.get('ic_type', 'polynomial')
    lambda_val = parameters.get('lambda', -1.0)  # Parameter
    
    # The Tricomi equation only has one solution type in the YAML
    if ic_type == 'polynomial' or ic_type == 'exp_decay':
        # Simple polynomial solutions with λ parameter
        # Use x³ + λ*x*y² which is a classic solution
        return f"x^3 + {lambda_val}*y^2*x"
    
    elif ic_type == 'exponential':
        # Exponential-based solution (approximation)
        a = parameters.get('a', 1.0)
        b = parameters.get('b', 2.0)
        
        # Approximate solution with manageable forcing
        return f"exp({a}*x + {b}*x^(3/2)*y)"
    
    elif ic_type == 'sine':
        # Approximate oscillatory solution
        k = parameters.get('wavenumbers', [1])[0]
        
        # A solution with oscillations
        return f"x^3*sin({k}*pi*y) + {lambda_val}*x*y^2"
    
    # Fallback with simple form
    return f"x^3 + {lambda_val}*y^2*x"
