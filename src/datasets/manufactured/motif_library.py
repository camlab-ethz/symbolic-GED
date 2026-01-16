"""
Motif Library for Manufactured Solutions

Physical motifs M1-M9 for generating u expressions that cover canonical PDE regimes:
- M1: Modal waves (traveling and standing)
- M2: Diffusion-smoothed modes (exponential decay)
- M3: Transported localized packet (Gabor)
- M4: Localized Gaussian load
- M5: Front / interface (tanh)
- M6: Multi-scale Fourier field
- M7: Mixed separable product
- M8: Chirp (nonstationary frequency)
- M9: Rational smooth bump
"""

import numpy as np
from typing import Dict, Optional
from sympy import (
    symbols, sin, cos, exp, tanh, pi, sqrt, Rational,
    Expr, S
)

# Define symbolic variables
x, y, t = symbols('x y t', real=True)


def sample_motif_params(motif_type: str, rng: np.random.Generator, 
                        dim: int = 1, temporal_order: int = 1) -> Dict:
    """
    Sample continuous parameters for a motif.
    
    Uses physically plausible ranges as specified in the plan.
    
    Args:
        motif_type: One of M1-M9
        rng: NumPy random generator
        dim: Spatial dimension
        temporal_order: 0 (steady), 1, or 2
    
    Returns:
        Dictionary of parameter values
    """
    params = {}
    
    # Common parameters
    params['A'] = rng.uniform(-2.0, 2.0)
    params['phi'] = rng.uniform(0, 2 * np.pi)
    params['k'] = rng.integers(1, 7)  # mode index 1-6
    params['kx'] = rng.integers(1, 7)
    params['ky'] = rng.integers(1, 7)
    
    if temporal_order > 0:
        params['omega'] = rng.uniform(0.5, 8.0)
        params['c'] = rng.uniform(-2.0, 2.0)  # advection speed
        params['lambda_decay'] = rng.uniform(0.5, 8.0)
    
    # Spatial localization
    params['alpha'] = np.exp(rng.uniform(np.log(2.0), np.log(50.0)))  # LogUniform
    params['delta'] = np.exp(rng.uniform(np.log(0.03), np.log(0.2)))  # LogUniform
    params['x0'] = rng.uniform(0.15, 0.85)
    params['y0'] = rng.uniform(0.15, 0.85)
    
    # Multi-scale parameters
    params['spectral_decay_p'] = rng.uniform(1.0, 2.5)
    params['n_modes'] = rng.integers(2, 5)  # 2-4 modes for M6
    
    # Chirp parameters
    params['k0'] = rng.integers(1, 4)
    params['k1'] = rng.uniform(0.5, 2.0)
    
    # Additional amplitudes for products
    params['a'] = rng.uniform(-1.0, 1.0)
    params['b'] = rng.uniform(-1.0, 1.0)
    
    return params


def M1_modal_wave(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M1: Modal wave (traveling or standing).
    
    Traveling: sin(2*pi*(k*x - omega*t) + phi)
    Standing: sin(2*pi*k*x) * cos(2*pi*omega*t)
    """
    A = Rational(params['A']).limit_denominator(1000)
    k = params['k']
    phi = params['phi']
    
    if temporal_order == 0:
        # Steady mode
        if dim == 1:
            return A * sin(2*pi*k*x + phi)
        else:
            kx, ky = params['kx'], params['ky']
            return A * sin(2*pi*(kx*x + ky*y) + phi)
    else:
        omega = params['omega']
        # Choose traveling or standing randomly based on phi
        if phi < np.pi:
            # Traveling wave
            if dim == 1:
                return A * sin(2*pi*(k*x - omega*t) + phi)
            else:
                kx, ky = params['kx'], params['ky']
                return A * sin(2*pi*(kx*x + ky*y - omega*t) + phi)
        else:
            # Standing wave
            if dim == 1:
                return A * sin(2*pi*k*x) * cos(2*pi*omega*t)
            else:
                kx, ky = params['kx'], params['ky']
                return A * sin(2*pi*kx*x) * sin(2*pi*ky*y) * cos(2*pi*omega*t)


def M2_diffusion_decay(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M2: Diffusion-smoothed mode with exponential decay.
    
    sin(2*pi*k*x) * exp(-lambda*t)
    """
    A = Rational(params['A']).limit_denominator(1000)
    k = params['k']
    
    if temporal_order == 0:
        # Without time: just a mode
        if dim == 1:
            return A * sin(2*pi*k*x)
        else:
            kx, ky = params['kx'], params['ky']
            return A * sin(2*pi*kx*x) * sin(2*pi*ky*y)
    else:
        lam = params['lambda_decay']
        if dim == 1:
            return A * sin(2*pi*k*x) * exp(-lam*t)
        else:
            kx, ky = params['kx'], params['ky']
            return A * sin(2*pi*kx*x) * sin(2*pi*ky*y) * exp(-lam*t)


def M3_gabor_packet(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M3: Transported localized packet (Gabor wavelet).
    
    exp(-alpha*(x-x0-c*t)^2) * sin(2*pi*k*(x-x0) - 2*pi*omega*t + phi)
    """
    A = Rational(params['A']).limit_denominator(1000)
    alpha = params['alpha']
    x0 = params['x0']
    k = params['k']
    phi = params['phi']
    
    if temporal_order == 0:
        # Static packet
        if dim == 1:
            envelope = exp(-alpha*(x - x0)**2)
            oscillation = sin(2*pi*k*(x - x0) + phi)
            return A * envelope * oscillation
        else:
            y0 = params['y0']
            envelope = exp(-alpha*((x - x0)**2 + (y - y0)**2))
            kx, ky = params['kx'], params['ky']
            oscillation = sin(2*pi*(kx*(x - x0) + ky*(y - y0)) + phi)
            return A * envelope * oscillation
    else:
        c = params['c']
        omega = params['omega']
        if dim == 1:
            envelope = exp(-alpha*(x - x0 - c*t)**2)
            oscillation = sin(2*pi*k*(x - x0) - 2*pi*omega*t + phi)
            return A * envelope * oscillation
        else:
            y0 = params['y0']
            envelope = exp(-alpha*((x - x0 - c*t)**2 + (y - y0)**2))
            kx, ky = params['kx'], params['ky']
            oscillation = sin(2*pi*(kx*(x - x0) + ky*(y - y0)) - 2*pi*omega*t + phi)
            return A * envelope * oscillation


def M4_gaussian_load(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M4: Localized Gaussian load/source.
    
    exp(-alpha*((x-x0)^2 + (y-y0)^2))
    """
    A = Rational(params['A']).limit_denominator(1000)
    alpha = params['alpha']
    x0 = params['x0']
    
    if dim == 1:
        base = exp(-alpha*(x - x0)**2)
    else:
        y0 = params['y0']
        base = exp(-alpha*((x - x0)**2 + (y - y0)**2))
    
    if temporal_order > 0:
        # Add mild temporal modulation
        omega = params['omega']
        base = base * (1 + Rational(1, 4) * cos(2*pi*omega*t))
    
    return A * base


def M5_front(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M5: Front / interface (tanh or logistic).
    
    tanh((x - c*t - x0) / delta)
    """
    A = Rational(params['A']).limit_denominator(1000)
    delta = params['delta']
    x0 = params['x0']
    
    if temporal_order == 0:
        # Static front
        if dim == 1:
            return A * tanh((x - x0) / delta)
        else:
            kx, ky = params['kx'], params['ky']
            s0 = x0  # Use x0 as offset
            return A * tanh((kx*x + ky*y - s0) / delta)
    else:
        c = params['c']
        if dim == 1:
            return A * tanh((x - c*t - x0) / delta)
        else:
            kx, ky = params['kx'], params['ky']
            s0 = x0
            return A * tanh((kx*x + ky*y - c*t - s0) / delta)


def M6_multiscale_fourier(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M6: Multi-scale Fourier field with spectral decay.
    
    sum_{n=1..K} a_n * sin(2*pi*(kx_n*x + ky_n*y - omega_n*t) + phi_n)
    with a_n ~ 1/||k_n||^p
    """
    p = params['spectral_decay_p']
    n_modes = params['n_modes']
    
    result = S(0)
    rng = np.random.default_rng(int(abs(params['A']) * 1e6) % (2**31))
    
    for n in range(1, n_modes + 1):
        # Sample mode indices
        kx_n = rng.integers(1, 7)
        if dim == 2:
            ky_n = rng.integers(1, 7)
            k_norm = np.sqrt(kx_n**2 + ky_n**2)
        else:
            ky_n = 0
            k_norm = kx_n
        
        # Amplitude with spectral decay
        a_n = Rational(1, int(k_norm**p * 10)).limit_denominator(100)
        phi_n = rng.uniform(0, 2 * np.pi)
        
        if temporal_order == 0:
            if dim == 1:
                term = a_n * sin(2*pi*kx_n*x + phi_n)
            else:
                term = a_n * sin(2*pi*(kx_n*x + ky_n*y) + phi_n)
        else:
            omega_n = rng.uniform(0.5, 4.0)
            if dim == 1:
                term = a_n * sin(2*pi*(kx_n*x - omega_n*t) + phi_n)
            else:
                term = a_n * sin(2*pi*(kx_n*x + ky_n*y - omega_n*t) + phi_n)
        
        result = result + term
    
    return result


def M7_separable_product(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M7: Mixed separable product (excites mixed derivatives).
    
    (1 + a*sin(2*pi*kx*x)) * (1 + b*cos(2*pi*ky*y)) * (1 + c*cos(2*pi*omega*t))
    """
    a = Rational(params['a']).limit_denominator(100)
    b = Rational(params['b']).limit_denominator(100)
    kx = params['kx']
    ky = params['ky']
    
    x_part = 1 + a * sin(2*pi*kx*x)
    
    if dim == 1:
        result = x_part
    else:
        y_part = 1 + b * cos(2*pi*ky*y)
        result = x_part * y_part
    
    if temporal_order > 0:
        omega = params['omega']
        c_amp = Rational(params['A'] / 4).limit_denominator(100)
        t_part = 1 + c_amp * cos(2*pi*omega*t)
        result = result * t_part
    
    return result


def M8_chirp(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M8: Chirp (nonstationary frequency).
    
    sin(2*pi*(k0*x + k1*x^2) - 2*pi*omega*t + phi)
    """
    A = Rational(params['A']).limit_denominator(1000)
    k0 = params['k0']
    k1 = params['k1']
    phi = params['phi']
    
    if temporal_order == 0:
        if dim == 1:
            return A * sin(2*pi*(k0*x + k1*x**2) + phi)
        else:
            return A * sin(2*pi*(k0*x + k1*x**2 + params['ky']*y) + phi)
    else:
        omega = params['omega']
        if dim == 1:
            return A * sin(2*pi*(k0*x + k1*x**2) - 2*pi*omega*t + phi)
        else:
            return A * sin(2*pi*(k0*x + k1*x**2 + params['ky']*y) - 2*pi*omega*t + phi)


def M9_rational_bump(params: Dict, dim: int, temporal_order: int) -> Expr:
    """
    M9: Rational smooth bump.
    
    a / (1 + alpha*((x-x0)^2 + (y-y0)^2))
    """
    A = Rational(params['A']).limit_denominator(1000)
    alpha = params['alpha']
    x0 = params['x0']
    
    if dim == 1:
        denominator = 1 + alpha*(x - x0)**2
    else:
        y0 = params['y0']
        denominator = 1 + alpha*((x - x0)**2 + (y - y0)**2)
    
    result = A / denominator
    
    if temporal_order > 0:
        # Add mild temporal oscillation
        omega = params['omega']
        result = result * (1 + Rational(1, 5) * sin(2*pi*omega*t))
    
    return result


def boundary_mask(dim: int) -> Expr:
    """
    Boundary mask for Dirichlet-like behavior.
    
    1D: x*(1-x)
    2D: x*(1-x)*y*(1-y)
    """
    if dim == 1:
        return x * (1 - x)
    else:
        return x * (1 - x) * y * (1 - y)


# Mapping of motif names to functions
MOTIF_FUNCTIONS = {
    'M1': M1_modal_wave,
    'M2': M2_diffusion_decay,
    'M3': M3_gabor_packet,
    'M4': M4_gaussian_load,
    'M5': M5_front,
    'M6': M6_multiscale_fourier,
    'M7': M7_separable_product,
    'M8': M8_chirp,
    'M9': M9_rational_bump,
}


def sample_motif(motif_name: str, rng: np.random.Generator, 
                 dim: int, temporal_order: int) -> Expr:
    """
    Sample a single motif with random parameters.
    
    Args:
        motif_name: One of M1-M9
        rng: NumPy random generator
        dim: Spatial dimension
        temporal_order: 0 (steady), 1, or 2
    
    Returns:
        SymPy expression
    """
    if motif_name not in MOTIF_FUNCTIONS:
        raise ValueError(f"Unknown motif: {motif_name}")
    
    params = sample_motif_params(motif_name, rng, dim, temporal_order)
    return MOTIF_FUNCTIONS[motif_name](params, dim, temporal_order)
