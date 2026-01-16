"""Motif Library for Manufactured Solutions - Physical motifs M1-M9"""
import numpy as np
from typing import Dict
from sympy import symbols, sin, cos, exp, tanh, pi, Float, Expr, S

x, y, t = symbols('x y t', real=True)

def sample_motif_params(motif_type: str, rng: np.random.Generator, dim: int = 1, temporal_order: int = 1) -> Dict:
    params = {}
    params['A'] = float(rng.uniform(-2.0, 2.0))
    params['phi'] = float(rng.uniform(0, 2 * np.pi))
    params['k'] = int(rng.integers(1, 7))
    params['kx'] = int(rng.integers(1, 7))
    params['ky'] = int(rng.integers(1, 7))
    if temporal_order > 0:
        params['omega'] = float(rng.uniform(0.5, 8.0))
        params['c'] = float(rng.uniform(-2.0, 2.0))
        params['lambda_decay'] = float(rng.uniform(0.5, 8.0))
    params['alpha'] = float(np.exp(rng.uniform(np.log(2.0), np.log(50.0))))
    params['delta'] = float(np.exp(rng.uniform(np.log(0.03), np.log(0.2))))
    params['x0'] = float(rng.uniform(0.15, 0.85))
    params['y0'] = float(rng.uniform(0.15, 0.85))
    params['spectral_decay_p'] = float(rng.uniform(1.0, 2.5))
    params['n_modes'] = int(rng.integers(2, 5))
    params['k0'] = int(rng.integers(1, 4))
    params['k1'] = float(rng.uniform(0.5, 2.0))
    params['a'] = float(rng.uniform(-1.0, 1.0))
    params['b'] = float(rng.uniform(-1.0, 1.0))
    return params

def M1_modal_wave(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, k, phi = Float(params['A']), params['k'], params['phi']
    if temporal_order == 0:
        return A * sin(2*pi*(params['kx']*x + params['ky']*y) + phi) if dim > 1 else A * sin(2*pi*k*x + phi)
    omega = params['omega']
    if phi < np.pi:
        return A * sin(2*pi*(params['kx']*x + params['ky']*y - omega*t) + phi) if dim > 1 else A * sin(2*pi*(k*x - omega*t) + phi)
    return A * sin(2*pi*params['kx']*x) * sin(2*pi*params['ky']*y) * cos(2*pi*omega*t) if dim > 1 else A * sin(2*pi*k*x) * cos(2*pi*omega*t)

def M2_diffusion_decay(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, k = Float(params['A']), params['k']
    if temporal_order == 0:
        return A * sin(2*pi*params['kx']*x) * sin(2*pi*params['ky']*y) if dim > 1 else A * sin(2*pi*k*x)
    lam = params['lambda_decay']
    return A * sin(2*pi*params['kx']*x) * sin(2*pi*params['ky']*y) * exp(-lam*t) if dim > 1 else A * sin(2*pi*k*x) * exp(-lam*t)

def M3_gabor_packet(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, alpha, x0, k, phi = Float(params['A']), params['alpha'], params['x0'], params['k'], params['phi']
    if temporal_order == 0:
        if dim == 1:
            return A * exp(-alpha*(x - x0)**2) * sin(2*pi*k*(x - x0) + phi)
        y0 = params['y0']
        return A * exp(-alpha*((x - x0)**2 + (y - y0)**2)) * sin(2*pi*(params['kx']*(x - x0) + params['ky']*(y - y0)) + phi)
    c, omega = params['c'], params['omega']
    if dim == 1:
        return A * exp(-alpha*(x - x0 - c*t)**2) * sin(2*pi*k*(x - x0) - 2*pi*omega*t + phi)
    y0 = params['y0']
    return A * exp(-alpha*((x - x0 - c*t)**2 + (y - y0)**2)) * sin(2*pi*(params['kx']*(x - x0) + params['ky']*(y - y0)) - 2*pi*omega*t + phi)

def M4_gaussian_load(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, alpha, x0 = Float(params['A']), params['alpha'], params['x0']
    base = exp(-alpha*(x - x0)**2) if dim == 1 else exp(-alpha*((x - x0)**2 + (y - params['y0'])**2))
    if temporal_order > 0:
        base = base * (1 + Float(0.25) * cos(2*pi*params['omega']*t))
    return A * base

def M5_front(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, delta, x0 = Float(params['A']), params['delta'], params['x0']
    if temporal_order == 0:
        return A * tanh((params['kx']*x + params['ky']*y - x0) / delta) if dim > 1 else A * tanh((x - x0) / delta)
    c = params['c']
    return A * tanh((params['kx']*x + params['ky']*y - c*t - x0) / delta) if dim > 1 else A * tanh((x - c*t - x0) / delta)

def M6_multiscale_fourier(params: Dict, dim: int, temporal_order: int, rng: np.random.Generator = None) -> Expr:
    if rng is None:
        raise ValueError("M6 requires rng for reproducibility")
    p, n_modes = params['spectral_decay_p'], params['n_modes']
    result = S(0)
    for _ in range(n_modes):
        kx_n = int(rng.integers(1, 7))
        ky_n = int(rng.integers(1, 7)) if dim > 1 else 0
        k_norm = float(np.sqrt(kx_n**2 + ky_n**2)) if dim > 1 else float(kx_n)
        sign = 1 if rng.random() > 0.5 else -1
        a_n = Float(sign * float(rng.uniform(0.3, 1.0)) / (k_norm ** p))
        phi_n = float(rng.uniform(0, 2 * np.pi))
        if temporal_order == 0:
            term = a_n * sin(2*pi*(kx_n*x + ky_n*y) + phi_n) if dim > 1 else a_n * sin(2*pi*kx_n*x + phi_n)
        else:
            omega_n = float(rng.uniform(0.5, 4.0))
            term = a_n * sin(2*pi*(kx_n*x + ky_n*y - omega_n*t) + phi_n) if dim > 1 else a_n * sin(2*pi*(kx_n*x - omega_n*t) + phi_n)
        result = result + term
    return result

def M7_separable_product(params: Dict, dim: int, temporal_order: int) -> Expr:
    a, b, kx, ky = Float(params['a']), Float(params['b']), params['kx'], params['ky']
    result = 1 + a * sin(2*pi*kx*x)
    if dim > 1:
        result = result * (1 + b * cos(2*pi*ky*y))
    if temporal_order > 0:
        result = result * (1 + Float(params['A'] / 4.0) * cos(2*pi*params['omega']*t))
    return result

def M8_chirp(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, k0, k1, phi = Float(params['A']), params['k0'], params['k1'], params['phi']
    spatial = 2*pi*(k0*x + k1*x**2 + (params['ky']*y if dim > 1 else 0)) + phi
    if temporal_order > 0:
        spatial = spatial - 2*pi*params['omega']*t
    return A * sin(spatial)

def M9_rational_bump(params: Dict, dim: int, temporal_order: int) -> Expr:
    A, alpha, x0 = Float(params['A']), params['alpha'], params['x0']
    denom = 1 + alpha*((x - x0)**2 + ((y - params['y0'])**2 if dim > 1 else 0))
    result = A / denom
    if temporal_order > 0:
        result = result * (1 + Float(0.2) * sin(2*pi*params['omega']*t))
    return result

def boundary_mask(dim: int) -> Expr:
    return x * (1 - x) * y * (1 - y) if dim > 1 else x * (1 - x)

MOTIF_FUNCTIONS = {'M1': M1_modal_wave, 'M2': M2_diffusion_decay, 'M3': M3_gabor_packet,
    'M4': M4_gaussian_load, 'M5': M5_front, 'M6': M6_multiscale_fourier,
    'M7': M7_separable_product, 'M8': M8_chirp, 'M9': M9_rational_bump}

def sample_motif(motif_name: str, rng: np.random.Generator, dim: int, temporal_order: int) -> Expr:
    if motif_name not in MOTIF_FUNCTIONS:
        raise ValueError(f"Unknown motif: {motif_name}")
    params = sample_motif_params(motif_name, rng, dim, temporal_order)
    if motif_name == 'M6':
        return MOTIF_FUNCTIONS[motif_name](params, dim, temporal_order, rng=rng)
    return MOTIF_FUNCTIONS[motif_name](params, dim, temporal_order)
