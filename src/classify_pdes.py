"""
Classify PDEs into families based on structural features.
Analyzes PDE strings to determine which family they belong to.
"""

import re
from typing import Dict, Optional


def analyze_pde_structure(pde: str) -> Dict[str, any]:
    """Extract structural features from a PDE string."""
    features = {
        'has_dt': 'dt(u)' in pde,
        'has_dtt': 'dtt(u)' in pde,
        'has_dx': 'dx(u)' in pde,
        'has_dy': 'dy(u)' in pde,
        'has_dz': 'dz(u)' in pde,
        'has_dxx': 'dxx(u)' in pde,
        'has_dyy': 'dyy(u)' in pde,
        'has_dzz': 'dzz(u)' in pde,
        'has_dxxx': 'dxxx(u)' in pde,
        'has_dyyy': 'dyyy(u)' in pde,
        'has_dzzz': 'dzzz(u)' in pde,
        'has_dxxxx': 'dxxxx(u)' in pde,
        'has_dyyyy': 'dyyyy(u)' in pde,
        'has_dzzzz': 'dzzzz(u)' in pde,
        'has_dxxyy': 'dxxyy(u)' in pde,
        'has_dxxzz': 'dxxzz(u)' in pde,
        'has_dyyzz': 'dyyzz(u)' in pde,
        'has_u_dx': 'u*dx(u)' in pde,
        'has_u_dy': 'u*dy(u)' in pde,
        'has_u_dz': 'u*dz(u)' in pde,
        'has_u2': 'u^2' in pde,
        'has_u3': 'u^3' in pde,
        'has_dx_squared': '(dx(u))^2' in pde,
        'has_dy_squared': '(dy(u))^2' in pde,
        'has_dz_squared': '(dz(u))^2' in pde,
        'has_laplacian': ('dxx(u)' in pde or 'dyy(u)' in pde or 'dzz(u)' in pde),
        'has_biharmonic': ('dxxxx(u)' in pde or 'dyyyy(u)' in pde or 'dzzzz(u)' in pde),
        'has_mixed_4th': ('dxxyy(u)' in pde or 'dxxzz(u)' in pde or 'dyyzz(u)' in pde),
        'num_spatial_dims': sum([('dx(u)' in pde or 'dxx(u)' in pde),
                                  ('dy(u)' in pde or 'dyy(u)' in pde),
                                  ('dz(u)' in pde or 'dzz(u)' in pde)]),
    }
    
    # Check for constant terms (coefficients without derivatives or u)
    # Pattern: number followed by space and = 0
    features['has_constant'] = bool(re.search(r'[-+]\s*\d+\.\d+\s*=\s*0', pde))
    
    return features


def classify_pde(pde: str) -> str:
    """Classify a PDE into one of the 16 families based on structural features."""
    f = analyze_pde_structure(pde)
    
    # Heat equation: dt(u) = k*Δu (no nonlinear terms)
    if (f['has_dt'] and not f['has_dtt'] and f['has_laplacian'] and 
        not f['has_u_dx'] and not f['has_u2'] and not f['has_u3'] and 
        not f['has_dx_squared'] and not f['has_biharmonic'] and not f['has_dxxx']):
        return 'heat'
    
    # Wave equation: dtt(u) = c²*Δu (no dt, no nonlinear)
    if (f['has_dtt'] and not f['has_dt'] and f['has_laplacian'] and 
        not f['has_u_dx'] and not f['has_u2'] and not f['has_u3'] and 
        not f['has_dx_squared'] and not f['has_biharmonic']):
        return 'wave'
    
    # Poisson equation: Δu = f (no temporal derivatives)
    if (not f['has_dt'] and not f['has_dtt'] and f['has_laplacian'] and 
        not f['has_biharmonic'] and f['has_constant'] and 
        not f['has_u_dx'] and not f['has_u2'] and not f['has_u3']):
        return 'poisson'
    
    # Advection equation: dt(u) + v·∇u = 0 (first order spatial only)
    if (f['has_dt'] and not f['has_dtt'] and 
        (f['has_dx'] or f['has_dy'] or f['has_dz']) and 
        not f['has_laplacian'] and not f['has_u_dx'] and not f['has_u2'] and 
        not f['has_u3'] and not f['has_dx_squared']):
        return 'advection'
    
    # Burgers equation: dt(u) + u*∇u = ν*Δu
    if (f['has_dt'] and not f['has_dtt'] and f['has_laplacian'] and 
        (f['has_u_dx'] or f['has_u_dy'] or f['has_u_dz']) and 
        not f['has_u2'] and not f['has_u3'] and not f['has_dx_squared'] and 
        not f['has_dxxx'] and not f['has_biharmonic']):
        return 'burgers'
    
    # KdV equation: dt(u) + u*dx(u) + δ*dxxx(u) = 0
    if (f['has_dt'] and not f['has_dtt'] and f['has_dxxx'] and f['has_u_dx'] and 
        not f['has_u2'] and not f['has_u3'] and not f['has_laplacian']):
        return 'kdv'
    
    # Check for linear u term (used by both Allen-Cahn and Fisher-KPP)
    has_linear_u = bool(re.search(r'[-+]\s*\d+\.\d+\*u(?:\s|=)', pde) or 
                        re.search(r'[-+]\s*u(?:\s|=)', pde))
    
    # Allen-Cahn equation: dt(u) = ε²*Δu + u - u³
    # This has both u and u³ terms (distinguishes from Schrödinger)
    # MUST CHECK BEFORE Schrödinger since both have u³
    if (f['has_dt'] and not f['has_dtt'] and f['has_laplacian'] and f['has_u3'] and 
        has_linear_u and not f['has_u_dx'] and not f['has_biharmonic']):
        return 'allen_cahn'
    
    # Cubic reaction-diffusion: dt(u) = Δu ± g*u³ (NO linear u term)
    if (f['has_dt'] and not f['has_dtt'] and f['has_laplacian'] and f['has_u3'] and 
        not has_linear_u and not f['has_u_dx'] and not f['has_u2'] and 
        not f['has_dx_squared'] and not f['has_biharmonic']):
        return 'reaction_diffusion_cubic'
    
    # Cahn-Hilliard equation: dt(u) = -γ*Δ²u (4th order, may have mixed)
    if (f['has_dt'] and not f['has_dtt'] and 
        (f['has_biharmonic'] or f['has_mixed_4th']) and 
        not f['has_u_dx'] and not f['has_u2'] and not f['has_u3'] and 
        not f['has_dx_squared'] and not f['has_laplacian']):
        return 'cahn_hilliard'
    
    # Fisher-KPP equation: dt(u) = D*Δu + r*u - r*u²
    # Sine-Gordon equation (true): dtt(u) - c^2*Δu + beta*sin(u) = 0
    if (f['has_dtt'] and f['has_laplacian'] and bool(re.search(r'sin\\(u\\)', pde))):
        return 'sine_gordon'

    if (f['has_dt'] and not f['has_dtt'] and f['has_laplacian'] and f['has_u2'] and 
        has_linear_u and not f['has_u3'] and not f['has_u_dx'] and 
        not f['has_biharmonic']):
        return 'fisher_kpp'
    
    # Kuramoto-Sivashinsky: dt(u) + ν*dxx(u) + γ*dxxxx(u) + α*(dx(u))²
    if (f['has_dt'] and not f['has_dtt'] and f['has_dx_squared'] and f['has_dxxxx'] and 
        not f['has_u_dx'] and not f['has_u2'] and not f['has_u3']):
        return 'kuramoto_sivashinsky'
    
    # Klein-Gordon equation: dtt(u) = c²*Δu - m²*u
    # Klein-Gordon removed from the 16-family dataset
    
    # Telegraph equation: dtt(u) + a*dt(u) = b²*Δu
    if (f['has_dtt'] and f['has_dt'] and f['has_laplacian'] and 
        not f['has_u_dx'] and not f['has_u2'] and not f['has_u3']):
        return 'telegraph'
    
    # Biharmonic equation: Δ²u = f (no temporal, 4th order spatial)
    if (not f['has_dt'] and not f['has_dtt'] and 
        (f['has_biharmonic'] or f['has_mixed_4th']) and 
        f['has_constant'] and not f['has_laplacian']):
        return 'biharmonic'
    
    # Sine-Gordon equation: dtt(u) = Δu - u + β*u³
    if (f['has_dtt'] and not f['has_dt'] and f['has_laplacian'] and 
        f['has_u3'] and has_linear_u and not f['has_u_dx']):
        return 'sine_gordon'
    
    # Default: unknown
    return 'unknown'


def classify_dataset(pde_list: list[str]) -> list[str]:
    """Classify a list of PDEs."""
    return [classify_pde(pde) for pde in pde_list]


if __name__ == '__main__':
    # Test classification
    test_pdes = [
        'dt(u) - 1.935*dxx(u) = 0',  # heat
        'dtt(u) - 4.759*dxx(u) - 4.759*dyy(u) = 0',  # wave
        'dt(u) - 1.467*dxx(u) + u*dx(u) = 0',  # burgers
        'dt(u) + 0.808*dxxx(u) + u*dx(u) = 0',  # kdv
        'dt(u) - dxx(u) - 0.342*u^3 = 0',  # schrodinger
        'dt(u) + 1.104*dxx(u) + 1.805*dxxxx(u) + 0.981*(dx(u))^2 = 0',  # kuramoto_sivashinsky
    ]
    
    for pde in test_pdes:
        category = classify_pde(pde)
        print(f"{category:20s} | {pde}")
