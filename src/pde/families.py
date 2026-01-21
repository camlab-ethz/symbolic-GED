"""
PDE Family Definitions
Defines 16 common PDE families with their properties and template functions
"""

from typing import Dict, Callable, Optional
from dataclasses import dataclass
from .normalize import fmt_coeff, canonicalize_operator_str


@dataclass
class PDEFamily:
    """Definition of a PDE family"""

    name: str
    temporal_order: int  # 0 (steady), 1 (first-order), 2 (second-order)
    spatial_order: int  # Highest spatial derivative order
    min_dimension: int  # Minimum spatial dimension required
    supports_nonlinear: bool
    template_fn: Callable  # Function to generate PDE string from coefficients


# ============================================================================
# PDE TEMPLATE FUNCTIONS
# ============================================================================


def _minus_chain(lhs: str, terms: list[str]) -> str:
    """Build 'lhs - t1 - t2 - ...' safely as a plain string.

    This avoids the common string-formatting bug where one writes:
      lhs - (t1 + t2)
    but without parentheses, it becomes:
      lhs - t1 + t2
    """
    if not terms:
        return lhs
    return lhs + " - " + " - ".join(terms)


def heat_template(dim: int, coeffs: Dict) -> str:
    """Heat equation: ∂u/∂t = k∇²u"""
    k = float(coeffs["k"])
    k_str = fmt_coeff(k)

    lhs = "dt(u)"
    spatial_terms = []

    if dim >= 1:
        spatial_terms.append(f"{k_str}*dxx(u)")
    if dim >= 2:
        spatial_terms.append(f"{k_str}*dyy(u)")
    if dim >= 3:
        spatial_terms.append(f"{k_str}*dzz(u)")

    result = f"{_minus_chain(lhs, spatial_terms)}" if spatial_terms else f"{lhs}"
    return canonicalize_operator_str(result)


def wave_template(dim: int, coeffs: Dict) -> str:
    """Wave equation: ∂²u/∂t² = c²∇²u"""
    c_sq = float(coeffs["c_sq"])
    c_sq_str = fmt_coeff(c_sq)

    lhs = "dtt(u)"
    spatial_terms = []

    if dim >= 1:
        spatial_terms.append(f"{c_sq_str}*dxx(u)")
    if dim >= 2:
        spatial_terms.append(f"{c_sq_str}*dyy(u)")
    if dim >= 3:
        spatial_terms.append(f"{c_sq_str}*dzz(u)")

    result = f"{_minus_chain(lhs, spatial_terms)}" if spatial_terms else f"{lhs}"
    return canonicalize_operator_str(result)


def poisson_template(dim: int, coeffs: Dict) -> str:
    """Poisson equation: ∇²u = f"""
    f = float(coeffs["f"])
    f_str = fmt_coeff(abs(f))
    terms = []

    if dim >= 1:
        terms.append("dxx(u)")
    if dim >= 2:
        terms.append("dyy(u)")
    if dim >= 3:
        terms.append("dzz(u)")

    result = " + ".join(terms)
    if f >= 0:
        result = f"{result} - {f_str}"
    else:
        result = f"{result} + {f_str}"
    return canonicalize_operator_str(result)


def advection_template(dim: int, coeffs: Dict) -> str:
    """Advection equation: ∂u/∂t + v·∇u = 0"""
    expr = "dt(u)"

    if dim >= 1 and "v_x" in coeffs:
        v_x = float(coeffs["v_x"])
        v_x_str = fmt_coeff(abs(v_x))
        expr += f" + {v_x_str}*dx(u)" if v_x >= 0 else f" - {v_x_str}*dx(u)"

    if dim >= 2 and "v_y" in coeffs:
        v_y = float(coeffs["v_y"])
        v_y_str = fmt_coeff(abs(v_y))
        expr += f" + {v_y_str}*dy(u)" if v_y >= 0 else f" - {v_y_str}*dy(u)"

    if dim >= 3 and "v_z" in coeffs:
        v_z = float(coeffs["v_z"])
        v_z_str = fmt_coeff(abs(v_z))
        expr += f" + {v_z_str}*dz(u)" if v_z >= 0 else f" - {v_z_str}*dz(u)"

    return canonicalize_operator_str(expr)


def burgers_template(dim: int, coeffs: Dict) -> str:
    """Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²"""
    nu = float(coeffs["nu"])
    nu_str = fmt_coeff(nu)
    result = f"dt(u) + u*dx(u) - {nu_str}*dxx(u)"
    return canonicalize_operator_str(result)


def kdv_template(dim: int, coeffs: Dict) -> str:
    """Korteweg-de Vries equation: ∂u/∂t + u∂u/∂x + δ∂³u/∂x³ = 0"""
    if dim != 1:
        raise ValueError("kdv is 1D-only in this benchmark (dim must be 1)")
    delta = float(coeffs["delta"])
    delta_str = fmt_coeff(delta)
    result = f"dt(u) + u*dx(u) + {delta_str}*dxxx(u)"
    return canonicalize_operator_str(result)


def cubic_reaction_diffusion_template(dim: int, coeffs: Dict) -> str:
    """Cubic reaction-diffusion: u_t - Δu ± g*u^3 = 0

    NOTE: This is NOT the (complex) Schrödinger equation (no i, no complex field).
    We keep it as a separate, clearly named family to avoid confusion/overlap.
    """
    g = float(coeffs["g"])
    g_str = fmt_coeff(abs(g))
    terms = ["dt(u)", "dxx(u)"]

    if dim >= 2:
        terms.append("dyy(u)")
    if dim >= 3:
        terms.append("dzz(u)")

    # Want: dt(u) - dxx(u) - dyy(u) - dzz(u) ...
    lap_part = _minus_chain("dt(u)", terms[1:])

    if g >= 0:
        result = f"{lap_part} + {g_str}*u^3"
    else:
        result = f"{lap_part} - {g_str}*u^3"
    return canonicalize_operator_str(result)


def reaction_diffusion_cubic_template(dim: int, coeffs: Dict) -> str:
    """Alias for backward compatibility (truthful name)."""
    return cubic_reaction_diffusion_template(dim, coeffs)


def allen_cahn_template(dim: int, coeffs: Dict) -> str:
    """Allen-Cahn equation: ∂u/∂t = ε²∇²u + u - u³"""
    eps_sq = float(coeffs["eps_sq"])
    eps_sq_str = fmt_coeff(eps_sq)
    terms = []

    if dim >= 1:
        terms.append(f"{eps_sq_str}*dxx(u)")
    if dim >= 2:
        terms.append(f"{eps_sq_str}*dyy(u)")
    if dim >= 3:
        terms.append(f"{eps_sq_str}*dzz(u)")

    lap_part = _minus_chain("dt(u)", terms)
    result = f"{lap_part} - u + u^3"
    return canonicalize_operator_str(result)


def cahn_hilliard_template(dim: int, coeffs: Dict) -> str:
    """Cahn–Hilliard residual operator (benchmark form).

    PDE: u_t = -γ Δ^2 u + Δ(u^3 - u)
    Residual operator:
      dt(u) + γ Δ^2 u + Δu - Δ(u^3)
    """
    gamma = float(coeffs["gamma"])
    gamma_str = fmt_coeff(gamma)
    # Collapse 2*gamma to single coefficient
    two_gamma = 2.0 * gamma
    two_gamma_str = fmt_coeff(two_gamma)

    if dim == 1:
        result = f"dt(u) + {gamma_str}*dxxxx(u) + dxx(u) - dxx(u^3)"
    elif dim == 2:
        # Δ^2 u in 2D: u_xxxx + 2 u_xxyy + u_yyyy
        # Collapse 2*gamma to single coefficient
        result = (
            f"dt(u) + {gamma_str}*dxxxx(u) + {gamma_str}*dyyyy(u) + {two_gamma_str}*dxxyy(u)"
            f" + dxx(u) + dyy(u) - dxx(u^3) - dyy(u^3)"
        )
    else:
        raise ValueError("cahn_hilliard currently supports dim=1 or dim=2 only")
    
    return canonicalize_operator_str(result)


def fisher_kpp_template(dim: int, coeffs: Dict) -> str:
    """Fisher-KPP equation: ∂u/∂t = D∇²u + ru(1-u)"""
    D = float(coeffs["D"])
    r = float(coeffs["r"])
    D_str = fmt_coeff(D)
    r_str = fmt_coeff(r)
    terms = []

    if dim >= 1:
        terms.append(f"{D_str}*dxx(u)")
    if dim >= 2:
        terms.append(f"{D_str}*dyy(u)")
    if dim >= 3:
        terms.append(f"{D_str}*dzz(u)")

    lap_part = _minus_chain("dt(u)", terms)
    result = f"{lap_part} - {r_str}*u + {r_str}*u^2"
    return canonicalize_operator_str(result)


def kuramoto_sivashinsky_template(dim: int, coeffs: Dict) -> str:
    """Kuramoto-Sivashinsky equation: ∂u/∂t + ν∂²u/∂x² + γ∂⁴u/∂x⁴ + α(∂u/∂x)² = 0"""
    if dim != 1:
        raise ValueError(
            "kuramoto_sivashinsky is 1D-only in this benchmark (dim must be 1)"
        )
    nu = float(coeffs["nu"])
    gamma = float(coeffs["gamma"])
    alpha = float(coeffs["alpha"])
    nu_str = fmt_coeff(nu)
    gamma_str = fmt_coeff(gamma)
    alpha_str = fmt_coeff(alpha)
    # Canonical KS uses u*u_x (not (u_x)^2) as the nonlinearity.
    result = f"dt(u) + {nu_str}*dxx(u) + {gamma_str}*dxxxx(u) + {alpha_str}*u*dx(u)"
    return canonicalize_operator_str(result)


def airy_template(dim: int, coeffs: Dict) -> str:
    """Airy equation (linear dispersive, 1D): u_t + α u_xxx = 0"""
    if dim != 1:
        raise ValueError("airy is 1D-only in this benchmark (dim must be 1)")
    alpha = float(coeffs["alpha"])
    alpha_str = fmt_coeff(alpha)
    result = f"dt(u) + {alpha_str}*dxxx(u)"
    return canonicalize_operator_str(result)


def telegraph_template(dim: int, coeffs: Dict) -> str:
    """Telegraph equation: ∂²u/∂t² + a∂u/∂t - b²∂²u/∂x² = 0"""
    a = float(coeffs["a"])
    b_sq = float(coeffs["b_sq"])
    a_str = fmt_coeff(a)
    b_sq_str = fmt_coeff(b_sq)
    # Multi-D telegraph uses the Laplacian, consistent with the other families.
    terms = [f"{b_sq_str}*dxx(u)"]
    if dim >= 2:
        terms.append(f"{b_sq_str}*dyy(u)")
    if dim >= 3:
        terms.append(f"{b_sq_str}*dzz(u)")
    lap_part = _minus_chain(f"dtt(u) + {a_str}*dt(u)", terms)
    return canonicalize_operator_str(lap_part)


def beam_plate_template(dim: int, coeffs: Dict) -> str:
    """Beam/Plate equation (hyperbolic 4th order).

    1D beam:  u_tt + κ u_xxxx = 0
    2D plate: u_tt + κ (u_xxxx + 2 u_xxyy + u_yyyy) = 0
    """
    kappa = float(coeffs["kappa"])
    kappa_str = fmt_coeff(kappa)
    # Collapse 2*kappa to single coefficient
    two_kappa = 2.0 * kappa
    two_kappa_str = fmt_coeff(two_kappa)
    
    if dim == 1:
        result = f"dtt(u) + {kappa_str}*dxxxx(u)"
    elif dim == 2:
        # Collapse 2*kappa to single coefficient
        result = f"dtt(u) + {kappa_str}*dxxxx(u) + {two_kappa_str}*dxxyy(u) + {kappa_str}*dyyyy(u)"
    else:
        raise ValueError("beam_plate currently supports dim=1 or dim=2 only")
    return canonicalize_operator_str(result)


def biharmonic_template(dim: int, coeffs: Dict) -> str:
    """Biharmonic equation: ∇⁴u = f"""
    f = float(coeffs["f"])
    f_str = fmt_coeff(abs(f))

    if dim == 1:
        if f >= 0:
            result = f"dxxxx(u) - {f_str}"
        else:
            result = f"dxxxx(u) + {f_str}"
    elif dim == 2:
        if f >= 0:
            # Δ^2 u in 2D: u_xxxx + 2 u_xxyy + u_yyyy
            result = f"dxxxx(u) + dyyyy(u) + 2.000*dxxyy(u) - {f_str}"
        else:
            result = f"dxxxx(u) + dyyyy(u) + 2.000*dxxyy(u) + {f_str}"
    else:
        if f >= 0:
            result = f"dxxxx(u) - {f_str}"
        else:
            result = f"dxxxx(u) + {f_str}"
    return canonicalize_operator_str(result)


def sine_gordon_template(dim: int, coeffs: Dict) -> str:
    """Sine-Gordon equation (true): u_tt - c^2 Δu + beta*sin(u) = 0"""
    c_sq = float(coeffs["c_sq"])
    beta = float(coeffs["beta"])
    c_sq_str = fmt_coeff(c_sq)
    beta_str = fmt_coeff(beta)

    spatial_terms = [f"{c_sq_str}*dxx(u)"]
    if dim >= 2:
        spatial_terms.append(f"{c_sq_str}*dyy(u)")
    if dim >= 3:
        spatial_terms.append(f"{c_sq_str}*dzz(u)")

    lap_part = _minus_chain("dtt(u)", spatial_terms)
    result = f"{lap_part} + {beta_str}*sin(u)"
    return canonicalize_operator_str(result)


# ============================================================================
# PDE FAMILY REGISTRY
# ============================================================================

PDE_FAMILIES = {
    "heat": PDEFamily(
        name="heat",
        temporal_order=1,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=heat_template,
    ),
    "wave": PDEFamily(
        name="wave",
        temporal_order=2,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=wave_template,
    ),
    "poisson": PDEFamily(
        name="poisson",
        temporal_order=0,  # Steady-state
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=poisson_template,
    ),
    "advection": PDEFamily(
        name="advection",
        temporal_order=1,
        spatial_order=1,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=advection_template,
    ),
    "burgers": PDEFamily(
        name="burgers",
        temporal_order=1,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=burgers_template,
    ),
    "kdv": PDEFamily(
        name="kdv",
        temporal_order=1,
        spatial_order=3,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=kdv_template,
    ),
    "reaction_diffusion_cubic": PDEFamily(
        name="reaction_diffusion_cubic",
        temporal_order=1,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=reaction_diffusion_cubic_template,
    ),
    "allen_cahn": PDEFamily(
        name="allen_cahn",
        temporal_order=1,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=allen_cahn_template,
    ),
    "cahn_hilliard": PDEFamily(
        name="cahn_hilliard",
        temporal_order=1,
        spatial_order=4,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=cahn_hilliard_template,
    ),
    "fisher_kpp": PDEFamily(
        name="fisher_kpp",
        temporal_order=1,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=fisher_kpp_template,
    ),
    "kuramoto_sivashinsky": PDEFamily(
        name="kuramoto_sivashinsky",
        temporal_order=1,
        spatial_order=4,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=kuramoto_sivashinsky_template,
    ),
    "airy": PDEFamily(
        name="airy",
        temporal_order=1,
        spatial_order=3,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=airy_template,
    ),
    "telegraph": PDEFamily(
        name="telegraph",
        temporal_order=2,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=telegraph_template,
    ),
    "biharmonic": PDEFamily(
        name="biharmonic",
        temporal_order=0,  # Steady-state
        spatial_order=4,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=biharmonic_template,
    ),
    "sine_gordon": PDEFamily(
        name="sine_gordon",
        temporal_order=2,
        spatial_order=2,
        min_dimension=1,
        supports_nonlinear=True,
        template_fn=sine_gordon_template,
    ),
    "beam_plate": PDEFamily(
        name="beam_plate",
        temporal_order=2,
        spatial_order=4,
        min_dimension=1,
        supports_nonlinear=False,
        template_fn=beam_plate_template,
    ),
}


def get_family(name: str) -> Optional[PDEFamily]:
    """Get PDE family by name"""
    return PDE_FAMILIES.get(name)


def list_families() -> list:
    """List all available PDE families"""
    return list(PDE_FAMILIES.keys())


if __name__ == "__main__":
    # Test family definitions
    print("Available PDE Families:")
    print("=" * 80)
    for name, family in PDE_FAMILIES.items():
        print(f"\n{name}:")
        print(f"  Temporal order: {family.temporal_order}")
        print(f"  Spatial order: {family.spatial_order}")
        print(f"  Min dimension: {family.min_dimension}")
        print(f"  Nonlinear: {family.supports_nonlinear}")

        # Generate example
        if name == "heat":
            example = family.template_fn(2, {"k": 1.5})
        elif name == "wave":
            example = family.template_fn(2, {"c_sq": 2.0})
        elif name == "poisson":
            example = family.template_fn(2, {"f": 3.0})
        elif name == "advection":
            example = family.template_fn(2, {"v_x": 1.0, "v_y": 0.5})
        elif name == "burgers":
            example = family.template_fn(1, {"nu": 0.1})
        elif name == "kdv":
            example = family.template_fn(1, {"delta": 0.022})
        elif name == "reaction_diffusion_cubic":
            example = family.template_fn(1, {"g": 1.0})
        elif name == "allen_cahn":
            example = family.template_fn(1, {"eps_sq": 0.01})
        elif name == "cahn_hilliard":
            example = family.template_fn(1, {"gamma": 0.01})
        elif name == "fisher_kpp":
            example = family.template_fn(1, {"D": 1.0, "r": 1.0})
        elif name == "kuramoto_sivashinsky":
            example = family.template_fn(1, {"nu": 1.0, "gamma": 0.01, "alpha": 1.0})
        elif name == "airy":
            example = family.template_fn(1, {"alpha": 0.5})
        elif name == "telegraph":
            example = family.template_fn(1, {"a": 1.0, "b_sq": 4.0})
        elif name == "biharmonic":
            example = family.template_fn(1, {"f": 1.0})
        elif name == "sine_gordon":
            example = family.template_fn(1, {"c_sq": 1.0, "beta": 0.1})
        elif name == "beam_plate":
            example = family.template_fn(2, {"kappa": 0.1})
        else:
            example = "N/A"

        print(f"  Example (operator-only): {example}")

    print(f"\n{'=' * 80}")
    print(f"Total families: {len(PDE_FAMILIES)}")
