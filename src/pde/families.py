"""
PDE Family Definitions
Defines 16 common PDE families with their properties and template functions
"""

from typing import Dict, Callable, Optional
from dataclasses import dataclass


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
    k = coeffs["k"]

    lhs = "dt(u)"
    spatial_terms = []

    if dim >= 1:
        spatial_terms.append(f"{k}*dxx(u)")
    if dim >= 2:
        spatial_terms.append(f"{k}*dyy(u)")
    if dim >= 3:
        spatial_terms.append(f"{k}*dzz(u)")

    if spatial_terms:
        return f"{_minus_chain(lhs, spatial_terms)}"
    else:
        return f"{lhs}"


def wave_template(dim: int, coeffs: Dict) -> str:
    """Wave equation: ∂²u/∂t² = c²∇²u"""
    c_sq = coeffs["c_sq"]

    lhs = "dtt(u)"
    spatial_terms = []

    if dim >= 1:
        spatial_terms.append(f"{c_sq}*dxx(u)")
    if dim >= 2:
        spatial_terms.append(f"{c_sq}*dyy(u)")
    if dim >= 3:
        spatial_terms.append(f"{c_sq}*dzz(u)")

    if spatial_terms:
        return f"{_minus_chain(lhs, spatial_terms)}"
    else:
        return f"{lhs}"


def poisson_template(dim: int, coeffs: Dict) -> str:
    """Poisson equation: ∇²u = f"""
    f = coeffs["f"]
    terms = []

    if dim >= 1:
        terms.append("dxx(u)")
    if dim >= 2:
        terms.append("dyy(u)")
    if dim >= 3:
        terms.append("dzz(u)")

    result = " + ".join(terms)
    if f >= 0:
        return f"{result} - {f}"
    else:
        return f"{result} + {abs(f)}"


def advection_template(dim: int, coeffs: Dict) -> str:
    """Advection equation: ∂u/∂t + v·∇u = 0"""
    expr = "dt(u)"

    if dim >= 1 and "v_x" in coeffs:
        v_x = float(coeffs["v_x"])
        expr += f" + {v_x}*dx(u)" if v_x >= 0 else f" - {abs(v_x)}*dx(u)"

    if dim >= 2 and "v_y" in coeffs:
        v_y = float(coeffs["v_y"])
        expr += f" + {v_y}*dy(u)" if v_y >= 0 else f" - {abs(v_y)}*dy(u)"

    if dim >= 3 and "v_z" in coeffs:
        v_z = float(coeffs["v_z"])
        expr += f" + {v_z}*dz(u)" if v_z >= 0 else f" - {abs(v_z)}*dz(u)"

    return expr


def burgers_template(dim: int, coeffs: Dict) -> str:
    """Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²"""
    nu = coeffs["nu"]
    return f"dt(u) + u*dx(u) - {nu}*dxx(u)"


def kdv_template(dim: int, coeffs: Dict) -> str:
    """Korteweg-de Vries equation: ∂u/∂t + u∂u/∂x + δ∂³u/∂x³ = 0"""
    if dim != 1:
        raise ValueError("kdv is 1D-only in this benchmark (dim must be 1)")
    delta = coeffs["delta"]
    return f"dt(u) + u*dx(u) + {delta}*dxxx(u)"


def cubic_reaction_diffusion_template(dim: int, coeffs: Dict) -> str:
    """Cubic reaction-diffusion: u_t - Δu ± g*u^3 = 0

    NOTE: This is NOT the (complex) Schrödinger equation (no i, no complex field).
    We keep it as a separate, clearly named family to avoid confusion/overlap.
    """
    g = coeffs["g"]
    terms = ["dt(u)", "dxx(u)"]

    if dim >= 2:
        terms.append("dyy(u)")
    if dim >= 3:
        terms.append("dzz(u)")

    # Want: dt(u) - dxx(u) - dyy(u) - dzz(u) ...
    lap_part = _minus_chain("dt(u)", terms[1:])

    if g >= 0:
        return f"{lap_part} + {g}*u**3"
    else:
        return f"{lap_part} - {abs(g)}*u**3"


def reaction_diffusion_cubic_template(dim: int, coeffs: Dict) -> str:
    """Alias for backward compatibility (truthful name)."""
    return cubic_reaction_diffusion_template(dim, coeffs)


def allen_cahn_template(dim: int, coeffs: Dict) -> str:
    """Allen-Cahn equation: ∂u/∂t = ε²∇²u + u - u³"""
    eps_sq = coeffs["eps_sq"]
    terms = []

    if dim >= 1:
        terms.append(f"{eps_sq}*dxx(u)")
    if dim >= 2:
        terms.append(f"{eps_sq}*dyy(u)")
    if dim >= 3:
        terms.append(f"{eps_sq}*dzz(u)")

    lap_part = _minus_chain("dt(u)", terms)
    return f"{lap_part} - u + u**3"


def cahn_hilliard_template(dim: int, coeffs: Dict) -> str:
    """Cahn–Hilliard residual operator (benchmark form).

    PDE: u_t = -γ Δ^2 u + Δ(u^3 - u)
    Residual operator:
      dt(u) + γ Δ^2 u + Δu - Δ(u**3)
    """
    gamma = coeffs["gamma"]

    if dim == 1:
        return f"dt(u) + {gamma}*dxxxx(u) + dxx(u) - dxx(u**3)"
    if dim == 2:
        # Δ^2 u in 2D: u_xxxx + 2 u_xxyy + u_yyyy
        # IMPORTANT: keep operator-only string but avoid parentheses, so grammar tokenization stays simple.
        return (
            f"dt(u) + {gamma}*dxxxx(u) + {gamma}*dyyyy(u) + 2*{gamma}*dxxyy(u)"
            f" + dxx(u) + dyy(u) - dxx(u**3) - dyy(u**3)"
        )
    raise ValueError("cahn_hilliard currently supports dim=1 or dim=2 only")


def fisher_kpp_template(dim: int, coeffs: Dict) -> str:
    """Fisher-KPP equation: ∂u/∂t = D∇²u + ru(1-u)"""
    D = coeffs["D"]
    r = coeffs["r"]
    terms = []

    if dim >= 1:
        terms.append(f"{D}*dxx(u)")
    if dim >= 2:
        terms.append(f"{D}*dyy(u)")
    if dim >= 3:
        terms.append(f"{D}*dzz(u)")

    lap_part = _minus_chain("dt(u)", terms)
    return f"{lap_part} - {r}*u + {r}*u**2"


def kuramoto_sivashinsky_template(dim: int, coeffs: Dict) -> str:
    """Kuramoto-Sivashinsky equation: ∂u/∂t + ν∂²u/∂x² + γ∂⁴u/∂x⁴ + α(∂u/∂x)² = 0"""
    if dim != 1:
        raise ValueError(
            "kuramoto_sivashinsky is 1D-only in this benchmark (dim must be 1)"
        )
    nu = coeffs["nu"]
    gamma = coeffs["gamma"]
    alpha = coeffs["alpha"]
    # Canonical KS uses u*u_x (not (u_x)^2) as the nonlinearity.
    return f"dt(u) + {nu}*dxx(u) + {gamma}*dxxxx(u) + {alpha}*u*dx(u)"


def airy_template(dim: int, coeffs: Dict) -> str:
    """Airy equation (linear dispersive, 1D): u_t + α u_xxx = 0"""
    if dim != 1:
        raise ValueError("airy is 1D-only in this benchmark (dim must be 1)")
    alpha = coeffs["alpha"]
    return f"dt(u) + {alpha}*dxxx(u)"


def telegraph_template(dim: int, coeffs: Dict) -> str:
    """Telegraph equation: ∂²u/∂t² + a∂u/∂t - b²∂²u/∂x² = 0"""
    a = coeffs["a"]
    b_sq = coeffs["b_sq"]
    # Multi-D telegraph uses the Laplacian, consistent with the other families.
    terms = [f"{b_sq}*dxx(u)"]
    if dim >= 2:
        terms.append(f"{b_sq}*dyy(u)")
    if dim >= 3:
        terms.append(f"{b_sq}*dzz(u)")
    lap_part = _minus_chain(f"dtt(u) + {a}*dt(u)", terms)
    return f"{lap_part}"


def beam_plate_template(dim: int, coeffs: Dict) -> str:
    """Beam/Plate equation (hyperbolic 4th order).

    1D beam:  u_tt + κ u_xxxx = 0
    2D plate: u_tt + κ (u_xxxx + 2 u_xxyy + u_yyyy) = 0
    """
    kappa = coeffs["kappa"]
    if dim == 1:
        return f"dtt(u) + {kappa}*dxxxx(u)"
    if dim == 2:
        # Avoid parentheses for grammar tokenization compatibility.
        return f"dtt(u) + {kappa}*dxxxx(u) + 2*{kappa}*dxxyy(u) + {kappa}*dyyyy(u)"
    raise ValueError("beam_plate currently supports dim=1 or dim=2 only")


def biharmonic_template(dim: int, coeffs: Dict) -> str:
    """Biharmonic equation: ∇⁴u = f"""
    f = coeffs["f"]

    if dim == 1:
        if f >= 0:
            return f"dxxxx(u) - {f}"
        else:
            return f"dxxxx(u) + {abs(f)}"
    elif dim == 2:
        if f >= 0:
            # Δ^2 u in 2D: u_xxxx + 2 u_xxyy + u_yyyy
            return f"dxxxx(u) + dyyyy(u) + 2*dxxyy(u) - {f}"
        else:
            return f"dxxxx(u) + dyyyy(u) + 2*dxxyy(u) + {abs(f)}"
    else:
        if f >= 0:
            return f"dxxxx(u) - {f}"
        else:
            return f"dxxxx(u) + {abs(f)}"


def sine_gordon_template(dim: int, coeffs: Dict) -> str:
    """Sine-Gordon equation (true): u_tt - c^2 Δu + beta*sin(u) = 0"""
    c_sq = coeffs["c_sq"]
    beta = coeffs["beta"]

    spatial_terms = [f"{c_sq}*dxx(u)"]
    if dim >= 2:
        spatial_terms.append(f"{c_sq}*dyy(u)")
    if dim >= 3:
        spatial_terms.append(f"{c_sq}*dzz(u)")

    lap_part = _minus_chain("dtt(u)", spatial_terms)
    return f"{lap_part} + {beta}*sin(u)"


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
