import re

import pytest

from pde.families import PDE_FAMILIES
from pde_string_utils import canonicalize, infer_orders_and_dim, signature


ALLOWED_PATTERN = re.compile(
    r"^[0-9eE\.\+\-\*/\(\)\s]*"
    r"(?:"
    r"dt|dtt|dx|dxx|dxxx|dxxxx|dy|dyy|dyyy|dyyyy|dz|dzz|dzzz|dzzzz|dxxyy|"
    r"sin|cos|exp|log|tanh|u"
    r"|[0-9eE\.\+\-\*/\(\)\s]"
    r")*$"
)


def _assert_operator_contract(s: str) -> None:
    assert "=" not in s
    assert ALLOWED_PATTERN.match(s) is not None


def _coeffs_for_family(name: str):
    # minimal deterministic coeff dict per family for tests
    if name == "heat":
        return {"k": 1.0}
    if name == "wave":
        return {"c_sq": 1.0}
    if name == "poisson":
        return {"f": 1.0}
    if name == "advection":
        return {"v_x": 1.0, "v_y": 1.0, "v_z": 1.0}
    if name == "burgers":
        return {"nu": 0.1}
    if name == "kdv":
        return {"delta": 0.1}
    if name == "reaction_diffusion_cubic":
        return {"g": 1.0}
    if name == "allen_cahn":
        return {"eps_sq": 0.1}
    if name == "cahn_hilliard":
        return {"gamma": 0.2}
    if name == "fisher_kpp":
        return {"D": 1.0, "r": 0.5}
    if name == "kuramoto_sivashinsky":
        return {"nu": 1.0, "gamma": 0.1, "alpha": 0.5}
    if name == "telegraph":
        return {"a": 1.0, "b_sq": 2.0}
    if name == "biharmonic":
        return {"f": 1.0}
    if name == "sine_gordon":
        return {"c_sq": 1.0, "beta": 0.5}
    if name == "airy":
        return {"alpha": 0.2}
    if name == "beam_plate":
        return {"kappa": 0.3}
    raise KeyError(name)


def test_operator_contract():
    for name, fam in PDE_FAMILIES.items():
        for dim in (1, 2, 3):
            try:
                s = fam.template_fn(dim, _coeffs_for_family(name))
            except ValueError:
                continue
            _assert_operator_contract(s)


def test_metadata_matches_inference():
    for name, fam in PDE_FAMILIES.items():
        for dim in (1, 2, 3):
            if dim < fam.min_dimension:
                with pytest.raises(Exception):
                    fam.template_fn(dim, _coeffs_for_family(name))
                continue

            # enforce invalid dims for benchmark-restricted families
            if name in {"kdv", "airy", "kuramoto_sivashinsky"} and dim != 1:
                with pytest.raises(ValueError):
                    fam.template_fn(dim, _coeffs_for_family(name))
                continue

            if name in {"cahn_hilliard", "beam_plate"} and dim > 2:
                with pytest.raises(ValueError):
                    fam.template_fn(dim, _coeffs_for_family(name))
                continue

            s = fam.template_fn(dim, _coeffs_for_family(name))
            meta = infer_orders_and_dim(s)

            assert meta["temporal_order"] == fam.temporal_order
            assert meta["spatial_order"] == fam.spatial_order
            # inferred_min_dim is string-dependent; it must be within [family.min_dimension, dim]
            assert fam.min_dimension <= meta["inferred_min_dim"] <= dim


def test_signature_uniqueness():
    # Representative valid dimension per family (avoid dim-restricted exceptions)
    rep_dim = {
        "heat": 2,
        "wave": 2,
        "poisson": 2,
        "advection": 2,
        "burgers": 1,
        "kdv": 1,
        "reaction_diffusion_cubic": 2,
        "allen_cahn": 2,
        "cahn_hilliard": 2,
        "fisher_kpp": 2,
        "kuramoto_sivashinsky": 1,
        "telegraph": 2,
        "biharmonic": 2,
        "sine_gordon": 2,
        "airy": 1,
        "beam_plate": 2,
    }

    sigs = {}
    for name, fam in PDE_FAMILIES.items():
        dim = rep_dim[name]
        s = fam.template_fn(dim, _coeffs_for_family(name))
        sigs[name] = signature(s)

    # Ensure uniqueness across families (no aliases expected here)
    seen = {}
    for name, sig in sigs.items():
        if sig in seen:
            raise AssertionError(f"Signature collision: {name} collides with {seen[sig]} ({sig})")
        seen[sig] = name


def test_canonicalize_idempotent():
    samples = [
        "dt(u)  +  1.0*dxx(u)",
        "dt(u)+-1.0*dxx(u)",
        "dtt(u) -  1.0*dxx(u) + 0.5*sin(u)",
        "dt(u) + (dxx(u) + dyy(u)) - (dxx(u**3) + dyy(u**3))",
    ]
    for s in samples:
        c1 = canonicalize(s)
        c2 = canonicalize(c1)
        assert c1 == c2
        assert "  " not in c1
        assert "+ -" not in c1

