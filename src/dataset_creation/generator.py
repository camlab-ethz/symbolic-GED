"""
Modular PDE Dataset Generator
Generates human-readable PDE strings from known families
"""

import random
import numpy as np
from typing import List, Dict, Optional
from pde.families import PDE_FAMILIES, get_family
from pde.normalize import normalize_pde_string


class PDEGenerator:
    """Generate PDEs from known families with configurable parameters"""

    def __init__(self, seed: Optional[int] = None, omit_eq0: bool = False):
        """
        Initialize the PDE generator

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.omit_eq0 = bool(omit_eq0)

    @staticmethod
    def _strip_eq0(pde: str) -> str:
        """Strip a trailing '= 0' (any whitespace) if present."""
        # Backward-compatible wrapper; keep behavior centralized.
        return normalize_pde_string(pde)

    def generate_coefficients(self, family_name: str, dim: int) -> Dict[str, float]:
        """
        Generate random coefficients for a PDE family

        Rules:
        - Coefficients in range [-5, 5] with 3 decimal places
        - Same coefficient value across dimensions (e.g., same diffusion k for x, y, z)

        Args:
            family_name: Name of the PDE family
            dim: Spatial dimension

        Returns:
            Dictionary of coefficient names to values
        """
        coeffs = {}

        # Generate base coefficients based on family
        if family_name == "heat":
            coeffs["k"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # Diffusion coefficient (positive)

        elif family_name == "wave":
            coeffs["c_sq"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # Wave speed squared (positive)

        elif family_name == "poisson":
            coeffs["f"] = round(np.random.uniform(-5.0, 5.0), 3)  # Source term

        elif family_name == "advection":
            # Velocity components (can be negative for direction)
            for ax in ["x", "y", "z"][:dim]:
                coeffs[f"v_{ax}"] = round(np.random.uniform(-5.0, 5.0), 3)

        elif family_name == "burgers":
            coeffs["nu"] = round(np.random.uniform(0.1, 5.0), 3)  # Viscosity (positive)

        elif family_name == "kdv":
            coeffs["delta"] = round(np.random.uniform(0.1, 5.0), 3)  # Dispersion

        elif family_name == "cubic_reaction_diffusion":
            coeffs["g"] = round(
                np.random.uniform(-5.0, 5.0), 3
            )  # Cubic reaction strength
        elif family_name == "reaction_diffusion_cubic":
            coeffs["g"] = round(np.random.uniform(-5.0, 5.0), 3)

        elif family_name == "allen_cahn":
            coeffs["eps_sq"] = round(
                np.random.uniform(0.01, 2.0), 3
            )  # Interface width squared

        elif family_name == "cahn_hilliard":
            coeffs["gamma"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # 4th order diffusion

        elif family_name == "fisher_kpp":
            coeffs["D"] = round(np.random.uniform(0.1, 5.0), 3)  # Diffusion
            coeffs["r"] = round(np.random.uniform(0.1, 2.0), 3)  # Growth rate

        elif family_name == "kuramoto_sivashinsky":
            coeffs["nu"] = round(np.random.uniform(0.1, 3.0), 3)  # 2nd order
            coeffs["gamma"] = round(np.random.uniform(0.01, 2.0), 3)  # 4th order
            coeffs["alpha"] = round(np.random.uniform(0.1, 2.0), 3)  # Nonlinear term

        # Navier–Stokes removed

        # klein_gordon excluded (not part of the 16-family dataset)

        elif family_name == "telegraph":
            coeffs["a"] = round(np.random.uniform(0.1, 3.0), 3)  # Damping
            coeffs["b_sq"] = round(np.random.uniform(0.1, 5.0), 3)  # Wave speed squared

        elif family_name == "biharmonic":
            coeffs["f"] = round(np.random.uniform(-5.0, 5.0), 3)  # Source term

        elif family_name == "sine_gordon":
            coeffs["c_sq"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # Wave speed squared (positive)
            coeffs["beta"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # sin(u) strength (positive)
        elif family_name == "airy":
            coeffs["alpha"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # positive dispersion
        elif family_name == "beam_plate":
            coeffs["kappa"] = round(
                np.random.uniform(0.1, 5.0), 3
            )  # positive stiffness

        return coeffs

    def generate_pde(self, family_name: str, dim: int = 2) -> Optional[Dict]:
        """
        Generate a single PDE from a family

        Args:
            family_name: Name of the PDE family
            dim: Spatial dimension (1, 2, or 3)

        Returns:
            Dictionary with 'pde', 'family', 'dim', 'coefficients'
            or None if family not found or invalid dimension
        """
        family = get_family(family_name)
        if family is None:
            print(f"Unknown family: {family_name}")
            return None

        if dim < family.min_dimension:
            print(f"Family {family_name} requires dimension >= {family.min_dimension}")
            return None

        # Special case: KdV is 1D only
        if family_name == "kdv" and dim > 1:
            dim = 1

        # Special case: Kuramoto-Sivashinsky is typically 1D
        if family_name == "kuramoto_sivashinsky" and dim > 1:
            dim = 1

        # Families that are implemented as 1D templates in `pde/families.py`
        # (until we add true multi-D versions).
        if family_name in ["burgers"] and dim > 1:
            dim = 1

        # Airy is 1D only
        if family_name == "airy" and dim > 1:
            dim = 1

        # Beam/plate: implemented up to 2D in template
        if family_name == "beam_plate" and dim > 2:
            dim = 2

        # Families with 2D templates but no 3D template (cap at 2D to keep labels consistent).
        if family_name in ["biharmonic", "cahn_hilliard"] and dim > 2:
            dim = 2

        # Generate coefficients
        coeffs = self.generate_coefficients(family_name, dim)

        # Generate PDE string using family template
        pde_string = family.template_fn(dim, coeffs)
        if self.omit_eq0:
            pde_string = self._strip_eq0(pde_string)

        # Recommended correctness: derive nonlinear flag from the actual string.
        import re

        # Derive nonlinear flag from the actual string (simple + robust).
        s = str(pde_string)
        nonlinear = (
            "sin(" in s
            or "u^" in s
            or "u**" in s
            or "u*u" in s
            or "u*dx(u)" in s
            or "u*dy(u)" in s
            or "u*dz(u)" in s
        )

        return {
            "pde": pde_string,
            "family": family_name,
            "dim": dim,
            "temporal_order": family.temporal_order,
            "spatial_order": family.spatial_order,
            "nonlinear": nonlinear,
            "coefficients": coeffs,
        }

    def generate_dataset(
        self,
        families: List[str],
        n_per_family: int = 100,
        dimensions: List[int] = [1, 2, 3],
        balance_dims: bool = True,
        ensure_unique: bool = True,
        max_attempts: int = 1000,
    ) -> List[Dict]:
        """
        Generate a dataset of PDEs

        Args:
            families: List of family names to include
            n_per_family: Number of PDEs per family
            dimensions: List of dimensions to use
            balance_dims: If True, balance PDEs across dimensions
            ensure_unique: If True, ensure all PDEs are unique
            max_attempts: Maximum attempts to generate unique PDE

        Returns:
            List of PDE dictionaries
        """
        dataset = []
        seen_pdes = set() if ensure_unique else None

        for family_name in families:
            family = get_family(family_name)
            if family is None:
                print(f"Skipping unknown family: {family_name}")
                continue

            # Filter valid dimensions for this family
            valid_dims = [d for d in dimensions if d >= family.min_dimension]

            if not valid_dims:
                print(f"No valid dimensions for {family_name}, skipping")
                continue

            # Generate PDEs
            generated_count = 0
            attempts = 0

            while (
                generated_count < n_per_family
                and attempts < max_attempts * n_per_family
            ):
                attempts += 1

                if balance_dims:
                    # Cycle through dimensions
                    dim = valid_dims[generated_count % len(valid_dims)]
                else:
                    # Random dimension
                    dim = random.choice(valid_dims)

                pde_data = self.generate_pde(family_name, dim)
                if pde_data:
                    # Check uniqueness
                    if ensure_unique:
                        pde_string = pde_data["pde"]
                        if pde_string in seen_pdes:
                            continue  # Skip duplicate
                        seen_pdes.add(pde_string)

                    dataset.append(pde_data)
                    generated_count += 1

            if generated_count < n_per_family:
                print(
                    f"Warning: Only generated {generated_count}/{n_per_family} unique PDEs for {family_name}"
                )

        return dataset

    def save_dataset(self, dataset: List[Dict], filename: str, format: str = "csv"):
        """
        Save dataset to file

        Args:
            dataset: List of PDE dictionaries
            filename: Output filename
            format: 'csv' or 'txt'
        """
        if format == "csv":
            import csv

            with open(filename, "w", newline="") as f:
                if not dataset:
                    return

                # Determine all available fields from first entry
                base_fieldnames = [
                    "pde",
                    "family",
                    "dim",
                    "temporal_order",
                    "spatial_order",
                    "nonlinear",
                ]

                # Add extra fields if present (e.g., tau, split for telegrapher bridge)
                extra_fields = set()
                for entry in dataset:
                    extra_fields.update(
                        k
                        for k in entry.keys()
                        if k not in base_fieldnames and k != "coefficients"
                    )

                fieldnames = base_fieldnames + sorted(extra_fields)

                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()

                # Write data
                for entry in dataset:
                    writer.writerow(entry)

        elif format == "txt":
            with open(filename, "w") as f:
                for entry in dataset:
                    f.write(f"{entry['pde']}\n")

        print(f"Saved {len(dataset)} PDEs to {filename}")

    def print_summary(self, dataset: List[Dict]):
        """Print summary statistics of the dataset"""
        from collections import Counter

        print(f"\nDataset Summary")
        print("=" * 80)
        print(f"Total PDEs: {len(dataset)}")

        # By family
        families = Counter(entry["family"] for entry in dataset)
        print(f"\nBy Family:")
        for family, count in sorted(families.items()):
            print(f"  {family:20s}: {count:4d}")

        # By dimension
        dims = Counter(entry["dim"] for entry in dataset)
        print(f"\nBy Dimension:")
        for dim, count in sorted(dims.items()):
            print(f"  {dim}D: {count:4d}")

        # By temporal order
        temporal = Counter(entry["temporal_order"] for entry in dataset)
        print(f"\nTemporal Order:")
        for order, count in sorted(temporal.items()):
            order_name = {0: "none", 1: "1st", 2: "2nd"}.get(order, str(order))
            print(f"  {order_name:5s}: {count:4d}")

        # Nonlinear
        nonlinear_count = sum(1 for e in dataset if e["nonlinear"])
        print(f"\nNonlinear: {nonlinear_count}/{len(dataset)}")

    def generate_telegrapher_bridge(
        self,
        tau_small: Optional[List[float]] = None,
        tau_mid: Optional[List[float]] = None,
        tau_large: Optional[List[float]] = None,
        c_sq: float = 1.0,
        dim: int = 1,
    ) -> List[Dict]:
        """
        Generate a telegrapher dataset for diffusion↔wave continuation.

        We parameterize the telegraph equation as:
            dtt(u) + a * dt(u) - c_sq * dxx(u) = 0
        and interpret a ≈ 1 / tau.

        Splits:
            - 'train_endpoints': small & large tau (seen during training)
            - 'test_middle': middle tau (unseen continuation region)

        Args:
            tau_small: List of small tau values (diffusion-like, e.g., [0.02, 0.05, 0.1])
            tau_mid: List of middle tau values (unseen continuation, e.g., [0.2, 0.5, 1.0, 2.0])
            tau_large: List of large tau values (wave-like, e.g., [5.0, 10.0])
            c_sq: Wave speed squared (default: 1.0)
            dim: Spatial dimension (default: 1)

        Returns:
            List of PDE dictionaries with additional 'tau' and 'split' fields
        """
        from copy import deepcopy

        if tau_small is None:
            tau_small = [0.02, 0.05, 0.1]  # diffusion-like
        if tau_mid is None:
            tau_mid = [0.2, 0.5, 1.0, 2.0]  # unseen continuation band
        if tau_large is None:
            tau_large = [5.0, 10.0]  # wave-like

        all_entries: List[Dict] = []

        family = get_family("telegraph")
        if family is None:
            raise ValueError("Telegraph family not found in PDE_FAMILIES.")

        def make_entry(tau: float, split: str) -> Dict:
            a = round(1.0 / tau, 3)  # damping parameter
            coeffs = {"a": a, "b_sq": c_sq}
            pde_string = family.template_fn(dim, coeffs)
            if self.omit_eq0:
                pde_string = self._strip_eq0(pde_string)
            return {
                "pde": pde_string,
                "family": "telegraph",
                "dim": dim,
                "temporal_order": family.temporal_order,
                "spatial_order": family.spatial_order,
                "nonlinear": family.supports_nonlinear,
                "coefficients": deepcopy(coeffs),
                "tau": tau,
                "split": split,
            }

        # endpoints → training
        for tau in tau_small + tau_large:
            all_entries.append(make_entry(tau, split="train_endpoints"))

        # middle → test
        for tau in tau_mid:
            all_entries.append(make_entry(tau, split="test_middle"))

        return all_entries


def main():
    """Generate PDE dataset with command-line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDE dataset")
    parser.add_argument(
        "--output", type=str, default="pde_dataset.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--num_per_family",
        type=int,
        default=3000,
        help="Number of PDEs to generate per family",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--omit-eq0",
        action="store_true",
        help="If set, remove a trailing '= 0' from PDE strings in the output CSV (recommended).",
    )

    args = parser.parse_args()

    # Create generator
    gen = PDEGenerator(seed=args.seed, omit_eq0=args.omit_eq0)

    # Generate dataset with all 16 families
    families = [
        "heat",
        "wave",
        "poisson",
        "advection",
        "burgers",
        "kdv",
        "reaction_diffusion_cubic",
        "allen_cahn",
        "cahn_hilliard",
        "fisher_kpp",
        "kuramoto_sivashinsky",
        "telegraph",
        "biharmonic",
        "sine_gordon",
        "airy",
        "beam_plate",
    ]

    print("Generating PDE dataset...")
    dataset = gen.generate_dataset(
        families=families,
        n_per_family=args.num_per_family,
        dimensions=[1, 2, 3],
        balance_dims=True,
        ensure_unique=True,  # Ensure all PDEs are unique
    )

    # Print summary
    gen.print_summary(dataset)

    # Save to files
    gen.save_dataset(dataset, args.output, format="csv")
    if args.output.endswith(".csv"):
        txt_output = args.output.replace(".csv", ".txt")
        gen.save_dataset(dataset, txt_output, format="txt")

    # Show some examples
    print("\nExample PDEs:")
    print("-" * 80)
    for i in range(min(10, len(dataset))):
        entry = dataset[i]
        print(f"{entry['family']:20s} ({entry['dim']}D): {entry['pde']}")

    # Generate telegrapher bridge dataset for diffusion↔wave continuation
    print("\n" + "=" * 80)
    print("Generating Telegrapher Bridge Dataset (Diffusion ↔ Wave Continuation)")
    print("=" * 80)
    bridge = gen.generate_telegrapher_bridge()
    gen.save_dataset(bridge, "telegrapher_bridge.csv", format="csv")

    print("\nTelegrapher Bridge Samples:")
    print("-" * 80)
    print(f"{'Split':<20s} {'Tau':<10s} {'PDE'}")
    print("-" * 80)
    for e in bridge:
        print(f"{e['split']:<20s} {e['tau']:<10.3f} {e['pde']}")

    # Summary
    train_count = sum(1 for e in bridge if e["split"] == "train_endpoints")
    test_count = sum(1 for e in bridge if e["split"] == "test_middle")
    print(f"\nBridge Dataset Summary:")
    print(f"  Train (endpoints): {train_count}")
    print(f"  Test (middle):     {test_count}")
    print(f"  Total:             {len(bridge)}")


if __name__ == "__main__":
    main()
