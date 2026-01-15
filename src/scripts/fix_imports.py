#!/usr/bin/env python3
"""
Fix import statements after reorganization.

Changes:
- from pde_families → from pde.families
- from pde_grammar → from pde.grammar
- import pde_grammar → from pde import grammar
- from pde_tokenizer → from pde.tokenizer
- from generator → from dataset_creation.generator
"""

import re
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent

# Files to update
FILES_TO_FIX = [
    "scripts/verify_tokenization_correctness.py",
    "analyze_interpolations.py",
    "evaluate_all_metrics.py",
    "create_visualizations.py",
    "comprehensive_final_analysis.py",
    "test_validity.py",
    "stress_test_vaes.py",
    "comprehensive_vae_analysis.py",
    "visualize_all_maps.py",
    "evaluate_interpolation.py",
    "evaluate_generation_proper.py",
    "evaluate_generation.py",
    "quick_generation_eval.py",
    "vae/decode_test_set.py",
    "vae/module.py",
    "examples/generate_telegrapher_bridge.py",
]

# Patterns and replacements
REPLACEMENTS = [
    # Direct imports
    (r"from pde.families import", "from pde.families import"),
    (r"from pde.grammar import", "from pde.grammar import"),
    (r"from pde.chr_tokenizer import", "from pde.chr_tokenizer import"),
    (
        r"from dataset_creation.generator import",
        "from dataset_creation.generator import",
    ),
    # Module imports
    (r"^import pde_grammar$", "from pde import grammar as pde_grammar"),
    (r"^import pde_grammar as", "from pde import grammar as"),
    # In sys.path.insert contexts (keep relative imports working)
    # Handle: from src.pde import grammar as pde_grammar
    (
        r"from src.pde import grammar as pde_grammar",
        "from src.pde import grammar as pde_grammar",
    ),
    (r"from src.pde.grammar", "from src.pde.grammar"),
    (r"from src.pde.chr_tokenizer", "from src.pde.chr_tokenizer"),
    (r"from src.pde.families", "from src.pde.families"),
    (r"from src.dataset_creation.generator", "from src.dataset_creation.generator"),
]


def fix_file(file_path: Path):
    """Fix imports in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        for pattern, replacement in REPLACEMENTS:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        if content != original:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix all files."""
    fixed_count = 0
    for rel_path in FILES_TO_FIX:
        file_path = BASE_DIR / rel_path
        if file_path.exists():
            if fix_file(file_path):
                print(f"✓ Fixed: {rel_path}")
                fixed_count += 1
            else:
                print(f"  No changes: {rel_path}")
        else:
            print(f"✗ Not found: {rel_path}")

    print(f"\n✅ Fixed {fixed_count} files")

    # Also check scripts directory
    scripts_dir = BASE_DIR / "scripts"
    if scripts_dir.exists():
        for script_file in scripts_dir.glob("*.py"):
            if fix_file(script_file):
                print(f"✓ Fixed: scripts/{script_file.name}")
                fixed_count += 1

    # Check analysis directory
    analysis_dir = BASE_DIR / "analysis"
    if analysis_dir.exists():
        for analysis_file in analysis_dir.glob("*.py"):
            if fix_file(analysis_file):
                print(f"✓ Fixed: analysis/{analysis_file.name}")
                fixed_count += 1

    # Check vae directory
    vae_dir = BASE_DIR / "vae"
    if vae_dir.exists():
        for vae_file in vae_dir.glob("*.py"):
            if fix_file(vae_file):
                print(f"✓ Fixed: vae/{vae_file.name}")
                fixed_count += 1


if __name__ == "__main__":
    main()
