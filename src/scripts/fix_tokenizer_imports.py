#!/usr/bin/env python3
"""
Fix imports after renaming tokenizer.py to chr_tokenizer.py

Changes:
- from pde.tokenizer → from pde.chr_tokenizer
- from pde import chr_tokenizer → from pde import chr_tokenizer
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Files to check (all Python files)
PYTHON_FILES = list(BASE_DIR.rglob("*.py"))
# Exclude __pycache__
PYTHON_FILES = [f for f in PYTHON_FILES if "__pycache__" not in str(f)]

REPLACEMENTS = [
    (r"from pde\.tokenizer import", "from pde.chr_tokenizer import"),
    (r"from pde import chr_tokenizer", "from pde import chr_tokenizer"),
    (r"from \.tokenizer import", "from .chr_tokenizer import"),
    (r"from src\.pde\.tokenizer", "from src.pde.chr_tokenizer"),
]


def fix_file(file_path: Path):
    """Fix imports in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        for pattern, replacement in REPLACEMENTS:
            content = re.sub(pattern, replacement, content)

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
    for file_path in PYTHON_FILES:
        if fix_file(file_path):
            rel_path = file_path.relative_to(BASE_DIR)
            print(f"✓ Fixed: {rel_path}")
            fixed_count += 1

    print(f"\n✅ Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
