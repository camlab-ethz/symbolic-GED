"""
Comprehensive Verification Script for Dataset Creation and Tokenization

This script verifies:
1. Mathematical correctness of PDE templates
2. Tokenization correctness (Lample & Charton style)
3. Roundtrip encoding/decoding accuracy
4. Dataset consistency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import re
from typing import List, Tuple, Dict
from pde.families import PDE_FAMILIES
from pde.chr_tokenizer import PDETokenizer
from pde.grammar import parse_to_productions, decode_production_sequence, PROD_COUNT
from analysis.pde_classifier import PDEClassifier


def verify_mathematical_correctness():
    """Verify that PDE templates match their mathematical definitions"""
    print("=" * 80)
    print("1. MATHEMATICAL CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    issues = []
    
    # Known correct forms
    expected_forms = {
        'heat': r'dt\(u\).*-.*dxx\(u\)',  # dt(u) - k*dxx(u) = 0
        'wave': r'dtt\(u\).*-.*dxx\(u\)',  # dtt(u) - c²*dxx(u) = 0
        'telegraph': r'dtt\(u\).*\+.*dt\(u\).*-.*dxx\(u\)',  # dtt(u) + a*dt(u) - b²*dxx(u) = 0
        'burgers': r'dt\(u\).*\+.*u.*\*.*dx\(u\).*-.*dxx\(u\)',  # dt(u) + u*dx(u) - nu*dxx(u) = 0
        'sine_gordon': r'dtt\(u\).*-.*dxx\(u\).*\+.*sin\(u\)',  # dtt(u) - c²*dxx(u) + beta*sin(u) = 0
        'airy': r'dt\(u\).*\+.*dxxx\(u\)',  # dt(u) + alpha*dxxx(u) = 0
        'beam_plate': r'dtt\(u\).*\+.*dxxxx\(u\)',  # dtt(u) + kappa*dxxxx(u) = 0
    }
    
    print("\nChecking PDE template mathematical forms...")
    for name, pattern in expected_forms.items():
        if name not in PDE_FAMILIES:
            issues.append(f"Missing family: {name}")
            continue
            
        family = PDE_FAMILIES[name]
        
        # Generate example
        if name == 'heat':
            pde = family.template_fn(1, {'k': 1.5})
        elif name == 'wave':
            pde = family.template_fn(1, {'c_sq': 2.0})
        elif name == 'telegraph':
            pde = family.template_fn(1, {'a': 1.0, 'b_sq': 4.0})
        elif name == 'burgers':
            pde = family.template_fn(1, {'nu': 0.1})
        elif name == 'sine_gordon':
            pde = family.template_fn(1, {'c_sq': 1.0, 'beta': 0.1})
        elif name == 'airy':
            pde = family.template_fn(1, {'alpha': 0.5})
        elif name == 'beam_plate':
            pde = family.template_fn(1, {'kappa': 0.1})
        else:
            continue
        
        # Remove " = 0"
        expr = pde.replace(' = 0', '').replace(' ', '')
        
        if not re.search(pattern, expr):
            issues.append(f"{name}: Pattern mismatch\n  Expected: {pattern}\n  Got: {expr}")
            print(f"  ❌ {name}: Pattern mismatch")
        else:
            print(f"  ✅ {name}: Matches expected form")
    
    if issues:
        print("\n⚠️  Mathematical Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All PDE templates are mathematically correct!")
    
    return len(issues) == 0


def verify_lample_charton_tokenization():
    """
    Verify that tokenization follows Lample & Charton's approach.
    
    Lample & Charton (2019) Deep Learning for Symbolic Mathematics:
    - Numbers are tokenized character-by-character: "1.5" → ['INT+', '1', '.', '5']
    - Expressions are converted to prefix (Polish) notation
    - Operators use names: 'add', 'sub', 'mul', 'div', 'pow'
    """
    print("\n" + "=" * 80)
    print("2. LAMPLE & CHARTON TOKENIZATION VERIFICATION")
    print("=" * 80)
    
    tokenizer = PDETokenizer()
    issues = []
    
    # Test 1: Number tokenization
    print("\nTest 1: Number Tokenization (Character-by-character)")
    print("-" * 80)
    test_numbers = [
        ("1.5", ["INT+", "1", ".", "5"]),
        ("-0.1", ["INT-", "0", ".", "1"]),
        ("2.345", ["INT+", "2", ".", "3", "4", "5"]),
        ("3", ["INT+", "3"]),
    ]
    
    all_pass = True
    for num_str, expected_tokens in test_numbers:
        actual = tokenizer._tokenize_number(num_str)
        if actual == expected_tokens:
            print(f"  ✅ {num_str:8s} → {actual}")
        else:
            print(f"  ❌ {num_str:8s} → {actual} (expected {expected_tokens})")
            issues.append(f"Number tokenization: {num_str}")
            all_pass = False
    
    # Test 2: Prefix notation conversion
    print("\nTest 2: Prefix Notation Conversion")
    print("-" * 80)
    test_expressions = [
        ("dt(u) - 1.5*dxx(u)", ["sub", "dt", "u", "mul", "INT+", "1", ".", "5", "dxx", "u"]),
        ("dtt(u) - 2.0*dxx(u)", ["sub", "dtt", "u", "mul", "INT+", "2", ".", "0", "dxx", "u"]),
    ]
    
    for infix, expected_prefix in test_expressions:
        actual_prefix = tokenizer.infix_to_prefix_simple(infix)
        # Normalize: remove spaces, compare token sequences
        actual_normalized = ' '.join(actual_prefix)
        expected_normalized = ' '.join(expected_prefix)
        
        if actual_normalized == expected_normalized:
            print(f"  ✅ {infix}")
            print(f"     → {actual_normalized}")
        else:
            print(f"  ⚠️  {infix}")
            print(f"     Got:      {actual_normalized}")
            print(f"     Expected: {expected_normalized}")
            # Don't fail if just spacing differs, but log it
            if actual_prefix != expected_prefix:
                issues.append(f"Prefix conversion mismatch: {infix}")
    
    # Test 3: Roundtrip encoding/decoding
    print("\nTest 3: Roundtrip Encoding/Decoding")
    print("-" * 80)
    test_pdes = [
        "dt(u) - 1.5*dxx(u)",
        "dtt(u) - 2.0*dxx(u)",
        "dt(u) + u*dx(u) - 0.1*dxx(u)",
        "dt(u) - dxx(u) + u^3 - u",
    ]
    
    roundtrip_pass = True
    for pde in test_pdes:
        ids = tokenizer.encode(pde, add_special_tokens=False)
        decoded = tokenizer.decode_to_infix(ids, skip_special_tokens=True)
        # Normalize: remove all spaces for comparison
        original_norm = pde.replace(' ', '')
        decoded_norm = decoded.replace(' ', '')
        
        if original_norm == decoded_norm:
            print(f"  ✅ {pde}")
        else:
            print(f"  ❌ {pde}")
            print(f"     Decoded: {decoded}")
            issues.append(f"Roundtrip failed: {pde}")
            roundtrip_pass = False
    
    # Test 4: Vocabulary size
    print("\nTest 4: Vocabulary Consistency")
    print("-" * 80)
    vocab_size = len(tokenizer.vocab)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Expected ~82 tokens (operators + derivatives + variables + numbers)")
    
    if 75 <= vocab_size <= 90:
        print(f"  ✅ Vocabulary size is reasonable")
    else:
        print(f"  ⚠️  Vocabulary size seems unusual")
        issues.append(f"Vocabulary size: {vocab_size}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} tokenization issues")
    else:
        print(f"\n✅ Tokenization follows Lample & Charton approach correctly!")
    
    return len(issues) == 0


def verify_dataset_consistency(dataset_path: str = 'pde_dataset_48444_fixed.csv'):
    """Verify dataset file consistency"""
    print("\n" + "=" * 80)
    print("3. DATASET CONSISTENCY VERIFICATION")
    print("=" * 80)
    
    if not os.path.exists(dataset_path):
        print(f"  ⚠️  Dataset file not found: {dataset_path}")
        return False
    
    issues = []
    classifier = PDEClassifier()
    
    print(f"\nReading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  Total PDEs: {len(rows)}")
    
    # Check 1: No " = 0" suffix
    print("\nCheck 1: PDE strings should NOT have ' = 0' suffix")
    has_zero_suffix = 0
    for i, row in enumerate(rows[:1000]):  # Check first 1000
        pde = row.get('pde', '')
        if ' = 0' in pde:
            has_zero_suffix += 1
            if has_zero_suffix <= 5:  # Show first 5 examples
                issues.append(f"Row {i}: Has ' = 0' suffix: {pde}")
    
    if has_zero_suffix == 0:
        print(f"  ✅ No ' = 0' suffixes found (checked {min(1000, len(rows))} rows)")
    else:
        print(f"  ❌ Found {has_zero_suffix} PDEs with ' = 0' suffix")
    
    # Check 2: Classification consistency
    print("\nCheck 2: Classification consistency (sample)")
    misclassified = 0
    for i, row in enumerate(rows[:100]):  # Check first 100
        pde = row.get('pde', '')
        expected_family = row.get('family', '')
        expected_nonlinear = row.get('nonlinear', 'False').lower() == 'true'
        
        result = classifier.classify(pde)
        # result is a PDELabels object, access attributes directly
        if result.family != expected_family:
            misclassified += 1
            if misclassified <= 3:
                issues.append(f"Row {i}: Family mismatch - Expected: {expected_family}, Got: {result.family}")
        
        expected_linearity = 'nonlinear' if expected_nonlinear else 'linear'
        if result.linearity != expected_linearity:
            misclassified += 1
            if misclassified <= 3:
                issues.append(f"Row {i}: Nonlinearity mismatch - Expected: {expected_linearity}, Got: {result.linearity}")
    
    if misclassified == 0:
        print(f"  ✅ All classifications match (checked 100 rows)")
    else:
        print(f"  ⚠️  Found {misclassified} classification mismatches (checked 100 rows)")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} dataset issues")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
    else:
        print(f"\n✅ Dataset is consistent!")
    
    return len(issues) == 0


def verify_grammar_tokenization():
    """Verify Grammar-based tokenization"""
    print("\n" + "=" * 80)
    print("4. GRAMMAR TOKENIZATION VERIFICATION")
    print("=" * 80)
    
    issues = []
    
    print(f"\nGrammar Info:")
    print(f"  Production count (P): {PROD_COUNT}")
    
    # Test roundtrip
    test_pdes = [
        "dt(u) - 1.5*dxx(u)",
        "dtt(u) - 2.0*dxx(u)",
    ]
    
    print("\nTest: Grammar Encoding/Decoding Roundtrip")
    print("-" * 80)
    for pde in test_pdes:
        try:
            prod_ids = parse_to_productions(pde)
            decoded = decode_production_sequence(prod_ids)
            
            # Normalize for comparison
            original_norm = pde.replace(' ', '')
            decoded_norm = decoded.replace(' ', '')
            
            if original_norm == decoded_norm:
                print(f"  ✅ {pde}")
            else:
                print(f"  ❌ {pde}")
                print(f"     Decoded: {decoded}")
                issues.append(f"Grammar roundtrip failed: {pde}")
        except Exception as e:
            print(f"  ❌ {pde}: Error - {e}")
            issues.append(f"Grammar error: {pde} - {e}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} grammar tokenization issues")
    else:
        print(f"\n✅ Grammar tokenization works correctly!")
    
    return len(issues) == 0


def main():
    """Run all verification checks"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATASET & TOKENIZATION VERIFICATION")
    print("=" * 80)
    
    results = {}
    
    # Run all checks
    results['mathematical'] = verify_mathematical_correctness()
    results['tokenization'] = verify_lample_charton_tokenization()
    results['dataset'] = verify_dataset_consistency()
    results['grammar'] = verify_grammar_tokenization()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_pass = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check.capitalize():20s}: {status}")
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ALL CHECKS PASSED - Dataset and tokenization are correct!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review issues above")
    print("=" * 80)
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
