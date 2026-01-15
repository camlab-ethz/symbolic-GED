"""Unit tests for skeletonize_pde function."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.utils_pde import skeletonize_pde


def test_basic_integer():
    """Test basic integer replacement."""
    pde = "dt(u)=1*dxx(u)"
    expected = "dt(u)=C*dxx(u)"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_float():
    """Test float replacement."""
    pde = "dt(u)=0.37*dxx(u)+1.1*u"
    expected = "dt(u)=C*dxx(u)+C*u"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_negative_numbers():
    """Test negative number replacement."""
    pde = "-2.0*u^2+3*dx(u)"
    expected = "-C*u^2+C*dx(u)"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_scientific_notation_lowercase():
    """Test scientific notation with lowercase e."""
    pde = "dt(u)=3e-2*dx(u)"
    expected = "dt(u)=C*dx(u)"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_scientific_notation_uppercase():
    """Test scientific notation with uppercase E."""
    pde = "dt(u)=1.2E+5*dxx(u)"
    expected = "dt(u)=C*dxx(u)"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_scientific_notation_negative():
    """Test scientific notation with negative exponent."""
    pde = "-4.5e-3*u+2.1E-2*dxx(u)"
    expected = "-C*u+C*dxx(u)"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_multiple_numbers():
    """Test PDE with multiple numbers."""
    pde = "dt(u)=0.5*dxx(u)+1.0*u-0.25*u^2"
    expected = "dt(u)=C*dxx(u)+C*u-C*u^2"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_strip_spaces():
    """Test that spaces are stripped."""
    pde = "dt(u) = 0.37 * dxx(u) + 1.1 * u"
    expected = "dt(u)=C*dxx(u)+C*u"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


def test_complex_example():
    """Test complex example with mixed number types."""
    pde = "dt(u)=-2.0*u^2+3e-2*dx(u)-1.5E+1*u"
    expected = "dt(u)=-C*u^2+C*dx(u)-C*u"
    result = skeletonize_pde(pde)
    assert result == expected, f"Expected {expected}, got {result}"


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_basic_integer,
        test_float,
        test_negative_numbers,
        test_scientific_notation_lowercase,
        test_scientific_notation_uppercase,
        test_scientific_notation_negative,
        test_multiple_numbers,
        test_strip_spaces,
        test_complex_example,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
