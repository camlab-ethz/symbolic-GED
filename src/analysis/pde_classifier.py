"""Rigorous PDE Classifier for Decoded Outputs.

This module provides a rule-based classifier that determines all physics
labels for a PDE string. Use this to classify decoded/generated PDEs
when you can't rely on ground truth labels.

Physics Labels Determined:
- family: heat, wave, poisson, burgers, kdv, fisher_kpp, etc. (16 families)
- pde_type: parabolic, hyperbolic, elliptic, dispersive
- linearity: linear, nonlinear
- temporal_order: 0, 1, 2
- spatial_order: 1, 2, 3, 4
- dimension: 1, 2, 3
- mechanisms: diffusion, advection, reaction, dispersion (list)

Usage:
    from analysis.pde_classifier import PDEClassifier
    
    classifier = PDEClassifier()
    labels = classifier.classify("∂u/∂t = 0.5∇²u + u(1-u)")
    # {'family': 'fisher_kpp', 'pde_type': 'parabolic', 'linearity': 'nonlinear', ...}
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class PDELabels:
    """All physics labels for a PDE."""
    family: str
    pde_type: str  # parabolic, hyperbolic, elliptic, dispersive
    linearity: str  # linear, nonlinear
    temporal_order: int  # 0, 1, 2
    spatial_order: int  # 1, 2, 3, 4
    dimension: int  # 1, 2, 3
    mechanisms: List[str]  # diffusion, advection, reaction, dispersion
    confidence: float  # 0-1, how confident we are
    raw_features: Dict[str, Any]  # For debugging
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'family': self.family,
            'pde_type': self.pde_type,
            'linearity': self.linearity,
            'temporal_order': self.temporal_order,
            'spatial_order': self.spatial_order,
            'dimension': self.dimension,
            'mechanisms': self.mechanisms,
            'confidence': self.confidence,
        }


class PDEClassifier:
    """Rule-based classifier for PDE strings.
    
    Determines physics labels by parsing the mathematical structure
    of the PDE string.
    """
    
    def __init__(self):
        """Initialize the classifier with pattern definitions."""
        
        # Temporal derivative patterns
        self.temporal_patterns = {
            'first_order': [
                r'∂u/∂t',
                r'∂_t\s*u',
                r'u_t\b',
                r'\\partial_t',
                r'du/dt',
                r'dt\(u\)',      # Functional notation
                r'dt\s*\(',
            ],
            'second_order': [
                r'∂²u/∂t²',
                r'∂\^2u/∂t\^2',
                r'∂_tt\s*u',
                r'u_tt\b',
                r'\\partial_tt',
                r'd²u/dt²',
                r'dtt\(u\)',     # Functional notation
                r'dtt\s*\(',
            ],
        }
        
        # Spatial derivative patterns
        self.spatial_patterns = {
            'laplacian': [
                r'∇²u',
                r'\\nabla\^2',
                r'\\Delta\s*u',
                r'Δu',
                r'∂²u/∂x²',
                r'∂\^2u/∂x\^2',
                r'u_xx',
                r'd²u/dx²',
                r'dxx\(u\)',     # Functional notation
                r'dxx\s*\(',
                r'dyy\s*\(',
                r'dzz\s*\(',
            ],
            'bilaplacian': [
                r'∇⁴u',
                r'∇\^4u',
                r'\\nabla\^4',
                r'Δ²u',
                r'∂⁴u/∂x⁴',
                r'u_xxxx',
                r'dxxxx\(u\)',   # Functional notation
                r'dxxxx\s*\(',
            ],
            'gradient': [
                r'∇u',
                r'\\nabla\s*u',
                r'∂u/∂x',
                r'u_x\b',
                r'du/dx',
                r'dx\(u\)',      # Functional notation
                r'dx\s*\(',
            ],
            'third_order': [
                r'∂³u/∂x³',
                r'∂\^3u/∂x\^3',
                r'u_xxx',
                r'd³u/dx³',
                r'dxxx\(u\)',    # Functional notation
                r'dxxx\s*\(',
            ],
        }
        
        # Nonlinearity patterns (handle both with/without spaces)
        self.nonlinear_patterns = [
            r'u²',
            r'u\s*\^\s*2',       # u^2 or u ^ 2
            r'u\s*\*\*\s*2',     # u**2 or u ** 2
            r'u³',
            r'u\s*\^\s*3',       # u^3 or u ^ 3
            r'u\s*\*\*\s*3',     # u**3 or u ** 3
            r'u\s*\*\s*u',       # u*u or u * u
            r'u\(1-u\)',
            r'u\(1\s*-\s*u\)',
            r'\(1-u\)',
            r'u∂u/∂x',
            r'u\s*u_x',
            r'u\s*∇u',
            r'sin\s*\(\s*u\s*\)',  # sin(u) with optional spaces
            r'cos\s*\(\s*u\s*\)',  # cos(u) with optional spaces
            r'\|u\|',
            r'\|∇u\|',
            r'u\s*\*\s*d[xyz]\(',  # u*dx(u), u * dx(u) - advection term
            r'\(d[xyz]+\(u\)\)\s*\^\s*2',  # (dx(u))^2 - squared derivatives
            r'\(d[xyz]+\(u\)\)\s*\*\s*\(d[xyz]+\(u\)\)',  # (dx(u))*(dx(u))
        ]
        
        # Dimension patterns (including mixed derivatives)
        self.dimension_patterns = {
            3: [r'∂.*/∂z', r'u_z', r'_z\b', r',\s*z\s*[,\)]', r'dzz\(', r'dz\(',
                r'dxxzz\(', r'dyyzz\(', r'dxyzz\(', r'dzzzz\('],  # Mixed derivatives with z
            2: [r'∂.*/∂y', r'u_y', r'_y\b', r',\s*y\s*[,\)]', r'dyy\(', r'dy\(',
                r'dxxyy\(', r'dxyyy\(', r'dyyyy\(', r'dxyy\('],  # Mixed derivatives with y
            1: [],  # Default
        }
        
        # Family-specific patterns (ordered by specificity - most specific first!)
        # Supports both symbolic (∂u/∂t) and functional (dt(u)) notation
        # Patterns designed for actual dataset format
        self.family_patterns = [
            # === FOURTH ORDER ===
            ('kuramoto_sivashinsky', [
                (r'dxxxx\(|∇⁴u|u_xxxx', True), (r'dxx\(|∇²u|u_xx', True),
            ]),
            # cahn_hilliard: HAS time derivative + bilaplacian
            ('cahn_hilliard', [
                (r'dt\(|∂u/∂t|u_t', True),  # HAS time derivative
                (r'dxxxx\(|dyyyy\(|dzzzz\(|∇⁴|u_xxxx', True), 
            ]),
            # biharmonic: NO time derivative, just bilaplacian
            ('biharmonic', [
                (r'dxxxx\(|∇⁴u|Δ²u|u_xxxx', True),
                (r'dt\(|∂u/∂t|u_t', False),  # NO time derivative
            ]),
            
            # === NONLINEAR PARABOLIC ===
            # fisher_kpp: dt - dxx + c*u^2 - c*u (same coefficient!)
            ('fisher_kpp', [
                (r'dt\(|∂u/∂t|u_t', True), (r'dxx\(|∇²u|u_xx', True), 
                (r'u\s*\^\s*2.*-.*u|u\s*\*\*\s*2.*-.*u|u\(1-u\)', True),  # u^2 and -u pattern
            ]),
            # allen_cahn: dt - c*dxx + u^3 - u (has -u at end!)
            ('allen_cahn', [
                (r'dt\(|∂u/∂t|u_t', True), (r'dxx\(|∇²u|u_xx', True), 
                (r'u\s*\^\s*3|u\s*\*\*\s*3|u³', True),  # u^3 with optional spaces
                (r'-\s*u\s*$|-\s*u[^*^0-9]', True),  # Has -u term
                (r'dxxxx', False),
            ]),
            
            
            # === DISPERSIVE ===
            ('kdv', [
                (r'dxxx\(|∂³|u_xxx', True),  # Third order spatial
            ]),
            # reaction_diffusion_cubic: dt - Δu ± g*u^3 (no -u term!)
            ('reaction_diffusion_cubic', [
                (r'dt\(|∂u/∂t', True), 
                (r'u\^3|u\*\*3', True),  # Has u^3
                (r'-\s*u\s*$|-\s*u[^*^0-9]', False),  # NO -u term (unlike allen_cahn)
                (r'dtt\(|∂²u/∂t²', False),  # Not second order in time
            ]),
            
            # === NONLINEAR HYPERBOLIC ===
            # sine_gordon (true): dtt - Δu + beta*sin(u)
            ('sine_gordon', [
                (r'dtt\(|∂²u/∂t²|u_tt', True), (r'dxx\(|∇²u|u_xx', True),
                (r'sin\s*\(\s*u\s*\)', True),
            ]),
            # airy: dt + alpha*dxxx(u) = 0 (linear dispersive, 1D)
            ('airy', [
                (r'dt\(|∂u/∂t|u_t', True),
                (r'dxxx\(|∂³|u_xxx', True),
                (r'dtt\(|∂²u/∂t²|u_tt', False),
                (r'u\^|u\*\*|sin\s*\(', False),
            ]),
            
            # === HYPERBOLIC (second order in time) ===
            ('telegraph', [
                (r'dtt\(|∂²u/∂t²|u_tt', True), (r'dt\(|∂u/∂t|u_t', True), (r'dxx\(|∇²u|u_xx', True),
            ]),
            ('wave', [
                (r'dtt\(|∂²u/∂t²|u_tt', True), (r'dxx\(|∇²u|u_xx', True),
            ]),

            # beam_plate: dtt + kappa*dxxxx (and 2*dxxyy + dyyyy in 2D)
            ('beam_plate', [
                (r'dtt\(|∂²u/∂t²|u_tt', True),
                (r'dxxxx\(|u_xxxx', True),
                (r'dt\(|∂u/∂t|u_t', False),
            ]),
            
            # === BURGERS (nonlinear advection, no constant) ===
            ('burgers', [
                (r'dt\(|∂u/∂t|u_t', True), (r'u\*dx\(|u\*dy\(|u∂u|u\s*u_x', True),
                (r'[+-]\s*\d+\.?\d*$', False),  # No constant at end
            ]),
            
            # === PARABOLIC (first order in time) ===
            ('heat', [
                (r'dt\(|∂u/∂t|u_t', True), (r'dxx\(|dyy\(|dzz\(|∇²u|Δu', True),
            ]),
            ('advection', [
                (r'dt\(|∂u/∂t|u_t', True), (r'dx\(|∂u/∂x|u_x', True), (r'dxx\(|∇²|u_xx|dyy\(', False),
            ]),
            
            # === ELLIPTIC (no time derivative) ===
            ('poisson', [
                (r'dxx\(|∇²u|Δu|u_xx', True), (r'dt\(|∂.*/∂t|u_t', False),
            ]),
        ]
    
    def _match_any(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _count_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match."""
        count = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count
    
    def _detect_temporal_order(self, pde: str) -> int:
        """Detect temporal derivative order (0, 1, or 2)."""
        if self._match_any(pde, self.temporal_patterns['second_order']):
            return 2
        if self._match_any(pde, self.temporal_patterns['first_order']):
            return 1
        return 0
    
    def _detect_spatial_order(self, pde: str) -> int:
        """Detect highest spatial derivative order."""
        if self._match_any(pde, self.spatial_patterns['bilaplacian']):
            return 4
        if self._match_any(pde, self.spatial_patterns['third_order']):
            return 3
        if self._match_any(pde, self.spatial_patterns['laplacian']):
            return 2
        if self._match_any(pde, self.spatial_patterns['gradient']):
            return 1
        return 0
    
    def _detect_dimension(self, pde: str) -> int:
        """Detect spatial dimension (1, 2, or 3)."""
        for dim in [3, 2]:
            if self._match_any(pde, self.dimension_patterns[dim]):
                return dim
        return 1
    
    def _detect_nonlinearity(self, pde: str) -> bool:
        """Detect if PDE is nonlinear."""
        return self._match_any(pde, self.nonlinear_patterns)
    
    def _detect_mechanisms(self, pde: str) -> List[str]:
        """Detect physical mechanisms present."""
        mechanisms = []
        
        # Diffusion: Laplacian term
        if self._match_any(pde, self.spatial_patterns['laplacian']):
            mechanisms.append('diffusion')
        
        # Advection: first-order spatial with nonlinearity or explicit advection
        if (self._match_any(pde, self.spatial_patterns['gradient']) or
            re.search(r'u\s*∂u/∂x|u\s*u_x|u∇u', pde)):
            mechanisms.append('advection')
        
        # Reaction: nonlinear source terms
        if re.search(r'u\(1-u\)|u²|u³|u\^2|u\^3|sin\(u\)|cos\(u\)', pde):
            mechanisms.append('reaction')
        
        # Dispersion: odd-order spatial derivatives
        if self._match_any(pde, self.spatial_patterns['third_order']):
            mechanisms.append('dispersion')
        
        return mechanisms if mechanisms else ['unknown']
    
    def _detect_pde_type(self, temporal_order: int, spatial_order: int, 
                         pde: str, mechanisms: List[str]) -> str:
        """Determine PDE type (parabolic, hyperbolic, elliptic, dispersive)."""
        
        # Schrödinger-type: dispersive
        if re.search(r'i\s*∂|i∂|ψ', pde):
            return 'dispersive'
        
        # KdV-type: dispersive
        if 'dispersion' in mechanisms and temporal_order == 1:
            return 'dispersive'
        
        # Second-order in time: hyperbolic
        if temporal_order == 2:
            return 'hyperbolic'
        
        # First-order in time with diffusion: parabolic
        if temporal_order == 1 and spatial_order >= 2:
            return 'parabolic'
        
        # First-order in time, first-order in space: hyperbolic (advection)
        if temporal_order == 1 and spatial_order == 1:
            return 'hyperbolic'
        
        # No time derivative: elliptic
        if temporal_order == 0:
            return 'elliptic'
        
        return 'unknown'
    
    def _detect_family(self, pde: str, features: Dict[str, Any]) -> Tuple[str, float]:
        """Detect PDE family based on patterns and features.
        
        Uses first-match-wins for families where all required patterns match.
        Families are ordered from most specific to most general.
        """
        
        for family, conditions in self.family_patterns:
            required_met = 0
            required_total = 0
            forbidden_violated = False
            
            for pattern, required in conditions:
                matches = bool(re.search(pattern, pde, re.IGNORECASE))
                if required:
                    required_total += 1
                    if matches:
                        required_met += 1
                else:
                    # Pattern should NOT be present
                    if matches:
                        forbidden_violated = True
                        break
            
            # If all required patterns match and no forbidden patterns present
            if required_total > 0 and required_met == required_total and not forbidden_violated:
                confidence = min(1.0, required_met / 3.0)  # More matches = higher confidence
                return family, confidence
        
        # Fallback based on features
        if features['temporal_order'] == 0:
            return 'poisson', 0.5
        elif features['temporal_order'] == 2:
            return 'wave', 0.5
        elif features['spatial_order'] >= 2:
            return 'heat', 0.5
        else:
            return 'advection', 0.5
    
    def classify(self, pde: str) -> PDELabels:
        """Classify a PDE string and return all physics labels.
        
        Args:
            pde: PDE string (can be in various formats)
            
        Returns:
            PDELabels with all physics properties
        """
        # Clean the input
        pde_clean = pde.strip()
        
        # Extract features
        temporal_order = self._detect_temporal_order(pde_clean)
        spatial_order = self._detect_spatial_order(pde_clean)
        dimension = self._detect_dimension(pde_clean)
        is_nonlinear = self._detect_nonlinearity(pde_clean)
        mechanisms = self._detect_mechanisms(pde_clean)
        
        features = {
            'temporal_order': temporal_order,
            'spatial_order': spatial_order,
            'dimension': dimension,
            'is_nonlinear': is_nonlinear,
            'mechanisms': mechanisms,
        }
        
        # Determine PDE type
        pde_type = self._detect_pde_type(temporal_order, spatial_order, 
                                          pde_clean, mechanisms)
        
        # Determine family
        family, confidence = self._detect_family(pde_clean, features)
        
        return PDELabels(
            family=family,
            pde_type=pde_type,
            linearity='nonlinear' if is_nonlinear else 'linear',
            temporal_order=temporal_order,
            spatial_order=spatial_order,
            dimension=dimension,
            mechanisms=mechanisms,
            confidence=confidence,
            raw_features=features,
        )
    
    def classify_batch(self, pdes: List[str]) -> List[PDELabels]:
        """Classify multiple PDEs."""
        return [self.classify(pde) for pde in pdes]
    
    def compare_to_true(self, pde: str, true_labels: Dict[str, Any]) -> Dict[str, bool]:
        """Compare predicted labels to true labels.
        
        Args:
            pde: PDE string
            true_labels: Dict with true labels
            
        Returns:
            Dict mapping label_name -> is_correct
        """
        predicted = self.classify(pde)
        
        comparisons = {}
        
        if 'family' in true_labels:
            comparisons['family'] = predicted.family == true_labels['family']
        if 'pde_type' in true_labels:
            comparisons['pde_type'] = predicted.pde_type == true_labels['pde_type']
        if 'nonlinear' in true_labels:
            is_nonlinear = true_labels['nonlinear']
            comparisons['linearity'] = (predicted.linearity == 'nonlinear') == is_nonlinear
        if 'dim' in true_labels:
            comparisons['dimension'] = predicted.dimension == true_labels['dim']
        if 'spatial_order' in true_labels:
            comparisons['spatial_order'] = predicted.spatial_order == true_labels['spatial_order']
        if 'temporal_order' in true_labels:
            comparisons['temporal_order'] = predicted.temporal_order == true_labels['temporal_order']
        
        return comparisons


def test_classifier():
    """Test the classifier on known PDEs."""
    
    classifier = PDEClassifier()
    
    test_cases = [
        # (pde_string, expected_family, expected_type)
        ("∂u/∂t = 0.5∇²u", "heat", "parabolic"),
        ("∂²u/∂t² = 4∇²u", "wave", "hyperbolic"),
        ("∇²u = f(x,y)", "poisson", "elliptic"),
        ("∂u/∂t = D∇²u + u(1-u)", "fisher_kpp", "parabolic"),
        ("∂u/∂t = ε∇²u - u³ + u", "allen_cahn", "parabolic"),
        ("∂u/∂t + u∂u/∂x = 0", "burgers", "hyperbolic"),
        ("∂u/∂t + u∂u/∂x = ν∂³u/∂x³", "kdv", "dispersive"),
        ("i∂ψ/∂t = -∇²ψ", "schrodinger", "dispersive"),
        ("∂²u/∂t² + γ∂u/∂t = c²∇²u", "telegraph", "hyperbolic"),
        ("∂²u/∂t² = c²∇²u - sin(u)", "sine_gordon", "hyperbolic"),
    ]
    
    print("=" * 80)
    print("PDE CLASSIFIER TEST")
    print("=" * 80)
    
    correct = 0
    total = len(test_cases)
    
    for pde, expected_family, expected_type in test_cases:
        labels = classifier.classify(pde)
        
        family_ok = labels.family == expected_family
        type_ok = labels.pde_type == expected_type
        
        if family_ok and type_ok:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n{status} PDE: {pde}")
        print(f"   Expected: family={expected_family}, type={expected_type}")
        print(f"   Got:      family={labels.family}, type={labels.pde_type}")
        print(f"   Full:     linearity={labels.linearity}, dim={labels.dimension}, "
              f"mechanisms={labels.mechanisms}")
    
    print(f"\n{'=' * 80}")
    print(f"ACCURACY: {correct}/{total} = {100*correct/total:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    test_classifier()
