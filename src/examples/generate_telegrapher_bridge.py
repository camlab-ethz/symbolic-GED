#!/usr/bin/env python3
"""
Example: Generate Telegrapher Bridge Dataset for Diffusion ↔ Wave Continuation

This demonstrates the specialized telegrapher dataset for testing symbolic continuation
between diffusion-like (strong damping) and wave-like (weak damping) regimes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_creation.generator import PDEGenerator


def main():
    print("=" * 80)
    print("TELEGRAPHER BRIDGE DATASET GENERATOR")
    print("Diffusion ↔ Wave Continuation Experiment")
    print("=" * 80)
    
    # Create generator
    gen = PDEGenerator(seed=42)
    
    # Generate bridge dataset with default tau values
    print("\n1. Generating with default tau ranges...")
    bridge_default = gen.generate_telegrapher_bridge()
    
    print(f"\nGenerated {len(bridge_default)} PDEs:")
    print(f"{'Split':<20s} {'Tau':<10s} {'a (1/tau)':<12s} {'Regime':<15s} {'PDE'}")
    print("-" * 100)
    
    for e in bridge_default:
        regime = ""
        if e['tau'] <= 0.1:
            regime = "Diffusion-like"
        elif e['tau'] >= 5.0:
            regime = "Wave-like"
        else:
            regime = "Continuation"
        
        print(f"{e['split']:<20s} {e['tau']:<10.3f} {e['coefficients']['a']:<12.3f} "
              f"{regime:<15s} {e['pde']}")
    
    # Save to CSV
    gen.save_dataset(bridge_default, 'telegrapher_bridge_default.csv', format='csv')
    print(f"\n✓ Saved to: telegrapher_bridge_default.csv")
    
    # Generate with custom tau values
    print("\n" + "=" * 80)
    print("2. Generating with custom tau ranges...")
    bridge_custom = gen.generate_telegrapher_bridge(
        tau_small=[0.01, 0.03, 0.08],      # More diffusion-like points
        tau_mid=[0.3, 0.7, 1.5, 3.0],      # More continuation points
        tau_large=[8.0, 15.0],             # More wave-like points
        c_sq=2.5,                          # Different wave speed
    )
    
    print(f"\nGenerated {len(bridge_custom)} PDEs with custom ranges")
    gen.save_dataset(bridge_custom, 'telegrapher_bridge_custom.csv', format='csv')
    print(f"✓ Saved to: telegrapher_bridge_custom.csv")
    
    # Statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    for name, dataset in [("Default", bridge_default), ("Custom", bridge_custom)]:
        train = [e for e in dataset if e['split'] == 'train_endpoints']
        test = [e for e in dataset if e['split'] == 'test_middle']
        
        print(f"\n{name} Dataset:")
        print(f"  Train endpoints: {len(train)}")
        print(f"    - Tau range: [{min(e['tau'] for e in train):.3f}, {max(e['tau'] for e in train):.3f}]")
        print(f"  Test middle:     {len(test)}")
        print(f"    - Tau range: [{min(e['tau'] for e in test):.3f}, {max(e['tau'] for e in test):.3f}]")
        print(f"  Total:           {len(dataset)}")
    
    # Show physical interpretation
    print("\n" + "=" * 80)
    print("PHYSICAL INTERPRETATION")
    print("=" * 80)
    print("""
Telegraph equation: ∂²u/∂t² + a*∂u/∂t - c²*∂²u/∂x² = 0

Where a = 1/τ (damping parameter):

1. Small τ (large a) → Strong damping → Diffusion-like behavior
   - Example: τ=0.02 → a=50.0 → dtt(u) + 50.0*dt(u) - 1.0*dxx(u) = 0
   - Overdamped: second derivative term negligible

2. Large τ (small a) → Weak damping → Wave-like behavior
   - Example: τ=10.0 → a=0.1 → dtt(u) + 0.1*dt(u) - 1.0*dxx(u) = 0
   - Underdamped: oscillatory solutions

3. Middle τ (moderate a) → Transition regime (UNSEEN during training)
   - Example: τ=1.0 → a=1.0 → dtt(u) + 1.0*dt(u) - 1.0*dxx(u) = 0
   - Tests model's ability to interpolate symbolic behavior

EXPERIMENT:
- Train VAE only on endpoints (tau_small + tau_large)
- Test reconstruction accuracy on middle τ values
- Compare: Grammar vs Token vs Tag tokenization
- Question: Which encoding captures the diffusion↔wave continuation better?
    """)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Encode dataset using both tokenization methods:
   - Grammar: python3 encode_grammar.py telegrapher_bridge_default.csv
   - Token:   python3 encode_token.py telegrapher_bridge_default.csv

2. Split data:
   - Train: filter split == 'train_endpoints'
   - Test:  filter split == 'test_middle'

3. Train VAE on endpoints only

4. Evaluate reconstruction accuracy:
   - Endpoints (should be high)
   - Middle (key test: can model extrapolate?)

5. Compare methods:
   - Which tokenization gives better continuation?
   - Grammar (structural) vs Token (sequential) vs Tag (discrete)
    """)


if __name__ == '__main__':
    main()
