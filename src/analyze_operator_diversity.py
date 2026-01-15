#!/usr/bin/env python3
"""Analyze operator diversity in generated PDEs and compare with training dataset."""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

def extract_operators_and_patterns(pde_str):
    """Extract all operators, derivatives, and patterns from a PDE string."""
    patterns = {
        'derivatives': [],
        'operators': [],
        'nonlinear_terms': [],
        'mixed_derivatives': [],
    }
    
    # Extract all derivative patterns
    deriv_patterns = re.findall(r'd[xyz]{1,4}\(|dt\(|dtt\(', pde_str)
    patterns['derivatives'] = deriv_patterns
    
    # Extract mixed derivatives (4th order)
    mixed_4th = re.findall(r'd[xyz]{4,}\(', pde_str)
    patterns['mixed_derivatives'] = mixed_4th
    
    # Extract operators
    operators = re.findall(r'[+\-*/^]', pde_str)
    patterns['operators'] = operators
    
    # Extract nonlinear terms
    if 'u*dx(' in pde_str or 'u*dy(' in pde_str or 'u*dz(' in pde_str:
        patterns['nonlinear_terms'].append('u*dx')
    if 'u^2' in pde_str or 'u**2' in pde_str:
        patterns['nonlinear_terms'].append('u^2')
    if 'u^3' in pde_str or 'u**3' in pde_str:
        patterns['nonlinear_terms'].append('u^3')
    if '(dx(u))^2' in pde_str or '(dy(u))^2' in pde_str or '(dz(u))^2' in pde_str:
        patterns['nonlinear_terms'].append('(dx)^2')
    if 'dxx(u^3)' in pde_str or 'dyy(u^3)' in pde_str or 'dzz(u^3)' in pde_str:
        patterns['nonlinear_terms'].append('dxx(u^3)')
    
    return patterns

def load_pdes_from_file(filepath):
    """Load PDEs from a generation results file."""
    pdes = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[4:]:  # Skip header
                if line.strip():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        pde = parts[1].strip()
                        if pde and not pde.startswith('['):
                            pdes.append(pde)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return pdes

def analyze_training_dataset(sample_size=None):
    """Analyze training dataset to extract operators.
    
    Args:
        sample_size: If None, analyzes full dataset. Otherwise samples first N PDEs.
    """
    training_ops = {
        'derivatives': Counter(),
        'operators': Counter(),
        'nonlinear_terms': Counter(),
        'mixed_derivatives': Counter(),
    }
    
    try:
        dataset_path = Path('data/raw/pde_dataset_45672.csv')
        if not dataset_path.exists():
            print(f"Warning: Training dataset not found at {dataset_path}")
            return {k: set() for k in training_ops}
        
        print(f"Analyzing training dataset from {dataset_path}...")
        if sample_size:
            print(f"  (sampling first {sample_size} PDEs)")
        else:
            print(f"  (full dataset)")
        
        with open(dataset_path, 'r') as f:
            header = f.readline()
            total = 0
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                parts = line.strip().split(',')
                if len(parts) > 0:
                    pde = parts[0]
                    patterns = extract_operators_and_patterns(pde)
                    training_ops['derivatives'].update(patterns['derivatives'])
                    training_ops['operators'].update(patterns['operators'])
                    training_ops['nonlinear_terms'].update(patterns['nonlinear_terms'])
                    training_ops['mixed_derivatives'].update(patterns['mixed_derivatives'])
                    total += 1
        
        print(f"  Analyzed {total} PDEs\n")
        
        # Convert Counter to set for comparison
        return {
            'derivatives': set(training_ops['derivatives'].keys()),
            'operators': set(training_ops['operators'].keys()),
            'nonlinear_terms': set(training_ops['nonlinear_terms'].keys()),
            'mixed_derivatives': set(training_ops['mixed_derivatives'].keys()),
            '_counts': training_ops,  # Keep counts for reporting
        }
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        return {k: set() for k in ['derivatives', 'operators', 'nonlinear_terms', 'mixed_derivatives']}

def main():
    models = [
        ('grammar_beta2e4', 'generation_results/grammar_beta2e4_valid_pdes_n1000_seed42.txt'),
        ('grammar_beta1e2', 'generation_results/grammar_beta1e2_valid_pdes_n1000_seed42.txt'),
        ('token_beta2e4', 'generation_results/token_beta2e4_valid_pdes_n1000_seed42.txt'),
        ('token_beta1e2', 'generation_results/token_beta1e2_valid_pdes_n1000_seed42.txt'),
    ]
    
    print("=" * 80)
    print("OPERATOR DIVERSITY ANALYSIS")
    print("=" * 80)
    
    # Load training dataset patterns (FULL dataset for accuracy)
    print("\n1. Loading training dataset patterns...")
    training_ops_result = analyze_training_dataset(sample_size=None)  # Full dataset
    training_ops = {k: v for k, v in training_ops_result.items() if k != '_counts'}
    training_counts = training_ops_result.get('_counts', {})
    
    print(f"   Training derivatives: {sorted(training_ops['derivatives'])}")
    if training_counts.get('derivatives'):
        print("   Derivative counts:")
        for deriv, count in sorted(training_counts['derivatives'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {deriv:12s}: {count:6d} occurrences")
    print(f"   Training operators: {sorted(training_ops['operators'])}")
    print(f"   Training nonlinear terms: {sorted(training_ops['nonlinear_terms'])}")
    if training_counts.get('nonlinear_terms'):
        print("   Nonlinear term counts:")
        for nl, count in sorted(training_counts['nonlinear_terms'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {nl:15s}: {count:6d} occurrences")
    print(f"   Training mixed derivatives: {sorted(training_ops['mixed_derivatives'])}")
    
    # Analyze each model
    print("\n2. Analyzing generated PDEs...")
    results = {}
    
    for model_name, filepath in models:
        print(f"\n   Analyzing {model_name}...")
        pdes = load_pdes_from_file(filepath)
        
        all_derivs = []
        all_ops = []
        all_nonlinear = []
        all_mixed = []
        unique_pdes = set()
        
        for pde in pdes:
            unique_pdes.add(pde)
            patterns = extract_operators_and_patterns(pde)
            all_derivs.extend(patterns['derivatives'])
            all_ops.extend(patterns['operators'])
            all_nonlinear.extend(patterns['nonlinear_terms'])
            all_mixed.extend(patterns['mixed_derivatives'])
        
        deriv_counts = Counter(all_derivs)
        op_counts = Counter(all_ops)
        nonlinear_counts = Counter(all_nonlinear)
        mixed_counts = Counter(all_mixed)
        
        # Find novel patterns
        unique_derivs = set(deriv_counts.keys())
        unique_nonlinear = set(nonlinear_counts.keys())
        unique_mixed = set(mixed_counts.keys())
        
        novel_derivs = unique_derivs - training_ops['derivatives']
        novel_nonlinear = unique_nonlinear - training_ops['nonlinear_terms']
        novel_mixed = unique_mixed - training_ops['mixed_derivatives']
        
        results[model_name] = {
            'total_pdes': len(pdes),
            'unique_pdes': len(unique_pdes),
            'unique_ratio': len(unique_pdes) / len(pdes) if pdes else 0,
            'derivatives': dict(deriv_counts),
            'operators': dict(op_counts),
            'nonlinear_terms': dict(nonlinear_counts),
            'mixed_derivatives': dict(mixed_counts),
            'novel_derivatives': novel_derivs,
            'novel_nonlinear': novel_nonlinear,
            'novel_mixed': novel_mixed,
            'num_unique_derivs': len(unique_derivs),
            'num_unique_nonlinear': len(unique_nonlinear),
            'num_unique_mixed': len(unique_mixed),
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n📊 UNIQUENESS:")
    for model_name, data in results.items():
        print(f"  {model_name:20s}: {data['unique_pdes']:4d} unique / {data['total_pdes']:4d} total ({100*data['unique_ratio']:5.1f}%)")
    
    print("\n🔬 DERIVATIVE DIVERSITY:")
    for model_name, data in results.items():
        print(f"  {model_name:20s}: {data['num_unique_derivs']:2d} unique derivative types")
        derivs = sorted(data['derivatives'].items(), key=lambda x: x[1], reverse=True)
        top_5 = derivs[:5]
        print(f"    Top: {', '.join([f'{d}({c})' for d, c in top_5])}")
    
    print("\n🔀 NONLINEAR TERM DIVERSITY:")
    for model_name, data in results.items():
        print(f"  {model_name:20s}: {data['num_unique_nonlinear']:2d} unique nonlinear patterns")
        if data['nonlinear_terms']:
            nl = sorted(data['nonlinear_terms'].items(), key=lambda x: x[1], reverse=True)
            print(f"    Patterns: {', '.join([f'{n}({c})' for n, c in nl])}")
        else:
            print(f"    (none)")
    
    print("\n🌐 MIXED DERIVATIVE DIVERSITY:")
    for model_name, data in results.items():
        print(f"  {model_name:20s}: {data['num_unique_mixed']:2d} unique mixed derivatives")
        if data['mixed_derivatives']:
            mixed = sorted(data['mixed_derivatives'].items(), key=lambda x: x[1], reverse=True)
            print(f"    Patterns: {', '.join([f'{m}({c})' for m, c in mixed])}")
        else:
            print(f"    (none)")
    
    print("\n🆕 NOVEL PATTERNS (not in training dataset):")
    has_novel = False
    for model_name, data in results.items():
        novel_count = len(data['novel_derivatives']) + len(data['novel_nonlinear']) + len(data['novel_mixed'])
        if novel_count > 0:
            has_novel = True
            print(f"  {model_name:20s}: {novel_count} novel patterns")
            if data['novel_derivatives']:
                print(f"    Novel derivatives: {sorted(data['novel_derivatives'])}")
            if data['novel_nonlinear']:
                print(f"    Novel nonlinear: {sorted(data['novel_nonlinear'])}")
            if data['novel_mixed']:
                print(f"    Novel mixed: {sorted(data['novel_mixed'])}")
        else:
            print(f"  {model_name:20s}: No novel patterns (all from training data)")
    
    print("\n📈 COVERAGE ANALYSIS (Generated vs Training):")
    for model_name, data in results.items():
        gen_derivs = set(data['derivatives'].keys())
        gen_nonlinear = set(data['nonlinear_terms'].keys())
        gen_mixed = set(data['mixed_derivatives'].keys())
        
        deriv_coverage = len(gen_derivs & training_ops['derivatives']) / len(training_ops['derivatives']) if training_ops['derivatives'] else 0
        nonlinear_coverage = len(gen_nonlinear & training_ops['nonlinear_terms']) / len(training_ops['nonlinear_terms']) if training_ops['nonlinear_terms'] else 0
        mixed_coverage = len(gen_mixed & training_ops['mixed_derivatives']) / len(training_ops['mixed_derivatives']) if training_ops['mixed_derivatives'] else 0
        
        print(f"  {model_name:20s}:")
        print(f"    Derivatives: {100*deriv_coverage:.1f}% ({len(gen_derivs & training_ops['derivatives'])}/{len(training_ops['derivatives'])} types)")
        print(f"    Nonlinear:   {100*nonlinear_coverage:.1f}% ({len(gen_nonlinear & training_ops['nonlinear_terms'])}/{len(training_ops['nonlinear_terms'])} types)")
        print(f"    Mixed derivs: {100*mixed_coverage:.1f}% ({len(gen_mixed & training_ops['mixed_derivatives'])}/{len(training_ops['mixed_derivatives'])} types)")
        
        # Check what's missing
        missing_nonlinear = training_ops['nonlinear_terms'] - gen_nonlinear
        if missing_nonlinear:
            print(f"    ⚠️  Missing nonlinear: {sorted(missing_nonlinear)}")
    
    # Winner analysis
    print("\n" + "=" * 80)
    print("🏆 WINNERS")
    print("=" * 80)
    
    best_unique = max(results.items(), key=lambda x: x[1]['unique_ratio'])
    best_deriv_diversity = max(results.items(), key=lambda x: x[1]['num_unique_derivs'])
    best_nonlinear_diversity = max(results.items(), key=lambda x: x[1]['num_unique_nonlinear'])
    best_mixed_diversity = max(results.items(), key=lambda x: x[1]['num_unique_mixed'])
    
    print(f"  Most unique PDEs: {best_unique[0]} ({100*best_unique[1]['unique_ratio']:.1f}%)")
    print(f"  Most derivative types: {best_deriv_diversity[0]} ({best_deriv_diversity[1]['num_unique_derivs']} types)")
    print(f"  Most nonlinear patterns: {best_nonlinear_diversity[0]} ({best_nonlinear_diversity[1]['num_unique_nonlinear']} patterns)")
    print(f"  Most mixed derivatives: {best_mixed_diversity[0]} ({best_mixed_diversity[1]['num_unique_mixed']} types)")
    
    # Save detailed report
    report_path = Path('generation_results/operator_diversity_report.txt')
    with open(report_path, 'w') as f:
        f.write("OPERATOR DIVERSITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, data in results.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total PDEs: {data['total_pdes']}\n")
            f.write(f"Unique PDEs: {data['unique_pdes']} ({100*data['unique_ratio']:.1f}%)\n\n")
            
            f.write("Derivatives:\n")
            for deriv, count in sorted(data['derivatives'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {deriv:15s}: {count:4d} occurrences\n")
            
            f.write("\nNonlinear Terms:\n")
            if data['nonlinear_terms']:
                for nl, count in sorted(data['nonlinear_terms'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {nl:15s}: {count:4d} occurrences\n")
            else:
                f.write("  (none)\n")
            
            f.write("\nMixed Derivatives:\n")
            if data['mixed_derivatives']:
                for mixed, count in sorted(data['mixed_derivatives'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {mixed:15s}: {count:4d} occurrences\n")
            else:
                f.write("  (none)\n")
            
            if data['novel_derivatives'] or data['novel_nonlinear'] or data['novel_mixed']:
                f.write("\nNovel Patterns:\n")
                if data['novel_derivatives']:
                    f.write(f"  Derivatives: {sorted(data['novel_derivatives'])}\n")
                if data['novel_nonlinear']:
                    f.write(f"  Nonlinear: {sorted(data['novel_nonlinear'])}\n")
                if data['novel_mixed']:
                    f.write(f"  Mixed: {sorted(data['novel_mixed'])}\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")

if __name__ == '__main__':
    main()
