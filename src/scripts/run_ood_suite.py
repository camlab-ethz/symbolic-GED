#!/usr/bin/env python3
"""Run OOD experiment suite (Phase 3).

This script:
1. Trains VAEs on multiple OOD scenarios (or uses existing checkpoints)
2. Extracts latents with is_ood masks
3. Runs analysis comparing Grammar vs Token on OOD samples
4. Generates per-scenario Markdown reports

Usage:
    # Run all scenarios:
    python scripts/run_ood_suite.py

    # Run specific scenarios:
    python scripts/run_ood_suite.py --scenarios hard_families dispersive_ood

    # Analysis only (skip training):
    python scripts/run_ood_suite.py --analysis-only
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def run_ood_training(scenario: str, tokenization: str, seed: int = 42,
                     output_dir: str = 'experiments', extract_latents: bool = True) -> str:
    """Run OOD VAE training for a scenario.

    Returns:
        Path to best checkpoint
    """
    cmd = [
        'python', '-m', 'training.train_vae_ood',
        '--scenario', scenario,
        '--tokenization', tokenization,
        '--seed', str(seed),
        '--output_dir', output_dir,
    ]

    if extract_latents:
        cmd.append('--extract_latents')

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None

    # Find checkpoint
    base_dir = Path(__file__).parent.parent
    ckpt_dir = base_dir / output_dir / scenario / f'seed_{seed}' / f'{tokenization}_checkpoints'

    best_ckpt = None
    best_acc = 0
    for ckpt in ckpt_dir.glob('*.ckpt'):
        name = str(ckpt)
        if 'seq_acc=' in name:
            try:
                acc = float(name.split('seq_acc=')[-1].split('.ckpt')[0])
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = str(ckpt)
            except:
                pass

    return best_ckpt


def run_ood_analysis(scenario: str, seed: int = 42,
                     output_dir: str = 'experiments') -> dict:
    """Run analysis on OOD scenario latents.

    Returns:
        Analysis results dict
    """
    from scripts.run_analysis import load_latent_file, analyze_latents
    from analysis.report_generator import generate_ood_report, save_report
    from ood import get_scenario

    base_dir = Path(__file__).parent.parent
    exp_dir = base_dir / output_dir / scenario / f'seed_{seed}'

    # Load latent files
    grammar_path = exp_dir / 'grammar_latents.npz'
    token_path = exp_dir / 'token_latents.npz'

    if not grammar_path.exists() or not token_path.exists():
        print(f"  Warning: Latent files not found for {scenario}")
        return None

    print(f"  Loading Grammar latents: {grammar_path}")
    grammar_data = load_latent_file(str(grammar_path))

    print(f"  Loading Token latents: {token_path}")
    token_data = load_latent_file(str(token_path))

    # Run analysis
    print("  Analyzing Grammar VAE...")
    grammar_results = analyze_latents(grammar_data)

    print("  Analyzing Token VAE...")
    token_results = analyze_latents(token_data)

    # Get scenario info
    scenario_obj = get_scenario(scenario)
    ood_families = list(scenario_obj.ood_families) if scenario_obj.ood_families else None

    # Generate report
    report = generate_ood_report(
        grammar_results,
        token_results,
        scenario,
        scenario_obj.description,
        ood_families
    )

    # Save report
    report_dir = base_dir / output_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f'ood_{scenario}_summary.md'
    save_report(report, str(report_path))

    return {
        'scenario': scenario,
        'grammar': grammar_results,
        'token': token_results,
        'report_path': str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Run OOD experiment suite')

    parser.add_argument('--scenarios', nargs='+',
                        default=['hard_families', 'dispersive_ood', 'dim_ood_1d2d_to_3d'],
                        help='OOD scenarios to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Skip training, only run analysis')
    parser.add_argument('--training-only', action='store_true',
                        help='Skip analysis, only run training')
    parser.add_argument('--list-scenarios', action='store_true',
                        help='List available scenarios and exit')

    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        from ood import list_scenarios, get_scenario
        print("Available OOD scenarios:")
        for name in list_scenarios():
            scenario = get_scenario(name)
            print(f"  {name}: {scenario.description}")
        return

    base_dir = Path(__file__).parent.parent

    print("=" * 60)
    print("OOD EXPERIMENT SUITE")
    print("=" * 60)
    print(f"Scenarios: {args.scenarios}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")

    all_results = {}

    for scenario in args.scenarios:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario}")
        print("=" * 60)

        # Training
        if not args.analysis_only:
            print("\n[TRAINING]")

            print(f"\nTraining Grammar VAE for {scenario}...")
            grammar_ckpt = run_ood_training(
                scenario, 'grammar', args.seed, args.output_dir
            )
            if grammar_ckpt:
                print(f"  Best checkpoint: {grammar_ckpt}")

            print(f"\nTraining Token VAE for {scenario}...")
            token_ckpt = run_ood_training(
                scenario, 'token', args.seed, args.output_dir
            )
            if token_ckpt:
                print(f"  Best checkpoint: {token_ckpt}")

        # Analysis
        if not args.training_only:
            print("\n[ANALYSIS]")
            results = run_ood_analysis(scenario, args.seed, args.output_dir)
            if results:
                all_results[scenario] = results
                print(f"  Report saved: {results['report_path']}")

    # Save combined results
    if all_results:
        json_path = base_dir / args.output_dir / 'reports' / 'ood_suite_results.json'

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(json_path, 'w') as f:
            json.dump(convert(all_results), f, indent=2)
        print(f"\nCombined JSON saved: {json_path}")

    print("\n" + "=" * 60)
    print("OOD SUITE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
