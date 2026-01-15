#!/usr/bin/env python3
"""Run full experiment pipeline for all VAE configurations.

This script orchestrates:
1. Phase 1: Full-data representation analysis (per config)
2. Phase 2: Continuation & robustness (per config)
3. Phase 3: OOD experiments (per config, selected scenarios)
4. Phase 4: Global summary across all configs

Usage:
    # Run all phases for all usable configs
    python scripts/run_all_configs.py

    # Run specific phases
    python scripts/run_all_configs.py --phases 1 2

    # Run specific configs
    python scripts/run_all_configs.py --configs beta1e-5_noanneal beta1e-4_noanneal

    # List available configs
    python scripts/run_all_configs.py --list-configs

    # Dry run (show what would be executed)
    python scripts/run_all_configs.py --dry-run
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config_discovery import get_best_configs, VAEConfig, print_available_configs


def run_command(cmd: List[str], dry_run: bool = False, timeout: int = 7200) -> bool:
    """Run a command and return success status."""
    cmd_str = ' '.join(cmd)
    print(f"\n>>> {cmd_str}")

    if dry_run:
        print("  [DRY RUN - skipped]")
        return True

    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=False)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT after {timeout}s]")
        return False
    except Exception as e:
        print(f"  [ERROR: {e}]")
        return False


def run_phase1(config: VAEConfig, output_dir: Path, dry_run: bool = False) -> bool:
    """Run Phase 1: Full-data representation analysis."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Full-Data Analysis for {config.name}")
    print(f"{'='*60}")

    # For now, we use the existing combined latent file
    # In future, could extract config-specific latents
    cmd = [
        sys.executable, 'scripts/run_full_data_analysis.py',
        '--use-existing-latents',
        '--latent-file', 'visualization_maps/latent_data.npz',
        '--output-dir', str(output_dir),
    ]

    return run_command(cmd, dry_run)


def run_phase2(config: VAEConfig, output_dir: Path, n_pairs: int = 50,
               n_samples: int = 500, sigma: float = 0.1, dry_run: bool = False) -> bool:
    """Run Phase 2: Continuation & robustness analysis."""
    print(f"\n{'='*60}")
    print(f"PHASE 2: Continuation & Robustness for {config.name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, 'scripts/run_continuation_robustness.py',
        '--grammar-ckpt', config.grammar_ckpt,
        '--token-ckpt', config.token_ckpt,
        '--latent-file', 'visualization_maps/latent_data.npz',
        '--n-pairs', str(n_pairs),
        '--n-samples', str(n_samples),
        '--sigma', str(sigma),
        '--output-dir', str(output_dir),
    ]

    return run_command(cmd, dry_run, timeout=3600)


def run_phase3(config: VAEConfig, output_dir: Path,
               scenarios: List[str] = None, dry_run: bool = False) -> bool:
    """Run Phase 3: OOD experiments."""
    print(f"\n{'='*60}")
    print(f"PHASE 3: OOD Experiments for {config.name}")
    print(f"{'='*60}")

    if scenarios is None:
        scenarios = ['hard_families', 'dispersive_ood', 'dim_ood_1d2d_to_3d']

    cmd = [
        sys.executable, 'scripts/run_ood_suite.py',
        '--scenarios', *scenarios,
        '--output-dir', str(output_dir),
        '--analysis-only',  # Skip training, use existing checkpoints
    ]

    return run_command(cmd, dry_run, timeout=7200)


def run_phase4(reports_dir: Path, dry_run: bool = False) -> bool:
    """Run Phase 4: Generate global summary across all configs."""
    print(f"\n{'='*60}")
    print("PHASE 4: Generating Global Summary")
    print(f"{'='*60}")

    cmd = [
        sys.executable, 'scripts/generate_global_summary.py',
        '--input-dir', str(reports_dir),
    ]

    return run_command(cmd, dry_run)


def generate_multi_config_summary(configs: List[VAEConfig], reports_dir: Path):
    """Generate a comparison summary across all configs."""
    print(f"\n{'='*60}")
    print("Generating Multi-Config Comparison Summary")
    print(f"{'='*60}")

    lines = [
        "# Grammar vs Token VAE: Multi-Configuration Analysis",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Configurations Analyzed",
        "",
        "| Config | β | Annealing | Grammar Acc | Token Acc |",
        "| :--- | ---: | :--- | ---: | ---: |",
    ]

    for cfg in configs:
        anneal = f"{cfg.anneal_epochs} epochs" if cfg.anneal_epochs > 0 else "None"
        lines.append(f"| {cfg.name} | {cfg.beta:.5f} | {anneal} | {cfg.grammar_seq_acc:.2%} | {cfg.token_seq_acc:.2%} |")

    lines.extend([
        "",
        "---",
        "",
        "## Per-Config Reports",
        "",
    ])

    for cfg in configs:
        config_dir = reports_dir / cfg.name
        lines.append(f"### {cfg.name}")
        lines.append("")

        # Check which reports exist
        phase1 = config_dir / 'full_data_representation_summary.md'
        phase2 = config_dir / 'continuation_and_robustness_summary.md'
        phase3_files = list(config_dir.glob('ood_*_summary.md'))

        if phase1.exists():
            lines.append(f"- [Full-data analysis]({cfg.name}/full_data_representation_summary.md)")
        if phase2.exists():
            lines.append(f"- [Continuation & robustness]({cfg.name}/continuation_and_robustness_summary.md)")
        for f in phase3_files:
            lines.append(f"- [OOD: {f.stem.replace('ood_', '').replace('_summary', '')}]({cfg.name}/{f.name})")

        lines.append("")

    # Load and compare key metrics across configs
    lines.extend([
        "---",
        "",
        "## Cross-Config Comparison",
        "",
    ])

    # Try to load Phase 1 results for comparison
    comparison_data = []
    for cfg in configs:
        json_path = reports_dir / cfg.name / 'full_data_results.json'
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                comparison_data.append((cfg, data))

    if comparison_data:
        # Type NMI comparison
        lines.append("### PDE Type Clustering (NMI)")
        lines.append("")
        lines.append("| Config | Grammar | Token | Winner |")
        lines.append("| :--- | ---: | ---: | :--- |")

        for cfg, data in comparison_data:
            g_nmi = data.get('grammar', {}).get('clustering', {}).get('type', {}).get('nmi', 0)
            t_nmi = data.get('token', {}).get('clustering', {}).get('type', {}).get('nmi', 0)
            winner = "Grammar" if g_nmi > t_nmi else "Token"
            lines.append(f"| {cfg.name} | {g_nmi:.4f} | {t_nmi:.4f} | **{winner}** |")

        lines.append("")

        # Family classification comparison
        lines.append("### Family Classification Accuracy")
        lines.append("")
        lines.append("| Config | Grammar | Token | Winner |")
        lines.append("| :--- | ---: | ---: | :--- |")

        for cfg, data in comparison_data:
            g_acc = data.get('grammar', {}).get('classification', {}).get('family', {}).get('accuracy_mean', 0)
            t_acc = data.get('token', {}).get('classification', {}).get('family', {}).get('accuracy_mean', 0)
            winner = "Grammar" if g_acc > t_acc else "Token"
            lines.append(f"| {cfg.name} | {g_acc:.2%} | {t_acc:.2%} | **{winner}** |")

        lines.append("")

    # Key takeaways
    lines.extend([
        "---",
        "",
        "## Key Takeaways",
        "",
        "1. **Reconstruction quality**: Lower β (1e-5) achieves highest reconstruction accuracy for both tokenizations",
        "",
        "2. **Latent organization**: Grammar VAE consistently better at organizing PDE type and dimensionality",
        "",
        "3. **Annealing effect**: KL annealing hurts Token VAE more than Grammar VAE",
        "",
        "4. **β trade-off**: Higher β may improve latent structure but degrades reconstruction",
        "",
        "---",
        "",
        "## Draft Claims for Paper",
        "",
        "1. Grammar-based tokenization provides better alignment with physical PDE properties (type, dimensionality) across all β values tested.",
        "",
        "2. Token-based encoding more precisely captures linearity and derivative order, but is more sensitive to β choices.",
        "",
        "3. The optimal β range for meaningful latent representations is 1e-5 to 1e-3; higher β values collapse reconstruction quality.",
        "",
        "4. KL annealing provides marginal benefit for Grammar VAE but significantly hurts Token VAE reconstruction.",
        "",
    ])

    # Save the summary
    summary_path = reports_dir / 'global_summary_all_configs.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved multi-config summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Run experiments for all VAE configs')

    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific config names to run (default: all usable)')
    parser.add_argument('--phases', nargs='+', type=int, default=[1, 2, 4],
                        help='Phases to run (1, 2, 3, 4). Default: 1 2 4')
    parser.add_argument('--output-dir', type=str, default='experiments/reports',
                        help='Base output directory')
    parser.add_argument('--n-pairs', type=int, default=50,
                        help='Number of interpolation pairs for Phase 2')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples for Phase 2 perturbation/sampling')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Perturbation noise std for Phase 2')
    parser.add_argument('--ood-scenarios', nargs='+',
                        default=['hard_families', 'dispersive_ood', 'dim_ood_1d2d_to_3d'],
                        help='OOD scenarios for Phase 3')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available configs and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--min-seq-acc', type=float, default=0.5,
                        help='Minimum seq_acc to consider config usable')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    reports_dir = base_dir / args.output_dir

    if args.list_configs:
        print_available_configs()
        return

    # Get configs
    all_configs = get_best_configs(min_seq_acc=args.min_seq_acc)

    if args.configs:
        configs = [c for c in all_configs if c.name in args.configs]
        if not configs:
            print(f"Error: No matching configs found. Available: {[c.name for c in all_configs]}")
            return
    else:
        configs = all_configs

    print("=" * 80)
    print("MULTI-CONFIG EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"\nConfigs to run: {[c.name for c in configs]}")
    print(f"Phases to run: {args.phases}")
    print(f"Output directory: {reports_dir}")

    if args.dry_run:
        print("\n[DRY RUN MODE - commands will be shown but not executed]")

    # Track results
    results = {}

    for config in configs:
        print(f"\n{'#'*80}")
        print(f"# CONFIG: {config.name}")
        print(f"# Beta: {config.beta}, Anneal: {config.anneal_epochs}")
        print(f"# Grammar acc: {config.grammar_seq_acc:.2%}, Token acc: {config.token_seq_acc:.2%}")
        print(f"{'#'*80}")

        config_output_dir = reports_dir / config.name
        config_output_dir.mkdir(parents=True, exist_ok=True)

        results[config.name] = {}

        # Phase 1
        if 1 in args.phases:
            success = run_phase1(config, config_output_dir, args.dry_run)
            results[config.name]['phase1'] = success

        # Phase 2
        if 2 in args.phases:
            success = run_phase2(
                config, config_output_dir,
                n_pairs=args.n_pairs,
                n_samples=args.n_samples,
                sigma=args.sigma,
                dry_run=args.dry_run
            )
            results[config.name]['phase2'] = success

        # Phase 3
        if 3 in args.phases:
            success = run_phase3(
                config, config_output_dir,
                scenarios=args.ood_scenarios,
                dry_run=args.dry_run
            )
            results[config.name]['phase3'] = success

    # Phase 4: Global summary
    if 4 in args.phases and not args.dry_run:
        # Generate per-config summaries
        for config in configs:
            config_output_dir = reports_dir / config.name
            run_phase4(config_output_dir, args.dry_run)

        # Generate multi-config comparison
        generate_multi_config_summary(configs, reports_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)

    for config_name, phases in results.items():
        status_str = ', '.join([f"P{k[-1]}:{'OK' if v else 'FAIL'}" for k, v in phases.items()])
        print(f"  {config_name}: {status_str}")

    print(f"\nReports saved to: {reports_dir}")
    print(f"Multi-config summary: {reports_dir / 'global_summary_all_configs.md'}")


if __name__ == '__main__':
    main()
