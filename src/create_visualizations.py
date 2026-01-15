"""Create visualizations for VAE evaluation results.

Generates:
1. t-SNE plots of latent space colored by family
2. Bar charts comparing metrics across models
3. Summary table
"""
import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import csv

sys.path.insert(0, str(Path(__file__).parent))

from vae.module import GrammarVAEModule
from pde import grammar as pde_grammar
from pde.chr_tokenizer import PDETokenizer


def load_and_encode_dataset(checkpoint_path, tokenization, csv_path='pde_dataset_48444_clean.csv', device='cuda'):
    """Load model and encode all PDEs."""
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']

    model = GrammarVAEModule(
        P=hparams['P'],
        max_length=hparams['max_length'],
        z_dim=hparams.get('z_dim', 26),
        lr=hparams.get('lr', 0.001),
        beta=hparams.get('beta', 1e-5),
        encoder_hidden=hparams.get('encoder_hidden', 128),
        encoder_conv_layers=hparams.get('encoder_conv_layers', 3),
        encoder_kernel=hparams.get('encoder_kernel', [7, 7, 7]),
        decoder_hidden=hparams.get('decoder_hidden', 80),
        decoder_layers=hparams.get('decoder_layers', 3),
        decoder_dropout=hparams.get('decoder_dropout', 0.1),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    max_length = hparams['max_length']
    P = hparams['P']

    # Load dataset
    pdes = []
    families = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdes.append(row['pde'])
            families.append(row['family'])

    # Encode PDEs
    tokenizer = PDETokenizer() if tokenization == 'token' else None

    all_mu = []
    valid_families = []

    with torch.no_grad():
        for pde, family in zip(pdes, families):
            try:
                if tokenization == 'grammar':
                    seq = pde_grammar.parse_to_productions(pde.replace(' ', ''))
                    x = torch.zeros(1, max_length, P)
                    for t, pid in enumerate(seq[:max_length]):
                        if 0 <= pid < P:
                            x[0, t, pid] = 1.0
                else:
                    ids = tokenizer.encode(pde)
                    x = torch.zeros(1, max_length, P)
                    for t, tid in enumerate(ids[:max_length]):
                        if 0 <= tid < P:
                            x[0, t, tid] = 1.0

                x = x.to(device)
                mu, _ = model.encoder(x)
                all_mu.append(mu.cpu().numpy())
                valid_families.append(family)
            except:
                continue

    return np.vstack(all_mu), valid_families


def create_tsne_plots(results_path, output_dir='plots'):
    """Create t-SNE visualizations for all models."""
    Path(output_dir).mkdir(exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Family color mapping
    family_colors = {
        'heat': '#e41a1c',
        'wave': '#377eb8',
        'advection': '#4daf4a',
        'burgers': '#984ea3',
        'kdv': '#ff7f00',
        'fisher_kpp': '#ffff33',
        'allen_cahn': '#a65628',
        'telegraph': '#f781bf',
        'reaction_diffusion_cubic': '#999999',
        'biharmonic': '#fc8d62',
        'poisson': '#8da0cb',
        'airy': '#e78ac3',
        'beam_plate': '#a6d854',
        'fokker_planck': '#ffd92f',
        'cahn_hilliard': '#e5c494',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    model_names = ['grammar_fixed', 'token_fixed', 'grammar_cyclical', 'token_cyclical']
    titles = ['Grammar VAE (β=1e-4)', 'Token VAE (β=1e-4)',
              'Grammar VAE Cyclical (β=1e-3)', 'Token VAE Cyclical (β=1e-3)']

    for idx, (model_name, title) in enumerate(zip(model_names, titles)):
        ax = axes[idx // 2, idx % 2]

        if model_name not in results:
            ax.set_title(f'{title}\n(Not available)')
            continue

        info = results[model_name]
        checkpoint = info['checkpoint']
        tokenization = info['tokenization']

        print(f"Encoding dataset for {model_name}...")
        try:
            embeddings, families = load_and_encode_dataset(checkpoint, tokenization, device=device)

            # Subsample if too large
            if len(embeddings) > 5000:
                indices = np.random.choice(len(embeddings), 5000, replace=False)
                embeddings = embeddings[indices]
                families = [families[i] for i in indices]

            print(f"  Running t-SNE on {len(embeddings)} samples...")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
            coords = tsne.fit_transform(embeddings)

            # Plot each family
            unique_families = list(set(families))
            for family in unique_families:
                mask = [f == family for f in families]
                color = family_colors.get(family, '#000000')
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=color, label=family, alpha=0.6, s=10)

            # Add metrics to title
            nmi = info['clustering']['family']['nmi']
            validity = info['generative']['validity']
            ax.set_title(f'{title}\nNMI={nmi:.3f}, Validity={validity:.1f}%')

        except Exception as e:
            ax.set_title(f'{title}\nError: {str(e)[:30]}')
            print(f"  Error: {e}")

    # Add legend to the right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/tsne_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/tsne_comparison.pdf', bbox_inches='tight')
    print(f"Saved t-SNE plots to {output_dir}/tsne_comparison.png")
    plt.close()


def create_bar_charts(results_path, output_dir='plots'):
    """Create bar charts comparing metrics across models."""
    Path(output_dir).mkdir(exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    models = ['grammar_fixed', 'token_fixed', 'grammar_cyclical', 'token_cyclical']
    labels = ['Grammar\nFixed', 'Token\nFixed', 'Grammar\nCyclical', 'Token\nCyclical']
    colors = ['#2ecc71', '#3498db', '#27ae60', '#2980b9']

    # Metrics to plot
    metrics = [
        ('Family NMI', [results[m]['clustering']['family']['nmi'] for m in models]),
        ('Family ARI', [results[m]['clustering']['family']['ari'] for m in models]),
        ('Validity %', [results[m]['generative']['validity'] for m in models]),
        ('Novel (exact) %', [results[m]['generative']['novelty_exact'] for m in models]),
        ('Novel (struct) %', [results[m]['generative']['novelty_structural'] for m in models]),
        ('Novel (family) %', [results[m]['generative']['novelty_family'] for m in models]),
        ('Diversity %', [results[m]['generative']['diversity'] for m in models]),
        ('Interp Valid %', [results[m]['interpolation']['interp_validity'] for m in models]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (metric_name, values) in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(range(len(models)), values, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if val < 10 else f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9)

    plt.suptitle('Grammar VAE vs Token VAE: Comprehensive Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/metrics_comparison.pdf', bbox_inches='tight')
    print(f"Saved bar charts to {output_dir}/metrics_comparison.png")
    plt.close()


def create_summary_table(results_path, output_dir='plots'):
    """Create a summary table as image."""
    Path(output_dir).mkdir(exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    # Table data
    headers = ['Metric', 'Grammar Fixed', 'Token Fixed', 'Grammar Cyclical', 'Token Cyclical']
    rows = [
        ['Seq Accuracy', '97.87%', '96.88%', '95.73%', '89.97%'],
        ['β (KL weight)', '1e-4', '1e-4', '1e-3', '1e-3'],
        ['Family NMI', f"{results['grammar_fixed']['clustering']['family']['nmi']:.3f}",
                       f"{results['token_fixed']['clustering']['family']['nmi']:.3f}",
                       f"{results['grammar_cyclical']['clustering']['family']['nmi']:.3f}",
                       f"{results['token_cyclical']['clustering']['family']['nmi']:.3f}"],
        ['Family ARI', f"{results['grammar_fixed']['clustering']['family']['ari']:.3f}",
                       f"{results['token_fixed']['clustering']['family']['ari']:.3f}",
                       f"{results['grammar_cyclical']['clustering']['family']['ari']:.3f}",
                       f"{results['token_cyclical']['clustering']['family']['ari']:.3f}"],
        ['Validity %', f"{results['grammar_fixed']['generative']['validity']:.1f}%",
                       f"{results['token_fixed']['generative']['validity']:.1f}%",
                       f"{results['grammar_cyclical']['generative']['validity']:.1f}%",
                       f"{results['token_cyclical']['generative']['validity']:.1f}%"],
        ['Novel (exact) %', f"{results['grammar_fixed']['generative']['novelty_exact']:.1f}%",
                            f"{results['token_fixed']['generative']['novelty_exact']:.1f}%",
                            f"{results['grammar_cyclical']['generative']['novelty_exact']:.1f}%",
                            f"{results['token_cyclical']['generative']['novelty_exact']:.1f}%"],
        ['Novel (struct) %', f"{results['grammar_fixed']['generative']['novelty_structural']:.1f}%",
                             f"{results['token_fixed']['generative']['novelty_structural']:.1f}%",
                             f"{results['grammar_cyclical']['generative']['novelty_structural']:.1f}%",
                             f"{results['token_cyclical']['generative']['novelty_structural']:.1f}%"],
        ['Novel (family) %', f"{results['grammar_fixed']['generative']['novelty_family']:.1f}%",
                             f"{results['token_fixed']['generative']['novelty_family']:.1f}%",
                             f"{results['grammar_cyclical']['generative']['novelty_family']:.1f}%",
                             f"{results['token_cyclical']['generative']['novelty_family']:.1f}%"],
        ['Interp Valid %', f"{results['grammar_fixed']['interpolation']['interp_validity']:.1f}%",
                           f"{results['token_fixed']['interpolation']['interp_validity']:.1f}%",
                           f"{results['grammar_cyclical']['interpolation']['interp_validity']:.1f}%",
                           f"{results['token_cyclical']['interpolation']['interp_validity']:.1f}%"],
        ['Active Units', f"{results['grammar_fixed']['geometry']['active_units']}/26",
                         f"{results['token_fixed']['geometry']['active_units']}/26",
                         f"{results['grammar_cyclical']['geometry']['active_units']}/26",
                         f"{results['token_cyclical']['geometry']['active_units']}/26"],
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers,
                    loc='center', cellLoc='center',
                    colColours=['#f0f0f0']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight best values
    for row_idx, row in enumerate(rows):
        if 'NMI' in row[0] or 'ARI' in row[0]:
            vals = [float(v) for v in row[1:]]
            best_idx = vals.index(max(vals)) + 1
            table[(row_idx + 1, best_idx)].set_facecolor('#90EE90')
        elif 'Validity' in row[0]:
            vals = [float(v.replace('%', '')) for v in row[1:]]
            best_idx = vals.index(max(vals)) + 1
            table[(row_idx + 1, best_idx)].set_facecolor('#90EE90')

    plt.title('Grammar VAE vs Token VAE: Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_table.png', dpi=150, bbox_inches='tight')
    print(f"Saved summary table to {output_dir}/summary_table.png")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='evaluation_results.json')
    parser.add_argument('--output', default='plots')
    parser.add_argument('--skip-tsne', action='store_true', help='Skip t-SNE (slow)')
    args = parser.parse_args()

    print("Creating visualizations...")

    # Bar charts (fast)
    create_bar_charts(args.results, args.output)

    # Summary table (fast)
    create_summary_table(args.results, args.output)

    # t-SNE plots (slow)
    if not args.skip_tsne:
        create_tsne_plots(args.results, args.output)
    else:
        print("Skipping t-SNE plots (--skip-tsne flag)")

    print("\nAll visualizations complete!")
