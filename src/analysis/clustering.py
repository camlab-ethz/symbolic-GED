"""Clustering metrics for latent space analysis.

This module provides functions for evaluating how well the latent space
organizes PDEs according to physics labels.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def compute_clustering_metrics(
    latents: np.ndarray,
    labels: Union[List, np.ndarray],
    label_name: str = 'label',
    max_samples: int = 10000,
    random_state: int = 42
) -> Dict:
    """Compute clustering metrics for given labels.

    Computes:
    - ARI (Adjusted Rand Index): How well clusters match true labels
    - NMI (Normalized Mutual Information): Information shared between clusters and labels
    - Purity: Fraction of samples in majority class per cluster
    - Silhouette: How well-separated clusters are

    Args:
        latents: (N, D) latent vectors
        labels: (N,) label array
        label_name: Name of label for reporting
        max_samples: Max samples for silhouette (memory-intensive)
        random_state: Random seed

    Returns:
        Dictionary with metrics
    """
    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {
            'label_name': label_name,
            'n_classes': n_classes,
            'classes': list(le.classes_),
            'ari': np.nan,
            'nmi': np.nan,
            'purity': np.nan,
            'silhouette': np.nan,
            'silhouette_wrt_labels': np.nan,
            'error': 'Need at least 2 classes'
        }

    # K-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=random_state, n_init=10)
    pred = kmeans.fit_predict(latents)

    # ARI and NMI
    ari = adjusted_rand_score(y, pred)
    nmi = normalized_mutual_info_score(y, pred)

    # Purity: for each cluster, count most common label
    contingency = defaultdict(lambda: defaultdict(int))
    for true, p in zip(y, pred):
        contingency[p][true] += 1
    purity = sum(max(contingency[c].values()) for c in contingency) / len(y)

    # Silhouette w.r.t. clusters (subsample if too large)
    n = len(latents)
    if n > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, max_samples, replace=False)
        sil_clusters = silhouette_score(latents[idx], pred[idx])
        sil_labels = silhouette_score(latents[idx], y[idx])
    else:
        sil_clusters = silhouette_score(latents, pred)
        sil_labels = silhouette_score(latents, y)

    return {
        'label_name': label_name,
        'n_classes': n_classes,
        'classes': list(le.classes_),
        'ari': float(ari),
        'nmi': float(nmi),
        'purity': float(purity),
        'silhouette': float(sil_clusters),  # Silhouette w.r.t. cluster assignments
        'silhouette_wrt_labels': float(sil_labels)  # Silhouette w.r.t. true labels
    }


def compute_all_clustering(
    latents: np.ndarray,
    labels_dict: Dict[str, Union[List, np.ndarray]],
    max_samples: int = 10000,
    random_state: int = 42
) -> Dict[str, Dict]:
    """Compute clustering metrics for multiple label types.

    Args:
        latents: (N, D) latent vectors
        labels_dict: Dictionary mapping label_name -> label array
        max_samples: Max samples for silhouette
        random_state: Random seed

    Returns:
        Dictionary mapping label_name -> metrics dict
    """
    results = {}

    for label_name, labels in labels_dict.items():
        # Skip if too few unique values
        unique = set(labels) if isinstance(labels, list) else set(labels.tolist())
        if len(unique) < 2:
            continue

        results[label_name] = compute_clustering_metrics(
            latents, labels, label_name, max_samples, random_state
        )

    return results


def compute_ood_aware_clustering(
    latents: np.ndarray,
    labels: Union[List, np.ndarray],
    is_ood: np.ndarray,
    label_name: str = 'label',
    **kwargs
) -> Dict[str, Dict]:
    """Compute clustering metrics separately for IID and OOD samples.

    Args:
        latents: (N, D) latent vectors
        labels: (N,) label array
        is_ood: (N,) boolean mask (True = OOD)
        label_name: Name of label
        **kwargs: Additional args for compute_clustering_metrics

    Returns:
        Dictionary with 'all', 'iid', 'ood' metrics
    """
    labels = np.array(labels)

    results = {
        'all': compute_clustering_metrics(latents, labels, f'{label_name}_all', **kwargs),
    }

    # IID only
    iid_mask = ~is_ood
    if iid_mask.sum() > 10:
        results['iid'] = compute_clustering_metrics(
            latents[iid_mask], labels[iid_mask], f'{label_name}_iid', **kwargs
        )

    # OOD only
    if is_ood.sum() > 10:
        results['ood'] = compute_clustering_metrics(
            latents[is_ood], labels[is_ood], f'{label_name}_ood', **kwargs
        )

    return results


def train_classifier(
    latents: np.ndarray,
    labels: Union[List, np.ndarray],
    label_name: str = 'label',
    cv: int = 5,
    max_iter: int = 1000,
    random_state: int = 42
) -> Dict:
    """Train classifier from latent z to physics label.

    Uses logistic regression with cross-validation.

    Args:
        latents: (N, D) latent vectors
        labels: (N,) label array
        label_name: Name of label
        cv: Number of cross-validation folds
        max_iter: Max iterations for logistic regression
        random_state: Random seed

    Returns:
        Dictionary with accuracy and classes
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {
            'label_name': label_name,
            'n_classes': n_classes,
            'accuracy_mean': np.nan,
            'accuracy_std': np.nan,
            'classes': list(le.classes_),
            'error': 'Need at least 2 classes'
        }

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state, n_jobs=-1)
    scores = cross_val_score(clf, latents, y, cv=cv, scoring='accuracy')

    return {
        'label_name': label_name,
        'n_classes': n_classes,
        'accuracy_mean': float(scores.mean()),
        'accuracy_std': float(scores.std()),
        'classes': list(le.classes_)
    }


def train_ood_aware_classifier(
    latents: np.ndarray,
    labels: Union[List, np.ndarray],
    is_ood: np.ndarray,
    label_name: str = 'label',
    **kwargs
) -> Dict[str, Dict]:
    """Train classifier and evaluate on IID vs OOD.

    Trains on IID data, evaluates on both IID (cross-val) and OOD (holdout).

    Args:
        latents: (N, D) latent vectors
        labels: (N,) label array
        is_ood: (N,) boolean mask
        label_name: Name of label
        **kwargs: Additional args for train_classifier

    Returns:
        Dictionary with 'iid' and 'ood' results
    """
    from sklearn.model_selection import train_test_split

    labels = np.array(labels)
    iid_mask = ~is_ood

    # Train on IID
    X_iid = latents[iid_mask]
    y_iid = labels[iid_mask]

    results = {
        'iid': train_classifier(X_iid, y_iid, f'{label_name}_iid', **kwargs)
    }

    # If OOD samples exist, evaluate transfer
    if is_ood.sum() > 0:
        X_ood = latents[is_ood]
        y_ood = labels[is_ood]

        # Train on all IID, test on OOD
        le = LabelEncoder()
        le.fit(y_iid)  # Fit on IID labels

        # Check if OOD has same labels
        ood_labels_in_train = set(y_ood) <= set(y_iid)

        if ood_labels_in_train:
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            clf.fit(X_iid, le.transform(y_iid))
            ood_acc = clf.score(X_ood, le.transform(y_ood))

            results['ood'] = {
                'label_name': f'{label_name}_ood',
                'accuracy': float(ood_acc),
                'n_samples': int(is_ood.sum()),
                'note': 'Trained on IID, tested on OOD'
            }
        else:
            results['ood'] = {
                'label_name': f'{label_name}_ood',
                'accuracy': np.nan,
                'n_samples': int(is_ood.sum()),
                'note': 'OOD contains unseen labels'
            }

    return results


def compare_models(
    grammar_latents: np.ndarray,
    token_latents: np.ndarray,
    labels_dict: Dict[str, Union[List, np.ndarray]],
    is_ood: np.ndarray = None
) -> Dict:
    """Compare Grammar VAE vs Token VAE on clustering metrics.

    Args:
        grammar_latents: Grammar VAE latents
        token_latents: Token VAE latents
        labels_dict: Dictionary of label arrays
        is_ood: Optional OOD mask

    Returns:
        Comparison results
    """
    results = {
        'grammar': {},
        'token': {},
        'comparison': {}
    }

    for label_name, labels in labels_dict.items():
        # Compute for both
        if is_ood is not None:
            g_metrics = compute_ood_aware_clustering(grammar_latents, labels, is_ood, label_name)
            t_metrics = compute_ood_aware_clustering(token_latents, labels, is_ood, label_name)
        else:
            g_metrics = compute_clustering_metrics(grammar_latents, labels, label_name)
            t_metrics = compute_clustering_metrics(token_latents, labels, label_name)

        results['grammar'][label_name] = g_metrics
        results['token'][label_name] = t_metrics

        # Compare
        if isinstance(g_metrics, dict) and 'all' in g_metrics:
            g_nmi = g_metrics['all'].get('nmi', 0)
            t_nmi = t_metrics['all'].get('nmi', 0)
        else:
            g_nmi = g_metrics.get('nmi', 0) if isinstance(g_metrics, dict) else 0
            t_nmi = t_metrics.get('nmi', 0) if isinstance(t_metrics, dict) else 0

        results['comparison'][label_name] = {
            'grammar_nmi': g_nmi,
            'token_nmi': t_nmi,
            'winner': 'grammar' if g_nmi > t_nmi else 'token',
            'diff': abs(g_nmi - t_nmi)
        }

    return results


def print_comparison_table(results: Dict, title: str = "Clustering Comparison"):
    """Print formatted comparison table.

    Args:
        results: Output from compare_models()
        title: Table title
    """
    print(f"\n{'='*80}")
    print(title)
    print('='*80)

    print(f"\n{'Label':<25} {'Metric':<10} {'Grammar VAE':>15} {'Token VAE':>15} {'Winner':>10}")
    print("-"*75)

    for label_name in results['comparison']:
        g = results['grammar'][label_name]
        t = results['token'][label_name]

        # Handle OOD-aware results
        if isinstance(g, dict) and 'all' in g:
            g = g['all']
            t = t['all']

        if isinstance(g, dict):
            for metric in ['ari', 'nmi', 'purity', 'silhouette']:
                if metric in g:
                    g_val = g[metric]
                    t_val = t[metric]
                    winner = "Grammar" if g_val > t_val else "Token"
                    print(f"{label_name:<25} {metric:<10} {g_val:>15.4f} {t_val:>15.4f} {winner:>10}")
            print()
