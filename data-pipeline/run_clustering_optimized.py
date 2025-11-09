#!/usr/bin/env python3
"""
Story 4.3: Optimized Wallet Clustering Analysis
Try multiple parameter configurations to find best clustering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("STORY 4.3: OPTIMIZED WALLET CLUSTERING ANALYSIS")
print("=" * 80)
print()

# Load data
DATA_DIR = Path("outputs/features")
OUTPUT_DIR = Path("outputs/clustering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

cleaned_files = list(DATA_DIR.glob("wallet_features_cleaned_*.csv"))
input_file = max(cleaned_files, key=lambda p: p.stat().st_mtime)
df = pd.read_csv(input_file)

print(f"Loaded: {len(df):,} wallets")

# Feature selection
exclude_cols = ["wallet_address", "activity_segment"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features: {len(feature_cols)}")
print()

# ============================================================================
# OPTIMIZATION 1: Try PCA for dimensionality reduction
# ============================================================================
print("OPTIMIZATION 1: Testing PCA dimensionality reduction...")
print("-" * 80)

pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X_scaled.shape[1]}")
print(f"PCA dimensions: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
print()

# ============================================================================
# OPTIMIZATION 2: Grid search for HDBSCAN parameters
# ============================================================================
print("OPTIMIZATION 2: HDBSCAN parameter grid search...")
print("-" * 80)

hdbscan_configs = [
    # Try different min_cluster_size values
    {'min_cluster_size': 30, 'min_samples': 5, 'data': 'original'},
    {'min_cluster_size': 40, 'min_samples': 8, 'data': 'original'},
    {'min_cluster_size': 60, 'min_samples': 12, 'data': 'original'},

    # Try with PCA-reduced features
    {'min_cluster_size': 30, 'min_samples': 5, 'data': 'pca'},
    {'min_cluster_size': 40, 'min_samples': 8, 'data': 'pca'},
    {'min_cluster_size': 50, 'min_samples': 10, 'data': 'pca'},
]

best_hdbscan = None
best_hdbscan_score = -1

for i, config in enumerate(hdbscan_configs, 1):
    data_choice = X_pca if config['data'] == 'pca' else X_scaled

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config['min_cluster_size'],
        min_samples=config['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )

    labels = clusterer.fit_predict(data_choice)
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / len(labels)

    if n_clusters > 1 and mask.sum() > 0:
        silhouette = silhouette_score(data_choice[mask], labels[mask])
    else:
        silhouette = 0.0

    print(f"Config {i}: min_size={config['min_cluster_size']}, "
          f"min_samples={config['min_samples']}, data={config['data']}")
    print(f"  Clusters: {n_clusters}, Noise: {noise_ratio:.1%}, Silhouette: {silhouette:.4f}")

    if silhouette > best_hdbscan_score:
        best_hdbscan_score = silhouette
        best_hdbscan = {
            'config': config,
            'labels': labels,
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': silhouette,
            'data': data_choice,
        }

print()
print(f"Best HDBSCAN: Silhouette = {best_hdbscan_score:.4f}")
print(f"  Config: {best_hdbscan['config']}")
print()

# ============================================================================
# OPTIMIZATION 3: Enhanced K-Means with PCA
# ============================================================================
print("OPTIMIZATION 3: K-Means on PCA-reduced features...")
print("-" * 80)

k_values = [3, 5, 7, 10]
kmeans_results = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_original = kmeans.fit_predict(X_scaled)
    labels_pca = kmeans.fit_predict(X_pca)

    sil_original = silhouette_score(X_scaled, labels_original)
    sil_pca = silhouette_score(X_pca, labels_pca)

    print(f"k={k}: Silhouette (original)={sil_original:.4f}, Silhouette (PCA)={sil_pca:.4f}")

    # Store better performing version
    if sil_pca > sil_original:
        kmeans_results[k] = {
            'labels': labels_pca,
            'silhouette': sil_pca,
            'data_type': 'PCA',
            'data': X_pca,
        }
    else:
        kmeans_results[k] = {
            'labels': labels_original,
            'silhouette': sil_original,
            'data_type': 'Original',
            'data': X_scaled,
        }

# Find best K-Means
best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
best_kmeans = kmeans_results[best_k]

print()
print(f"Best K-Means: k={best_k}, Silhouette = {best_kmeans['silhouette']:.4f} ({best_kmeans['data_type']})")
print()

# ============================================================================
# FINAL SELECTION
# ============================================================================
print("FINAL ALGORITHM SELECTION...")
print("-" * 80)

print(f"HDBSCAN (best): Silhouette = {best_hdbscan['silhouette']:.4f}, Noise = {best_hdbscan['noise_ratio']:.1%}")
print(f"K-Means (best): Silhouette = {best_kmeans['silhouette']:.4f}, k = {best_k}")
print()

# Selection logic: Prefer HDBSCAN if silhouette is close AND noise is acceptable
if (best_hdbscan['silhouette'] >= 0.4 and best_hdbscan['noise_ratio'] < 0.30) or \
   (best_hdbscan['silhouette'] > best_kmeans['silhouette'] * 1.1):
    selected_algorithm = f"HDBSCAN ({best_hdbscan['config']['data']})"
    selected_labels = best_hdbscan['labels']
    selected_metrics = {
        'n_clusters': best_hdbscan['n_clusters'],
        'silhouette': best_hdbscan['silhouette'],
        'noise_ratio': best_hdbscan['noise_ratio'],
    }
    X_final = best_hdbscan['data']
    print(f"✅ Selected: HDBSCAN with {best_hdbscan['config']}")
else:
    selected_algorithm = f"K-Means (k={best_k}, {best_kmeans['data_type']})"
    selected_labels = best_kmeans['labels']
    selected_metrics = {
        'n_clusters': best_k,
        'silhouette': best_kmeans['silhouette'],
        'noise_ratio': 0.0,
    }
    X_final = best_kmeans['data']
    print(f"✅ Selected: K-Means with k={best_k} on {best_kmeans['data_type']} features")

print()

# ============================================================================
# CLUSTER PROFILING
# ============================================================================
print("Generating cluster profiles...")
print("-" * 80)

df['cluster'] = selected_labels
cluster_sizes = df['cluster'].value_counts().sort_index()

print("Cluster Sizes:")
for cluster_id, size in cluster_sizes.items():
    if cluster_id == -1:
        print(f"  Noise: {size:,} ({size/len(df):.1%})")
    else:
        print(f"  Cluster {cluster_id}: {size:,} ({size/len(df):.1%})")
print()

# Improved cluster naming with more granular analysis
cluster_profiles = df.groupby('cluster')[feature_cols].mean()
cluster_names = {}
cluster_descriptions = {}

for cluster_id in cluster_profiles.index:
    if cluster_id == -1:
        cluster_names[cluster_id] = "Outliers"
        cluster_descriptions[cluster_id] = "Wallets with unique characteristics"
        continue

    profile = cluster_profiles.loc[cluster_id]

    # Multi-factor naming based on strongest characteristics
    characteristics = []

    # Performance
    if profile['roi_percent'] > df['roi_percent'].quantile(0.75):
        characteristics.append(('High Performers', 'Exceptional returns'))
    elif profile['roi_percent'] < df['roi_percent'].quantile(0.25):
        characteristics.append(('Underperformers', 'Below-average returns'))

    # Activity
    if profile['trade_frequency'] > df['trade_frequency'].quantile(0.75):
        characteristics.append(('Active Traders', 'High trading frequency'))
    elif profile['trade_frequency'] < df['trade_frequency'].quantile(0.25):
        characteristics.append(('Passive Investors', 'Low trading frequency'))

    # Holding behavior
    if profile['avg_holding_period_days'] > df['avg_holding_period_days'].quantile(0.75):
        characteristics.append(('Long-term Holders', 'Extended holding periods'))

    # Concentration
    if profile['portfolio_hhi'] > 0.6:
        characteristics.append(('Concentrated', 'Few tokens'))
    elif profile['narrative_diversity_score'] > df['narrative_diversity_score'].quantile(0.75):
        characteristics.append(('Diversified', 'Multiple narratives'))

    # Narrative focus
    if profile.get('defi_exposure_pct', 0) > 0.5:
        characteristics.append(('DeFi Focus', 'DeFi-heavy portfolio'))
    elif profile.get('ai_exposure_pct', 0) > 0.5:
        characteristics.append(('AI Focus', 'AI-heavy portfolio'))
    elif profile.get('meme_exposure_pct', 0) > 0.5:
        characteristics.append(('Meme Focus', 'Meme-heavy portfolio'))

    # Select top 2 characteristics
    if len(characteristics) >= 2:
        name = f"{characteristics[0][0]} + {characteristics[1][0].split()[0]}"
        desc = f"{characteristics[0][1]}, {characteristics[1][1].lower()}"
    elif len(characteristics) == 1:
        name, desc = characteristics[0]
    else:
        name = f"Cluster {cluster_id}"
        desc = "Mixed strategy"

    cluster_names[cluster_id] = name
    cluster_descriptions[cluster_id] = desc

print("Cluster Names:")
for cluster_id in sorted(cluster_names.keys()):
    size = cluster_sizes.get(cluster_id, 0)
    print(f"  {cluster_id}: {cluster_names[cluster_id]} ({size:,})")
    print(f"      → {cluster_descriptions[cluster_id]}")
print()

df['cluster_name'] = df['cluster'].map(cluster_names)

# ============================================================================
# EXPORT
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

assignments_file = OUTPUT_DIR / f"wallet_cluster_assignments_optimized_{timestamp}.csv"
df[['wallet_address', 'cluster', 'cluster_name']].to_csv(assignments_file, index=False)

profiles_file = OUTPUT_DIR / f"cluster_profiles_optimized_{timestamp}.csv"
cluster_profiles['cluster_name'] = cluster_profiles.index.map(cluster_names)
cluster_profiles['cluster_size'] = cluster_profiles.index.map(cluster_sizes)
cluster_profiles.to_csv(profiles_file)

metadata = {
    'timestamp': timestamp,
    'algorithm': selected_algorithm,
    'n_wallets': int(len(df)),
    'n_features_original': int(len(feature_cols)),
    'n_features_used': int(X_final.shape[1]),
    'n_clusters': int(selected_metrics['n_clusters']),
    'silhouette_score': float(selected_metrics['silhouette']),
    'noise_ratio': float(selected_metrics['noise_ratio']),
    'cluster_names': cluster_names,
    'cluster_descriptions': cluster_descriptions,
    'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.items()},
    'optimization_notes': {
        'pca_components': int(X_pca.shape[1]) if best_hdbscan['config']['data'] == 'pca' or best_kmeans['data_type'] == 'PCA' else None,
        'pca_variance_explained': float(pca.explained_variance_ratio_.sum()) if best_hdbscan['config']['data'] == 'pca' or best_kmeans['data_type'] == 'PCA' else None,
    }
}

metadata_file = OUTPUT_DIR / f"clustering_metadata_optimized_{timestamp}.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

full_output_file = OUTPUT_DIR / f"wallet_features_with_clusters_optimized_{timestamp}.csv"
df.to_csv(full_output_file, index=False)

print("✅ Exported results:")
print(f"  - {assignments_file.name}")
print(f"  - {profiles_file.name}")
print(f"  - {metadata_file.name}")
print(f"  - {full_output_file.name}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("Generating visualizations...")
print("-" * 80)

viz_mask = selected_labels != -1
X_viz = X_final[viz_mask]
labels_viz = selected_labels[viz_mask]

# t-SNE
print("Creating t-SNE projection...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_viz)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_viz,
                     cmap='tab10', alpha=0.6, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Optimized Wallet Clusters - t-SNE ({selected_algorithm})', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
tsne_file = VIZ_DIR / f"tsne_optimized_{timestamp}.png"
plt.savefig(tsne_file, dpi=300, bbox_inches='tight')
plt.close()

# Cluster sizes
plt.figure(figsize=(14, 6))
sizes_plot = cluster_sizes[cluster_sizes.index != -1].sort_values(ascending=False)
names_plot = [cluster_names.get(i, f"Cluster {i}") for i in sizes_plot.index]
bars = plt.bar(range(len(sizes_plot)), sizes_plot.values, color='steelblue', alpha=0.8)
plt.xticks(range(len(sizes_plot)), names_plot, rotation=45, ha='right')
plt.ylabel('Number of Wallets', fontsize=12)
plt.title(f'Optimized Cluster Sizes ({selected_algorithm})', fontsize=16, fontweight='bold')
plt.grid(True, axis='y', alpha=0.3)
for i, (bar, v) in enumerate(zip(bars, sizes_plot.values)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{v:,}\n({v/len(df):.1%})',
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
sizes_file = VIZ_DIR / f"cluster_sizes_optimized_{timestamp}.png"
plt.savefig(sizes_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved: {tsne_file.name}")
print(f"✅ Saved: {sizes_file.name}")
print()

# ============================================================================
# SUCCESS CRITERIA
# ============================================================================
print("=" * 80)
print("SUCCESS CRITERIA VERIFICATION")
print("=" * 80)

criteria = {
    '5-7 clusters': 5 <= selected_metrics['n_clusters'] <= 7,
    'Silhouette ≥ 0.5': selected_metrics['silhouette'] >= 0.5,
    'Silhouette ≥ 0.3 (acceptable)': selected_metrics['silhouette'] >= 0.3,
    'Clusters > 5% each': all(s >= len(df) * 0.05 for c, s in cluster_sizes.items() if c != -1),
    'Noise < 20%': selected_metrics['noise_ratio'] < 0.2,
}

for criterion, passed in criteria.items():
    status = "✅" if passed else "⚠️ "
    print(f"{status} {criterion}")

print()
print(f"Final Silhouette Score: {selected_metrics['silhouette']:.4f}")
print(f"Final Cluster Count: {selected_metrics['n_clusters']}")
print(f"Noise Ratio: {selected_metrics['noise_ratio']:.1%}")
print()
print("=" * 80)
print("✅ STORY 4.3 COMPLETE - Optimized clustering results exported")
print("=" * 80)
