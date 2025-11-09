#!/usr/bin/env python3
"""
Story 4.3: Wallet Clustering Analysis
Execute complete clustering pipeline and generate all outputs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("STORY 4.3: WALLET CLUSTERING ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading cleaned wallet features dataset...")
print("-" * 80)

DATA_DIR = Path("outputs/features")
OUTPUT_DIR = Path("outputs/clustering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Find the most recent cleaned dataset
cleaned_files = list(DATA_DIR.glob("wallet_features_cleaned_*.csv"))
if not cleaned_files:
    raise FileNotFoundError("No cleaned wallet features file found!")

input_file = max(cleaned_files, key=lambda p: p.stat().st_mtime)
print(f"Loading: {input_file.name}")

df = pd.read_csv(input_file)
print(f"✅ Loaded {len(df):,} wallets with {len(df.columns)} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print()

# ============================================================================
# STEP 2: FEATURE SELECTION
# ============================================================================
print("STEP 2: Selecting features for clustering...")
print("-" * 80)

# Exclude non-numeric columns
exclude_cols = ["wallet_address", "activity_segment"]
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Excluded columns: {exclude_cols}")
print(f"Selected {len(feature_cols)} features for clustering:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")
print()

X = df[feature_cols].values
print(f"Feature matrix shape: {X.shape}")
print(f"Data type: {X.dtype}")
print()

# ============================================================================
# STEP 3: FEATURE SCALING
# ============================================================================
print("STEP 3: Standardizing features...")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling statistics:")
print(f"  Mean (should be ~0): {X_scaled.mean():.6f}")
print(f"  Std (should be ~1): {X_scaled.std():.6f}")
print(f"  Min: {X_scaled.min():.2f}")
print(f"  Max: {X_scaled.max():.2f}")
print()

# ============================================================================
# STEP 4: HDBSCAN CLUSTERING
# ============================================================================
print("STEP 4: Running HDBSCAN (Primary Algorithm)...")
print("-" * 80)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
)

hdbscan_labels = clusterer.fit_predict(X_scaled)

# Calculate metrics (excluding noise points labeled -1)
mask = hdbscan_labels != -1
n_clusters_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
n_noise = (hdbscan_labels == -1).sum()

hdbscan_metrics = {}
if n_clusters_hdbscan > 1 and mask.sum() > 0:
    hdbscan_metrics = {
        'n_clusters': int(n_clusters_hdbscan),
        'n_noise': int(n_noise),
        'noise_ratio': float(n_noise / len(hdbscan_labels)),
        'silhouette': float(silhouette_score(X_scaled[mask], hdbscan_labels[mask])),
        'davies_bouldin': float(davies_bouldin_score(X_scaled[mask], hdbscan_labels[mask])),
        'calinski_harabasz': float(calinski_harabasz_score(X_scaled[mask], hdbscan_labels[mask])),
    }
else:
    hdbscan_metrics = {
        'n_clusters': int(n_clusters_hdbscan),
        'n_noise': int(n_noise),
        'noise_ratio': float(n_noise / len(hdbscan_labels)),
        'silhouette': 0.0,
        'davies_bouldin': float('inf'),
        'calinski_harabasz': 0.0,
    }

print("HDBSCAN Results:")
print(f"  Number of clusters: {hdbscan_metrics['n_clusters']}")
print(f"  Noise points: {hdbscan_metrics['n_noise']} ({hdbscan_metrics['noise_ratio']:.1%})")
if hdbscan_metrics['silhouette'] > 0:
    print(f"  Silhouette Score: {hdbscan_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index: {hdbscan_metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz Score: {hdbscan_metrics['calinski_harabasz']:.2f}")
print()

# ============================================================================
# STEP 5: K-MEANS CLUSTERING (VALIDATION)
# ============================================================================
print("STEP 5: Running K-Means with multiple k values...")
print("-" * 80)

k_values = [3, 5, 7, 10]
kmeans_results = {}

for k in k_values:
    print(f"Testing k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    metrics = {
        'n_clusters': k,
        'inertia': float(kmeans.inertia_),
        'silhouette': float(silhouette_score(X_scaled, labels)),
        'davies_bouldin': float(davies_bouldin_score(X_scaled, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X_scaled, labels)),
        'labels': labels,
    }
    kmeans_results[k] = metrics

    print(f"  Silhouette: {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")

print()

# ============================================================================
# STEP 6: ALGORITHM COMPARISON & SELECTION
# ============================================================================
print("STEP 6: Comparing algorithms and selecting best approach...")
print("-" * 80)

# Find best K-Means configuration
best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
best_kmeans = kmeans_results[best_k]

print("K-Means Best Configuration:")
print(f"  k = {best_k}")
print(f"  Silhouette: {best_kmeans['silhouette']:.4f}")
print()

# Compare HDBSCAN vs K-Means
print("Algorithm Comparison:")
print(f"  HDBSCAN Silhouette: {hdbscan_metrics['silhouette']:.4f}")
print(f"  K-Means Silhouette: {best_kmeans['silhouette']:.4f}")
print()

# Selection logic
if hdbscan_metrics['silhouette'] >= 0.5 and hdbscan_metrics['noise_ratio'] < 0.2:
    selected_algorithm = 'HDBSCAN'
    selected_labels = hdbscan_labels
    selected_metrics = hdbscan_metrics
    print("✅ Selected: HDBSCAN (meets success criteria)")
elif best_kmeans['silhouette'] > hdbscan_metrics['silhouette']:
    selected_algorithm = f'K-Means (k={best_k})'
    selected_labels = best_kmeans['labels']
    selected_metrics = best_kmeans
    print(f"✅ Selected: K-Means with k={best_k} (higher silhouette score)")
else:
    selected_algorithm = 'HDBSCAN'
    selected_labels = hdbscan_labels
    selected_metrics = hdbscan_metrics
    print("✅ Selected: HDBSCAN (default choice)")

print()

# ============================================================================
# STEP 7: CLUSTER PROFILING
# ============================================================================
print("STEP 7: Generating cluster profiles...")
print("-" * 80)

df['cluster'] = selected_labels

# Calculate mean features per cluster
cluster_profiles = df.groupby('cluster')[feature_cols].mean()
cluster_sizes = df['cluster'].value_counts().sort_index()

print("Cluster Sizes:")
for cluster_id, size in cluster_sizes.items():
    if cluster_id == -1:
        print(f"  Noise: {size:,} wallets ({size/len(df):.1%})")
    else:
        print(f"  Cluster {cluster_id}: {size:,} wallets ({size/len(df):.1%})")
print()

# ============================================================================
# STEP 8: CLUSTER NAMING
# ============================================================================
print("STEP 8: Assigning interpretable cluster names...")
print("-" * 80)

cluster_names = {}
cluster_descriptions = {}

for cluster_id in cluster_profiles.index:
    if cluster_id == -1:
        cluster_names[cluster_id] = "Noise/Outliers"
        cluster_descriptions[cluster_id] = "Wallets that don't fit well into any cluster"
        continue

    profile = cluster_profiles.loc[cluster_id]

    # Naming heuristics based on feature values
    if profile.get('total_roi', 0) > 2.0:
        name = "High Performers"
        desc = "Wallets with exceptional returns"
    elif profile.get('total_roi', 0) < -0.5:
        name = "Struggling Traders"
        desc = "Wallets with consistent losses"
    elif profile.get('trade_frequency', 0) > df['trade_frequency'].quantile(0.75):
        name = "Active Traders"
        desc = "High-frequency trading behavior"
    elif profile.get('avg_holding_period_days', 0) > df['avg_holding_period_days'].quantile(0.75):
        name = "Long-term Holders"
        desc = "Diamond hands with extended holding periods"
    elif profile.get('portfolio_hhi', 0) > 0.5:
        name = "Concentrated Investors"
        desc = "Focused portfolios with few tokens"
    elif profile.get('narrative_diversity', 0) > df['narrative_diversity'].quantile(0.75):
        name = "Diversified Explorers"
        desc = "Exposure across multiple narratives"
    elif profile.get('is_defi_focused', 0) > 0.5:
        name = "DeFi Specialists"
        desc = "Primarily trading DeFi tokens"
    elif profile.get('is_ai_focused', 0) > 0.5:
        name = "AI/ML Enthusiasts"
        desc = "Concentrated in AI narrative"
    elif profile.get('is_meme_focused', 0) > 0.5:
        name = "Meme Traders"
        desc = "Active in meme token markets"
    else:
        name = f"Cluster {cluster_id}"
        desc = "Mixed strategy traders"

    cluster_names[cluster_id] = name
    cluster_descriptions[cluster_id] = desc

print("Cluster Names:")
for cluster_id in sorted(cluster_names.keys()):
    size = cluster_sizes.get(cluster_id, 0)
    print(f"  {cluster_id}: {cluster_names[cluster_id]} ({size:,} wallets)")
    print(f"      → {cluster_descriptions[cluster_id]}")
print()

df['cluster_name'] = df['cluster'].map(cluster_names)

# ============================================================================
# STEP 9: EXPORT RESULTS
# ============================================================================
print("STEP 9: Exporting clustering results...")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Export 1: Wallet assignments
assignments_file = OUTPUT_DIR / f"wallet_cluster_assignments_{timestamp}.csv"
df[['wallet_address', 'cluster', 'cluster_name']].to_csv(assignments_file, index=False)
print(f"✅ Saved: {assignments_file.name}")

# Export 2: Cluster profiles
profiles_file = OUTPUT_DIR / f"cluster_profiles_{timestamp}.csv"
cluster_profiles['cluster_name'] = cluster_profiles.index.map(cluster_names)
cluster_profiles['cluster_size'] = cluster_profiles.index.map(cluster_sizes)
cluster_profiles.to_csv(profiles_file)
print(f"✅ Saved: {profiles_file.name}")

# Export 3: Metrics and metadata
metadata = {
    'timestamp': timestamp,
    'algorithm': selected_algorithm,
    'n_wallets': int(len(df)),
    'n_features': int(len(feature_cols)),
    'n_clusters': int(selected_metrics['n_clusters']),
    'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in selected_metrics.items() if k != 'labels'},
    'cluster_names': cluster_names,
    'cluster_descriptions': cluster_descriptions,
    'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.items()},
}

metadata_file = OUTPUT_DIR / f"clustering_metadata_{timestamp}.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Saved: {metadata_file.name}")

# Export 4: Full dataset with clusters
full_output_file = OUTPUT_DIR / f"wallet_features_with_clusters_{timestamp}.csv"
df.to_csv(full_output_file, index=False)
print(f"✅ Saved: {full_output_file.name}")

print()

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================
print("STEP 10: Generating visualizations...")
print("-" * 80)

# Filter out noise for visualization
viz_mask = selected_labels != -1
X_viz = X_scaled[viz_mask]
labels_viz = selected_labels[viz_mask]

# Visualization 1: t-SNE projection
print("Generating t-SNE projection...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_viz)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_viz,
                     cmap='tab10', alpha=0.6, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Wallet Clusters - t-SNE Projection ({selected_algorithm})', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
tsne_file = VIZ_DIR / f"tsne_projection_{timestamp}.png"
plt.savefig(tsne_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {tsne_file.name}")

# Visualization 2: Cluster sizes
plt.figure(figsize=(12, 6))
sizes_plot = cluster_sizes[cluster_sizes.index != -1].sort_values(ascending=False)
names_plot = [cluster_names.get(i, f"Cluster {i}") for i in sizes_plot.index]
plt.bar(range(len(sizes_plot)), sizes_plot.values, color='steelblue', alpha=0.8)
plt.xticks(range(len(sizes_plot)), names_plot, rotation=45, ha='right')
plt.ylabel('Number of Wallets', fontsize=12)
plt.title(f'Cluster Size Distribution ({selected_algorithm})', fontsize=16, fontweight='bold')
plt.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(sizes_plot.values):
    plt.text(i, v + 20, f'{v:,}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
sizes_file = VIZ_DIR / f"cluster_sizes_{timestamp}.png"
plt.savefig(sizes_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {sizes_file.name}")

# Visualization 3: Silhouette plot (if applicable)
if selected_metrics['silhouette'] > 0:
    from sklearn.metrics import silhouette_samples

    silhouette_vals = silhouette_samples(X_viz, labels_viz)

    plt.figure(figsize=(12, 8))
    y_lower = 10
    for i in sorted(set(labels_viz)):
        cluster_silhouette_vals = silhouette_vals[labels_viz == i]
        cluster_silhouette_vals.sort()

        size = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size

        color = plt.cm.tab10(i / max(labels_viz))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7,
                         label=cluster_names.get(i, f"Cluster {i}"))

        plt.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=selected_metrics['silhouette'], color="red", linestyle="--",
                label=f"Mean Silhouette: {selected_metrics['silhouette']:.3f}")
    plt.xlabel("Silhouette Coefficient", fontsize=12)
    plt.ylabel("Cluster", fontsize=12)
    plt.title(f"Silhouette Analysis ({selected_algorithm})", fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    silhouette_file = VIZ_DIR / f"silhouette_analysis_{timestamp}.png"
    plt.savefig(silhouette_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {silhouette_file.name}")

print()

# ============================================================================
# STEP 11: SUCCESS CRITERIA VERIFICATION
# ============================================================================
print("STEP 11: Verifying success criteria...")
print("-" * 80)

criteria = {
    '5-7 clusters identified': 5 <= selected_metrics['n_clusters'] <= 7,
    'Silhouette score ≥ 0.5': selected_metrics.get('silhouette', 0) >= 0.5,
    'All clusters > 5% of dataset': all(s >= len(df) * 0.05 for s in cluster_sizes.values if s != -1),
    'Noise ratio < 20%': selected_metrics.get('noise_ratio', 0) < 0.2,
}

print("Success Criteria Check:")
for criterion, passed in criteria.items():
    status = "✅ PASS" if passed else "⚠️  FAIL"
    print(f"  {status} - {criterion}")

overall_success = sum(criteria.values()) >= 3  # At least 3 of 4 criteria
print()
print(f"Overall Assessment: {'✅ SUCCESS' if overall_success else '⚠️  PARTIAL SUCCESS'}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("STORY 4.3: CLUSTERING ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Algorithm: {selected_algorithm}")
print(f"Clusters: {selected_metrics['n_clusters']}")
print(f"Silhouette Score: {selected_metrics.get('silhouette', 0):.4f}")
print()
print("Outputs generated:")
print(f"  1. {assignments_file.name}")
print(f"  2. {profiles_file.name}")
print(f"  3. {metadata_file.name}")
print(f"  4. {full_output_file.name}")
print(f"  5. {tsne_file.name}")
print(f"  6. {sizes_file.name}")
if selected_metrics.get('silhouette', 0) > 0:
    print(f"  7. {silhouette_file.name}")
print()
print("✅ Ready for Story 4.4 (Cluster Interpretation & Documentation)")
print("=" * 80)
