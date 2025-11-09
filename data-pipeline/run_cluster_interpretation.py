#!/usr/bin/env python3
"""
Story 4.4: Cluster Interpretation & Documentation
Comprehensive analysis of clustering results with validation and insights
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STORY 4.4: CLUSTER INTERPRETATION & DOCUMENTATION")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD CLUSTERING RESULTS
# ============================================================================
print("STEP 1: Loading clustering results...")
print("-" * 80)

CLUSTERING_DIR = Path("outputs/clustering")
OUTPUT_DIR = Path("outputs/cluster_interpretation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load HDBSCAN optimized (primary)
hdbscan_files = list(CLUSTERING_DIR.glob("wallet_features_with_clusters_optimized_*.csv"))
if not hdbscan_files:
    raise FileNotFoundError("HDBSCAN optimized results not found!")
hdbscan_file = max(hdbscan_files, key=lambda p: p.stat().st_mtime)
df_hdbscan = pd.read_csv(hdbscan_file)

# Load K-Means k=5 (validation)
kmeans_files = list(CLUSTERING_DIR.glob("wallet_features_with_clusters_final_*.csv"))
if not kmeans_files:
    raise FileNotFoundError("K-Means final results not found!")
kmeans_file = max(kmeans_files, key=lambda p: p.stat().st_mtime)
df_kmeans = pd.read_csv(kmeans_file)

print(f"✅ HDBSCAN Optimized: {len(df_hdbscan):,} wallets from {hdbscan_file.name}")
print(f"✅ K-Means (k=5): {len(df_kmeans):,} wallets from {kmeans_file.name}")
print()

# Verify same wallet addresses
assert (df_hdbscan['wallet_address'] == df_kmeans['wallet_address']).all(), "Wallet mismatch!"

# ============================================================================
# STEP 2: FEATURE VALUE VALIDATION
# ============================================================================
print("STEP 2: Validating feature value ranges...")
print("-" * 80)

features_to_check = {
    'portfolio_hhi': (0, 1, 'Herfindahl-Hirschman Index'),
    'portfolio_gini': (0, 1, 'Gini coefficient'),
    'win_rate': (0, 100, 'Win rate percentage'),
    'defi_exposure_pct': (0, 100, 'DeFi exposure percentage'),
    'ai_exposure_pct': (0, 100, 'AI exposure percentage'),
    'meme_exposure_pct': (0, 100, 'Meme exposure percentage'),
    'weekend_activity_ratio': (0, 1, 'Weekend activity ratio'),
    'night_trading_ratio': (0, 1, 'Night trading ratio'),
    'stablecoin_usage_ratio': (0, 1, 'Stablecoin usage ratio'),
}

validation_issues = []

for feature, (min_val, max_val, description) in features_to_check.items():
    if feature not in df_hdbscan.columns:
        validation_issues.append(f"⚠️  {feature} not found in dataset")
        continue

    actual_min = df_hdbscan[feature].min()
    actual_max = df_hdbscan[feature].max()

    issues = []
    if actual_min < min_val:
        issues.append(f"min={actual_min:.2f} < expected {min_val}")
    if actual_max > max_val:
        issues.append(f"max={actual_max:.2f} > expected {max_val}")

    if issues:
        validation_issues.append(f"⚠️  {feature} ({description}): {', '.join(issues)}")
        print(f"⚠️  {feature}: Range [{actual_min:.2f}, {actual_max:.2f}] "
              f"(expected [{min_val}, {max_val}])")
    else:
        print(f"✅ {feature}: [{actual_min:.2f}, {actual_max:.2f}]")

print()

if validation_issues:
    print(f"Found {len(validation_issues)} validation issues:")
    for issue in validation_issues:
        print(f"  {issue}")
    print()
    print("NOTE: These anomalies will be documented but won't block interpretation.")
    print("      They indicate potential feature engineering refinements for future work.")
    print()

# ============================================================================
# STEP 3: DETAILED CLUSTER PROFILES
# ============================================================================
print("STEP 3: Generating detailed cluster profiles...")
print("-" * 80)

# Rename cluster column for clarity
df_hdbscan = df_hdbscan.rename(columns={'cluster': 'hdbscan_cluster', 'cluster_name': 'hdbscan_cluster_name'})
df_kmeans = df_kmeans.rename(columns={'cluster': 'kmeans_cluster', 'cluster_name': 'kmeans_cluster_name'})

# Merge for comparison
df = df_hdbscan.copy()
df['kmeans_cluster'] = df_kmeans['kmeans_cluster']
df['kmeans_cluster_name'] = df_kmeans['kmeans_cluster_name']

# Exclude non-feature columns
exclude_cols = ['wallet_address', 'activity_segment', 'hdbscan_cluster',
                'hdbscan_cluster_name', 'kmeans_cluster', 'kmeans_cluster_name']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Analyzing {len(feature_cols)} features across clusters")
print()

# HDBSCAN cluster profiles
hdbscan_clusters = sorted(df['hdbscan_cluster'].unique())
print(f"HDBSCAN Clusters: {len(hdbscan_clusters)} (including noise)")

cluster_profiles_hdbscan = {}
for cluster_id in hdbscan_clusters:
    cluster_data = df[df['hdbscan_cluster'] == cluster_id]

    profile = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_data),
        'percentage': len(cluster_data) / len(df) * 100,

        # Performance metrics
        'roi_mean': cluster_data['roi_percent'].mean(),
        'roi_median': cluster_data['roi_percent'].median(),
        'roi_std': cluster_data['roi_percent'].std(),
        'win_rate_mean': cluster_data['win_rate'].mean(),
        'sharpe_mean': cluster_data['sharpe_ratio'].mean(),
        'pnl_mean': cluster_data['total_pnl_usd'].mean(),
        'pnl_median': cluster_data['total_pnl_usd'].median(),

        # Activity metrics
        'trade_freq_mean': cluster_data['trade_frequency'].mean(),
        'trade_freq_median': cluster_data['trade_frequency'].median(),
        'holding_days_mean': cluster_data['avg_holding_period_days'].mean(),
        'holding_days_median': cluster_data['avg_holding_period_days'].median(),
        'weekend_ratio': cluster_data['weekend_activity_ratio'].mean(),
        'night_ratio': cluster_data['night_trading_ratio'].mean(),

        # Portfolio metrics
        'hhi_mean': cluster_data['portfolio_hhi'].mean(),
        'gini_mean': cluster_data['portfolio_gini'].mean(),
        'num_tokens_mean': cluster_data['num_tokens_avg'].mean(),
        'narrative_diversity_mean': cluster_data['narrative_diversity_score'].mean(),

        # Narrative exposure
        'defi_exposure': cluster_data['defi_exposure_pct'].mean(),
        'ai_exposure': cluster_data['ai_exposure_pct'].mean(),
        'meme_exposure': cluster_data['meme_exposure_pct'].mean(),
        'stablecoin_ratio': cluster_data['stablecoin_usage_ratio'].mean(),

        # Behavior flags
        'pct_profitable': (cluster_data['is_profitable'] == 1).mean() * 100,
        'pct_active': (cluster_data['is_active'] == 1).mean() * 100,
        'pct_multi_token': (cluster_data['is_multi_token'] == 1).mean() * 100,
    }

    cluster_profiles_hdbscan[cluster_id] = profile

print(f"✅ Generated {len(cluster_profiles_hdbscan)} HDBSCAN cluster profiles")
print()

# ============================================================================
# STEP 4: IDENTIFY REPRESENTATIVE WALLETS
# ============================================================================
print("STEP 4: Identifying representative wallets per cluster...")
print("-" * 80)

from sklearn.metrics.pairwise import euclidean_distances

representative_wallets = {}

for cluster_id in hdbscan_clusters:
    if cluster_id == -1:  # Skip noise for now
        continue

    cluster_mask = df['hdbscan_cluster'] == cluster_id
    cluster_data = df[cluster_mask]

    if len(cluster_data) < 3:
        # For very small clusters, just pick the first wallet
        rep_wallet = cluster_data.iloc[0]['wallet_address']
        representative_wallets[cluster_id] = {
            'centroid': rep_wallet,
            'top_performers': [],
            'typical': []
        }
        continue

    # Calculate cluster centroid (mean of features)
    cluster_features = cluster_data[feature_cols].values
    centroid = cluster_features.mean(axis=0)

    # Find wallet closest to centroid
    distances = euclidean_distances(cluster_features, centroid.reshape(1, -1))
    centroid_idx = distances.argmin()
    centroid_wallet = cluster_data.iloc[centroid_idx]['wallet_address']

    # Find top 3 performers by ROI
    top_performers = cluster_data.nlargest(min(3, len(cluster_data)), 'roi_percent')['wallet_address'].tolist()

    # Find 3 typical wallets (closest to median ROI)
    median_roi = cluster_data['roi_percent'].median()
    cluster_data_sorted = cluster_data.copy()
    cluster_data_sorted['roi_distance'] = (cluster_data_sorted['roi_percent'] - median_roi).abs()
    typical_wallets = cluster_data_sorted.nsmallest(min(3, len(cluster_data)), 'roi_distance')['wallet_address'].tolist()

    representative_wallets[cluster_id] = {
        'centroid': centroid_wallet,
        'top_performers': top_performers,
        'typical': typical_wallets,
    }

    print(f"Cluster {cluster_id}: Centroid={centroid_wallet[:10]}..., "
          f"Top performers={len(top_performers)}, Typical={len(typical_wallets)}")

print()
print(f"✅ Identified representative wallets for {len(representative_wallets)} clusters")
print()

# ============================================================================
# STEP 5: CLUSTER PERSONAS
# ============================================================================
print("STEP 5: Creating cluster personas with rich narratives...")
print("-" * 80)

def create_persona(cluster_id, profile, rep_wallets):
    """Generate rich cluster persona based on statistics."""

    if cluster_id == -1:
        return {
            'name': 'Unique Strategists (Noise)',
            'archetype': 'Outliers',
            'tagline': 'Wallets with unique, non-conforming strategies',
            'description': (
                f"This group contains {profile['size']:,} wallets ({profile['percentage']:.1f}%) "
                "that don't fit well into any standard cluster pattern. These wallets employ "
                "unique or hybrid strategies that defy categorization. In crypto markets, "
                "where innovation is rewarded, these outliers may represent the most adaptive traders."
            ),
            'characteristics': [
                'Highly diverse trading patterns',
                'Don\'t conform to typical wallet behavior',
                'May represent innovative or experimental strategies',
                'Could include both exceptional performers and unique failures',
            ],
            'investment_style': 'Non-standard, experimental',
            'risk_profile': 'Variable',
            'recommendation': 'Study individually for unique insights',
        }

    # Determine archetype based on key metrics
    roi = profile['roi_mean']
    trade_freq = profile['trade_freq_mean']
    holding_days = profile['holding_days_mean']
    hhi = profile['hhi_mean']
    profitable_pct = profile['pct_profitable']

    # Performance-based classification
    if roi > 100:
        performance_tier = 'Elite'
    elif roi > 50:
        performance_tier = 'High'
    elif roi > 0:
        performance_tier = 'Moderate'
    else:
        performance_tier = 'Struggling'

    # Activity-based classification
    if trade_freq > 10:
        activity_level = 'Hyperactive'
    elif trade_freq > 5:
        activity_level = 'Active'
    elif trade_freq > 2:
        activity_level = 'Moderate'
    else:
        activity_level = 'Passive'

    # Concentration classification
    if hhi > 0.6:
        concentration = 'Highly Concentrated'
    elif hhi > 0.4:
        concentration = 'Moderately Concentrated'
    else:
        concentration = 'Diversified'

    # Create persona name
    if performance_tier == 'Elite' and activity_level in ['Active', 'Hyperactive']:
        name = f"Elite Active Traders"
        archetype = "High-Frequency Winners"
    elif performance_tier == 'Elite' and holding_days > 30:
        name = f"Strategic Long-term Winners"
        archetype = "Patient Accumulators"
    elif performance_tier == 'Elite':
        name = f"Elite Performers"
        archetype = "Exceptional Traders"
    elif activity_level == 'Hyperactive':
        name = f"Hyperactive Traders"
        archetype = "High-Frequency Operators"
    elif holding_days > 30:
        name = f"Long-term Holders"
        archetype = "Diamond Hands"
    elif concentration == 'Highly Concentrated':
        name = f"Focused Specialists"
        archetype = "Concentrated Portfolios"
    else:
        name = f"{performance_tier} {activity_level} Traders"
        archetype = f"{concentration} Investors"

    # Generate tagline
    tagline = f"{performance_tier} performers with {activity_level.lower()} trading style"

    # Generate rich description
    description = (
        f"This cluster contains {profile['size']:,} wallets ({profile['percentage']:.1f}%) "
        f"characterized by {performance_tier.lower()} performance metrics "
        f"(average ROI: {roi:.1f}%, {profitable_pct:.0f}% profitable). "
    )

    if activity_level in ['Hyperactive', 'Active']:
        description += (
            f"These wallets trade frequently (avg {trade_freq:.1f} trades) "
            f"with short holding periods (avg {holding_days:.0f} days). "
        )
    elif activity_level == 'Passive':
        description += (
            f"These wallets trade infrequently (avg {trade_freq:.1f} trades) "
            f"with extended holding periods (avg {holding_days:.0f} days). "
        )

    if concentration == 'Highly Concentrated':
        description += (
            f"Portfolios are highly concentrated (HHI: {hhi:.2f}), "
            f"focusing on few tokens (avg {profile['num_tokens_mean']:.1f}). "
        )
    elif concentration == 'Diversified':
        description += (
            f"Portfolios are well-diversified across multiple tokens "
            f"(avg {profile['num_tokens_mean']:.1f} tokens). "
        )

    # Narrative exposure
    narratives = []
    if profile['defi_exposure'] > 40:
        narratives.append(f"DeFi ({profile['defi_exposure']:.0f}%)")
    if profile['ai_exposure'] > 40:
        narratives.append(f"AI ({profile['ai_exposure']:.0f}%)")
    if profile['meme_exposure'] > 40:
        narratives.append(f"Meme ({profile['meme_exposure']:.0f}%)")

    if narratives:
        description += f"Strong exposure to: {', '.join(narratives)}. "

    # Characteristics
    characteristics = []
    characteristics.append(f"Average ROI: {roi:.1f}%")
    characteristics.append(f"Win rate: {profile['win_rate_mean']:.1f}%")
    characteristics.append(f"Sharpe ratio: {profile['sharpe_mean']:.2f}")
    characteristics.append(f"Trade frequency: {trade_freq:.1f} trades")
    characteristics.append(f"Holding period: {holding_days:.0f} days")
    characteristics.append(f"Portfolio concentration (HHI): {hhi:.2f}")
    characteristics.append(f"{profitable_pct:.0f}% are profitable")

    if profile['weekend_ratio'] > 0.3:
        characteristics.append(f"High weekend activity ({profile['weekend_ratio']:.1%})")
    if profile['night_ratio'] > 0.3:
        characteristics.append(f"Significant night trading ({profile['night_ratio']:.1%})")

    # Investment style
    if activity_level in ['Hyperactive', 'Active']:
        investment_style = "Active trading with frequent position changes"
    elif holding_days > 30:
        investment_style = "Buy-and-hold with long-term conviction"
    else:
        investment_style = "Balanced approach with selective entries/exits"

    # Risk profile
    if profile['sharpe_mean'] > 2:
        risk_profile = "High risk-adjusted returns (Sharpe > 2)"
    elif profile['sharpe_mean'] > 1:
        risk_profile = "Moderate risk-adjusted returns"
    else:
        risk_profile = "Lower risk-adjusted performance"

    # Recommendations
    if performance_tier == 'Elite':
        recommendation = "Study strategies for replication; identify alpha sources"
    elif performance_tier == 'Struggling':
        recommendation = "Avoid mimicking; analyze failure modes for risk management"
    elif activity_level == 'Hyperactive':
        recommendation = "Monitor for market-moving activities; potential frontrunning targets"
    else:
        recommendation = "Baseline behavior; useful for comparative analysis"

    return {
        'name': name,
        'archetype': archetype,
        'tagline': tagline,
        'description': description,
        'characteristics': characteristics,
        'investment_style': investment_style,
        'risk_profile': risk_profile,
        'recommendation': recommendation,
        'representative_wallets': rep_wallets,
    }

personas = {}
for cluster_id, profile in cluster_profiles_hdbscan.items():
    rep_wallets = representative_wallets.get(cluster_id, {})
    persona = create_persona(cluster_id, profile, rep_wallets)
    personas[cluster_id] = persona

    print(f"\nCluster {cluster_id}: {persona['name']}")
    print(f"  {persona['tagline']}")
    print(f"  Size: {profile['size']:,} wallets ({profile['percentage']:.1f}%)")

print()
print(f"✅ Created {len(personas)} detailed cluster personas")
print()

# ============================================================================
# STEP 6: COMPARE HDBSCAN VS K-MEANS
# ============================================================================
print("STEP 6: Comparing HDBSCAN vs K-Means cluster mappings...")
print("-" * 80)

# Create cross-tabulation
cross_tab = pd.crosstab(df['hdbscan_cluster'], df['kmeans_cluster'],
                        margins=True, margins_name='Total')

print("HDBSCAN vs K-Means Cluster Mapping:")
print(cross_tab)
print()

# Calculate overlap metrics
overlap_analysis = []
for hdb_cluster in hdbscan_clusters:
    if hdb_cluster == -1:
        continue

    hdb_wallets = df[df['hdbscan_cluster'] == hdb_cluster]
    if len(hdb_wallets) == 0:
        continue

    # Find which K-Means cluster has most overlap
    kmeans_distribution = hdb_wallets['kmeans_cluster'].value_counts()
    dominant_kmeans = kmeans_distribution.index[0]
    overlap_count = kmeans_distribution.iloc[0]
    overlap_pct = (overlap_count / len(hdb_wallets)) * 100

    overlap_analysis.append({
        'hdbscan_cluster': int(hdb_cluster),
        'size': len(hdb_wallets),
        'dominant_kmeans_cluster': int(dominant_kmeans),
        'overlap_count': int(overlap_count),
        'overlap_percentage': float(overlap_pct),
        'fragmentation': len(kmeans_distribution),
    })

overlap_df = pd.DataFrame(overlap_analysis)
print("Cluster Overlap Analysis:")
print(overlap_df.to_string(index=False))
print()

# ============================================================================
# STEP 7: ANALYZE NOISE CLUSTER
# ============================================================================
print("STEP 7: Deep dive into 'noise' cluster (unique strategists)...")
print("-" * 80)

noise_wallets = df[df['hdbscan_cluster'] == -1]
print(f"Noise cluster size: {len(noise_wallets):,} ({len(noise_wallets)/len(df)*100:.1f}%)")
print()

if len(noise_wallets) > 0:
    # Characterize noise wallets
    noise_stats = {
        'roi_mean': noise_wallets['roi_percent'].mean(),
        'roi_median': noise_wallets['roi_percent'].median(),
        'roi_std': noise_wallets['roi_percent'].std(),
        'pct_profitable': (noise_wallets['is_profitable'] == 1).mean() * 100,
        'pct_high_roi': (noise_wallets['roi_percent'] > 100).mean() * 100,
        'pct_negative_roi': (noise_wallets['roi_percent'] < 0).mean() * 100,
        'trade_freq_mean': noise_wallets['trade_frequency'].mean(),
        'num_tokens_mean': noise_wallets['num_tokens_avg'].mean(),
    }

    print("Noise Cluster Characteristics:")
    print(f"  Average ROI: {noise_stats['roi_mean']:.1f}% (median: {noise_stats['roi_median']:.1f}%)")
    print(f"  Std deviation: {noise_stats['roi_std']:.1f}% (high variance)")
    print(f"  Profitable: {noise_stats['pct_profitable']:.0f}%")
    print(f"  Exceptional (ROI > 100%): {noise_stats['pct_high_roi']:.0f}%")
    print(f"  Negative ROI: {noise_stats['pct_negative_roi']:.0f}%")
    print(f"  Avg trades: {noise_stats['trade_freq_mean']:.1f}")
    print(f"  Avg tokens: {noise_stats['num_tokens_mean']:.1f}")
    print()

    # Find extreme outliers in noise
    top_noise_performers = noise_wallets.nlargest(10, 'roi_percent')[['wallet_address', 'roi_percent', 'trade_frequency']]
    worst_noise_performers = noise_wallets.nsmallest(10, 'roi_percent')[['wallet_address', 'roi_percent', 'trade_frequency']]

    print("Top 10 Noise Performers:")
    print(top_noise_performers.to_string(index=False))
    print()

    print("Worst 10 Noise Performers:")
    print(worst_noise_performers.to_string(index=False))
    print()

# ============================================================================
# STEP 8: GENERATE ACTIONABLE INSIGHTS
# ============================================================================
print("STEP 8: Generating actionable insights per cluster...")
print("-" * 80)

insights = {}

for cluster_id, profile in cluster_profiles_hdbscan.items():
    persona = personas[cluster_id]

    cluster_insights = {
        'cluster_id': int(cluster_id),
        'cluster_name': persona['name'],
        'size': profile['size'],
        'percentage': profile['percentage'],

        'key_insights': [],
        'trading_implications': [],
        'research_questions': [],
        'data_opportunities': [],
    }

    # Generate insights based on characteristics
    roi = profile['roi_mean']
    profitable_pct = profile['pct_profitable']
    trade_freq = profile['trade_freq_mean']

    # Key insights
    if cluster_id == -1:
        cluster_insights['key_insights'].append(
            f"{profile['percentage']:.1f}% of wallets defy standard categorization"
        )
        cluster_insights['key_insights'].append(
            "High variance suggests diverse experimental strategies"
        )
        cluster_insights['trading_implications'].append(
            "These wallets may identify emerging trends before they become mainstream"
        )
    elif roi > 100:
        cluster_insights['key_insights'].append(
            f"Exceptional returns ({roi:.0f}% ROI) significantly outperform market"
        )
        cluster_insights['trading_implications'].append(
            "Study token selection and entry/exit timing for alpha signals"
        )
        cluster_insights['research_questions'].append(
            "What tokens or narratives drove exceptional performance?"
        )
    elif roi < 0:
        cluster_insights['key_insights'].append(
            f"Consistent underperformance ({roi:.0f}% ROI) despite market opportunities"
        )
        cluster_insights['trading_implications'].append(
            "Analyze failure patterns to avoid similar mistakes"
        )
        cluster_insights['research_questions'].append(
            "What common mistakes or risk factors led to losses?"
        )

    if trade_freq > 10:
        cluster_insights['key_insights'].append(
            f"High trading frequency ({trade_freq:.0f} trades) suggests active management"
        )
        cluster_insights['trading_implications'].append(
            "Monitor for market-making activity or arbitrage strategies"
        )

    if profile['hhi_mean'] > 0.7:
        cluster_insights['key_insights'].append(
            "Highly concentrated portfolios indicate conviction-based investing"
        )
        cluster_insights['research_questions'].append(
            "What drives concentrated allocation decisions?"
        )

    # Data opportunities
    rep_wallet = representative_wallets.get(cluster_id, {}).get('centroid', 'N/A')
    if rep_wallet != 'N/A':
        cluster_insights['data_opportunities'].append(
            f"Deep dive into {rep_wallet[:10]}... (centroid wallet)"
        )
    else:
        cluster_insights['data_opportunities'].append(
            "Identify representative wallet for case study"
        )
    cluster_insights['data_opportunities'].append(
        "Analyze token overlap within cluster for narrative trends"
    )
    cluster_insights['data_opportunities'].append(
        "Track cluster migration over time for strategy evolution"
    )

    insights[cluster_id] = cluster_insights

    print(f"\nCluster {cluster_id}: {persona['name']}")
    print(f"  Key Insights: {len(cluster_insights['key_insights'])}")
    print(f"  Trading Implications: {len(cluster_insights['trading_implications'])}")
    print(f"  Research Questions: {len(cluster_insights['research_questions'])}")

print()
print(f"✅ Generated actionable insights for {len(insights)} clusters")
print()

# ============================================================================
# STEP 9: EXPORT COMPREHENSIVE DOCUMENTATION
# ============================================================================
print("STEP 9: Exporting comprehensive documentation...")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Export 1: Cluster profiles
profiles_export = []
for cluster_id, profile in cluster_profiles_hdbscan.items():
    profile_export = profile.copy()
    profile_export['cluster_name'] = personas[cluster_id]['name']
    profile_export['archetype'] = personas[cluster_id]['archetype']
    profiles_export.append(profile_export)

profiles_df = pd.DataFrame(profiles_export)
profiles_file = OUTPUT_DIR / f"cluster_profiles_detailed_{timestamp}.csv"
profiles_df.to_csv(profiles_file, index=False)
print(f"✅ {profiles_file.name}")

# Export 2: Cluster personas
personas_file = OUTPUT_DIR / f"cluster_personas_{timestamp}.json"
with open(personas_file, 'w') as f:
    # Convert to serializable format
    personas_export = {}
    for k, v in personas.items():
        personas_export[str(k)] = {
            key: (value if not isinstance(value, (np.integer, np.floating)) else float(value))
            for key, value in v.items()
        }
    json.dump(personas_export, f, indent=2)
print(f"✅ {personas_file.name}")

# Export 3: Actionable insights
insights_file = OUTPUT_DIR / f"cluster_insights_{timestamp}.json"
with open(insights_file, 'w') as f:
    insights_export = {}
    for k, v in insights.items():
        insights_export[str(k)] = v
    json.dump(insights_export, f, indent=2)
print(f"✅ {insights_file.name}")

# Export 4: Representative wallets
rep_wallets_file = OUTPUT_DIR / f"representative_wallets_{timestamp}.json"
with open(rep_wallets_file, 'w') as f:
    rep_export = {}
    for k, v in representative_wallets.items():
        rep_export[str(k)] = v
    json.dump(rep_export, f, indent=2)
print(f"✅ {rep_wallets_file.name}")

# Export 5: Cluster comparison
comparison_file = OUTPUT_DIR / f"hdbscan_kmeans_comparison_{timestamp}.csv"
cross_tab.to_csv(comparison_file)
print(f"✅ {comparison_file.name}")

overlap_file = OUTPUT_DIR / f"cluster_overlap_analysis_{timestamp}.csv"
overlap_df.to_csv(overlap_file, index=False)
print(f"✅ {overlap_file.name}")

# Export 6: Validation report
validation_file = OUTPUT_DIR / f"feature_validation_report_{timestamp}.txt"
with open(validation_file, 'w') as f:
    f.write("FEATURE VALIDATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    if validation_issues:
        f.write(f"Found {len(validation_issues)} validation issues:\n\n")
        for issue in validation_issues:
            f.write(f"{issue}\n")
        f.write("\nRecommendation: Review feature engineering logic for affected features.\n")
    else:
        f.write("✅ All features validated successfully\n")
        f.write("No range anomalies detected.\n")

print(f"✅ {validation_file.name}")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("STORY 4.4: CLUSTER INTERPRETATION COMPLETE")
print("=" * 80)
print()
print(f"✅ Analyzed {len(hdbscan_clusters)} HDBSCAN clusters")
print(f"✅ Created {len(personas)} detailed personas")
print(f"✅ Identified representative wallets for {len(representative_wallets)} clusters")
print(f"✅ Generated actionable insights")
print(f"✅ Compared HDBSCAN vs K-Means mappings")
print(f"✅ Deep-dived into noise cluster ({len(noise_wallets):,} wallets)")
print()

if validation_issues:
    print(f"⚠️  {len(validation_issues)} feature validation issues detected")
    print("   See feature_validation_report for details")
    print()

print("Outputs:")
print(f"  • {profiles_file.name}")
print(f"  • {personas_file.name}")
print(f"  • {insights_file.name}")
print(f"  • {rep_wallets_file.name}")
print(f"  • {comparison_file.name}")
print(f"  • {overlap_file.name}")
print(f"  • {validation_file.name}")
print()
print("=" * 80)
print("✅ READY FOR FINAL DOCUMENTATION & PRESENTATION")
print("=" * 80)
