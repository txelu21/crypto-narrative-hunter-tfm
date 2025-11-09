"""
Detailed Wallet Behavior Analysis for Clustering
================================================

Deep dive into wallet behavior patterns to validate clustering potential
and identify key differentiating features for archetypes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/BMAD_TFM/data-collection")
DATA_PATH = BASE_PATH / "outputs/csv"
OUTPUT_PATH = BASE_PATH / "outputs/eda"

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DETAILED WALLET BEHAVIOR ANALYSIS FOR CLUSTERING")
print("="*80)

# Load data
print("\nLoading datasets...")
wallets_df = pd.read_csv(DATA_PATH / 'wallets.csv')
txn_df = pd.read_csv(DATA_PATH / 'transactions.csv')
balances_df = pd.read_csv(DATA_PATH / 'wallet_token_balances.csv')
tokens_df = pd.read_csv(DATA_PATH / 'tokens.csv')

print(f"✓ Wallets: {len(wallets_df):,}")
print(f"✓ Transactions: {len(txn_df):,}")
print(f"✓ Balance snapshots: {len(balances_df):,}")
print(f"✓ Tokens: {len(tokens_df):,}")

# Parse timestamps
if 'timestamp' in txn_df.columns:
    txn_df['timestamp'] = pd.to_datetime(txn_df['timestamp'])

# ==============================================================================
# SECTION 1: WALLET TRADING INTENSITY SEGMENTATION
# ==============================================================================

print("\n" + "="*80)
print("SECTION 1: WALLET TRADING INTENSITY SEGMENTATION")
print("="*80)

# Calculate transactions per wallet
wallet_txn_count = txn_df.groupby('wallet_address').size().reset_index(name='txn_count')

print(f"\nTransaction count distribution:")
print(wallet_txn_count['txn_count'].describe())

# Segment wallets by trading intensity
intensity_segments = []

for idx, row in wallet_txn_count.iterrows():
    count = row['txn_count']
    if count < 5:
        segment = 'Minimal (1-4)'
    elif count < 20:
        segment = 'Low (5-19)'
    elif count < 100:
        segment = 'Medium (20-99)'
    elif count < 500:
        segment = 'High (100-499)'
    else:
        segment = 'Very High (500+)'

    intensity_segments.append({
        'wallet_address': row['wallet_address'],
        'txn_count': count,
        'intensity_segment': segment
    })

intensity_df = pd.DataFrame(intensity_segments)

# Distribution of segments
segment_dist = intensity_df['intensity_segment'].value_counts().sort_index()
print(f"\nTrading Intensity Segments:")
for segment, count in segment_dist.items():
    pct = count / len(intensity_df) * 100
    print(f"  {segment:20} {count:>6,} wallets ({pct:>5.1f}%)")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of segments
segment_dist.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
ax1.set_title('Wallet Distribution by Trading Intensity', fontweight='bold', fontsize=14)
ax1.set_xlabel('Trading Intensity Segment')
ax1.set_ylabel('Number of Wallets')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Log-scale histogram for granularity
ax2.hist(np.log10(wallet_txn_count['txn_count'] + 1), bins=50, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Log10(Transaction Count + 1)')
ax2.set_ylabel('Frequency')
ax2.set_title('Transaction Count Distribution (Log Scale)', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '14_trading_intensity_segments.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 14_trading_intensity_segments.png")
plt.close()

# Clustering Insight
high_variance_indicator = wallet_txn_count['txn_count'].max() / wallet_txn_count['txn_count'].median()
print(f"\nClustering Insight:")
print(f"  Max/Median ratio: {high_variance_indicator:.1f}x")
print(f"  Coefficient of Variation: {wallet_txn_count['txn_count'].std() / wallet_txn_count['txn_count'].mean():.2f}")
if high_variance_indicator > 100:
    print("  ✓ EXCELLENT variance - trading frequency is strong clustering feature")

# ==============================================================================
# SECTION 2: TOKEN SPECIALIZATION ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("SECTION 2: TOKEN SPECIALIZATION vs DIVERSIFICATION")
print("="*80)

# Calculate unique tokens traded per wallet
# Need to consider both token_in and token_out
all_tokens = pd.concat([
    txn_df[['wallet_address', 'token_in']].rename(columns={'token_in': 'token'}),
    txn_df[['wallet_address', 'token_out']].rename(columns={'token_out': 'token'})
])

wallet_token_diversity = all_tokens.groupby('wallet_address')['token'].nunique().reset_index()
wallet_token_diversity.columns = ['wallet_address', 'unique_tokens']

print(f"\nToken diversity distribution:")
print(wallet_token_diversity['unique_tokens'].describe())

# Segment by diversification
diversification_segments = []

for idx, row in wallet_token_diversity.iterrows():
    count = row['unique_tokens']
    if count == 1:
        segment = 'Specialist (1)'
    elif count <= 3:
        segment = 'Focused (2-3)'
    elif count <= 10:
        segment = 'Moderate (4-10)'
    elif count <= 30:
        segment = 'Diversified (11-30)'
    else:
        segment = 'Hyper-diversified (31+)'

    diversification_segments.append({
        'wallet_address': row['wallet_address'],
        'unique_tokens': count,
        'diversification_segment': segment
    })

diversification_df = pd.DataFrame(diversification_segments)

# Distribution
div_segment_dist = diversification_df['diversification_segment'].value_counts()
print(f"\nDiversification Segments:")
for segment in ['Specialist (1)', 'Focused (2-3)', 'Moderate (4-10)',
                'Diversified (11-30)', 'Hyper-diversified (31+)']:
    if segment in div_segment_dist.index:
        count = div_segment_dist[segment]
        pct = count / len(diversification_df) * 100
        print(f"  {segment:25} {count:>6,} wallets ({pct:>5.1f}%)")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Segment distribution
div_segment_order = ['Specialist (1)', 'Focused (2-3)', 'Moderate (4-10)',
                     'Diversified (11-30)', 'Hyper-diversified (31+)']
div_segment_dist_ordered = div_segment_dist.reindex(div_segment_order, fill_value=0)
div_segment_dist_ordered.plot(kind='bar', ax=ax1, color='green', alpha=0.7)
ax1.set_title('Wallet Distribution by Token Diversification', fontweight='bold', fontsize=14)
ax1.set_xlabel('Diversification Level')
ax1.set_ylabel('Number of Wallets')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Distribution curve
unique_tokens = wallet_token_diversity['unique_tokens']
ax2.hist(unique_tokens[unique_tokens <= 50], bins=50, color='purple', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Unique Tokens Traded')
ax2.set_ylabel('Frequency')
ax2.set_title('Token Diversification Distribution (≤50 tokens)', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axvline(unique_tokens.median(), color='red', linestyle='--', linewidth=2, label=f'Median = {unique_tokens.median():.0f}')
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '15_token_diversification.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 15_token_diversification.png")
plt.close()

# Clustering Insight
div_variance = wallet_token_diversity['unique_tokens'].max() / wallet_token_diversity['unique_tokens'].median()
print(f"\nClustering Insight:")
print(f"  Max/Median ratio: {div_variance:.1f}x")
if div_variance > 50:
    print("  ✓ EXCELLENT variance - token diversification is strong clustering feature")

# ==============================================================================
# SECTION 3: CROSS-ANALYSIS - Intensity vs Diversification
# ==============================================================================

print("\n" + "="*80)
print("SECTION 3: TRADING INTENSITY vs TOKEN DIVERSIFICATION")
print("="*80)

# Merge intensity and diversification
combined_df = intensity_df.merge(diversification_df, on='wallet_address')

# Create 2D heatmap
intensity_order = ['Minimal (1-4)', 'Low (5-19)', 'Medium (20-99)', 'High (100-499)', 'Very High (500+)']
div_order = ['Specialist (1)', 'Focused (2-3)', 'Moderate (4-10)',
             'Diversified (11-30)', 'Hyper-diversified (31+)']

# Cross-tabulation
cross_tab = pd.crosstab(combined_df['intensity_segment'],
                        combined_df['diversification_segment'])

# Reindex to ensure order
cross_tab = cross_tab.reindex(index=intensity_order, columns=div_order, fill_value=0)

print(f"\nCross-tabulation (Intensity x Diversification):")
print(cross_tab)

# Visualize heatmap
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Number of Wallets'})
ax.set_title('Wallet Archetypes: Trading Intensity vs Token Diversification',
             fontweight='bold', fontsize=14)
ax.set_xlabel('Token Diversification', fontsize=12)
ax.set_ylabel('Trading Intensity', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / '16_intensity_vs_diversification_heatmap.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 16_intensity_vs_diversification_heatmap.png")
plt.close()

# Identify potential archetype patterns
print(f"\nPotential Archetype Patterns:")
for intensity in intensity_order:
    for div in div_order:
        count = cross_tab.loc[intensity, div]
        if count > 20:  # Significant cluster
            pct = count / len(combined_df) * 100
            print(f"  {intensity} + {div}: {count} wallets ({pct:.1f}%)")

# ==============================================================================
# SECTION 4: GAS EFFICIENCY PATTERNS
# ==============================================================================

print("\n" + "="*80)
print("SECTION 4: GAS EFFICIENCY - Trader Sophistication Indicator")
print("="*80)

if 'gas_used' in txn_df.columns and 'gas_price_gwei' in txn_df.columns:
    # Calculate gas cost per transaction
    txn_df['gas_cost_eth'] = txn_df['gas_used'] * txn_df['gas_price_gwei'] / 1e9

    # Wallet-level gas metrics
    wallet_gas = txn_df.groupby('wallet_address').agg({
        'gas_price_gwei': ['mean', 'median', 'std'],
        'gas_used': ['mean', 'median'],
        'gas_cost_eth': ['sum', 'mean']
    }).reset_index()

    wallet_gas.columns = ['wallet_address', 'avg_gas_price', 'median_gas_price', 'std_gas_price',
                          'avg_gas_used', 'median_gas_used', 'total_gas_cost_eth', 'avg_gas_cost_eth']

    print(f"\nWallet Gas Metrics:")
    print(wallet_gas[['avg_gas_price', 'median_gas_price', 'avg_gas_used', 'total_gas_cost_eth']].describe())

    # Gas optimization score = inverse of median gas price paid
    wallet_gas['gas_optimization_percentile'] = wallet_gas['median_gas_price'].rank(pct=True) * 100

    # Segment by gas efficiency
    wallet_gas['gas_efficiency_segment'] = pd.cut(
        wallet_gas['gas_optimization_percentile'],
        bins=[0, 25, 50, 75, 100],
        labels=['Highly Optimized (Q1)', 'Optimized (Q2)', 'Standard (Q3)', 'Inefficient (Q4)']
    )

    gas_segment_dist = wallet_gas['gas_efficiency_segment'].value_counts().sort_index()
    print(f"\nGas Efficiency Segments:")
    for segment, count in gas_segment_dist.items():
        pct = count / len(wallet_gas) * 100
        print(f"  {segment:25} {count:>6,} wallets ({pct:>5.1f}%)")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gas price distribution
    axes[0, 0].hist(np.log10(wallet_gas['avg_gas_price'] + 0.01), bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Log10(Average Gas Price Gwei + 0.01)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Average Gas Price Distribution (Log Scale)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Gas used distribution
    axes[0, 1].hist(wallet_gas['avg_gas_used'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Average Gas Used')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Average Gas Used Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Total gas cost
    axes[1, 0].hist(np.log10(wallet_gas['total_gas_cost_eth'] + 0.001), bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Log10(Total Gas Cost ETH + 0.001)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Gas Cost Distribution (Log Scale)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Gas efficiency segments
    gas_segment_dist.plot(kind='bar', ax=axes[1, 1], color='green', alpha=0.7)
    axes[1, 1].set_title('Gas Efficiency Segments', fontweight='bold')
    axes[1, 1].set_xlabel('Efficiency Level')
    axes[1, 1].set_ylabel('Number of Wallets')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '17_gas_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 17_gas_efficiency_analysis.png")
    plt.close()

    # Clustering Insight
    print(f"\nClustering Insight:")
    print(f"  Gas price CV: {wallet_gas['avg_gas_price'].std() / wallet_gas['avg_gas_price'].mean():.2f}")
    print("  ✓ Gas efficiency can differentiate sophisticated vs naive traders")

# ==============================================================================
# SECTION 5: BALANCE EVOLUTION - Portfolio Growth Patterns
# ==============================================================================

print("\n" + "="*80)
print("SECTION 5: PORTFOLIO EVOLUTION - Growth Patterns")
print("="*80)

if 'snapshot_date' in balances_df.columns and 'balance_formatted' in balances_df.columns:
    # Parse dates
    balances_df['snapshot_date'] = pd.to_datetime(balances_df['snapshot_date'])

    # Calculate total portfolio value per wallet per day
    # Note: We need USD values, but if not available, we'll use token counts as proxy
    wallet_daily_portfolio = balances_df.groupby(['wallet_address', 'snapshot_date']).agg({
        'token_address': 'nunique',  # Number of tokens held
        'balance_formatted': 'count'  # Number of positions
    }).reset_index()

    wallet_daily_portfolio.columns = ['wallet_address', 'snapshot_date', 'num_unique_tokens', 'num_positions']

    # Get first and last snapshot for each wallet
    wallet_first_last = wallet_daily_portfolio.groupby('wallet_address').agg({
        'snapshot_date': ['min', 'max'],
        'num_unique_tokens': ['first', 'last']
    }).reset_index()

    wallet_first_last.columns = ['wallet_address', 'first_date', 'last_date', 'initial_tokens', 'final_tokens']

    # Calculate token count change
    wallet_first_last['token_count_change'] = wallet_first_last['final_tokens'] - wallet_first_last['initial_tokens']
    wallet_first_last['token_count_change_pct'] = (
        wallet_first_last['token_count_change'] / wallet_first_last['initial_tokens'] * 100
    ).replace([np.inf, -np.inf], np.nan)

    print(f"\nPortfolio Token Count Evolution:")
    print(wallet_first_last[['initial_tokens', 'final_tokens', 'token_count_change']].describe())

    # Classify growth patterns
    growth_patterns = []
    for idx, row in wallet_first_last.iterrows():
        change = row['token_count_change']
        if pd.isna(change):
            pattern = 'Unknown'
        elif change > 5:
            pattern = 'Aggressive Expansion'
        elif change > 0:
            pattern = 'Moderate Growth'
        elif change == 0:
            pattern = 'Stable'
        elif change > -3:
            pattern = 'Moderate Contraction'
        else:
            pattern = 'Aggressive Contraction'

        growth_patterns.append(pattern)

    wallet_first_last['growth_pattern'] = growth_patterns

    pattern_dist = wallet_first_last['growth_pattern'].value_counts()
    print(f"\nGrowth Pattern Distribution:")
    for pattern, count in pattern_dist.items():
        pct = count / len(wallet_first_last) * 100
        print(f"  {pattern:25} {count:>6,} wallets ({pct:>5.1f}%)")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Token count change distribution
    changes = wallet_first_last['token_count_change'].dropna()
    changes_limited = changes[(changes >= -20) & (changes <= 20)]
    ax1.hist(changes_limited, bins=40, color='teal', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Token Count Change (Final - Initial)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Portfolio Token Count Evolution (±20 tokens)', fontweight='bold')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Growth pattern distribution
    pattern_dist.plot(kind='barh', ax=ax2, color='coral', alpha=0.7)
    ax2.set_title('Portfolio Growth Pattern Distribution', fontweight='bold')
    ax2.set_xlabel('Number of Wallets')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '18_portfolio_evolution_patterns.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 18_portfolio_evolution_patterns.png")
    plt.close()

    # Clustering Insight
    print(f"\nClustering Insight:")
    print("  Portfolio evolution patterns can identify:")
    print("    - Accumulators (expanding positions)")
    print("    - Distributors (contracting positions)")
    print("    - Stable holders (diamond hands)")
    print("  ✓ This addresses RQ5 directly")

# ==============================================================================
# SUMMARY STATISTICS FOR FEATURE ENGINEERING
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY: KEY METRICS FOR FEATURE ENGINEERING")
print("="*80)

# Compile key statistics
summary_stats = {
    'Trading Intensity': {
        'Median': wallet_txn_count['txn_count'].median(),
        'Mean': wallet_txn_count['txn_count'].mean(),
        'Max': wallet_txn_count['txn_count'].max(),
        'CV': wallet_txn_count['txn_count'].std() / wallet_txn_count['txn_count'].mean()
    },
    'Token Diversification': {
        'Median': wallet_token_diversity['unique_tokens'].median(),
        'Mean': wallet_token_diversity['unique_tokens'].mean(),
        'Max': wallet_token_diversity['unique_tokens'].max(),
        'CV': wallet_token_diversity['unique_tokens'].std() / wallet_token_diversity['unique_tokens'].mean()
    }
}

if 'gas_used' in txn_df.columns:
    summary_stats['Gas Efficiency'] = {
        'Median Gas Price': wallet_gas['median_gas_price'].median(),
        'Mean Gas Price': wallet_gas['avg_gas_price'].mean(),
        'Std Gas Price': wallet_gas['avg_gas_price'].std(),
        'CV': wallet_gas['avg_gas_price'].std() / wallet_gas['avg_gas_price'].mean()
    }

print("\nKey Variance Metrics (High CV = Good for Clustering):")
for category, metrics in summary_stats.items():
    print(f"\n{category}:")
    for metric, value in metrics.items():
        if metric == 'CV':
            quality = "EXCELLENT" if value > 1.0 else "GOOD" if value > 0.5 else "FAIR"
            print(f"  {metric:20} {value:>10.3f}  ({quality})")
        else:
            print(f"  {metric:20} {value:>10.2f}")

print("\n" + "="*80)
print("DETAILED WALLET ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated visualizations:")
print("  14. 14_trading_intensity_segments.png")
print("  15. 15_token_diversification.png")
print("  16. 16_intensity_vs_diversification_heatmap.png")
print("  17. 17_gas_efficiency_analysis.png")
print("  18. 18_portfolio_evolution_patterns.png")

print(f"\nAll outputs saved to: {OUTPUT_PATH}")
print("\n" + "="*80)
