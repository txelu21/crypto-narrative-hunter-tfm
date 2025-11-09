"""
Comprehensive EDA for Crypto Narrative Hunter Thesis
====================================================

Author: Claude Code - Blockchain Data Analyst
Project: Master's Thesis - Smart Money Archetypes & Narrative Analysis
Date: 2025-10-19

Research Questions:
- RQ1: Can we identify distinct smart money archetypes based on trading behavior?
- RQ2: Do specific archetypes show preferences for certain narratives?
- RQ3: Do early adopters achieve higher risk-adjusted returns?
- RQ4: How does portfolio concentration correlate with performance?
- RQ5: What accumulation/distribution patterns distinguish top performers?

ML Task: Unsupervised clustering (HDBSCAN + K-Means) → Narrative affinity analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Paths
BASE_PATH = Path("/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/BMAD_TFM/data-collection")
DATA_PATH = BASE_PATH / "outputs/csv"
OUTPUT_PATH = BASE_PATH / "outputs/eda"
OUTPUT_PATH.mkdir(exist_ok=True)

print("="*80)
print("CRYPTO NARRATIVE HUNTER - COMPREHENSIVE EDA")
print("="*80)
print(f"\nAnalysis Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==============================================================================
# PHASE 1: DATA LOADING & INITIAL ASSESSMENT
# ==============================================================================

print("\n" + "="*80)
print("PHASE 1: DATA LOADING & INITIAL ASSESSMENT")
print("="*80)

# Load all datasets
print("\n[1.1] Loading datasets...")

datasets = {
    'tokens': 'tokens.csv',
    'wallets': 'wallets.csv',
    'transactions': 'transactions.csv',
    'balances': 'wallet_token_balances.csv',
    'pools': 'token_pools.csv',
    'eth_prices': 'eth_prices.csv'
}

data = {}
for name, filename in datasets.items():
    filepath = DATA_PATH / filename
    print(f"  Loading {name}...", end=" ")
    try:
        data[name] = pd.read_csv(filepath)
        print(f"✓ {len(data[name]):,} rows x {len(data[name].columns)} columns")
    except Exception as e:
        print(f"✗ Error: {e}")
        data[name] = None

# Quick overview
print("\n[1.2] Dataset Overview:")
print("-" * 80)
for name, df in data.items():
    if df is not None:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"{name:15} | {len(df):>10,} rows | {len(df.columns):>3} cols | {memory_mb:>8.2f} MB")

# ==============================================================================
# PHASE 2: DATA QUALITY DEEP DIVE
# ==============================================================================

print("\n" + "="*80)
print("PHASE 2: DATA QUALITY ASSESSMENT")
print("="*80)

def assess_data_quality(df, name):
    """Comprehensive data quality assessment"""
    print(f"\n[2.{name}] Quality Assessment: {name.upper()}")
    print("-" * 80)

    # Basic info
    print(f"Shape: {df.shape}")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())

    # Missing data
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if len(missing_df) > 0:
        print(f"\n⚠️  MISSING DATA DETECTED:")
        print(missing_df.to_string())
    else:
        print("\n✓ No missing data")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"\n⚠️  DUPLICATES: {dup_count:,} duplicate rows ({dup_count/len(df)*100:.2f}%)")
    else:
        print("\n✓ No duplicate rows")

    # Sample data
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    return missing_df

# Assess each dataset
quality_reports = {}
for i, (name, df) in enumerate(data.items(), 1):
    if df is not None:
        quality_reports[name] = assess_data_quality(df, name)

# ==============================================================================
# PHASE 3: TOKENS ANALYSIS - Narrative Quality Assessment
# ==============================================================================

print("\n" + "="*80)
print("PHASE 3: TOKENS ANALYSIS - Narrative Classification Quality")
print("="*80)

tokens_df = data['tokens']

print("\n[3.1] Token Dataset Structure:")
print("-" * 80)
print(f"Columns: {tokens_df.columns.tolist()}")

# Narrative distribution
print("\n[3.2] Narrative Classification Distribution:")
print("-" * 80)

if 'narrative_category' in tokens_df.columns:
    narrative_dist = tokens_df['narrative_category'].value_counts()
    narrative_pct = (narrative_dist / len(tokens_df) * 100).round(2)

    narrative_summary = pd.DataFrame({
        'Count': narrative_dist,
        'Percentage': narrative_pct
    })
    print(narrative_summary.to_string())

    # Critical finding
    other_count = narrative_dist.get('Other', 0)
    other_pct = other_count / len(tokens_df) * 100

    if other_pct > 50:
        print(f"\n⚠️  CRITICAL NARRATIVE QUALITY ISSUE:")
        print(f"   {other_count:,} tokens ({other_pct:.1f}%) classified as 'Other'")
        print(f"   This will significantly limit narrative affinity analysis (RQ2)")
        print(f"   RECOMMENDATION: Prioritize narrative reclassification before clustering")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    narrative_summary.sort_values('Count', ascending=True).plot(
        kind='barh', y='Count', ax=ax1, legend=False
    )
    ax1.set_title('Token Count by Narrative Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Tokens')
    ax1.set_ylabel('Narrative Category')

    # Pie chart (excluding Other for clarity)
    narrative_dist_no_other = narrative_dist[narrative_dist.index != 'Other']
    ax2.pie(narrative_dist_no_other.values, labels=narrative_dist_no_other.index,
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Narrative Distribution (Excluding "Other")', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '01_narrative_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 01_narrative_distribution.png")
    plt.close()

# Token metrics analysis
print("\n[3.3] Token Metrics Analysis:")
print("-" * 80)

numeric_cols = tokens_df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print("\nNumeric columns available:")
    for col in numeric_cols:
        print(f"  - {col}")

    # Basic statistics
    print("\nDescriptive Statistics:")
    print(tokens_df[numeric_cols].describe().to_string())

# ==============================================================================
# PHASE 4: WALLETS ANALYSIS - Smart Money Characteristics
# ==============================================================================

print("\n" + "="*80)
print("PHASE 4: WALLETS ANALYSIS - Smart Money Wallet Characteristics")
print("="*80)

wallets_df = data['wallets']

print("\n[4.1] Wallet Dataset Structure:")
print("-" * 80)
print(f"Total wallets: {len(wallets_df):,}")
print(f"Columns: {wallets_df.columns.tolist()}")

# Identify key metrics
wallet_metrics = wallets_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric metrics ({len(wallet_metrics)}):")
for metric in wallet_metrics:
    print(f"  - {metric}")

print("\n[4.2] Wallet Activity Distribution:")
print("-" * 80)

# Key metrics to analyze
key_metrics = ['total_trades', 'unique_tokens', 'total_volume_usd', 'active_days']
available_metrics = [m for m in key_metrics if m in wallets_df.columns]

if available_metrics:
    stats_summary = wallets_df[available_metrics].describe().T
    stats_summary['skewness'] = wallets_df[available_metrics].apply(skew)
    stats_summary['kurtosis'] = wallets_df[available_metrics].apply(kurtosis)
    print(stats_summary.to_string())

    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics[:4]):
        ax = axes[idx]

        # Histogram with KDE
        data_to_plot = wallets_df[metric].dropna()

        # Use log scale if highly skewed
        if data_to_plot.max() / data_to_plot.median() > 100:
            data_to_plot_log = np.log10(data_to_plot + 1)
            ax.hist(data_to_plot_log, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'Log10({metric} + 1)')
            ax.set_title(f'{metric} Distribution (Log Scale)', fontweight='bold')
        else:
            ax.hist(data_to_plot, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Distribution', fontweight='bold')

        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Median: {data_to_plot.median():.1f}\nMean: {data_to_plot.mean():.1f}\nMax: {data_to_plot.max():.1f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '02_wallet_activity_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 02_wallet_activity_distributions.png")
    plt.close()

# Identify variance for clustering potential
print("\n[4.3] Clustering Potential Assessment:")
print("-" * 80)
print("High variance indicates good clustering potential")

if available_metrics:
    variance_analysis = pd.DataFrame({
        'Metric': available_metrics,
        'CV (Coefficient of Variation)': [
            wallets_df[m].std() / wallets_df[m].mean() if wallets_df[m].mean() > 0 else 0
            for m in available_metrics
        ],
        'Range Ratio (Max/Median)': [
            wallets_df[m].max() / wallets_df[m].median() if wallets_df[m].median() > 0 else 0
            for m in available_metrics
        ]
    }).sort_values('CV (Coefficient of Variation)', ascending=False)

    print(variance_analysis.to_string(index=False))

    high_variance = variance_analysis[variance_analysis['CV (Coefficient of Variation)'] > 1.0]
    if len(high_variance) > 0:
        print(f"\n✓ {len(high_variance)} metrics with CV > 1.0 (excellent for clustering)")
    else:
        print(f"\n⚠️  Low variance detected - may need feature engineering for clustering")

# ==============================================================================
# PHASE 5: TRANSACTIONS ANALYSIS - Trading Behavior Patterns
# ==============================================================================

print("\n" + "="*80)
print("PHASE 5: TRANSACTIONS ANALYSIS - Trading Behavior Patterns")
print("="*80)

txn_df = data['transactions']

print("\n[5.1] Transaction Dataset Structure:")
print("-" * 80)
print(f"Total transactions: {len(txn_df):,}")
print(f"Columns: {txn_df.columns.tolist()}")

# Transaction type distribution
print("\n[5.2] Transaction Type Distribution:")
print("-" * 80)

if 'transaction_type' in txn_df.columns:
    txn_type_dist = txn_df['transaction_type'].value_counts()
    txn_type_pct = (txn_type_dist / len(txn_df) * 100).round(2)

    txn_summary = pd.DataFrame({
        'Count': txn_type_dist,
        'Percentage': txn_type_pct
    })
    print(txn_summary.to_string())

# DEX distribution
print("\n[5.3] DEX Usage Distribution:")
print("-" * 80)

if 'dex' in txn_df.columns:
    dex_dist = txn_df['dex'].value_counts()
    dex_pct = (dex_dist / len(txn_df) * 100).round(2)

    dex_summary = pd.DataFrame({
        'Count': dex_dist,
        'Percentage': dex_pct
    })
    print(dex_summary.to_string())

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    dex_summary.plot(kind='bar', y='Count', ax=ax, legend=False)
    ax.set_title('Transaction Count by DEX', fontsize=14, fontweight='bold')
    ax.set_xlabel('DEX')
    ax.set_ylabel('Number of Transactions')
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (idx, row) in enumerate(dex_summary.iterrows()):
        ax.text(i, row['Count'], f"{row['Percentage']:.1f}%",
                ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '03_dex_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 03_dex_distribution.png")
    plt.close()

# Gas analysis
print("\n[5.4] Gas Efficiency Analysis:")
print("-" * 80)

gas_metrics = ['gas_used', 'gas_price_gwei', 'gas_cost_usd']
available_gas = [m for m in gas_metrics if m in txn_df.columns]

if available_gas:
    print("\nGas Statistics:")
    print(txn_df[available_gas].describe().to_string())

    # Gas efficiency visualization
    if 'gas_used' in txn_df.columns and 'gas_price_gwei' in txn_df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Gas used distribution
        gas_used = txn_df['gas_used'].dropna()
        ax1.hist(gas_used, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Gas Used')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Gas Used Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Gas price distribution
        gas_price = txn_df['gas_price_gwei'].dropna()
        ax2.hist(gas_price, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Gas Price (Gwei)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Gas Price Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / '04_gas_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: 04_gas_analysis.png")
        plt.close()

# Transaction value analysis
print("\n[5.5] Transaction Value Economics:")
print("-" * 80)

if 'amount_usd' in txn_df.columns:
    value_stats = txn_df['amount_usd'].describe()
    print(value_stats.to_string())

    # Value distribution (log scale due to power-law)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    values = txn_df['amount_usd'].dropna()
    values_clean = values[values > 0]

    ax1.hist(values_clean, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Transaction Value (USD)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Transaction Value Distribution (Linear Scale)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.hist(np.log10(values_clean + 1), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Log10(Transaction Value USD + 1)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Transaction Value Distribution (Log Scale)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '05_transaction_values.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 05_transaction_values.png")
    plt.close()

    # Blockchain insight
    print("\nBlockchain Insight:")
    small_txn = (values_clean < 100).sum()
    medium_txn = ((values_clean >= 100) & (values_clean < 10000)).sum()
    large_txn = (values_clean >= 10000).sum()

    print(f"  Small (<$100):        {small_txn:>8,} ({small_txn/len(values_clean)*100:>5.1f}%)")
    print(f"  Medium ($100-$10K):   {medium_txn:>8,} ({medium_txn/len(values_clean)*100:>5.1f}%)")
    print(f"  Large (>$10K):        {large_txn:>8,} ({large_txn/len(values_clean)*100:>5.1f}%)")
    print("\n  → Classic power-law distribution (many small, few large)")

print("\n" + "="*80)
print("PHASE 5 COMPLETE - Continuing with temporal & balance analysis...")
print("="*80)
