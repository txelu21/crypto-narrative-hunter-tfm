"""
Comprehensive EDA - Part 2: Temporal, Balance & Multivariate Analysis
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_PATH = Path("/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/BMAD_TFM/data-collection")
DATA_PATH = BASE_PATH / "outputs/csv"
OUTPUT_PATH = BASE_PATH / "outputs/eda"

# Load data
print("\nLoading datasets for temporal & multivariate analysis...")
txn_df = pd.read_csv(DATA_PATH / 'transactions.csv')
wallets_df = pd.read_csv(DATA_PATH / 'wallets.csv')
balances_df = pd.read_csv(DATA_PATH / 'wallet_token_balances.csv')
eth_prices_df = pd.read_csv(DATA_PATH / 'eth_prices.csv')
tokens_df = pd.read_csv(DATA_PATH / 'tokens.csv')

# ==============================================================================
# PHASE 6: TEMPORAL ANALYSIS - Time-Based Patterns
# ==============================================================================

print("\n" + "="*80)
print("PHASE 6: TEMPORAL ANALYSIS - Trading Patterns Over Time")
print("="*80)

# Parse timestamps
print("\n[6.1] Parsing temporal data...")

if 'block_time' in txn_df.columns:
    txn_df['timestamp'] = pd.to_datetime(txn_df['block_time'])
    txn_df['date'] = txn_df['timestamp'].dt.date
    txn_df['hour'] = txn_df['timestamp'].dt.hour
    txn_df['day_of_week'] = txn_df['timestamp'].dt.dayofweek
    txn_df['day_name'] = txn_df['timestamp'].dt.day_name()

    print(f"✓ Temporal range: {txn_df['timestamp'].min()} to {txn_df['timestamp'].max()}")
    print(f"  Duration: {(txn_df['timestamp'].max() - txn_df['timestamp'].min()).days} days")

    # Daily transaction volume
    print("\n[6.2] Daily Transaction Activity:")
    print("-" * 80)

    daily_txn = txn_df.groupby('date').agg({
        'transaction_hash': 'count',
        'amount_usd': 'sum',
        'wallet_address': 'nunique'
    }).rename(columns={
        'transaction_hash': 'txn_count',
        'amount_usd': 'total_volume_usd',
        'wallet_address': 'active_wallets'
    })

    print(daily_txn.describe().to_string())

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Transaction count
    daily_txn['txn_count'].plot(ax=axes[0], marker='o', linewidth=2)
    axes[0].set_title('Daily Transaction Count', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Transactions')
    axes[0].grid(True, alpha=0.3)

    # Volume
    daily_txn['total_volume_usd'].plot(ax=axes[1], marker='o', linewidth=2, color='green')
    axes[1].set_title('Daily Trading Volume (USD)', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Volume (USD)')
    axes[1].grid(True, alpha=0.3)

    # Active wallets
    daily_txn['active_wallets'].plot(ax=axes[2], marker='o', linewidth=2, color='orange')
    axes[2].set_title('Daily Active Wallets', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Unique Wallets')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '06_temporal_daily_activity.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 06_temporal_daily_activity.png")
    plt.close()

    # Hour of day patterns
    print("\n[6.3] Intraday Patterns (Hour of Day):")
    print("-" * 80)

    hourly_txn = txn_df.groupby('hour').size()
    print(hourly_txn.to_string())

    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_txn.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
    ax.set_title('Transaction Distribution by Hour of Day (UTC)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Transaction Count')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '07_hourly_pattern.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 07_hourly_pattern.png")
    plt.close()

    # Day of week patterns
    print("\n[6.4] Weekly Patterns (Day of Week):")
    print("-" * 80)

    dow_txn = txn_df.groupby('day_name').size().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    print(dow_txn.to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    dow_txn.plot(kind='bar', ax=ax, color='coral', alpha=0.7)
    ax.set_title('Transaction Distribution by Day of Week', fontweight='bold', fontsize=14)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Transaction Count')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '08_weekly_pattern.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 08_weekly_pattern.png")
    plt.close()

    # Blockchain insight
    weekday_avg = dow_txn[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()
    weekend_avg = dow_txn[['Saturday', 'Sunday']].mean()
    print(f"\nBlockchain Insight:")
    print(f"  Weekday avg: {weekday_avg:.0f} txn/day")
    print(f"  Weekend avg: {weekend_avg:.0f} txn/day")
    print(f"  Weekday/Weekend ratio: {weekday_avg/weekend_avg:.2f}x")
    if weekday_avg > weekend_avg:
        print("  → Institutional/professional trading pattern detected")
    else:
        print("  → Retail-dominated trading pattern")

# ==============================================================================
# PHASE 7: BALANCE SNAPSHOTS - Portfolio Evolution Analysis
# ==============================================================================

print("\n" + "="*80)
print("PHASE 7: BALANCE SNAPSHOTS - Portfolio Evolution Analysis")
print("="*80)

print("\n[7.1] Balance Dataset Structure:")
print("-" * 80)
print(f"Total snapshots: {len(balances_df):,}")
print(f"Columns: {balances_df.columns.tolist()}")

# Parse snapshot dates
if 'snapshot_date' in balances_df.columns:
    balances_df['snapshot_date'] = pd.to_datetime(balances_df['snapshot_date'])

    print(f"\nSnapshot range: {balances_df['snapshot_date'].min()} to {balances_df['snapshot_date'].max()}")

    # Wallet coverage
    unique_wallets_in_balances = balances_df['wallet_address'].nunique()
    print(f"Unique wallets tracked: {unique_wallets_in_balances:,}")

    # Snapshot completeness
    unique_dates = balances_df['snapshot_date'].nunique()
    print(f"Unique snapshot dates: {unique_dates}")

    # Daily snapshot count
    daily_snapshots = balances_df.groupby('snapshot_date').size()
    print(f"\nSnapshots per day:")
    print(f"  Mean: {daily_snapshots.mean():.0f}")
    print(f"  Min:  {daily_snapshots.min():.0f}")
    print(f"  Max:  {daily_snapshots.max():.0f}")

    if daily_snapshots.std() / daily_snapshots.mean() > 0.1:
        print(f"\n⚠️  Snapshot count variance detected (CV: {daily_snapshots.std() / daily_snapshots.mean():.2f})")
        print("     May indicate wallet churn or data collection issues")

# Portfolio concentration analysis
print("\n[7.2] Portfolio Concentration Metrics:")
print("-" * 80)

if 'balance_usd' in balances_df.columns:
    # Calculate daily portfolio metrics per wallet
    wallet_daily_metrics = balances_df.groupby(['wallet_address', 'snapshot_date']).agg({
        'token_address': 'count',  # Number of tokens held
        'balance_usd': 'sum'       # Total portfolio value
    }).rename(columns={
        'token_address': 'num_tokens',
        'balance_usd': 'total_value_usd'
    }).reset_index()

    print(f"Calculated metrics for {len(wallet_daily_metrics):,} wallet-day combinations")

    # Portfolio size distribution
    print("\nPortfolio Value Distribution:")
    print(wallet_daily_metrics['total_value_usd'].describe().to_string())

    # Number of tokens held
    print("\nTokens Held Distribution:")
    print(wallet_daily_metrics['num_tokens'].describe().to_string())

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Portfolio value (log scale)
    values = wallet_daily_metrics['total_value_usd'].dropna()
    values = values[values > 0]
    axes[0].hist(np.log10(values + 1), bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Log10(Portfolio Value USD + 1)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Portfolio Value Distribution (Daily Snapshots)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Number of tokens
    axes[1].hist(wallet_daily_metrics['num_tokens'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Tokens Held')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Portfolio Diversification (Daily Snapshots)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '09_portfolio_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 09_portfolio_distributions.png")
    plt.close()

# Portfolio evolution for sample wallets
print("\n[7.3] Portfolio Evolution Examples:")
print("-" * 80)

if 'snapshot_date' in balances_df.columns and 'balance_usd' in balances_df.columns:
    # Get top 10 wallets by average portfolio value
    top_wallets = wallet_daily_metrics.groupby('wallet_address')['total_value_usd'].mean().nlargest(10).index

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Portfolio value evolution
    for wallet in top_wallets[:5]:  # Plot top 5
        wallet_data = wallet_daily_metrics[wallet_daily_metrics['wallet_address'] == wallet].sort_values('snapshot_date')
        axes[0].plot(wallet_data['snapshot_date'], wallet_data['total_value_usd'],
                    marker='o', linewidth=2, label=f"{wallet[:8]}...", alpha=0.7)

    axes[0].set_title('Top 5 Wallets - Portfolio Value Evolution', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Portfolio Value (USD)')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Token count evolution
    for wallet in top_wallets[:5]:
        wallet_data = wallet_daily_metrics[wallet_daily_metrics['wallet_address'] == wallet].sort_values('snapshot_date')
        axes[1].plot(wallet_data['snapshot_date'], wallet_data['num_tokens'],
                    marker='o', linewidth=2, label=f"{wallet[:8]}...", alpha=0.7)

    axes[1].set_title('Top 5 Wallets - Token Count Evolution', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Number of Tokens')
    axes[1].set_xlabel('Date')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '10_portfolio_evolution_samples.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 10_portfolio_evolution_samples.png")
    plt.close()

# ==============================================================================
# PHASE 8: MULTIVARIATE ANALYSIS - Feature Correlations
# ==============================================================================

print("\n" + "="*80)
print("PHASE 8: MULTIVARIATE ANALYSIS - Feature Correlations")
print("="*80)

print("\n[8.1] Wallet-Level Correlation Analysis:")
print("-" * 80)

# Select numeric features from wallets
wallet_numeric = wallets_df.select_dtypes(include=[np.number])

if len(wallet_numeric.columns) > 0:
    # Calculate correlation matrix
    corr_matrix = wallet_numeric.corr()

    print(f"Correlation matrix computed for {len(corr_matrix)} features")

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'fontsize': 8})
    ax.set_title('Wallet Metrics Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '11_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 11_correlation_matrix.png")
    plt.close()

    # Identify strong correlations (potential multicollinearity)
    print("\n[8.2] Multicollinearity Detection:")
    print("-" * 80)

    # Find pairs with |correlation| > 0.85
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.85:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        print(f"\n⚠️  {len(high_corr_pairs)} feature pairs with |r| > 0.85 detected:")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.to_string(index=False))
        print("\nRECOMMENDATION: Consider removing one feature from each pair for clustering")
    else:
        print("\n✓ No severe multicollinearity detected (all |r| < 0.85)")

# Scatter plots for interesting relationships
print("\n[8.3] Feature Interaction Visualizations:")
print("-" * 80)

# Define interesting pairs to visualize
interesting_pairs = [
    ('total_trades', 'total_volume_usd'),
    ('total_trades', 'unique_tokens'),
    ('unique_tokens', 'total_volume_usd'),
    ('active_days', 'total_trades')
]

available_pairs = [(f1, f2) for f1, f2 in interesting_pairs
                   if f1 in wallets_df.columns and f2 in wallets_df.columns]

if available_pairs:
    n_pairs = len(available_pairs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (feat1, feat2) in enumerate(available_pairs[:4]):
        ax = axes[idx]

        # Create scatter plot
        x = wallets_df[feat1].dropna()
        y = wallets_df[feat2].dropna()

        # Align data
        valid_idx = x.index.intersection(y.index)
        x = x.loc[valid_idx]
        y = y.loc[valid_idx]

        # Use log scale if highly skewed
        if x.max() / x.median() > 100:
            x = np.log10(x + 1)
            feat1_label = f"Log10({feat1} + 1)"
        else:
            feat1_label = feat1

        if y.max() / y.median() > 100:
            y = np.log10(y + 1)
            feat2_label = f"Log10({feat2} + 1)"
        else:
            feat2_label = feat2

        ax.scatter(x, y, alpha=0.5, s=20)
        ax.set_xlabel(feat1_label)
        ax.set_ylabel(feat2_label)
        ax.set_title(f'{feat1} vs {feat2}', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr_coef = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '12_feature_interactions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 12_feature_interactions.png")
    plt.close()

print("\n" + "="*80)
print("PART 2 COMPLETE - Proceeding to ML readiness assessment...")
print("="*80)
