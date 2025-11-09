#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA)
Crypto Narrative Hunter - Master Thesis Project

Author: Txelu Sanchez
Date: October 19, 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Paths
DATA_DIR = Path('/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/BMAD_TFM/data-collection/outputs/csv')
OUTPUT_DIR = Path('/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/BMAD_TFM/data-collection/outputs/eda')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("COMPREHENSIVE EDA - CRYPTO NARRATIVE HUNTER")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output Directory: {OUTPUT_DIR}")
print()

# Load all datasets
print("Loading datasets...")
print("-" * 80)

try:
    # Tokens
    print("📊 Loading tokens data...")
    df_tokens = pd.read_csv(DATA_DIR / 'tokens.csv')
    print(f"   ✅ Loaded {len(df_tokens):,} tokens")

    # Wallets
    print("💼 Loading wallets data...")
    df_wallets = pd.read_csv(DATA_DIR / 'wallets.csv')
    print(f"   ✅ Loaded {len(df_wallets):,} wallets")

    # Transactions
    print("💱 Loading transactions data...")
    df_transactions = pd.read_csv(DATA_DIR / 'transactions.csv')
    print(f"   ✅ Loaded {len(df_transactions):,} transactions")

    # Balance snapshots (sample for memory efficiency)
    print("📈 Loading balance snapshots (sampling)...")
    df_balances = pd.read_csv(DATA_DIR / 'wallet_token_balances.csv', nrows=100000)
    print(f"   ✅ Loaded {len(df_balances):,} balance snapshots (sampled)")

    # DEX pools
    print("🏊 Loading DEX pools data...")
    df_pools = pd.read_csv(DATA_DIR / 'token_pools.csv')
    print(f"   ✅ Loaded {len(df_pools):,} DEX pools")

    # ETH prices
    print("💰 Loading ETH prices...")
    df_eth_prices = pd.read_csv(DATA_DIR / 'eth_prices.csv')
    print(f"   ✅ Loaded {len(df_eth_prices):,} ETH price records")

    print("\n✅ All datasets loaded successfully\n")

except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    sys.exit(1)

# ============================================================================
# 1. DATA QUALITY ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("1. DATA QUALITY ASSESSMENT")
print("="*80)

def data_quality_summary(df, name):
    """Generate data quality summary"""
    print(f"\n{name}:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_cols = missing[missing > 0]

    if len(missing_cols) > 0:
        print(f"  ⚠️  Columns with missing values: {len(missing_cols)}")
        for col, miss_count in list(missing_cols.items())[:5]:
            print(f"     - {col}: {miss_count:,} ({missing_pct[col]:.2f}%)")
    else:
        print("  ✅ No missing values")

    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠️  Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    else:
        print("  ✅ No duplicate rows")

data_quality_summary(df_tokens, "TOKENS")
data_quality_summary(df_wallets, "WALLETS")
data_quality_summary(df_transactions, "TRANSACTIONS")
data_quality_summary(df_balances, "BALANCE SNAPSHOTS (Sampled)")
data_quality_summary(df_pools, "DEX POOLS")
data_quality_summary(df_eth_prices, "ETH PRICES")

# ============================================================================
# 2. TOKEN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("2. TOKEN ANALYSIS")
print("="*80)

# Narrative distribution
print("\n📊 Narrative Distribution:")
print("-" * 80)
narrative_dist = df_tokens['narrative_category'].value_counts()
narrative_pct = (narrative_dist / len(df_tokens) * 100).round(2)

for narrative, count in narrative_dist.items():
    print(f"  {narrative:20s}: {count:5d} ({narrative_pct[narrative]:5.2f}%)")

print(f"\nTotal Narratives: {df_tokens['narrative_category'].nunique()}")
other_count = narrative_dist.get('Other', 0)
print(f"⚠️  'Other' category: {other_count:,} tokens ({narrative_pct.get('Other', 0):.2f}%) - requires manual review")

# Visualize narrative distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
narrative_dist.plot(kind='bar', ax=axes[0], color=sns.color_palette("husl", len(narrative_dist)))
axes[0].set_title('Token Distribution by Narrative Category', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Narrative Category', fontsize=12)
axes[0].set_ylabel('Number of Tokens', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(narrative_dist.values):
    axes[0].text(i, v + 10, f"{v:,}\n({narrative_pct.values[i]:.1f}%)",
                ha='center', va='bottom', fontsize=9)

# Pie chart (excluding 'Other')
narrative_dist_no_other = narrative_dist.drop('Other', errors='ignore')
axes[1].pie(narrative_dist_no_other.values, labels=narrative_dist_no_other.index, autopct='%1.1f%%',
           colors=sns.color_palette("husl", len(narrative_dist_no_other)), startangle=90)
axes[1].set_title('Narrative Distribution (Excluding "Other")', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_narrative_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {OUTPUT_DIR / '01_narrative_distribution.png'}")
plt.close()

# Market cap rank distribution
if 'market_cap_rank' in df_tokens.columns:
    print("\n📊 Market Cap Rank Distribution:")
    print("-" * 80)

    rank_bins = [0, 100, 250, 500, 750, 1000, 1500]
    rank_labels = ['1-100', '101-250', '251-500', '501-750', '751-1000', '1001-1500']

    df_tokens['rank_category'] = pd.cut(df_tokens['market_cap_rank'],
                                        bins=rank_bins,
                                        labels=rank_labels,
                                        include_lowest=True)

    rank_dist = df_tokens['rank_category'].value_counts().sort_index()

    for rank, count in rank_dist.items():
        print(f"  {rank:15s}: {count:5d}")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    rank_dist.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(rank_dist)))
    ax.set_title('Token Distribution by Market Cap Rank Range', fontsize=14, fontweight='bold')
    ax.set_xlabel('Market Cap Rank Range', fontsize=12)
    ax.set_ylabel('Number of Tokens', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(rank_dist.values):
        ax.text(i, v + 5, f"{v:,}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_rank_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / '02_rank_distribution.png'}")
    plt.close()

# ============================================================================
# 3. WALLET ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("3. WALLET ANALYSIS")
print("="*80)

print(f"\n💼 Total Wallets: {len(df_wallets):,}")

# Tier distribution
if 'has_transactions' in df_wallets.columns:
    tier1_count = df_wallets['has_transactions'].sum()
    tier2_count = len(df_wallets) - tier1_count

    print(f"\n📊 Wallet Tier Distribution:")
    print(f"  Tier 1 (Complete data): {tier1_count:,} ({tier1_count/len(df_wallets)*100:.2f}%)")
    print(f"  Tier 2 (Aggregate data): {tier2_count:,} ({tier2_count/len(df_wallets)*100:.2f}%)")

# Activity metrics
if 'total_trades_30d' in df_wallets.columns:
    print(f"\n📊 Wallet Activity Metrics:")
    print(f"  Total Trades (30d):")
    print(f"    Mean: {df_wallets['total_trades_30d'].mean():.2f}")
    print(f"    Median: {df_wallets['total_trades_30d'].median():.2f}")
    print(f"    Min: {df_wallets['total_trades_30d'].min():.0f}")
    print(f"    Max: {df_wallets['total_trades_30d'].max():.0f}")

if 'unique_tokens_traded' in df_wallets.columns:
    print(f"\n  Unique Tokens Traded:")
    print(f"    Mean: {df_wallets['unique_tokens_traded'].mean():.2f}")
    print(f"    Median: {df_wallets['unique_tokens_traded'].median():.2f}")
    print(f"    Min: {df_wallets['unique_tokens_traded'].min():.0f}")
    print(f"    Max: {df_wallets['unique_tokens_traded'].max():.0f}")

# Visualize wallet metrics
if 'total_trades_30d' in df_wallets.columns and 'unique_tokens_traded' in df_wallets.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_wallets['total_trades_30d'].hist(bins=50, ax=axes[0], edgecolor='black', color='lightgreen')
    axes[0].set_title('Distribution of Total Trades (30 days)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Total Trades')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)

    df_wallets['unique_tokens_traded'].hist(bins=50, ax=axes[1], edgecolor='black', color='lightblue')
    axes[1].set_title('Distribution of Unique Tokens Traded', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Unique Tokens')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_wallet_activity.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR / '03_wallet_activity.png'}")
    plt.close()

# ============================================================================
# 4. TRANSACTION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("4. TRANSACTION ANALYSIS")
print("="*80)

print(f"\n💱 Total Transactions: {len(df_transactions):,}")

# Parse timestamp
if 'block_time' in df_transactions.columns:
    df_transactions['timestamp'] = pd.to_datetime(df_transactions['block_time'])
elif 'timestamp' in df_transactions.columns:
    df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

# Unique wallets
if 'wallet_address' in df_transactions.columns:
    unique_wallets = df_transactions['wallet_address'].nunique()
    print(f"Unique wallets: {unique_wallets:,}")
    print(f"Avg transactions/wallet: {len(df_transactions)/unique_wallets:.2f}")

# DEX distribution
if 'dex_name' in df_transactions.columns:
    print(f"\n🏊 DEX Distribution:")
    print("-" * 80)
    dex_dist = df_transactions['dex_name'].value_counts()
    dex_pct = (dex_dist / len(df_transactions) * 100).round(2)

    for dex, count in dex_dist.items():
        print(f"  {dex:20s}: {count:7,} ({dex_pct[dex]:5.2f}%)")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    dex_dist.plot(kind='bar', ax=ax, color=sns.color_palette("Set2", len(dex_dist)))
    ax.set_title('Transaction Distribution by DEX', fontsize=14, fontweight='bold')
    ax.set_xlabel('DEX Name', fontsize=12)
    ax.set_ylabel('Number of Transactions', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(dex_dist.values):
        ax.text(i, v + 100, f"{v:,}\n({dex_pct.values[i]:.1f}%)",
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_dex_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR / '04_dex_distribution.png'}")
    plt.close()

# Gas analysis
if 'gas_used' in df_transactions.columns and 'gas_price_gwei' in df_transactions.columns:
    df_transactions['gas_cost_eth'] = (df_transactions['gas_used'] *
                                       df_transactions['gas_price_gwei']) / 1e9

    print(f"\n⛽ Gas Fee Analysis:")
    print(f"  Total gas spent: {df_transactions['gas_cost_eth'].sum():.4f} ETH")
    print(f"  Average gas/tx: {df_transactions['gas_cost_eth'].mean():.6f} ETH")
    print(f"  Median gas/tx: {df_transactions['gas_cost_eth'].median():.6f} ETH")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    df_transactions['gas_used'].hist(bins=50, ax=axes[0], edgecolor='black')
    axes[0].set_title('Distribution of Gas Used', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Gas Used')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)

    df_transactions['gas_price_gwei'].hist(bins=50, ax=axes[1], edgecolor='black', color='coral')
    axes[1].set_title('Distribution of Gas Price (Gwei)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Gas Price (Gwei)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)

    df_transactions['gas_cost_eth'].hist(bins=50, ax=axes[2], edgecolor='black', color='lightgreen')
    axes[2].set_title('Distribution of Gas Cost (ETH)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Gas Cost (ETH)')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_gas_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / '05_gas_analysis.png'}")
    plt.close()

# Temporal analysis
if 'timestamp' in df_transactions.columns:
    min_date = df_transactions['timestamp'].min()
    max_date = df_transactions['timestamp'].max()

    print(f"\n📅 Temporal Coverage:")
    print(f"  From: {min_date}")
    print(f"  To: {max_date}")
    print(f"  Duration: {(max_date - min_date).days} days")

    # Daily transaction volume
    df_transactions['date'] = df_transactions['timestamp'].dt.date
    daily_txs = df_transactions.groupby('date').size()

    print(f"\n  Daily transaction statistics:")
    print(f"    Mean: {daily_txs.mean():.0f} transactions/day")
    print(f"    Median: {daily_txs.median():.0f} transactions/day")
    print(f"    Min: {daily_txs.min()} transactions/day")
    print(f"    Max: {daily_txs.max()} transactions/day")

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 6))
    daily_txs.plot(ax=ax, color='steelblue', linewidth=2)
    ax.set_title('Daily Transaction Count Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Transactions', fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_temporal_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / '06_temporal_analysis.png'}")
    plt.close()

# ============================================================================
# 5. BALANCE SNAPSHOTS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("5. BALANCE SNAPSHOTS ANALYSIS (Sampled)")
print("="*80)

print(f"\n⚠️  Analyzing {len(df_balances):,} sampled records out of ~1.77M total")

# Parse snapshot date
if 'snapshot_date' in df_balances.columns:
    df_balances['snapshot_date'] = pd.to_datetime(df_balances['snapshot_date'])

    print(f"\n📅 Snapshot Period:")
    print(f"  From: {df_balances['snapshot_date'].min()}")
    print(f"  To: {df_balances['snapshot_date'].max()}")
    print(f"  Duration: {(df_balances['snapshot_date'].max() - df_balances['snapshot_date'].min()).days} days")

# Unique counts
if 'wallet_address' in df_balances.columns:
    print(f"\nUnique wallets (sample): {df_balances['wallet_address'].nunique():,}")

if 'token_address' in df_balances.columns:
    print(f"Unique tokens (sample): {df_balances['token_address'].nunique():,}")

# Portfolio value over time
if 'balance_usd' in df_balances.columns and 'snapshot_date' in df_balances.columns:
    daily_portfolio = df_balances.groupby('snapshot_date')['balance_usd'].sum()

    print(f"\n💰 Portfolio Value Statistics (Sample):")
    print(f"  Mean daily value: ${daily_portfolio.mean():,.2f}")
    print(f"  Min daily value: ${daily_portfolio.min():,.2f}")
    print(f"  Max daily value: ${daily_portfolio.max():,.2f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    daily_portfolio.plot(ax=ax, color='steelblue', linewidth=2)
    ax.set_title('Total Portfolio Value Over Time (Sampled Wallets)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Value (USD)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_portfolio_evolution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR / '07_portfolio_evolution.png'}")
    plt.close()

# ============================================================================
# 6. DEX POOLS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("6. DEX POOLS ANALYSIS")
print("="*80)

print(f"\n🏊 Total DEX Pools: {len(df_pools):,}")

# Pool type distribution
if 'pool_type' in df_pools.columns:
    print(f"\n📊 Pool Type Distribution:")
    print("-" * 80)
    pool_type_dist = df_pools['pool_type'].value_counts()
    pool_type_pct = (pool_type_dist / len(df_pools) * 100).round(2)

    for pool_type, count in pool_type_dist.items():
        print(f"  {pool_type:20s}: {count:5d} ({pool_type_pct[pool_type]:5.2f}%)")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    pool_type_dist.plot(kind='bar', ax=ax, color=sns.color_palette("Set3", len(pool_type_dist)))
    ax.set_title('DEX Pool Type Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Pool Type', fontsize=12)
    ax.set_ylabel('Number of Pools', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(pool_type_dist.values):
        ax.text(i, v + 10, f"{v:,}\n({pool_type_pct.values[i]:.1f}%)",
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_pool_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR / '08_pool_distribution.png'}")
    plt.close()

# TVL analysis
if 'tvl_usd' in df_pools.columns:
    total_tvl = df_pools['tvl_usd'].sum()

    print(f"\n💰 Total Value Locked (TVL):")
    print(f"  Total TVL: ${total_tvl:,.2f}")
    print(f"  Average TVL/pool: ${df_pools['tvl_usd'].mean():,.2f}")
    print(f"  Median TVL/pool: ${df_pools['tvl_usd'].median():,.2f}")

# ============================================================================
# 7. ETH PRICE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("7. ETH PRICE ANALYSIS")
print("="*80)

if 'timestamp' in df_eth_prices.columns:
    df_eth_prices['timestamp'] = pd.to_datetime(df_eth_prices['timestamp'])

    print(f"\n📅 Price Period:")
    print(f"  From: {df_eth_prices['timestamp'].min()}")
    print(f"  To: {df_eth_prices['timestamp'].max()}")
    print(f"  Records: {len(df_eth_prices):,}")

if 'price_usd' in df_eth_prices.columns:
    print(f"\n💰 ETH/USD Price Statistics:")
    print(f"  Mean: ${df_eth_prices['price_usd'].mean():,.2f}")
    print(f"  Median: ${df_eth_prices['price_usd'].median():,.2f}")
    print(f"  Min: ${df_eth_prices['price_usd'].min():,.2f}")
    print(f"  Max: ${df_eth_prices['price_usd'].max():,.2f}")

    # Calculate volatility
    df_eth_prices['returns'] = df_eth_prices['price_usd'].pct_change()
    volatility_annual = df_eth_prices['returns'].std() * np.sqrt(365 * 24) * 100
    volatility_daily = df_eth_prices['returns'].std() * np.sqrt(24) * 100

    print(f"\n📊 Price Volatility:")
    print(f"  Annualized: {volatility_annual:.2f}%")
    print(f"  Daily: {volatility_daily:.2f}%")

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_eth_prices['timestamp'], df_eth_prices['price_usd'],
           color='steelblue', linewidth=1.5, alpha=0.7)
    ax.set_title('ETH/USD Price Over Time (Hourly)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_eth_price.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {OUTPUT_DIR / '09_eth_price.png'}")
    plt.close()

# ============================================================================
# 8. DATA READINESS ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("8. DATA READINESS ASSESSMENT FOR EPIC 4")
print("="*80)

# Calculate tier 1 wallets
tier1_wallets = len(df_wallets)
if 'has_transactions' in df_wallets.columns:
    tier1_wallets = df_wallets['has_transactions'].sum()

# Gas completeness
gas_completeness = 0
if 'gas_used' in df_transactions.columns:
    gas_completeness = (df_transactions['gas_used'].notna().sum() / len(df_transactions) * 100)

# Narrative classification
other_count = (df_tokens['narrative_category'] == 'Other').sum()
other_pct = other_count / len(df_tokens) * 100
narrative_complete_pct = 100 - other_pct

readiness_report = []

print("\n1️⃣  Wallet Data (Tier 1):")
status = '✅' if tier1_wallets >= 2000 else '⚠️'
print(f"   {status} {tier1_wallets:,} wallets available for deep analysis")
readiness_report.append({
    'Component': 'Wallet Data (Tier 1)',
    'Status': status,
    'Details': f"{tier1_wallets:,} wallets"
})

print("\n2️⃣  Transaction Data:")
status = '✅' if len(df_transactions) >= 30000 else '⚠️'
print(f"   {status} {len(df_transactions):,} transactions available")
print(f"   {'✅' if gas_completeness > 95 else '⚠️'} Gas data completeness: {gas_completeness:.2f}%")
readiness_report.append({
    'Component': 'Transaction Data',
    'Status': status,
    'Details': f"{len(df_transactions):,} txs, {gas_completeness:.1f}% gas"
})

print("\n3️⃣  Balance Snapshots:")
print("   ✅ Daily balance data available (~1.77M total snapshots)")
readiness_report.append({
    'Component': 'Balance Snapshots',
    'Status': '✅',
    'Details': '~1.77M snapshots'
})

print("\n4️⃣  Token Metadata & Narratives:")
status = '⚠️' if other_pct > 50 else '✅'
print(f"   {status} {narrative_complete_pct:.1f}% narratives classified")
print(f"   ⚠️  {other_count:,} tokens need manual review")
readiness_report.append({
    'Component': 'Token Narratives',
    'Status': status,
    'Details': f"{narrative_complete_pct:.1f}% classified"
})

print("\n5️⃣  DEX Pools:")
print(f"   ✅ {len(df_pools):,} pools available")
readiness_report.append({
    'Component': 'DEX Pools',
    'Status': '✅',
    'Details': f"{len(df_pools):,} pools"
})

print("\n6️⃣  ETH Prices:")
print(f"   ✅ {len(df_eth_prices):,} hourly price records")
readiness_report.append({
    'Component': 'ETH Prices',
    'Status': '✅',
    'Details': f"{len(df_eth_prices):,} records"
})

# Calculate readiness score
ready_count = sum(1 for r in readiness_report if r['Status'] == '✅')
total_count = len(readiness_report)
readiness_score = (ready_count / total_count * 100)

print("\n" + "="*80)
print("READINESS SUMMARY")
print("="*80)

df_readiness = pd.DataFrame(readiness_report)
print(df_readiness.to_string(index=False))

print(f"\n🎯 Overall Readiness Score: {readiness_score:.1f}% ({ready_count}/{total_count} components ready)")

if readiness_score >= 80:
    print("\n✅ DATA IS READY FOR EPIC 4: FEATURE ENGINEERING & CLUSTERING")
    print("\nNext Steps:")
    print("  1. Story 4.1: Calculate wallet performance metrics")
    print("  2. Story 4.2: Manual narrative reclassification")
    print("  3. Story 4.3: Execute clustering analysis")
    print("  4. Story 4.4: Cluster-narrative affinity analysis")
else:
    print("\n⚠️  SOME COMPONENTS NEED ATTENTION")

# Save readiness report
df_readiness.to_csv(OUTPUT_DIR / 'data_readiness_report.csv', index=False)
print(f"\n✅ Saved: {OUTPUT_DIR / 'data_readiness_report.csv'}")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("9. SUMMARY STATISTICS")
print("="*80)

summary = {
    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'Data Period': 'September 3 - October 3, 2025',
    'Total Tokens': len(df_tokens),
    'Unique Narratives': df_tokens['narrative_category'].nunique(),
    'Tokens Classified': len(df_tokens) - other_count,
    'Tokens Needing Review': other_count,
    'Total Wallets': len(df_wallets),
    'Tier 1 Wallets': tier1_wallets,
    'Total Transactions': len(df_transactions),
    'Unique Wallets (Txs)': df_transactions['wallet_address'].nunique() if 'wallet_address' in df_transactions.columns else 'N/A',
    'Balance Snapshots (Sampled)': len(df_balances),
    'DEX Pools': len(df_pools),
    'ETH Price Records': len(df_eth_prices),
    'Data Readiness Score': f"{readiness_score:.1f}%",
    'Ready for Epic 4': 'Yes' if readiness_score >= 80 else 'Needs Attention'
}

print()
for key, value in summary.items():
    print(f"{key:.<40s} {str(value):.>38s}")

# Save summary
df_summary = pd.DataFrame([summary]).T
df_summary.columns = ['Value']
df_summary.to_csv(OUTPUT_DIR / 'eda_summary.csv')
print(f"\n✅ Saved: {OUTPUT_DIR / 'eda_summary.csv'}")

# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "="*80)
print("EDA ANALYSIS COMPLETE")
print("="*80)
print(f"\n📊 Generated {len(list(OUTPUT_DIR.glob('*.png')))} visualizations")
print(f"📊 Generated {len(list(OUTPUT_DIR.glob('*.csv')))} reports")
print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
print("\n✅ EDA analysis completed successfully!")
print("="*80)
