#!/usr/bin/env python3
"""
Dataset Export Script

Exports all collected data to Parquet and CSV formats for thesis analysis.
"""

import psycopg
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_datasets():
    """Export all datasets to Parquet and CSV"""

    db_url = os.getenv('DATABASE_URL')
    conn = psycopg.connect(db_url)

    # Create output directories
    os.makedirs('outputs/parquet', exist_ok=True)
    os.makedirs('outputs/csv', exist_ok=True)

    logger.info("=" * 80)
    logger.info("DATASET EXPORT STARTED")
    logger.info("=" * 80)

    datasets = []

    # 1. Export Tokens
    logger.info("\n📦 Exporting tokens...")
    df_tokens = pd.read_sql("""
        SELECT
            token_address,
            symbol,
            name,
            decimals,
            narrative_category,
            market_cap_rank,
            avg_daily_volume_usd,
            liquidity_tier,
            validation_status,
            classification_confidence,
            classification_method,
            requires_manual_review,
            created_at
        FROM tokens
        ORDER BY market_cap_rank NULLS LAST
    """, conn)

    df_tokens.to_parquet('outputs/parquet/tokens.parquet', index=False, compression='snappy')
    df_tokens.to_csv('outputs/csv/tokens.csv', index=False)
    datasets.append(('Tokens', len(df_tokens)))
    logger.info(f"✅ Exported {len(df_tokens)} tokens")

    # 2. Export Wallets (with transactions only)
    logger.info("\n📦 Exporting wallets...")
    df_wallets = pd.read_sql("""
        SELECT
            w.wallet_address,
            w.first_seen_date,
            w.last_active_date,
            w.total_trades_30d,
            w.avg_daily_volume_eth,
            w.unique_tokens_traded,
            w.is_smart_money,
            w.created_at
        FROM wallets w
        WHERE w.wallet_address IN (SELECT DISTINCT wallet_address FROM transactions)
        ORDER BY w.total_trades_30d DESC
    """, conn)

    df_wallets.to_parquet('outputs/parquet/wallets.parquet', index=False, compression='snappy')
    df_wallets.to_csv('outputs/csv/wallets.csv', index=False)
    datasets.append(('Wallets', len(df_wallets)))
    logger.info(f"✅ Exported {len(df_wallets)} wallets")

    # 3. Export Transactions
    logger.info("\n📦 Exporting transactions...")
    df_transactions = pd.read_sql("""
        SELECT
            tx_hash,
            block_number,
            timestamp,
            wallet_address,
            dex_name,
            pool_address,
            token_in,
            amount_in,
            token_out,
            amount_out,
            gas_used,
            gas_price_gwei,
            eth_value_in,
            eth_value_out,
            transaction_status,
            usd_value_in,
            usd_value_out,
            mev_type,
            mev_damage_eth,
            slippage_percentage
        FROM transactions
        ORDER BY timestamp DESC
    """, conn)

    df_transactions.to_parquet('outputs/parquet/transactions.parquet', index=False, compression='snappy')
    df_transactions.to_csv('outputs/csv/transactions.csv', index=False)
    datasets.append(('Transactions', len(df_transactions)))
    logger.info(f"✅ Exported {len(df_transactions)} transactions")

    # 4. Export Wallet Performance
    logger.info("\n📦 Exporting wallet performance...")
    df_performance = pd.read_sql("""
        SELECT
            wallet_address,
            total_trades,
            unique_tokens_traded,
            total_gas_cost_usd,
            calculation_date,
            time_period
        FROM wallet_performance
        WHERE time_period = 'all_time'
        ORDER BY total_trades DESC
    """, conn)

    df_performance.to_parquet('outputs/parquet/wallet_performance.parquet', index=False, compression='snappy')
    df_performance.to_csv('outputs/csv/wallet_performance.csv', index=False)
    datasets.append(('Wallet Performance', len(df_performance)))
    logger.info(f"✅ Exported {len(df_performance)} wallet performance records")

    # 5. Export ETH Prices
    logger.info("\n📦 Exporting ETH prices...")
    df_prices = pd.read_sql("""
        SELECT
            timestamp,
            price_usd,
            created_at
        FROM eth_prices
        ORDER BY timestamp
    """, conn)

    df_prices.to_parquet('outputs/parquet/eth_prices.parquet', index=False, compression='snappy')
    df_prices.to_csv('outputs/csv/eth_prices.csv', index=False)
    datasets.append(('ETH Prices', len(df_prices)))
    logger.info(f"✅ Exported {len(df_prices)} ETH price points")

    # 6. Export DEX Pools
    logger.info("\n📦 Exporting DEX pools...")
    df_pools = pd.read_sql("""
        SELECT
            pool_address,
            token_address,
            pair_token,
            dex_name,
            tvl_usd,
            volume_24h_usd,
            price_eth,
            last_updated,
            created_at
        FROM token_pools
        ORDER BY tvl_usd DESC NULLS LAST
    """, conn)

    df_pools.to_parquet('outputs/parquet/token_pools.parquet', index=False, compression='snappy')
    df_pools.to_csv('outputs/csv/token_pools.csv', index=False)
    datasets.append(('DEX Pools', len(df_pools)))
    logger.info(f"✅ Exported {len(df_pools)} DEX pools")

    # 7. Create combined dataset for analysis
    logger.info("\n📦 Creating combined wallet analysis dataset...")
    df_combined = pd.read_sql("""
        SELECT
            w.wallet_address,
            w.first_seen_date,
            w.last_active_date,
            wp.total_trades,
            wp.unique_tokens_traded,
            wp.total_gas_cost_usd,
            COUNT(DISTINCT t.dex_name) as dex_platforms_used,
            MIN(t.timestamp) as first_transaction,
            MAX(t.timestamp) as last_transaction,
            COUNT(DISTINCT DATE(t.timestamp)) as active_days
        FROM wallets w
        JOIN wallet_performance wp ON w.wallet_address = wp.wallet_address
        LEFT JOIN transactions t ON w.wallet_address = t.wallet_address
        WHERE wp.time_period = 'all_time'
        GROUP BY w.wallet_address, w.first_seen_date, w.last_active_date,
                 wp.total_trades, wp.unique_tokens_traded, wp.total_gas_cost_usd
        ORDER BY wp.total_trades DESC
    """, conn)

    df_combined.to_parquet('outputs/parquet/wallet_analysis_combined.parquet', index=False, compression='snappy')
    df_combined.to_csv('outputs/csv/wallet_analysis_combined.csv', index=False)
    datasets.append(('Combined Wallet Analysis', len(df_combined)))
    logger.info(f"✅ Created combined analysis dataset with {len(df_combined)} wallets")

    conn.close()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT COMPLETE - SUMMARY")
    logger.info("=" * 80)

    for dataset_name, count in datasets:
        logger.info(f"✅ {dataset_name}: {count:,} records")

    logger.info("\n📁 Output locations:")
    logger.info("   - Parquet: outputs/parquet/")
    logger.info("   - CSV: outputs/csv/")

    # Calculate file sizes
    parquet_size = sum(os.path.getsize(f'outputs/parquet/{f}') for f in os.listdir('outputs/parquet/') if f.endswith('.parquet'))
    csv_size = sum(os.path.getsize(f'outputs/csv/{f}') for f in os.listdir('outputs/csv/') if f.endswith('.csv'))

    logger.info(f"\n💾 Storage:")
    logger.info(f"   - Parquet: {parquet_size / 1024 / 1024:.2f} MB (compressed)")
    logger.info(f"   - CSV: {csv_size / 1024 / 1024:.2f} MB")

    logger.info("\n🎉 All datasets exported successfully!")
    logger.info("=" * 80)

    # Create export manifest
    manifest = {
        'export_date': datetime.now().isoformat(),
        'datasets': {name: count for name, count in datasets},
        'formats': ['parquet', 'csv'],
        'compression': 'snappy (parquet only)',
        'total_records': sum(count for _, count in datasets)
    }

    import json
    with open('outputs/export_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info("📄 Export manifest created: outputs/export_manifest.json")

if __name__ == '__main__':
    export_datasets()
