#!/usr/bin/env python3
"""
Simple Wallet Performance Calculator

Calculates basic performance metrics without complex dependencies.
Direct database calculation approach.
"""

import psycopg
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting wallet performance calculation...")

    db_url = os.getenv('DATABASE_URL')
    conn = psycopg.connect(db_url)

    try:
        with conn.cursor() as cur:
            # Calculate performance metrics directly with SQL
            logger.info("Calculating performance metrics...")

            cur.execute("""
                INSERT INTO wallet_performance (
                    wallet_address,
                    calculation_date,
                    time_period,
                    total_trades,
                    total_gas_cost_usd,
                    unique_tokens_traded
                )
                SELECT
                    wallet_address,
                    NOW() as calculation_date,
                    'all_time' as time_period,
                    COUNT(*) as total_trades,
                    SUM(COALESCE(gas_used, 0) * COALESCE(gas_price_gwei, 0) / 1e9 * 2500) as total_gas_cost_usd,  -- rough ETH price estimate
                    COUNT(DISTINCT COALESCE(token_in, token_out)) as unique_tokens_traded
                FROM transactions
                WHERE wallet_address IN (SELECT wallet_address FROM wallets)  -- Only process wallets that exist
                GROUP BY wallet_address
                ON CONFLICT (wallet_address, time_period) DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    total_gas_cost_usd = EXCLUDED.total_gas_cost_usd,
                    unique_tokens_traded = EXCLUDED.unique_tokens_traded,
                    calculation_date = EXCLUDED.calculation_date,
                    updated_at = NOW()
            """)

            rows_affected = cur.rowcount
            conn.commit()

            logger.info(f"✅ Performance metrics calculated for {rows_affected} wallets")

            # Show summary
            cur.execute("""
                SELECT
                    COUNT(*) as total_wallets,
                    AVG(total_trades) as avg_trades,
                    AVG(unique_tokens_traded) as avg_unique_tokens,
                    MAX(total_trades) as max_trades
                FROM wallet_performance
                WHERE time_period = 'all_time'
            """)

            summary = cur.fetchone()
            logger.info(f"""
Summary Statistics:
- Total wallets: {summary[0]}
- Avg trades per wallet: {summary[1]:.1f}
- Avg unique tokens: {summary[2]:.1f}
- Max trades (single wallet): {summary[3]}
            """)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    main()
