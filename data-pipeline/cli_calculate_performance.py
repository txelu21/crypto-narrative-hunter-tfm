#!/usr/bin/env python3
"""
Wallet Performance Metrics Calculator CLI

Calculates comprehensive trading performance metrics for all wallets with transaction data.
Processes transactions from the database and stores results in wallet_performance table.

Usage:
    python cli_calculate_performance.py [--batch-size 100] [--time-period all_time]
"""

import argparse
import logging
import sys
from datetime import datetime
from decimal import Decimal
from typing import Dict, List
from collections import defaultdict

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
import os

from services.wallets.performance_calculator import (
    WalletPerformanceCalculator,
    Trade,
    PerformanceMetrics
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/performance_calculation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PerformanceCalculationCLI:
    """CLI for calculating wallet performance metrics"""

    def __init__(self, batch_size: int = 100, time_period: str = 'all_time'):
        self.batch_size = batch_size
        self.time_period = time_period
        self.calculator = WalletPerformanceCalculator(risk_free_rate=0.02)

        # Database connection
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment")

        self.conn = psycopg.connect(self.db_url, autocommit=False)

        self.total_wallets_processed = 0
        self.total_wallets_failed = 0

    def load_eth_prices(self) -> Dict[datetime, Decimal]:
        """Load ETH/USD prices from database"""
        logger.info("Loading ETH prices from database...")

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT timestamp, price_usd
                FROM eth_prices
                ORDER BY timestamp
            """)

            prices = {}
            for row in cur.fetchall():
                # Use date only for easier matching
                date_key = row['timestamp'].date()
                prices[date_key] = Decimal(str(row['price_usd']))

        logger.info(f"Loaded {len(prices)} ETH price points")
        return prices

    def get_wallets_with_transactions(self) -> List[str]:
        """Get list of wallet addresses that have transactions"""
        logger.info("Fetching wallets with transaction data...")

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT wallet_address
                FROM transactions
                ORDER BY wallet_address
            """)

            wallets = [row[0] for row in cur.fetchall()]

        logger.info(f"Found {len(wallets)} wallets with transactions")
        return wallets

    def load_wallet_transactions(self, wallet_address: str) -> List[Trade]:
        """Load all transactions for a wallet and convert to Trade objects"""

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT
                    tx_hash as transaction_hash,
                    timestamp,
                    token_in,
                    token_out,
                    amount_in,
                    amount_out,
                    eth_value_in,
                    eth_value_out,
                    usd_value_in,
                    usd_value_out,
                    gas_used,
                    gas_price_gwei
                FROM transactions
                WHERE wallet_address = %s
                ORDER BY timestamp
            """, (wallet_address,))

            transactions = cur.fetchall()

        trades = []

        for tx in transactions:
            # Create buy trade for token_out (token received)
            if tx['token_out'] and tx['amount_out']:
                # Use a simple token identifier - in production you'd join with tokens table
                token_symbol = tx['token_out'][-8:]  # Last 8 chars of address as identifier

                trades.append(Trade(
                    timestamp=tx['timestamp'],
                    token_address=tx['token_out'],
                    token_symbol=token_symbol,
                    is_buy=True,
                    amount=Decimal(str(tx['amount_out'])) if tx['amount_out'] else Decimal(0),
                    price_eth=Decimal(str(tx['eth_value_out'])) if tx['eth_value_out'] else Decimal(0),
                    price_usd=Decimal(str(tx['usd_value_out'])) if tx['usd_value_out'] else Decimal(0),
                    gas_used=tx['gas_used'] or 0,
                    gas_price_gwei=Decimal(str(tx['gas_price_gwei'])) if tx['gas_price_gwei'] else Decimal(0),
                    transaction_hash=tx['transaction_hash']
                ))

            # Create sell trade for token_in (token given)
            if tx['token_in'] and tx['amount_in']:
                token_symbol = tx['token_in'][-8:]

                trades.append(Trade(
                    timestamp=tx['timestamp'],
                    token_address=tx['token_in'],
                    token_symbol=token_symbol,
                    is_buy=False,
                    amount=Decimal(str(tx['amount_in'])) if tx['amount_in'] else Decimal(0),
                    price_eth=Decimal(str(tx['eth_value_in'])) if tx['eth_value_in'] else Decimal(0),
                    price_usd=Decimal(str(tx['usd_value_in'])) if tx['usd_value_in'] else Decimal(0),
                    gas_used=tx['gas_used'] or 0,
                    gas_price_gwei=Decimal(str(tx['gas_price_gwei'])) if tx['gas_price_gwei'] else Decimal(0),
                    transaction_hash=tx['transaction_hash']
                ))

        return trades

    def save_performance_metrics(self, metrics: PerformanceMetrics):
        """Save performance metrics to database"""

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO wallet_performance (
                    wallet_address, calculation_date, time_period,
                    total_trades, win_rate, avg_return_per_trade, total_return, annualized_return,
                    volatility, sharpe_ratio, sortino_ratio, max_drawdown, var_95, calmar_ratio,
                    total_gas_cost_usd, volume_per_gas, net_return_after_costs,
                    unique_tokens_traded, hhi_concentration, max_position_size,
                    avg_holding_period_days, profit_factor
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT (wallet_address, time_period)
                DO UPDATE SET
                    calculation_date = EXCLUDED.calculation_date,
                    total_trades = EXCLUDED.total_trades,
                    win_rate = EXCLUDED.win_rate,
                    avg_return_per_trade = EXCLUDED.avg_return_per_trade,
                    total_return = EXCLUDED.total_return,
                    annualized_return = EXCLUDED.annualized_return,
                    volatility = EXCLUDED.volatility,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    sortino_ratio = EXCLUDED.sortino_ratio,
                    max_drawdown = EXCLUDED.max_drawdown,
                    var_95 = EXCLUDED.var_95,
                    calmar_ratio = EXCLUDED.calmar_ratio,
                    total_gas_cost_usd = EXCLUDED.total_gas_cost_usd,
                    volume_per_gas = EXCLUDED.volume_per_gas,
                    net_return_after_costs = EXCLUDED.net_return_after_costs,
                    unique_tokens_traded = EXCLUDED.unique_tokens_traded,
                    hhi_concentration = EXCLUDED.hhi_concentration,
                    max_position_size = EXCLUDED.max_position_size,
                    avg_holding_period_days = EXCLUDED.avg_holding_period_days,
                    profit_factor = EXCLUDED.profit_factor,
                    updated_at = NOW()
            """, (
                metrics.wallet_address, metrics.calculation_date, metrics.time_period,
                metrics.total_trades, metrics.win_rate, metrics.avg_return_per_trade,
                metrics.total_return, metrics.annualized_return,
                metrics.volatility, metrics.sharpe_ratio, metrics.sortino_ratio,
                metrics.max_drawdown, metrics.var_95, metrics.calmar_ratio,
                metrics.total_gas_cost_usd, metrics.volume_per_gas, metrics.net_return_after_costs,
                metrics.unique_tokens_traded, metrics.hhi_concentration, metrics.max_position_size,
                metrics.avg_holding_period_days, metrics.profit_factor
            ))

    def calculate_wallet_performance(self, wallet_address: str, eth_prices: Dict) -> bool:
        """Calculate performance metrics for a single wallet"""
        try:
            # Load wallet transactions
            trades = self.load_wallet_transactions(wallet_address)

            if not trades:
                logger.warning(f"No trades found for wallet {wallet_address}")
                return False

            # Calculate performance metrics
            metrics = self.calculator.calculate_wallet_performance(
                wallet_address=wallet_address,
                trades=trades,
                eth_prices=eth_prices,
                time_period=self.time_period
            )

            # Save to database
            self.save_performance_metrics(metrics)

            logger.info(
                f"✅ {wallet_address}: "
                f"Trades={metrics.total_trades}, "
                f"WinRate={metrics.win_rate:.2%}, "
                f"Return={metrics.total_return:.2%}, "
                f"Sharpe={metrics.sharpe_ratio:.2f if metrics.sharpe_ratio else 'N/A'}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Failed to calculate performance for {wallet_address}: {e}")
            return False

    def run(self):
        """Main execution flow"""
        logger.info("=" * 80)
        logger.info("WALLET PERFORMANCE CALCULATION STARTED")
        logger.info("=" * 80)

        try:
            # Load ETH prices
            eth_prices = self.load_eth_prices()

            # Get wallets with transactions
            wallets = self.get_wallets_with_transactions()
            total_wallets = len(wallets)

            logger.info(f"\nProcessing {total_wallets} wallets in batches of {self.batch_size}")
            logger.info(f"Time period: {self.time_period}\n")

            # Process wallets in batches
            for i in range(0, total_wallets, self.batch_size):
                batch = wallets[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (total_wallets + self.batch_size - 1) // self.batch_size

                logger.info(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} wallets)")

                for wallet in batch:
                    success = self.calculate_wallet_performance(wallet, eth_prices)

                    if success:
                        self.total_wallets_processed += 1
                    else:
                        self.total_wallets_failed += 1

                # Commit batch
                self.conn.commit()
                logger.info(f"✓ Batch {batch_num} committed to database")

            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("PERFORMANCE CALCULATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Total wallets processed: {self.total_wallets_processed}")
            logger.info(f"Total wallets failed: {self.total_wallets_failed}")
            logger.info(f"Success rate: {self.total_wallets_processed/total_wallets*100:.1f}%")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Fatal error during execution: {e}", exc_info=True)
            self.conn.rollback()
            raise

        finally:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate wallet performance metrics from transaction data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of wallets to process per batch (default: 100)'
    )
    parser.add_argument(
        '--time-period',
        type=str,
        default='all_time',
        choices=['30d', '90d', 'all_time'],
        help='Time period for analysis (default: all_time)'
    )

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Run calculation
    cli = PerformanceCalculationCLI(
        batch_size=args.batch_size,
        time_period=args.time_period
    )
    cli.run()


if __name__ == '__main__':
    main()
