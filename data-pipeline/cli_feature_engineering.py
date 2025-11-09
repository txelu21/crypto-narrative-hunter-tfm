"""
CLI for Wallet Feature Engineering
Epic 4, Story 4.1

Usage:
    python cli_feature_engineering.py --category performance
    python cli_feature_engineering.py --category all --output wallet_features.csv
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from data_collection.common.logging_setup import setup_logging, get_logger
from services.feature_engineering.performance_calculator_v2 import calculate_performance_metrics_batch
from services.feature_engineering.behavioral_analyzer import calculate_behavioral_metrics_batch
from services.feature_engineering.concentration_calculator import calculate_concentration_metrics_batch
from services.feature_engineering.narrative_analyzer import calculate_narrative_metrics_batch
from services.feature_engineering.accumulation_detector import calculate_accumulation_metrics_batch

# Setup logging
setup_logging()
logger = get_logger(__name__)


class FeatureEngineeringCLI:
    """CLI for generating wallet features"""

    def __init__(self):
        self.data_dir = Path("outputs/csv")
        self.output_dir = Path("outputs/features")
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load all required datasets"""
        logger.info("Loading datasets...")

        try:
            # Load wallets
            self.wallets_df = pd.read_csv(self.data_dir / "wallets.csv")
            logger.info(f"Loaded {len(self.wallets_df)} wallets")

            # Load transactions
            self.transactions_df = pd.read_csv(self.data_dir / "transactions.csv")
            logger.info(f"Loaded {len(self.transactions_df)} transactions")

            # Load balance snapshots
            logger.info("Loading balance snapshots (this may take a minute)...")
            self.balances_df = pd.read_csv(self.data_dir / "wallet_token_balances.csv")
            logger.info(f"Loaded {len(self.balances_df)} balance snapshots")

            # Load ETH prices
            self.eth_prices_df = pd.read_csv(self.data_dir / "eth_prices.csv")
            logger.info(f"Loaded {len(self.eth_prices_df)} ETH prices")

            # Load tokens
            self.tokens_df = pd.read_csv(self.data_dir / "tokens.csv")
            logger.info(f"Loaded {len(self.tokens_df)} tokens")

            logger.info("✓ All datasets loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def generate_performance_features(self, output_file: str = None):
        """Generate Category 1: Performance Metrics"""
        logger.info("=== Generating Category 1: Performance Metrics ===")
        logger.info("Features: ROI, Win Rate, Sharpe Ratio, Max Drawdown, Total PnL, Avg Trade Size, Volume Consistency")

        # Get unique wallet addresses
        wallet_addresses = self.wallets_df['wallet_address'].unique().tolist()
        logger.info(f"Processing {len(wallet_addresses)} wallets")

        # Calculate metrics
        performance_df = calculate_performance_metrics_batch(
            wallet_addresses=wallet_addresses,
            balances_df=self.balances_df,
            transactions_df=self.transactions_df,
            eth_prices_df=self.eth_prices_df,
            tokens_df=self.tokens_df
        )

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_features_{timestamp}.csv"

        output_path = self.output_dir / output_file
        performance_df.to_csv(output_path, index=False)

        logger.info(f"✓ Performance features saved to: {output_path}")
        logger.info(f"✓ Generated {len(performance_df.columns)-1} features for {len(performance_df)} wallets")

        # Display summary statistics
        self._display_summary(performance_df, "Performance Metrics")

        return performance_df

    def generate_behavioral_features(self, output_file: str = None):
        """Generate Category 2: Behavioral Features"""
        logger.info("=== Generating Category 2: Behavioral Features ===")
        logger.info("Features: Trade Frequency, Holding Period, Diamond Hands, Rotation, Weekend/Night Activity, Gas Optimization, DEX Diversity")

        # Get unique wallet addresses
        wallet_addresses = self.wallets_df['wallet_address'].unique().tolist()
        logger.info(f"Processing {len(wallet_addresses)} wallets")

        # Calculate metrics
        behavioral_df = calculate_behavioral_metrics_batch(
            wallet_addresses=wallet_addresses,
            transactions_df=self.transactions_df,
            balances_df=self.balances_df
        )

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"behavioral_features_{timestamp}.csv"

        output_path = self.output_dir / output_file
        behavioral_df.to_csv(output_path, index=False)

        logger.info(f"✓ Behavioral features saved to: {output_path}")
        logger.info(f"✓ Generated {len(behavioral_df.columns)-1} features for {len(behavioral_df)} wallets")

        # Display summary statistics
        self._display_summary(behavioral_df, "Behavioral Metrics")

        return behavioral_df

    def generate_concentration_features(self, output_file: str = None):
        """Generate Category 3: Portfolio Concentration Features"""
        logger.info("=== Generating Category 3: Portfolio Concentration Features ===")
        logger.info("Features: HHI, Gini, Top3 Concentration, Token Count Stats, Portfolio Turnover")

        # Get unique wallet addresses
        wallet_addresses = self.wallets_df['wallet_address'].unique().tolist()
        logger.info(f"Processing {len(wallet_addresses)} wallets")

        # Calculate metrics
        concentration_df = calculate_concentration_metrics_batch(
            wallet_addresses=wallet_addresses,
            balances_df=self.balances_df
        )

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"concentration_features_{timestamp}.csv"

        output_path = self.output_dir / output_file
        concentration_df.to_csv(output_path, index=False)

        logger.info(f"✓ Concentration features saved to: {output_path}")
        logger.info(f"✓ Generated {len(concentration_df.columns)-1} features for {len(concentration_df)} wallets")

        # Display summary statistics
        self._display_summary(concentration_df, "Concentration Metrics")

        return concentration_df

    def generate_narrative_features(self, output_file: str = None):
        """Generate Category 4: Narrative Exposure Features"""
        logger.info("=== Generating Category 4: Narrative Exposure Features ===")
        logger.info("Features: Narrative Diversity, Primary Narrative %, DeFi/AI/Meme Exposure, Stablecoin Usage")

        # Get unique wallet addresses
        wallet_addresses = self.wallets_df['wallet_address'].unique().tolist()
        logger.info(f"Processing {len(wallet_addresses)} wallets")

        # Calculate metrics
        narrative_df = calculate_narrative_metrics_batch(
            wallet_addresses=wallet_addresses,
            balances_df=self.balances_df,
            tokens_df=self.tokens_df
        )

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"narrative_features_{timestamp}.csv"

        output_path = self.output_dir / output_file
        narrative_df.to_csv(output_path, index=False)

        logger.info(f"✓ Narrative features saved to: {output_path}")
        logger.info(f"✓ Generated {len(narrative_df.columns)-1} features for {len(narrative_df)} wallets")

        # Display summary statistics
        self._display_summary(narrative_df, "Narrative Metrics")

        return narrative_df

    def generate_accumulation_features(self, output_file: str = None):
        """Generate Category 5: Accumulation/Distribution Features"""
        logger.info("=== Generating Category 5: Accumulation/Distribution Features ===")
        logger.info("Features: Accumulation/Distribution Days & Intensity, Balance Volatility, Trend Direction")

        # Get unique wallet addresses
        wallet_addresses = self.wallets_df['wallet_address'].unique().tolist()
        logger.info(f"Processing {len(wallet_addresses)} wallets")

        # Calculate metrics
        accumulation_df = calculate_accumulation_metrics_batch(
            wallet_addresses=wallet_addresses,
            balances_df=self.balances_df
        )

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"accumulation_features_{timestamp}.csv"

        output_path = self.output_dir / output_file
        accumulation_df.to_csv(output_path, index=False)

        logger.info(f"✓ Accumulation features saved to: {output_path}")
        logger.info(f"✓ Generated {len(accumulation_df.columns)-1} features for {len(accumulation_df)} wallets")

        # Display summary statistics
        self._display_summary(accumulation_df, "Accumulation Metrics")

        return accumulation_df

    def _display_summary(self, df: pd.DataFrame, category: str):
        """Display summary statistics for features"""
        logger.info(f"\n=== {category} Summary ===")

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'wallet_address']

        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe()
            logger.info(f"\n{summary.to_string()}")

            # Check for missing values
            missing = df[numeric_cols].isna().sum()
            if missing.any():
                logger.warning(f"\nMissing values:\n{missing[missing > 0]}")
            else:
                logger.info("\n✓ No missing values detected")


def main():
    parser = argparse.ArgumentParser(
        description="Generate wallet features for clustering analysis"
    )

    parser.add_argument(
        "--category",
        choices=["performance", "behavioral", "concentration", "narrative", "accumulation", "all"],
        default="performance",
        help="Feature category to generate"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (default: auto-generated with timestamp)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of wallets (for testing)"
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = FeatureEngineeringCLI()

    # Load data
    if not cli.load_data():
        logger.error("Failed to load data. Exiting.")
        return 1

    # Apply limit if specified (for testing)
    if args.limit:
        logger.info(f"Limiting to first {args.limit} wallets for testing")
        cli.wallets_df = cli.wallets_df.head(args.limit)

    # Generate features based on category
    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE ENGINEERING - Category: {args.category.upper()}")
    logger.info(f"{'='*60}\n")

    if args.category == "performance" or args.category == "all":
        performance_df = cli.generate_performance_features(args.output)

        if args.category == "performance":
            logger.info("\n✅ Category 1 (Performance Metrics) COMPLETE")
            logger.info(f"Output: {cli.output_dir / (args.output or 'performance_features_*.csv')}")

    if args.category == "behavioral" or args.category == "all":
        behavioral_df = cli.generate_behavioral_features(args.output)

        if args.category == "behavioral":
            logger.info("\n✅ Category 2 (Behavioral Features) COMPLETE")
            logger.info(f"Output: {cli.output_dir / (args.output or 'behavioral_features_*.csv')}")

    if args.category == "concentration" or args.category == "all":
        concentration_df = cli.generate_concentration_features(args.output)

        if args.category == "concentration":
            logger.info("\n✅ Category 3 (Portfolio Concentration) COMPLETE")
            logger.info(f"Output: {cli.output_dir / (args.output or 'concentration_features_*.csv')}")

    if args.category == "narrative" or args.category == "all":
        narrative_df = cli.generate_narrative_features(args.output)

        if args.category == "narrative":
            logger.info("\n✅ Category 4 (Narrative Exposure) COMPLETE")
            logger.info(f"Output: {cli.output_dir / (args.output or 'narrative_features_*.csv')}")

    if args.category == "accumulation" or args.category == "all":
        accumulation_df = cli.generate_accumulation_features(args.output)

        if args.category == "accumulation":
            logger.info("\n✅ Category 5 (Accumulation/Distribution) COMPLETE")
            logger.info(f"Output: {cli.output_dir / (args.output or 'accumulation_features_*.csv')}")

    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
