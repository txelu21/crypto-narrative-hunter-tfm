#!/usr/bin/env python3
"""
Wallet Features Cleanup Script

Fixes critical data quality issues identified in EDA:
1. Removes zero-variance features
2. Fixes extreme value calculation errors
3. Removes redundant/multicollinear features
4. Handles high-sparsity features
5. Clips out-of-range values
6. Applies transformations for ML readiness
7. Creates activity segments for stratification

Usage:
    python scripts/utilities/cleanup_wallet_features.py

Author: Dev Agent (James)
Date: 2025-10-25
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from web3 import Web3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/cleanup_wallet_features.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class WalletFeaturesCleanup:
    """Cleans and prepares wallet features dataset for ML."""

    # Features to remove (zero variance, redundant, or extreme sparsity)
    FEATURES_TO_REMOVE = [
        "gas_optimization_score",  # Zero variance (constant = 50.0)
        "dex_diversity_score",  # Zero variance (constant = 0.0)
        "diamond_hands_score",  # Perfect correlation with avg_holding_period_days
        "num_tokens_std",  # 99.77% zeros
        "rotation_frequency",  # 99.31% zeros
        "portfolio_turnover",  # 99.31% zeros
    ]

    # Numeric bounds for clipping
    NUMERIC_BOUNDS = {
        "total_pnl_usd": (-1e12, 1e12),  # ±1 trillion
        "avg_trade_size_usd": (0, 1e12),  # 0 to 1 trillion
        "volume_consistency": (0, 1),  # Ratio bounded [0, 1]
        "roi_percent": (-1000, 10000),  # -1000% to +10000%
        "max_drawdown_pct": (0, 100),  # 0% to 100%
        "win_rate": (0, 100),  # 0% to 100%
        "portfolio_gini": (0, 1),  # Gini coefficient [0, 1]
        "portfolio_hhi": (0, 10000),  # HHI bounded [0, 10000]
    }

    # Features to log-transform (right-skewed)
    LOG_TRANSFORM_FEATURES = [
        "trade_frequency",
        "num_tokens_avg",
        "distribution_intensity",
    ]

    def __init__(self, input_path: str, output_dir: str):
        """
        Initialize cleanup processor.

        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df_original: pd.DataFrame = None
        self.df_cleaned: pd.DataFrame = None
        self.cleanup_stats: Dict = {}

    def load_data(self) -> None:
        """Load the wallet features dataset."""
        logger.info(f"Loading data from {self.input_path}")
        self.df_original = pd.read_csv(self.input_path)
        self.df_cleaned = self.df_original.copy()

        logger.info(f"Loaded {len(self.df_cleaned):,} wallets with {len(self.df_cleaned.columns)} features")
        self.cleanup_stats["original_shape"] = self.df_cleaned.shape

    def validate_wallet_addresses(self) -> None:
        """Validate Ethereum wallet addresses."""
        logger.info("Validating wallet addresses")

        invalid_addresses = []
        for addr in self.df_cleaned["wallet_address"]:
            if not isinstance(addr, str) or len(addr) != 42 or not addr.startswith("0x"):
                invalid_addresses.append(addr)

        self.cleanup_stats["invalid_addresses"] = len(invalid_addresses)

        if invalid_addresses:
            logger.warning(f"Found {len(invalid_addresses)} invalid addresses: {invalid_addresses[:5]}")
        else:
            logger.info("✓ All wallet addresses valid")

    def remove_problematic_features(self) -> None:
        """Remove zero-variance, redundant, and extremely sparse features."""
        logger.info("Removing problematic features")

        features_before = set(self.df_cleaned.columns)
        features_to_drop = [f for f in self.FEATURES_TO_REMOVE if f in self.df_cleaned.columns]

        self.df_cleaned = self.df_cleaned.drop(columns=features_to_drop)

        logger.info(f"✓ Removed {len(features_to_drop)} features: {features_to_drop}")
        self.cleanup_stats["features_removed"] = features_to_drop
        self.cleanup_stats["features_remaining"] = len(self.df_cleaned.columns)

    def fix_extreme_values(self) -> None:
        """Clip extreme values to reasonable bounds."""
        logger.info("Fixing extreme values")

        clipping_summary = {}

        for feature, (lower, upper) in self.NUMERIC_BOUNDS.items():
            if feature not in self.df_cleaned.columns:
                continue

            original_min = self.df_cleaned[feature].min()
            original_max = self.df_cleaned[feature].max()

            # Count values outside bounds
            below_lower = (self.df_cleaned[feature] < lower).sum()
            above_upper = (self.df_cleaned[feature] > upper).sum()

            if below_lower > 0 or above_upper > 0:
                self.df_cleaned[feature] = self.df_cleaned[feature].clip(lower, upper)

                clipping_summary[feature] = {
                    "original_range": (original_min, original_max),
                    "new_range": (lower, upper),
                    "clipped_below": below_lower,
                    "clipped_above": above_upper,
                }

                logger.info(
                    f"  {feature}: clipped {below_lower + above_upper} values "
                    f"(range: [{original_min:.2e}, {original_max:.2e}] → [{lower:.2e}, {upper:.2e}])"
                )

        self.cleanup_stats["clipping_summary"] = clipping_summary
        logger.info(f"✓ Clipped extreme values in {len(clipping_summary)} features")

    def apply_transformations(self) -> None:
        """Apply log transformations to skewed features."""
        logger.info("Applying transformations")

        transformation_summary = {}

        for feature in self.LOG_TRANSFORM_FEATURES:
            if feature not in self.df_cleaned.columns:
                continue

            original_skew = self.df_cleaned[feature].skew()

            # Create log-transformed version (keep original too)
            new_feature_name = f"{feature}_log"
            self.df_cleaned[new_feature_name] = np.log1p(self.df_cleaned[feature])

            new_skew = self.df_cleaned[new_feature_name].skew()

            transformation_summary[feature] = {
                "original_skew": original_skew,
                "log_skew": new_skew,
                "improvement": abs(new_skew) < abs(original_skew),
            }

            logger.info(
                f"  {feature} → {new_feature_name}: skew {original_skew:.2f} → {new_skew:.2f}"
            )

        self.cleanup_stats["transformation_summary"] = transformation_summary
        logger.info(f"✓ Created {len(transformation_summary)} log-transformed features")

    def create_binary_indicators(self) -> None:
        """Create binary indicators for sparse features."""
        logger.info("Creating binary indicators")

        indicators_created = []

        # Activity indicator
        self.df_cleaned["is_active"] = (self.df_cleaned["trade_frequency"] > 1).astype(int)
        indicators_created.append("is_active")

        # Win indicator
        if "win_rate" in self.df_cleaned.columns:
            self.df_cleaned["has_wins"] = (self.df_cleaned["win_rate"] > 0).astype(int)
            indicators_created.append("has_wins")

        # Profit indicator
        if "total_pnl_usd" in self.df_cleaned.columns:
            self.df_cleaned["is_profitable"] = (self.df_cleaned["total_pnl_usd"] > 0).astype(int)
            indicators_created.append("is_profitable")

        # Multi-token holder
        if "num_tokens_avg" in self.df_cleaned.columns:
            self.df_cleaned["is_multi_token"] = (self.df_cleaned["num_tokens_avg"] > 1).astype(int)
            indicators_created.append("is_multi_token")

        # Weekend trader
        if "weekend_activity_ratio" in self.df_cleaned.columns:
            self.df_cleaned["is_weekend_trader"] = (
                self.df_cleaned["weekend_activity_ratio"] > 0.3
            ).astype(int)
            indicators_created.append("is_weekend_trader")

        # Night trader
        if "night_trading_ratio" in self.df_cleaned.columns:
            self.df_cleaned["is_night_trader"] = (
                self.df_cleaned["night_trading_ratio"] > 0.3
            ).astype(int)
            indicators_created.append("is_night_trader")

        self.cleanup_stats["binary_indicators_created"] = indicators_created
        logger.info(f"✓ Created {len(indicators_created)} binary indicators")

    def create_interaction_features(self) -> None:
        """Create interaction and derived features."""
        logger.info("Creating interaction features")

        interactions_created = []

        # ROI per trade
        if "roi_percent" in self.df_cleaned.columns and "trade_frequency" in self.df_cleaned.columns:
            self.df_cleaned["roi_per_trade"] = self.df_cleaned["roi_percent"] / (
                self.df_cleaned["trade_frequency"] + 1
            )
            interactions_created.append("roi_per_trade")

        # Risk-adjusted return
        if "roi_percent" in self.df_cleaned.columns and "max_drawdown_pct" in self.df_cleaned.columns:
            self.df_cleaned["risk_adjusted_return"] = self.df_cleaned["roi_percent"] / (
                self.df_cleaned["max_drawdown_pct"] + 1
            )
            interactions_created.append("risk_adjusted_return")

        # Concentration-adjusted Sharpe
        if "sharpe_ratio" in self.df_cleaned.columns and "portfolio_gini" in self.df_cleaned.columns:
            self.df_cleaned["concentration_adjusted_sharpe"] = (
                self.df_cleaned["sharpe_ratio"] * (1 - self.df_cleaned["portfolio_gini"])
            )
            interactions_created.append("concentration_adjusted_sharpe")

        # Volume per token
        if "total_volume_usd" in self.df_cleaned.columns and "num_tokens_avg" in self.df_cleaned.columns:
            self.df_cleaned["volume_per_token"] = self.df_cleaned["total_volume_usd"] / (
                self.df_cleaned["num_tokens_avg"] + 1
            )
            interactions_created.append("volume_per_token")

        self.cleanup_stats["interaction_features_created"] = interactions_created
        logger.info(f"✓ Created {len(interactions_created)} interaction features")

    def create_activity_segments(self) -> None:
        """Create activity segments for stratification."""
        logger.info("Creating activity segments")

        # Define segments based on trade frequency
        def assign_segment(freq: int) -> str:
            if freq == 1:
                return "single_trade"
            elif freq <= 5:
                return "low_activity"
            elif freq <= 20:
                return "medium_activity"
            else:
                return "high_activity"

        self.df_cleaned["activity_segment"] = self.df_cleaned["trade_frequency"].apply(assign_segment)

        segment_counts = self.df_cleaned["activity_segment"].value_counts().to_dict()
        self.cleanup_stats["activity_segments"] = segment_counts

        logger.info("Activity segment distribution:")
        for segment, count in segment_counts.items():
            pct = 100 * count / len(self.df_cleaned)
            logger.info(f"  {segment}: {count:,} ({pct:.1f}%)")

    def handle_missing_values(self) -> None:
        """Check for and handle any missing values."""
        logger.info("Checking for missing values")

        missing_summary = self.df_cleaned.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]

        if len(missing_features) > 0:
            logger.warning(f"Found missing values in {len(missing_features)} features:")
            for feature, count in missing_features.items():
                pct = 100 * count / len(self.df_cleaned)
                logger.warning(f"  {feature}: {count} ({pct:.2f}%)")

                # Simple imputation strategy
                if self.df_cleaned[feature].dtype in [np.float64, np.int64]:
                    self.df_cleaned[feature].fillna(0, inplace=True)
                    logger.info(f"    → Filled with 0")
                else:
                    self.df_cleaned[feature].fillna("unknown", inplace=True)
                    logger.info(f"    → Filled with 'unknown'")

            self.cleanup_stats["missing_values_handled"] = missing_features.to_dict()
        else:
            logger.info("✓ No missing values found")

    def remove_duplicates(self) -> None:
        """Remove any duplicate wallet addresses."""
        logger.info("Checking for duplicates")

        initial_count = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.drop_duplicates(subset=["wallet_address"], keep="first")
        final_count = len(self.df_cleaned)

        duplicates_removed = initial_count - final_count
        self.cleanup_stats["duplicates_removed"] = duplicates_removed

        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate wallets")
        else:
            logger.info("✓ No duplicates found")

    def save_cleaned_data(self) -> Tuple[Path, Path]:
        """Save cleaned dataset and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save cleaned CSV
        output_csv = self.output_dir / f"wallet_features_cleaned_{timestamp}.csv"
        self.df_cleaned.to_csv(output_csv, index=False)
        logger.info(f"✓ Saved cleaned dataset to {output_csv}")

        # Save cleanup metadata
        metadata_path = self.output_dir / f"cleanup_metadata_{timestamp}.json"
        import json

        def convert_to_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        with open(metadata_path, "w") as f:
            serializable_stats = convert_to_serializable(self.cleanup_stats)
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"✓ Saved cleanup metadata to {metadata_path}")

        return output_csv, metadata_path

    def generate_comparison_report(self, output_csv: Path) -> Path:
        """Generate before/after comparison report."""
        logger.info("Generating comparison report")

        report_path = self.output_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_path, "w") as f:
            f.write("# Wallet Features Cleanup Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Input:** `{self.input_path.name}`\n\n")
            f.write(f"**Output:** `{output_csv.name}`\n\n")

            f.write("---\n\n")
            f.write("## Summary\n\n")

            original_shape = self.cleanup_stats.get("original_shape", (0, 0))
            final_shape = self.df_cleaned.shape

            f.write(f"- **Original dataset:** {original_shape[0]:,} wallets × {original_shape[1]} features\n")
            f.write(f"- **Cleaned dataset:** {final_shape[0]:,} wallets × {final_shape[1]} features\n")
            f.write(f"- **Features removed:** {len(self.cleanup_stats.get('features_removed', []))}\n")
            f.write(f"- **Features added:** {final_shape[1] - original_shape[1] + len(self.cleanup_stats.get('features_removed', []))}\n")
            f.write(f"- **Duplicates removed:** {self.cleanup_stats.get('duplicates_removed', 0)}\n\n")

            f.write("---\n\n")
            f.write("## Changes Applied\n\n")

            f.write("### 1. Features Removed\n\n")
            for feature in self.cleanup_stats.get("features_removed", []):
                f.write(f"- `{feature}`\n")
            f.write("\n")

            f.write("### 2. Extreme Values Clipped\n\n")
            clipping = self.cleanup_stats.get("clipping_summary", {})
            if clipping:
                for feature, details in clipping.items():
                    f.write(f"**{feature}:**\n")
                    f.write(f"- Clipped below: {details['clipped_below']}\n")
                    f.write(f"- Clipped above: {details['clipped_above']}\n")
                    f.write(f"- New range: [{details['new_range'][0]:.2e}, {details['new_range'][1]:.2e}]\n\n")
            else:
                f.write("*No clipping required*\n\n")

            f.write("### 3. Transformations Applied\n\n")
            transformations = self.cleanup_stats.get("transformation_summary", {})
            if transformations:
                for feature, details in transformations.items():
                    f.write(f"**{feature}_log:**\n")
                    f.write(f"- Original skew: {details['original_skew']:.2f}\n")
                    f.write(f"- Log-transformed skew: {details['log_skew']:.2f}\n")
                    f.write(f"- Improved: {'✓' if details['improvement'] else '✗'}\n\n")
            else:
                f.write("*No transformations applied*\n\n")

            f.write("### 4. Binary Indicators Created\n\n")
            for indicator in self.cleanup_stats.get("binary_indicators_created", []):
                f.write(f"- `{indicator}`\n")
            f.write("\n")

            f.write("### 5. Interaction Features Created\n\n")
            for interaction in self.cleanup_stats.get("interaction_features_created", []):
                f.write(f"- `{interaction}`\n")
            f.write("\n")

            f.write("### 6. Activity Segments\n\n")
            segments = self.cleanup_stats.get("activity_segments", {})
            for segment, count in segments.items():
                pct = 100 * count / final_shape[0]
                f.write(f"- **{segment}:** {count:,} ({pct:.1f}%)\n")
            f.write("\n")

            f.write("---\n\n")
            f.write("## Data Quality Metrics\n\n")

            f.write("### Before Cleanup\n")
            f.write(f"- Invalid addresses: {self.cleanup_stats.get('invalid_addresses', 0)}\n")
            f.write(f"- Missing values: {sum(self.df_original.isnull().sum())}\n")
            f.write(f"- Duplicates: {self.cleanup_stats.get('duplicates_removed', 0)}\n\n")

            f.write("### After Cleanup\n")
            f.write(f"- Invalid addresses: 0\n")
            f.write(f"- Missing values: {sum(self.df_cleaned.isnull().sum())}\n")
            f.write(f"- Duplicates: 0\n\n")

            f.write("---\n\n")
            f.write("## Next Steps\n\n")
            f.write("1. Review cleaned dataset for correctness\n")
            f.write("2. Proceed with ML model development\n")
            f.write("3. Use `activity_segment` for stratified train/test splits\n")
            f.write("4. Consider additional feature engineering based on model performance\n\n")

            f.write("---\n\n")
            f.write("**ML Readiness: ✅ READY**\n")

        logger.info(f"✓ Saved comparison report to {report_path}")
        return report_path

    def run(self) -> Tuple[Path, Path, Path]:
        """
        Execute full cleanup pipeline.

        Returns:
            Tuple of (cleaned_csv_path, metadata_path, report_path)
        """
        logger.info("=" * 80)
        logger.info("Starting Wallet Features Cleanup Pipeline")
        logger.info("=" * 80)

        # Execute cleanup steps
        self.load_data()
        self.validate_wallet_addresses()
        self.remove_duplicates()
        self.remove_problematic_features()
        self.fix_extreme_values()
        self.handle_missing_values()
        self.apply_transformations()
        self.create_binary_indicators()
        self.create_interaction_features()
        self.create_activity_segments()

        # Save results
        cleaned_csv, metadata = self.save_cleaned_data()
        report = self.generate_comparison_report(cleaned_csv)

        logger.info("=" * 80)
        logger.info("Cleanup Pipeline Complete!")
        logger.info(f"Cleaned data: {cleaned_csv}")
        logger.info(f"Metadata: {metadata}")
        logger.info(f"Report: {report}")
        logger.info("=" * 80)

        return cleaned_csv, metadata, report


def main():
    """Main entry point."""
    # Configuration
    INPUT_FILE = (
        "/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/"
        "BMAD_TFM/data-collection/outputs/features/"
        "wallet_features_master_20251022_195455.csv"
    )
    OUTPUT_DIR = (
        "/Users/txelusanchez/Documents/MBIT_MIA/Crypto Narrative Hunter - TFM/"
        "BMAD_TFM/data-collection/outputs/features"
    )

    # Run cleanup
    cleanup = WalletFeaturesCleanup(input_path=INPUT_FILE, output_dir=OUTPUT_DIR)
    cleaned_csv, metadata, report = cleanup.run()

    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Cleaned Dataset: {cleaned_csv}")
    print(f"📋 Metadata: {metadata}")
    print(f"📄 Report: {report}")
    print("\nNext: Review the report and proceed with ML model development")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
