"""
Combine Feature Files into Master Dataset
Merges all 5 category feature files into a single master dataset for clustering
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def combine_feature_files():
    """Combine all feature category files into master dataset"""

    features_dir = Path("outputs/features")

    # Load the latest file from each category
    logger.info("=" * 80)
    logger.info("COMBINING FEATURE FILES INTO MASTER DATASET")
    logger.info("=" * 80)
    logger.info("")

    # Category 1: Performance
    performance_files = sorted(features_dir.glob("performance_features_*.csv"))
    if not performance_files:
        raise FileNotFoundError("No performance features file found")
    performance_file = performance_files[-1]
    logger.info(f"📊 Category 1 (Performance): {performance_file.name}")
    performance_df = pd.read_csv(performance_file)
    logger.info(f"   Loaded: {len(performance_df)} wallets, {len(performance_df.columns)-1} features")

    # Category 2: Behavioral
    behavioral_files = sorted(features_dir.glob("behavioral_features_*.csv"))
    if not behavioral_files:
        raise FileNotFoundError("No behavioral features file found")
    behavioral_file = behavioral_files[-1]
    logger.info(f"📊 Category 2 (Behavioral): {behavioral_file.name}")
    behavioral_df = pd.read_csv(behavioral_file)
    logger.info(f"   Loaded: {len(behavioral_df)} wallets, {len(behavioral_df.columns)-1} features")

    # Category 3: Concentration
    concentration_files = sorted(features_dir.glob("concentration_features_*.csv"))
    if not concentration_files:
        raise FileNotFoundError("No concentration features file found")
    concentration_file = concentration_files[-1]
    logger.info(f"📊 Category 3 (Concentration): {concentration_file.name}")
    concentration_df = pd.read_csv(concentration_file)
    logger.info(f"   Loaded: {len(concentration_df)} wallets, {len(concentration_df.columns)-1} features")

    # Category 4: Narrative
    narrative_files = sorted(features_dir.glob("narrative_features_*.csv"))
    if not narrative_files:
        raise FileNotFoundError("No narrative features file found")
    narrative_file = narrative_files[-1]
    logger.info(f"📊 Category 4 (Narrative): {narrative_file.name}")
    narrative_df = pd.read_csv(narrative_file)
    logger.info(f"   Loaded: {len(narrative_df)} wallets, {len(narrative_df.columns)-1} features")

    # Category 5: Accumulation
    accumulation_files = sorted(features_dir.glob("accumulation_features_*.csv"))
    if not accumulation_files:
        raise FileNotFoundError("No accumulation features file found")
    accumulation_file = accumulation_files[-1]
    logger.info(f"📊 Category 5 (Accumulation): {accumulation_file.name}")
    accumulation_df = pd.read_csv(accumulation_file)
    logger.info(f"   Loaded: {len(accumulation_df)} wallets, {len(accumulation_df.columns)-1} features")

    logger.info("")
    logger.info("-" * 80)
    logger.info("MERGING DATASETS...")
    logger.info("-" * 80)

    # Merge all dataframes on wallet_address
    # Start with performance (largest dataset)
    master_df = performance_df.copy()
    logger.info(f"Starting with: {len(master_df)} wallets from Performance")

    # Merge behavioral
    master_df = master_df.merge(behavioral_df, on='wallet_address', how='inner')
    logger.info(f"After Behavioral merge: {len(master_df)} wallets")

    # Merge concentration
    master_df = master_df.merge(concentration_df, on='wallet_address', how='inner')
    logger.info(f"After Concentration merge: {len(master_df)} wallets")

    # Merge narrative
    master_df = master_df.merge(narrative_df, on='wallet_address', how='inner')
    logger.info(f"After Narrative merge: {len(master_df)} wallets")

    # Merge accumulation
    master_df = master_df.merge(accumulation_df, on='wallet_address', how='inner')
    logger.info(f"After Accumulation merge: {len(master_df)} wallets")

    logger.info("")
    logger.info("-" * 80)
    logger.info("MASTER DATASET SUMMARY")
    logger.info("-" * 80)
    logger.info(f"✓ Total wallets: {len(master_df)}")
    logger.info(f"✓ Total features: {len(master_df.columns) - 1} (excluding wallet_address)")
    logger.info(f"✓ Total columns: {len(master_df.columns)}")

    # Check for missing values
    missing = master_df.isna().sum().sum()
    if missing > 0:
        logger.warning(f"⚠️  Missing values detected: {missing}")
        logger.warning("\nMissing values by column:")
        missing_cols = master_df.isna().sum()
        for col, count in missing_cols[missing_cols > 0].items():
            logger.warning(f"  - {col}: {count} missing")
    else:
        logger.info("✓ No missing values detected")

    # Save master dataset in both CSV and Parquet formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    csv_file = features_dir / f"wallet_features_master_{timestamp}.csv"
    master_df.to_csv(csv_file, index=False)
    csv_size = csv_file.stat().st_size

    # Save Parquet
    parquet_file = features_dir / f"wallet_features_master_{timestamp}.parquet"
    master_df.to_parquet(parquet_file, engine='pyarrow', compression='snappy', index=False)
    parquet_size = parquet_file.stat().st_size

    # Calculate compression
    compression_ratio = (1 - parquet_size / csv_size) * 100

    logger.info("")
    logger.info("-" * 80)
    logger.info("✓ MASTER DATASET SAVED (DUAL FORMAT)")
    logger.info("-" * 80)
    logger.info(f"CSV:     {csv_file.name}")
    logger.info(f"         Size: {csv_size / 1024:.1f} KB")
    logger.info(f"Parquet: {parquet_file.name}")
    logger.info(f"         Size: {parquet_size / 1024:.1f} KB")
    logger.info(f"         Compression: {compression_ratio:.1f}% smaller")
    logger.info("-" * 80)

    # Display feature categories breakdown
    logger.info("")
    logger.info("FEATURE BREAKDOWN BY CATEGORY:")
    logger.info("")

    performance_cols = [col for col in performance_df.columns if col != 'wallet_address']
    behavioral_cols = [col for col in behavioral_df.columns if col != 'wallet_address']
    concentration_cols = [col for col in concentration_df.columns if col != 'wallet_address']
    narrative_cols = [col for col in narrative_df.columns if col != 'wallet_address']
    accumulation_cols = [col for col in accumulation_df.columns if col != 'wallet_address']

    logger.info(f"Category 1 - Performance ({len(performance_cols)} features):")
    for col in performance_cols:
        logger.info(f"  - {col}")

    logger.info(f"\nCategory 2 - Behavioral ({len(behavioral_cols)} features):")
    for col in behavioral_cols:
        logger.info(f"  - {col}")

    logger.info(f"\nCategory 3 - Concentration ({len(concentration_cols)} features):")
    for col in concentration_cols:
        logger.info(f"  - {col}")

    logger.info(f"\nCategory 4 - Narrative ({len(narrative_cols)} features):")
    for col in narrative_cols:
        logger.info(f"  - {col}")

    logger.info(f"\nCategory 5 - Accumulation ({len(accumulation_cols)} features):")
    for col in accumulation_cols:
        logger.info(f"  - {col}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ FEATURE ENGINEERING COMPLETE - READY FOR CLUSTERING")
    logger.info("=" * 80)

    return master_df, csv_file, parquet_file


if __name__ == "__main__":
    master_df, csv_file, parquet_file = combine_feature_files()
