#!/usr/bin/env python3
"""
Additional Token Collection Script
Expands token universe from 500 to 1,500 tokens (ranks 501-1500 by market cap).

This script:
1. Fetches tokens ranked 501-1500 from CoinGecko
2. Applies same narrative classification and liquidity tier logic
3. Merges with existing tokens.csv (avoiding duplicates)
4. Updates PostgreSQL database

Expected Runtime: ~3-5 minutes
"""

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.tokens.coingecko_client import CoinGeckoClient
from services.tokens.token_fetcher import TokenFetcher
from services.tokens.narrative_classifier import NarrativeClassifier
from data_collection.common.db import get_cursor
from data_collection.common.logging_setup import get_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = get_logger("cli_collect_additional_tokens")


def load_existing_tokens(csv_path: Path) -> pd.DataFrame:
    """Load existing tokens to avoid duplicates"""
    if csv_path.exists():
        logger.info(f"Loading existing tokens from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} existing tokens")
        return df
    else:
        logger.info("No existing tokens file found")
        return pd.DataFrame()


def fetch_additional_tokens(target_count: int = 1000, existing_count: int = 500):
    """
    Fetch additional tokens to expand universe

    Args:
        target_count: Number of new tokens to fetch (default: 1000)
        existing_count: Number of already collected tokens (default: 500)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Token Universe Expansion: {existing_count} → {existing_count + target_count} tokens")
    logger.info(f"{'='*60}\n")

    # Initialize services
    # Note: CoinGeckoClient reads API key from settings/environment
    client = CoinGeckoClient()
    fetcher = TokenFetcher(client)
    classifier = NarrativeClassifier()

    # Paths
    csv_path = Path(__file__).parent / "outputs" / "csv" / "tokens.csv"
    backup_path = csv_path.with_suffix('.csv.backup_before_expansion')

    # Load existing tokens
    existing_df = load_existing_tokens(csv_path)
    existing_addresses = set(existing_df['token_address'].values) if not existing_df.empty else set()
    logger.info(f"Existing token addresses: {len(existing_addresses)}")

    # Create backup
    if csv_path.exists() and not backup_path.exists():
        logger.info(f"Creating backup: {backup_path}")
        existing_df.to_csv(backup_path, index=False)

    # Fetch tokens (we'll fetch more than needed and filter to ranks 501-1500)
    # CoinGecko returns tokens sorted by market cap, so we fetch 1500 total and skip first 500
    logger.info(f"\nFetching top {existing_count + target_count} tokens from CoinGecko...")
    logger.info(f"(Will filter to ranks {existing_count + 1}-{existing_count + target_count})\n")

    all_tokens, fetch_stats = fetcher.fetch_top_ethereum_tokens(
        target_count=existing_count + target_count
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Fetch Statistics:")
    logger.info(f"  Total tokens fetched: {len(all_tokens)}")
    logger.info(f"  Pages fetched: {fetch_stats.total_pages_fetched}")
    logger.info(f"  Ethereum tokens found: {fetch_stats.ethereum_tokens_found}")
    logger.info(f"  Duration: {fetch_stats.duration_seconds:.1f}s")
    logger.info(f"{'='*60}\n")

    # Filter to ranks 501-1500 (new tokens only)
    new_tokens = [t for t in all_tokens if t.market_cap_rank and t.market_cap_rank > existing_count]
    logger.info(f"Filtered to {len(new_tokens)} tokens ranked >{existing_count}")

    # Remove tokens that already exist (by address)
    unique_new_tokens = [t for t in new_tokens if t.token_address not in existing_addresses]
    logger.info(f"Removed {len(new_tokens) - len(unique_new_tokens)} duplicates")
    logger.info(f"New unique tokens to add: {len(unique_new_tokens)}")

    if len(unique_new_tokens) == 0:
        logger.warning("No new tokens to add!")
        return False

    # Convert to DataFrame
    new_tokens_data = []
    for token in unique_new_tokens:
        # Classify narrative
        classification_result = classifier.classify_token(
            token_address=token.token_address,
            symbol=token.symbol,
            name=token.name
        )

        # Determine liquidity tier (using volume as proxy)
        volume_usd = float(token.volume_24h_usd) if token.volume_24h_usd else 0
        if volume_usd > 10_000_000:
            liquidity_tier = "Tier 1"
        elif volume_usd > 1_000_000:
            liquidity_tier = "Tier 2"
        elif volume_usd > 10_000:
            liquidity_tier = "Tier 3"
        else:
            liquidity_tier = None

        new_tokens_data.append({
            "token_address": token.token_address,
            "symbol": token.symbol,
            "name": token.name,
            "decimals": token.decimals,
            "narrative_category": classification_result.category.value,
            "market_cap_rank": token.market_cap_rank,
            "avg_daily_volume_usd": volume_usd,
            "liquidity_tier": liquidity_tier,
            "validation_status": "pending",
            "classification_confidence": classification_result.confidence,
            "requires_manual_review": classifier.is_manual_review_required(classification_result),
            "created_at": datetime.now()
        })

    new_df = pd.DataFrame(new_tokens_data)

    # Merge with existing
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Sort by market cap rank
    combined_df = combined_df.sort_values('market_cap_rank').reset_index(drop=True)

    logger.info(f"\nSaving combined dataset ({len(combined_df)} tokens) to {csv_path}...")
    combined_df.to_csv(csv_path, index=False)

    # Update database
    logger.info(f"\nUpdating PostgreSQL database...")
    try:
        with get_cursor() as cur:
            # Insert new tokens only
            for _, row in new_df.iterrows():
                upsert_query = """
                    INSERT INTO tokens (
                        token_address, symbol, name, decimals, narrative_category,
                        market_cap_rank, avg_daily_volume_usd, liquidity_tier,
                        classification_confidence, requires_manual_review, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (token_address)
                    DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        name = EXCLUDED.name,
                        narrative_category = EXCLUDED.narrative_category,
                        market_cap_rank = EXCLUDED.market_cap_rank,
                        avg_daily_volume_usd = EXCLUDED.avg_daily_volume_usd,
                        liquidity_tier = EXCLUDED.liquidity_tier,
                        classification_confidence = EXCLUDED.classification_confidence,
                        requires_manual_review = EXCLUDED.requires_manual_review,
                        updated_at = NOW()
                """

                cur.execute(upsert_query, (
                    row['token_address'],
                    row['symbol'],
                    row['name'],
                    row['decimals'],
                    row['narrative_category'],
                    int(row['market_cap_rank']) if pd.notna(row['market_cap_rank']) else None,
                    float(row['avg_daily_volume_usd']) if pd.notna(row['avg_daily_volume_usd']) else None,
                    row['liquidity_tier'] if pd.notna(row['liquidity_tier']) else None,
                    float(row['classification_confidence']) if pd.notna(row['classification_confidence']) else None,
                    bool(row['requires_manual_review']),
                    row['created_at']
                ))

        logger.info(f"✅ Database updated with {len(new_df)} new tokens")

    except Exception as e:
        logger.error(f"❌ Database update failed: {e}")
        logger.info(f"CSV file saved successfully, but database not updated")
        return False

    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Token Universe Expansion Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"   Previous count: {len(existing_df) if not existing_df.empty else 0}")
    logger.info(f"   New tokens added: {len(new_df)}")
    logger.info(f"   Total tokens: {len(combined_df)}")
    logger.info(f"   Backup saved: {backup_path}")

    # Narrative distribution
    logger.info(f"\nNarrative Distribution (NEW tokens only):")
    narrative_counts = new_df['narrative_category'].value_counts()
    for narrative, count in narrative_counts.items():
        pct = count / len(new_df) * 100
        requires_review = new_df[new_df['narrative_category'] == narrative]['requires_manual_review'].sum()
        logger.info(f"   {narrative}: {count} ({pct:.1f}%) - {requires_review} need review")

    # Liquidity tier distribution
    logger.info(f"\nLiquidity Tier Distribution (NEW tokens only):")
    tier_counts = new_df['liquidity_tier'].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(new_df) * 100
        logger.info(f"   {tier or 'Untiered'}: {count} ({pct:.1f}%)")

    # Manual review warning
    review_needed = new_df['requires_manual_review'].sum()
    if review_needed > 0:
        logger.info(f"\n⚠️  NOTE: {review_needed} new tokens require manual narrative review (Epic 4.2)")

    logger.info(f"\n✅ Ready for metrics enhancement: python cli_enhance_token_metrics.py")

    return True


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    success = fetch_additional_tokens(target_count=1000, existing_count=500)
    sys.exit(0 if success else 1)
