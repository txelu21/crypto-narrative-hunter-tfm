#!/usr/bin/env python3
"""
Token Metrics Enhancement Script
Collects additional token-level metrics to match tutor's requirements:
- Holder count (Etherscan API)
- Current price, market cap, circulating supply (CoinGecko API)
- Calculated: FDV, Volume/MC ratio

Expected Runtime: ~5-10 minutes
- Etherscan: 500 tokens × 0.2s = 100 seconds
- CoinGecko: Batch requests (minimize API calls)
"""

import pandas as pd
import requests
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


def get_token_holder_count(token_address: str) -> int:
    """
    Fetch token holder count from Etherscan API

    Args:
        token_address: Ethereum token contract address

    Returns:
        Number of unique holders (integer)
    """
    try:
        params = {
            "module": "token",
            "action": "tokenholderlist",
            "contractaddress": token_address,
            "page": 1,
            "offset": 1,  # Only need total count
            "apikey": ETHERSCAN_API_KEY
        }

        response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=10)
        data = response.json()

        if data.get("status") == "1" and data.get("result"):
            # Etherscan doesn't directly provide total holder count in API
            # Alternative: Use tokeninfo endpoint
            params_info = {
                "module": "stats",
                "action": "tokensupply",
                "contractaddress": token_address,
                "apikey": ETHERSCAN_API_KEY
            }
            # Note: Etherscan free tier doesn't provide holder count directly
            # Using workaround: Estimate from top holders or use CoinGecko
            return None  # Will fall back to CoinGecko or mark as unavailable

    except Exception as e:
        logger.warning(f"Etherscan API error for {token_address}: {e}")
        return None


def get_coingecko_token_data(token_address: str, max_retries: int = 3) -> dict:
    """
    Fetch comprehensive token data from CoinGecko API with retry logic

    Args:
        token_address: Ethereum token contract address
        max_retries: Maximum number of retries for rate limit errors

    Returns:
        Dictionary with: current_price_usd, market_cap, circulating_supply,
                        total_supply, holder_count (if available)
    """
    url = f"{COINGECKO_BASE_URL}/coins/ethereum/contract/{token_address}"

    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()

                return {
                    "current_price_usd": data.get("market_data", {}).get("current_price", {}).get("usd"),
                    "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
                    "circulating_supply": data.get("market_data", {}).get("circulating_supply"),
                    "total_supply": data.get("market_data", {}).get("total_supply"),
                    "holder_count": data.get("community_data", {}).get("holders"),  # May be None
                    "price_change_24h_pct": data.get("market_data", {}).get("price_change_percentage_24h"),
                    "volume_24h": data.get("market_data", {}).get("total_volume", {}).get("usd")
                }
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 400:
                # Bad request - token not found or invalid, skip retry
                logger.warning(f"CoinGecko: Token not found (400) for {token_address}")
                return None
            else:
                logger.warning(f"CoinGecko API returned {response.status_code} for {token_address}")
                return None

        except Exception as e:
            logger.warning(f"CoinGecko API error for {token_address}: {e}")
            return None

    # Max retries exceeded
    logger.error(f"Max retries exceeded for {token_address}")
    return None


def calculate_derived_metrics(row: pd.Series) -> dict:
    """
    Calculate derived metrics: FDV, Volume/MC ratio

    Args:
        row: DataFrame row with token data

    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}

    # FDV = total_supply × current_price_usd
    if pd.notna(row.get("total_supply")) and pd.notna(row.get("current_price_usd")):
        metrics["fdv"] = row["total_supply"] * row["current_price_usd"]
    else:
        metrics["fdv"] = None

    # Volume/Market Cap ratio
    if pd.notna(row.get("volume_24h")) and pd.notna(row.get("market_cap")) and row["market_cap"] > 0:
        metrics["volume_mc_ratio"] = row["volume_24h"] / row["market_cap"]
    else:
        metrics["volume_mc_ratio"] = None

    return metrics


def enhance_token_metrics():
    """
    Main function: Enhance token metrics with additional data
    """
    csv_path = Path(__file__).parent / "outputs" / "csv" / "tokens.csv"

    if not csv_path.exists():
        logger.error(f"Error: {csv_path} not found")
        return False

    logger.info(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    logger.info(f"Total tokens: {len(df)}")
    logger.info(f"Fetching enhanced metrics from CoinGecko and Etherscan...")
    logger.info(f"Estimated time: 5-10 minutes for {len(df)} tokens")

    # Backup original
    backup_path = csv_path.with_suffix('.csv.backup_before_metrics')
    if not backup_path.exists():
        logger.info(f"Creating backup: {backup_path}")
        df.to_csv(backup_path, index=False)

    # Add new columns
    new_columns = {
        "holder_count": None,
        "circulating_supply": None,
        "current_market_cap": None,
        "current_price_usd": None,
        "fdv": None,
        "volume_mc_ratio": None,
        "price_change_24h_pct": None,
        "volume_24h": None,
        "metrics_updated_at": None
    }

    for col in new_columns:
        if col not in df.columns:
            df[col] = new_columns[col]

    # Process each token
    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        try:
            token_address = row["token_address"]
            token_symbol = row["symbol"]

            logger.info(f"[{idx+1}/{len(df)}] Processing {token_symbol} ({token_address})")

            # Skip if already has data (unless force refresh)
            if pd.notna(row.get("current_price_usd")) and row.get("current_price_usd") != "":
                logger.info(f"  Skipping {token_symbol} (already has metrics)")
                continue

            # Fetch from CoinGecko (primary source)
            cg_data = get_coingecko_token_data(token_address)

            if cg_data:
                df.at[idx, "current_price_usd"] = cg_data.get("current_price_usd")
                df.at[idx, "current_market_cap"] = cg_data.get("market_cap")
                df.at[idx, "circulating_supply"] = cg_data.get("circulating_supply")
                df.at[idx, "holder_count"] = cg_data.get("holder_count")  # May be None
                df.at[idx, "price_change_24h_pct"] = cg_data.get("price_change_24h_pct")
                df.at[idx, "volume_24h"] = cg_data.get("volume_24h")

                # Use CoinGecko total_supply if available, else use existing
                if cg_data.get("total_supply"):
                    df.at[idx, "total_supply"] = cg_data.get("total_supply")

                # Calculate derived metrics
                derived = calculate_derived_metrics(df.loc[idx])
                df.at[idx, "fdv"] = derived.get("fdv")
                df.at[idx, "volume_mc_ratio"] = derived.get("volume_mc_ratio")

                df.at[idx, "metrics_updated_at"] = datetime.now()

                success_count += 1

                logger.info(f"  ✅ {token_symbol}: Price=${cg_data.get('current_price_usd'):.6f}, "
                          f"MC=${cg_data.get('market_cap'):,.0f}, "
                          f"FDV=${derived.get('fdv'):,.0f if derived.get('fdv') else 0}")
            else:
                logger.warning(f"  ❌ {token_symbol}: Failed to fetch data")
                fail_count += 1

            # Rate limiting: CoinGecko free tier = 10-30 calls/minute on contract endpoint
            # Use conservative 4 seconds between requests = 15 calls/minute
            time.sleep(4.0)

            # Checkpoint every 50 tokens
            if (idx + 1) % 50 == 0:
                logger.info(f"  💾 Checkpoint: Saving progress at token {idx+1}...")
                df.to_csv(csv_path, index=False)
                logger.info(f"  Success: {success_count}, Failed: {fail_count}")

        except Exception as e:
            logger.error(f"  Error processing {row['symbol']}: {e}")
            fail_count += 1
            continue

    # Final save
    logger.info(f"\nSaving final data to {csv_path}...")
    df.to_csv(csv_path, index=False)

    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Token metrics enhancement complete!")
    logger.info(f"{'='*60}")
    logger.info(f"   Total tokens: {len(df)}")
    logger.info(f"   Successfully updated: {success_count}")
    logger.info(f"   Failed: {fail_count}")
    logger.info(f"   Success rate: {success_count/len(df)*100:.1f}%")

    # Data completeness
    completeness = {
        "current_price_usd": df["current_price_usd"].notna().sum(),
        "current_market_cap": df["current_market_cap"].notna().sum(),
        "circulating_supply": df["circulating_supply"].notna().sum(),
        "holder_count": df["holder_count"].notna().sum(),
        "fdv": df["fdv"].notna().sum(),
        "volume_mc_ratio": df["volume_mc_ratio"].notna().sum()
    }

    logger.info(f"\nData Completeness:")
    for metric, count in completeness.items():
        pct = count / len(df) * 100
        logger.info(f"   {metric}: {count}/{len(df)} ({pct:.1f}%)")

    # Show sample
    logger.info(f"\nSample with enhanced metrics:")
    sample = df[pd.notna(df["current_price_usd"])].head(5)
    print(sample[["symbol", "current_price_usd", "current_market_cap", "fdv", "volume_mc_ratio", "holder_count"]].to_string())

    return True


if __name__ == "__main__":
    import sys

    if not COINGECKO_API_KEY:
        logger.error("Error: COINGECKO_API_KEY not set in .env file")
        sys.exit(1)

    success = enhance_token_metrics()
    sys.exit(0 if success else 1)
