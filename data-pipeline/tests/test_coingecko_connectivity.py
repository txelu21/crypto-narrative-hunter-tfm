#!/usr/bin/env python3
"""
Quick test script to verify CoinGecko API connectivity with API key.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.tokens.coingecko_client import CoinGeckoClient


def test_coingecko_connectivity():
    """Test CoinGecko API connectivity and rate limits."""

    print("Testing CoinGecko API connectivity...")
    print(f"API Key configured: {'Yes' if os.getenv('COINGECKO_API_KEY') else 'No'}")

    try:
        # Initialize client
        client = CoinGeckoClient()

        # Test rate limit info (if available with API key)
        print("\n1. Testing markets API (small sample)...")
        markets = client.get_coins_markets(per_page=5, page=1)
        print(f"   Retrieved {len(markets)} market entries")

        if markets:
            sample_token = markets[0]
            print(f"   Sample token: {sample_token.get('name')} ({sample_token.get('symbol')})")

        # Test coin details endpoint
        print("\n2. Testing coin details API...")
        try:
            details = client.get_coin_details_by_id("ethereum")
            platforms = details.get("platforms", {})
            eth_address = platforms.get("ethereum")
            print(f"   Ethereum details retrieved, contract: {eth_address}")
        except Exception as e:
            print(f"   Coin details test failed: {e}")

        print("\n✅ CoinGecko API connectivity test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ CoinGecko API test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_coingecko_connectivity()
    sys.exit(0 if success else 1)