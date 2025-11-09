#!/usr/bin/env python3
"""
Test script to verify the complete token fetcher pipeline.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.tokens.token_collection_service import TokenCollectionService


def test_token_fetcher_pipeline():
    """Test the complete token fetcher pipeline with a small sample."""

    print("Testing Token Fetcher Pipeline...")
    print(f"Database URL: {os.getenv('DATABASE_URL')}")
    print(f"CoinGecko API Key: {'Configured' if os.getenv('COINGECKO_API_KEY') else 'Not configured'}")

    try:
        # Initialize service
        service = TokenCollectionService()

        # Run a small test collection (10 tokens)
        print("\n🚀 Starting token collection (10 tokens for testing)...")
        result = service.collect_tokens(target_count=10)

        if result:
            print(f"\n✅ Token collection completed successfully!")
            print(f"   Tokens collected: {result.get('tokens_collected', 0)}")
            print(f"   Collection took: {result.get('duration_seconds', 0):.2f} seconds")
            print(f"   Ethereum tokens found: {result.get('ethereum_tokens_found', 0)}")
            print(f"   Duplicates found: {result.get('duplicates_found', 0)}")

            # Show sample of collected tokens
            if 'sample_tokens' in result:
                print(f"\n📊 Sample tokens collected:")
                for i, token in enumerate(result['sample_tokens'][:3], 1):
                    print(f"   {i}. {token['name']} ({token['symbol']}) - Rank: {token.get('market_cap_rank', 'N/A')}")

        else:
            print("❌ Token collection returned no results")
            return False

        return True

    except Exception as e:
        print(f"\n❌ Token fetcher pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_token_fetcher_pipeline()
    sys.exit(0 if success else 1)