#!/usr/bin/env python3
"""
Run the full token collection for 500 tokens.
"""

from services.tokens.token_collection_service import TokenCollectionService

def main():
    print("🚀 Starting token collection for top 500 Ethereum tokens...")
    print("⏱️  This will take several minutes due to API rate limits...")

    service = TokenCollectionService()
    result = service.collect_tokens(target_count=500)

    print(f"\n✅ Collection completed!")
    print(f"📊 Tokens collected: {result['tokens_collected']}")
    print(f"⏱️  Duration: {result['duration_seconds']:.1f} seconds")
    print(f"🔍 Ethereum tokens found: {result['ethereum_tokens_found']}")
    print(f"🔄 Duplicates found: {result['duplicates_found']}")

    if result.get('csv_file'):
        print(f"📁 CSV export: {result['csv_file']}")

if __name__ == "__main__":
    main()