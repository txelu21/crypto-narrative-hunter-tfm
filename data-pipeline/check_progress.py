#!/usr/bin/env python3
"""Quick progress check for balance collection"""
import os
import asyncpg
import asyncio
from datetime import datetime

async def check_progress():
    db_url = os.getenv("DATABASE_URL", "postgresql://txelusanchez@localhost:5432/crypto_narratives")

    conn = await asyncpg.connect(db_url)

    # Get stats
    row = await conn.fetchrow("""
        SELECT
            COUNT(DISTINCT wallet_address) as wallets,
            COUNT(*) as snapshots,
            COUNT(DISTINCT token_address) as tokens
        FROM wallet_token_balances
    """)

    await conn.close()

    wallets = row['wallets']
    snapshots = row['snapshots']
    tokens = row['tokens']
    total_wallets = 2343
    pct = (wallets / total_wallets) * 100

    print("=" * 50)
    print("  BALANCE COLLECTION PROGRESS")
    print("=" * 50)
    print(f"\n✅ Wallets completed: {wallets:,} / {total_wallets:,} ({pct:.2f}%)")
    print(f"📊 Total snapshots:   {snapshots:,}")
    print(f"🪙 Unique tokens:     {tokens:,}")
    print(f"📈 Avg per wallet:    {snapshots/max(wallets,1):.0f} balances")

    # Estimate completion
    if wallets > 0:
        elapsed_hours = wallets * 10 / 3600  # 10 sec per wallet
        remaining_wallets = total_wallets - wallets
        remaining_hours = remaining_wallets * 10 / 3600
        print(f"\n⏱️  Estimated remaining: ~{remaining_hours:.1f} hours")

    print("\nTo watch live: watch -n 10 python3 check_progress.py")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(check_progress())
