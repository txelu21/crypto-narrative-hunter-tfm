#!/usr/bin/env python3
"""
Daily Balance Snapshot Collection Script

Collects historical ERC20 token balances for smart money wallets at daily intervals
using Alchemy's alchemy_getTokenBalances API.

Usage:
    python cli_collect_daily_balances.py --start-date 2025-09-03 --end-date 2025-10-03

Features:
    - Daily balance snapshots for all Tier 1 wallets
    - Checkpoint/resume functionality for interrupted collections
    - Rate limiting and error handling
    - Progress tracking and logging
    - Database persistence with conflict resolution
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set
import json

import aiohttp
import asyncpg
import structlog
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from services.block_mapper import BlockMapper
from data_collection.common.config import settings

# Load environment variables
load_dotenv()

logger = structlog.get_logger()


class BalanceCollector:
    """Collects historical token balances using Alchemy API."""

    def __init__(
        self,
        api_key: str,
        db_pool: asyncpg.Pool,
        rate_limit: int = 10,  # Conservative: 10 req/sec (vs 25 limit)
        checkpoint_dir: str = "data/checkpoints/balances"
    ):
        """
        Initialize balance collector.

        Args:
            api_key: Alchemy API key
            db_pool: Database connection pool
            rate_limit: Max requests per second
            checkpoint_dir: Directory for checkpoint files
        """
        self.api_key = api_key
        self.base_url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
        self.db_pool = db_pool
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.checkpoint_dir = checkpoint_dir
        self.block_mapper = BlockMapper()

        # Statistics
        self.total_calls = 0
        self.total_balances_found = 0
        self.errors = 0

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.logger = logger.bind(component="balance_collector")

    async def get_token_balances(
        self,
        wallet_address: str,
        block_number: int,
        session: aiohttp.ClientSession
    ) -> List[Dict]:
        """
        Get all ERC20 token balances for a wallet at a specific block.

        Args:
            wallet_address: Ethereum wallet address
            block_number: Block number for historical query
            session: aiohttp session

        Returns:
            List of token balances with metadata
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getTokenBalances",
            "params": [
                wallet_address,
                "erc20",  # Get ALL ERC20 tokens
                {
                    "blockNumber": hex(block_number)
                }
            ]
        }

        async with self.semaphore:
            try:
                async with session.post(self.base_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if "error" in data:
                        error_msg = data["error"].get("message", "Unknown error")
                        self.logger.error(
                            "api_error",
                            wallet=wallet_address,
                            block=block_number,
                            error=error_msg
                        )
                        self.errors += 1
                        return []

                    result = data.get("result", {})
                    token_balances = result.get("tokenBalances", [])

                    self.total_calls += 1

                    # Filter out zero balances
                    non_zero_balances = [
                        tb for tb in token_balances
                        if tb.get("tokenBalance") and tb["tokenBalance"] != "0x0"
                    ]

                    self.total_balances_found += len(non_zero_balances)

                    return non_zero_balances

            except Exception as e:
                self.logger.error(
                    "request_failed",
                    wallet=wallet_address,
                    block=block_number,
                    error=str(e)
                )
                self.errors += 1
                return []

            # Small delay to respect rate limits
            await asyncio.sleep(0.05)  # 20 req/sec max

    async def get_token_metadata(
        self,
        token_address: str,
        session: aiohttp.ClientSession
    ) -> Dict[str, any]:
        """
        Get token metadata (symbol, decimals, name) using Alchemy Token API.

        Args:
            token_address: ERC20 token contract address
            session: aiohttp session

        Returns:
            Dict with token metadata
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getTokenMetadata",
            "params": [token_address]
        }

        async with self.semaphore:
            try:
                async with session.post(self.base_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if "error" in data:
                        return {
                            "symbol": None,
                            "name": None,
                            "decimals": 18  # Default
                        }

                    result = data.get("result", {})
                    return {
                        "symbol": result.get("symbol"),
                        "name": result.get("name"),
                        "decimals": result.get("decimals", 18)
                    }

            except Exception:
                return {
                    "symbol": None,
                    "name": None,
                    "decimals": 18
                }

    async def store_balances(
        self,
        wallet_address: str,
        snapshot_date: datetime,
        block_number: int,
        balances: List[Dict],
        token_metadata_cache: Dict[str, Dict]
    ):
        """
        Store balance snapshots in database.

        Args:
            wallet_address: Wallet address
            snapshot_date: Date of snapshot
            block_number: Block number
            balances: List of token balances
            token_metadata_cache: Cache of token metadata
        """
        if not balances:
            return

        records = []
        for balance_data in balances:
            token_address = balance_data.get("contractAddress")
            balance_hex = balance_data.get("tokenBalance", "0x0")

            # Convert hex balance to integer
            balance_raw = int(balance_hex, 16)

            # Get token metadata
            metadata = token_metadata_cache.get(token_address, {})
            decimals = metadata.get("decimals") or 18  # Default to 18 if None
            symbol = metadata.get("symbol")
            name = metadata.get("name")

            # Calculate formatted balance
            balance_formatted = Decimal(balance_raw) / Decimal(10 ** decimals)

            records.append((
                wallet_address.lower(),
                token_address.lower(),
                snapshot_date.date(),
                block_number,
                balance_raw,
                float(balance_formatted),
                symbol,
                name,
                decimals
            ))

        # Batch insert with ON CONFLICT DO NOTHING
        insert_query = """
            INSERT INTO wallet_token_balances (
                wallet_address,
                token_address,
                snapshot_date,
                block_number,
                balance_raw,
                balance_formatted,
                token_symbol,
                token_name,
                decimals
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (wallet_address, token_address, snapshot_date)
            DO NOTHING
        """

        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(insert_query, records)

            self.logger.info(
                "stored_balances",
                wallet=wallet_address,
                date=snapshot_date.date(),
                count=len(records)
            )

        except Exception as e:
            self.logger.error(
                "failed_to_store_balances",
                wallet=wallet_address,
                error=str(e)
            )

    def save_checkpoint(self, date: datetime, completed_wallets: Set[str]):
        """Save checkpoint for date."""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{date.strftime('%Y-%m-%d')}.json"
        )
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "date": date.strftime("%Y-%m-%d"),
                "completed_wallets": list(completed_wallets),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    def load_checkpoint(self, date: datetime) -> Set[str]:
        """Load checkpoint for date if exists."""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{date.strftime('%Y-%m-%d')}.json"
        )
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get("completed_wallets", []))
        return set()

    async def collect_daily_snapshot(
        self,
        date: datetime,
        wallets: List[str],
        session: aiohttp.ClientSession,
        token_metadata_cache: Dict[str, Dict]
    ):
        """
        Collect balance snapshots for all wallets on a specific date.

        Args:
            date: Snapshot date
            wallets: List of wallet addresses
            session: aiohttp session
            token_metadata_cache: Cache for token metadata
        """
        # Get block number for date
        block_number = self.block_mapper.date_to_block(date)

        self.logger.info(
            "collecting_daily_snapshot",
            date=date.date(),
            block=block_number,
            wallet_count=len(wallets)
        )

        # Load checkpoint
        completed_wallets = self.load_checkpoint(date)
        remaining_wallets = [w for w in wallets if w.lower() not in completed_wallets]

        if not remaining_wallets:
            self.logger.info("snapshot_already_complete", date=date.date())
            return

        self.logger.info(
            "resuming_from_checkpoint",
            date=date.date(),
            completed=len(completed_wallets),
            remaining=len(remaining_wallets)
        )

        # Progress bar
        pbar = tqdm(
            total=len(remaining_wallets),
            desc=f"{date.date()}",
            unit="wallet"
        )

        for wallet in remaining_wallets:
            # Get balances
            balances = await self.get_token_balances(wallet, block_number, session)

            # Fetch metadata for new tokens
            for balance_data in balances:
                token_address = balance_data.get("contractAddress")
                if token_address and token_address not in token_metadata_cache:
                    metadata = await self.get_token_metadata(token_address, session)
                    token_metadata_cache[token_address] = metadata

            # Store in database
            await self.store_balances(
                wallet,
                date,
                block_number,
                balances,
                token_metadata_cache
            )

            # Update checkpoint
            completed_wallets.add(wallet.lower())
            if len(completed_wallets) % 10 == 0:  # Save every 10 wallets
                self.save_checkpoint(date, completed_wallets)

            pbar.update(1)

        pbar.close()

        # Final checkpoint save
        self.save_checkpoint(date, completed_wallets)

        self.logger.info(
            "completed_daily_snapshot",
            date=date.date(),
            wallets_processed=len(wallets)
        )

    async def collect_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        wallets: List[str]
    ):
        """
        Collect balance snapshots for date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            wallets: List of wallet addresses
        """
        self.logger.info(
            "starting_collection",
            start_date=start_date.date(),
            end_date=end_date.date(),
            wallet_count=len(wallets),
            total_snapshots=(end_date - start_date).days + 1
        )

        # Token metadata cache
        token_metadata_cache = {}

        # Create aiohttp session
        async with aiohttp.ClientSession() as session:
            current_date = start_date

            while current_date <= end_date:
                await self.collect_daily_snapshot(
                    current_date,
                    wallets,
                    session,
                    token_metadata_cache
                )
                current_date += timedelta(days=1)

        # Print summary
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        print(f"Total API calls: {self.total_calls:,}")
        print(f"Total balances found: {self.total_balances_found:,}")
        print(f"Errors: {self.errors}")
        print(f"Success rate: {(1 - self.errors/max(self.total_calls, 1))*100:.2f}%")
        print("="*60)


async def get_tier1_wallets(db_pool: asyncpg.Pool) -> List[str]:
    """
    Get Tier 1 wallets (those with transaction data).

    Returns:
        List of wallet addresses
    """
    query = """
        SELECT DISTINCT wallet_address
        FROM transactions
        ORDER BY wallet_address
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query)
        return [row['wallet_address'] for row in rows]


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Collect daily balance snapshots")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-09-03",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-10-03",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--wallet-file",
        type=str,
        help="Optional: File with wallet addresses (one per line)"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=10,
        help="Requests per second (default: 10)"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Get API key
    api_key = os.getenv("ALCHEMY_API_KEY")
    if not api_key:
        print("❌ ALCHEMY_API_KEY environment variable not set")
        print("Please set your Alchemy API key in .env file")
        sys.exit(1)

    # Connect to database
    db_url = settings.database_url
    db_pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)

    try:
        # Get wallets
        if args.wallet_file:
            with open(args.wallet_file, 'r') as f:
                wallets = [line.strip() for line in f if line.strip()]
        else:
            print("📊 Loading Tier 1 wallets from database...")
            wallets = await get_tier1_wallets(db_pool)

        print(f"✓ Loaded {len(wallets):,} wallets")

        # Calculate expected API calls
        num_days = (end_date - start_date).days + 1
        expected_calls = len(wallets) * num_days
        expected_cu = expected_calls * 20  # 20 CU per call

        print(f"\n📅 Collection Plan:")
        print(f"  Date range: {start_date.date()} to {end_date.date()} ({num_days} days)")
        print(f"  Wallets: {len(wallets):,}")
        print(f"  Expected API calls: {expected_calls:,}")
        print(f"  Expected Compute Units: {expected_cu:,} CUs")
        print(f"  Alchemy Free Tier: 30,000,000 CUs/month")
        print(f"  Usage: {(expected_cu/30_000_000)*100:.2f}% of free tier")
        print(f"  Estimated runtime: ~{expected_calls/(args.rate_limit*60):.0f} minutes")
        print()

        # Confirm
        response = input("Proceed with collection? (y/n): ")
        if response.lower() != 'y':
            print("Collection cancelled")
            sys.exit(0)

        # Create collector
        collector = BalanceCollector(
            api_key=api_key,
            db_pool=db_pool,
            rate_limit=args.rate_limit
        )

        # Run collection
        await collector.collect_date_range(start_date, end_date, wallets)

        print("\n✅ Collection complete!")

    finally:
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
