#!/usr/bin/env python3
"""
Transaction Collection CLI

Collects DEX swap transactions for all discovered smart money wallets.
Uses Alchemy RPC, Uniswap subgraphs, and Curve API for comprehensive data collection.
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncpg
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.common.logging_setup import get_logger
from data_collection.common.config import get_config
from services.transactions.batch_processor import BatchProcessor
from services.transactions.alchemy_client import AlchemyClient
from services.transactions.uniswap_client import UniswapClient
from services.transactions.curve_client import CurveClient
from services.transactions.db_manager import TransactionDatabaseManager
from services.transactions.checkpoint_manager import TransactionCheckpointManager

logger = get_logger(__name__)


async def get_database_pool():
    """Get asyncpg database connection pool"""
    config = get_config()
    db_config = config.get("database", {})

    return await asyncpg.create_pool(
        user=db_config.get('user', 'txelusanchez'),
        host=db_config.get('host', 'localhost'),
        port=db_config.get('port', 5432),
        database=db_config.get('database', 'crypto_narratives'),
        min_size=5,
        max_size=20
    )


async def load_wallets_from_database(pool: asyncpg.Pool, limit: int = None) -> List[str]:
    """Load smart money wallet addresses from database"""
    logger.info("Loading wallet addresses from database...")

    query = """
        SELECT wallet_address
        FROM wallets
        WHERE is_smart_money = true
        ORDER BY total_trades_30d DESC
    """

    if limit:
        query += f" LIMIT {limit}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        wallets = [row['wallet_address'] for row in rows]

    logger.info(f"Loaded {len(wallets)} wallet addresses")
    return wallets


async def is_wallet_completed(pool: asyncpg.Pool, wallet_address: str) -> bool:
    """Check if wallet has already been processed successfully"""
    query = """
        SELECT status
        FROM collection_checkpoints
        WHERE collection_type = 'transaction_extraction'
        AND metadata->>'wallet_address' = $1
        AND status = 'completed'
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(query, wallet_address)
        return result is not None


async def mark_wallet_completed(pool: asyncpg.Pool, wallet_address: str, tx_count: int):
    """Mark wallet as completed"""
    metadata = {
        "wallet_address": wallet_address,
        "transactions_count": tx_count,
        "completed_at": datetime.now().isoformat()
    }

    query = """
        INSERT INTO collection_checkpoints
        (collection_type, last_processed_block, records_collected, status, metadata, updated_at)
        VALUES ('transaction_extraction', 0, $1, 'completed', $2, NOW())
        ON CONFLICT (collection_type, COALESCE((metadata->>'wallet_address')::text, ''))
        DO UPDATE SET
            records_collected = EXCLUDED.records_collected,
            status = 'completed',
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """

    import json
    async with pool.acquire() as conn:
        await conn.execute(query, tx_count, json.dumps(metadata))


async def mark_wallet_failed(pool: asyncpg.Pool, wallet_address: str, error: str):
    """Mark wallet as failed"""
    metadata = {
        "wallet_address": wallet_address,
        "error": error,
        "failed_at": datetime.now().isoformat()
    }

    query = """
        INSERT INTO collection_checkpoints
        (collection_type, last_processed_block, records_collected, status, metadata, updated_at)
        VALUES ('transaction_extraction', 0, 0, 'failed', $1, NOW())
        ON CONFLICT (collection_type, COALESCE((metadata->>'wallet_address')::text, ''))
        DO UPDATE SET
            status = 'failed',
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """

    import json
    async with pool.acquire() as conn:
        await conn.execute(query, json.dumps(metadata))


async def process_wallet_transactions(
    wallet_address: str,
    alchemy_client: AlchemyClient,
    uniswap_client: UniswapClient,
    curve_client: CurveClient,
    db_manager: TransactionDatabaseManager,
    pool: asyncpg.Pool
) -> Dict[str, Any]:
    """
    Process all transactions for a single wallet

    Returns dict with wallet_address, tx_count, status
    """
    try:
        # Check if already processed
        if await is_wallet_completed(pool, wallet_address):
            logger.debug(f"Wallet {wallet_address} already processed, skipping")
            return {
                "wallet_address": wallet_address,
                "tx_count": 0,
                "status": "skipped",
                "reason": "already_processed"
            }

        logger.info(f"Processing wallet: {wallet_address}")

        # Define date range (last 90 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        all_transactions = []

        # 1. Get Uniswap transactions
        try:
            uniswap_txs = await uniswap_client.get_wallet_swaps(
                wallet_address,
                start_date,
                end_date
            )
            all_transactions.extend(uniswap_txs)
            logger.debug(f"  Found {len(uniswap_txs)} Uniswap transactions")
        except Exception as e:
            logger.warning(f"  Failed to get Uniswap txs for {wallet_address}: {e}")

        # 2. Get Curve transactions
        try:
            curve_txs = await curve_client.get_wallet_swaps(
                wallet_address,
                start_date,
                end_date
            )
            all_transactions.extend(curve_txs)
            logger.debug(f"  Found {len(curve_txs)} Curve transactions")
        except Exception as e:
            logger.warning(f"  Failed to get Curve txs for {wallet_address}: {e}")

        # 3. Get direct blockchain transactions via Alchemy
        try:
            alchemy_txs = await alchemy_client.get_wallet_transactions(
                wallet_address,
                start_date,
                end_date
            )
            all_transactions.extend(alchemy_txs)
            logger.debug(f"  Found {len(alchemy_txs)} direct transactions")
        except Exception as e:
            logger.warning(f"  Failed to get Alchemy txs for {wallet_address}: {e}")

        # 4. Deduplicate by transaction hash
        unique_txs = {}
        for tx in all_transactions:
            tx_hash = tx.get('transaction_hash') or tx.get('hash')
            if tx_hash and tx_hash not in unique_txs:
                unique_txs[tx_hash] = tx

        transaction_list = list(unique_txs.values())

        # 5. Store in database
        if transaction_list:
            await db_manager.upsert_transactions(transaction_list)

        # 6. Mark as completed
        await mark_wallet_completed(pool, wallet_address, len(transaction_list))

        logger.info(f"✓ Processed {wallet_address}: {len(transaction_list)} transactions")

        return {
            "wallet_address": wallet_address,
            "tx_count": len(transaction_list),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"✗ Failed to process {wallet_address}: {e}")
        await mark_wallet_failed(pool, wallet_address, str(e))
        return {
            "wallet_address": wallet_address,
            "tx_count": 0,
            "status": "failed",
            "error": str(e)
        }


async def main():
    """Main execution function"""
    print("=" * 80)
    print("TRANSACTION COLLECTION - DEX Swap History")
    print("=" * 80)
    print()

    pool = None
    try:
        # Initialize config
        config = get_config()

        # Create database pool
        logger.info("Creating database connection pool...")
        pool = await get_database_pool()

        # Load wallets
        wallets = await load_wallets_from_database(pool)

        if not wallets:
            logger.error("No wallets found in database!")
            print("\n❌ No smart money wallets found in database")
            print("Run wallet import first: python import_wallets_from_cache.py")
            return 1

        print(f"\n📊 Transaction Collection Summary")
        print(f"=" * 80)
        print(f"Wallets to process:  {len(wallets):,}")
        print(f"Date range:          Last 90 days")
        print(f"Data sources:        Uniswap, Curve, Alchemy RPC")
        print(f"Estimated time:      12-24 hours")
        print(f"=" * 80)
        print()

        # Initialize clients
        logger.info("Initializing API clients...")
        alchemy_client = AlchemyClient()
        uniswap_client = UniswapClient()
        curve_client = CurveClient()
        db_manager = TransactionDatabaseManager(pool)

        # Ensure schema exists
        await db_manager.create_schema()

        # Process wallets with progress bar
        print("🚀 Starting transaction collection...")
        print(f"Processing wallets with checkpointing")
        print(f"Progress will be saved automatically\n")

        results = []

        # Process with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent wallets

        async def process_with_semaphore(wallet):
            async with semaphore:
                return await process_wallet_transactions(
                    wallet,
                    alchemy_client,
                    uniswap_client,
                    curve_client,
                    db_manager,
                    pool
                )

        # Create tasks for all wallets
        tasks = [process_with_semaphore(wallet) for wallet in wallets]

        # Process with progress bar
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Collecting transactions"):
            result = await coro
            results.append(result)

        # Summary statistics
        successful = sum(1 for r in results if r and r['status'] == 'success')
        failed = sum(1 for r in results if r and r['status'] == 'failed')
        skipped = sum(1 for r in results if r and r['status'] == 'skipped')
        total_txs = sum(r['tx_count'] for r in results if r and 'tx_count' in r)

        print("\n" + "=" * 80)
        print("COLLECTION COMPLETE")
        print("=" * 80)
        print(f"✓ Successful:        {successful:,} wallets")
        print(f"✗ Failed:            {failed:,} wallets")
        print(f"⊘ Skipped:           {skipped:,} wallets (already processed)")
        print(f"📊 Total transactions: {total_txs:,}")
        print("=" * 80)

        # Verify database
        stats = await db_manager.get_statistics()
        print(f"\n📁 Database Status:")
        print(f"   Transactions stored: {stats.get('total_transactions', 0):,}")
        print(f"   Wallets with data:   {stats.get('unique_wallets', 0):,}")
        print(f"   Successful txs:      {stats.get('successful', 0):,}")
        print(f"   Failed txs:          {stats.get('failed', 0):,}")

        print("\n✓ Transaction collection complete!")
        print("Ready for performance metrics calculation\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("Progress has been saved - safe to restart later")
        return 130

    except Exception as e:
        logger.exception("Transaction collection failed")
        print(f"\n❌ Error: {e}")
        print("Check logs for details: logs/collection_*.log")
        return 1

    finally:
        if pool:
            await pool.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
