#!/usr/bin/env python3
"""
Transaction Collection CLI - Dune Analytics Based

Collects DEX swap transactions for all discovered smart money wallets using Dune Analytics.
This is the WORKING implementation that uses proven Dune queries instead of incomplete RPC clients.
"""

import sys
import asyncio
import yaml
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
from decimal import Decimal
import asyncpg
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.common.logging_setup import get_logger
from data_collection.common.config import get_config
from services.tokens.dune_client import DuneClient

logger = get_logger(__name__)


def parse_timestamp(ts_str: str) -> datetime:
    """Parse Dune timestamp string to datetime object"""
    if isinstance(ts_str, datetime):
        return ts_str
    # Format: '2025-10-02 23:54:23.000 UTC'
    return datetime.strptime(ts_str.replace(' UTC', ''), '%Y-%m-%d %H:%M:%S.%f')


def safe_decimal(value: Any) -> Decimal:
    """Safely convert value to Decimal, handling None and out-of-range values"""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except:
        return None


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


async def load_wallets_from_database(pool: asyncpg.Pool) -> List[str]:
    """Load smart money wallet addresses from database"""
    logger.info("Loading wallet addresses from database...")

    query = """
        SELECT wallet_address
        FROM wallets
        WHERE is_smart_money = true
        ORDER BY total_trades_30d DESC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        wallets = [row['wallet_address'] for row in rows]

    logger.info(f"Loaded {len(wallets)} wallet addresses")
    return wallets


def collect_transactions_for_batch(
    batch_offset: int,
    batch_size: int,
    dune_client: DuneClient,
    start_date: date,
    end_date: date,
    query_id: int
) -> List[Dict[str, Any]]:
    """Collect transactions for a batch of wallets using Dune with pagination"""

    logger.info(f"Collecting transactions for batch offset {batch_offset}, size {batch_size}...")

    # Create query parameters with batching
    params = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'batch_size': batch_size,
        'batch_offset': batch_offset
    }

    try:
        # Execute the batched query and wait for results
        df_result = dune_client.execute_and_wait(
            query_id=query_id,
            parameters=params,
            timeout_seconds=180,  # 3 minutes timeout per batch
            use_cache=False  # Don't cache - we want fresh data
        )

        if df_result is not None and not df_result.empty:
            # Convert DataFrame to list of dicts
            transactions = df_result.to_dict('records')
            logger.info(f"Retrieved {len(transactions)} transactions for batch {batch_offset}")
            return transactions
        else:
            logger.warning(f"No data returned for batch {batch_offset}")
            return []

    except Exception as e:
        logger.error(f"Failed to collect batch {batch_offset}: {e}")
        raise


async def insert_transactions(pool: asyncpg.Pool, transactions: List[Dict[str, Any]]) -> int:
    """Insert transactions into database"""

    if not transactions:
        return 0

    logger.info(f"Inserting {len(transactions)} transactions...")

    inserted = 0
    async with pool.acquire() as conn:
        for tx in transactions:
            try:
                await conn.execute("""
                    INSERT INTO transactions (
                        tx_hash, block_number, timestamp, wallet_address,
                        dex_name, pool_address,
                        token_in, amount_in, token_out, amount_out,
                        gas_used, gas_price_gwei, transaction_status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (tx_hash) DO NOTHING
                """,
                    tx.get('tx_hash'),
                    tx.get('block_number'),
                    parse_timestamp(tx.get('timestamp')),  # Parse timestamp string to datetime
                    tx.get('wallet_address'),
                    tx.get('dex_name'),
                    tx.get('pool_address'),
                    tx.get('token_in'),
                    safe_decimal(tx.get('amount_in')),  # Use Decimal for large numbers
                    tx.get('token_out'),
                    safe_decimal(tx.get('amount_out')),  # Use Decimal for large numbers
                    tx.get('gas_used'),
                    safe_decimal(tx.get('gas_price_gwei')),  # Use Decimal
                    tx.get('status', 'success')
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert transaction {tx.get('tx_hash')}: {e}")
                continue

    logger.info(f"Inserted {inserted} new transactions")
    return inserted


async def get_last_completed_batch(pool: asyncpg.Pool) -> int:
    """Get the last successfully completed batch offset for resuming"""
    query = """
        SELECT metadata->>'last_batch_offset' as last_offset
        FROM collection_checkpoints
        WHERE collection_type = 'transactions'
        ORDER BY updated_at DESC
        LIMIT 1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query)
        if row and row['last_offset']:
            return int(row['last_offset'])
        return 0


async def save_batch_checkpoint(pool: asyncpg.Pool, batch_offset: int, batch_size: int, tx_count: int):
    """Save checkpoint after each successful batch"""
    query = """
        INSERT INTO collection_checkpoints (
            collection_type, last_processed_date, records_collected,
            status, metadata, created_at, updated_at
        ) VALUES (
            'transactions', NOW(), $1::integer, 'in_progress',
            jsonb_build_object('last_batch_offset', $2::integer, 'batch_size', $3::integer),
            NOW(), NOW()
        )
        ON CONFLICT (collection_type, COALESCE(metadata->>'wallet_address', ''))
        DO UPDATE SET
            records_collected = collection_checkpoints.records_collected + $1::integer,
            metadata = jsonb_build_object('last_batch_offset', $2::integer, 'batch_size', $3::integer),
            updated_at = NOW()
    """
    async with pool.acquire() as conn:
        await conn.execute(query, tx_count, batch_offset, batch_size)


async def main():
    """Main execution function"""
    print("=" * 80)
    print("TRANSACTION COLLECTION - Dune Analytics Batched")
    print("=" * 80)
    print()

    pool = None
    try:
        # Initialize config
        config = get_config()

        # Create database pool
        logger.info("Creating database connection pool...")
        pool = await get_database_pool()

        # Load query ID from config file
        try:
            config_file = Path(__file__).parent / "config" / "dune_query_ids.yaml"
            with open(config_file, 'r') as f:
                query_config = yaml.safe_load(f)
            query_id = query_config['transaction_collection']['wallet_transactions_batch']
            logger.info(f"Loaded query ID from config: {query_id}")
        except Exception as e:
            logger.error(f"Failed to load query ID from config: {e}")
            print("\n❌ Could not load query ID from config/dune_query_ids.yaml")
            print(f"Error: {e}")
            return 1

        # Count total wallets in uploaded dataset (we know it's 25,161 from import)
        total_wallets = 25161

        print(f"\n📊 Transaction Collection Summary")
        print(f"=" * 80)
        print(f"Total wallets:       {total_wallets:,}")
        print(f"Date range:          Last 30 days")
        print(f"Data source:         Dune Analytics (Query ID: {query_id})")
        print(f"Batch size:          10 wallets per query (with 2k tx/wallet limit)")
        print(f"=" * 80)
        print()

        # Date range
        end_date = date.today()
        start_date = end_date - timedelta(days=30)  # Reduced to 30 days to minimize datapoints

        # Initialize Dune client
        logger.info("Initializing Dune client...")
        dune_client = DuneClient()

        # Batching configuration
        # With per-wallet transaction limiting (2000 max), we can safely batch 10 wallets
        # Max datapoints: 10 wallets × 2000 txs × 11 columns = 220,000 (under 250k limit)
        batch_size = 10  # Process 10 wallets at a time with transaction limiting
        num_batches = (total_wallets + batch_size - 1) // batch_size

        # Check for previous checkpoint
        last_completed_offset = await get_last_completed_batch(pool)

        if last_completed_offset > 0:
            print(f"📍 Resuming from offset {last_completed_offset} (batch {last_completed_offset // batch_size + 1}/{num_batches})")
            print()

        print(f"📈 Execution Plan:")
        print(f"   Total batches:    {num_batches}")
        print(f"   Batch size:       {batch_size} wallets")
        print(f"   Estimated time:   ~{num_batches * 1.5} minutes ({num_batches * 1.5 / 60:.1f} hours)")
        print(f"   Dune credits:     ~{num_batches * 1} credits")
        print()
        print("   ⚙️  Per-wallet transaction limiting (max 2k txs) filters HFT/MEV bots")
        print("   📊 Batches will run sequentially with checkpointing")
        print("   🔄 Safe to interrupt (Ctrl+C) and resume later")
        print()

        # Execute batched collection
        total_txs = 0
        start_offset = last_completed_offset if last_completed_offset > 0 else 0

        for batch_offset in range(start_offset, total_wallets, batch_size):
            batch_num = batch_offset // batch_size + 1

            print(f"\n{'='*80}")
            print(f"Processing Batch {batch_num}/{num_batches} (offset {batch_offset})")
            print(f"{'='*80}")

            try:
                # Collect transactions for this batch (synchronous call)
                txs = collect_transactions_for_batch(
                    batch_offset=batch_offset,
                    batch_size=batch_size,
                    dune_client=dune_client,
                    start_date=start_date,
                    end_date=end_date,
                    query_id=query_id
                )

                # Insert into database
                inserted = await insert_transactions(pool, txs)
                total_txs += inserted

                # Save checkpoint
                await save_batch_checkpoint(pool, batch_offset + batch_size, batch_size, inserted)

                print(f"✓ Batch {batch_num}/{num_batches}: {inserted:,} transactions inserted")
                print(f"  Progress: {batch_offset + batch_size}/{total_wallets} wallets ({100*(batch_offset+batch_size)/total_wallets:.1f}%)")
                print(f"  Total transactions so far: {total_txs:,}")

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                print(f"\n❌ Batch {batch_num} failed: {e}")
                print(f"   Last successful offset: {batch_offset}")
                print(f"   You can resume by re-running this script.")
                raise

        print(f"\n{'='*80}")
        print(f"✅ COLLECTION COMPLETE!")
        print(f"{'='*80}")
        print(f"Total transactions collected: {total_txs:,}")
        print(f"Total batches processed:      {num_batches}")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("Run this script again to resume from the last checkpoint.")
        return 130

    except Exception as e:
        logger.exception("Transaction collection failed")
        print(f"\n❌ Error: {e}")
        return 1

    finally:
        if pool:
            await pool.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
