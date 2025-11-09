"""
Database storage and optimization for transactions.

Provides:
- Efficient batch insertion with conflict resolution
- Proper indexing for query performance
- Partitioning strategy for large datasets
- UPSERT operations for idempotency
- Connection pooling and transaction management
"""

import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal
import asyncpg

logger = logging.getLogger(__name__)


class TransactionDatabaseManager:
    """Manages database operations for transaction storage."""

    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize database manager.

        Args:
            db_pool: AsyncPG connection pool
        """
        self.pool = db_pool

    async def create_schema(self) -> None:
        """Create transaction tables and indexes."""
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS transactions (
                tx_hash VARCHAR(66) PRIMARY KEY,
                block_number BIGINT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                wallet_address VARCHAR(42) NOT NULL,

                -- DEX and pool information
                dex_name VARCHAR(20) NOT NULL,
                pool_address VARCHAR(42),

                -- Swap details
                token_in VARCHAR(42),
                amount_in NUMERIC(78,0),
                token_out VARCHAR(42),
                amount_out NUMERIC(78,0),

                -- Gas and execution
                gas_used INTEGER,
                gas_price_gwei NUMERIC(15,6),
                transaction_status VARCHAR(10) CHECK (transaction_status IN ('success', 'failed')),

                -- Value calculations
                eth_value_in NUMERIC(36,18),
                eth_value_out NUMERIC(36,18),
                usd_value_in NUMERIC(36,18),
                usd_value_out NUMERIC(36,18),

                -- MEV and quality flags
                mev_type VARCHAR(20),
                mev_damage_eth NUMERIC(36,18),
                slippage_percentage NUMERIC(8,4),

                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """

        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_transactions_wallet_time ON transactions(wallet_address, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_block ON transactions(block_number);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_dex ON transactions(dex_name);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_tokens ON transactions(token_in, token_out);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(transaction_status);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_mev ON transactions(mev_type) WHERE mev_type IS NOT NULL;"
        ]

        async with self.pool.acquire() as conn:
            await conn.execute(create_table_sql)

            for index_sql in create_indexes_sql:
                await conn.execute(index_sql)

        logger.info("Created transactions table and indexes")

    async def upsert_transactions(
        self,
        transactions: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Insert or update transactions in batches.

        Args:
            transactions: List of transaction dictionaries
            batch_size: Number of transactions per batch

        Returns:
            Number of transactions inserted/updated
        """
        if not transactions:
            return 0

        total_inserted = 0

        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]

            try:
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        inserted = await self._upsert_batch(conn, batch)
                        total_inserted += inserted

                logger.info(f"Inserted batch {i // batch_size + 1}: {inserted} transactions")

            except Exception as e:
                logger.error(f"Error inserting batch {i // batch_size + 1}: {e}")

        return total_inserted

    async def _upsert_batch(
        self,
        conn: asyncpg.Connection,
        batch: List[Dict[str, Any]]
    ) -> int:
        """
        Upsert a batch of transactions.

        Args:
            conn: Database connection
            batch: List of transactions

        Returns:
            Number of transactions upserted
        """
        upsert_sql = """
            INSERT INTO transactions (
                tx_hash, block_number, timestamp, wallet_address,
                dex_name, pool_address,
                token_in, amount_in, token_out, amount_out,
                gas_used, gas_price_gwei, transaction_status,
                eth_value_in, eth_value_out, usd_value_in, usd_value_out,
                mev_type, mev_damage_eth, slippage_percentage
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            ON CONFLICT (tx_hash)
            DO UPDATE SET
                dex_name = EXCLUDED.dex_name,
                pool_address = EXCLUDED.pool_address,
                eth_value_in = EXCLUDED.eth_value_in,
                eth_value_out = EXCLUDED.eth_value_out,
                usd_value_in = EXCLUDED.usd_value_in,
                usd_value_out = EXCLUDED.usd_value_out,
                mev_type = EXCLUDED.mev_type,
                mev_damage_eth = EXCLUDED.mev_damage_eth,
                slippage_percentage = EXCLUDED.slippage_percentage,
                updated_at = NOW()
        """

        values = []
        for tx in batch:
            values.append((
                tx.get("tx_hash"),
                tx.get("block_number"),
                tx.get("timestamp"),
                tx.get("wallet_address"),
                tx.get("dex_name"),
                tx.get("pool_address"),
                tx.get("token_in"),
                tx.get("amount_in"),
                tx.get("token_out"),
                tx.get("amount_out"),
                tx.get("gas_used"),
                tx.get("gas_price_gwei"),
                tx.get("transaction_status"),
                tx.get("eth_value_in"),
                tx.get("eth_value_out"),
                tx.get("usd_value_in"),
                tx.get("usd_value_out"),
                tx.get("mev_type"),
                tx.get("mev_damage_eth"),
                tx.get("slippage_percentage")
            ))

        await conn.executemany(upsert_sql, values)
        return len(values)

    async def get_wallet_transactions(
        self,
        wallet_address: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get transactions for a wallet.

        Args:
            wallet_address: Wallet address
            limit: Maximum number of transactions
            offset: Number of transactions to skip

        Returns:
            List of transaction dictionaries
        """
        query = """
            SELECT *
            FROM transactions
            WHERE wallet_address = $1
            ORDER BY timestamp DESC
            LIMIT $2 OFFSET $3
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, wallet_address, limit, offset)

        return [dict(row) for row in rows]

    async def get_transaction_count(
        self,
        wallet_address: Optional[str] = None
    ) -> int:
        """
        Get transaction count.

        Args:
            wallet_address: Optional wallet address filter

        Returns:
            Transaction count
        """
        if wallet_address:
            query = "SELECT COUNT(*) FROM transactions WHERE wallet_address = $1"
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, wallet_address)
        else:
            query = "SELECT COUNT(*) FROM transactions"
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query)

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get transaction statistics.

        Returns:
            Dictionary of statistics
        """
        query = """
            SELECT
                COUNT(*) as total_transactions,
                COUNT(DISTINCT wallet_address) as unique_wallets,
                COUNT(DISTINCT dex_name) as unique_dexs,
                SUM(CASE WHEN transaction_status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN transaction_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN mev_type IS NOT NULL THEN 1 ELSE 0 END) as mev_affected,
                AVG(CAST(gas_price_gwei AS FLOAT)) as avg_gas_price_gwei,
                SUM(CAST(eth_value_in AS FLOAT)) as total_eth_volume
            FROM transactions
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query)

        return dict(row) if row else {}

    async def delete_old_transactions(
        self,
        days_to_keep: int = 365
    ) -> int:
        """
        Delete transactions older than specified days.

        Args:
            days_to_keep: Number of days to keep

        Returns:
            Number of transactions deleted
        """
        query = """
            DELETE FROM transactions
            WHERE timestamp < NOW() - INTERVAL '$1 days'
        """

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, days_to_keep)

        deleted_count = int(result.split()[-1])
        logger.info(f"Deleted {deleted_count} old transactions")

        return deleted_count