"""
Checkpoint and resume system for transaction extraction.

Provides:
- Per-wallet checkpoint tracking
- Atomic checkpoint updates
- Resume logic for interrupted processes
- Data deduplication
- Checkpoint validation
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class TransactionCheckpointManager:
    """Manages checkpoints for transaction extraction."""

    def __init__(self, db_connection: Any):
        """
        Initialize checkpoint manager.

        Args:
            db_connection: Database connection
        """
        self.db = db_connection
        self.collection_type = "transaction_extraction"

    async def save_checkpoint(
        self,
        wallet_address: str,
        last_block: int,
        transactions_extracted: int,
        status: str = "in_progress"
    ) -> None:
        """
        Save extraction progress for a wallet.

        Args:
            wallet_address: Wallet address
            last_block: Last processed block
            transactions_extracted: Number of transactions extracted
            status: Extraction status
        """
        metadata = {
            "wallet_address": wallet_address,
            "last_processed_block": last_block,
            "transactions_count": transactions_extracted,
            "last_updated": datetime.now().isoformat()
        }

        query = """
            INSERT INTO collection_checkpoints
            (collection_type, last_processed_block, records_collected, status, metadata, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (collection_type, COALESCE((metadata->>'wallet_address')::text, ''))
            DO UPDATE SET
                last_processed_block = EXCLUDED.last_processed_block,
                records_collected = EXCLUDED.records_collected,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """

        await self.db.execute(
            query,
            self.collection_type,
            last_block,
            transactions_extracted,
            status,
            json.dumps(metadata)
        )

        logger.debug(f"Saved checkpoint for wallet {wallet_address} at block {last_block}")

    async def resume_from_checkpoint(
        self,
        wallet_address: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resume extraction from last checkpoint.

        Args:
            wallet_address: Optional wallet address to resume for specific wallet

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        if wallet_address:
            query = """
                SELECT last_processed_block, records_collected, metadata, status
                FROM collection_checkpoints
                WHERE collection_type = $1
                AND metadata->>'wallet_address' = $2
                AND status = 'in_progress'
            """
            result = await self.db.fetchrow(query, self.collection_type, wallet_address)
        else:
            query = """
                SELECT last_processed_block, records_collected, metadata, status
                FROM collection_checkpoints
                WHERE collection_type = $1
                AND status = 'in_progress'
                ORDER BY updated_at DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, self.collection_type)

        if result:
            metadata = json.loads(result["metadata"])
            return {
                "resume_block": result["last_processed_block"],
                "records_collected": result["records_collected"],
                "wallet_address": metadata.get("wallet_address"),
                "status": result["status"]
            }

        return None

    async def mark_complete(self, wallet_address: str) -> None:
        """
        Mark wallet extraction as complete.

        Args:
            wallet_address: Wallet address
        """
        query = """
            UPDATE collection_checkpoints
            SET status = 'completed', updated_at = NOW()
            WHERE collection_type = $1
            AND metadata->>'wallet_address' = $2
        """

        await self.db.execute(query, self.collection_type, wallet_address)
        logger.info(f"Marked wallet {wallet_address} extraction as complete")

    async def get_incomplete_wallets(self) -> List[str]:
        """
        Get list of wallets with incomplete extraction.

        Returns:
            List of wallet addresses
        """
        query = """
            SELECT metadata->>'wallet_address' as wallet_address
            FROM collection_checkpoints
            WHERE collection_type = $1
            AND status = 'in_progress'
        """

        rows = await self.db.fetch(query, self.collection_type)
        return [row["wallet_address"] for row in rows if row["wallet_address"]]

    async def validate_checkpoint(
        self,
        wallet_address: str,
        expected_block: int
    ) -> bool:
        """
        Validate checkpoint integrity.

        Args:
            wallet_address: Wallet address
            expected_block: Expected block number

        Returns:
            True if checkpoint is valid
        """
        query = """
            SELECT last_processed_block
            FROM collection_checkpoints
            WHERE collection_type = $1
            AND metadata->>'wallet_address' = $2
        """

        result = await self.db.fetchrow(query, self.collection_type, wallet_address)

        if not result:
            return False

        return result["last_processed_block"] <= expected_block

    async def deduplicate_transactions(
        self,
        wallet_address: str,
        tx_hashes: List[str]
    ) -> List[str]:
        """
        Remove duplicate transactions already in database.

        Args:
            wallet_address: Wallet address
            tx_hashes: List of transaction hashes

        Returns:
            List of new transaction hashes not in database
        """
        if not tx_hashes:
            return []

        query = """
            SELECT tx_hash
            FROM transactions
            WHERE wallet_address = $1
            AND tx_hash = ANY($2)
        """

        existing_rows = await self.db.fetch(query, wallet_address, tx_hashes)
        existing_hashes = {row["tx_hash"] for row in existing_rows}

        new_hashes = [tx_hash for tx_hash in tx_hashes if tx_hash not in existing_hashes]

        logger.info(
            f"Deduplication for {wallet_address}: "
            f"{len(tx_hashes)} total, {len(existing_hashes)} existing, {len(new_hashes)} new"
        )

        return new_hashes

    async def get_checkpoint_stats(self) -> Dict[str, Any]:
        """
        Get checkpoint statistics.

        Returns:
            Dictionary of checkpoint stats
        """
        query = """
            SELECT
                COUNT(*) as total_checkpoints,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                SUM(records_collected) as total_records
            FROM collection_checkpoints
            WHERE collection_type = $1
        """

        result = await self.db.fetchrow(query, self.collection_type)

        return {
            "total_checkpoints": result["total_checkpoints"] or 0,
            "completed": result["completed"] or 0,
            "in_progress": result["in_progress"] or 0,
            "total_records": result["total_records"] or 0
        }