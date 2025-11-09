import hashlib
import json
import os
import time
from datetime import date, datetime
from typing import Optional, Dict, Any, Set, List
from pathlib import Path
from .db import get_cursor, execute_with_retry
from .logging_setup import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages collection checkpoints with resume capability and idempotency"""

    def __init__(self, collection_type: str):
        self.collection_type = collection_type
        self._dedup_cache: Set[str] = set()  # In-memory deduplication cache
        self._staging_dir = Path("./tmp") / collection_type
        self._staging_dir.mkdir(parents=True, exist_ok=True)

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Read the last checkpoint for this collection type"""
        query = """
            SELECT * FROM collection_checkpoints
            WHERE collection_type = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """
        result = execute_with_retry(query, (self.collection_type,))
        if result:
            checkpoint = result[0]
            logger.log_operation(
                operation="get_checkpoint",
                params={"collection_type": self.collection_type},
                status="found",
                message=f"Found checkpoint at block {checkpoint.get('last_processed_block')}"
            )
            return checkpoint

    def load_checkpoint(self, checkpoint_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a specific key or default to collection type"""
        key = checkpoint_key if checkpoint_key else self.collection_type
        query = """
            SELECT * FROM collection_checkpoints
            WHERE collection_type = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """
        result = execute_with_retry(query, (key,))
        if result:
            return result[0]
        return None

    def save_checkpoint(self, checkpoint_key: str, checkpoint_data: Dict[str, Any]) -> None:
        """Save checkpoint data"""
        # Temporarily change collection_type for this save operation
        original_type = self.collection_type
        self.collection_type = checkpoint_key

        # Get block number, use None if 0 or not provided (to satisfy constraint that block > 0)
        block = checkpoint_data.get('last_processed_block')
        if block is not None and block <= 0:
            block = None

        # Map status to valid checkpoint statuses
        status = checkpoint_data.get('status', 'running')
        if status == 'in_progress':
            status = 'running'

        # Use update_checkpoint method
        self.update_checkpoint(
            last_processed_block=block,
            last_processed_date=checkpoint_data.get('last_processed_date'),
            records_collected=checkpoint_data.get('records_collected', 0),
            status=status
        )

        # Restore original collection_type
        self.collection_type = original_type

        logger.log_operation(
            operation="save_checkpoint",
            params={"collection_type": checkpoint_key},
            status="saved",
            message=f"Checkpoint saved for {checkpoint_key}"
        )

    def clear_checkpoint(self, checkpoint_key: Optional[str] = None) -> None:
        """Clear checkpoint for a specific key"""
        key = checkpoint_key if checkpoint_key else self.collection_type
        query = "DELETE FROM collection_checkpoints WHERE collection_type = %s"
        execute_with_retry(query, (key,), fetch=False)
        logger.log_operation(
            operation="clear_checkpoint",
            params={"collection_type": key},
            status="cleared",
            message=f"Checkpoint cleared for {key}"
        )

    def get_last_checkpoint_old(self) -> Optional[Dict[str, Any]]:
        """Read the last checkpoint for this collection type (kept for compatibility)"""
        logger.log_operation(
            operation="get_checkpoint",
            params={"collection_type": self.collection_type},
            status="not_found",
            message="No checkpoint found, starting from beginning"
        )
        return None

    def update_checkpoint(
        self,
        last_processed_block: Optional[int] = None,
        last_processed_date: Optional[date] = None,
        records_collected: int = 0,
        status: str = "running"
    ) -> None:
        """Atomically update checkpoint only on successful completion"""
        query = """
            INSERT INTO collection_checkpoints (
                collection_type,
                last_processed_block,
                last_processed_date,
                records_collected,
                status,
                created_at,
                updated_at
            ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (collection_type)
            DO UPDATE SET
                last_processed_block = EXCLUDED.last_processed_block,
                last_processed_date = EXCLUDED.last_processed_date,
                records_collected = collection_checkpoints.records_collected + EXCLUDED.records_collected,
                status = EXCLUDED.status,
                updated_at = NOW()
        """

        execute_with_retry(
            query,
            (
                self.collection_type,
                last_processed_block,
                last_processed_date,
                records_collected,
                status
            ),
            fetch=False
        )

        logger.log_operation(
            operation="update_checkpoint",
            params={
                "collection_type": self.collection_type,
                "block": last_processed_block,
                "date": str(last_processed_date) if last_processed_date else None
            },
            status="completed",
            message=f"Checkpoint updated: {records_collected} records, status={status}"
        )

    def get_resume_position(self) -> Dict[str, Any]:
        """Get the position to resume from based on last checkpoint"""
        checkpoint = self.get_last_checkpoint()

        if checkpoint and checkpoint.get('status') != 'failed':
            return {
                'start_block': checkpoint.get('last_processed_block', 0) + 1,
                'start_date': checkpoint.get('last_processed_date'),
                'total_collected': checkpoint.get('records_collected', 0)
            }
        else:
            # Start from beginning
            return {
                'start_block': 0,
                'start_date': None,
                'total_collected': 0
            }

    def is_duplicate(self, record_hash: str) -> bool:
        """Check if record has already been processed using in-memory cache"""
        if record_hash in self._dedup_cache:
            return True

        # Add to cache (with size limit to prevent memory issues)
        if len(self._dedup_cache) > 100000:
            # Clear oldest 20% when cache is full
            cache_list = list(self._dedup_cache)
            self._dedup_cache = set(cache_list[-80000:])

        self._dedup_cache.add(record_hash)
        return False

    def compute_record_hash(self, record: Dict[str, Any]) -> str:
        """Compute hash of record for deduplication"""
        # Sort keys for consistent hashing
        record_str = json.dumps(record, sort_keys=True, default=str)
        return hashlib.md5(record_str.encode()).hexdigest()

    def write_staging(self, data: List[Dict[str, Any]], batch_id: str) -> Path:
        """Write data to staging file for partial write protection"""
        staging_file = self._staging_dir / f"batch_{batch_id}.json"

        with open(staging_file, 'w') as f:
            json.dump(data, f)

        logger.log_operation(
            operation="write_staging",
            params={"batch_id": batch_id, "records": len(data)},
            status="completed",
            message=f"Wrote {len(data)} records to staging"
        )

        return staging_file

    def commit_staging(self, staging_file: Path) -> None:
        """Move staging file to final location after successful processing"""
        # In a real implementation, this would move to outputs/
        # For now, just delete the staging file
        if staging_file.exists():
            staging_file.unlink()

    def cleanup_staging(self) -> None:
        """Clean up old staging files"""
        cutoff_time = time.time() - (24 * 3600)  # 24 hours ago

        for file in self._staging_dir.glob("batch_*.json"):
            if file.stat().st_mtime < cutoff_time:
                file.unlink()
                logger.info(f"Cleaned up old staging file: {file.name}")


def ensure_checkpoint_table() -> None:
    """Ensure the checkpoint table exists with all constraints"""
    # This is already handled by schema.sql, but we keep this for compatibility
    query = """
    CREATE TABLE IF NOT EXISTS collection_checkpoints (
        id SERIAL PRIMARY KEY,
        collection_type VARCHAR(50) NOT NULL UNIQUE,
        last_processed_block BIGINT CHECK (last_processed_block > 0),
        last_processed_date DATE,
        records_collected INT DEFAULT 0 CHECK (records_collected >= 0),
        status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    execute_with_retry(query, fetch=False)


# Backward compatibility functions
def get_checkpoint(collection_type: str) -> Optional[dict]:
    """Legacy function for getting checkpoint"""
    manager = CheckpointManager(collection_type)
    return manager.get_last_checkpoint()


def upsert_checkpoint(
    collection_type: str,
    last_processed_block: Optional[int] = None,
    last_processed_date: Optional[str] = None,
    records_collected: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Legacy function for updating checkpoint"""
    manager = CheckpointManager(collection_type)
    manager.update_checkpoint(
        last_processed_block=last_processed_block,
        last_processed_date=datetime.strptime(last_processed_date, '%Y-%m-%d').date() if last_processed_date else None,
        records_collected=records_collected or 0,
        status=status or "running"
    )
