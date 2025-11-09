"""Balance storage and position management.

This module handles efficient storage of balance snapshots with zero balance
optimization, incremental position tracking, and bulk insert operations.
"""

from typing import Dict, List, Optional, Tuple, Set
from datetime import date, datetime
import psycopg
from psycopg.rows import dict_row
from decimal import Decimal

from data_collection.common.logging_setup import get_logger
from .schema import ensure_partition_exists

logger = get_logger(__name__)


class BalanceStorageService:
    """Service for storing and managing wallet balance snapshots."""

    def __init__(self, conn: psycopg.Connection):
        """Initialize storage service.

        Args:
            conn: Database connection
        """
        self.conn = conn

    def store_balance_snapshot(
        self,
        wallet_address: str,
        token_address: str,
        snapshot_date: date,
        block_number: int,
        balance: int,
        eth_value: Optional[Decimal] = None,
        price_eth: Optional[Decimal] = None,
    ) -> None:
        """Store a single balance snapshot.

        Args:
            wallet_address: Wallet address
            token_address: Token address (or 'ETH')
            snapshot_date: Date of snapshot
            block_number: Block number at snapshot
            balance: Token balance (raw amount)
            eth_value: ETH-denominated value
            price_eth: Token price in ETH
        """
        # Ensure partition exists
        ensure_partition_exists(self.conn, snapshot_date.isoformat())

        insert_sql = """
        INSERT INTO wallet_balances (
            wallet_address, token_address, snapshot_date, block_number,
            balance, eth_value, price_eth
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (wallet_address, token_address, snapshot_date, block_number)
        DO UPDATE SET
            balance = EXCLUDED.balance,
            eth_value = EXCLUDED.eth_value,
            price_eth = EXCLUDED.price_eth,
            updated_at = NOW()
        """

        with self.conn.cursor() as cur:
            cur.execute(
                insert_sql,
                (
                    wallet_address,
                    token_address,
                    snapshot_date,
                    block_number,
                    balance,
                    eth_value,
                    price_eth,
                ),
            )
            self.conn.commit()

    def store_balance_batch(
        self,
        balances: List[Dict],
        batch_size: int = 1000,
    ) -> int:
        """Store multiple balance snapshots efficiently.

        Args:
            balances: List of balance dictionaries with keys:
                - wallet_address
                - token_address
                - snapshot_date
                - block_number
                - balance
                - eth_value (optional)
                - price_eth (optional)
            batch_size: Number of records per batch

        Returns:
            Number of records inserted/updated
        """
        if not balances:
            return 0

        # Ensure partition exists for all dates
        unique_dates = set(b['snapshot_date'] for b in balances)
        for snap_date in unique_dates:
            if isinstance(snap_date, date):
                ensure_partition_exists(self.conn, snap_date.isoformat())
            else:
                ensure_partition_exists(self.conn, snap_date)

        total_inserted = 0

        insert_sql = """
        INSERT INTO wallet_balances (
            wallet_address, token_address, snapshot_date, block_number,
            balance, eth_value, price_eth
        ) VALUES %s
        ON CONFLICT (wallet_address, token_address, snapshot_date, block_number)
        DO UPDATE SET
            balance = EXCLUDED.balance,
            eth_value = EXCLUDED.eth_value,
            price_eth = EXCLUDED.price_eth,
            updated_at = NOW()
        """

        # Process in batches
        for i in range(0, len(balances), batch_size):
            batch = balances[i:i + batch_size]

            # Prepare values for batch insert
            values = [
                (
                    b['wallet_address'],
                    b['token_address'],
                    b['snapshot_date'],
                    b['block_number'],
                    b['balance'],
                    b.get('eth_value'),
                    b.get('price_eth'),
                )
                for b in batch
            ]

            with self.conn.cursor() as cur:
                # Use execute_values for efficient batch insert
                from psycopg.sql import SQL, Identifier

                # Build values placeholder
                values_template = "(%s, %s, %s, %s, %s, %s, %s)"
                values_clause = ", ".join([values_template] * len(values))

                full_sql = insert_sql.replace("%s", values_clause, 1)

                # Flatten values
                flat_values = []
                for val_tuple in values:
                    flat_values.extend(val_tuple)

                cur.execute(full_sql, flat_values)
                total_inserted += cur.rowcount

            self.conn.commit()

            logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch)} records")

        logger.info(f"Stored {total_inserted} balance snapshots")
        return total_inserted

    def get_previous_balances(
        self,
        wallet_address: str,
        before_date: date,
        token_addresses: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Get the most recent balances before a given date.

        Args:
            wallet_address: Wallet address
            before_date: Get balances before this date
            token_addresses: Optional list of token addresses to filter

        Returns:
            Dictionary mapping token addresses to balances
        """
        query = """
        SELECT DISTINCT ON (token_address)
            token_address, balance
        FROM wallet_balances
        WHERE wallet_address = %s
        AND snapshot_date < %s
        """

        params = [wallet_address, before_date]

        if token_addresses:
            query += " AND token_address = ANY(%s)"
            params.append(token_addresses)

        query += """
        ORDER BY token_address, snapshot_date DESC, block_number DESC
        """

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            results = cur.fetchall()

        return {row['token_address']: int(row['balance']) for row in results}

    def update_position_status(
        self,
        wallet_address: str,
        token_address: str,
        snapshot_date: date,
        status: str,
    ) -> None:
        """Update or create position tracking record.

        Args:
            wallet_address: Wallet address
            token_address: Token address
            snapshot_date: Date of position change
            status: Position status ('active' or 'closed')
        """
        upsert_sql = """
        INSERT INTO wallet_positions (
            wallet_address, token_address, first_seen_date, last_seen_date, status
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (wallet_address, token_address)
        DO UPDATE SET
            last_seen_date = EXCLUDED.last_seen_date,
            status = EXCLUDED.status,
            updated_at = NOW()
        """

        with self.conn.cursor() as cur:
            cur.execute(
                upsert_sql,
                (wallet_address, token_address, snapshot_date, snapshot_date, status),
            )
            self.conn.commit()

    def analyze_position_changes(
        self,
        wallet_address: str,
        current_balances: Dict[str, int],
        snapshot_date: date,
    ) -> Dict[str, dict]:
        """Analyze position changes compared to previous snapshot.

        Args:
            wallet_address: Wallet address
            current_balances: Current token balances
            snapshot_date: Current snapshot date

        Returns:
            Dictionary of changes by token with type: 'new', 'changed', 'closed'
        """
        # Get previous balances
        previous_balances = self.get_previous_balances(wallet_address, snapshot_date)

        changes = {}
        all_tokens = set(previous_balances.keys()) | set(current_balances.keys())

        for token in all_tokens:
            prev_balance = previous_balances.get(token, 0)
            curr_balance = current_balances.get(token, 0)

            if prev_balance == 0 and curr_balance > 0:
                changes[token] = {
                    'type': 'new_position',
                    'balance': curr_balance,
                    'previous_balance': 0,
                }
                # Update position tracking
                self.update_position_status(wallet_address, token, snapshot_date, 'active')

            elif prev_balance > 0 and curr_balance == 0:
                changes[token] = {
                    'type': 'position_closed',
                    'balance': 0,
                    'previous_balance': prev_balance,
                }
                # Update position tracking
                self.update_position_status(wallet_address, token, snapshot_date, 'closed')

            elif prev_balance != curr_balance and curr_balance > 0:
                changes[token] = {
                    'type': 'position_change',
                    'balance': curr_balance,
                    'previous_balance': prev_balance,
                    'change': curr_balance - prev_balance,
                }
                # Update position tracking
                self.update_position_status(wallet_address, token, snapshot_date, 'active')

        logger.info(
            f"Position changes for {wallet_address}: "
            f"{sum(1 for c in changes.values() if c['type'] == 'new_position')} new, "
            f"{sum(1 for c in changes.values() if c['type'] == 'position_closed')} closed, "
            f"{sum(1 for c in changes.values() if c['type'] == 'position_change')} changed"
        )

        return changes

    def create_snapshot_metadata(
        self,
        snapshot_date: date,
        block_number: int,
        block_timestamp: int,
    ) -> int:
        """Create snapshot metadata record.

        Args:
            snapshot_date: Snapshot date
            block_number: Block number
            block_timestamp: Block timestamp

        Returns:
            Snapshot metadata ID
        """
        insert_sql = """
        INSERT INTO balance_snapshots_meta (
            snapshot_date, block_number, block_timestamp, status
        ) VALUES (%s, %s, %s, 'pending')
        ON CONFLICT (snapshot_date)
        DO UPDATE SET
            block_number = EXCLUDED.block_number,
            block_timestamp = EXCLUDED.block_timestamp,
            status = 'pending',
            updated_at = NOW()
        RETURNING id
        """

        with self.conn.cursor() as cur:
            cur.execute(insert_sql, (snapshot_date, block_number, block_timestamp))
            snapshot_id = cur.fetchone()[0]
            self.conn.commit()

        return snapshot_id

    def update_snapshot_metadata(
        self,
        snapshot_date: date,
        status: str,
        wallets_processed: Optional[int] = None,
        tokens_tracked: Optional[int] = None,
        total_balances: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update snapshot metadata.

        Args:
            snapshot_date: Snapshot date
            status: Status ('pending', 'processing', 'completed', 'failed')
            wallets_processed: Number of wallets processed
            tokens_tracked: Number of tokens tracked
            total_balances: Total balance records
            error_message: Error message if failed
        """
        update_sql = """
        UPDATE balance_snapshots_meta
        SET status = %s,
            updated_at = NOW()
        """

        params = [status]

        if wallets_processed is not None:
            update_sql += ", wallets_processed = %s"
            params.append(wallets_processed)

        if tokens_tracked is not None:
            update_sql += ", tokens_tracked = %s"
            params.append(tokens_tracked)

        if total_balances is not None:
            update_sql += ", total_balances = %s"
            params.append(total_balances)

        if error_message is not None:
            update_sql += ", error_message = %s"
            params.append(error_message)

        if status == 'processing':
            update_sql += ", started_at = NOW()"
        elif status in ['completed', 'failed']:
            update_sql += ", completed_at = NOW()"

        update_sql += " WHERE snapshot_date = %s"
        params.append(snapshot_date)

        with self.conn.cursor() as cur:
            cur.execute(update_sql, params)
            self.conn.commit()

    def get_active_tokens_for_wallet(
        self,
        wallet_address: str,
    ) -> List[str]:
        """Get list of tokens with active positions for a wallet.

        Args:
            wallet_address: Wallet address

        Returns:
            List of token addresses with active positions
        """
        query = """
        SELECT token_address
        FROM wallet_positions
        WHERE wallet_address = %s
        AND status = 'active'
        ORDER BY last_seen_date DESC
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (wallet_address,))
            return [row[0] for row in cur.fetchall()]

    def get_balances_for_date(
        self,
        wallet_address: str,
        snapshot_date: date,
    ) -> Dict[str, Dict]:
        """Get all balances for a wallet on a specific date.

        Args:
            wallet_address: Wallet address
            snapshot_date: Snapshot date

        Returns:
            Dictionary of token balances with metadata
        """
        query = """
        SELECT token_address, balance, eth_value, price_eth, block_number
        FROM wallet_balances
        WHERE wallet_address = %s
        AND snapshot_date = %s
        ORDER BY token_address
        """

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (wallet_address, snapshot_date))
            results = cur.fetchall()

        return {
            row['token_address']: {
                'balance': int(row['balance']),
                'eth_value': row['eth_value'],
                'price_eth': row['price_eth'],
                'block_number': row['block_number'],
            }
            for row in results
        }

    def get_snapshot_statistics(
        self,
        snapshot_date: date,
    ) -> Optional[Dict]:
        """Get statistics for a snapshot.

        Args:
            snapshot_date: Snapshot date

        Returns:
            Dictionary with snapshot statistics or None if not found
        """
        query = """
        SELECT
            snapshot_date,
            block_number,
            block_timestamp,
            wallets_processed,
            tokens_tracked,
            total_balances,
            status,
            started_at,
            completed_at,
            error_message
        FROM balance_snapshots_meta
        WHERE snapshot_date = %s
        """

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (snapshot_date,))
            result = cur.fetchone()

        return dict(result) if result else None