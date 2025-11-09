"""Historical balance backfill and reconstruction.

This module provides functionality for reconstructing historical balances
from transaction history and backfilling balance snapshots.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import date, datetime, timedelta
from decimal import Decimal
import psycopg
from psycopg.rows import dict_row

from data_collection.common.logging_setup import get_logger
from data_collection.common.checkpoints import CheckpointManager
from .multicall_client import MulticallClient
from .block_timing import BlockTimingClient
from .storage import BalanceStorageService
from .pricing import PricingService

logger = get_logger(__name__)


class BackfillService:
    """Service for historical balance backfill."""

    def __init__(
        self,
        conn: psycopg.Connection,
        multicall_client: MulticallClient,
        block_timing_client: BlockTimingClient,
    ):
        """Initialize backfill service.

        Args:
            conn: Database connection
            multicall_client: Multicall client for balance queries
            block_timing_client: Block timing client
        """
        self.conn = conn
        self.multicall_client = multicall_client
        self.block_timing_client = block_timing_client
        self.storage = BalanceStorageService(conn)
        self.pricing = PricingService(conn)
        self.checkpoint_manager = CheckpointManager(conn)

    async def backfill_wallet_balances(
        self,
        wallet_address: str,
        token_addresses: List[str],
        start_date: date,
        end_date: date,
        checkpoint_key: Optional[str] = None,
    ) -> int:
        """Backfill balance snapshots for a wallet over a date range.

        Args:
            wallet_address: Wallet address
            token_addresses: List of token addresses to track
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            checkpoint_key: Optional checkpoint key for resume capability

        Returns:
            Number of snapshots created
        """
        log_context = {
            'wallet': wallet_address,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'num_tokens': len(token_addresses),
        }
        logger.info("Starting balance backfill", **log_context)

        if checkpoint_key is None:
            checkpoint_key = f"backfill_{wallet_address}_{start_date}_{end_date}"

        # Get checkpoint
        checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_key)
        last_processed_date = None

        if checkpoint and checkpoint.get('last_date'):
            last_processed_date = datetime.strptime(
                checkpoint['last_date'], "%Y-%m-%d"
            ).date()
            logger.info(f"Resuming from checkpoint: {last_processed_date}")

        # Get daily blocks for date range
        daily_blocks = await self.block_timing_client.get_daily_blocks(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time()),
            snapshot_time='end_of_day',
        )

        snapshots_created = 0
        current_date = start_date

        while current_date <= end_date:
            # Skip already processed dates
            if last_processed_date and current_date <= last_processed_date:
                current_date += timedelta(days=1)
                continue

            date_str = current_date.isoformat()

            if date_str not in daily_blocks:
                logger.warning(f"No block found for {date_str}, skipping")
                current_date += timedelta(days=1)
                continue

            block_info = daily_blocks[date_str]
            block_number = block_info['number']

            try:
                # Fetch balances at this block
                balances = await self.multicall_client.get_wallet_balances(
                    wallet_address,
                    token_addresses,
                    block_number=block_number,
                    skip_zero_balances=True,
                )

                # Get prices for this date
                prices = self.pricing.get_token_prices_batch(
                    list(balances.keys()),
                    current_date,
                )

                # Prepare balance records
                balance_records = []
                for token_address, balance in balances.items():
                    price_eth = prices.get(token_address)
                    eth_value = self.pricing.calculate_eth_value(
                        token_address,
                        balance,
                        price_eth,
                    )

                    balance_records.append({
                        'wallet_address': wallet_address,
                        'token_address': token_address,
                        'snapshot_date': current_date,
                        'block_number': block_number,
                        'balance': balance,
                        'eth_value': eth_value,
                        'price_eth': price_eth,
                    })

                # Store balances
                if balance_records:
                    self.storage.store_balance_batch(balance_records)
                    snapshots_created += len(balance_records)

                logger.info(
                    f"Backfilled {current_date}: {len(balance_records)} balances"
                )

                # Update checkpoint
                self.checkpoint_manager.update_checkpoint(
                    checkpoint_key,
                    {
                        'last_date': date_str,
                        'snapshots_created': snapshots_created,
                    },
                )

            except Exception as e:
                logger.error(f"Backfill failed for {current_date}: {e}")
                raise

            current_date += timedelta(days=1)

        logger.info(
            f"Backfill complete: {snapshots_created} snapshots created",
            **log_context,
        )

        return snapshots_created

    async def backfill_cohort_balances(
        self,
        wallet_addresses: List[str],
        token_addresses: List[str],
        start_date: date,
        end_date: date,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, int]:
        """Backfill balances for entire wallet cohort.

        Args:
            wallet_addresses: List of wallet addresses
            token_addresses: List of token addresses
            start_date: Start date
            end_date: End date
            progress_callback: Optional progress callback(completed, total)

        Returns:
            Dictionary mapping wallet addresses to snapshot counts
        """
        logger.info(
            f"Starting cohort backfill: {len(wallet_addresses)} wallets, "
            f"{start_date} to {end_date}"
        )

        results = {}

        for idx, wallet in enumerate(wallet_addresses):
            try:
                count = await self.backfill_wallet_balances(
                    wallet,
                    token_addresses,
                    start_date,
                    end_date,
                )
                results[wallet] = count

                if progress_callback:
                    progress_callback(idx + 1, len(wallet_addresses))

            except Exception as e:
                logger.error(f"Cohort backfill failed for {wallet}: {e}")
                results[wallet] = 0

        total_snapshots = sum(results.values())
        logger.info(
            f"Cohort backfill complete: {total_snapshots} total snapshots"
        )

        return results

    def reconstruct_balance_from_transactions(
        self,
        wallet_address: str,
        token_address: str,
        up_to_date: date,
    ) -> int:
        """Reconstruct balance from transaction history.

        Args:
            wallet_address: Wallet address
            token_address: Token address
            up_to_date: Reconstruct up to this date

        Returns:
            Reconstructed balance
        """
        query = """
        SELECT
            tx_type,
            token_in_address,
            token_out_address,
            amount_in,
            amount_out,
            block_timestamp
        FROM wallet_transactions
        WHERE wallet_address = %s
        AND (token_in_address = %s OR token_out_address = %s)
        AND DATE(to_timestamp(block_timestamp)) <= %s
        ORDER BY block_timestamp ASC, log_index ASC
        """

        balance = 0

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                query,
                (wallet_address, token_address, token_address, up_to_date),
            )
            transactions = cur.fetchall()

            for tx in transactions:
                # Token received
                if tx['token_in_address'] == token_address and tx['amount_in']:
                    balance += int(tx['amount_in'])

                # Token sent
                if tx['token_out_address'] == token_address and tx['amount_out']:
                    balance -= int(tx['amount_out'])

        logger.debug(
            f"Reconstructed balance for {wallet_address}/{token_address}: {balance}"
        )

        return balance

    async def backfill_from_transactions(
        self,
        wallet_address: str,
        start_date: date,
        end_date: date,
    ) -> int:
        """Backfill balance snapshots by reconstructing from transactions.

        This method doesn't require RPC calls - it builds balances purely
        from the transaction history.

        Args:
            wallet_address: Wallet address
            start_date: Start date
            end_date: End date

        Returns:
            Number of snapshots created
        """
        logger.info(
            f"Transaction-based backfill for {wallet_address}: "
            f"{start_date} to {end_date}"
        )

        # Get all tokens the wallet has interacted with
        tokens_query = """
        SELECT DISTINCT
            COALESCE(token_in_address, token_out_address) AS token_address
        FROM wallet_transactions
        WHERE wallet_address = %s
        AND DATE(to_timestamp(block_timestamp)) BETWEEN %s AND %s
        """

        with self.conn.cursor() as cur:
            cur.execute(tokens_query, (wallet_address, start_date, end_date))
            token_addresses = [row[0] for row in cur.fetchall() if row[0]]

        logger.info(f"Found {len(token_addresses)} tokens to reconstruct")

        # Get daily blocks
        daily_blocks = await self.block_timing_client.get_daily_blocks(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time()),
            snapshot_time='end_of_day',
        )

        snapshots_created = 0
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.isoformat()

            if date_str not in daily_blocks:
                logger.warning(f"No block found for {date_str}")
                current_date += timedelta(days=1)
                continue

            block_number = daily_blocks[date_str]['number']

            # Reconstruct balances for all tokens
            balance_records = []

            for token_address in token_addresses:
                balance = self.reconstruct_balance_from_transactions(
                    wallet_address,
                    token_address,
                    current_date,
                )

                # Skip zero balances
                if balance == 0:
                    continue

                # Get price
                price_eth = self.pricing.get_token_price_eth(
                    token_address,
                    current_date,
                )
                eth_value = self.pricing.calculate_eth_value(
                    token_address,
                    balance,
                    price_eth,
                )

                balance_records.append({
                    'wallet_address': wallet_address,
                    'token_address': token_address,
                    'snapshot_date': current_date,
                    'block_number': block_number,
                    'balance': balance,
                    'eth_value': eth_value,
                    'price_eth': price_eth,
                })

            # Store balances
            if balance_records:
                self.storage.store_balance_batch(balance_records)
                snapshots_created += len(balance_records)

            logger.debug(
                f"Transaction backfill {current_date}: {len(balance_records)} balances"
            )

            current_date += timedelta(days=1)

        logger.info(
            f"Transaction backfill complete: {snapshots_created} snapshots"
        )

        return snapshots_created

    def get_backfill_progress(self, checkpoint_key: str) -> Optional[Dict]:
        """Get backfill progress from checkpoint.

        Args:
            checkpoint_key: Checkpoint key

        Returns:
            Progress information or None
        """
        checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_key)

        if checkpoint:
            return {
                'last_date': checkpoint.get('last_date'),
                'snapshots_created': checkpoint.get('snapshots_created', 0),
                'updated_at': checkpoint.get('updated_at'),
            }

        return None

    def clear_backfill_checkpoint(self, checkpoint_key: str) -> None:
        """Clear backfill checkpoint.

        Args:
            checkpoint_key: Checkpoint key
        """
        self.checkpoint_manager.delete_checkpoint(checkpoint_key)
        logger.info(f"Cleared checkpoint: {checkpoint_key}")