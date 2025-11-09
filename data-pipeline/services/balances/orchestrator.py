"""Balance snapshot collection orchestrator.

This module provides the main orchestration logic for daily balance snapshot
collection with quality assurance, monitoring, and error handling.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import date, datetime
from decimal import Decimal
import psycopg

from data_collection.common.logging_setup import get_logger
from .multicall_client import MulticallClient
from .block_timing import BlockTimingClient
from .storage import BalanceStorageService
from .pricing import PricingService
from .validation import BalanceValidationService
from .backfill import BackfillService

logger = get_logger(__name__)


class BalanceSnapshotOrchestrator:
    """Orchestrator for balance snapshot collection workflow."""

    def __init__(
        self,
        conn: psycopg.Connection,
        multicall_client: MulticallClient,
        block_timing_client: BlockTimingClient,
    ):
        """Initialize orchestrator.

        Args:
            conn: Database connection
            multicall_client: Multicall client
            block_timing_client: Block timing client
        """
        self.conn = conn
        self.multicall_client = multicall_client
        self.block_timing_client = block_timing_client
        self.storage = BalanceStorageService(conn)
        self.pricing = PricingService(conn)
        self.validation = BalanceValidationService(conn)
        self.backfill = BackfillService(conn, multicall_client, block_timing_client)

    async def collect_daily_snapshot(
        self,
        wallet_addresses: List[str],
        token_addresses: List[str],
        snapshot_date: Optional[date] = None,
        validate_sample: bool = True,
        sample_size: int = 10,
    ) -> Dict:
        """Collect daily balance snapshots for wallet cohort.

        Args:
            wallet_addresses: List of wallet addresses
            token_addresses: List of token addresses to track
            snapshot_date: Date for snapshot (default: yesterday)
            validate_sample: Whether to validate a sample of results
            sample_size: Number of wallets to sample for validation

        Returns:
            Collection results summary
        """
        if snapshot_date is None:
            snapshot_date = date.today() - timedelta(days=1)

        logger.info(
            f"Starting daily snapshot collection: {len(wallet_addresses)} wallets, "
            f"{len(token_addresses)} tokens, date={snapshot_date}"
        )

        # Get block for snapshot time
        snapshot_datetime = datetime.combine(snapshot_date, datetime.min.time())
        block_info = await self.block_timing_client.get_end_of_day_block(
            snapshot_datetime
        )
        block_number = block_info['number']

        logger.info(f"Snapshot block: {block_number} at {snapshot_date}")

        # Create snapshot metadata
        snapshot_id = self.storage.create_snapshot_metadata(
            snapshot_date,
            block_number,
            block_info['timestamp'],
        )

        # Update status to processing
        self.storage.update_snapshot_metadata(
            snapshot_date,
            status='processing',
        )

        total_balances = 0
        failed_wallets = []

        try:
            # Collect balances for all wallets
            for idx, wallet in enumerate(wallet_addresses):
                try:
                    # Fetch balances
                    balances = await self.multicall_client.get_wallet_balances(
                        wallet,
                        token_addresses,
                        block_number=block_number,
                        skip_zero_balances=True,
                    )

                    if not balances:
                        logger.debug(f"No balances for {wallet}")
                        continue

                    # Enrich with prices
                    enriched = self.pricing.enrich_balances_with_prices(
                        balances,
                        snapshot_date,
                        use_fallback=True,
                    )

                    # Analyze position changes
                    self.storage.analyze_position_changes(
                        wallet,
                        balances,
                        snapshot_date,
                    )

                    # Prepare balance records
                    balance_records = []
                    for token_address, data in enriched.items():
                        balance_records.append({
                            'wallet_address': wallet,
                            'token_address': token_address,
                            'snapshot_date': snapshot_date,
                            'block_number': block_number,
                            'balance': data['balance'],
                            'eth_value': data['eth_value'],
                            'price_eth': data['price_eth'],
                        })

                    # Store balances
                    if balance_records:
                        count = self.storage.store_balance_batch(balance_records)
                        total_balances += count

                    if (idx + 1) % 10 == 0:
                        logger.info(
                            f"Progress: {idx + 1}/{len(wallet_addresses)} wallets processed"
                        )

                except Exception as e:
                    logger.error(f"Failed to process wallet {wallet}: {e}")
                    failed_wallets.append(wallet)

            # Update snapshot metadata
            self.storage.update_snapshot_metadata(
                snapshot_date,
                status='completed',
                wallets_processed=len(wallet_addresses) - len(failed_wallets),
                tokens_tracked=len(token_addresses),
                total_balances=total_balances,
            )

            # Validate sample if requested
            validation_report = None
            if validate_sample and sample_size > 0:
                logger.info(f"Validating sample of {sample_size} wallets")
                validation_report = self.validation.generate_validation_report(
                    snapshot_date,
                    sample_size=sample_size,
                )

            logger.info(
                f"Snapshot collection complete: {total_balances} balances, "
                f"{len(failed_wallets)} failures"
            )

            return {
                'snapshot_date': str(snapshot_date),
                'block_number': block_number,
                'wallets_processed': len(wallet_addresses) - len(failed_wallets),
                'wallets_failed': len(failed_wallets),
                'total_balances': total_balances,
                'validation_report': validation_report,
                'failed_wallets': failed_wallets,
            }

        except Exception as e:
            logger.error(f"Snapshot collection failed: {e}")
            self.storage.update_snapshot_metadata(
                snapshot_date,
                status='failed',
                error_message=str(e),
            )
            raise

    async def run_quality_checks(
        self,
        snapshot_date: date,
    ) -> Dict:
        """Run comprehensive quality checks on a snapshot.

        Args:
            snapshot_date: Snapshot date to check

        Returns:
            Quality check results
        """
        logger.info(f"Running quality checks for {snapshot_date}")

        results = {
            'snapshot_date': str(snapshot_date),
            'checks': {},
            'overall_status': 'passed',
        }

        # Check 1: Data completeness
        logger.info("Checking data completeness")
        completeness = self.validation.validate_snapshot_completeness(snapshot_date)
        results['checks']['completeness'] = completeness

        if not completeness['is_complete']:
            results['overall_status'] = 'warning'

        # Check 2: Detect outliers
        logger.info("Detecting outliers")
        outliers = self.validation.detect_balance_outliers(snapshot_date)
        results['checks']['outliers'] = {
            'count': len(outliers),
            'outliers': outliers[:10],  # Top 10
        }

        if len(outliers) > 100:
            results['overall_status'] = 'warning'

        # Check 3: Validate sample
        logger.info("Validating balance sample")
        validation_report = self.validation.generate_validation_report(
            snapshot_date,
            sample_size=20,
        )
        results['checks']['validation'] = validation_report

        if validation_report['validation_summary']['validation_rate'] < 95:
            results['overall_status'] = 'failed'

        # Check 4: Pricing coverage
        logger.info("Checking pricing coverage")
        pricing_stats = self._get_pricing_stats(snapshot_date)
        results['checks']['pricing'] = pricing_stats

        if pricing_stats['coverage_rate'] < 80:
            results['overall_status'] = 'warning'

        logger.info(f"Quality checks complete: {results['overall_status']}")
        return results

    def _get_pricing_stats(self, snapshot_date: date) -> Dict:
        """Get pricing statistics for a snapshot.

        Args:
            snapshot_date: Snapshot date

        Returns:
            Pricing statistics
        """
        query = """
        SELECT
            COUNT(*) AS total_balances,
            COUNT(price_eth) AS priced_balances,
            COUNT(eth_value) AS valued_balances,
            SUM(CASE WHEN price_eth IS NULL THEN 1 ELSE 0 END) AS missing_prices
        FROM wallet_balances
        WHERE snapshot_date = %s
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (snapshot_date,))
            row = cur.fetchone()

            total = row[0] if row else 0
            priced = row[1] if row else 0
            valued = row[2] if row else 0
            missing = row[3] if row else 0

            coverage_rate = (priced / total * 100) if total > 0 else 0

            return {
                'total_balances': total,
                'priced_balances': priced,
                'valued_balances': valued,
                'missing_prices': missing,
                'coverage_rate': round(coverage_rate, 2),
            }

    def generate_monitoring_report(
        self,
        start_date: date,
        end_date: date,
    ) -> Dict:
        """Generate monitoring report for date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Monitoring report
        """
        logger.info(f"Generating monitoring report: {start_date} to {end_date}")

        # Get snapshot statistics
        query = """
        SELECT
            snapshot_date,
            block_number,
            wallets_processed,
            tokens_tracked,
            total_balances,
            status,
            started_at,
            completed_at,
            EXTRACT(EPOCH FROM (completed_at - started_at)) AS duration_seconds
        FROM balance_snapshots_meta
        WHERE snapshot_date BETWEEN %s AND %s
        ORDER BY snapshot_date DESC
        """

        snapshots = []
        with self.conn.cursor() as cur:
            cur.execute(query, (start_date, end_date))
            for row in cur.fetchall():
                snapshots.append({
                    'snapshot_date': str(row[0]),
                    'block_number': row[1],
                    'wallets_processed': row[2],
                    'tokens_tracked': row[3],
                    'total_balances': row[4],
                    'status': row[5],
                    'duration_seconds': float(row[8]) if row[8] else None,
                })

        # Aggregate statistics
        total_snapshots = len(snapshots)
        completed_snapshots = sum(1 for s in snapshots if s['status'] == 'completed')
        failed_snapshots = sum(1 for s in snapshots if s['status'] == 'failed')
        total_balances = sum(s['total_balances'] or 0 for s in snapshots)

        avg_duration = None
        if completed_snapshots > 0:
            durations = [
                s['duration_seconds']
                for s in snapshots
                if s['status'] == 'completed' and s['duration_seconds']
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)

        return {
            'period': {
                'start_date': str(start_date),
                'end_date': str(end_date),
            },
            'summary': {
                'total_snapshots': total_snapshots,
                'completed_snapshots': completed_snapshots,
                'failed_snapshots': failed_snapshots,
                'success_rate': round(
                    (completed_snapshots / total_snapshots * 100) if total_snapshots > 0 else 0,
                    2,
                ),
                'total_balances': total_balances,
                'avg_duration_seconds': round(avg_duration, 2) if avg_duration else None,
            },
            'snapshots': snapshots,
        }