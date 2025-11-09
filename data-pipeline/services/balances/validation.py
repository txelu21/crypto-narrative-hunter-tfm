"""Balance validation and cross-checking.

This module provides validation mechanisms to ensure balance snapshot accuracy
through transaction reconciliation and statistical analysis.
"""

from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
from decimal import Decimal
import psycopg
from psycopg.rows import dict_row

from data_collection.common.logging_setup import get_logger

logger = get_logger(__name__)


class BalanceValidationService:
    """Service for validating balance snapshot accuracy."""

    def __init__(self, conn: psycopg.Connection):
        """Initialize validation service.

        Args:
            conn: Database connection
        """
        self.conn = conn

    def reconstruct_balance_from_transactions(
        self,
        wallet_address: str,
        token_address: str,
        up_to_date: date,
        initial_balance: int = 0,
    ) -> int:
        """Reconstruct token balance from transaction history.

        Args:
            wallet_address: Wallet address
            token_address: Token address
            up_to_date: Reconstruct balance up to this date
            initial_balance: Starting balance (default: 0)

        Returns:
            Reconstructed balance
        """
        # Query all transactions affecting this token
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
        ORDER BY block_timestamp ASC
        """

        balance = initial_balance

        try:
            with self.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    query,
                    (wallet_address, token_address, token_address, up_to_date),
                )
                transactions = cur.fetchall()

                for tx in transactions:
                    # Token coming in
                    if tx['token_in_address'] == token_address and tx['amount_in']:
                        balance += int(tx['amount_in'])

                    # Token going out
                    if tx['token_out_address'] == token_address and tx['amount_out']:
                        balance -= int(tx['amount_out'])

        except Exception as e:
            logger.error(
                f"Transaction reconstruction failed for {wallet_address}/{token_address}: {e}"
            )
            raise

        return balance

    def validate_balance_snapshot(
        self,
        wallet_address: str,
        snapshot_date: date,
        tolerance: Decimal = Decimal('0.01'),
    ) -> Dict[str, Dict]:
        """Validate balance snapshot against transaction history.

        Args:
            wallet_address: Wallet address
            snapshot_date: Snapshot date to validate
            tolerance: Acceptable relative difference (default: 1%)

        Returns:
            Validation results by token
        """
        # Get snapshot balances
        snapshot_query = """
        SELECT token_address, balance
        FROM wallet_balances
        WHERE wallet_address = %s
        AND snapshot_date = %s
        """

        validation_results = {}

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(snapshot_query, (wallet_address, snapshot_date))
            snapshot_balances = cur.fetchall()

        for row in snapshot_balances:
            token_address = row['token_address']
            snapshot_balance = int(row['balance'])

            try:
                # Reconstruct balance from transactions
                reconstructed_balance = self.reconstruct_balance_from_transactions(
                    wallet_address,
                    token_address,
                    snapshot_date,
                )

                # Calculate difference
                max_balance = max(snapshot_balance, reconstructed_balance)
                if max_balance > 0:
                    relative_diff = Decimal(abs(snapshot_balance - reconstructed_balance)) / Decimal(max_balance)
                else:
                    relative_diff = Decimal('0')

                is_valid = relative_diff <= tolerance

                validation_results[token_address] = {
                    'snapshot_balance': snapshot_balance,
                    'reconstructed_balance': reconstructed_balance,
                    'difference': snapshot_balance - reconstructed_balance,
                    'relative_difference': float(relative_diff),
                    'valid': is_valid,
                }

            except Exception as e:
                logger.error(
                    f"Validation failed for {wallet_address}/{token_address}: {e}"
                )
                validation_results[token_address] = {
                    'snapshot_balance': snapshot_balance,
                    'reconstructed_balance': None,
                    'difference': None,
                    'relative_difference': None,
                    'valid': False,
                    'error': str(e),
                }

        # Overall validation status
        all_valid = all(
            result.get('valid', False) for result in validation_results.values()
        )
        tokens_validated = len(validation_results)
        tokens_passed = sum(1 for r in validation_results.values() if r.get('valid', False))

        logger.info(
            f"Validation for {wallet_address} on {snapshot_date}: "
            f"{tokens_passed}/{tokens_validated} tokens passed"
        )

        return {
            'wallet_address': wallet_address,
            'snapshot_date': str(snapshot_date),
            'overall_valid': all_valid,
            'token_results': validation_results,
            'tokens_validated': tokens_validated,
            'tokens_passed': tokens_passed,
        }

    def validate_balance_reasonableness(
        self,
        wallet_address: str,
        token_address: str,
        balance: int,
        snapshot_date: date,
        max_change_factor: Decimal = Decimal('100'),
    ) -> Tuple[bool, Optional[str]]:
        """Validate that balance change is reasonable compared to history.

        Args:
            wallet_address: Wallet address
            token_address: Token address
            balance: Balance to validate
            snapshot_date: Snapshot date
            max_change_factor: Maximum acceptable change factor

        Returns:
            Tuple of (is_valid, reason)
        """
        # Get recent historical balance
        query = """
        SELECT balance
        FROM wallet_balances
        WHERE wallet_address = %s
        AND token_address = %s
        AND snapshot_date < %s
        ORDER BY snapshot_date DESC
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (wallet_address, token_address, snapshot_date))
                result = cur.fetchone()

                if result and result[0] is not None:
                    historical_balance = int(result[0])

                    # Check for unreasonable changes
                    if historical_balance > 0 and balance > 0:
                        change_ratio = Decimal(balance) / Decimal(historical_balance)

                        if change_ratio > max_change_factor or change_ratio < (1 / max_change_factor):
                            return (
                                False,
                                f"Balance changed by {float(change_ratio):.2f}x, exceeds threshold {float(max_change_factor)}x",
                            )

        except Exception as e:
            logger.warning(
                f"Reasonableness check failed for {wallet_address}/{token_address}: {e}"
            )
            # Don't block on validation errors
            return True, None

        return True, None

    def detect_balance_outliers(
        self,
        snapshot_date: date,
        z_score_threshold: float = 3.0,
    ) -> List[Dict]:
        """Detect statistical outliers in balance changes.

        Args:
            snapshot_date: Snapshot date to analyze
            z_score_threshold: Z-score threshold for outlier detection

        Returns:
            List of outlier records
        """
        # This is a simplified implementation
        # In production, you'd calculate z-scores for balance changes
        query = """
        WITH balance_changes AS (
            SELECT
                curr.wallet_address,
                curr.token_address,
                curr.balance AS current_balance,
                prev.balance AS previous_balance,
                curr.balance - prev.balance AS balance_change,
                curr.eth_value AS current_eth_value,
                prev.eth_value AS previous_eth_value
            FROM wallet_balances curr
            LEFT JOIN wallet_balances prev
                ON curr.wallet_address = prev.wallet_address
                AND curr.token_address = prev.token_address
                AND prev.snapshot_date = (
                    SELECT MAX(snapshot_date)
                    FROM wallet_balances
                    WHERE wallet_address = curr.wallet_address
                    AND token_address = curr.token_address
                    AND snapshot_date < curr.snapshot_date
                )
            WHERE curr.snapshot_date = %s
        )
        SELECT
            wallet_address,
            token_address,
            current_balance,
            previous_balance,
            balance_change,
            current_eth_value,
            previous_eth_value
        FROM balance_changes
        WHERE previous_balance > 0
        AND ABS(balance_change::NUMERIC / previous_balance::NUMERIC) > 10
        ORDER BY ABS(balance_change::NUMERIC / previous_balance::NUMERIC) DESC
        LIMIT 100
        """

        outliers = []

        try:
            with self.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (snapshot_date,))
                results = cur.fetchall()

                for row in results:
                    outliers.append({
                        'wallet_address': row['wallet_address'],
                        'token_address': row['token_address'],
                        'current_balance': int(row['current_balance']) if row['current_balance'] else None,
                        'previous_balance': int(row['previous_balance']) if row['previous_balance'] else None,
                        'balance_change': int(row['balance_change']) if row['balance_change'] else None,
                        'current_eth_value': row['current_eth_value'],
                        'previous_eth_value': row['previous_eth_value'],
                    })

        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")

        logger.info(f"Detected {len(outliers)} balance outliers for {snapshot_date}")
        return outliers

    def validate_snapshot_completeness(
        self,
        snapshot_date: date,
        expected_wallets: Optional[List[str]] = None,
    ) -> Dict:
        """Validate that snapshot is complete for all expected wallets.

        Args:
            snapshot_date: Snapshot date to validate
            expected_wallets: Optional list of expected wallet addresses

        Returns:
            Completeness validation results
        """
        # Count wallets in snapshot
        count_query = """
        SELECT COUNT(DISTINCT wallet_address) AS wallet_count
        FROM wallet_balances
        WHERE snapshot_date = %s
        """

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(count_query, (snapshot_date,))
            result = cur.fetchone()
            actual_wallet_count = result['wallet_count'] if result else 0

        if expected_wallets:
            expected_count = len(expected_wallets)

            # Check which wallets are missing
            missing_query = """
            SELECT %s AS expected_wallet
            WHERE NOT EXISTS (
                SELECT 1 FROM wallet_balances
                WHERE wallet_address = %s
                AND snapshot_date = %s
            )
            """

            missing_wallets = []
            for wallet in expected_wallets:
                with self.conn.cursor() as cur:
                    cur.execute(missing_query, (wallet, wallet, snapshot_date))
                    if cur.fetchone():
                        missing_wallets.append(wallet)

            is_complete = len(missing_wallets) == 0

            return {
                'snapshot_date': str(snapshot_date),
                'expected_wallets': expected_count,
                'actual_wallets': actual_wallet_count,
                'missing_wallets': missing_wallets,
                'is_complete': is_complete,
            }
        else:
            return {
                'snapshot_date': str(snapshot_date),
                'actual_wallets': actual_wallet_count,
                'is_complete': actual_wallet_count > 0,
            }

    def generate_validation_report(
        self,
        snapshot_date: date,
        sample_size: int = 10,
    ) -> Dict:
        """Generate comprehensive validation report for a snapshot.

        Args:
            snapshot_date: Snapshot date to validate
            sample_size: Number of wallets to sample for detailed validation

        Returns:
            Comprehensive validation report
        """
        logger.info(f"Generating validation report for {snapshot_date}")

        # Get sample of wallets
        sample_query = """
        SELECT DISTINCT wallet_address
        FROM wallet_balances
        WHERE snapshot_date = %s
        ORDER BY RANDOM()
        LIMIT %s
        """

        with self.conn.cursor() as cur:
            cur.execute(sample_query, (snapshot_date, sample_size))
            sample_wallets = [row[0] for row in cur.fetchall()]

        # Validate sample
        sample_validations = []
        for wallet in sample_wallets:
            validation = self.validate_balance_snapshot(wallet, snapshot_date)
            sample_validations.append(validation)

        # Detect outliers
        outliers = self.detect_balance_outliers(snapshot_date)

        # Check completeness
        completeness = self.validate_snapshot_completeness(snapshot_date)

        # Aggregate results
        total_validated = len(sample_validations)
        total_passed = sum(1 for v in sample_validations if v['overall_valid'])
        validation_rate = (total_passed / total_validated * 100) if total_validated > 0 else 0

        report = {
            'snapshot_date': str(snapshot_date),
            'validation_summary': {
                'sample_size': total_validated,
                'passed': total_passed,
                'failed': total_validated - total_passed,
                'validation_rate': round(validation_rate, 2),
            },
            'outliers_detected': len(outliers),
            'outliers': outliers[:10],  # Top 10 outliers
            'completeness': completeness,
            'sample_validations': sample_validations,
        }

        logger.info(
            f"Validation report complete: {total_passed}/{total_validated} passed "
            f"({validation_rate:.1f}%), {len(outliers)} outliers"
        )

        return report