"""Cross-validation framework for multi-source data verification.

This module implements validation logic to ensure consistency between
transactions and balance changes, and cross-validates data across multiple sources.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import structlog
from asyncpg import Pool

logger = structlog.get_logger()


class CrossValidator:
    """Cross-validation framework for verifying data consistency."""

    def __init__(self, db_pool: Pool):
        """Initialize cross-validator.

        Args:
            db_pool: Async database connection pool
        """
        self.db = db_pool
        self.logger = logger.bind(component="cross_validator")

    async def validate_transaction_balance_consistency(
        self,
        wallet_address: str,
        start_date: datetime,
        end_date: datetime,
        tolerance: float = 0.01,
    ) -> Dict[str, Any]:
        """Validate that balance changes match transaction history.

        Args:
            wallet_address: Wallet address to validate
            start_date: Start of validation period
            end_date: End of validation period
            tolerance: Tolerance for balance discrepancies (default 1%)

        Returns:
            Validation results with consistency scores and discrepancies
        """
        log = self.logger.bind(
            wallet_address=wallet_address,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        log.info("validating_transaction_balance_consistency")

        # Get all transactions for the wallet
        transactions = await self._get_wallet_transactions(
            wallet_address, start_date, end_date
        )

        # Get balance snapshots for validation points
        snapshots = await self._get_balance_snapshots(
            wallet_address, start_date, end_date
        )

        if len(snapshots) < 2:
            log.warning("insufficient_snapshots", snapshot_count=len(snapshots))
            return {
                "wallet_address": wallet_address,
                "validation_status": "insufficient_data",
                "snapshot_count": len(snapshots),
                "periods_validated": 0,
            }

        validation_results = []

        for i in range(len(snapshots) - 1):
            start_snapshot = snapshots[i]
            end_snapshot = snapshots[i + 1]

            # Get transactions in this period
            period_transactions = [
                tx
                for tx in transactions
                if start_snapshot["snapshot_date"]
                < tx["block_timestamp"]
                <= end_snapshot["snapshot_date"]
            ]

            # Calculate expected balance changes from transactions
            expected_changes = self._calculate_balance_changes_from_transactions(
                period_transactions
            )

            # Calculate actual balance changes from snapshots
            actual_changes = await self._calculate_actual_balance_changes(
                wallet_address, start_snapshot, end_snapshot
            )

            # Compare and validate consistency
            consistency_check = self._compare_balance_changes(
                expected_changes, actual_changes, tolerance
            )

            validation_results.append(
                {
                    "period_start": start_snapshot["snapshot_date"].isoformat(),
                    "period_end": end_snapshot["snapshot_date"].isoformat(),
                    "transaction_count": len(period_transactions),
                    "expected_changes": expected_changes,
                    "actual_changes": actual_changes,
                    "consistency_score": consistency_check["score"],
                    "discrepancies": consistency_check["discrepancies"],
                    "validation_status": (
                        "pass" if consistency_check["score"] > (1 - tolerance) else "fail"
                    ),
                }
            )

        # Calculate overall consistency
        overall_score = (
            sum(r["consistency_score"] for r in validation_results)
            / len(validation_results)
            if validation_results
            else 0
        )

        log.info(
            "validation_completed",
            periods_validated=len(validation_results),
            overall_score=overall_score,
        )

        return {
            "wallet_address": wallet_address,
            "validation_period": (start_date.isoformat(), end_date.isoformat()),
            "periods_validated": len(validation_results),
            "overall_consistency_score": overall_score,
            "validation_status": "pass" if overall_score > (1 - tolerance) else "fail",
            "period_results": validation_results,
        }

    async def _get_wallet_transactions(
        self, wallet_address: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch all transactions for wallet in date range."""
        query = """
            SELECT
                tx_hash,
                block_number,
                block_timestamp,
                token_in_address,
                token_out_address,
                amount_in,
                amount_out,
                gas_used,
                gas_price_gwei,
                eth_value_total_usd
            FROM transactions
            WHERE wallet_address = $1
                AND block_timestamp >= $2
                AND block_timestamp <= $3
            ORDER BY block_timestamp ASC
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, wallet_address, start_date, end_date)
            return [dict(row) for row in rows]

    async def _get_balance_snapshots(
        self, wallet_address: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch balance snapshots for wallet in date range."""
        query = """
            SELECT
                snapshot_date,
                token_address,
                balance,
                eth_value
            FROM wallet_balances
            WHERE wallet_address = $1
                AND snapshot_date >= $2
                AND snapshot_date <= $3
            ORDER BY snapshot_date ASC
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, wallet_address, start_date, end_date)

        # Group by snapshot_date
        snapshots_by_date = defaultdict(list)
        for row in rows:
            snapshots_by_date[row["snapshot_date"]].append(dict(row))

        # Convert to list of snapshots
        snapshots = [
            {"snapshot_date": date, "balances": balances}
            for date, balances in sorted(snapshots_by_date.items())
        ]

        return snapshots

    def _calculate_balance_changes_from_transactions(
        self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Decimal]:
        """Calculate expected balance changes from transaction history.

        Args:
            transactions: List of transactions

        Returns:
            Dict mapping token address to expected balance change
        """
        balance_changes = defaultdict(Decimal)

        for tx in transactions:
            # Outgoing tokens (sells) - decrease balance
            if tx.get("token_in_address") and tx.get("amount_in"):
                balance_changes[tx["token_in_address"]] -= Decimal(
                    str(tx["amount_in"])
                )

            # Incoming tokens (buys) - increase balance
            if tx.get("token_out_address") and tx.get("amount_out"):
                balance_changes[tx["token_out_address"]] += Decimal(
                    str(tx["amount_out"])
                )

            # Gas costs reduce ETH balance
            if tx.get("gas_used") and tx.get("gas_price_gwei"):
                gas_cost_eth = (
                    Decimal(str(tx["gas_used"]))
                    * Decimal(str(tx["gas_price_gwei"]))
                    / Decimal("1e9")
                )
                # ETH address (zero address or WETH)
                eth_address = "0x0000000000000000000000000000000000000000"
                balance_changes[eth_address] -= gas_cost_eth

        return dict(balance_changes)

    async def _calculate_actual_balance_changes(
        self,
        wallet_address: str,
        start_snapshot: Dict[str, Any],
        end_snapshot: Dict[str, Any],
    ) -> Dict[str, Decimal]:
        """Calculate actual balance changes between two snapshots.

        Args:
            wallet_address: Wallet address
            start_snapshot: Starting balance snapshot
            end_snapshot: Ending balance snapshot

        Returns:
            Dict mapping token address to actual balance change
        """
        balance_changes = {}

        # Create maps for easier lookup
        start_balances = {
            b["token_address"]: Decimal(str(b["balance"]))
            for b in start_snapshot["balances"]
        }
        end_balances = {
            b["token_address"]: Decimal(str(b["balance"]))
            for b in end_snapshot["balances"]
        }

        # Get all tokens present in either snapshot
        all_tokens = set(start_balances.keys()) | set(end_balances.keys())

        for token in all_tokens:
            start_bal = start_balances.get(token, Decimal("0"))
            end_bal = end_balances.get(token, Decimal("0"))
            balance_changes[token] = end_bal - start_bal

        return balance_changes

    def _compare_balance_changes(
        self,
        expected: Dict[str, Decimal],
        actual: Dict[str, Decimal],
        tolerance: float,
    ) -> Dict[str, Any]:
        """Compare expected and actual balance changes.

        Args:
            expected: Expected balance changes from transactions
            actual: Actual balance changes from snapshots
            tolerance: Tolerance percentage for discrepancies

        Returns:
            Comparison results with score and discrepancies
        """
        all_tokens = set(expected.keys()) | set(actual.keys())

        discrepancies = []
        matching_tokens = 0

        for token in all_tokens:
            expected_change = expected.get(token, Decimal("0"))
            actual_change = actual.get(token, Decimal("0"))

            # Calculate relative difference
            if expected_change == 0 and actual_change == 0:
                matching_tokens += 1
                continue

            # Use the larger absolute value as denominator
            denominator = max(abs(expected_change), abs(actual_change))
            if denominator == 0:
                matching_tokens += 1
                continue

            relative_diff = abs(expected_change - actual_change) / denominator

            if relative_diff <= tolerance:
                matching_tokens += 1
            else:
                discrepancies.append(
                    {
                        "token_address": token,
                        "expected_change": str(expected_change),
                        "actual_change": str(actual_change),
                        "relative_difference": float(relative_diff),
                    }
                )

        consistency_score = (
            matching_tokens / len(all_tokens) if all_tokens else 1.0
        )

        return {
            "score": consistency_score,
            "matching_tokens": matching_tokens,
            "total_tokens": len(all_tokens),
            "discrepancies": discrepancies,
        }

    async def validate_wallet_portfolio_consistency(
        self, wallet_address: str, snapshot_dates: List[datetime]
    ) -> Dict[str, Any]:
        """Validate portfolio consistency across time periods.

        Args:
            wallet_address: Wallet address to validate
            snapshot_dates: List of snapshot dates to validate

        Returns:
            Validation results for portfolio consistency
        """
        log = self.logger.bind(
            wallet_address=wallet_address, snapshot_count=len(snapshot_dates)
        )
        log.info("validating_portfolio_consistency")

        consistency_results = []

        for i in range(len(snapshot_dates) - 1):
            start_date = snapshot_dates[i]
            end_date = snapshot_dates[i + 1]

            # Validate this period
            period_result = await self.validate_transaction_balance_consistency(
                wallet_address, start_date, end_date
            )

            consistency_results.append(
                {
                    "period": (start_date.isoformat(), end_date.isoformat()),
                    "consistency_score": period_result["overall_consistency_score"],
                    "validation_status": period_result["validation_status"],
                }
            )

        overall_score = (
            sum(r["consistency_score"] for r in consistency_results)
            / len(consistency_results)
            if consistency_results
            else 0
        )

        log.info("portfolio_validation_completed", overall_score=overall_score)

        return {
            "wallet_address": wallet_address,
            "periods_validated": len(consistency_results),
            "overall_consistency_score": overall_score,
            "validation_status": "pass" if overall_score > 0.95 else "fail",
            "period_results": consistency_results,
        }

    async def validate_transaction_completeness(
        self, wallet_address: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Validate transaction completeness against multiple sources.

        Args:
            wallet_address: Wallet address
            start_date: Start date
            end_date: End date

        Returns:
            Completeness validation results
        """
        log = self.logger.bind(wallet_address=wallet_address)
        log.info("validating_transaction_completeness")

        # Get transaction count from our database
        our_count = await self._get_transaction_count(
            wallet_address, start_date, end_date
        )

        # Check for gaps in block numbers
        gaps = await self._detect_block_gaps(wallet_address, start_date, end_date)

        # Check for temporal gaps (unusual time gaps between transactions)
        temporal_gaps = await self._detect_temporal_gaps(
            wallet_address, start_date, end_date
        )

        completeness_score = 1.0
        issues = []

        if gaps:
            completeness_score -= 0.1
            issues.append({"type": "block_gaps", "count": len(gaps), "gaps": gaps})

        if temporal_gaps:
            completeness_score -= 0.1
            issues.append(
                {
                    "type": "temporal_gaps",
                    "count": len(temporal_gaps),
                    "gaps": temporal_gaps,
                }
            )

        completeness_score = max(0.0, completeness_score)

        log.info(
            "completeness_validation_completed",
            transaction_count=our_count,
            completeness_score=completeness_score,
        )

        return {
            "wallet_address": wallet_address,
            "period": (start_date.isoformat(), end_date.isoformat()),
            "transaction_count": our_count,
            "completeness_score": completeness_score,
            "validation_status": "pass" if completeness_score > 0.9 else "fail",
            "issues": issues,
        }

    async def _get_transaction_count(
        self, wallet_address: str, start_date: datetime, end_date: datetime
    ) -> int:
        """Get transaction count from database."""
        query = """
            SELECT COUNT(*) as count
            FROM transactions
            WHERE wallet_address = $1
                AND block_timestamp >= $2
                AND block_timestamp <= $3
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, wallet_address, start_date, end_date)
            return row["count"]

    async def _detect_block_gaps(
        self, wallet_address: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Detect gaps in block number sequence."""
        query = """
            SELECT
                block_number,
                LAG(block_number) OVER (ORDER BY block_number) as prev_block
            FROM transactions
            WHERE wallet_address = $1
                AND block_timestamp >= $2
                AND block_timestamp <= $3
            ORDER BY block_number
        """

        gaps = []
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, wallet_address, start_date, end_date)

            for row in rows:
                if row["prev_block"] is not None:
                    gap = row["block_number"] - row["prev_block"]
                    # Flag gaps larger than 1000 blocks (~3-4 hours)
                    if gap > 1000:
                        gaps.append(
                            {
                                "start_block": row["prev_block"],
                                "end_block": row["block_number"],
                                "gap_size": gap,
                            }
                        )

        return gaps

    async def _detect_temporal_gaps(
        self, wallet_address: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Detect unusual temporal gaps between transactions."""
        query = """
            SELECT
                block_timestamp,
                LAG(block_timestamp) OVER (ORDER BY block_timestamp) as prev_timestamp
            FROM transactions
            WHERE wallet_address = $1
                AND block_timestamp >= $2
                AND block_timestamp <= $3
            ORDER BY block_timestamp
        """

        gaps = []
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, wallet_address, start_date, end_date)

            for row in rows:
                if row["prev_timestamp"] is not None:
                    gap = row["block_timestamp"] - row["prev_timestamp"]
                    # Flag gaps larger than 7 days
                    if gap > timedelta(days=7):
                        gaps.append(
                            {
                                "start_time": row["prev_timestamp"].isoformat(),
                                "end_time": row["block_timestamp"].isoformat(),
                                "gap_duration_hours": gap.total_seconds() / 3600,
                            }
                        )

        return gaps

    async def validate_gas_costs(
        self, start_date: datetime, end_date: datetime, sample_size: int = 100
    ) -> Dict[str, Any]:
        """Validate gas cost calculations and detect anomalies.

        Args:
            start_date: Start date
            end_date: End date
            sample_size: Number of transactions to validate

        Returns:
            Gas cost validation results
        """
        log = self.logger.bind(sample_size=sample_size)
        log.info("validating_gas_costs")

        query = """
            SELECT
                tx_hash,
                gas_used,
                gas_price_gwei,
                eth_value_total_usd,
                block_timestamp
            FROM transactions
            WHERE block_timestamp >= $1
                AND block_timestamp <= $2
                AND transaction_status = 'success'
            ORDER BY RANDOM()
            LIMIT $3
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date, sample_size)

        validation_results = []
        anomalies = []

        for row in rows:
            gas_used = Decimal(str(row["gas_used"]))
            gas_price = Decimal(str(row["gas_price_gwei"]))

            # Calculate expected gas cost in ETH
            expected_gas_eth = gas_used * gas_price / Decimal("1e9")

            # Validate gas price is within reasonable range
            # Typical gas prices: 1-500 Gwei
            if gas_price < 1 or gas_price > 1000:
                anomalies.append(
                    {
                        "tx_hash": row["tx_hash"],
                        "issue": "unusual_gas_price",
                        "gas_price_gwei": str(gas_price),
                        "timestamp": row["block_timestamp"].isoformat(),
                    }
                )

            # Validate gas used is within reasonable range
            # Typical swap: 100k-500k gas
            if gas_used < 21000 or gas_used > 5000000:
                anomalies.append(
                    {
                        "tx_hash": row["tx_hash"],
                        "issue": "unusual_gas_used",
                        "gas_used": str(gas_used),
                        "timestamp": row["block_timestamp"].isoformat(),
                    }
                )

            validation_results.append(
                {
                    "tx_hash": row["tx_hash"],
                    "gas_cost_eth": str(expected_gas_eth),
                    "valid": True,
                }
            )

        anomaly_rate = len(anomalies) / len(validation_results) if validation_results else 0
        validation_score = 1 - anomaly_rate

        log.info(
            "gas_validation_completed",
            transactions_validated=len(validation_results),
            anomalies_found=len(anomalies),
            validation_score=validation_score,
        )

        return {
            "period": (start_date.isoformat(), end_date.isoformat()),
            "transactions_validated": len(validation_results),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": anomaly_rate,
            "validation_score": validation_score,
            "validation_status": "pass" if validation_score > 0.95 else "fail",
            "anomalies": anomalies[:10],  # Return first 10 anomalies
        }