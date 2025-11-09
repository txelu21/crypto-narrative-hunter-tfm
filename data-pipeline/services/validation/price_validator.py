"""Comprehensive price and value validation system.

This module wraps the existing price validator and adds comprehensive
validation for portfolio values, price reasonableness, and value calculations.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog
from asyncpg import Pool

logger = structlog.get_logger()


class PriceValidationFramework:
    """Comprehensive price and value validation framework."""

    def __init__(self, db_pool: Pool):
        """Initialize price validation framework.

        Args:
            db_pool: Async database connection pool
        """
        self.db = db_pool
        self.logger = logger.bind(component="price_validation_framework")

    async def validate_price_data_quality(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Validate overall price data quality.

        Args:
            start_date: Start of validation period
            end_date: End of validation period

        Returns:
            Price data quality validation results
        """
        log = self.logger.bind(
            start_date=start_date.isoformat(), end_date=end_date.isoformat()
        )
        log.info("validating_price_data_quality")

        # Check price data completeness
        completeness = await self._check_price_completeness(start_date, end_date)

        # Check price reasonableness
        reasonableness = await self._check_price_reasonableness(
            start_date, end_date
        )

        # Check price volatility
        volatility = await self._check_price_volatility(start_date, end_date)

        # Check for price gaps
        gaps = await self._detect_price_gaps(start_date, end_date)

        # Overall quality score
        quality_metrics = {
            "completeness_score": completeness["completeness_score"],
            "reasonableness_score": reasonableness["reasonableness_score"],
            "volatility_score": volatility["volatility_score"],
            "gap_penalty": min(0.2, len(gaps) * 0.05),
        }

        overall_quality = (
            quality_metrics["completeness_score"] * 0.3
            + quality_metrics["reasonableness_score"] * 0.3
            + quality_metrics["volatility_score"] * 0.2
            - quality_metrics["gap_penalty"]
        )
        overall_quality = max(0, min(1, overall_quality))

        log.info("price_quality_validation_completed", overall_quality=overall_quality)

        return {
            "period": (start_date.isoformat(), end_date.isoformat()),
            "completeness": completeness,
            "reasonableness": reasonableness,
            "volatility": volatility,
            "gaps_detected": len(gaps),
            "gaps": gaps[:10],
            "quality_metrics": quality_metrics,
            "overall_quality_score": overall_quality,
            "validation_status": "pass" if overall_quality > 0.8 else "fail",
        }

    async def _check_price_completeness(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check completeness of price data."""
        # Calculate expected hourly data points
        expected_hours = int((end_date - start_date).total_seconds() / 3600)

        query = """
            SELECT COUNT(*) as actual_count
            FROM eth_prices
            WHERE timestamp >= $1
                AND timestamp <= $2
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, start_date, end_date)
            actual_count = row["actual_count"]

        completeness_score = (
            actual_count / expected_hours if expected_hours > 0 else 0
        )

        return {
            "expected_data_points": expected_hours,
            "actual_data_points": actual_count,
            "completeness_score": min(1.0, completeness_score),
            "meets_threshold": completeness_score >= 0.95,
        }

    async def _check_price_reasonableness(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check if prices are within reasonable ranges."""
        query = """
            SELECT
                MIN(price_usd) as min_price,
                MAX(price_usd) as max_price,
                AVG(price_usd) as avg_price,
                STDDEV(price_usd) as std_price
            FROM eth_prices
            WHERE timestamp >= $1
                AND timestamp <= $2
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, start_date, end_date)

        # Define reasonable price ranges for ETH (adjust as needed)
        reasonable_min = 500  # $500
        reasonable_max = 10000  # $10,000

        min_price = float(row["min_price"]) if row["min_price"] else 0
        max_price = float(row["max_price"]) if row["max_price"] else 0
        avg_price = float(row["avg_price"]) if row["avg_price"] else 0

        is_reasonable = reasonable_min <= min_price and max_price <= reasonable_max

        # Calculate reasonableness score
        if is_reasonable:
            reasonableness_score = 1.0
        else:
            # Penalize based on how far outside reasonable range
            lower_violation = max(0, reasonable_min - min_price) / reasonable_min
            upper_violation = max(0, max_price - reasonable_max) / reasonable_max
            max_violation = max(lower_violation, upper_violation)
            reasonableness_score = max(0, 1 - max_violation)

        return {
            "min_price": min_price,
            "max_price": max_price,
            "avg_price": avg_price,
            "std_price": float(row["std_price"]) if row["std_price"] else 0,
            "reasonable_range": (reasonable_min, reasonable_max),
            "is_reasonable": is_reasonable,
            "reasonableness_score": reasonableness_score,
        }

    async def _check_price_volatility(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check price volatility for smoothness."""
        query = """
            SELECT
                timestamp,
                price_usd,
                LAG(price_usd) OVER (ORDER BY timestamp) as prev_price
            FROM eth_prices
            WHERE timestamp >= $1
                AND timestamp <= $2
            ORDER BY timestamp
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date)

        if len(rows) < 2:
            return {
                "data_points": len(rows),
                "volatility_score": 1.0,
                "validation_status": "insufficient_data",
            }

        # Calculate hourly price changes
        price_changes = []
        large_spikes = []

        for row in rows:
            if row["prev_price"] is not None:
                prev = float(row["prev_price"])
                curr = float(row["price_usd"])

                if prev > 0:
                    pct_change = abs((curr - prev) / prev)
                    price_changes.append(pct_change)

                    # Flag spikes >10% in one hour
                    if pct_change > 0.10:
                        large_spikes.append(
                            {
                                "timestamp": row["timestamp"].isoformat(),
                                "prev_price": prev,
                                "curr_price": curr,
                                "pct_change": pct_change,
                            }
                        )

        # Calculate volatility metrics
        import numpy as np

        avg_volatility = np.mean(price_changes) if price_changes else 0
        max_volatility = max(price_changes) if price_changes else 0

        # Score based on volatility (lower is better for smoothness)
        # Penalize if average hourly change > 2% or max change > 20%
        volatility_score = 1.0
        if avg_volatility > 0.02:
            volatility_score -= 0.2
        if max_volatility > 0.20:
            volatility_score -= 0.3
        volatility_score = max(0, volatility_score)

        return {
            "data_points": len(rows),
            "avg_hourly_volatility": avg_volatility,
            "max_hourly_volatility": max_volatility,
            "large_spikes_count": len(large_spikes),
            "large_spikes": large_spikes[:5],
            "volatility_score": volatility_score,
            "validation_status": (
                "pass" if len(large_spikes) < len(rows) * 0.05 else "warning"
            ),
        }

    async def _detect_price_gaps(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Detect gaps in price data."""
        query = """
            SELECT
                timestamp,
                LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
            FROM eth_prices
            WHERE timestamp >= $1
                AND timestamp <= $2
            ORDER BY timestamp
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date)

        gaps = []
        for row in rows:
            if row["prev_timestamp"] is not None:
                gap_duration = row["timestamp"] - row["prev_timestamp"]

                # Flag gaps longer than 2 hours
                if gap_duration > timedelta(hours=2):
                    gaps.append(
                        {
                            "start_time": row["prev_timestamp"].isoformat(),
                            "end_time": row["timestamp"].isoformat(),
                            "gap_duration_hours": gap_duration.total_seconds()
                            / 3600,
                        }
                    )

        return gaps

    async def validate_portfolio_values(
        self, wallet_address: str, snapshot_date: datetime
    ) -> Dict[str, Any]:
        """Validate portfolio value calculations for a wallet.

        Args:
            wallet_address: Wallet address
            snapshot_date: Date of the snapshot to validate

        Returns:
            Portfolio value validation results
        """
        log = self.logger.bind(
            wallet_address=wallet_address, snapshot_date=snapshot_date.isoformat()
        )
        log.info("validating_portfolio_values")

        # Get wallet balances for the snapshot
        balances = await self._get_wallet_balances(wallet_address, snapshot_date)

        if len(balances) == 0:
            log.warning("no_balances_found")
            return {
                "wallet_address": wallet_address,
                "snapshot_date": snapshot_date.isoformat(),
                "validation_status": "no_data",
            }

        # Recalculate portfolio value independently
        calculated_value = await self._calculate_portfolio_value(
            balances, snapshot_date
        )

        # Get stored portfolio value from database
        stored_value = await self._get_stored_portfolio_value(
            wallet_address, snapshot_date
        )

        # Compare calculated vs stored
        if stored_value is not None:
            relative_diff = (
                abs(calculated_value - stored_value) / max(stored_value, 0.01)
            )
            is_valid = relative_diff <= 0.05  # 5% tolerance
        else:
            relative_diff = None
            is_valid = False

        log.info(
            "portfolio_validation_completed",
            calculated_value=calculated_value,
            stored_value=stored_value,
            is_valid=is_valid,
        )

        return {
            "wallet_address": wallet_address,
            "snapshot_date": snapshot_date.isoformat(),
            "token_count": len(balances),
            "calculated_value_eth": calculated_value,
            "stored_value_eth": stored_value,
            "relative_difference": relative_diff,
            "is_valid": is_valid,
            "validation_status": "pass" if is_valid else "fail",
        }

    async def _get_wallet_balances(
        self, wallet_address: str, snapshot_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get wallet balances for a snapshot date."""
        query = """
            SELECT
                token_address,
                balance,
                eth_value
            FROM wallet_balances
            WHERE wallet_address = $1
                AND snapshot_date = $2
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, wallet_address, snapshot_date)
            return [dict(row) for row in rows]

    async def _calculate_portfolio_value(
        self, balances: List[Dict[str, Any]], snapshot_date: datetime
    ) -> Decimal:
        """Calculate total portfolio value in ETH."""
        total_value = Decimal("0")

        for balance in balances:
            eth_value = Decimal(str(balance.get("eth_value", 0)))
            total_value += eth_value

        return total_value

    async def _get_stored_portfolio_value(
        self, wallet_address: str, snapshot_date: datetime
    ) -> Optional[Decimal]:
        """Get stored portfolio value from database."""
        query = """
            SELECT SUM(eth_value) as total_value
            FROM wallet_balances
            WHERE wallet_address = $1
                AND snapshot_date = $2
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, wallet_address, snapshot_date)

            if row and row["total_value"] is not None:
                return Decimal(str(row["total_value"]))
            return None

    async def validate_currency_conversions(
        self, start_date: datetime, end_date: datetime, sample_size: int = 100
    ) -> Dict[str, Any]:
        """Validate accuracy of ETH to USD conversions.

        Args:
            start_date: Start date
            end_date: End date
            sample_size: Number of transactions to validate

        Returns:
            Currency conversion validation results
        """
        log = self.logger.bind(sample_size=sample_size)
        log.info("validating_currency_conversions")

        # Sample transactions with USD values
        query = """
            SELECT
                tx_hash,
                block_timestamp,
                eth_value_total_usd,
                gas_used,
                gas_price_gwei
            FROM transactions
            WHERE block_timestamp >= $1
                AND block_timestamp <= $2
                AND eth_value_total_usd IS NOT NULL
            ORDER BY RANDOM()
            LIMIT $3
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date, sample_size)

        if len(rows) == 0:
            log.warning("no_transactions_found")
            return {
                "validation_status": "no_data",
            }

        validation_results = []
        discrepancies = []

        for row in rows:
            # Get ETH price at transaction time
            eth_price = await self._get_eth_price_at_time(row["block_timestamp"])

            if eth_price is None:
                continue

            # Recalculate USD value
            gas_cost_eth = (
                Decimal(str(row["gas_used"]))
                * Decimal(str(row["gas_price_gwei"]))
                / Decimal("1e9")
            )
            calculated_usd = float(gas_cost_eth) * eth_price

            stored_usd = float(row["eth_value_total_usd"])

            # Check for discrepancy
            if stored_usd > 0:
                relative_diff = abs(calculated_usd - stored_usd) / stored_usd

                if relative_diff > 0.05:  # >5% difference
                    discrepancies.append(
                        {
                            "tx_hash": row["tx_hash"],
                            "timestamp": row["block_timestamp"].isoformat(),
                            "calculated_usd": calculated_usd,
                            "stored_usd": stored_usd,
                            "relative_diff": relative_diff,
                        }
                    )

            validation_results.append(
                {
                    "tx_hash": row["tx_hash"],
                    "is_valid": (
                        relative_diff <= 0.05 if stored_usd > 0 else False
                    ),
                }
            )

        accuracy_rate = (
            sum(1 for r in validation_results if r["is_valid"])
            / len(validation_results)
            if validation_results
            else 0
        )

        log.info(
            "conversion_validation_completed",
            transactions_validated=len(validation_results),
            accuracy_rate=accuracy_rate,
        )

        return {
            "period": (start_date.isoformat(), end_date.isoformat()),
            "transactions_validated": len(validation_results),
            "discrepancies_found": len(discrepancies),
            "accuracy_rate": accuracy_rate,
            "discrepancies": discrepancies[:10],
            "validation_status": "pass" if accuracy_rate > 0.95 else "fail",
        }

    async def _get_eth_price_at_time(
        self, timestamp: datetime
    ) -> Optional[float]:
        """Get ETH price at a specific timestamp."""
        query = """
            SELECT price_usd
            FROM eth_prices
            WHERE timestamp <= $1
            ORDER BY timestamp DESC
            LIMIT 1
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query, timestamp)

            if row:
                return float(row["price_usd"])
            return None