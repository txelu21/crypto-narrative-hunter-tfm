"""
Data validation and quality assurance for transaction extraction.

Provides:
- Transaction completeness validation
- Cross-source verification
- Data quality metrics
- Anomaly detection
- Statistical validation
"""

import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TransactionValidator:
    """Validates transaction extraction data quality."""

    def __init__(
        self,
        alchemy_client: Any = None,
        etherscan_client: Any = None,
        subgraph_client: Any = None
    ):
        """
        Initialize validator with API clients.

        Args:
            alchemy_client: Alchemy client for verification
            etherscan_client: Etherscan client for cross-validation
            subgraph_client: Subgraph client for comparison
        """
        self.alchemy_client = alchemy_client
        self.etherscan_client = etherscan_client
        self.subgraph_client = subgraph_client

    async def validate_transaction_completeness(
        self,
        wallet_address: str,
        extracted_txs: List[Dict[str, Any]],
        time_period: tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Validate transaction extraction completeness across sources.

        Args:
            wallet_address: Wallet address
            extracted_txs: List of extracted transactions
            time_period: (start_date, end_date) tuple

        Returns:
            Validation results dictionary
        """
        extracted_count = len(extracted_txs)

        # Get transaction counts from different sources
        etherscan_count = await self._count_etherscan_transactions(
            wallet_address, time_period
        ) if self.etherscan_client else None

        subgraph_count = await self._count_subgraph_swaps(
            wallet_address, time_period
        ) if self.subgraph_client else None

        # Calculate validation metrics
        validation_results = {
            "wallet_address": wallet_address,
            "extracted_count": extracted_count,
            "etherscan_count": etherscan_count,
            "subgraph_count": subgraph_count,
            "completeness_score": 1.0,
            "variance_percentage": 0.0,
            "is_complete": True
        }

        # Cross-validate counts
        if etherscan_count is not None:
            variance = abs(extracted_count - etherscan_count) / max(etherscan_count, 1)
            validation_results["variance_percentage"] = variance * 100

            # Allow 10% variance
            validation_results["is_complete"] = variance < 0.10

            if etherscan_count > 0:
                validation_results["completeness_score"] = min(
                    extracted_count / etherscan_count, 1.0
                )

        # Spot check individual transactions
        if extracted_txs:
            spot_check_results = await self._spot_check_transactions(
                extracted_txs[:min(10, len(extracted_txs))]
            )
            validation_results["spot_check_accuracy"] = spot_check_results

        return validation_results

    async def _count_etherscan_transactions(
        self,
        wallet_address: str,
        time_period: tuple[datetime, datetime]
    ) -> int:
        """Count transactions via Etherscan."""
        # TODO: Implement Etherscan API call
        return 0

    async def _count_subgraph_swaps(
        self,
        wallet_address: str,
        time_period: tuple[datetime, datetime]
    ) -> int:
        """Count swaps via Uniswap subgraph."""
        # TODO: Implement subgraph query
        return 0

    async def _spot_check_transactions(
        self,
        transactions: List[Dict[str, Any]]
    ) -> float:
        """
        Spot check random transactions for accuracy.

        Args:
            transactions: List of transactions to check

        Returns:
            Accuracy percentage (0-1)
        """
        if not transactions or not self.etherscan_client:
            return 1.0

        correct_count = 0

        for tx in transactions:
            is_correct = await self._verify_transaction_on_etherscan(tx["tx_hash"])
            if is_correct:
                correct_count += 1

        return correct_count / len(transactions)

    async def _verify_transaction_on_etherscan(self, tx_hash: str) -> bool:
        """Verify transaction exists on Etherscan."""
        # TODO: Implement Etherscan verification
        return True

    def detect_anomalies(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Detect anomalies in transaction data.

        Args:
            transactions: List of transactions

        Returns:
            Dictionary of anomaly types and affected tx hashes
        """
        anomalies: Dict[str, List[str]] = {
            "extreme_gas": [],
            "unusual_amounts": [],
            "timing_outliers": [],
            "duplicate_hashes": []
        }

        if not transactions:
            return anomalies

        # Extract metrics
        gas_prices = [
            Decimal(tx.get("gas_price_gwei", 0))
            for tx in transactions
            if tx.get("gas_price_gwei")
        ]

        amounts = [
            Decimal(tx.get("eth_value_in", 0))
            for tx in transactions
            if tx.get("eth_value_in")
        ]

        # Detect extreme gas prices (>3 std dev from mean)
        if gas_prices:
            mean_gas = sum(gas_prices) / len(gas_prices)
            std_gas = self._calculate_std_dev(gas_prices, mean_gas)

            for tx in transactions:
                gas_price = Decimal(tx.get("gas_price_gwei", 0))
                if abs(gas_price - mean_gas) > 3 * std_gas:
                    anomalies["extreme_gas"].append(tx["tx_hash"])

        # Detect unusual amounts (>3 std dev from mean)
        if amounts:
            mean_amount = sum(amounts) / len(amounts)
            std_amount = self._calculate_std_dev(amounts, mean_amount)

            for tx in transactions:
                amount = Decimal(tx.get("eth_value_in", 0))
                if abs(amount - mean_amount) > 3 * std_amount:
                    anomalies["unusual_amounts"].append(tx["tx_hash"])

        # Detect duplicate hashes
        seen_hashes = set()
        for tx in transactions:
            tx_hash = tx["tx_hash"]
            if tx_hash in seen_hashes:
                anomalies["duplicate_hashes"].append(tx_hash)
            seen_hashes.add(tx_hash)

        return anomalies

    def _calculate_std_dev(
        self,
        values: List[Decimal],
        mean: Decimal
    ) -> Decimal:
        """Calculate standard deviation."""
        if not values:
            return Decimal(0)

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance.sqrt()

    def calculate_quality_metrics(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate data quality metrics.

        Args:
            transactions: List of transactions

        Returns:
            Dictionary of quality metrics
        """
        total_count = len(transactions)

        if total_count == 0:
            return {
                "total_transactions": 0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0
            }

        # Check field completeness
        required_fields = [
            "tx_hash", "block_number", "wallet_address",
            "dex_name", "token_in", "token_out"
        ]

        complete_count = 0
        for tx in transactions:
            if all(tx.get(field) for field in required_fields):
                complete_count += 1

        completeness = complete_count / total_count

        # Check for failed transactions
        failed_count = sum(
            1 for tx in transactions
            if tx.get("transaction_status") == "failed"
        )

        # Check timestamp consistency
        timestamps = [
            tx.get("timestamp")
            for tx in transactions
            if tx.get("timestamp")
        ]

        consistency = 1.0
        if timestamps:
            # Check if timestamps are in order
            is_ordered = all(
                timestamps[i] <= timestamps[i + 1]
                for i in range(len(timestamps) - 1)
            )
            consistency = 1.0 if is_ordered else 0.8

        return {
            "total_transactions": total_count,
            "complete_transactions": complete_count,
            "completeness": completeness,
            "failed_transaction_rate": failed_count / total_count if total_count > 0 else 0,
            "consistency": consistency,
            "quality_score": (completeness + consistency) / 2
        }