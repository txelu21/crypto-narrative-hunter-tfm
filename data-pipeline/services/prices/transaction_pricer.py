"""
Transaction Value Calculation Engine
Converts ETH transaction amounts to USD using historical prices
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Transaction data structure"""
    tx_hash: str
    timestamp: int
    block_number: int
    eth_value_in: float
    eth_value_out: float
    gas_cost_eth: float


@dataclass
class EnrichedTransaction(Transaction):
    """Transaction with USD values"""
    eth_value_in_usd: float
    eth_value_out_usd: float
    gas_cost_usd: float
    net_value_usd: float
    price_timestamp: int
    price_source: str
    eth_price: float


class TransactionPricer:
    """
    Engine for calculating USD values of transactions

    Features:
    - Transaction-time price lookup
    - Block number to timestamp mapping
    - Batch value calculation
    - Historical repricing capabilities
    """

    def __init__(self, price_data: Dict[int, Dict]):
        """
        Initialize transaction pricer

        Args:
            price_data: Dict of {timestamp -> price_data}
        """
        self.price_data = price_data
        logger.info(f"Transaction pricer initialized with {len(price_data)} price points")

    def calculate_transaction_values(
        self,
        transactions: List[Transaction]
    ) -> List[EnrichedTransaction]:
        """
        Calculate USD values for all transactions

        Args:
            transactions: List of Transaction objects

        Returns:
            List of EnrichedTransaction objects with USD values
        """
        enriched_transactions = []
        missing_prices = []

        for tx in transactions:
            try:
                # Get ETH price at transaction time
                eth_price_data = self.get_price_at_timestamp(tx.timestamp)

                if eth_price_data:
                    # Calculate USD values
                    eth_value_in_usd = tx.eth_value_in * eth_price_data['price_usd']
                    eth_value_out_usd = tx.eth_value_out * eth_price_data['price_usd']
                    gas_cost_usd = tx.gas_cost_eth * eth_price_data['price_usd']

                    # Calculate net value (out - in - gas)
                    net_value_usd = eth_value_out_usd - eth_value_in_usd - gas_cost_usd

                    enriched_tx = EnrichedTransaction(
                        tx_hash=tx.tx_hash,
                        timestamp=tx.timestamp,
                        block_number=tx.block_number,
                        eth_value_in=tx.eth_value_in,
                        eth_value_out=tx.eth_value_out,
                        gas_cost_eth=tx.gas_cost_eth,
                        eth_value_in_usd=eth_value_in_usd,
                        eth_value_out_usd=eth_value_out_usd,
                        gas_cost_usd=gas_cost_usd,
                        net_value_usd=net_value_usd,
                        price_timestamp=eth_price_data['timestamp'],
                        price_source=eth_price_data.get('source', 'unknown'),
                        eth_price=eth_price_data['price_usd']
                    )

                    enriched_transactions.append(enriched_tx)

                else:
                    missing_prices.append(tx.tx_hash)
                    logger.warning(
                        f"No price data for transaction {tx.tx_hash} at {tx.timestamp}"
                    )

            except Exception as e:
                logger.error(f"Error pricing transaction {tx.tx_hash}: {e}")
                missing_prices.append(tx.tx_hash)

        if missing_prices:
            logger.warning(
                f"Failed to price {len(missing_prices)}/{len(transactions)} transactions"
            )

        logger.info(
            f"Successfully priced {len(enriched_transactions)}/{len(transactions)} transactions"
        )

        return enriched_transactions

    def get_price_at_timestamp(self, timestamp: int) -> Optional[Dict]:
        """
        Get the closest price to the target timestamp

        Args:
            timestamp: Unix timestamp

        Returns:
            Price data dict or None
        """
        # Round to nearest hour
        target_hour = int(timestamp // 3600) * 3600

        # Exact match
        if target_hour in self.price_data:
            return self.price_data[target_hour]

        # Find closest hour within reasonable range (±2 hours)
        for offset in [3600, -3600, 7200, -7200]:
            candidate_hour = target_hour + offset
            if candidate_hour in self.price_data:
                logger.debug(
                    f"Using price from {candidate_hour} for timestamp {timestamp} "
                    f"(offset: {offset/3600:.1f}h)"
                )
                return self.price_data[candidate_hour]

        # No price within 2 hours
        logger.warning(f"No price within 2 hours of timestamp {timestamp}")
        return None

    def calculate_single_transaction_value(
        self,
        transaction: Transaction
    ) -> Optional[EnrichedTransaction]:
        """
        Calculate USD value for a single transaction

        Args:
            transaction: Transaction object

        Returns:
            EnrichedTransaction or None if price unavailable
        """
        result = self.calculate_transaction_values([transaction])
        return result[0] if result else None

    def reprice_transactions(
        self,
        transactions: List[EnrichedTransaction],
        new_price_data: Dict[int, Dict]
    ) -> List[EnrichedTransaction]:
        """
        Reprice transactions with updated price data

        Args:
            transactions: List of already-priced transactions
            new_price_data: Updated price data

        Returns:
            List of repriced transactions
        """
        logger.info(f"Repricing {len(transactions)} transactions with updated prices")

        # Temporarily swap price data
        old_price_data = self.price_data
        self.price_data = new_price_data

        # Convert back to base transactions and reprice
        base_transactions = [
            Transaction(
                tx_hash=tx.tx_hash,
                timestamp=tx.timestamp,
                block_number=tx.block_number,
                eth_value_in=tx.eth_value_in,
                eth_value_out=tx.eth_value_out,
                gas_cost_eth=tx.gas_cost_eth
            )
            for tx in transactions
        ]

        repriced = self.calculate_transaction_values(base_transactions)

        # Restore original price data
        self.price_data = old_price_data

        logger.info(f"Repriced {len(repriced)} transactions")
        return repriced

    def get_value_summary(
        self,
        transactions: List[EnrichedTransaction]
    ) -> Dict:
        """
        Calculate summary statistics for transaction values

        Args:
            transactions: List of priced transactions

        Returns:
            Dict with summary statistics
        """
        if not transactions:
            return {
                'total_transactions': 0,
                'total_eth_in': 0,
                'total_eth_out': 0,
                'total_gas_eth': 0,
                'total_usd_in': 0,
                'total_usd_out': 0,
                'total_gas_usd': 0,
                'net_value_usd': 0
            }

        total_eth_in = sum(tx.eth_value_in for tx in transactions)
        total_eth_out = sum(tx.eth_value_out for tx in transactions)
        total_gas_eth = sum(tx.gas_cost_eth for tx in transactions)

        total_usd_in = sum(tx.eth_value_in_usd for tx in transactions)
        total_usd_out = sum(tx.eth_value_out_usd for tx in transactions)
        total_gas_usd = sum(tx.gas_cost_usd for tx in transactions)

        net_value_usd = total_usd_out - total_usd_in - total_gas_usd

        return {
            'total_transactions': len(transactions),
            'total_eth_in': total_eth_in,
            'total_eth_out': total_eth_out,
            'total_gas_eth': total_gas_eth,
            'total_usd_in': total_usd_in,
            'total_usd_out': total_usd_out,
            'total_gas_usd': total_gas_usd,
            'net_value_usd': net_value_usd,
            'avg_tx_value_usd': net_value_usd / len(transactions) if transactions else 0
        }

    def filter_by_value_range(
        self,
        transactions: List[EnrichedTransaction],
        min_usd: Optional[float] = None,
        max_usd: Optional[float] = None
    ) -> List[EnrichedTransaction]:
        """
        Filter transactions by USD value range

        Args:
            transactions: List of priced transactions
            min_usd: Minimum net USD value (inclusive)
            max_usd: Maximum net USD value (inclusive)

        Returns:
            Filtered list of transactions
        """
        filtered = transactions

        if min_usd is not None:
            filtered = [tx for tx in filtered if tx.net_value_usd >= min_usd]

        if max_usd is not None:
            filtered = [tx for tx in filtered if tx.net_value_usd <= max_usd]

        logger.info(
            f"Filtered to {len(filtered)}/{len(transactions)} transactions "
            f"(min: ${min_usd}, max: ${max_usd})"
        )

        return filtered