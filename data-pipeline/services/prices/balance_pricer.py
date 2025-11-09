"""
Balance Value Calculation System
Converts token balances to USD using snapshot-time prices
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BalanceSnapshot:
    """Balance snapshot data structure"""
    wallet_address: str
    token_address: str
    token_symbol: str
    balance: float
    timestamp: int
    block_number: int


@dataclass
class ValuedBalance(BalanceSnapshot):
    """Balance with USD valuation"""
    token_price_usd: float
    balance_value_usd: float
    price_timestamp: int
    price_source: str


@dataclass
class PortfolioSnapshot:
    """Portfolio valuation at a point in time"""
    wallet_address: str
    timestamp: int
    balances: List[ValuedBalance]
    total_value_usd: float
    eth_price: float


class BalancePricer:
    """
    Engine for calculating USD values of token balances

    Features:
    - Snapshot-time price matching
    - Portfolio value calculation
    - Time-weighted value analysis
    - Historical repricing capabilities
    - Value aggregation and summation
    - Portfolio tracking and trend analysis
    """

    def __init__(self, eth_price_data: Dict[int, Dict], token_price_data: Optional[Dict] = None):
        """
        Initialize balance pricer

        Args:
            eth_price_data: Dict of {timestamp -> eth_price_data}
            token_price_data: Optional dict of {token_address -> {timestamp -> price_data}}
        """
        self.eth_price_data = eth_price_data
        self.token_price_data = token_price_data or {}
        logger.info(
            f"Balance pricer initialized with {len(eth_price_data)} ETH price points "
            f"and {len(self.token_price_data)} token price feeds"
        )

    def price_balance_snapshot(
        self,
        balance: BalanceSnapshot,
        use_eth_as_default: bool = True
    ) -> Optional[ValuedBalance]:
        """
        Price a single balance snapshot

        Args:
            balance: BalanceSnapshot object
            use_eth_as_default: If True, use ETH price for ETH balances

        Returns:
            ValuedBalance or None if price unavailable
        """
        try:
            # Get price at snapshot time
            price_data = self._get_token_price_at_timestamp(
                balance.token_address,
                balance.timestamp,
                balance.token_symbol
            )

            if not price_data:
                logger.warning(
                    f"No price data for {balance.token_symbol} at timestamp {balance.timestamp}"
                )
                return None

            # Calculate balance value
            balance_value_usd = balance.balance * price_data['price_usd']

            return ValuedBalance(
                wallet_address=balance.wallet_address,
                token_address=balance.token_address,
                token_symbol=balance.token_symbol,
                balance=balance.balance,
                timestamp=balance.timestamp,
                block_number=balance.block_number,
                token_price_usd=price_data['price_usd'],
                balance_value_usd=balance_value_usd,
                price_timestamp=price_data['timestamp'],
                price_source=price_data.get('source', 'unknown')
            )

        except Exception as e:
            logger.error(f"Error pricing balance snapshot: {e}")
            return None

    def calculate_portfolio_value(
        self,
        balances: List[BalanceSnapshot],
        timestamp: Optional[int] = None
    ) -> Optional[PortfolioSnapshot]:
        """
        Calculate total portfolio value across all token holdings

        Args:
            balances: List of balance snapshots for a wallet
            timestamp: Optional timestamp (uses balances' timestamp if not provided)

        Returns:
            PortfolioSnapshot with aggregated value
        """
        if not balances:
            logger.warning("No balances provided for portfolio calculation")
            return None

        # Use provided timestamp or first balance's timestamp
        portfolio_timestamp = timestamp or balances[0].timestamp
        wallet_address = balances[0].wallet_address

        valued_balances = []
        total_value_usd = 0.0

        for balance in balances:
            valued_balance = self.price_balance_snapshot(balance)
            if valued_balance:
                valued_balances.append(valued_balance)
                total_value_usd += valued_balance.balance_value_usd

        # Get ETH price for reference
        eth_price_data = self._get_price_at_timestamp(self.eth_price_data, portfolio_timestamp)
        eth_price = eth_price_data['price_usd'] if eth_price_data else 0.0

        portfolio = PortfolioSnapshot(
            wallet_address=wallet_address,
            timestamp=portfolio_timestamp,
            balances=valued_balances,
            total_value_usd=total_value_usd,
            eth_price=eth_price
        )

        logger.info(
            f"Portfolio value for {wallet_address}: ${total_value_usd:,.2f} "
            f"({len(valued_balances)}/{len(balances)} balances priced)"
        )

        return portfolio

    def calculate_time_weighted_value(
        self,
        portfolio_snapshots: List[PortfolioSnapshot]
    ) -> Dict:
        """
        Calculate time-weighted average portfolio value

        Args:
            portfolio_snapshots: List of portfolio snapshots over time

        Returns:
            Dict with time-weighted metrics
        """
        if not portfolio_snapshots:
            return {
                'time_weighted_value': 0.0,
                'average_value': 0.0,
                'total_time_seconds': 0,
                'num_snapshots': 0
            }

        # Sort by timestamp
        sorted_snapshots = sorted(portfolio_snapshots, key=lambda x: x.timestamp)

        time_weighted_sum = 0.0
        total_time = 0

        for i in range(len(sorted_snapshots) - 1):
            current = sorted_snapshots[i]
            next_snapshot = sorted_snapshots[i + 1]

            # Time period in seconds
            time_period = next_snapshot.timestamp - current.timestamp

            # Weight current value by time period
            time_weighted_sum += current.total_value_usd * time_period
            total_time += time_period

        # Calculate average
        time_weighted_average = time_weighted_sum / total_time if total_time > 0 else 0.0
        simple_average = sum(s.total_value_usd for s in sorted_snapshots) / len(sorted_snapshots)

        return {
            'time_weighted_value': time_weighted_average,
            'average_value': simple_average,
            'total_time_seconds': total_time,
            'num_snapshots': len(sorted_snapshots),
            'start_timestamp': sorted_snapshots[0].timestamp,
            'end_timestamp': sorted_snapshots[-1].timestamp,
            'start_value': sorted_snapshots[0].total_value_usd,
            'end_value': sorted_snapshots[-1].total_value_usd,
            'value_change': sorted_snapshots[-1].total_value_usd - sorted_snapshots[0].total_value_usd,
            'value_change_pct': (
                (sorted_snapshots[-1].total_value_usd - sorted_snapshots[0].total_value_usd) /
                sorted_snapshots[0].total_value_usd * 100
                if sorted_snapshots[0].total_value_usd > 0 else 0.0
            )
        }

    def reprice_balances(
        self,
        balances: List[BalanceSnapshot],
        new_price_data: Dict[int, Dict]
    ) -> List[ValuedBalance]:
        """
        Reprice balances with updated price data

        Args:
            balances: List of balance snapshots
            new_price_data: Updated price data

        Returns:
            List of repriced balances
        """
        logger.info(f"Repricing {len(balances)} balances with updated prices")

        # Temporarily swap price data
        old_eth_price_data = self.eth_price_data
        self.eth_price_data = new_price_data

        repriced = []
        for balance in balances:
            valued = self.price_balance_snapshot(balance)
            if valued:
                repriced.append(valued)

        # Restore original price data
        self.eth_price_data = old_eth_price_data

        logger.info(f"Repriced {len(repriced)}/{len(balances)} balances")
        return repriced

    def aggregate_values_by_token(
        self,
        valued_balances: List[ValuedBalance]
    ) -> Dict[str, Dict]:
        """
        Aggregate balance values by token

        Args:
            valued_balances: List of priced balances

        Returns:
            Dict of {token_symbol -> aggregated_data}
        """
        aggregated = {}

        for balance in valued_balances:
            token = balance.token_symbol

            if token not in aggregated:
                aggregated[token] = {
                    'token_address': balance.token_address,
                    'token_symbol': token,
                    'total_balance': 0.0,
                    'total_value_usd': 0.0,
                    'num_holders': 0,
                    'avg_price_usd': 0.0
                }

            aggregated[token]['total_balance'] += balance.balance
            aggregated[token]['total_value_usd'] += balance.balance_value_usd
            aggregated[token]['num_holders'] += 1

        # Calculate average prices
        for token, data in aggregated.items():
            if data['total_balance'] > 0:
                data['avg_price_usd'] = data['total_value_usd'] / data['total_balance']

        return aggregated

    def aggregate_values_by_wallet(
        self,
        valued_balances: List[ValuedBalance]
    ) -> Dict[str, Dict]:
        """
        Aggregate balance values by wallet

        Args:
            valued_balances: List of priced balances

        Returns:
            Dict of {wallet_address -> aggregated_data}
        """
        aggregated = {}

        for balance in valued_balances:
            wallet = balance.wallet_address

            if wallet not in aggregated:
                aggregated[wallet] = {
                    'wallet_address': wallet,
                    'total_value_usd': 0.0,
                    'num_tokens': 0,
                    'tokens': []
                }

            aggregated[wallet]['total_value_usd'] += balance.balance_value_usd
            aggregated[wallet]['num_tokens'] += 1
            aggregated[wallet]['tokens'].append({
                'token_symbol': balance.token_symbol,
                'balance': balance.balance,
                'value_usd': balance.balance_value_usd
            })

        return aggregated

    def track_portfolio_trend(
        self,
        portfolio_snapshots: List[PortfolioSnapshot],
        interval: str = 'daily'
    ) -> List[Dict]:
        """
        Track portfolio value trends over time

        Args:
            portfolio_snapshots: List of portfolio snapshots
            interval: Aggregation interval ('hourly', 'daily', 'weekly')

        Returns:
            List of trend data points
        """
        if not portfolio_snapshots:
            return []

        # Sort by timestamp
        sorted_snapshots = sorted(portfolio_snapshots, key=lambda x: x.timestamp)

        # Determine interval seconds
        interval_seconds = {
            'hourly': 3600,
            'daily': 86400,
            'weekly': 604800
        }.get(interval, 86400)

        # Group snapshots by interval
        trend_data = []
        current_interval_start = sorted_snapshots[0].timestamp
        interval_snapshots = []

        for snapshot in sorted_snapshots:
            if snapshot.timestamp >= current_interval_start + interval_seconds:
                # Process current interval
                if interval_snapshots:
                    avg_value = sum(s.total_value_usd for s in interval_snapshots) / len(interval_snapshots)
                    trend_data.append({
                        'timestamp': current_interval_start,
                        'interval_start': current_interval_start,
                        'interval_end': current_interval_start + interval_seconds,
                        'avg_value_usd': avg_value,
                        'min_value_usd': min(s.total_value_usd for s in interval_snapshots),
                        'max_value_usd': max(s.total_value_usd for s in interval_snapshots),
                        'num_snapshots': len(interval_snapshots)
                    })

                # Start new interval
                current_interval_start += interval_seconds
                interval_snapshots = []

            interval_snapshots.append(snapshot)

        # Process final interval
        if interval_snapshots:
            avg_value = sum(s.total_value_usd for s in interval_snapshots) / len(interval_snapshots)
            trend_data.append({
                'timestamp': current_interval_start,
                'interval_start': current_interval_start,
                'interval_end': current_interval_start + interval_seconds,
                'avg_value_usd': avg_value,
                'min_value_usd': min(s.total_value_usd for s in interval_snapshots),
                'max_value_usd': max(s.total_value_usd for s in interval_snapshots),
                'num_snapshots': len(interval_snapshots)
            })

        logger.info(f"Generated {len(trend_data)} trend data points at {interval} intervals")
        return trend_data

    def _get_token_price_at_timestamp(
        self,
        token_address: str,
        timestamp: int,
        token_symbol: str
    ) -> Optional[Dict]:
        """
        Get token price at specific timestamp

        Args:
            token_address: Token contract address
            timestamp: Unix timestamp
            token_symbol: Token symbol (for ETH detection)

        Returns:
            Price data dict or None
        """
        # Special handling for ETH
        if token_symbol.upper() in ['ETH', 'WETH', 'ETHEREUM']:
            return self._get_price_at_timestamp(self.eth_price_data, timestamp)

        # Check if we have price data for this token
        if token_address in self.token_price_data:
            return self._get_price_at_timestamp(
                self.token_price_data[token_address],
                timestamp
            )

        # No price data available
        logger.debug(f"No price data available for token {token_symbol} ({token_address})")
        return None

    def _get_price_at_timestamp(
        self,
        price_data: Dict[int, Dict],
        timestamp: int
    ) -> Optional[Dict]:
        """
        Get the closest price to target timestamp

        Args:
            price_data: Dict of {timestamp -> price_data}
            timestamp: Target unix timestamp

        Returns:
            Price data dict or None
        """
        # Round to nearest hour
        target_hour = int(timestamp // 3600) * 3600

        # Exact match
        if target_hour in price_data:
            return price_data[target_hour]

        # Find closest hour within reasonable range (±2 hours)
        for offset in [3600, -3600, 7200, -7200]:
            candidate_hour = target_hour + offset
            if candidate_hour in price_data:
                return price_data[candidate_hour]

        # No price within 2 hours
        return None

    def get_portfolio_composition(
        self,
        portfolio: PortfolioSnapshot
    ) -> List[Dict]:
        """
        Get portfolio composition breakdown by token

        Args:
            portfolio: PortfolioSnapshot object

        Returns:
            List of composition data sorted by value
        """
        if portfolio.total_value_usd == 0:
            return []

        composition = []
        for balance in portfolio.balances:
            percentage = (balance.balance_value_usd / portfolio.total_value_usd) * 100
            composition.append({
                'token_symbol': balance.token_symbol,
                'token_address': balance.token_address,
                'balance': balance.balance,
                'value_usd': balance.balance_value_usd,
                'percentage': percentage,
                'price_usd': balance.token_price_usd
            })

        # Sort by value descending
        composition.sort(key=lambda x: x['value_usd'], reverse=True)

        return composition