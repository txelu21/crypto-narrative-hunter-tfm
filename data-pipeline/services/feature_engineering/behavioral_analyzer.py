"""
Behavioral Features Analyzer
Category 2: 8 features for wallet trading behavior analysis

Features:
1. trade_frequency - Trades per active day
2. avg_holding_period_days - Average time holding tokens
3. diamond_hands_score - Long-term holding tendency (0-100)
4. rotation_frequency - Portfolio churn rate
5. weekend_activity_ratio - Weekend vs weekday trading
6. night_trading_ratio - After-hours trading (0-1)
7. gas_optimization_score - Gas efficiency percentile
8. dex_diversity_score - DEX usage diversity (entropy)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, time
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class BehavioralMetrics:
    """Container for wallet behavioral metrics"""
    wallet_address: str
    trade_frequency: float
    avg_holding_period_days: float
    diamond_hands_score: float
    rotation_frequency: float
    weekend_activity_ratio: float
    night_trading_ratio: float
    gas_optimization_score: float
    dex_diversity_score: float


class BehavioralAnalyzer:
    """Analyze wallet trading behavior patterns"""

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        balances_df: pd.DataFrame
    ):
        """
        Initialize analyzer with required data

        Args:
            transactions_df: Transaction history
            balances_df: Daily balance snapshots
        """
        self.transactions_df = transactions_df
        self.balances_df = balances_df

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and validate input data"""
        # Convert timestamps
        if 'timestamp' in self.transactions_df.columns:
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])

        if 'snapshot_date' in self.balances_df.columns:
            self.balances_df['snapshot_date'] = pd.to_datetime(self.balances_df['snapshot_date'])

        # Add derived time fields
        if 'timestamp' in self.transactions_df.columns:
            self.transactions_df['date'] = self.transactions_df['timestamp'].dt.date
            self.transactions_df['day_of_week'] = self.transactions_df['timestamp'].dt.dayofweek
            self.transactions_df['hour'] = self.transactions_df['timestamp'].dt.hour

    def calculate_all_metrics(self, wallet_address: str) -> Optional[BehavioralMetrics]:
        """
        Calculate all behavioral metrics for a wallet

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            BehavioralMetrics object or None if insufficient data
        """
        try:
            # Get wallet data
            wallet_txs = self.transactions_df[
                self.transactions_df['wallet_address'].str.lower() == wallet_address.lower()
            ].copy().sort_values('timestamp')

            wallet_balances = self.balances_df[
                self.balances_df['wallet_address'].str.lower() == wallet_address.lower()
            ].copy()

            if len(wallet_txs) == 0:
                logger.warning(f"No transaction data for wallet {wallet_address}")
                return None

            # Calculate each metric
            trade_freq = self._calculate_trade_frequency(wallet_txs)
            holding_period = self._calculate_avg_holding_period(wallet_txs)
            diamond_hands = self._calculate_diamond_hands_score(wallet_txs, wallet_balances)
            rotation = self._calculate_rotation_frequency(wallet_txs, wallet_balances)
            weekend_ratio = self._calculate_weekend_activity_ratio(wallet_txs)
            night_ratio = self._calculate_night_trading_ratio(wallet_txs)
            gas_score = self._calculate_gas_optimization_score(wallet_txs)
            dex_diversity = self._calculate_dex_diversity_score(wallet_txs)

            return BehavioralMetrics(
                wallet_address=wallet_address,
                trade_frequency=trade_freq,
                avg_holding_period_days=holding_period,
                diamond_hands_score=diamond_hands,
                rotation_frequency=rotation,
                weekend_activity_ratio=weekend_ratio,
                night_trading_ratio=night_ratio,
                gas_optimization_score=gas_score,
                dex_diversity_score=dex_diversity
            )

        except Exception as e:
            logger.error(f"Error calculating behavioral metrics for {wallet_address}: {e}")
            return None

    def _calculate_trade_frequency(self, transactions: pd.DataFrame) -> float:
        """
        Calculate trades per active day

        Active day = day with at least one transaction
        """
        if len(transactions) == 0:
            return 0.0

        # Count unique trading days
        unique_days = transactions['date'].nunique()

        if unique_days == 0:
            return 0.0

        # Trades per active day
        trade_frequency = len(transactions) / unique_days
        return float(trade_frequency)

    def _calculate_avg_holding_period(self, transactions: pd.DataFrame) -> float:
        """
        Calculate average holding period in days

        Track token positions and measure time between buy and sell
        """
        if len(transactions) == 0:
            return 0.0

        # Track positions: {token: [(buy_timestamp, amount)]}
        positions = defaultdict(list)
        holding_periods = []

        eth_address = '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'

        for _, tx in transactions.iterrows():
            token_in = tx.get('token_in', '').lower() if pd.notna(tx.get('token_in')) else None
            token_out = tx.get('token_out', '').lower() if pd.notna(tx.get('token_out')) else None
            timestamp = tx['timestamp']

            # BUY: Add position
            if token_in and token_in != eth_address:
                positions[token_in].append(timestamp)

            # SELL: Calculate holding period
            if token_out and token_out != eth_address:
                if len(positions[token_out]) > 0:
                    # Use FIFO: sell from earliest buy
                    buy_timestamp = positions[token_out].pop(0)
                    holding_days = (timestamp - buy_timestamp).total_seconds() / 86400
                    holding_periods.append(holding_days)

        if len(holding_periods) == 0:
            # If no closed positions, estimate from token age
            return self._estimate_holding_from_balances(transactions)

        avg_holding = np.mean(holding_periods)
        return float(avg_holding)

    def _estimate_holding_from_balances(self, transactions: pd.DataFrame) -> float:
        """Estimate holding period from transaction timespan"""
        if len(transactions) < 2:
            return 0.0

        first_tx = transactions['timestamp'].min()
        last_tx = transactions['timestamp'].max()
        timespan_days = (last_tx - first_tx).total_seconds() / 86400

        # Rough estimate: half the timespan
        return float(timespan_days / 2)

    def _calculate_diamond_hands_score(
        self,
        transactions: pd.DataFrame,
        balances: pd.DataFrame
    ) -> float:
        """
        Calculate diamond hands score (0-100)

        Higher score = holds tokens longer (diamond hands)
        Lower score = trades frequently (paper hands)

        Based on:
        - Average holding period (longer = higher score)
        - Sell frequency (less selling = higher score)
        """
        if len(transactions) == 0:
            return 0.0

        # Component 1: Holding period score (0-50 points)
        avg_holding = self._calculate_avg_holding_period(transactions)
        # Score: 0 days = 0 points, 30+ days = 50 points
        holding_score = min(50, (avg_holding / 30) * 50)

        # Component 2: Sell frequency score (0-50 points)
        eth_address = '0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        total_trades = len(transactions)
        sells = sum(
            1 for _, tx in transactions.iterrows()
            if tx.get('token_out', '').lower() != eth_address
        )

        if total_trades > 0:
            sell_ratio = sells / total_trades
            # Lower sell ratio = higher score
            sell_score = (1 - sell_ratio) * 50
        else:
            sell_score = 0

        diamond_score = holding_score + sell_score
        return float(diamond_score)

    def _calculate_rotation_frequency(
        self,
        transactions: pd.DataFrame,
        balances: pd.DataFrame
    ) -> float:
        """
        Calculate portfolio rotation frequency (churn rate)

        Measures how often the portfolio composition changes
        Higher value = more active rotation
        """
        if len(balances) == 0:
            return 0.0

        # Group balances by date
        daily_portfolios = balances.groupby('snapshot_date')['token_address'].apply(set)

        if len(daily_portfolios) < 2:
            return 0.0

        # Calculate daily changes
        changes = 0
        for i in range(1, len(daily_portfolios)):
            prev_tokens = daily_portfolios.iloc[i-1]
            curr_tokens = daily_portfolios.iloc[i]

            # Count tokens added or removed
            tokens_added = len(curr_tokens - prev_tokens)
            tokens_removed = len(prev_tokens - curr_tokens)
            changes += tokens_added + tokens_removed

        # Average changes per day
        num_days = len(daily_portfolios) - 1
        rotation_freq = changes / num_days if num_days > 0 else 0.0

        return float(rotation_freq)

    def _calculate_weekend_activity_ratio(self, transactions: pd.DataFrame) -> float:
        """
        Calculate weekend vs weekday trading ratio

        0.0 = only weekday trading
        1.0 = only weekend trading
        0.5 = equal weekend/weekday
        """
        if len(transactions) == 0:
            return 0.0

        # Weekend = Saturday (5), Sunday (6)
        weekend_txs = sum(transactions['day_of_week'] >= 5)
        weekday_txs = sum(transactions['day_of_week'] < 5)

        total_txs = len(transactions)

        if total_txs == 0:
            return 0.0

        weekend_ratio = weekend_txs / total_txs
        return float(weekend_ratio)

    def _calculate_night_trading_ratio(self, transactions: pd.DataFrame) -> float:
        """
        Calculate night trading ratio (0-1)

        Night hours: 22:00 - 06:00 (10pm - 6am UTC)

        0.0 = no night trading
        1.0 = only night trading
        """
        if len(transactions) == 0:
            return 0.0

        # Night = 22:00 - 06:00 (hours 22, 23, 0, 1, 2, 3, 4, 5)
        night_hours = {22, 23, 0, 1, 2, 3, 4, 5}
        night_txs = sum(transactions['hour'].isin(night_hours))

        total_txs = len(transactions)

        if total_txs == 0:
            return 0.0

        night_ratio = night_txs / total_txs
        return float(night_ratio)

    def _calculate_gas_optimization_score(self, transactions: pd.DataFrame) -> float:
        """
        Calculate gas optimization score (0-100)

        Based on gas price paid vs median gas price
        Higher score = better at optimizing gas (paying less)
        """
        if len(transactions) == 0:
            return 0.0

        # Extract gas prices
        gas_prices = []
        for _, tx in transactions.iterrows():
            gas_price = tx.get('gas_price')
            if pd.notna(gas_price) and gas_price > 0:
                gas_prices.append(float(gas_price))

        if len(gas_prices) == 0:
            return 50.0  # Default score if no gas data

        # Calculate wallet's average gas price
        wallet_avg_gas = np.mean(gas_prices)

        # Get global median from all transactions
        all_gas_prices = self.transactions_df['gas_price'].dropna()
        if len(all_gas_prices) > 0:
            global_median_gas = all_gas_prices.median()
        else:
            return 50.0

        if global_median_gas == 0:
            return 50.0

        # Score: paying less than median = higher score
        # 50% of median = 100 points
        # Equal to median = 50 points
        # 2x median = 0 points
        ratio = wallet_avg_gas / global_median_gas

        if ratio <= 0.5:
            score = 100.0
        elif ratio <= 1.0:
            # Linear from 100 (at 0.5) to 50 (at 1.0)
            score = 100 - (ratio - 0.5) * 100
        elif ratio <= 2.0:
            # Linear from 50 (at 1.0) to 0 (at 2.0)
            score = 50 - (ratio - 1.0) * 50
        else:
            score = 0.0

        return float(max(0, min(100, score)))

    def _calculate_dex_diversity_score(self, transactions: pd.DataFrame) -> float:
        """
        Calculate DEX diversity score using entropy

        Higher score = uses many different DEXs
        Lower score = concentrated on few DEXs

        Range: 0 (only one DEX) to ~3.5 (many DEXs evenly)
        """
        if len(transactions) == 0:
            return 0.0

        # Extract DEX addresses from transactions
        # DEX = contract addresses that facilitate swaps
        dex_addresses = []

        for _, tx in transactions.iterrows():
            # In our data, the 'from' might be the user, 'to' is often the DEX contract
            contract = tx.get('contract_address')
            if pd.notna(contract):
                dex_addresses.append(contract.lower())

        if len(dex_addresses) == 0:
            return 0.0

        # Count usage of each DEX
        dex_counts = Counter(dex_addresses)
        total_txs = len(dex_addresses)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in dex_counts.values():
            probability = count / total_txs
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return float(entropy)


def calculate_behavioral_metrics_batch(
    wallet_addresses: list,
    transactions_df: pd.DataFrame,
    balances_df: pd.DataFrame,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Calculate behavioral metrics for multiple wallets

    Args:
        wallet_addresses: List of wallet addresses
        transactions_df: Transactions DataFrame
        balances_df: Balances DataFrame
        batch_size: Number of wallets to process at once

    Returns:
        DataFrame with behavioral metrics for all wallets
    """
    analyzer = BehavioralAnalyzer(
        transactions_df=transactions_df,
        balances_df=balances_df
    )

    results = []
    total = len(wallet_addresses)

    for i, wallet in enumerate(wallet_addresses):
        if i % 100 == 0:
            logger.info(f"Processing wallet {i+1}/{total}")

        metrics = analyzer.calculate_all_metrics(wallet)

        if metrics:
            results.append({
                'wallet_address': metrics.wallet_address,
                'trade_frequency': metrics.trade_frequency,
                'avg_holding_period_days': metrics.avg_holding_period_days,
                'diamond_hands_score': metrics.diamond_hands_score,
                'rotation_frequency': metrics.rotation_frequency,
                'weekend_activity_ratio': metrics.weekend_activity_ratio,
                'night_trading_ratio': metrics.night_trading_ratio,
                'gas_optimization_score': metrics.gas_optimization_score,
                'dex_diversity_score': metrics.dex_diversity_score
            })

    logger.info(f"Calculated behavioral metrics for {len(results)}/{total} wallets")

    return pd.DataFrame(results)
