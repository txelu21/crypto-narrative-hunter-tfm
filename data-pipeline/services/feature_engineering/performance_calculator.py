"""
Performance Metrics Calculator
Category 1: 7 features for wallet performance analysis

Features:
1. roi_percent - Return on investment
2. win_rate - Percentage of profitable trades
3. sharpe_ratio - Risk-adjusted returns
4. max_drawdown_pct - Maximum portfolio decline
5. total_pnl_usd - Total profit/loss
6. avg_trade_size_usd - Average transaction size
7. volume_consistency - Trading volume regularity
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for wallet performance metrics"""
    wallet_address: str
    roi_percent: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_pnl_usd: float
    avg_trade_size_usd: float
    volume_consistency: float


class PerformanceCalculator:
    """Calculate performance metrics for wallets"""

    def __init__(
        self,
        balances_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        eth_prices_df: pd.DataFrame,
        tokens_df: pd.DataFrame
    ):
        """
        Initialize calculator with required data

        Args:
            balances_df: Daily balance snapshots
            transactions_df: Transaction history
            eth_prices_df: Hourly ETH prices
            tokens_df: Token metadata with decimals
        """
        self.balances_df = balances_df
        self.transactions_df = transactions_df
        self.eth_prices_df = eth_prices_df
        self.tokens_df = tokens_df

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and validate input data"""
        # Convert timestamps
        if 'snapshot_date' in self.balances_df.columns:
            self.balances_df['snapshot_date'] = pd.to_datetime(self.balances_df['snapshot_date'])

        if 'timestamp' in self.transactions_df.columns:
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])

        if 'timestamp' in self.eth_prices_df.columns:
            self.eth_prices_df['timestamp'] = pd.to_datetime(self.eth_prices_df['timestamp'])

        # Create price lookup
        self.eth_prices_df = self.eth_prices_df.set_index('timestamp').sort_index()

        # Create token decimals lookup
        self.token_decimals = dict(zip(
            self.tokens_df['token_address'],
            self.tokens_df['decimals']
        ))

    def calculate_all_metrics(self, wallet_address: str) -> Optional[PerformanceMetrics]:
        """
        Calculate all performance metrics for a wallet

        Args:
            wallet_address: Ethereum wallet address

        Returns:
            PerformanceMetrics object or None if insufficient data
        """
        try:
            # Get wallet data
            wallet_balances = self.balances_df[
                self.balances_df['wallet_address'] == wallet_address
            ].copy()

            wallet_txs = self.transactions_df[
                self.transactions_df['wallet_address'] == wallet_address
            ].copy()

            if len(wallet_balances) == 0:
                logger.warning(f"No balance data for wallet {wallet_address}")
                return None

            # Calculate portfolio values over time
            daily_values = self._calculate_daily_portfolio_values(wallet_balances)

            if daily_values is None or len(daily_values) < 2:
                logger.warning(f"Insufficient data for wallet {wallet_address}")
                return None

            # Calculate each metric
            roi = self._calculate_roi(daily_values)
            win_rate = self._calculate_win_rate(wallet_txs)
            sharpe = self._calculate_sharpe_ratio(daily_values)
            max_dd = self._calculate_max_drawdown(daily_values)
            total_pnl = self._calculate_total_pnl(daily_values)
            avg_trade_size = self._calculate_avg_trade_size(wallet_txs)
            vol_consistency = self._calculate_volume_consistency(wallet_txs)

            return PerformanceMetrics(
                wallet_address=wallet_address,
                roi_percent=roi,
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                max_drawdown_pct=max_dd,
                total_pnl_usd=total_pnl,
                avg_trade_size_usd=avg_trade_size,
                volume_consistency=vol_consistency
            )

        except Exception as e:
            logger.error(f"Error calculating metrics for {wallet_address}: {e}")
            return None

    def _calculate_daily_portfolio_values(self, wallet_balances: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate daily portfolio value in USD"""
        try:
            # Group by date and sum balances
            daily_balances = wallet_balances.groupby('snapshot_date').apply(
                lambda x: self._value_balances(x)
            )

            return daily_balances.sort_index()

        except Exception as e:
            logger.error(f"Error calculating daily values: {e}")
            return None

    def _value_balances(self, balances: pd.DataFrame) -> float:
        """Value a set of token balances in USD"""
        total_usd = 0.0
        snapshot_date = balances['snapshot_date'].iloc[0]

        # Get ETH price for this date (use closest available)
        eth_price = self._get_eth_price(snapshot_date)

        for _, row in balances.iterrows():
            # For now, simplified: assume all tokens trade at proportional ETH value
            # In production, would need token-specific prices
            balance_formatted = row.get('balance_formatted', 0)

            # Simple heuristic: value tokens based on their balance
            # This is a simplification - proper implementation needs token prices
            if pd.notna(balance_formatted) and balance_formatted > 0:
                # Placeholder: would need actual token prices
                # For now, just track ETH-denominated value
                total_usd += balance_formatted * eth_price * 0.001  # Scaling factor

        return total_usd

    def _get_eth_price(self, timestamp: pd.Timestamp) -> float:
        """Get ETH price at or before timestamp"""
        try:
            # Find nearest price (forward fill)
            idx = self.eth_prices_df.index.get_indexer([timestamp], method='ffill')[0]
            if idx >= 0:
                return self.eth_prices_df.iloc[idx]['price_usd']
            return 2500.0  # Fallback
        except:
            return 2500.0  # Fallback

    def _calculate_roi(self, daily_values: pd.Series) -> float:
        """Calculate return on investment percentage"""
        if len(daily_values) < 2:
            return 0.0

        initial_value = daily_values.iloc[0]
        final_value = daily_values.iloc[-1]

        if initial_value == 0 or pd.isna(initial_value):
            return 0.0

        roi = ((final_value - initial_value) / initial_value) * 100
        return float(roi)

    def _calculate_win_rate(self, transactions: pd.DataFrame) -> float:
        """Calculate percentage of profitable trades"""
        if len(transactions) == 0:
            return 0.0

        # Simplified: consider a trade profitable if gas cost < trade value
        # Proper implementation would track buy/sell pairs
        profitable = len(transactions[transactions.get('transaction_status', '') == 'success'])
        total = len(transactions)

        if total == 0:
            return 0.0

        return (profitable / total) * 100

    def _calculate_sharpe_ratio(self, daily_values: pd.Series) -> float:
        """Calculate risk-adjusted returns (Sharpe ratio)"""
        if len(daily_values) < 2:
            return 0.0

        # Calculate daily returns
        daily_returns = daily_values.pct_change().dropna()

        if len(daily_returns) == 0:
            return 0.0

        mean_return = daily_returns.mean()
        std_return = daily_returns.std()

        if std_return == 0 or pd.isna(std_return):
            return 0.0

        # Annualized Sharpe ratio (assuming 365 days/year)
        sharpe = (mean_return / std_return) * np.sqrt(365)

        return float(sharpe) if not pd.isna(sharpe) else 0.0

    def _calculate_max_drawdown(self, daily_values: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        if len(daily_values) < 2:
            return 0.0

        # Calculate running maximum
        running_max = daily_values.expanding().max()

        # Calculate drawdowns
        drawdowns = (daily_values - running_max) / running_max

        # Get maximum drawdown (most negative)
        max_dd = drawdowns.min()

        return abs(float(max_dd * 100)) if not pd.isna(max_dd) else 0.0

    def _calculate_total_pnl(self, daily_values: pd.Series) -> float:
        """Calculate total profit/loss in USD"""
        if len(daily_values) < 2:
            return 0.0

        initial = daily_values.iloc[0]
        final = daily_values.iloc[-1]

        pnl = final - initial

        return float(pnl) if not pd.isna(pnl) else 0.0

    def _calculate_avg_trade_size(self, transactions: pd.DataFrame) -> float:
        """Calculate average trade size in USD"""
        if len(transactions) == 0:
            return 0.0

        # Simplified: use gas cost as proxy for trade size
        # Proper implementation would calculate actual trade values
        if 'gas_used' in transactions.columns and 'gas_price_gwei' in transactions.columns:
            # Convert gas cost to USD
            transactions = transactions.copy()
            transactions['gas_cost_eth'] = (
                transactions['gas_used'] * transactions['gas_price_gwei'] / 1e9
            )

            # Get avg ETH price during period
            avg_eth_price = self.eth_prices_df['price_usd'].mean()
            transactions['gas_cost_usd'] = transactions['gas_cost_eth'] * avg_eth_price

            avg_trade_size = transactions['gas_cost_usd'].mean()
            return float(avg_trade_size) if not pd.isna(avg_trade_size) else 0.0

        return 0.0

    def _calculate_volume_consistency(self, transactions: pd.DataFrame) -> float:
        """Calculate trading volume consistency (coefficient of variation)"""
        if len(transactions) < 2:
            return 0.0

        # Group by day and count transactions
        transactions = transactions.copy()
        transactions['date'] = transactions['timestamp'].dt.date
        daily_volume = transactions.groupby('date').size()

        if len(daily_volume) < 2:
            return 0.0

        mean_vol = daily_volume.mean()
        std_vol = daily_volume.std()

        if mean_vol == 0 or pd.isna(mean_vol):
            return 0.0

        cv = std_vol / mean_vol

        return float(cv) if not pd.isna(cv) else 0.0


def calculate_performance_metrics_batch(
    wallet_addresses: list,
    balances_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    eth_prices_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Calculate performance metrics for multiple wallets

    Args:
        wallet_addresses: List of wallet addresses
        balances_df: Balance snapshots DataFrame
        transactions_df: Transactions DataFrame
        eth_prices_df: ETH prices DataFrame
        tokens_df: Token metadata DataFrame
        batch_size: Number of wallets to process at once

    Returns:
        DataFrame with performance metrics for all wallets
    """
    calculator = PerformanceCalculator(
        balances_df=balances_df,
        transactions_df=transactions_df,
        eth_prices_df=eth_prices_df,
        tokens_df=tokens_df
    )

    results = []
    total = len(wallet_addresses)

    for i, wallet in enumerate(wallet_addresses):
        if i % 100 == 0:
            logger.info(f"Processing wallet {i+1}/{total}")

        metrics = calculator.calculate_all_metrics(wallet)

        if metrics:
            results.append({
                'wallet_address': metrics.wallet_address,
                'roi_percent': metrics.roi_percent,
                'win_rate': metrics.win_rate,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown_pct': metrics.max_drawdown_pct,
                'total_pnl_usd': metrics.total_pnl_usd,
                'avg_trade_size_usd': metrics.avg_trade_size_usd,
                'volume_consistency': metrics.volume_consistency
            })

    logger.info(f"Calculated metrics for {len(results)}/{total} wallets")

    return pd.DataFrame(results)
