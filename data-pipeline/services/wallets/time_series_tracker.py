"""
Time-series performance tracking module for wallet analysis.

This module provides functionality to track wallet performance metrics
over time, including daily portfolio values, rolling returns, and
cumulative performance calculations across multiple time horizons.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from decimal import Decimal
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TimeSeriesTracker:
    """
    Tracks and analyzes wallet performance over time.

    This class provides methods for calculating time-series performance
    metrics including daily portfolio values, rolling returns, and
    performance attribution across different time horizons.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the TimeSeriesTracker.

        Args:
            risk_free_rate: Annual risk-free rate for calculations (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 365

    def calculate_daily_portfolio_values(
        self,
        transactions: List[Dict],
        price_data: Dict[str, pd.Series],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Calculate daily portfolio values from transaction history.

        Args:
            transactions: List of transaction dictionaries
            price_data: Dictionary mapping token symbols to price series
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)

        Returns:
            Series of daily portfolio values indexed by date
        """
        if not transactions:
            logger.warning("No transactions provided for portfolio calculation")
            return pd.Series(dtype=float)

        # Sort transactions by timestamp
        sorted_txns = sorted(transactions, key=lambda x: x['timestamp'])

        # Determine date range
        if not start_date:
            start_date = sorted_txns[0]['timestamp'].date()
        if not end_date:
            end_date = datetime.now().date()

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Track token positions over time
        positions = defaultdict(lambda: defaultdict(float))

        # Process transactions to build position history
        for txn in sorted_txns:
            date = txn['timestamp'].date()
            token = txn['token_symbol']

            if txn['type'] == 'buy':
                positions[date][token] += txn['amount']
            elif txn['type'] == 'sell':
                positions[date][token] -= txn['amount']

        # Calculate cumulative positions
        cumulative_positions = defaultdict(lambda: defaultdict(float))
        current_positions = defaultdict(float)

        for date in date_range:
            date_as_date = date.date() if hasattr(date, 'date') else date

            # Update positions if there were transactions
            if date_as_date in positions:
                for token, amount_change in positions[date_as_date].items():
                    current_positions[token] += amount_change

            # Record current positions
            for token, amount in current_positions.items():
                if amount != 0:  # Only track non-zero positions
                    cumulative_positions[date][token] = amount

        # Calculate portfolio values
        portfolio_values = []

        for date in date_range:
            daily_value = 0.0

            for token, amount in cumulative_positions[date].items():
                if token in price_data:
                    # Get price for the date
                    try:
                        if date in price_data[token].index:
                            price = price_data[token].loc[date]
                        else:
                            # Use last available price
                            available_prices = price_data[token][price_data[token].index <= date]
                            if not available_prices.empty:
                                price = available_prices.iloc[-1]
                            else:
                                price = 0

                        daily_value += amount * price
                    except Exception as e:
                        logger.warning(f"Error getting price for {token} on {date}: {e}")

            portfolio_values.append(daily_value)

        return pd.Series(portfolio_values, index=date_range, name='portfolio_value')

    def calculate_rolling_returns(
        self,
        portfolio_values: pd.Series,
        windows: List[int] = [7, 30, 90]
    ) -> Dict[int, pd.Series]:
        """
        Calculate rolling returns over multiple time windows.

        Args:
            portfolio_values: Series of portfolio values
            windows: List of window sizes in days

        Returns:
            Dictionary mapping window size to rolling return series
        """
        if portfolio_values.empty:
            return {window: pd.Series(dtype=float) for window in windows}

        # Calculate daily returns
        daily_returns = portfolio_values.pct_change().fillna(0)

        rolling_returns = {}

        for window in windows:
            # Calculate rolling returns
            rolling_ret = daily_returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1
            )
            rolling_returns[window] = rolling_ret

        return rolling_returns

    def calculate_cumulative_returns(
        self,
        portfolio_values: pd.Series
    ) -> pd.Series:
        """
        Calculate cumulative returns from the first trade date.

        Args:
            portfolio_values: Series of portfolio values

        Returns:
            Series of cumulative returns
        """
        if portfolio_values.empty or portfolio_values.iloc[0] == 0:
            return pd.Series(dtype=float)

        # Find first non-zero value
        first_value_idx = portfolio_values[portfolio_values > 0].first_valid_index()

        if first_value_idx is None:
            return pd.Series(dtype=float)

        first_value = portfolio_values.loc[first_value_idx]

        # Calculate cumulative returns
        cumulative_returns = (portfolio_values / first_value - 1) * 100

        # Set returns before first value to 0
        cumulative_returns[:first_value_idx] = 0

        return cumulative_returns

    def calculate_time_weighted_returns(
        self,
        portfolio_values: pd.Series,
        cash_flows: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate time-weighted returns accounting for cash flows.

        Args:
            portfolio_values: Series of portfolio values
            cash_flows: Optional list of cash flow events

        Returns:
            Annualized time-weighted return
        """
        if portfolio_values.empty or len(portfolio_values) < 2:
            return 0.0

        # If no cash flows, simple return calculation
        if not cash_flows:
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            days = (portfolio_values.index[-1] - portfolio_values.index[0]).days

            if days == 0:
                return 0.0

            # Annualize the return
            return ((1 + total_return) ** (365 / days)) - 1

        # With cash flows, calculate sub-period returns
        cash_flow_dates = sorted([cf['date'] for cf in cash_flows])

        # Add start and end dates
        all_dates = [portfolio_values.index[0]] + cash_flow_dates + [portfolio_values.index[-1]]

        sub_period_returns = []

        for i in range(len(all_dates) - 1):
            start_date = all_dates[i]
            end_date = all_dates[i + 1]

            # Get values for sub-period
            start_value = portfolio_values.loc[start_date]
            end_value = portfolio_values.loc[end_date]

            if start_value > 0:
                period_return = (end_value / start_value) - 1
                sub_period_returns.append(1 + period_return)

        if not sub_period_returns:
            return 0.0

        # Calculate geometric mean
        twr = np.prod(sub_period_returns) - 1

        # Annualize
        total_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days

        if total_days == 0:
            return 0.0

        return ((1 + twr) ** (365 / total_days)) - 1

    def track_performance_consistency(
        self,
        daily_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Track performance consistency metrics over time.

        Args:
            daily_returns: Series of daily returns

        Returns:
            Dictionary of consistency metrics
        """
        if daily_returns.empty:
            return {
                'positive_days_pct': 0.0,
                'consistency_score': 0.0,
                'return_autocorrelation': 0.0,
                'streak_ratio': 0.0
            }

        # Calculate percentage of positive days
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        positive_days_pct = (positive_days / total_days * 100) if total_days > 0 else 0

        # Calculate consistency score (lower volatility of returns)
        if daily_returns.std() > 0:
            consistency_score = daily_returns.mean() / daily_returns.std()
        else:
            consistency_score = 0.0

        # Calculate return autocorrelation (trending behavior)
        if len(daily_returns) > 1:
            try:
                return_autocorr = daily_returns.autocorr(lag=1)
                if pd.isna(return_autocorr):
                    return_autocorr = 0.0
            except:
                return_autocorr = 0.0
        else:
            return_autocorr = 0.0

        # Calculate winning/losing streak ratio
        streaks = []
        current_streak = 0

        for ret in daily_returns:
            if ret > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 1
            elif ret < 0:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    streaks.append(current_streak)
                    current_streak = -1

        if current_streak != 0:
            streaks.append(current_streak)

        # Calculate average positive vs negative streak length
        positive_streaks = [s for s in streaks if s > 0]
        negative_streaks = [abs(s) for s in streaks if s < 0]

        avg_positive = np.mean(positive_streaks) if positive_streaks else 0
        avg_negative = np.mean(negative_streaks) if negative_streaks else 1

        streak_ratio = avg_positive / avg_negative if avg_negative > 0 else avg_positive

        return {
            'positive_days_pct': positive_days_pct,
            'consistency_score': consistency_score,
            'return_autocorrelation': return_autocorr,
            'streak_ratio': streak_ratio
        }

    def calculate_performance_attribution(
        self,
        transactions: List[Dict],
        price_data: Dict[str, pd.Series],
        narrative_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance attribution by token and narrative.

        Args:
            transactions: List of transaction dictionaries
            price_data: Dictionary mapping token symbols to price series
            narrative_mapping: Optional mapping of tokens to narrative categories

        Returns:
            Dictionary with attribution by token and narrative
        """
        attribution = {
            'by_token': defaultdict(float),
            'by_narrative': defaultdict(float),
            'total_return': 0.0
        }

        if not transactions:
            return attribution

        # Group transactions by token
        token_transactions = defaultdict(list)

        for txn in transactions:
            token_transactions[txn['token_symbol']].append(txn)

        total_pnl = 0.0

        # Calculate P&L for each token
        for token, txns in token_transactions.items():
            # Sort by timestamp
            txns_sorted = sorted(txns, key=lambda x: x['timestamp'])

            # Track position and cost basis
            position = 0.0
            cost_basis = 0.0
            realized_pnl = 0.0

            for txn in txns_sorted:
                if txn['type'] == 'buy':
                    # Update position and cost basis
                    position += txn['amount']
                    cost_basis += txn['amount'] * txn['price']

                elif txn['type'] == 'sell':
                    # Calculate realized P&L
                    if position > 0:
                        avg_cost = cost_basis / position if position > 0 else 0
                        sell_value = txn['amount'] * txn['price']
                        sell_cost = txn['amount'] * avg_cost

                        realized_pnl += sell_value - sell_cost

                        # Update position and cost basis
                        position -= txn['amount']
                        cost_basis -= sell_cost

            # Add unrealized P&L if position remains
            if position > 0 and token in price_data:
                try:
                    current_price = price_data[token].iloc[-1]
                    unrealized_pnl = (position * current_price) - cost_basis
                    total_token_pnl = realized_pnl + unrealized_pnl
                except:
                    total_token_pnl = realized_pnl
            else:
                total_token_pnl = realized_pnl

            attribution['by_token'][token] = total_token_pnl
            total_pnl += total_token_pnl

            # Add to narrative attribution if mapping provided
            if narrative_mapping and token in narrative_mapping:
                narrative = narrative_mapping[token]
                attribution['by_narrative'][narrative] += total_token_pnl

        attribution['total_return'] = total_pnl

        # Convert to regular dict
        attribution['by_token'] = dict(attribution['by_token'])
        attribution['by_narrative'] = dict(attribution['by_narrative'])

        return attribution

    def calculate_volatility_over_time(
        self,
        daily_returns: pd.Series,
        window: int = 30
    ) -> pd.Series:
        """
        Calculate rolling volatility over time.

        Args:
            daily_returns: Series of daily returns
            window: Rolling window size in days

        Returns:
            Series of annualized rolling volatility
        """
        if daily_returns.empty:
            return pd.Series(dtype=float)

        # Calculate rolling standard deviation
        rolling_std = daily_returns.rolling(window=window).std()

        # Annualize volatility
        annualized_vol = rolling_std * np.sqrt(365)

        return annualized_vol

    def identify_regime_changes(
        self,
        daily_returns: pd.Series,
        lookback: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Identify performance regime changes.

        Args:
            daily_returns: Series of daily returns
            lookback: Lookback period for regime detection

        Returns:
            List of regime change events
        """
        if len(daily_returns) < lookback * 2:
            return []

        regime_changes = []

        # Calculate rolling statistics
        rolling_mean = daily_returns.rolling(window=lookback).mean()
        rolling_std = daily_returns.rolling(window=lookback).std()

        # Detect significant changes in mean return
        mean_change = rolling_mean.diff()
        std_change = rolling_std.diff()

        # Define thresholds for regime change
        mean_threshold = rolling_mean.std() * 2
        vol_threshold = rolling_std.std() * 2

        for i in range(lookback, len(daily_returns)):
            date = daily_returns.index[i]

            # Check for significant mean change
            if abs(mean_change.iloc[i]) > mean_threshold:
                regime_changes.append({
                    'date': date,
                    'type': 'return_regime_change',
                    'old_mean': rolling_mean.iloc[i-1],
                    'new_mean': rolling_mean.iloc[i],
                    'magnitude': mean_change.iloc[i]
                })

            # Check for significant volatility change
            if abs(std_change.iloc[i]) > vol_threshold:
                regime_changes.append({
                    'date': date,
                    'type': 'volatility_regime_change',
                    'old_vol': rolling_std.iloc[i-1],
                    'new_vol': rolling_std.iloc[i],
                    'magnitude': std_change.iloc[i]
                })

        return regime_changes

    def generate_time_series_summary(
        self,
        portfolio_values: pd.Series,
        daily_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Generate comprehensive time-series performance summary.

        Args:
            portfolio_values: Series of portfolio values
            daily_returns: Series of daily returns

        Returns:
            Dictionary containing time-series summary statistics
        """
        if portfolio_values.empty or daily_returns.empty:
            return {}

        # Calculate various metrics
        cumulative_returns = self.calculate_cumulative_returns(portfolio_values)
        rolling_returns = self.calculate_rolling_returns(portfolio_values)
        consistency_metrics = self.track_performance_consistency(daily_returns)
        volatility_series = self.calculate_volatility_over_time(daily_returns)
        twr = self.calculate_time_weighted_returns(portfolio_values)

        summary = {
            'time_period': {
                'start_date': portfolio_values.index[0].strftime('%Y-%m-%d'),
                'end_date': portfolio_values.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(portfolio_values)
            },
            'returns': {
                'cumulative_return': cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0,
                'time_weighted_return': twr * 100,
                'best_day': daily_returns.max() * 100,
                'worst_day': daily_returns.min() * 100,
                'average_daily': daily_returns.mean() * 100
            },
            'rolling_returns': {
                f'{window}d_return': (
                    rolling_returns[window].iloc[-1] * 100
                    if window in rolling_returns and not rolling_returns[window].empty
                    else 0
                )
                for window in [7, 30, 90]
            },
            'consistency': consistency_metrics,
            'volatility': {
                'current': volatility_series.iloc[-1] if not volatility_series.empty else 0,
                'average': volatility_series.mean() if not volatility_series.empty else 0,
                'max': volatility_series.max() if not volatility_series.empty else 0,
                'min': volatility_series.min() if not volatility_series.empty else 0
            }
        }

        return summary