"""
Wallet Performance Metrics Calculation Engine

This module implements comprehensive trading performance analysis for smart money wallets,
calculating win rates, returns, risk metrics, and efficiency measures.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import logging
from math import sqrt, log

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade transaction"""
    timestamp: datetime
    token_address: str
    token_symbol: str
    is_buy: bool
    amount: Decimal
    price_eth: Decimal
    price_usd: Decimal
    gas_used: int
    gas_price_gwei: Decimal
    transaction_hash: str
    amount_remaining: Decimal = None

    def __post_init__(self):
        if self.amount_remaining is None:
            self.amount_remaining = self.amount


@dataclass
class TradeReturn:
    """Represents a realized trade return"""
    entry_trade: Trade
    exit_trade: Trade
    amount_traded: Decimal
    return_pct: Decimal
    return_eth: Decimal
    return_usd: Decimal
    holding_period_days: float
    gas_cost_eth: Decimal
    gas_cost_usd: Decimal
    net_return_pct: Decimal


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a wallet"""
    wallet_address: str
    calculation_date: datetime
    time_period: str

    # Basic performance
    total_trades: int
    win_rate: float
    avg_return_per_trade: float
    total_return: float
    annualized_return: float

    # Risk metrics
    volatility: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: float
    var_95: float

    # Efficiency metrics
    total_gas_cost_usd: float
    volume_per_gas: float
    net_return_after_costs: float

    # Diversification
    unique_tokens_traded: int
    hhi_concentration: float
    max_position_size: float

    # Additional metrics
    calmar_ratio: Optional[float] = None
    avg_holding_period_days: float = 0.0
    profit_factor: float = 0.0


class WalletPerformanceCalculator:
    """
    Comprehensive wallet performance calculation engine

    Calculates trading metrics, risk-adjusted returns, gas efficiency,
    and portfolio diversification measures for smart money wallets.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 365

    def calculate_wallet_performance(
        self,
        wallet_address: str,
        trades: List[Trade],
        eth_prices: Dict[datetime, Decimal],
        time_period: str = "all_time"
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for a wallet

        Args:
            wallet_address: Wallet address to analyze
            trades: List of all trades for the wallet
            eth_prices: Dictionary mapping dates to ETH/USD prices
            time_period: Time period for analysis ('30d', '90d', 'all_time')

        Returns:
            PerformanceMetrics object with complete performance data
        """
        if not trades:
            return self._create_empty_metrics(wallet_address, time_period)

        # Filter trades by time period
        filtered_trades = self._filter_trades_by_period(trades, time_period)
        if not filtered_trades:
            return self._create_empty_metrics(wallet_address, time_period)

        # Calculate trade returns using FIFO accounting
        trade_returns = self._calculate_trade_returns(filtered_trades)

        # Calculate daily portfolio values for time-series analysis
        daily_portfolio_values = self._calculate_daily_portfolio_values(
            filtered_trades, eth_prices
        )

        # Calculate basic performance metrics
        basic_metrics = self._calculate_basic_performance(trade_returns)

        # Calculate risk-adjusted metrics
        risk_metrics = self._calculate_risk_metrics(
            trade_returns, daily_portfolio_values
        )

        # Calculate gas efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            filtered_trades, eth_prices
        )

        # Calculate diversification metrics
        diversification_metrics = self._calculate_diversification_metrics(
            filtered_trades
        )

        return PerformanceMetrics(
            wallet_address=wallet_address,
            calculation_date=datetime.now(),
            time_period=time_period,
            **basic_metrics,
            **risk_metrics,
            **efficiency_metrics,
            **diversification_metrics
        )

    def _filter_trades_by_period(
        self,
        trades: List[Trade],
        time_period: str
    ) -> List[Trade]:
        """Filter trades based on specified time period"""
        if time_period == "all_time":
            return trades

        cutoff_date = datetime.now()
        if time_period == "30d":
            cutoff_date -= timedelta(days=30)
        elif time_period == "90d":
            cutoff_date -= timedelta(days=90)
        else:
            # Default to all time for unknown periods
            return trades

        return [trade for trade in trades if trade.timestamp >= cutoff_date]

    def _calculate_trade_returns(self, trades: List[Trade]) -> List[TradeReturn]:
        """
        Calculate realized trade returns using FIFO accounting method

        Matches buy and sell trades for each token to calculate returns
        """
        # Group trades by token
        token_trades = defaultdict(list)
        for trade in sorted(trades, key=lambda x: x.timestamp):
            token_trades[trade.token_address].append(trade)

        all_returns = []

        for token_address, token_trade_list in token_trades.items():
            returns = self._calculate_token_returns(token_trade_list)
            all_returns.extend(returns)

        return all_returns

    def _calculate_token_returns(self, trades: List[Trade]) -> List[TradeReturn]:
        """Calculate returns for a specific token using FIFO accounting"""
        buy_queue = deque()
        returns = []

        for trade in trades:
            if trade.is_buy:
                # Add buy trade to queue
                buy_queue.append(Trade(
                    timestamp=trade.timestamp,
                    token_address=trade.token_address,
                    token_symbol=trade.token_symbol,
                    is_buy=trade.is_buy,
                    amount=trade.amount,
                    price_eth=trade.price_eth,
                    price_usd=trade.price_usd,
                    gas_used=trade.gas_used,
                    gas_price_gwei=trade.gas_price_gwei,
                    transaction_hash=trade.transaction_hash,
                    amount_remaining=trade.amount
                ))
            else:
                # Process sell trade against buy queue
                sell_remaining = trade.amount

                while sell_remaining > 0 and buy_queue:
                    buy_trade = buy_queue[0]

                    # Determine amount to match
                    match_amount = min(sell_remaining, buy_trade.amount_remaining)

                    # Calculate return for this portion
                    trade_return = self._calculate_single_return(
                        buy_trade, trade, match_amount
                    )
                    returns.append(trade_return)

                    # Update remaining amounts
                    sell_remaining -= match_amount
                    buy_trade.amount_remaining -= match_amount

                    # Remove buy trade if fully matched
                    if buy_trade.amount_remaining <= 0:
                        buy_queue.popleft()

        return returns

    def _calculate_single_return(
        self,
        buy_trade: Trade,
        sell_trade: Trade,
        amount: Decimal
    ) -> TradeReturn:
        """Calculate return for a single buy-sell trade pair"""

        # Calculate price difference
        price_diff_eth = sell_trade.price_eth - buy_trade.price_eth
        price_diff_usd = sell_trade.price_usd - buy_trade.price_usd

        # Calculate returns
        return_pct = float(price_diff_eth / buy_trade.price_eth)
        return_eth = price_diff_eth * amount
        return_usd = price_diff_usd * amount

        # Calculate holding period
        holding_period = (sell_trade.timestamp - buy_trade.timestamp).total_seconds() / 86400

        # Calculate gas costs (proportional to trade amount)
        buy_gas_eth = (buy_trade.gas_used * buy_trade.gas_price_gwei) / Decimal(1e9)
        sell_gas_eth = (sell_trade.gas_used * sell_trade.gas_price_gwei) / Decimal(1e9)
        total_gas_eth = buy_gas_eth + sell_gas_eth

        # Convert gas cost to USD using average price
        avg_eth_price = (buy_trade.price_usd + sell_trade.price_usd) / 2
        gas_cost_usd = total_gas_eth * avg_eth_price

        # Calculate net return after gas costs
        net_return_eth = return_eth - total_gas_eth
        net_return_pct = float(net_return_eth / (buy_trade.price_eth * amount))

        return TradeReturn(
            entry_trade=buy_trade,
            exit_trade=sell_trade,
            amount_traded=amount,
            return_pct=return_pct,
            return_eth=return_eth,
            return_usd=return_usd,
            holding_period_days=holding_period,
            gas_cost_eth=total_gas_eth,
            gas_cost_usd=gas_cost_usd,
            net_return_pct=net_return_pct
        )

    def _calculate_daily_portfolio_values(
        self,
        trades: List[Trade],
        eth_prices: Dict[datetime, Decimal]
    ) -> pd.Series:
        """Calculate daily portfolio values for time-series analysis"""
        if not trades:
            return pd.Series(dtype=float)

        # Get date range
        start_date = min(trade.timestamp.date() for trade in trades)
        end_date = max(trade.timestamp.date() for trade in trades)

        # Create daily portfolio tracking
        portfolio_values = []
        portfolio_dates = []
        current_positions = defaultdict(Decimal)

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.timestamp)
        trade_idx = 0

        current_date = start_date
        while current_date <= end_date:
            # Process all trades for current date
            while (trade_idx < len(sorted_trades) and
                   sorted_trades[trade_idx].timestamp.date() <= current_date):
                trade = sorted_trades[trade_idx]

                if trade.is_buy:
                    current_positions[trade.token_address] += trade.amount
                else:
                    current_positions[trade.token_address] -= trade.amount

                trade_idx += 1

            # Calculate portfolio value for current date
            eth_price = eth_prices.get(datetime.combine(current_date, datetime.min.time()))
            if eth_price:
                portfolio_value = self._calculate_portfolio_value_at_date(
                    current_positions, current_date, eth_price
                )
                portfolio_values.append(portfolio_value)
                portfolio_dates.append(current_date)

            current_date += timedelta(days=1)

        return pd.Series(portfolio_values, index=pd.to_datetime(portfolio_dates))

    def _calculate_portfolio_value_at_date(
        self,
        positions: Dict[str, Decimal],
        date: datetime.date,
        eth_price_usd: Decimal
    ) -> float:
        """Calculate total portfolio value at a specific date"""
        # This is a simplified calculation - in practice, you'd need
        # token prices at the specific date
        total_value_eth = sum(abs(amount) for amount in positions.values())
        return float(total_value_eth * eth_price_usd)

    def _calculate_basic_performance(
        self,
        trade_returns: List[TradeReturn]
    ) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        if not trade_returns:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'total_return': 0.0,
                'annualized_return': 0.0,
                'avg_holding_period_days': 0.0,
                'profit_factor': 0.0
            }

        returns = [tr.return_pct for tr in trade_returns]
        net_returns = [tr.net_return_pct for tr in trade_returns]

        # Basic metrics
        total_trades = len(trade_returns)
        win_rate = len([r for r in returns if r > 0]) / total_trades
        avg_return = np.mean(returns)
        total_return = np.prod([1 + r for r in returns]) - 1

        # Calculate annualized return
        if trade_returns:
            total_days = (trade_returns[-1].exit_trade.timestamp -
                         trade_returns[0].entry_trade.timestamp).days
            if total_days > 0:
                annualized_return = (1 + total_return) ** (365 / total_days) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0

        # Average holding period
        avg_holding_period = np.mean([tr.holding_period_days for tr in trade_returns])

        # Profit factor (gross profit / gross loss)
        profits = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]
        profit_factor = (sum(profits) / sum(losses)) if losses else float('inf')

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_holding_period_days': avg_holding_period,
            'profit_factor': profit_factor
        }

    def _calculate_risk_metrics(
        self,
        trade_returns: List[TradeReturn],
        daily_values: pd.Series
    ) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        if not trade_returns:
            return {
                'volatility': 0.0,
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'calmar_ratio': None
            }

        returns = np.array([tr.return_pct for tr in trade_returns])

        # Calculate volatility (annualized)
        volatility = np.std(returns) * sqrt(252)  # Assuming ~252 trading days

        # Sharpe ratio
        excess_returns = returns - self.daily_rf_rate
        sharpe_ratio = (np.mean(excess_returns) * 252) / (np.std(returns) * sqrt(252)) if len(returns) > 1 else None

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_dev = np.std(downside_returns) * sqrt(252)
            if downside_dev > 0:
                sortino_ratio = (np.mean(excess_returns) * 252) / downside_dev
            else:
                sortino_ratio = None
        else:
            sortino_ratio = None

        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(daily_values)

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0

        # Calmar ratio (annual return / max drawdown)
        annual_return = np.mean(returns) * 252
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else None

        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'calmar_ratio': calmar_ratio
        }

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown from peak portfolio value"""
        if len(portfolio_values) == 0:
            return 0.0

        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return float(drawdown.min())

    def _calculate_efficiency_metrics(
        self,
        trades: List[Trade],
        eth_prices: Dict[datetime, Decimal]
    ) -> Dict[str, Any]:
        """Calculate gas efficiency and cost analysis metrics"""
        if not trades:
            return {
                'total_gas_cost_usd': 0.0,
                'volume_per_gas': 0.0,
                'net_return_after_costs': 0.0
            }

        total_gas_cost_eth = Decimal(0)
        total_volume_eth = Decimal(0)

        for trade in trades:
            # Calculate gas cost in ETH
            gas_cost_eth = (trade.gas_used * trade.gas_price_gwei) / Decimal(1e9)
            total_gas_cost_eth += gas_cost_eth

            # Calculate trading volume in ETH
            volume_eth = trade.amount * trade.price_eth
            total_volume_eth += volume_eth

        # Convert total gas cost to USD using average ETH price
        avg_eth_price = np.mean([
            float(eth_prices.get(datetime.combine(trade.timestamp.date(), datetime.min.time()), 0))
            for trade in trades
        ])
        total_gas_cost_usd = float(total_gas_cost_eth) * avg_eth_price

        # Volume per gas efficiency
        volume_per_gas = float(total_volume_eth / total_gas_cost_eth) if total_gas_cost_eth > 0 else 0.0

        # This would need trade returns to calculate net return after costs
        # For now, setting as placeholder
        net_return_after_costs = 0.0

        return {
            'total_gas_cost_usd': total_gas_cost_usd,
            'volume_per_gas': volume_per_gas,
            'net_return_after_costs': net_return_after_costs
        }

    def _calculate_diversification_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate portfolio diversification metrics"""
        if not trades:
            return {
                'unique_tokens_traded': 0,
                'hhi_concentration': 0.0,
                'max_position_size': 0.0
            }

        # Count unique tokens
        unique_tokens = len(set(trade.token_address for trade in trades))

        # Calculate token concentration (simplified)
        token_volumes = defaultdict(Decimal)
        total_volume = Decimal(0)

        for trade in trades:
            volume = trade.amount * trade.price_eth
            token_volumes[trade.token_address] += volume
            total_volume += volume

        # Herfindahl-Hirschman Index
        if total_volume > 0:
            market_shares = [float(vol / total_volume) for vol in token_volumes.values()]
            hhi = sum(share ** 2 for share in market_shares)
            max_position_size = max(market_shares)
        else:
            hhi = 0.0
            max_position_size = 0.0

        return {
            'unique_tokens_traded': unique_tokens,
            'hhi_concentration': hhi,
            'max_position_size': max_position_size
        }

    def _create_empty_metrics(
        self,
        wallet_address: str,
        time_period: str
    ) -> PerformanceMetrics:
        """Create empty metrics for wallets with no trades"""
        return PerformanceMetrics(
            wallet_address=wallet_address,
            calculation_date=datetime.now(),
            time_period=time_period,
            total_trades=0,
            win_rate=0.0,
            avg_return_per_trade=0.0,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            max_drawdown=0.0,
            var_95=0.0,
            total_gas_cost_usd=0.0,
            volume_per_gas=0.0,
            net_return_after_costs=0.0,
            unique_tokens_traded=0,
            hhi_concentration=0.0,
            max_position_size=0.0,
            calmar_ratio=None,
            avg_holding_period_days=0.0,
            profit_factor=0.0
        )


def validate_performance_metrics(performance: PerformanceMetrics) -> List[str]:
    """
    Validate calculated performance metrics for reasonableness

    Returns list of validation warnings/errors
    """
    warnings = []

    # Check for extreme outliers
    if abs(performance.total_return) > 10:  # 1000% return
        warnings.append(f"Extreme return detected: {performance.total_return:.2%}")

    # Validate Sharpe ratio bounds
    if performance.sharpe_ratio and abs(performance.sharpe_ratio) > 5:
        warnings.append(f"Unrealistic Sharpe ratio: {performance.sharpe_ratio:.2f}")

    # Check metric consistency
    if not (0 <= performance.win_rate <= 1):
        warnings.append(f"Invalid win rate: {performance.win_rate:.2f}")

    if performance.max_drawdown > 0:
        warnings.append(f"Positive max drawdown: {performance.max_drawdown:.2f}")

    if performance.hhi_concentration > 1:
        warnings.append(f"HHI concentration > 1: {performance.hhi_concentration:.3f}")

    return warnings