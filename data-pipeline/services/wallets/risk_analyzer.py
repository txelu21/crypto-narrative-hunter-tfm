"""
Advanced Risk-Adjusted Performance Metrics Module

This module implements sophisticated risk analysis metrics including Sharpe ratio,
Sortino ratio, maximum drawdown, Value at Risk, and other advanced risk measures
for comprehensive wallet performance assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from math import sqrt, log
import logging
from scipy import stats
from scipy.stats import norm
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Complete risk analysis metrics for a trading strategy"""
    # Volatility metrics
    total_volatility: float
    downside_volatility: float
    upside_volatility: float
    volatility_skew: float

    # Risk-adjusted return ratios
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    sterling_ratio: Optional[float]
    treynor_ratio: Optional[float]

    # Drawdown analysis
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    recovery_factor: float
    pain_index: float

    # Value at Risk measures
    var_95: float
    var_99: float
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float

    # Distribution metrics
    skewness: float
    kurtosis: float
    tail_ratio: float
    gain_to_pain_ratio: float

    # Time-based metrics
    positive_periods: float
    consecutive_losses_max: int
    profit_factor: float
    expectancy: float


@dataclass
class DrawdownPeriod:
    """Represents a drawdown period with detailed metrics"""
    start_date: datetime
    end_date: datetime
    trough_date: datetime
    recovery_date: Optional[datetime]
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]
    is_recovered: bool


class AdvancedRiskAnalyzer:
    """
    Advanced risk analysis engine for trading strategies

    Provides comprehensive risk metrics beyond basic Sharpe ratio,
    including downside risk measures, drawdown analysis, and tail risk metrics.
    """

    def __init__(self, risk_free_rate: float = 0.02, confidence_levels: List[float] = None):
        """
        Initialize risk analyzer

        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
            confidence_levels: VaR confidence levels (default: [0.95, 0.99])
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 365
        self.confidence_levels = confidence_levels or [0.95, 0.99]

    def calculate_comprehensive_risk_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series

        Args:
            returns: Daily returns series
            portfolio_values: Daily portfolio values
            benchmark_returns: Optional benchmark returns for beta calculation

        Returns:
            RiskMetrics object with complete risk analysis
        """
        if len(returns) == 0:
            return self._create_empty_risk_metrics()

        # Basic volatility measures
        volatility_metrics = self._calculate_volatility_metrics(returns)

        # Risk-adjusted ratios
        ratio_metrics = self._calculate_risk_adjusted_ratios(
            returns, benchmark_returns
        )

        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(portfolio_values)

        # Value at Risk measures
        var_metrics = self._calculate_var_metrics(returns)

        # Distribution analysis
        distribution_metrics = self._calculate_distribution_metrics(returns)

        # Time-based performance metrics
        time_metrics = self._calculate_time_based_metrics(returns)

        return RiskMetrics(
            **volatility_metrics,
            **ratio_metrics,
            **drawdown_metrics,
            **var_metrics,
            **distribution_metrics,
            **time_metrics
        )

    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate various volatility measures"""
        # Annualized total volatility
        total_volatility = returns.std() * sqrt(252)

        # Handle near-zero volatility due to floating point precision
        if total_volatility < 1e-10:
            total_volatility = 0.0

        # Downside and upside volatility
        downside_returns = returns[returns < 0]
        upside_returns = returns[returns > 0]

        downside_volatility = (
            downside_returns.std() * sqrt(252) if len(downside_returns) > 0 else 0.0
        )
        upside_volatility = (
            upside_returns.std() * sqrt(252) if len(upside_returns) > 0 else 0.0
        )

        # Handle near-zero volatility
        if downside_volatility < 1e-10:
            downside_volatility = 0.0
        if upside_volatility < 1e-10:
            upside_volatility = 0.0

        # Volatility skew (upside vs downside volatility)
        volatility_skew = (
            upside_volatility / downside_volatility
            if downside_volatility > 0 else float('inf')
        )

        return {
            'total_volatility': total_volatility,
            'downside_volatility': downside_volatility,
            'upside_volatility': upside_volatility,
            'volatility_skew': volatility_skew
        }

    def _calculate_risk_adjusted_ratios(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Optional[float]]:
        """Calculate risk-adjusted performance ratios"""
        if len(returns) < 2:
            return {
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'calmar_ratio': None,
                'sterling_ratio': None,
                'treynor_ratio': None
            }

        excess_returns = returns - self.daily_rf_rate
        annual_excess_return = excess_returns.mean() * 252

        # Sharpe Ratio
        volatility = returns.std() * sqrt(252)
        sharpe_ratio = (
            annual_excess_return / volatility
            if volatility > 1e-10 else None
        )

        # Sortino Ratio
        downside_returns = returns[returns < self.daily_rf_rate]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = annual_excess_return / (downside_returns.std() * sqrt(252))
        else:
            sortino_ratio = None

        # Calmar Ratio (will be calculated after drawdown analysis)
        calmar_ratio = None

        # Sterling Ratio (average drawdown instead of max drawdown)
        sterling_ratio = None

        # Treynor Ratio (requires benchmark)
        treynor_ratio = None
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            beta = self._calculate_beta(returns, benchmark_returns)
            if beta != 0:
                treynor_ratio = annual_excess_return / beta

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'treynor_ratio': treynor_ratio
        }

    def _calculate_beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate beta relative to benchmark"""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0

        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        return covariance / benchmark_variance if benchmark_variance > 0 else 0.0

    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown analysis"""
        if len(portfolio_values) == 0:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'recovery_factor': 0.0,
                'pain_index': 0.0
            }

        # Calculate running maximum (peak values)
        peak = portfolio_values.expanding().max()

        # Calculate drawdown series
        drawdown = (portfolio_values - peak) / peak

        # Maximum drawdown
        max_drawdown = drawdown.min()
        if pd.isna(max_drawdown):
            max_drawdown = 0.0

        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
        if pd.isna(avg_drawdown):
            avg_drawdown = 0.0

        # Detailed drawdown period analysis
        drawdown_periods = self._identify_drawdown_periods(portfolio_values, drawdown)

        # Maximum drawdown duration
        max_drawdown_duration = (
            max([dp.duration_days for dp in drawdown_periods])
            if drawdown_periods else 0
        )

        # Recovery factor (total return / max drawdown)
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        recovery_factor = (
            abs(total_return / max_drawdown) if max_drawdown != 0 else 0.0
        )

        # Pain Index (average of all negative drawdowns)
        pain_index = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
        if pd.isna(pain_index):
            pain_index = 0.0

        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'recovery_factor': recovery_factor,
            'pain_index': pain_index
        }

    def _identify_drawdown_periods(
        self,
        portfolio_values: pd.Series,
        drawdown: pd.Series
    ) -> List[DrawdownPeriod]:
        """Identify and analyze individual drawdown periods"""
        periods = []
        in_drawdown = False
        current_peak = None
        drawdown_start = None
        trough_value = None
        trough_date = None

        for date, value in portfolio_values.items():
            dd = drawdown[date]

            if not in_drawdown and dd < 0:
                # Start of new drawdown
                in_drawdown = True
                current_peak = portfolio_values[:date].max()
                drawdown_start = date
                trough_value = value
                trough_date = date

            elif in_drawdown:
                # Update trough if this is lower
                if value < trough_value:
                    trough_value = value
                    trough_date = date

                # Check if drawdown has recovered
                if dd >= 0:
                    # End of drawdown period
                    if isinstance(date, (int, float)):
                        # Handle integer index
                        duration = date - drawdown_start
                        recovery_days = date - trough_date if trough_date is not None else 0
                    else:
                        # Handle datetime index
                        duration = (date - drawdown_start).days
                        recovery_days = (date - trough_date).days if trough_date is not None else 0

                    period = DrawdownPeriod(
                        start_date=drawdown_start,
                        end_date=date,
                        trough_date=trough_date,
                        recovery_date=date,
                        peak_value=current_peak,
                        trough_value=trough_value,
                        drawdown_pct=(trough_value - current_peak) / current_peak,
                        duration_days=duration,
                        recovery_days=recovery_days,
                        is_recovered=True
                    )
                    periods.append(period)

                    in_drawdown = False
                    current_peak = None
                    drawdown_start = None
                    trough_value = None
                    trough_date = None

        # Handle case where still in drawdown at end of period
        if in_drawdown:
            last_date = portfolio_values.index[-1]
            if isinstance(last_date, (int, float)):
                duration = last_date - drawdown_start
            else:
                duration = (last_date - drawdown_start).days
            period = DrawdownPeriod(
                start_date=drawdown_start,
                end_date=portfolio_values.index[-1],
                trough_date=trough_date,
                recovery_date=None,
                peak_value=current_peak,
                trough_value=trough_value,
                drawdown_pct=(trough_value - current_peak) / current_peak,
                duration_days=duration,
                recovery_days=None,
                is_recovered=False
            )
            periods.append(period)

        return periods

    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional VaR metrics"""
        if len(returns) == 0:
            return {
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0
            }

        # Calculate VaR at different confidence levels
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR

        # Calculate Conditional VaR (Expected Shortfall)
        # Average of returns below VaR threshold
        cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        cvar_99 = returns[returns <= var_99].mean() if (returns <= var_99).any() else var_99

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }

    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return distribution characteristics"""
        if len(returns) < 4:
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'tail_ratio': 0.0,
                'gain_to_pain_ratio': 0.0
            }

        # Skewness and kurtosis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)

            # Handle NaN values
            if np.isnan(skewness):
                skewness = 0.0
            if np.isnan(kurtosis):
                kurtosis = 0.0

        # Tail ratio (90th percentile / 10th percentile)
        p90 = np.percentile(returns, 90)
        p10 = np.percentile(returns, 10)
        tail_ratio = abs(p90 / p10) if p10 != 0 else 0.0

        # Gain-to-pain ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(positive_returns) > 0 and len(negative_returns) > 0:
            gain_to_pain_ratio = positive_returns.sum() / abs(negative_returns.sum())
        else:
            gain_to_pain_ratio = 0.0

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'gain_to_pain_ratio': gain_to_pain_ratio
        }

    def _calculate_time_based_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate time-based performance metrics"""
        if len(returns) == 0:
            return {
                'positive_periods': 0.0,
                'consecutive_losses_max': 0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }

        # Percentage of positive periods
        positive_periods = (returns > 0).sum() / len(returns)

        # Maximum consecutive losses
        consecutive_losses_max = self._calculate_max_consecutive_losses(returns)

        # Profit factor (gross profit / gross loss)
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy (average win * win rate - average loss * loss rate)
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(returns)
            loss_rate = len(losses) / len(returns)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
        else:
            expectancy = 0.0

        return {
            'positive_periods': positive_periods,
            'consecutive_losses_max': consecutive_losses_max,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }

    def _calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum number of consecutive losing periods"""
        max_consecutive = 0
        current_consecutive = 0

        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def calculate_rolling_risk_metrics(
        self,
        returns: pd.Series,
        window_days: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics over time

        Args:
            returns: Daily returns series
            window_days: Rolling window size in days

        Returns:
            DataFrame with rolling risk metrics
        """
        if len(returns) < window_days:
            return pd.DataFrame()

        rolling_metrics = []

        for i in range(window_days, len(returns) + 1):
            window_returns = returns.iloc[i-window_days:i]
            date = returns.index[i-1]

            # Calculate key rolling metrics
            volatility = window_returns.std() * sqrt(252)
            sharpe = self._calculate_rolling_sharpe(window_returns)
            max_dd = self._calculate_rolling_max_drawdown(window_returns)
            var_95 = np.percentile(window_returns, 5)

            rolling_metrics.append({
                'date': date,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'var_95': var_95
            })

        return pd.DataFrame(rolling_metrics).set_index('date')

    def _calculate_rolling_sharpe(self, returns: pd.Series) -> Optional[float]:
        """Calculate Sharpe ratio for rolling window"""
        if len(returns) < 2 or returns.std() == 0:
            return None

        excess_returns = returns - self.daily_rf_rate
        return (excess_returns.mean() * 252) / (returns.std() * sqrt(252))

    def _calculate_rolling_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for rolling window"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """Create empty risk metrics for edge cases"""
        return RiskMetrics(
            total_volatility=0.0,
            downside_volatility=0.0,
            upside_volatility=0.0,
            volatility_skew=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            sterling_ratio=None,
            treynor_ratio=None,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            max_drawdown_duration=0,
            recovery_factor=0.0,
            pain_index=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            skewness=0.0,
            kurtosis=0.0,
            tail_ratio=0.0,
            gain_to_pain_ratio=0.0,
            positive_periods=0.0,
            consecutive_losses_max=0,
            profit_factor=0.0,
            expectancy=0.0
        )

    def generate_risk_report(
        self,
        risk_metrics: RiskMetrics,
        returns: pd.Series,
        portfolio_values: pd.Series
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk analysis report

        Args:
            risk_metrics: Calculated risk metrics
            returns: Return series
            portfolio_values: Portfolio value series

        Returns:
            Dictionary with formatted risk analysis report
        """
        report = {
            'summary': {
                'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1,
                'volatility': risk_metrics.total_volatility,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'var_95': risk_metrics.var_95
            },
            'risk_adjusted_returns': {
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'calmar_ratio': risk_metrics.calmar_ratio,
                'treynor_ratio': risk_metrics.treynor_ratio
            },
            'volatility_analysis': {
                'total_volatility': risk_metrics.total_volatility,
                'downside_volatility': risk_metrics.downside_volatility,
                'upside_volatility': risk_metrics.upside_volatility,
                'volatility_skew': risk_metrics.volatility_skew
            },
            'drawdown_analysis': {
                'max_drawdown': risk_metrics.max_drawdown,
                'avg_drawdown': risk_metrics.avg_drawdown,
                'max_drawdown_duration': risk_metrics.max_drawdown_duration,
                'recovery_factor': risk_metrics.recovery_factor,
                'pain_index': risk_metrics.pain_index
            },
            'tail_risk': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'cvar_95': risk_metrics.cvar_95,
                'cvar_99': risk_metrics.cvar_99
            },
            'distribution_stats': {
                'skewness': risk_metrics.skewness,
                'kurtosis': risk_metrics.kurtosis,
                'tail_ratio': risk_metrics.tail_ratio
            },
            'trading_metrics': {
                'positive_periods': risk_metrics.positive_periods,
                'consecutive_losses_max': risk_metrics.consecutive_losses_max,
                'profit_factor': risk_metrics.profit_factor,
                'expectancy': risk_metrics.expectancy,
                'gain_to_pain_ratio': risk_metrics.gain_to_pain_ratio
            }
        }

        return report