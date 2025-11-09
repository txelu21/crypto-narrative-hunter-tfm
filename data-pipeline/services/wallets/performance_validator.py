"""
Performance Consistency Validation System

This module implements validation algorithms to ensure consistent skill-based performance
over different time periods and market conditions, filtering out luck-based gains.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
from collections import defaultdict

from .performance_calculator import PerformanceMetrics, WalletPerformanceCalculator
from .risk_analyzer import RiskMetrics, AdvancedRiskAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TimeWindowMetrics:
    """Performance metrics for a specific time window."""
    window_name: str
    start_date: datetime
    end_date: datetime
    total_return: Decimal
    sharpe_ratio: Decimal
    win_rate: Decimal
    max_drawdown: Decimal
    volatility: Decimal
    trades_count: int
    alpha: Optional[Decimal] = None
    beta: Optional[Decimal] = None


@dataclass
class ConsistencyResult:
    """Result of performance consistency validation."""
    wallet_address: str
    is_consistent: bool
    consistency_score: float
    validation_details: Dict[str, Any] = field(default_factory=dict)
    time_window_metrics: List[TimeWindowMetrics] = field(default_factory=list)
    statistical_significance: Dict[str, Any] = field(default_factory=dict)
    skill_indicators: Dict[str, Any] = field(default_factory=dict)
    validation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketCondition:
    """Represents different market conditions for robustness testing."""
    condition_name: str
    start_date: datetime
    end_date: datetime
    market_return: Decimal
    volatility: Decimal
    condition_type: str  # "bull", "bear", "sideways", "crash"


class PerformanceValidator:
    """
    Validates performance consistency and skill vs luck across time periods.

    This class implements statistical tests and consistency checks to identify
    wallets with genuine skill-based performance rather than luck-based gains.
    """

    def __init__(self,
                 min_trades_per_window: int = 10,
                 min_statistical_significance: float = 0.05,
                 min_consistency_score: float = 0.6,
                 time_windows: Optional[List[Tuple[str, int]]] = None):
        """
        Initialize the performance validator.

        Args:
            min_trades_per_window: Minimum trades required per time window
            min_statistical_significance: Maximum p-value for statistical significance
            min_consistency_score: Minimum consistency score for validation
            time_windows: List of (window_name, days) tuples for analysis periods
        """
        self.min_trades_per_window = min_trades_per_window
        self.min_p_value = min_statistical_significance
        self.min_consistency_score = min_consistency_score

        # Default time windows for analysis
        self.time_windows = time_windows or [
            ("30_day", 30),
            ("90_day", 90),
            ("180_day", 180),
            ("365_day", 365),
            ("all_time", None)  # None means all available data
        ]

        self.performance_calculator = WalletPerformanceCalculator()
        self.risk_analyzer = AdvancedRiskAnalyzer()

        # Market condition data (would be loaded from external source)
        self.market_conditions = self._initialize_market_conditions()

        logger.info(f"PerformanceValidator initialized with {len(self.time_windows)} time windows")

    def validate_performance_consistency(self, wallet_address: str,
                                       trades: List[Any],
                                       reference_date: Optional[datetime] = None) -> ConsistencyResult:
        """
        Validate performance consistency across multiple time windows.

        Args:
            wallet_address: Wallet address to validate
            trades: List of trade objects
            reference_date: Reference date for time window calculations

        Returns:
            ConsistencyResult with validation findings
        """
        if not trades:
            return ConsistencyResult(
                wallet_address=wallet_address,
                is_consistent=False,
                consistency_score=0.0,
                validation_details={'error': 'no_trades_available'}
            )

        reference_date = reference_date or datetime.now()

        try:
            # Calculate metrics for each time window
            window_metrics = self._calculate_time_window_metrics(trades, reference_date)

            # Perform statistical significance tests
            statistical_tests = self._perform_statistical_tests(trades)

            # Assess skill vs luck indicators
            skill_indicators = self._assess_skill_indicators(window_metrics, trades)

            # Test robustness across market conditions
            market_robustness = self._test_market_condition_robustness(trades)

            # Calculate overall consistency score
            consistency_score = self._calculate_consistency_score(
                window_metrics, statistical_tests, skill_indicators, market_robustness
            )

            is_consistent = (
                consistency_score >= self.min_consistency_score and
                statistical_tests.get('statistically_significant', False)
            )

            return ConsistencyResult(
                wallet_address=wallet_address,
                is_consistent=is_consistent,
                consistency_score=consistency_score,
                validation_details={
                    'market_robustness': market_robustness,
                    'windows_analyzed': len(window_metrics),
                    'sufficient_data': len([w for w in window_metrics if w.trades_count >= self.min_trades_per_window])
                },
                time_window_metrics=window_metrics,
                statistical_significance=statistical_tests,
                skill_indicators=skill_indicators
            )

        except Exception as e:
            logger.error(f"Error validating performance consistency for {wallet_address}: {e}")
            return ConsistencyResult(
                wallet_address=wallet_address,
                is_consistent=False,
                consistency_score=0.0,
                validation_details={'error': str(e)}
            )

    def _calculate_time_window_metrics(self, trades: List[Any],
                                     reference_date: datetime) -> List[TimeWindowMetrics]:
        """Calculate performance metrics for each time window."""
        window_metrics = []

        for window_name, window_days in self.time_windows:
            if window_days is None:
                # All-time window
                window_trades = trades
                start_date = min(trade.timestamp for trade in trades)
                end_date = max(trade.timestamp for trade in trades)
            else:
                # Specific time window
                start_date = reference_date - timedelta(days=window_days)
                end_date = reference_date
                window_trades = [
                    trade for trade in trades
                    if start_date <= trade.timestamp <= end_date
                ]

            if len(window_trades) < self.min_trades_per_window:
                logger.debug(f"Insufficient trades for {window_name} window: {len(window_trades)}")
                continue

            try:
                # Calculate performance metrics for this window
                performance_metrics = self.performance_calculator.calculate_wallet_performance(
                    window_trades
                )
                risk_metrics = self.risk_analyzer.calculate_comprehensive_risk_metrics(window_trades)

                # Calculate alpha and beta if possible
                alpha, beta = self._calculate_alpha_beta(window_trades, start_date, end_date)

                window_metric = TimeWindowMetrics(
                    window_name=window_name,
                    start_date=start_date,
                    end_date=end_date,
                    total_return=performance_metrics.total_return_pct,
                    sharpe_ratio=performance_metrics.sharpe_ratio,
                    win_rate=performance_metrics.win_rate,
                    max_drawdown=risk_metrics.max_drawdown,
                    volatility=risk_metrics.volatility,
                    trades_count=len(window_trades),
                    alpha=alpha,
                    beta=beta
                )

                window_metrics.append(window_metric)

            except Exception as e:
                logger.warning(f"Error calculating metrics for {window_name} window: {e}")
                continue

        return window_metrics

    def _perform_statistical_tests(self, trades: List[Any]) -> Dict[str, Any]:
        """Perform statistical significance tests on trading returns."""
        if len(trades) < 30:  # Minimum sample size for meaningful statistics
            return {
                'statistically_significant': False,
                'reason': 'insufficient_sample_size',
                'sample_size': len(trades)
            }

        try:
            # Calculate daily returns
            daily_returns = self._calculate_daily_returns(trades)

            if len(daily_returns) < 10:
                return {
                    'statistically_significant': False,
                    'reason': 'insufficient_daily_returns',
                    'daily_returns_count': len(daily_returns)
                }

            # Test 1: One-sample t-test against zero (no skill hypothesis)
            t_stat, p_value = stats.ttest_1samp(daily_returns, 0)

            # Test 2: Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(daily_returns[:5000])  # Max 5000 samples

            # Test 3: Test for autocorrelation (skill should show some persistence)
            autocorr = self._calculate_autocorrelation(daily_returns)

            # Test 4: Information ratio (excess return per unit of tracking error)
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            information_ratio = mean_return / std_return if std_return > 0 else 0

            # Test 5: Consistency of positive returns
            positive_days = sum(1 for ret in daily_returns if ret > 0)
            positive_ratio = positive_days / len(daily_returns)

            statistically_significant = (
                p_value < self.min_p_value and
                t_stat > 0 and  # Positive performance
                abs(information_ratio) > 0.1  # Meaningful information ratio
            )

            return {
                'statistically_significant': statistically_significant,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'autocorrelation': float(autocorr),
                'information_ratio': float(information_ratio),
                'positive_days_ratio': float(positive_ratio),
                'sample_size': len(daily_returns)
            }

        except Exception as e:
            logger.error(f"Error in statistical tests: {e}")
            return {
                'statistically_significant': False,
                'error': str(e)
            }

    def _assess_skill_indicators(self, window_metrics: List[TimeWindowMetrics],
                               trades: List[Any]) -> Dict[str, Any]:
        """Assess indicators of genuine skill vs luck."""
        if not window_metrics:
            return {'skill_score': 0.0, 'indicators': {}}

        try:
            # Indicator 1: Consistency across time windows
            sharpe_ratios = [float(w.sharpe_ratio) for w in window_metrics if w.sharpe_ratio]
            win_rates = [float(w.win_rate) for w in window_metrics if w.win_rate]

            sharpe_consistency = 1.0 - (np.std(sharpe_ratios) / max(np.mean(sharpe_ratios), 0.1)) if sharpe_ratios else 0.0
            win_rate_consistency = 1.0 - np.std(win_rates) if win_rates else 0.0

            # Indicator 2: Positive alpha across windows
            alphas = [float(w.alpha) for w in window_metrics if w.alpha and w.alpha > 0]
            positive_alpha_ratio = len(alphas) / len(window_metrics) if window_metrics else 0.0

            # Indicator 3: Downside protection (better risk-adjusted returns in bear markets)
            downside_protection = self._assess_downside_protection(trades)

            # Indicator 4: Performance regression over time (improving vs declining)
            performance_trend = self._analyze_performance_trend(window_metrics)

            # Indicator 5: Skill persistence (performance that doesn't degrade over time)
            persistence_score = self._calculate_persistence_score(window_metrics)

            # Calculate overall skill score
            skill_score = (
                sharpe_consistency * 0.25 +
                win_rate_consistency * 0.2 +
                positive_alpha_ratio * 0.2 +
                downside_protection * 0.15 +
                performance_trend * 0.1 +
                persistence_score * 0.1
            )

            return {
                'skill_score': min(skill_score, 1.0),
                'indicators': {
                    'sharpe_consistency': sharpe_consistency,
                    'win_rate_consistency': win_rate_consistency,
                    'positive_alpha_ratio': positive_alpha_ratio,
                    'downside_protection': downside_protection,
                    'performance_trend': performance_trend,
                    'persistence_score': persistence_score
                }
            }

        except Exception as e:
            logger.error(f"Error assessing skill indicators: {e}")
            return {'skill_score': 0.0, 'error': str(e)}

    def _test_market_condition_robustness(self, trades: List[Any]) -> Dict[str, Any]:
        """Test performance robustness across different market conditions."""
        try:
            condition_performance = {}

            for condition in self.market_conditions:
                condition_trades = [
                    trade for trade in trades
                    if condition.start_date <= trade.timestamp <= condition.end_date
                ]

                if len(condition_trades) < self.min_trades_per_window:
                    continue

                # Calculate performance during this market condition
                metrics = self.performance_calculator.calculate_wallet_performance(condition_trades)

                condition_performance[condition.condition_name] = {
                    'total_return': float(metrics.total_return_pct),
                    'sharpe_ratio': float(metrics.sharpe_ratio) if metrics.sharpe_ratio else 0.0,
                    'win_rate': float(metrics.win_rate) if metrics.win_rate else 0.0,
                    'trades_count': len(condition_trades),
                    'market_return': float(condition.market_return),
                    'outperformance': float(metrics.total_return_pct) - float(condition.market_return)
                }

            # Calculate robustness score
            if condition_performance:
                positive_conditions = sum(
                    1 for perf in condition_performance.values()
                    if perf['outperformance'] > 0
                )
                robustness_score = positive_conditions / len(condition_performance)
            else:
                robustness_score = 0.0

            return {
                'robustness_score': robustness_score,
                'conditions_tested': len(condition_performance),
                'condition_performance': condition_performance
            }

        except Exception as e:
            logger.error(f"Error testing market condition robustness: {e}")
            return {'robustness_score': 0.0, 'error': str(e)}

    def _calculate_consistency_score(self, window_metrics: List[TimeWindowMetrics],
                                   statistical_tests: Dict[str, Any],
                                   skill_indicators: Dict[str, Any],
                                   market_robustness: Dict[str, Any]) -> float:
        """Calculate overall consistency score."""
        try:
            # Score components
            weights = {
                'statistical_significance': 0.3,
                'skill_indicators': 0.3,
                'time_window_consistency': 0.25,
                'market_robustness': 0.15
            }

            # Statistical significance score
            stat_score = 1.0 if statistical_tests.get('statistically_significant', False) else 0.0

            # Skill indicators score
            skill_score = skill_indicators.get('skill_score', 0.0)

            # Time window consistency score
            if len(window_metrics) >= 2:
                returns = [float(w.total_return) for w in window_metrics]
                sharpes = [float(w.sharpe_ratio) for w in window_metrics if w.sharpe_ratio]

                # Positive performance in majority of windows
                positive_windows = sum(1 for ret in returns if ret > 0)
                positive_ratio = positive_windows / len(returns)

                # Consistent Sharpe ratios
                sharpe_consistency = 1.0 - (np.std(sharpes) / max(np.mean(sharpes), 0.1)) if sharpes else 0.0

                time_consistency_score = (positive_ratio + sharpe_consistency) / 2
            else:
                time_consistency_score = 0.0

            # Market robustness score
            robustness_score = market_robustness.get('robustness_score', 0.0)

            # Weighted combination
            consistency_score = (
                stat_score * weights['statistical_significance'] +
                skill_score * weights['skill_indicators'] +
                time_consistency_score * weights['time_window_consistency'] +
                robustness_score * weights['market_robustness']
            )

            return min(consistency_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0

    def _calculate_daily_returns(self, trades: List[Any]) -> List[float]:
        """Calculate daily returns from trades."""
        # Group trades by date and calculate daily P&L
        daily_pnl = defaultdict(float)

        for trade in trades:
            trade_date = trade.timestamp.date()
            # Simplified P&L calculation (would need proper implementation)
            pnl = float(trade.amount) * 0.01 if trade.is_buy else float(trade.amount) * -0.01
            daily_pnl[trade_date] += pnl

        return list(daily_pnl.values())

    def _calculate_autocorrelation(self, returns: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation of returns."""
        if len(returns) <= lag:
            return 0.0

        returns_array = np.array(returns)
        return float(np.corrcoef(returns_array[:-lag], returns_array[lag:])[0, 1])

    def _calculate_alpha_beta(self, trades: List[Any], start_date: datetime,
                            end_date: datetime) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate alpha and beta vs market benchmark."""
        try:
            # This is a simplified implementation
            # In practice, would need market benchmark data
            daily_returns = self._calculate_daily_returns(trades)

            if len(daily_returns) < 10:
                return None, None

            # Mock market returns (would be real benchmark data)
            market_returns = np.random.normal(0.001, 0.02, len(daily_returns))

            # Linear regression to find alpha and beta
            X = market_returns.reshape(-1, 1)
            y = np.array(daily_returns)

            model = LinearRegression().fit(X, y)
            beta = Decimal(str(model.coef_[0]))
            alpha = Decimal(str(model.intercept_))

            return alpha, beta

        except Exception as e:
            logger.warning(f"Error calculating alpha/beta: {e}")
            return None, None

    def _assess_downside_protection(self, trades: List[Any]) -> float:
        """Assess downside protection during market stress."""
        # Simplified implementation
        # Would analyze performance during market downturns
        daily_returns = self._calculate_daily_returns(trades)

        if not daily_returns:
            return 0.0

        # Calculate downside deviation
        negative_returns = [ret for ret in daily_returns if ret < 0]
        if not negative_returns:
            return 1.0

        downside_deviation = np.std(negative_returns)
        total_deviation = np.std(daily_returns)

        # Lower downside deviation relative to total = better protection
        protection_score = 1.0 - (downside_deviation / total_deviation) if total_deviation > 0 else 0.0

        return max(protection_score, 0.0)

    def _analyze_performance_trend(self, window_metrics: List[TimeWindowMetrics]) -> float:
        """Analyze if performance is improving, stable, or declining over time."""
        if len(window_metrics) < 3:
            return 0.5  # Neutral score for insufficient data

        # Sort by window duration (shorter to longer)
        sorted_metrics = sorted(window_metrics, key=lambda w: w.end_date - w.start_date)

        sharpe_ratios = [float(w.sharpe_ratio) for w in sorted_metrics if w.sharpe_ratio]

        if len(sharpe_ratios) < 3:
            return 0.5

        # Calculate trend using linear regression
        x = np.arange(len(sharpe_ratios))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, sharpe_ratios)

        # Positive slope = improving performance
        if slope > 0:
            return min(0.5 + slope * 2, 1.0)
        else:
            return max(0.5 + slope * 2, 0.0)

    def _calculate_persistence_score(self, window_metrics: List[TimeWindowMetrics]) -> float:
        """Calculate persistence of performance across time windows."""
        if len(window_metrics) < 2:
            return 0.0

        # Count windows with positive risk-adjusted returns
        positive_sharpe_windows = sum(
            1 for w in window_metrics
            if w.sharpe_ratio and float(w.sharpe_ratio) > 0
        )

        persistence_ratio = positive_sharpe_windows / len(window_metrics)

        # Bonus for high consistency
        if persistence_ratio >= 0.8:
            return min(persistence_ratio + 0.2, 1.0)
        else:
            return persistence_ratio

    def _initialize_market_conditions(self) -> List[MarketCondition]:
        """Initialize predefined market conditions for robustness testing."""
        # This would be loaded from historical market data
        # Simplified example conditions
        base_date = datetime.now() - timedelta(days=365*2)

        return [
            MarketCondition(
                condition_name="bull_market_2023",
                start_date=base_date,
                end_date=base_date + timedelta(days=90),
                market_return=Decimal("0.15"),
                volatility=Decimal("0.20"),
                condition_type="bull"
            ),
            MarketCondition(
                condition_name="bear_market_2022",
                start_date=base_date + timedelta(days=91),
                end_date=base_date + timedelta(days=180),
                market_return=Decimal("-0.25"),
                volatility=Decimal("0.35"),
                condition_type="bear"
            ),
            MarketCondition(
                condition_name="sideways_2024",
                start_date=base_date + timedelta(days=181),
                end_date=base_date + timedelta(days=270),
                market_return=Decimal("0.02"),
                volatility=Decimal("0.15"),
                condition_type="sideways"
            )
        ]