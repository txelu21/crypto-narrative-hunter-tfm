"""
Wallet Filtering and Validation System

This module implements multi-criteria filtering for smart money wallet datasets,
including performance thresholds, activity patterns, sybil detection, and quality assessment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from collections import defaultdict
from enum import Enum

from .performance_calculator import PerformanceMetrics, WalletPerformanceCalculator
from .risk_analyzer import RiskMetrics, AdvancedRiskAnalyzer
from .gas_analyzer import GasMetrics, MEVImpactAnalysis, GasEfficiencyAnalyzer
from .diversification_analyzer import DiversificationAnalyzer

logger = logging.getLogger(__name__)


class FilterReason(Enum):
    """Enumeration of wallet exclusion reasons."""
    PERFORMANCE_INSUFFICIENT = "performance_insufficient"
    ACTIVITY_INSUFFICIENT = "activity_insufficient"
    SYBIL_DETECTED = "sybil_detected"
    QUALITY_INSUFFICIENT = "quality_insufficient"
    MANUAL_EXCLUSION = "manual_exclusion"
    PASSED_ALL_FILTERS = "passed_all_filters"


@dataclass
class FilterCriteria:
    """Configuration for wallet filtering criteria."""

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 0.55
    min_total_return: float = 0.10
    max_drawdown: float = -0.50

    # Activity thresholds
    min_trading_days: int = 30
    min_trades: int = 20
    min_unique_tokens: int = 5
    max_inactive_days: int = 14

    # Quality thresholds
    min_volume_per_gas: float = 1000.0
    max_mev_impact: float = 0.05
    min_portfolio_diversity: float = 0.3

    # Activity patterns
    max_trades_per_day: int = 100  # Bot exclusion
    min_trading_span_days: int = 60

    # Portfolio sophistication
    min_hhi_diversity: float = 0.7  # 1 - HHI concentration
    min_narrative_categories: int = 3

    # Statistical significance
    min_trade_count_for_stats: int = 30
    max_p_value_for_significance: float = 0.05


@dataclass
class FilterResult:
    """Result of wallet filtering process."""
    wallet_address: str
    passed: bool
    reason: FilterReason
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WalletData:
    """Comprehensive wallet data for filtering analysis."""
    wallet_address: str
    trades: List[Any]  # Trade objects
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    gas_metrics: GasMetrics
    mev_analysis: MEVImpactAnalysis
    diversification_metrics: Dict[str, Any]
    activity_metrics: Dict[str, Any]
    first_seen_date: datetime
    last_active_date: datetime


class WalletFilter:
    """
    Multi-criteria wallet filtering system for smart money validation.

    This class implements configurable filtering rules for performance,
    activity patterns, portfolio quality, and sybil detection to identify
    high-quality smart money wallets for analysis.
    """

    def __init__(self, criteria: Optional[FilterCriteria] = None):
        """
        Initialize the wallet filter with configurable criteria.

        Args:
            criteria: FilterCriteria object with thresholds, defaults to standard criteria
        """
        self.criteria = criteria or FilterCriteria()
        self.performance_calculator = WalletPerformanceCalculator()
        self.risk_analyzer = AdvancedRiskAnalyzer()
        self.gas_analyzer = GasEfficiencyAnalyzer()
        self.diversification_analyzer = DiversificationAnalyzer()

        # Filter statistics
        self.filter_stats = {
            'total_processed': 0,
            'passed_performance': 0,
            'passed_activity': 0,
            'passed_quality': 0,
            'passed_sybil': 0,
            'final_passed': 0
        }

        logger.info(f"WalletFilter initialized with criteria: {self.criteria}")

    def filter_wallet(self, wallet_data: WalletData) -> FilterResult:
        """
        Apply complete filtering pipeline to a single wallet.

        Args:
            wallet_data: Complete wallet data for analysis

        Returns:
            FilterResult with pass/fail decision and detailed scores
        """
        self.filter_stats['total_processed'] += 1

        try:
            # Apply filters in sequence
            performance_result = self._check_performance_criteria(wallet_data)
            if not performance_result.passed:
                return performance_result

            activity_result = self._check_activity_criteria(wallet_data)
            if not activity_result.passed:
                return activity_result

            quality_result = self._check_quality_criteria(wallet_data)
            if not quality_result.passed:
                return quality_result

            # If all filters passed, combine scores
            combined_scores = {
                **performance_result.scores,
                **activity_result.scores,
                **quality_result.scores
            }

            self.filter_stats['final_passed'] += 1

            return FilterResult(
                wallet_address=wallet_data.wallet_address,
                passed=True,
                reason=FilterReason.PASSED_ALL_FILTERS,
                scores=combined_scores,
                details={
                    'performance_details': performance_result.details,
                    'activity_details': activity_result.details,
                    'quality_details': quality_result.details
                }
            )

        except Exception as e:
            logger.error(f"Error filtering wallet {wallet_data.wallet_address}: {e}")
            return FilterResult(
                wallet_address=wallet_data.wallet_address,
                passed=False,
                reason=FilterReason.QUALITY_INSUFFICIENT,
                details={'error': str(e)}
            )

    def _check_performance_criteria(self, wallet_data: WalletData) -> FilterResult:
        """Check if wallet meets performance thresholds."""
        metrics = wallet_data.performance_metrics
        risk_metrics = wallet_data.risk_metrics

        # Calculate performance scores
        scores = {
            'sharpe_ratio': float(metrics.sharpe_ratio) if metrics.sharpe_ratio else 0.0,
            'win_rate': float(metrics.win_rate) if metrics.win_rate else 0.0,
            'total_return': float(metrics.total_return_pct) if metrics.total_return_pct else 0.0,
            'max_drawdown': float(risk_metrics.max_drawdown) if risk_metrics.max_drawdown else -1.0
        }

        # Check minimum thresholds
        performance_checks = {
            'sharpe_ratio': scores['sharpe_ratio'] >= self.criteria.min_sharpe_ratio,
            'win_rate': scores['win_rate'] >= self.criteria.min_win_rate,
            'total_return': scores['total_return'] >= self.criteria.min_total_return,
            'max_drawdown': scores['max_drawdown'] >= self.criteria.max_drawdown
        }

        passed = all(performance_checks.values())

        if passed:
            self.filter_stats['passed_performance'] += 1

        return FilterResult(
            wallet_address=wallet_data.wallet_address,
            passed=passed,
            reason=FilterReason.PASSED_ALL_FILTERS if passed else FilterReason.PERFORMANCE_INSUFFICIENT,
            scores=scores,
            details={
                'checks': performance_checks,
                'criteria': {
                    'min_sharpe_ratio': self.criteria.min_sharpe_ratio,
                    'min_win_rate': self.criteria.min_win_rate,
                    'min_total_return': self.criteria.min_total_return,
                    'max_drawdown': self.criteria.max_drawdown
                }
            }
        )

    def _check_activity_criteria(self, wallet_data: WalletData) -> FilterResult:
        """Check if wallet meets activity and consistency thresholds."""
        activity_metrics = wallet_data.activity_metrics
        trades = wallet_data.trades

        # Calculate activity scores
        trading_days = len(set(trade.timestamp.date() for trade in trades))
        total_trades = len(trades)
        unique_tokens = len(set(trade.token_address for trade in trades))

        # Calculate time span coverage
        first_trade = min(trades, key=lambda t: t.timestamp)
        last_trade = max(trades, key=lambda t: t.timestamp)
        trading_span_days = (last_trade.timestamp - first_trade.timestamp).days

        # Calculate trading frequency patterns
        daily_trade_counts = defaultdict(int)
        for trade in trades:
            daily_trade_counts[trade.timestamp.date()] += 1

        max_daily_trades = max(daily_trade_counts.values()) if daily_trade_counts else 0

        # Calculate inactive periods
        trade_dates = sorted(set(trade.timestamp.date() for trade in trades))
        max_inactive_days = 0
        for i in range(1, len(trade_dates)):
            gap = (trade_dates[i] - trade_dates[i-1]).days - 1
            max_inactive_days = max(max_inactive_days, gap)

        scores = {
            'trading_days': trading_days,
            'total_trades': total_trades,
            'unique_tokens': unique_tokens,
            'trading_span_days': trading_span_days,
            'max_daily_trades': max_daily_trades,
            'max_inactive_days': max_inactive_days
        }

        # Check activity thresholds
        activity_checks = {
            'min_trading_days': trading_days >= self.criteria.min_trading_days,
            'min_trades': total_trades >= self.criteria.min_trades,
            'min_unique_tokens': unique_tokens >= self.criteria.min_unique_tokens,
            'max_inactive_days': max_inactive_days <= self.criteria.max_inactive_days,
            'not_bot': max_daily_trades <= self.criteria.max_trades_per_day,
            'sufficient_span': trading_span_days >= self.criteria.min_trading_span_days
        }

        passed = all(activity_checks.values())

        if passed:
            self.filter_stats['passed_activity'] += 1

        return FilterResult(
            wallet_address=wallet_data.wallet_address,
            passed=passed,
            reason=FilterReason.PASSED_ALL_FILTERS if passed else FilterReason.ACTIVITY_INSUFFICIENT,
            scores=scores,
            details={
                'checks': activity_checks,
                'criteria': {
                    'min_trading_days': self.criteria.min_trading_days,
                    'min_trades': self.criteria.min_trades,
                    'min_unique_tokens': self.criteria.min_unique_tokens,
                    'max_inactive_days': self.criteria.max_inactive_days,
                    'max_trades_per_day': self.criteria.max_trades_per_day,
                    'min_trading_span_days': self.criteria.min_trading_span_days
                }
            }
        )

    def _check_quality_criteria(self, wallet_data: WalletData) -> FilterResult:
        """Check volume efficiency, MEV impact, and portfolio quality."""
        gas_metrics = wallet_data.gas_metrics
        mev_analysis = wallet_data.mev_analysis
        diversification_metrics = wallet_data.diversification_metrics

        # Calculate quality scores
        volume_per_gas = (
            float(gas_metrics.total_volume_eth) / float(gas_metrics.total_gas_cost_eth)
            if gas_metrics.total_gas_cost_eth and float(gas_metrics.total_gas_cost_eth) > 0
            else 0.0
        )

        mev_impact = float(mev_analysis.total_mev_impact_pct) if mev_analysis.total_mev_impact_pct else 0.0

        # Portfolio diversity (1 - HHI concentration)
        hhi = diversification_metrics.get('hhi', 1.0)
        portfolio_diversity = 1.0 - hhi

        scores = {
            'volume_per_gas': volume_per_gas,
            'mev_impact': mev_impact,
            'portfolio_diversity': portfolio_diversity,
            'hhi_concentration': hhi
        }

        # Check quality thresholds
        quality_checks = {
            'sufficient_volume_efficiency': volume_per_gas >= self.criteria.min_volume_per_gas,
            'acceptable_mev_impact': mev_impact <= self.criteria.max_mev_impact,
            'sufficient_diversification': portfolio_diversity >= self.criteria.min_portfolio_diversity
        }

        passed = all(quality_checks.values())

        if passed:
            self.filter_stats['passed_quality'] += 1

        return FilterResult(
            wallet_address=wallet_data.wallet_address,
            passed=passed,
            reason=FilterReason.PASSED_ALL_FILTERS if passed else FilterReason.QUALITY_INSUFFICIENT,
            scores=scores,
            details={
                'checks': quality_checks,
                'criteria': {
                    'min_volume_per_gas': self.criteria.min_volume_per_gas,
                    'max_mev_impact': self.criteria.max_mev_impact,
                    'min_portfolio_diversity': self.criteria.min_portfolio_diversity
                }
            }
        )

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics and success rates."""
        total = self.filter_stats['total_processed']
        if total == 0:
            return self.filter_stats

        return {
            **self.filter_stats,
            'pass_rates': {
                'performance': self.filter_stats['passed_performance'] / total,
                'activity': self.filter_stats['passed_activity'] / total,
                'quality': self.filter_stats['passed_quality'] / total,
                'final': self.filter_stats['final_passed'] / total
            }
        }

    def update_criteria(self, **kwargs) -> None:
        """Update filtering criteria with new thresholds."""
        for key, value in kwargs.items():
            if hasattr(self.criteria, key):
                setattr(self.criteria, key, value)
                logger.info(f"Updated criteria {key} to {value}")
            else:
                logger.warning(f"Unknown criteria parameter: {key}")


class BatchWalletFilter:
    """Batch processing for wallet filtering with optimization."""

    def __init__(self, criteria: Optional[FilterCriteria] = None):
        """Initialize batch filter."""
        self.filter = WalletFilter(criteria)
        self.results = []

    def filter_wallets(self, wallet_data_list: List[WalletData]) -> List[FilterResult]:
        """
        Filter a batch of wallets efficiently.

        Args:
            wallet_data_list: List of WalletData objects to filter

        Returns:
            List of FilterResult objects
        """
        logger.info(f"Starting batch filtering of {len(wallet_data_list)} wallets")

        results = []
        for i, wallet_data in enumerate(wallet_data_list):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(wallet_data_list)} wallets")

            result = self.filter.filter_wallet(wallet_data)
            results.append(result)

        self.results = results
        logger.info(f"Batch filtering complete. {len([r for r in results if r.passed])} wallets passed")

        return results

    def get_passed_wallets(self) -> List[FilterResult]:
        """Get list of wallets that passed all filters."""
        return [r for r in self.results if r.passed]

    def get_failed_wallets(self) -> List[FilterResult]:
        """Get list of wallets that failed filters."""
        return [r for r in self.results if not r.passed]

    def get_failure_analysis(self) -> Dict[FilterReason, int]:
        """Analyze failure reasons across the batch."""
        failure_counts = defaultdict(int)
        for result in self.results:
            if not result.passed:
                failure_counts[result.reason] += 1

        return dict(failure_counts)