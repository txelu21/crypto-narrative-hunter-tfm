"""
Activity Pattern and Quality Analysis System

This module analyzes trading frequency, activity consistency, portfolio sophistication,
and market timing skills to assess the quality of smart money traders.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from collections import defaultdict, Counter
from scipy import stats
from sklearn.cluster import KMeans
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ActivityMetrics:
    """Comprehensive activity metrics for a wallet."""
    wallet_address: str

    # Frequency metrics
    total_trades: int
    trading_days: int
    avg_trades_per_day: float
    trading_frequency_score: float

    # Time span metrics
    first_trade_date: datetime
    last_trade_date: datetime
    total_trading_span_days: int
    active_days_ratio: float
    max_inactive_period_days: int

    # Activity consistency
    frequency_consistency_score: float
    temporal_distribution_score: float
    activity_pattern_score: float

    # Portfolio sophistication
    unique_tokens_traded: int
    narrative_diversity_score: float
    position_sizing_sophistication: float
    risk_management_score: float

    # Market timing
    entry_timing_score: float
    exit_timing_score: float
    contrarian_behavior_score: float

    # Overall quality scores
    overall_activity_score: float
    sophistication_score: float

    calculation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradingPattern:
    """Represents a specific trading pattern."""
    pattern_type: str
    frequency: int
    confidence: float
    description: str
    quality_indicator: str  # "positive", "neutral", "negative"


@dataclass
class NarrativeCategory:
    """Represents a narrative category for diversification analysis."""
    category_name: str
    tokens: Set[str]
    trade_count: int
    volume_allocation: Decimal
    performance: Decimal


class ActivityAnalyzer:
    """
    Comprehensive activity pattern and quality analyzer for smart money wallets.

    This class evaluates trading frequency, consistency, sophistication, and market
    timing skills to identify high-quality smart money traders.
    """

    def __init__(self,
                 min_trading_days: int = 30,
                 min_trades: int = 20,
                 min_unique_tokens: int = 5,
                 sophistication_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the activity analyzer with quality thresholds.

        Args:
            min_trading_days: Minimum trading days for quality assessment
            min_trades: Minimum total trades required
            min_unique_tokens: Minimum unique tokens for diversification
            sophistication_thresholds: Custom thresholds for sophistication scoring
        """
        self.min_trading_days = min_trading_days
        self.min_trades = min_trades
        self.min_unique_tokens = min_unique_tokens

        # Sophistication scoring thresholds
        self.sophistication_thresholds = sophistication_thresholds or {
            'min_hhi_diversity': 0.7,  # 1 - HHI concentration
            'min_position_sizing_consistency': 0.6,
            'min_risk_management_score': 0.5,
            'min_narrative_categories': 3
        }

        # Narrative categories for diversification analysis
        self.narrative_categories = self._initialize_narrative_categories()

        logger.info(f"ActivityAnalyzer initialized with thresholds: {self.sophistication_thresholds}")

    def analyze_activity_patterns(self, wallet_address: str, trades: List[Any]) -> ActivityMetrics:
        """
        Perform comprehensive activity pattern analysis for a wallet.

        Args:
            wallet_address: Wallet address to analyze
            trades: List of trade objects

        Returns:
            ActivityMetrics with complete analysis results
        """
        if not trades:
            return self._create_empty_metrics(wallet_address)

        try:
            # Basic activity metrics
            frequency_metrics = self._analyze_trading_frequency(trades)
            timespan_metrics = self._analyze_timespan_coverage(trades)
            consistency_metrics = self._analyze_activity_consistency(trades)

            # Sophistication analysis
            portfolio_metrics = self._analyze_portfolio_sophistication(trades)
            narrative_metrics = self._analyze_narrative_diversification(trades)

            # Market timing analysis
            timing_metrics = self._analyze_market_timing(trades)

            # Calculate overall scores
            overall_activity_score = self._calculate_overall_activity_score(
                frequency_metrics, timespan_metrics, consistency_metrics
            )

            sophistication_score = self._calculate_sophistication_score(
                portfolio_metrics, narrative_metrics, timing_metrics
            )

            return ActivityMetrics(
                wallet_address=wallet_address,

                # Frequency metrics
                total_trades=len(trades),
                trading_days=frequency_metrics['trading_days'],
                avg_trades_per_day=frequency_metrics['avg_trades_per_day'],
                trading_frequency_score=frequency_metrics['frequency_score'],

                # Time span metrics
                first_trade_date=timespan_metrics['first_trade_date'],
                last_trade_date=timespan_metrics['last_trade_date'],
                total_trading_span_days=timespan_metrics['total_span_days'],
                active_days_ratio=timespan_metrics['active_days_ratio'],
                max_inactive_period_days=timespan_metrics['max_inactive_period'],

                # Activity consistency
                frequency_consistency_score=consistency_metrics['frequency_consistency'],
                temporal_distribution_score=consistency_metrics['temporal_distribution'],
                activity_pattern_score=consistency_metrics['pattern_score'],

                # Portfolio sophistication
                unique_tokens_traded=portfolio_metrics['unique_tokens'],
                narrative_diversity_score=narrative_metrics['diversity_score'],
                position_sizing_sophistication=portfolio_metrics['position_sizing_score'],
                risk_management_score=portfolio_metrics['risk_management_score'],

                # Market timing
                entry_timing_score=timing_metrics['entry_score'],
                exit_timing_score=timing_metrics['exit_score'],
                contrarian_behavior_score=timing_metrics['contrarian_score'],

                # Overall scores
                overall_activity_score=overall_activity_score,
                sophistication_score=sophistication_score
            )

        except Exception as e:
            logger.error(f"Error analyzing activity patterns for {wallet_address}: {e}")
            return self._create_empty_metrics(wallet_address)

    def _analyze_trading_frequency(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze trading frequency patterns and consistency."""
        if not trades:
            return {'trading_days': 0, 'avg_trades_per_day': 0.0, 'frequency_score': 0.0}

        # Group trades by date
        daily_trades = defaultdict(int)
        for trade in trades:
            trade_date = trade.timestamp.date()
            daily_trades[trade_date] += 1

        trading_days = len(daily_trades)
        total_trades = len(trades)
        avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0

        # Analyze frequency distribution
        trade_counts = list(daily_trades.values())
        frequency_std = np.std(trade_counts) if len(trade_counts) > 1 else 0
        frequency_mean = np.mean(trade_counts)

        # Score frequency consistency (lower std relative to mean is better)
        if frequency_mean > 0:
            frequency_consistency = 1.0 - min(frequency_std / frequency_mean, 1.0)
        else:
            frequency_consistency = 0.0

        # Score based on reasonable trading frequency (not too low, not bot-like)
        if avg_trades_per_day < 0.5:
            frequency_score = avg_trades_per_day * 2  # Low frequency penalty
        elif avg_trades_per_day > 50:
            frequency_score = max(0.1, 1.0 - (avg_trades_per_day - 50) / 50)  # Bot-like penalty
        else:
            frequency_score = 1.0  # Optimal range

        # Combine with consistency
        final_frequency_score = (frequency_score * 0.7 + frequency_consistency * 0.3)

        return {
            'trading_days': trading_days,
            'avg_trades_per_day': avg_trades_per_day,
            'frequency_score': final_frequency_score,
            'frequency_consistency': frequency_consistency,
            'trade_distribution': dict(daily_trades)
        }

    def _analyze_timespan_coverage(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze time span coverage and activity distribution."""
        if not trades:
            return {
                'first_trade_date': datetime.now(),
                'last_trade_date': datetime.now(),
                'total_span_days': 0,
                'active_days_ratio': 0.0,
                'max_inactive_period': 0
            }

        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        first_trade = sorted_trades[0].timestamp
        last_trade = sorted_trades[-1].timestamp
        total_span = (last_trade - first_trade).days + 1

        # Calculate active days
        active_dates = set(trade.timestamp.date() for trade in trades)
        active_days = len(active_dates)
        active_days_ratio = active_days / total_span if total_span > 0 else 0

        # Find maximum inactive period
        sorted_dates = sorted(active_dates)
        max_inactive_period = 0

        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i-1]).days - 1
            max_inactive_period = max(max_inactive_period, gap)

        return {
            'first_trade_date': first_trade,
            'last_trade_date': last_trade,
            'total_span_days': total_span,
            'active_days_ratio': active_days_ratio,
            'max_inactive_period': max_inactive_period,
            'active_days': active_days
        }

    def _analyze_activity_consistency(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze consistency of activity patterns over time."""
        if not trades:
            return {
                'frequency_consistency': 0.0,
                'temporal_distribution': 0.0,
                'pattern_score': 0.0
            }

        # Analyze hourly distribution
        hourly_distribution = [0] * 24
        for trade in trades:
            hourly_distribution[trade.timestamp.hour] += 1

        # Calculate temporal distribution score (penalize extreme concentration)
        total_trades = len(trades)
        if total_trades > 0:
            hourly_ratios = [count / total_trades for count in hourly_distribution]
            # Calculate Herfindahl-Hirschman Index for temporal concentration
            hhi = sum(ratio ** 2 for ratio in hourly_ratios)
            temporal_score = 1.0 - hhi  # Lower concentration = higher score
        else:
            temporal_score = 0.0

        # Analyze weekly patterns
        weekly_distribution = [0] * 7  # Monday = 0, Sunday = 6
        for trade in trades:
            weekly_distribution[trade.timestamp.weekday()] += 1

        # Weekly consistency score
        if total_trades > 0:
            weekly_ratios = [count / total_trades for count in weekly_distribution]
            weekly_hhi = sum(ratio ** 2 for ratio in weekly_ratios)
            weekly_score = 1.0 - weekly_hhi
        else:
            weekly_score = 0.0

        # Overall pattern score
        pattern_score = (temporal_score + weekly_score) / 2

        return {
            'frequency_consistency': 1.0,  # Calculated in frequency analysis
            'temporal_distribution': temporal_score,
            'weekly_distribution': weekly_score,
            'pattern_score': pattern_score,
            'hourly_distribution': hourly_distribution,
            'weekly_distribution_counts': weekly_distribution
        }

    def _analyze_portfolio_sophistication(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze portfolio sophistication and position management."""
        if not trades:
            return {
                'unique_tokens': 0,
                'position_sizing_score': 0.0,
                'risk_management_score': 0.0
            }

        # Token diversification
        unique_tokens = len(set(trade.token_address for trade in trades))

        # Position sizing analysis
        position_sizes = [float(trade.amount) for trade in trades]
        position_sizing_score = self._analyze_position_sizing_sophistication(position_sizes)

        # Risk management assessment
        risk_management_score = self._assess_risk_management_sophistication(trades)

        return {
            'unique_tokens': unique_tokens,
            'position_sizing_score': position_sizing_score,
            'risk_management_score': risk_management_score,
            'position_sizes': position_sizes
        }

    def _analyze_narrative_diversification(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze diversification across narrative categories."""
        if not trades:
            return {'diversity_score': 0.0, 'narrative_distribution': {}}

        # Categorize tokens by narrative
        narrative_distribution = defaultdict(lambda: {'count': 0, 'volume': 0.0})

        for trade in trades:
            narrative = self._get_token_narrative(trade.token_address)
            narrative_distribution[narrative]['count'] += 1
            narrative_distribution[narrative]['volume'] += float(trade.amount)

        # Calculate narrative diversity score
        total_volume = sum(data['volume'] for data in narrative_distribution.values())
        narrative_count = len(narrative_distribution)

        if total_volume > 0 and narrative_count > 0:
            # Calculate HHI for narrative concentration
            volume_ratios = [data['volume'] / total_volume for data in narrative_distribution.values()]
            hhi = sum(ratio ** 2 for ratio in volume_ratios)
            diversity_score = (1.0 - hhi) * min(narrative_count / 5, 1.0)  # Bonus for multiple narratives
        else:
            diversity_score = 0.0

        return {
            'diversity_score': diversity_score,
            'narrative_distribution': dict(narrative_distribution),
            'narrative_count': narrative_count
        }

    def _analyze_market_timing(self, trades: List[Any]) -> Dict[str, Any]:
        """Analyze market timing and trading sophistication."""
        if len(trades) < 10:
            return {
                'entry_score': 0.0,
                'exit_score': 0.0,
                'contrarian_score': 0.0
            }

        try:
            # Group trades by token for entry/exit analysis
            token_trades = defaultdict(list)
            for trade in trades:
                token_trades[trade.token_address].append(trade)

            entry_scores = []
            exit_scores = []
            contrarian_scores = []

            for token_address, token_trade_list in token_trades.items():
                if len(token_trade_list) < 4:  # Need sufficient trades for analysis
                    continue

                # Sort trades by timestamp
                sorted_trades = sorted(token_trade_list, key=lambda t: t.timestamp)

                # Analyze entry timing (buy decisions)
                entry_score = self._analyze_entry_timing(sorted_trades)
                entry_scores.append(entry_score)

                # Analyze exit timing (sell decisions)
                exit_score = self._analyze_exit_timing(sorted_trades)
                exit_scores.append(exit_score)

                # Analyze contrarian behavior
                contrarian_score = self._analyze_contrarian_behavior(sorted_trades)
                contrarian_scores.append(contrarian_score)

            return {
                'entry_score': np.mean(entry_scores) if entry_scores else 0.0,
                'exit_score': np.mean(exit_scores) if exit_scores else 0.0,
                'contrarian_score': np.mean(contrarian_scores) if contrarian_scores else 0.0
            }

        except Exception as e:
            logger.warning(f"Error in market timing analysis: {e}")
            return {
                'entry_score': 0.0,
                'exit_score': 0.0,
                'contrarian_score': 0.0
            }

    def _analyze_position_sizing_sophistication(self, position_sizes: List[float]) -> float:
        """Analyze sophistication of position sizing strategy."""
        if len(position_sizes) < 5:
            return 0.0

        # Check for position sizing consistency and strategy
        sizes_array = np.array(position_sizes)

        # Look for evidence of Kelly criterion or other sophisticated sizing
        # Check for log-normal distribution (common in sophisticated sizing)
        log_sizes = np.log(sizes_array + 1)  # Add 1 to avoid log(0)

        # Test for normality of log sizes (indicates systematic sizing)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(log_sizes[:5000])  # Max 5000 samples
            normality_score = 1.0 - shapiro_p if shapiro_p < 0.05 else 0.5
        except:
            normality_score = 0.0

        # Check for position sizing variance (not too uniform, not too random)
        cv = np.std(position_sizes) / np.mean(position_sizes) if np.mean(position_sizes) > 0 else 0

        # Optimal coefficient of variation for position sizing
        if 0.2 <= cv <= 1.0:
            variance_score = 1.0
        elif cv < 0.2:
            variance_score = cv / 0.2  # Too uniform
        else:
            variance_score = max(0.1, 1.0 - (cv - 1.0) / 2.0)  # Too random

        return (normality_score * 0.4 + variance_score * 0.6)

    def _assess_risk_management_sophistication(self, trades: List[Any]) -> float:
        """Assess evidence of sophisticated risk management."""
        if len(trades) < 10:
            return 0.0

        # Group trades by token to analyze position management
        token_positions = defaultdict(list)
        for trade in trades:
            token_positions[trade.token_address].append(trade)

        risk_scores = []

        for token_trades in token_positions.values():
            if len(token_trades) < 3:
                continue

            # Sort by timestamp
            sorted_trades = sorted(token_trades, key=lambda t: t.timestamp)

            # Look for evidence of stop-losses and position scaling
            position_score = self._analyze_position_management(sorted_trades)
            risk_scores.append(position_score)

        return np.mean(risk_scores) if risk_scores else 0.0

    def _analyze_entry_timing(self, token_trades: List[Any]) -> float:
        """Analyze quality of entry timing for a specific token."""
        buy_trades = [t for t in token_trades if t.is_buy]

        if len(buy_trades) < 2:
            return 0.5  # Neutral score

        # Simplified analysis - would need price data for proper implementation
        # Look for evidence of accumulation patterns, DCA, etc.

        # Check for distribution of entry times (avoid FOMO clustering)
        entry_times = [trade.timestamp for trade in buy_trades]
        time_diffs = [(entry_times[i] - entry_times[i-1]).total_seconds() / 3600
                     for i in range(1, len(entry_times))]

        if time_diffs:
            # Prefer distributed entries over clustered entries
            avg_gap = np.mean(time_diffs)
            if avg_gap > 24:  # More than 24 hours between entries on average
                return 0.8
            elif avg_gap > 6:  # More than 6 hours
                return 0.6
            else:
                return 0.3  # Too clustered (potential FOMO)

        return 0.5

    def _analyze_exit_timing(self, token_trades: List[Any]) -> float:
        """Analyze quality of exit timing for a specific token."""
        sell_trades = [t for t in token_trades if not t.is_buy]

        if len(sell_trades) < 2:
            return 0.5

        # Similar simplified analysis for exits
        # Look for evidence of profit-taking strategy, stop-losses

        # Check for staged exits vs panic selling
        exit_times = [trade.timestamp for trade in sell_trades]
        time_diffs = [(exit_times[i] - exit_times[i-1]).total_seconds() / 3600
                     for i in range(1, len(exit_times))]

        if time_diffs:
            avg_gap = np.mean(time_diffs)
            if avg_gap > 48:  # Staged exits
                return 0.8
            elif avg_gap > 12:  # Reasonable distribution
                return 0.6
            else:
                return 0.3  # Potential panic selling

        return 0.5

    def _analyze_contrarian_behavior(self, token_trades: List[Any]) -> float:
        """Analyze evidence of contrarian trading behavior."""
        # Simplified analysis - would need market sentiment data
        # Look for evidence of buying during drawdowns, selling during rallies

        if len(token_trades) < 5:
            return 0.5

        # Analyze trade timing patterns
        buy_trades = [t for t in token_trades if t.is_buy]
        sell_trades = [t for t in token_trades if not t.is_buy]

        # Simple heuristic: contrarian traders often have more buys early in position
        if buy_trades and sell_trades:
            avg_buy_time = np.mean([t.timestamp.timestamp() for t in buy_trades])
            avg_sell_time = np.mean([t.timestamp.timestamp() for t in sell_trades])

            if avg_buy_time < avg_sell_time:  # Buys before sells (accumulation)
                return 0.7
            else:
                return 0.3

        return 0.5

    def _analyze_position_management(self, token_trades: List[Any]) -> float:
        """Analyze position management sophistication for a token."""
        if len(token_trades) < 3:
            return 0.0

        # Look for evidence of systematic position management
        position_sizes = [float(trade.amount) for trade in token_trades]

        # Check for scaling in/out patterns
        buy_sizes = [float(t.amount) for t in token_trades if t.is_buy]
        sell_sizes = [float(t.amount) for t in token_trades if not t.is_buy]

        sophistication_indicators = []

        # Indicator 1: Variable position sizing
        if len(position_sizes) > 1:
            cv = np.std(position_sizes) / np.mean(position_sizes)
            if 0.2 <= cv <= 0.8:  # Reasonable variation
                sophistication_indicators.append(0.8)
            else:
                sophistication_indicators.append(0.4)

        # Indicator 2: Scaling patterns
        if len(buy_sizes) > 1:
            # Check for increasing or decreasing buy sizes (scaling strategy)
            correlation = np.corrcoef(range(len(buy_sizes)), buy_sizes)[0, 1]
            if abs(correlation) > 0.3:  # Evidence of systematic scaling
                sophistication_indicators.append(0.7)
            else:
                sophistication_indicators.append(0.3)

        return np.mean(sophistication_indicators) if sophistication_indicators else 0.0

    def _get_token_narrative(self, token_address: str) -> str:
        """Get narrative category for a token address."""
        # Simplified mapping - would use real narrative classification
        for category_name, category in self.narrative_categories.items():
            if token_address in category.tokens:
                return category_name

        # Default to "other" if not found
        return "other"

    def _calculate_overall_activity_score(self, frequency_metrics: Dict,
                                        timespan_metrics: Dict,
                                        consistency_metrics: Dict) -> float:
        """Calculate overall activity quality score."""
        weights = {
            'frequency': 0.3,
            'timespan': 0.3,
            'consistency': 0.4
        }

        # Frequency score
        frequency_score = frequency_metrics.get('frequency_score', 0.0)

        # Timespan score
        active_ratio = timespan_metrics.get('active_days_ratio', 0.0)
        timespan_score = min(active_ratio * 2, 1.0)  # Bonus for high activity ratio

        # Consistency score
        consistency_score = consistency_metrics.get('pattern_score', 0.0)

        overall_score = (
            frequency_score * weights['frequency'] +
            timespan_score * weights['timespan'] +
            consistency_score * weights['consistency']
        )

        return min(overall_score, 1.0)

    def _calculate_sophistication_score(self, portfolio_metrics: Dict,
                                      narrative_metrics: Dict,
                                      timing_metrics: Dict) -> float:
        """Calculate overall sophistication score."""
        weights = {
            'portfolio': 0.4,
            'narrative': 0.3,
            'timing': 0.3
        }

        # Portfolio sophistication
        portfolio_score = (
            portfolio_metrics.get('position_sizing_score', 0.0) * 0.5 +
            portfolio_metrics.get('risk_management_score', 0.0) * 0.5
        )

        # Narrative diversification
        narrative_score = narrative_metrics.get('diversity_score', 0.0)

        # Market timing
        timing_score = (
            timing_metrics.get('entry_score', 0.0) * 0.4 +
            timing_metrics.get('exit_score', 0.0) * 0.4 +
            timing_metrics.get('contrarian_score', 0.0) * 0.2
        )

        sophistication_score = (
            portfolio_score * weights['portfolio'] +
            narrative_score * weights['narrative'] +
            timing_score * weights['timing']
        )

        return min(sophistication_score, 1.0)

    def _create_empty_metrics(self, wallet_address: str) -> ActivityMetrics:
        """Create empty activity metrics for wallets with no data."""
        return ActivityMetrics(
            wallet_address=wallet_address,
            total_trades=0,
            trading_days=0,
            avg_trades_per_day=0.0,
            trading_frequency_score=0.0,
            first_trade_date=datetime.now(),
            last_trade_date=datetime.now(),
            total_trading_span_days=0,
            active_days_ratio=0.0,
            max_inactive_period_days=0,
            frequency_consistency_score=0.0,
            temporal_distribution_score=0.0,
            activity_pattern_score=0.0,
            unique_tokens_traded=0,
            narrative_diversity_score=0.0,
            position_sizing_sophistication=0.0,
            risk_management_score=0.0,
            entry_timing_score=0.0,
            exit_timing_score=0.0,
            contrarian_behavior_score=0.0,
            overall_activity_score=0.0,
            sophistication_score=0.0
        )

    def _initialize_narrative_categories(self) -> Dict[str, NarrativeCategory]:
        """Initialize narrative categories for token classification."""
        # Simplified narrative categories - would be loaded from external data
        return {
            "defi": NarrativeCategory(
                category_name="defi",
                tokens=set(),  # Would be populated with real token addresses
                trade_count=0,
                volume_allocation=Decimal("0"),
                performance=Decimal("0")
            ),
            "gaming": NarrativeCategory(
                category_name="gaming",
                tokens=set(),
                trade_count=0,
                volume_allocation=Decimal("0"),
                performance=Decimal("0")
            ),
            "ai": NarrativeCategory(
                category_name="ai",
                tokens=set(),
                trade_count=0,
                volume_allocation=Decimal("0"),
                performance=Decimal("0")
            ),
            "meme": NarrativeCategory(
                category_name="meme",
                tokens=set(),
                trade_count=0,
                volume_allocation=Decimal("0"),
                performance=Decimal("0")
            ),
            "other": NarrativeCategory(
                category_name="other",
                tokens=set(),
                trade_count=0,
                volume_allocation=Decimal("0"),
                performance=Decimal("0")
            )
        }