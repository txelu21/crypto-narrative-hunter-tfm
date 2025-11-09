"""
Cohort Size Optimization and Ranking System

This module implements dynamic threshold adjustment, composite quality scoring,
and iterative filtering to achieve target cohort sizes with balanced representation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from collections import defaultdict
from enum import Enum
import json

from .wallet_filter import FilterResult, FilterCriteria, WalletFilter
from .performance_validator import ConsistencyResult
from .activity_analyzer import ActivityMetrics
from .sybil_detector import SybilDetectionResult

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Strategies for cohort optimization."""
    QUALITY_FIRST = "quality_first"
    SIZE_FIRST = "size_first"
    BALANCED = "balanced"
    PERCENTILE_BASED = "percentile_based"


@dataclass
class QualityWeights:
    """Weights for composite quality scoring."""
    performance_weight: float = 0.25
    consistency_weight: float = 0.20
    activity_weight: float = 0.15
    sophistication_weight: float = 0.15
    sybil_safety_weight: float = 0.10
    diversity_weight: float = 0.10
    manual_review_weight: float = 0.05


@dataclass
class CohortTargets:
    """Target parameters for cohort optimization."""
    min_size: int = 8000
    max_size: int = 12000
    target_size: int = 10000
    min_quality_score: float = 0.6
    performance_tier_distribution: Dict[str, float] = field(default_factory=lambda: {
        "top": 0.20,      # Top 20% performers
        "high": 0.30,     # High performers
        "medium": 0.35,   # Medium performers
        "emerging": 0.15  # Emerging performers
    })


@dataclass
class WalletScore:
    """Comprehensive scoring for a wallet."""
    wallet_address: str
    composite_score: float
    component_scores: Dict[str, float]
    performance_tier: str
    rank: Optional[int] = None
    percentile: Optional[float] = None
    selected: bool = False
    selection_reason: str = ""


@dataclass
class CohortResult:
    """Result of cohort optimization process."""
    selected_wallets: List[WalletScore]
    rejected_wallets: List[WalletScore]
    target_achieved: bool
    final_size: int
    quality_distribution: Dict[str, float]
    tier_distribution: Dict[str, int]
    optimization_iterations: int
    final_thresholds: Dict[str, float]
    selection_statistics: Dict[str, Any]


@dataclass
class OptimizationIteration:
    """Single iteration of the optimization process."""
    iteration: int
    threshold_adjustments: Dict[str, float]
    resulting_size: int
    avg_quality_score: float
    tier_distribution: Dict[str, int]
    convergence_metrics: Dict[str, float]


class CohortOptimizer:
    """
    Comprehensive cohort optimization system for achieving target wallet cohort sizes
    with balanced quality distribution and representation across performance tiers.
    """

    def __init__(self,
                 targets: Optional[CohortTargets] = None,
                 quality_weights: Optional[QualityWeights] = None,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                 max_iterations: int = 20,
                 convergence_tolerance: float = 0.05):
        """
        Initialize the cohort optimizer.

        Args:
            targets: Target cohort parameters
            quality_weights: Weights for composite scoring
            optimization_strategy: Strategy for optimization
            max_iterations: Maximum optimization iterations
            convergence_tolerance: Tolerance for convergence detection
        """
        self.targets = targets or CohortTargets()
        self.quality_weights = quality_weights or QualityWeights()
        self.strategy = optimization_strategy
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance

        # Optimization state
        self.current_thresholds: Dict[str, float] = {}
        self.optimization_history: List[OptimizationIteration] = []
        self.wallet_scores: Dict[str, WalletScore] = {}

        logger.info(f"CohortOptimizer initialized with strategy {optimization_strategy.value}, "
                   f"target size {self.targets.target_size}")

    def optimize_cohort(self,
                       filter_results: Dict[str, FilterResult],
                       consistency_results: Dict[str, ConsistencyResult],
                       activity_metrics: Dict[str, ActivityMetrics],
                       sybil_results: Dict[str, SybilDetectionResult],
                       manual_overrides: Optional[Dict[str, bool]] = None) -> CohortResult:
        """
        Optimize wallet cohort to achieve target size and quality distribution.

        Args:
            filter_results: Results from wallet filtering
            consistency_results: Performance consistency validation results
            activity_metrics: Activity pattern analysis results
            sybil_results: Sybil detection results
            manual_overrides: Manual inclusion/exclusion decisions

        Returns:
            CohortResult with optimized wallet selection
        """
        try:
            logger.info(f"Starting cohort optimization for {len(filter_results)} wallets")

            # Step 1: Calculate composite quality scores
            self._calculate_composite_scores(
                filter_results, consistency_results, activity_metrics,
                sybil_results, manual_overrides or {}
            )

            # Step 2: Assign performance tiers
            self._assign_performance_tiers()

            # Step 3: Iterative optimization
            optimization_result = self._perform_iterative_optimization()

            # Step 4: Final cohort selection
            final_cohort = self._select_final_cohort(optimization_result)

            # Step 5: Validate quality distribution
            validation_result = self._validate_quality_distribution(final_cohort)

            logger.info(f"Cohort optimization complete: {len(final_cohort.selected_wallets)} wallets selected")

            return final_cohort

        except Exception as e:
            logger.error(f"Error in cohort optimization: {e}")
            raise

    def adjust_thresholds_dynamically(self,
                                    current_size: int,
                                    target_size: int,
                                    current_quality: float,
                                    target_quality: float) -> Dict[str, float]:
        """
        Dynamically adjust filtering thresholds to achieve targets.

        Args:
            current_size: Current cohort size
            target_size: Target cohort size
            current_quality: Current average quality score
            target_quality: Target minimum quality score

        Returns:
            Dictionary of threshold adjustments
        """
        try:
            adjustments = {}

            # Size-based adjustments
            size_ratio = current_size / target_size

            if size_ratio > 1.1:  # Too many wallets - tighten thresholds
                adjustment_factor = 0.95  # 5% tighter
            elif size_ratio < 0.9:  # Too few wallets - loosen thresholds
                adjustment_factor = 1.05  # 5% looser
            else:
                adjustment_factor = 1.0  # No adjustment needed

            # Quality-based adjustments
            quality_ratio = current_quality / target_quality

            if quality_ratio < 0.95:  # Quality too low - tighten quality thresholds
                quality_adjustment = 0.98
            else:
                quality_adjustment = 1.0

            # Apply adjustments to key thresholds
            adjustments = {
                'min_sharpe_ratio': 0.5 * adjustment_factor * quality_adjustment,
                'min_win_rate': 0.55 * adjustment_factor * quality_adjustment,
                'min_total_return': 0.10 * adjustment_factor * quality_adjustment,
                'min_trading_days': int(30 / adjustment_factor),
                'min_trades': int(20 / adjustment_factor),
                'min_unique_tokens': int(5 / adjustment_factor)
            }

            # Ensure thresholds stay within reasonable bounds
            adjustments['min_sharpe_ratio'] = max(0.2, min(2.0, adjustments['min_sharpe_ratio']))
            adjustments['min_win_rate'] = max(0.45, min(0.75, adjustments['min_win_rate']))
            adjustments['min_total_return'] = max(0.05, min(0.30, adjustments['min_total_return']))
            adjustments['min_trading_days'] = max(15, min(90, adjustments['min_trading_days']))
            adjustments['min_trades'] = max(10, min(100, adjustments['min_trades']))
            adjustments['min_unique_tokens'] = max(3, min(15, adjustments['min_unique_tokens']))

            logger.info(f"Threshold adjustments calculated: size_ratio={size_ratio:.2f}, "
                       f"quality_ratio={quality_ratio:.2f}")

            return adjustments

        except Exception as e:
            logger.error(f"Error adjusting thresholds: {e}")
            return {}

    def rank_wallets_by_quality(self) -> List[WalletScore]:
        """
        Rank all wallets by composite quality score.

        Returns:
            List of WalletScore objects ranked by quality
        """
        try:
            # Sort wallets by composite score (descending)
            ranked_wallets = sorted(self.wallet_scores.values(),
                                  key=lambda w: w.composite_score, reverse=True)

            # Assign ranks and percentiles
            total_wallets = len(ranked_wallets)
            for i, wallet in enumerate(ranked_wallets):
                wallet.rank = i + 1
                wallet.percentile = (total_wallets - i) / total_wallets

            logger.info(f"Ranked {total_wallets} wallets by quality score")

            return ranked_wallets

        except Exception as e:
            logger.error(f"Error ranking wallets: {e}")
            return []

    def select_by_percentiles(self,
                            percentile_thresholds: Dict[str, float]) -> List[WalletScore]:
        """
        Select wallets based on percentile thresholds for different tiers.

        Args:
            percentile_thresholds: Percentile cutoffs for each tier

        Returns:
            List of selected WalletScore objects
        """
        try:
            ranked_wallets = self.rank_wallets_by_quality()
            selected_wallets = []

            for tier, threshold in percentile_thresholds.items():
                tier_wallets = [w for w in ranked_wallets
                              if w.performance_tier == tier and w.percentile >= threshold]

                # Calculate target count for this tier
                tier_target = int(self.targets.target_size *
                                self.targets.performance_tier_distribution.get(tier, 0))

                # Select top wallets from this tier
                tier_selected = tier_wallets[:tier_target]

                for wallet in tier_selected:
                    wallet.selected = True
                    wallet.selection_reason = f"percentile_tier_{tier}"

                selected_wallets.extend(tier_selected)

            logger.info(f"Selected {len(selected_wallets)} wallets using percentile filtering")

            return selected_wallets

        except Exception as e:
            logger.error(f"Error in percentile selection: {e}")
            return []

    def _calculate_composite_scores(self,
                                  filter_results: Dict[str, FilterResult],
                                  consistency_results: Dict[str, ConsistencyResult],
                                  activity_metrics: Dict[str, ActivityMetrics],
                                  sybil_results: Dict[str, SybilDetectionResult],
                                  manual_overrides: Dict[str, bool]) -> None:
        """Calculate composite quality scores for all wallets."""
        try:
            self.wallet_scores = {}

            all_wallets = set(filter_results.keys()) | set(consistency_results.keys()) | set(activity_metrics.keys())

            for wallet_address in all_wallets:
                # Get results for this wallet
                filter_result = filter_results.get(wallet_address)
                consistency_result = consistency_results.get(wallet_address)
                activity_metric = activity_metrics.get(wallet_address)
                sybil_result = sybil_results.get(wallet_address)
                manual_override = manual_overrides.get(wallet_address)

                # Calculate component scores
                component_scores = self._calculate_component_scores(
                    filter_result, consistency_result, activity_metric, sybil_result
                )

                # Calculate composite score
                composite_score = self._calculate_composite_score(component_scores, manual_override)

                # Create wallet score object
                wallet_score = WalletScore(
                    wallet_address=wallet_address,
                    composite_score=composite_score,
                    component_scores=component_scores,
                    performance_tier=""  # Will be assigned later
                )

                self.wallet_scores[wallet_address] = wallet_score

            logger.info(f"Calculated composite scores for {len(self.wallet_scores)} wallets")

        except Exception as e:
            logger.error(f"Error calculating composite scores: {e}")

    def _calculate_component_scores(self,
                                  filter_result: Optional[FilterResult],
                                  consistency_result: Optional[ConsistencyResult],
                                  activity_metric: Optional[ActivityMetrics],
                                  sybil_result: Optional[SybilDetectionResult]) -> Dict[str, float]:
        """Calculate individual component scores."""
        scores = {}

        # Performance score
        if filter_result:
            performance_indicators = [
                filter_result.scores.get('sharpe_ratio', 0) / 2.0,  # Normalize to ~0-1
                filter_result.scores.get('win_rate', 0),
                filter_result.scores.get('total_return', 0) * 2.0,  # Amplify for 0-1 range
            ]
            scores['performance'] = min(np.mean([max(0, min(1, x)) for x in performance_indicators]), 1.0)
        else:
            scores['performance'] = 0.0

        # Consistency score
        if consistency_result:
            scores['consistency'] = consistency_result.consistency_score
        else:
            scores['consistency'] = 0.0

        # Activity score
        if activity_metric:
            scores['activity'] = activity_metric.overall_activity_score
        else:
            scores['activity'] = 0.0

        # Sophistication score
        if activity_metric:
            scores['sophistication'] = activity_metric.sophistication_score
        else:
            scores['sophistication'] = 0.0

        # Sybil safety score (inverted sybil score)
        if sybil_result:
            scores['sybil_safety'] = 1.0 - sybil_result.sybil_score
        else:
            scores['sybil_safety'] = 1.0  # Assume safe if no sybil data

        # Diversity score
        if filter_result:
            scores['diversity'] = filter_result.scores.get('portfolio_diversity', 0)
        else:
            scores['diversity'] = 0.0

        return scores

    def _calculate_composite_score(self,
                                 component_scores: Dict[str, float],
                                 manual_override: Optional[bool]) -> float:
        """Calculate weighted composite score."""
        try:
            weights = self.quality_weights

            composite = (
                component_scores.get('performance', 0) * weights.performance_weight +
                component_scores.get('consistency', 0) * weights.consistency_weight +
                component_scores.get('activity', 0) * weights.activity_weight +
                component_scores.get('sophistication', 0) * weights.sophistication_weight +
                component_scores.get('sybil_safety', 0) * weights.sybil_safety_weight +
                component_scores.get('diversity', 0) * weights.diversity_weight
            )

            # Apply manual override bonus/penalty
            if manual_override is not None:
                manual_adjustment = 0.1 if manual_override else -0.1
                composite += manual_adjustment * weights.manual_review_weight

            return max(0.0, min(1.0, composite))

        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0.0

    def _assign_performance_tiers(self) -> None:
        """Assign performance tiers based on composite scores."""
        try:
            scores = [w.composite_score for w in self.wallet_scores.values()]

            if not scores:
                return

            # Calculate percentile thresholds
            p80 = np.percentile(scores, 80)
            p60 = np.percentile(scores, 60)
            p40 = np.percentile(scores, 40)

            for wallet in self.wallet_scores.values():
                score = wallet.composite_score

                if score >= p80:
                    wallet.performance_tier = "top"
                elif score >= p60:
                    wallet.performance_tier = "high"
                elif score >= p40:
                    wallet.performance_tier = "medium"
                else:
                    wallet.performance_tier = "emerging"

            logger.info(f"Assigned performance tiers: p80={p80:.3f}, p60={p60:.3f}, p40={p40:.3f}")

        except Exception as e:
            logger.error(f"Error assigning performance tiers: {e}")

    def _perform_iterative_optimization(self) -> OptimizationIteration:
        """Perform iterative optimization to achieve target cohort size."""
        try:
            # Initialize thresholds
            self.current_thresholds = {
                'min_composite_score': self.targets.min_quality_score,
                'tier_percentiles': {
                    'top': 0.9,
                    'high': 0.7,
                    'medium': 0.5,
                    'emerging': 0.3
                }
            }

            best_iteration = None
            target_achieved = False

            for iteration in range(self.max_iterations):
                # Apply current thresholds and calculate resulting cohort
                current_selection = self._apply_current_thresholds()
                current_size = len(current_selection)

                # Calculate metrics
                avg_quality = np.mean([w.composite_score for w in current_selection]) if current_selection else 0
                tier_distribution = self._calculate_tier_distribution(current_selection)

                # Create iteration record
                iteration_result = OptimizationIteration(
                    iteration=iteration,
                    threshold_adjustments=dict(self.current_thresholds),
                    resulting_size=current_size,
                    avg_quality_score=avg_quality,
                    tier_distribution=tier_distribution,
                    convergence_metrics=self._calculate_convergence_metrics(current_size, avg_quality)
                )

                self.optimization_history.append(iteration_result)

                # Check if target achieved
                size_in_range = self.targets.min_size <= current_size <= self.targets.max_size
                quality_met = avg_quality >= self.targets.min_quality_score

                if size_in_range and quality_met:
                    target_achieved = True
                    best_iteration = iteration_result
                    logger.info(f"Target achieved at iteration {iteration}: size={current_size}, quality={avg_quality:.3f}")
                    break

                # Adjust thresholds for next iteration
                if not target_achieved:
                    self._adjust_thresholds_for_next_iteration(
                        current_size, avg_quality, tier_distribution
                    )

                # Check convergence
                if self._check_convergence():
                    logger.info(f"Optimization converged at iteration {iteration}")
                    break

            # Return best iteration or last iteration
            return best_iteration or self.optimization_history[-1]

        except Exception as e:
            logger.error(f"Error in iterative optimization: {e}")
            raise

    def _apply_current_thresholds(self) -> List[WalletScore]:
        """Apply current thresholds to select wallets."""
        selected = []

        min_score = self.current_thresholds['min_composite_score']
        tier_percentiles = self.current_thresholds['tier_percentiles']

        # Rank wallets for percentile calculations
        ranked_wallets = self.rank_wallets_by_quality()

        for wallet in ranked_wallets:
            # Check minimum quality score
            if wallet.composite_score < min_score:
                continue

            # Check tier-specific percentile
            tier_threshold = tier_percentiles.get(wallet.performance_tier, 0.5)
            if wallet.percentile < tier_threshold:
                continue

            selected.append(wallet)

        return selected

    def _calculate_tier_distribution(self, selected_wallets: List[WalletScore]) -> Dict[str, int]:
        """Calculate distribution of wallets across performance tiers."""
        distribution = defaultdict(int)
        for wallet in selected_wallets:
            distribution[wallet.performance_tier] += 1
        return dict(distribution)

    def _calculate_convergence_metrics(self, current_size: int, avg_quality: float) -> Dict[str, float]:
        """Calculate metrics for convergence assessment."""
        return {
            'size_deviation': abs(current_size - self.targets.target_size) / self.targets.target_size,
            'quality_deviation': abs(avg_quality - self.targets.min_quality_score) / self.targets.min_quality_score
        }

    def _adjust_thresholds_for_next_iteration(self,
                                            current_size: int,
                                            avg_quality: float,
                                            tier_distribution: Dict[str, int]) -> None:
        """Adjust thresholds for the next optimization iteration."""
        # Size-based adjustments
        size_ratio = current_size / self.targets.target_size

        if size_ratio > 1.1:  # Too many - tighten
            self.current_thresholds['min_composite_score'] *= 1.02
            for tier in self.current_thresholds['tier_percentiles']:
                self.current_thresholds['tier_percentiles'][tier] *= 1.01
        elif size_ratio < 0.9:  # Too few - loosen
            self.current_thresholds['min_composite_score'] *= 0.98
            for tier in self.current_thresholds['tier_percentiles']:
                self.current_thresholds['tier_percentiles'][tier] *= 0.99

        # Quality-based adjustments
        if avg_quality < self.targets.min_quality_score:
            self.current_thresholds['min_composite_score'] *= 1.01

        # Ensure bounds
        self.current_thresholds['min_composite_score'] = max(0.3, min(0.9, self.current_thresholds['min_composite_score']))

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history) < 3:
            return False

        recent_iterations = self.optimization_history[-3:]
        size_variations = [abs(it.resulting_size - self.targets.target_size) for it in recent_iterations]

        return all(var / self.targets.target_size < self.convergence_tolerance for var in size_variations)

    def _select_final_cohort(self, optimization_result: OptimizationIteration) -> CohortResult:
        """Select final cohort based on optimization results."""
        try:
            # Apply final thresholds
            selected_wallets = self._apply_current_thresholds()

            # Mark selected wallets
            for wallet in selected_wallets:
                wallet.selected = True
                wallet.selection_reason = "optimization_threshold"

            # Get rejected wallets
            rejected_wallets = [w for w in self.wallet_scores.values() if not w.selected]

            # Calculate statistics
            quality_scores = [w.composite_score for w in selected_wallets]
            quality_distribution = {
                'mean': np.mean(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0,
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0,
                'q25': np.percentile(quality_scores, 25) if quality_scores else 0,
                'q50': np.percentile(quality_scores, 50) if quality_scores else 0,
                'q75': np.percentile(quality_scores, 75) if quality_scores else 0
            }

            tier_distribution = self._calculate_tier_distribution(selected_wallets)

            target_achieved = (
                self.targets.min_size <= len(selected_wallets) <= self.targets.max_size and
                quality_distribution['mean'] >= self.targets.min_quality_score
            )

            return CohortResult(
                selected_wallets=selected_wallets,
                rejected_wallets=rejected_wallets,
                target_achieved=target_achieved,
                final_size=len(selected_wallets),
                quality_distribution=quality_distribution,
                tier_distribution=tier_distribution,
                optimization_iterations=len(self.optimization_history),
                final_thresholds=dict(self.current_thresholds),
                selection_statistics={
                    'selection_rate': len(selected_wallets) / len(self.wallet_scores) if self.wallet_scores else 0,
                    'quality_improvement': quality_distribution['mean'] - self.targets.min_quality_score
                }
            )

        except Exception as e:
            logger.error(f"Error selecting final cohort: {e}")
            raise

    def _validate_quality_distribution(self, cohort_result: CohortResult) -> CohortResult:
        """Validate that the quality distribution meets requirements."""
        try:
            selected = cohort_result.selected_wallets

            # Check tier distribution targets
            actual_tier_counts = cohort_result.tier_distribution
            total_selected = len(selected)

            distribution_valid = True
            for tier, target_ratio in self.targets.performance_tier_distribution.items():
                actual_count = actual_tier_counts.get(tier, 0)
                actual_ratio = actual_count / total_selected if total_selected > 0 else 0

                # Allow 10% deviation from target
                if abs(actual_ratio - target_ratio) > 0.1:
                    distribution_valid = False
                    logger.warning(f"Tier {tier} distribution deviation: "
                                 f"target={target_ratio:.2f}, actual={actual_ratio:.2f}")

            # Update validation in selection statistics
            cohort_result.selection_statistics['distribution_valid'] = distribution_valid

            return cohort_result

        except Exception as e:
            logger.error(f"Error validating quality distribution: {e}")
            return cohort_result