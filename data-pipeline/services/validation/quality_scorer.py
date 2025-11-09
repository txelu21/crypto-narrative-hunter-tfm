"""Quality scoring and assessment framework.

This module provides composite quality scoring across all data dimensions.
"""

from typing import Any, Dict, List

import structlog

logger = structlog.get_logger()


class QualityScorer:
    """Quality scoring and assessment framework."""

    def __init__(self):
        """Initialize quality scorer."""
        self.logger = logger.bind(component="quality_scorer")

        # Define weights for each quality dimension
        self.quality_weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.20,
            "timeliness": 0.15,
            "validity": 0.15,
        }

    def calculate_composite_quality_score(
        self, validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate weighted composite quality score.

        Args:
            validation_results: Dict containing all validation results

        Returns:
            Composite quality score with breakdown
        """
        log = self.logger
        log.info("calculating_composite_quality_score")

        dimension_scores = {}

        # Completeness score
        dimension_scores["completeness"] = self._calculate_completeness_score(
            validation_results.get("completeness", {})
        )

        # Accuracy score
        dimension_scores["accuracy"] = self._calculate_accuracy_score(
            validation_results.get("accuracy", {})
        )

        # Consistency score
        dimension_scores["consistency"] = self._calculate_consistency_score(
            validation_results.get("consistency", {})
        )

        # Timeliness score
        dimension_scores["timeliness"] = self._calculate_timeliness_score(
            validation_results.get("timeliness", {})
        )

        # Validity score
        dimension_scores["validity"] = self._calculate_validity_score(
            validation_results.get("validity", {})
        )

        # Calculate weighted composite score
        composite_score = sum(
            score * self.quality_weights[dimension]
            for dimension, score in dimension_scores.items()
        )

        # Assign quality grade
        quality_grade = self._assign_quality_grade(composite_score)

        # Generate recommendations
        recommendations = self._generate_quality_recommendations(
            dimension_scores
        )

        log.info(
            "quality_score_calculated",
            composite_score=composite_score,
            quality_grade=quality_grade,
        )

        return {
            "composite_score": composite_score,
            "dimension_scores": dimension_scores,
            "quality_grade": quality_grade,
            "quality_weights": self.quality_weights,
            "recommendations": recommendations,
            "confidence_interval": self._calculate_confidence_interval(
                dimension_scores
            ),
        }

    def _calculate_completeness_score(
        self, completeness_data: Dict[str, Any]
    ) -> float:
        """Calculate completeness dimension score.

        Args:
            completeness_data: Completeness validation results

        Returns:
            Completeness score (0-1)
        """
        if not completeness_data:
            return 0.5  # Default middle score if no data

        # Extract completeness rate from various sources
        overall_completeness = completeness_data.get("overall_completeness", 0)
        coverage_percentage = completeness_data.get("coverage_percentage", 0)

        # Average the available metrics
        metrics = [overall_completeness, coverage_percentage]
        valid_metrics = [m for m in metrics if m > 0]

        if valid_metrics:
            return sum(valid_metrics) / len(valid_metrics)
        return 0.5

    def _calculate_accuracy_score(self, accuracy_data: Dict[str, Any]) -> float:
        """Calculate accuracy dimension score.

        Args:
            accuracy_data: Accuracy validation results

        Returns:
            Accuracy score (0-1)
        """
        if not accuracy_data:
            return 0.5

        # Extract accuracy metrics
        price_accuracy = accuracy_data.get("price_accuracy_rate", 0)
        validation_score = accuracy_data.get("validation_score", 0)

        metrics = [price_accuracy, validation_score]
        valid_metrics = [m for m in metrics if m > 0]

        if valid_metrics:
            return sum(valid_metrics) / len(valid_metrics)
        return 0.5

    def _calculate_consistency_score(
        self, consistency_data: Dict[str, Any]
    ) -> float:
        """Calculate consistency dimension score.

        Args:
            consistency_data: Consistency validation results

        Returns:
            Consistency score (0-1)
        """
        if not consistency_data:
            return 0.5

        # Extract consistency metrics
        cross_validation_score = consistency_data.get(
            "overall_consistency_score", 0
        )
        integrity_score = consistency_data.get("overall_integrity_score", 0)

        metrics = [cross_validation_score, integrity_score]
        valid_metrics = [m for m in metrics if m > 0]

        if valid_metrics:
            return sum(valid_metrics) / len(valid_metrics)
        return 0.5

    def _calculate_timeliness_score(
        self, timeliness_data: Dict[str, Any]
    ) -> float:
        """Calculate timeliness dimension score.

        Args:
            timeliness_data: Timeliness validation results

        Returns:
            Timeliness score (0-1)
        """
        if not timeliness_data:
            return 0.5

        # Check data freshness
        is_fresh = timeliness_data.get("is_fresh", False)
        staleness_hours = timeliness_data.get("staleness_hours", 24)
        max_staleness = timeliness_data.get("max_staleness_hours", 24)

        if is_fresh:
            return 1.0

        # Score decreases as data gets staler
        if staleness_hours <= max_staleness:
            return 1.0 - (staleness_hours / (max_staleness * 2))
        else:
            # Significant penalty for very stale data
            return max(0, 0.5 - ((staleness_hours - max_staleness) / 100))

    def _calculate_validity_score(self, validity_data: Dict[str, Any]) -> float:
        """Calculate validity dimension score.

        Args:
            validity_data: Validity validation results

        Returns:
            Validity score (0-1)
        """
        if not validity_data:
            return 0.5

        # Extract validity metrics
        constraint_score = validity_data.get("constraint_compliance_score", 0)
        schema_compliance = validity_data.get("schema_compliance", 0)

        metrics = [constraint_score, schema_compliance]
        valid_metrics = [m for m in metrics if m > 0]

        if valid_metrics:
            return sum(valid_metrics) / len(valid_metrics)
        return 0.5

    def _assign_quality_grade(self, score: float) -> str:
        """Assign quality grade based on composite score.

        Args:
            score: Composite quality score (0-1)

        Returns:
            Quality grade (A+ to F)
        """
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "B+"
        elif score >= 0.80:
            return "B"
        elif score >= 0.70:
            return "C"
        elif score >= 0.60:
            return "D"
        else:
            return "F"

    def _generate_quality_recommendations(
        self, dimension_scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on quality scores.

        Args:
            dimension_scores: Scores for each quality dimension

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        for dimension, score in dimension_scores.items():
            if score < 0.80:
                if dimension == "completeness":
                    recommendations.append(
                        f"Improve data collection to address {dimension} issues "
                        f"(current score: {score:.2f}). Consider reviewing collection "
                        f"processes and identifying missing data sources."
                    )
                elif dimension == "accuracy":
                    recommendations.append(
                        f"Implement additional validation checks for {dimension} "
                        f"(current score: {score:.2f}). Review price validation "
                        f"and cross-validation processes."
                    )
                elif dimension == "consistency":
                    recommendations.append(
                        f"Review data integration processes to improve {dimension} "
                        f"(current score: {score:.2f}). Check for discrepancies "
                        f"between transactions and balance changes."
                    )
                elif dimension == "timeliness":
                    recommendations.append(
                        f"Update data collection schedule to improve {dimension} "
                        f"(current score: {score:.2f}). Data may be stale."
                    )
                elif dimension == "validity":
                    recommendations.append(
                        f"Review and enforce data constraints for {dimension} "
                        f"(current score: {score:.2f}). Check for constraint violations."
                    )

        if not recommendations:
            recommendations.append(
                "All quality dimensions meet acceptable thresholds. "
                "Continue monitoring for quality deterioration."
            )

        return recommendations

    def _calculate_confidence_interval(
        self, dimension_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate confidence interval for quality estimate.

        Args:
            dimension_scores: Scores for each quality dimension

        Returns:
            Dict with lower_bound and upper_bound
        """
        scores = list(dimension_scores.values())
        avg_score = sum(scores) / len(scores) if scores else 0

        # Calculate variance
        variance = (
            sum((s - avg_score) ** 2 for s in scores) / len(scores)
            if scores
            else 0
        )
        std_dev = variance**0.5

        # 95% confidence interval (approx 2 standard deviations)
        margin = 2 * std_dev

        return {
            "lower_bound": max(0, avg_score - margin),
            "upper_bound": min(1, avg_score + margin),
            "confidence_level": 0.95,
        }

    def calculate_quality_trend(
        self, historical_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate quality trend over time.

        Args:
            historical_scores: List of historical quality scores with timestamps

        Returns:
            Quality trend analysis
        """
        log = self.logger
        log.info("calculating_quality_trend", data_points=len(historical_scores))

        if len(historical_scores) < 2:
            return {
                "trend": "insufficient_data",
                "data_points": len(historical_scores),
            }

        # Extract scores and timestamps
        scores = [s["composite_score"] for s in historical_scores]
        timestamps = [s["timestamp"] for s in historical_scores]

        # Calculate trend direction
        recent_avg = sum(scores[-3:]) / len(scores[-3:])
        historical_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else recent_avg

        trend_direction = "stable"
        if recent_avg > historical_avg + 0.05:
            trend_direction = "improving"
        elif recent_avg < historical_avg - 0.05:
            trend_direction = "deteriorating"

        # Calculate rate of change
        if len(scores) >= 2:
            rate_of_change = (scores[-1] - scores[0]) / len(scores)
        else:
            rate_of_change = 0

        log.info(
            "quality_trend_calculated",
            trend_direction=trend_direction,
            rate_of_change=rate_of_change,
        )

        return {
            "trend": trend_direction,
            "data_points": len(scores),
            "recent_avg": recent_avg,
            "historical_avg": historical_avg,
            "rate_of_change": rate_of_change,
            "current_score": scores[-1] if scores else 0,
            "peak_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
        }