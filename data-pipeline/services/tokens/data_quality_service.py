"""
Data quality reporting and metrics service.

Implements:
- Categorization distribution report by narrative
- Completeness metrics for all token fields
- Validation summary with pass/fail counts
- Liquidity tier distribution analysis
- Manual review statistics and coverage metrics
- Data quality dashboard for ongoing monitoring
"""

import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from data_collection.common.db import execute_with_retry
from data_collection.common.logging_setup import get_logger


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics"""
    # Basic counts
    total_tokens: int = 0
    classified_tokens: int = 0
    validated_tokens: int = 0

    # Completeness rates
    completeness_rates: Dict[str, float] = None

    # Category distribution
    category_distribution: Dict[str, int] = None
    category_percentages: Dict[str, float] = None

    # Validation statistics
    validation_summary: Dict[str, int] = None

    # Quality scores
    overall_quality_score: float = 0.0
    classification_quality_score: float = 0.0
    validation_quality_score: float = 0.0
    completeness_quality_score: float = 0.0

    # Timestamps
    report_timestamp: str = ""
    last_update_timestamp: str = ""

    def __post_init__(self):
        if self.completeness_rates is None:
            self.completeness_rates = {}
        if self.category_distribution is None:
            self.category_distribution = {}
        if self.category_percentages is None:
            self.category_percentages = {}
        if self.validation_summary is None:
            self.validation_summary = {}
        if not self.report_timestamp:
            self.report_timestamp = datetime.now().isoformat()


@dataclass
class LiquidityAnalysisResult:
    """Liquidity tier distribution analysis"""
    tier_distribution: Dict[str, int] = None
    tier_percentages: Dict[str, float] = None
    tier_volume_stats: Dict[str, Dict[str, float]] = None
    unassigned_liquidity_count: int = 0

    def __post_init__(self):
        if self.tier_distribution is None:
            self.tier_distribution = {}
        if self.tier_percentages is None:
            self.tier_percentages = {}
        if self.tier_volume_stats is None:
            self.tier_volume_stats = {}


@dataclass
class ManualReviewMetrics:
    """Manual review progress and accuracy metrics"""
    total_pending_review: int = 0
    total_manually_reviewed: int = 0
    review_completion_rate: float = 0.0

    # Reviewer activity
    reviewer_stats: Dict[str, int] = None

    # Review accuracy (based on confidence changes)
    avg_confidence_before_review: float = 0.0
    avg_confidence_after_review: float = 0.0
    confidence_improvement: float = 0.0

    # Recent activity
    recent_reviews: List[Dict[str, Any]] = None
    daily_review_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.reviewer_stats is None:
            self.reviewer_stats = {}
        if self.recent_reviews is None:
            self.recent_reviews = []
        if self.daily_review_counts is None:
            self.daily_review_counts = {}


class DataQualityService:
    """
    Service for generating comprehensive data quality reports and metrics.

    Provides insights into:
    - Data completeness and validation status
    - Narrative classification distribution and accuracy
    - Manual review progress and effectiveness
    - Liquidity tier analysis
    - Overall data quality scoring
    """

    def __init__(self, reports_dir: str = "data/quality_reports"):
        self.logger = get_logger("data_quality_service")
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Quality score weights
        self.quality_weights = {
            'completeness': 0.4,
            'classification': 0.3,
            'validation': 0.3
        }

        # Target thresholds for quality scoring
        self.quality_targets = {
            'completeness_threshold': 95.0,
            'classification_threshold': 90.0,
            'validation_threshold': 95.0,
            'manual_review_coverage': 10.0  # 10% manual review coverage target
        }

    def generate_comprehensive_report(self) -> DataQualityMetrics:
        """
        Generate a comprehensive data quality report.

        Returns:
            DataQualityMetrics with complete quality assessment
        """
        start_time = time.time()

        self.logger.log_operation(
            operation="generate_comprehensive_report",
            status="started",
            message="Starting comprehensive data quality report generation"
        )

        try:
            metrics = DataQualityMetrics()

            # 1. Basic token counts
            self._collect_basic_metrics(metrics)

            # 2. Data completeness analysis
            self._analyze_data_completeness(metrics)

            # 3. Classification analysis
            self._analyze_classification_quality(metrics)

            # 4. Validation analysis
            self._analyze_validation_quality(metrics)

            # 5. Calculate overall quality scores
            self._calculate_quality_scores(metrics)

            # 6. Set timestamps
            metrics.report_timestamp = datetime.now().isoformat()
            metrics.last_update_timestamp = self._get_last_update_timestamp()

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.log_operation(
                operation="generate_comprehensive_report",
                params={
                    "total_tokens": metrics.total_tokens,
                    "overall_quality": metrics.overall_quality_score,
                    "classified_tokens": metrics.classified_tokens
                },
                status="completed",
                duration_ms=duration_ms,
                message=f"Data quality report generated: {metrics.overall_quality_score:.1f}% overall quality"
            )

            return metrics

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="generate_comprehensive_report",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def analyze_liquidity_distribution(self) -> LiquidityAnalysisResult:
        """
        Analyze liquidity tier distribution and volume statistics.

        Returns:
            LiquidityAnalysisResult with tier analysis
        """
        try:
            result = LiquidityAnalysisResult()

            # Get tier distribution
            tier_query = """
                SELECT
                    COALESCE(liquidity_tier::text, 'Unassigned') as tier,
                    COUNT(*) as count,
                    AVG(avg_daily_volume_usd) as avg_volume,
                    MIN(avg_daily_volume_usd) as min_volume,
                    MAX(avg_daily_volume_usd) as max_volume
                FROM tokens
                GROUP BY liquidity_tier
                ORDER BY liquidity_tier NULLS LAST
            """

            tier_result = execute_with_retry(tier_query)

            if tier_result:
                total_tokens = sum(row['count'] for row in tier_result)

                for row in tier_result:
                    tier = row['tier']
                    count = row['count']

                    result.tier_distribution[tier] = count
                    result.tier_percentages[tier] = (count / total_tokens) * 100 if total_tokens > 0 else 0

                    if tier == 'Unassigned':
                        result.unassigned_liquidity_count = count
                    else:
                        result.tier_volume_stats[tier] = {
                            'avg_volume': float(row['avg_volume']) if row['avg_volume'] else 0,
                            'min_volume': float(row['min_volume']) if row['min_volume'] else 0,
                            'max_volume': float(row['max_volume']) if row['max_volume'] else 0
                        }

            self.logger.log_operation(
                operation="analyze_liquidity_distribution",
                params={"tier_count": len(result.tier_distribution)},
                status="completed",
                message="Liquidity distribution analysis completed"
            )

            return result

        except Exception as e:
            self.logger.log_operation(
                operation="analyze_liquidity_distribution",
                status="error",
                error=str(e)
            )
            raise

    def analyze_manual_review_progress(self) -> ManualReviewMetrics:
        """
        Analyze manual review progress and effectiveness.

        Returns:
            ManualReviewMetrics with review analysis
        """
        try:
            metrics = ManualReviewMetrics()

            # Basic review counts
            review_counts_query = """
                SELECT
                    COUNT(CASE WHEN manual_review_status = 'pending' THEN 1 END) as pending_review,
                    COUNT(CASE WHEN manual_review_status LIKE 'manual%' OR manual_review_status LIKE 'reviewed%' THEN 1 END) as manually_reviewed,
                    COUNT(*) as total_tokens
                FROM tokens
            """

            counts_result = execute_with_retry(review_counts_query)
            if counts_result:
                counts = counts_result[0]
                metrics.total_pending_review = counts['pending_review']
                metrics.total_manually_reviewed = counts['manually_reviewed']
                total_tokens = counts['total_tokens']
                metrics.review_completion_rate = (metrics.total_manually_reviewed / total_tokens) * 100 if total_tokens > 0 else 0

            # Reviewer activity
            reviewer_query = """
                SELECT reviewer, COUNT(*) as review_count
                FROM tokens
                WHERE reviewer IS NOT NULL
                GROUP BY reviewer
                ORDER BY review_count DESC
            """

            reviewer_result = execute_with_retry(reviewer_query)
            if reviewer_result:
                metrics.reviewer_stats = {row['reviewer']: row['review_count'] for row in reviewer_result}

            # Review effectiveness (confidence analysis)
            confidence_query = """
                SELECT
                    AVG(CASE WHEN manual_review_status = 'auto_classified' THEN classification_confidence END) as avg_auto_confidence,
                    AVG(CASE WHEN manual_review_status LIKE 'manual%' OR manual_review_status LIKE 'reviewed%' THEN classification_confidence END) as avg_manual_confidence
                FROM tokens
                WHERE classification_confidence IS NOT NULL
            """

            confidence_result = execute_with_retry(confidence_query)
            if confidence_result:
                conf = confidence_result[0]
                metrics.avg_confidence_before_review = float(conf['avg_auto_confidence']) if conf['avg_auto_confidence'] else 0
                metrics.avg_confidence_after_review = float(conf['avg_manual_confidence']) if conf['avg_manual_confidence'] else 0
                metrics.confidence_improvement = metrics.avg_confidence_after_review - metrics.avg_confidence_before_review

            # Recent review activity
            recent_activity_query = """
                SELECT
                    DATE(created_at) as review_date,
                    COUNT(*) as reviews_count
                FROM classification_audit
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY review_date DESC
            """

            recent_result = execute_with_retry(recent_activity_query)
            if recent_result:
                metrics.daily_review_counts = {
                    str(row['review_date']): row['reviews_count']
                    for row in recent_result
                }

            self.logger.log_operation(
                operation="analyze_manual_review_progress",
                params={
                    "pending_reviews": metrics.total_pending_review,
                    "completion_rate": metrics.review_completion_rate
                },
                status="completed",
                message="Manual review analysis completed"
            )

            return metrics

        except Exception as e:
            self.logger.log_operation(
                operation="analyze_manual_review_progress",
                status="error",
                error=str(e)
            )
            raise

    def _collect_basic_metrics(self, metrics: DataQualityMetrics) -> None:
        """Collect basic token count metrics"""
        basic_query = """
            SELECT
                COUNT(*) as total_tokens,
                COUNT(narrative_category) as classified_tokens,
                COUNT(CASE WHEN validation_status = 'valid' THEN 1 END) as validated_tokens
            FROM tokens
        """

        result = execute_with_retry(basic_query)
        if result:
            data = result[0]
            metrics.total_tokens = data['total_tokens']
            metrics.classified_tokens = data['classified_tokens']
            metrics.validated_tokens = data['validated_tokens']

    def _analyze_data_completeness(self, metrics: DataQualityMetrics) -> None:
        """Analyze data completeness for all fields"""
        completeness_query = """
            SELECT
                COUNT(*) as total_tokens,
                COUNT(symbol) as has_symbol,
                COUNT(name) as has_name,
                COUNT(decimals) as has_decimals,
                COUNT(market_cap_rank) as has_market_cap_rank,
                COUNT(avg_daily_volume_usd) as has_volume,
                COUNT(narrative_category) as has_narrative_category,
                COUNT(liquidity_tier) as has_liquidity_tier
            FROM tokens
        """

        result = execute_with_retry(completeness_query)
        if result:
            data = result[0]
            total = data['total_tokens']

            if total > 0:
                metrics.completeness_rates = {
                    'symbol': (data['has_symbol'] / total) * 100,
                    'name': (data['has_name'] / total) * 100,
                    'decimals': (data['has_decimals'] / total) * 100,
                    'market_cap_rank': (data['has_market_cap_rank'] / total) * 100,
                    'avg_daily_volume_usd': (data['has_volume'] / total) * 100,
                    'narrative_category': (data['has_narrative_category'] / total) * 100,
                    'liquidity_tier': (data['has_liquidity_tier'] / total) * 100
                }

    def _analyze_classification_quality(self, metrics: DataQualityMetrics) -> None:
        """Analyze narrative classification quality"""
        # Category distribution
        category_query = """
            SELECT
                narrative_category,
                COUNT(*) as count
            FROM tokens
            WHERE narrative_category IS NOT NULL
            GROUP BY narrative_category
            ORDER BY count DESC
        """

        result = execute_with_retry(category_query)
        if result:
            total_classified = sum(row['count'] for row in result)

            for row in result:
                category = row['narrative_category']
                count = row['count']

                metrics.category_distribution[category] = count
                metrics.category_percentages[category] = (count / total_classified) * 100 if total_classified > 0 else 0

    def _analyze_validation_quality(self, metrics: DataQualityMetrics) -> None:
        """Analyze validation quality"""
        validation_query = """
            SELECT
                COUNT(*) as total_tokens,
                COUNT(CASE WHEN validation_status = 'valid' THEN 1 END) as valid_tokens,
                COUNT(CASE WHEN validation_status = 'invalid' THEN 1 END) as invalid_tokens,
                COUNT(CASE WHEN validation_last_run IS NOT NULL THEN 1 END) as validated_tokens,
                COUNT(CASE WHEN validation_flags IS NOT NULL AND validation_flags != '' THEN 1 END) as tokens_with_flags
            FROM tokens
        """

        result = execute_with_retry(validation_query)
        if result:
            metrics.validation_summary = dict(result[0])

    def _calculate_quality_scores(self, metrics: DataQualityMetrics) -> None:
        """Calculate overall quality scores"""
        # Completeness quality score (average of key field completeness)
        key_fields = ['symbol', 'name', 'decimals', 'narrative_category']
        if metrics.completeness_rates:
            completeness_scores = [metrics.completeness_rates.get(field, 0) for field in key_fields]
            metrics.completeness_quality_score = sum(completeness_scores) / len(completeness_scores)

        # Classification quality score
        if metrics.total_tokens > 0:
            classification_rate = (metrics.classified_tokens / metrics.total_tokens) * 100
            metrics.classification_quality_score = min(classification_rate, 100.0)

        # Validation quality score
        if metrics.validation_summary and metrics.total_tokens > 0:
            valid_tokens = metrics.validation_summary.get('valid_tokens', 0)
            validation_rate = (valid_tokens / metrics.total_tokens) * 100
            metrics.validation_quality_score = min(validation_rate, 100.0)

        # Overall quality score (weighted average)
        metrics.overall_quality_score = (
            metrics.completeness_quality_score * self.quality_weights['completeness'] +
            metrics.classification_quality_score * self.quality_weights['classification'] +
            metrics.validation_quality_score * self.quality_weights['validation']
        )

    def _get_last_update_timestamp(self) -> str:
        """Get timestamp of last token update"""
        query = "SELECT MAX(updated_at) as last_update FROM tokens"
        result = execute_with_retry(query)

        if result and result[0]['last_update']:
            return result[0]['last_update'].isoformat()

        return datetime.now().isoformat()

    def save_report_to_file(self, metrics: DataQualityMetrics, filename: Optional[str] = None) -> str:
        """
        Save data quality report to JSON file.

        Args:
            metrics: DataQualityMetrics to save
            filename: Optional filename, defaults to timestamped filename

        Returns:
            Path to saved file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_quality_report_{timestamp}.json"

            file_path = self.reports_dir / filename

            # Convert metrics to dictionary and save
            report_data = {
                "metadata": {
                    "report_type": "data_quality_comprehensive",
                    "generated_at": metrics.report_timestamp,
                    "last_data_update": metrics.last_update_timestamp
                },
                "metrics": asdict(metrics)
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)

            self.logger.log_operation(
                operation="save_report_to_file",
                params={"filename": filename, "file_size": file_path.stat().st_size},
                status="completed",
                message=f"Data quality report saved to {filename}"
            )

            return str(file_path)

        except Exception as e:
            self.logger.log_operation(
                operation="save_report_to_file",
                status="error",
                error=str(e)
            )
            raise

    def generate_daily_summary(self) -> Dict[str, Any]:
        """
        Generate a concise daily summary report.

        Returns:
            Dictionary with key daily metrics
        """
        try:
            # Basic counts
            basic_query = """
                SELECT
                    COUNT(*) as total_tokens,
                    COUNT(narrative_category) as classified_tokens,
                    COUNT(CASE WHEN validation_status = 'valid' THEN 1 END) as valid_tokens,
                    COUNT(CASE WHEN manual_review_status = 'pending' THEN 1 END) as pending_reviews
                FROM tokens
            """

            basic_result = execute_with_retry(basic_query)
            basic_stats = basic_result[0] if basic_result else {}

            # Recent activity (last 24 hours)
            activity_query = """
                SELECT
                    COUNT(*) as recent_updates
                FROM tokens
                WHERE updated_at >= NOW() - INTERVAL '24 hours'
            """

            activity_result = execute_with_retry(activity_query)
            recent_updates = activity_result[0]['recent_updates'] if activity_result else 0

            # Classification distribution (top 5)
            top_categories_query = """
                SELECT narrative_category, COUNT(*) as count
                FROM tokens
                WHERE narrative_category IS NOT NULL
                GROUP BY narrative_category
                ORDER BY count DESC
                LIMIT 5
            """

            categories_result = execute_with_retry(top_categories_query)
            top_categories = {row['narrative_category']: row['count'] for row in categories_result or []}

            summary = {
                "summary_date": datetime.now().strftime("%Y-%m-%d"),
                "basic_statistics": basic_stats,
                "recent_activity": {
                    "tokens_updated_24h": recent_updates
                },
                "top_categories": top_categories,
                "data_quality_indicators": {
                    "classification_rate": (basic_stats.get('classified_tokens', 0) / basic_stats.get('total_tokens', 1)) * 100,
                    "validation_rate": (basic_stats.get('valid_tokens', 0) / basic_stats.get('total_tokens', 1)) * 100,
                    "pending_manual_reviews": basic_stats.get('pending_reviews', 0)
                }
            }

            return summary

        except Exception as e:
            self.logger.log_operation(
                operation="generate_daily_summary",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}

    def get_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Get data quality trends over specified number of days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        try:
            # Daily token counts
            daily_counts_query = """
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as tokens_added
                FROM tokens
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """

            daily_result = execute_with_retry(daily_counts_query, (days,))
            daily_counts = {str(row['date']): row['tokens_added'] for row in daily_result or []}

            # Daily classification activity
            classification_activity_query = """
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as classifications
                FROM classification_audit
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """

            classification_result = execute_with_retry(classification_activity_query, (days,))
            classification_activity = {str(row['date']): row['classifications'] for row in classification_result or []}

            return {
                "trend_period_days": days,
                "daily_token_additions": daily_counts,
                "daily_classification_activity": classification_activity
            }

        except Exception as e:
            self.logger.log_operation(
                operation="get_quality_trends",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}