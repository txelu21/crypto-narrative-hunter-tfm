"""
Price Quality Assurance and Benchmark Validation
Validates price accuracy against external benchmarks and ensures data quality
"""

import logging
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class QualityGrade(Enum):
    """Quality grade levels"""
    A = "A"  # Excellent (95-100%)
    B = "B"  # Good (90-95%)
    C = "C"  # Acceptable (80-90%)
    D = "D"  # Poor (70-80%)
    F = "F"  # Fail (<70%)


@dataclass
class QualityMetric:
    """Quality metric data structure"""
    name: str
    score: float  # 0-1
    weight: float  # Contribution to overall score
    status: str  # 'pass' or 'fail'
    details: Dict


@dataclass
class BenchmarkComparison:
    """Benchmark comparison result"""
    timestamp: int
    our_price: float
    benchmark_price: float
    deviation: float
    deviation_pct: float
    within_tolerance: bool


class PriceQualityAssurance:
    """
    Comprehensive quality assurance for price data

    Features:
    - Price accuracy validation against benchmarks
    - Statistical consistency checks
    - Coverage analysis
    - Anomaly rate monitoring
    - Quality scoring and grading
    - Automated testing framework
    """

    # Quality targets
    TARGET_COVERAGE = 0.99  # 99% coverage
    TARGET_ACCURACY = 0.98  # 98% within tolerance
    TARGET_ANOMALY_RATE = 0.01  # <1% anomalies
    TARGET_CONSISTENCY = 0.95  # 95% consistency

    def __init__(
        self,
        price_data: Dict[int, Dict],
        accuracy_tolerance: float = 0.02  # 2% tolerance
    ):
        """
        Initialize QA system

        Args:
            price_data: Dict of {timestamp -> price_data}
            accuracy_tolerance: Acceptable price deviation (e.g., 0.02 = 2%)
        """
        self.price_data = price_data
        self.accuracy_tolerance = accuracy_tolerance
        logger.info(
            f"QA system initialized with {len(price_data)} prices, "
            f"tolerance={accuracy_tolerance*100}%"
        )

    def run_full_qa(
        self,
        benchmark_data: Optional[Dict[int, float]] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> Dict:
        """
        Run comprehensive QA validation

        Args:
            benchmark_data: Optional benchmark prices for comparison
            start_timestamp: Optional start timestamp
            end_timestamp: Optional end timestamp

        Returns:
            Complete QA report
        """
        logger.info("Starting full QA validation")

        metrics = []

        # 1. Coverage Analysis
        coverage_metric = self._check_coverage(start_timestamp, end_timestamp)
        metrics.append(coverage_metric)

        # 2. Accuracy Validation
        if benchmark_data:
            accuracy_metric = self._check_accuracy(benchmark_data)
            metrics.append(accuracy_metric)

        # 3. Consistency Checks
        consistency_metric = self._check_consistency()
        metrics.append(consistency_metric)

        # 4. Anomaly Rate
        anomaly_metric = self._check_anomaly_rate()
        metrics.append(anomaly_metric)

        # 5. Data Integrity
        integrity_metric = self._check_data_integrity()
        metrics.append(integrity_metric)

        # Calculate overall quality score
        overall_score = sum(m.score * m.weight for m in metrics)
        quality_grade = self._assign_grade(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        report = {
            'timestamp': int(datetime.now().timestamp()),
            'overall_score': overall_score,
            'quality_grade': quality_grade.value,
            'metrics': [
                {
                    'name': m.name,
                    'score': m.score,
                    'weight': m.weight,
                    'status': m.status,
                    'details': m.details
                }
                for m in metrics
            ],
            'recommendations': recommendations,
            'passed': all(m.status == 'pass' for m in metrics),
            'total_data_points': len(self.price_data)
        }

        logger.info(
            f"QA complete: Grade {quality_grade.value}, Score {overall_score:.2%}, "
            f"Status {'PASS' if report['passed'] else 'FAIL'}"
        )

        return report

    def _check_coverage(
        self,
        start_timestamp: Optional[int],
        end_timestamp: Optional[int]
    ) -> QualityMetric:
        """
        Check data coverage

        Returns:
            Coverage quality metric
        """
        if not start_timestamp or not end_timestamp:
            # Use data boundaries
            timestamps = sorted(self.price_data.keys())
            start_timestamp = timestamps[0]
            end_timestamp = timestamps[-1]

        # Calculate expected vs actual data points (hourly)
        expected_hours = (end_timestamp - start_timestamp) // 3600
        actual_hours = len(self.price_data)

        coverage_ratio = actual_hours / expected_hours if expected_hours > 0 else 0
        gap_count = expected_hours - actual_hours

        # Score based on coverage ratio
        score = min(1.0, coverage_ratio / self.TARGET_COVERAGE)
        status = 'pass' if coverage_ratio >= self.TARGET_COVERAGE else 'fail'

        return QualityMetric(
            name='Coverage',
            score=score,
            weight=0.25,
            status=status,
            details={
                'expected_data_points': expected_hours,
                'actual_data_points': actual_hours,
                'gap_count': gap_count,
                'coverage_percentage': coverage_ratio * 100,
                'target_coverage': self.TARGET_COVERAGE * 100
            }
        )

    def _check_accuracy(
        self,
        benchmark_data: Dict[int, float]
    ) -> QualityMetric:
        """
        Check price accuracy against benchmarks

        Args:
            benchmark_data: Dict of {timestamp -> benchmark_price}

        Returns:
            Accuracy quality metric
        """
        comparisons = []

        for timestamp, price_info in self.price_data.items():
            if timestamp in benchmark_data:
                our_price = price_info['price_usd']
                benchmark_price = benchmark_data[timestamp]

                deviation = abs(our_price - benchmark_price)
                deviation_pct = deviation / benchmark_price if benchmark_price > 0 else 0

                within_tolerance = deviation_pct <= self.accuracy_tolerance

                comparisons.append(BenchmarkComparison(
                    timestamp=timestamp,
                    our_price=our_price,
                    benchmark_price=benchmark_price,
                    deviation=deviation,
                    deviation_pct=deviation_pct,
                    within_tolerance=within_tolerance
                ))

        if not comparisons:
            return QualityMetric(
                name='Accuracy',
                score=0.0,
                weight=0.30,
                status='fail',
                details={'error': 'No benchmark data for comparison'}
            )

        # Calculate accuracy metrics
        within_tolerance_count = sum(1 for c in comparisons if c.within_tolerance)
        accuracy_ratio = within_tolerance_count / len(comparisons)

        avg_deviation = statistics.mean(c.deviation_pct for c in comparisons)
        max_deviation = max(c.deviation_pct for c in comparisons)

        score = min(1.0, accuracy_ratio / self.TARGET_ACCURACY)
        status = 'pass' if accuracy_ratio >= self.TARGET_ACCURACY else 'fail'

        return QualityMetric(
            name='Accuracy',
            score=score,
            weight=0.30,
            status=status,
            details={
                'comparisons_count': len(comparisons),
                'within_tolerance': within_tolerance_count,
                'accuracy_percentage': accuracy_ratio * 100,
                'avg_deviation_pct': avg_deviation * 100,
                'max_deviation_pct': max_deviation * 100,
                'tolerance_threshold': self.accuracy_tolerance * 100,
                'target_accuracy': self.TARGET_ACCURACY * 100
            }
        )

    def _check_consistency(self) -> QualityMetric:
        """
        Check price consistency (no extreme fluctuations)

        Returns:
            Consistency quality metric
        """
        sorted_data = sorted(self.price_data.items())

        if len(sorted_data) < 2:
            return QualityMetric(
                name='Consistency',
                score=0.0,
                weight=0.20,
                status='fail',
                details={'error': 'Insufficient data for consistency check'}
            )

        # Check hourly changes
        extreme_changes = 0
        hourly_changes = []

        for i in range(1, len(sorted_data)):
            prev_price = sorted_data[i-1][1]['price_usd']
            curr_price = sorted_data[i][1]['price_usd']

            if prev_price > 0:
                change_pct = abs(curr_price - prev_price) / prev_price
                hourly_changes.append(change_pct)

                # Flag extreme changes (>15% per hour)
                if change_pct > 0.15:
                    extreme_changes += 1

        consistency_ratio = 1.0 - (extreme_changes / len(hourly_changes))
        avg_change = statistics.mean(hourly_changes)
        max_change = max(hourly_changes)

        score = min(1.0, consistency_ratio / self.TARGET_CONSISTENCY)
        status = 'pass' if consistency_ratio >= self.TARGET_CONSISTENCY else 'fail'

        return QualityMetric(
            name='Consistency',
            score=score,
            weight=0.20,
            status=status,
            details={
                'extreme_changes': extreme_changes,
                'total_changes': len(hourly_changes),
                'consistency_percentage': consistency_ratio * 100,
                'avg_hourly_change_pct': avg_change * 100,
                'max_hourly_change_pct': max_change * 100,
                'target_consistency': self.TARGET_CONSISTENCY * 100
            }
        )

    def _check_anomaly_rate(self) -> QualityMetric:
        """
        Check anomaly rate in data

        Returns:
            Anomaly rate quality metric
        """
        # Count prices marked as corrected (indicates anomalies)
        anomaly_count = sum(
            1 for p in self.price_data.values()
            if p.get('corrected', False)
        )

        total_count = len(self.price_data)
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0

        # Score inversely proportional to anomaly rate
        score = max(0.0, 1.0 - (anomaly_rate / self.TARGET_ANOMALY_RATE))
        status = 'pass' if anomaly_rate <= self.TARGET_ANOMALY_RATE else 'fail'

        return QualityMetric(
            name='Anomaly Rate',
            score=score,
            weight=0.15,
            status=status,
            details={
                'anomaly_count': anomaly_count,
                'total_count': total_count,
                'anomaly_rate_pct': anomaly_rate * 100,
                'target_anomaly_rate_pct': self.TARGET_ANOMALY_RATE * 100
            }
        )

    def _check_data_integrity(self) -> QualityMetric:
        """
        Check overall data integrity

        Returns:
            Data integrity quality metric
        """
        issues = []

        # Check for null/zero prices
        invalid_prices = sum(
            1 for p in self.price_data.values()
            if p['price_usd'] <= 0 or p['price_usd'] > 1000000
        )

        if invalid_prices > 0:
            issues.append(f"{invalid_prices} invalid price values")

        # Check for missing sources
        missing_sources = sum(
            1 for p in self.price_data.values()
            if not p.get('source')
        )

        if missing_sources > 0:
            issues.append(f"{missing_sources} missing source attribution")

        # Check for duplicate timestamps
        timestamps = list(self.price_data.keys())
        duplicates = len(timestamps) - len(set(timestamps))

        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps")

        # Calculate integrity score
        total_issues = invalid_prices + missing_sources + duplicates
        integrity_ratio = 1.0 - (total_issues / len(self.price_data))

        score = max(0.0, integrity_ratio)
        status = 'pass' if integrity_ratio >= 0.99 else 'fail'

        return QualityMetric(
            name='Data Integrity',
            score=score,
            weight=0.10,
            status=status,
            details={
                'total_issues': total_issues,
                'integrity_percentage': integrity_ratio * 100,
                'issues': issues if issues else ['No integrity issues']
            }
        )

    def _assign_grade(self, score: float) -> QualityGrade:
        """Assign quality grade based on score"""
        if score >= 0.95:
            return QualityGrade.A
        elif score >= 0.90:
            return QualityGrade.B
        elif score >= 0.80:
            return QualityGrade.C
        elif score >= 0.70:
            return QualityGrade.D
        else:
            return QualityGrade.F

    def _generate_recommendations(
        self,
        metrics: List[QualityMetric]
    ) -> List[str]:
        """
        Generate recommendations based on QA results

        Args:
            metrics: List of quality metrics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        for metric in metrics:
            if metric.status == 'fail':
                if metric.name == 'Coverage':
                    gap_count = metric.details.get('gap_count', 0)
                    recommendations.append(
                        f"Fill {gap_count} data gaps to improve coverage from "
                        f"{metric.details['coverage_percentage']:.1f}% to target "
                        f"{metric.details['target_coverage']:.1f}%"
                    )

                elif metric.name == 'Accuracy':
                    avg_dev = metric.details.get('avg_deviation_pct', 0)
                    recommendations.append(
                        f"Review price sources - average deviation {avg_dev:.2f}% "
                        f"exceeds tolerance {metric.details['tolerance_threshold']:.2f}%"
                    )

                elif metric.name == 'Consistency':
                    extreme = metric.details.get('extreme_changes', 0)
                    recommendations.append(
                        f"Investigate {extreme} extreme price changes that may indicate "
                        f"data quality issues"
                    )

                elif metric.name == 'Anomaly Rate':
                    anomaly_count = metric.details.get('anomaly_count', 0)
                    recommendations.append(
                        f"Review and correct {anomaly_count} anomalous price points"
                    )

                elif metric.name == 'Data Integrity':
                    issues = metric.details.get('issues', [])
                    recommendations.append(
                        f"Address data integrity issues: {', '.join(issues)}"
                    )

        if not recommendations:
            recommendations.append("All quality metrics passed - no action required")

        return recommendations

    def validate_against_external_benchmark(
        self,
        external_source_name: str,
        external_data: Dict[int, float]
    ) -> Dict:
        """
        Validate against specific external benchmark

        Args:
            external_source_name: Name of external source
            external_data: External price data

        Returns:
            Validation report
        """
        logger.info(f"Validating against {external_source_name}")

        comparisons = []
        mismatches = []

        for timestamp, price_info in self.price_data.items():
            if timestamp in external_data:
                our_price = price_info['price_usd']
                external_price = external_data[timestamp]

                deviation_pct = abs(our_price - external_price) / external_price

                if deviation_pct > self.accuracy_tolerance:
                    mismatches.append({
                        'timestamp': timestamp,
                        'our_price': our_price,
                        'external_price': external_price,
                        'deviation_pct': deviation_pct * 100
                    })

                comparisons.append(deviation_pct)

        if comparisons:
            avg_deviation = statistics.mean(comparisons)
            max_deviation = max(comparisons)
            accuracy = sum(1 for d in comparisons if d <= self.accuracy_tolerance) / len(comparisons)

            return {
                'external_source': external_source_name,
                'comparisons_count': len(comparisons),
                'avg_deviation_pct': avg_deviation * 100,
                'max_deviation_pct': max_deviation * 100,
                'accuracy_percentage': accuracy * 100,
                'mismatches_count': len(mismatches),
                'top_mismatches': sorted(mismatches, key=lambda x: x['deviation_pct'], reverse=True)[:10]
            }
        else:
            return {
                'external_source': external_source_name,
                'error': 'No overlapping timestamps for comparison'
            }

    def generate_quality_dashboard(self) -> Dict:
        """
        Generate quality dashboard data

        Returns:
            Dashboard data dict
        """
        prices = [p['price_usd'] for p in self.price_data.values()]
        timestamps = sorted(self.price_data.keys())

        return {
            'summary': {
                'total_data_points': len(self.price_data),
                'time_range': {
                    'start': timestamps[0],
                    'end': timestamps[-1],
                    'duration_hours': (timestamps[-1] - timestamps[0]) // 3600
                },
                'price_range': {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': statistics.mean(prices),
                    'std': statistics.stdev(prices) if len(prices) > 1 else 0
                }
            },
            'source_breakdown': self._get_source_breakdown(),
            'recent_anomalies': self._get_recent_anomalies(limit=10),
            'quality_targets': {
                'coverage': self.TARGET_COVERAGE * 100,
                'accuracy': self.TARGET_ACCURACY * 100,
                'consistency': self.TARGET_CONSISTENCY * 100,
                'max_anomaly_rate': self.TARGET_ANOMALY_RATE * 100
            }
        }

    def _get_source_breakdown(self) -> Dict[str, int]:
        """Get breakdown of prices by source"""
        breakdown = {}
        for price_info in self.price_data.values():
            source = price_info.get('source', 'unknown')
            breakdown[source] = breakdown.get(source, 0) + 1
        return breakdown

    def _get_recent_anomalies(self, limit: int = 10) -> List[Dict]:
        """Get most recent anomalies"""
        anomalies = [
            {
                'timestamp': ts,
                'price': info['price_usd'],
                'original_price': info.get('original_price'),
                'source': info.get('source')
            }
            for ts, info in self.price_data.items()
            if info.get('corrected', False)
        ]

        # Sort by timestamp descending and limit
        return sorted(anomalies, key=lambda x: x['timestamp'], reverse=True)[:limit]