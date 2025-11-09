"""
Price Anomaly Detection and Quality Control
Detects price anomalies, outliers, and data quality issues
"""

import logging
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrectionAction(Enum):
    """Correction action types"""
    INTERPOLATE = "interpolate"
    USE_FALLBACK = "use_fallback"
    MANUAL_REVIEW = "manual_review"
    REMOVE = "remove"
    ACCEPT = "accept"


@dataclass
class PriceAnomaly:
    """Price anomaly data structure"""
    timestamp: int
    price: float
    expected_range: Tuple[float, float]
    z_score: float
    severity: AnomalySeverity
    detection_method: str
    suggested_action: CorrectionAction
    context: Dict


@dataclass
class ValidationRule:
    """Price validation rule"""
    name: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_hourly_change_pct: Optional[float] = None
    max_z_score: Optional[float] = None


class PriceAnomalyDetector:
    """
    Comprehensive price anomaly detection and quality control

    Features:
    - Statistical anomaly detection (z-score, IQR)
    - Sudden change detection
    - Validation rules enforcement
    - Automated correction suggestions
    - Data integrity monitoring
    """

    def __init__(
        self,
        price_data: Dict[int, Dict],
        validation_rules: Optional[List[ValidationRule]] = None
    ):
        """
        Initialize anomaly detector

        Args:
            price_data: Dict of {timestamp -> price_data}
            validation_rules: Optional list of validation rules
        """
        self.price_data = price_data
        self.validation_rules = validation_rules or self._default_validation_rules()
        logger.info(
            f"Anomaly detector initialized with {len(price_data)} price points "
            f"and {len(self.validation_rules)} validation rules"
        )

    def _default_validation_rules(self) -> List[ValidationRule]:
        """Default validation rules for ETH/USD"""
        return [
            ValidationRule(
                name="eth_price_bounds",
                min_price=10.0,  # ETH should be > $10
                max_price=100000.0  # ETH should be < $100k (reasonable upper bound)
            ),
            ValidationRule(
                name="hourly_volatility",
                max_hourly_change_pct=20.0  # Max 20% change per hour
            ),
            ValidationRule(
                name="statistical_outlier",
                max_z_score=3.0  # 3 standard deviations
            )
        ]

    def detect_all_anomalies(
        self,
        window_hours: int = 24,
        std_threshold: float = 3.0
    ) -> List[PriceAnomaly]:
        """
        Detect all types of anomalies in price data

        Args:
            window_hours: Rolling window size for statistics
            std_threshold: Standard deviation threshold for z-score

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Statistical anomalies (z-score based)
        stat_anomalies = self.detect_statistical_anomalies(window_hours, std_threshold)
        anomalies.extend(stat_anomalies)

        # Sudden changes
        sudden_changes = self.detect_sudden_changes()
        anomalies.extend(sudden_changes)

        # Validation rule violations
        rule_violations = self.detect_rule_violations()
        anomalies.extend(rule_violations)

        # Remove duplicates (same timestamp)
        unique_anomalies = self._deduplicate_anomalies(anomalies)

        logger.info(
            f"Detected {len(unique_anomalies)} anomalies: "
            f"{sum(1 for a in unique_anomalies if a.severity == AnomalySeverity.CRITICAL)} critical, "
            f"{sum(1 for a in unique_anomalies if a.severity == AnomalySeverity.HIGH)} high, "
            f"{sum(1 for a in unique_anomalies if a.severity == AnomalySeverity.MEDIUM)} medium"
        )

        return unique_anomalies

    def detect_statistical_anomalies(
        self,
        window_hours: int = 24,
        std_threshold: float = 3.0
    ) -> List[PriceAnomaly]:
        """
        Detect anomalies using statistical methods (z-score)

        Args:
            window_hours: Rolling window size for statistics
            std_threshold: Standard deviation threshold

        Returns:
            List of statistical anomalies
        """
        anomalies = []
        sorted_data = sorted(self.price_data.items())

        for i, (timestamp, price_info) in enumerate(sorted_data):
            current_price = price_info['price_usd']

            # Calculate rolling window statistics
            start_idx = max(0, i - window_hours)
            end_idx = min(len(sorted_data), i + window_hours + 1)
            window_prices = [
                sorted_data[j][1]['price_usd']
                for j in range(start_idx, end_idx)
            ]

            if len(window_prices) < 3:
                continue

            mean_price = statistics.mean(window_prices)
            std_price = statistics.stdev(window_prices)

            # Calculate z-score
            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price

                if z_score > std_threshold:
                    # Determine severity
                    severity = self._calculate_severity_from_zscore(z_score)

                    # Expected range
                    expected_range = (
                        mean_price - std_threshold * std_price,
                        mean_price + std_threshold * std_price
                    )

                    anomalies.append(PriceAnomaly(
                        timestamp=timestamp,
                        price=current_price,
                        expected_range=expected_range,
                        z_score=z_score,
                        severity=severity,
                        detection_method="z_score",
                        suggested_action=self._suggest_action(severity),
                        context={
                            'window_mean': mean_price,
                            'window_std': std_price,
                            'window_size': len(window_prices)
                        }
                    ))

        return anomalies

    def detect_sudden_changes(
        self,
        change_threshold: float = 0.10
    ) -> List[PriceAnomaly]:
        """
        Detect sudden price changes between consecutive data points

        Args:
            change_threshold: Percentage change threshold (e.g., 0.10 = 10%)

        Returns:
            List of sudden change anomalies
        """
        anomalies = []
        sorted_data = sorted(self.price_data.items())

        for i in range(1, len(sorted_data)):
            prev_timestamp, prev_info = sorted_data[i-1]
            curr_timestamp, curr_info = sorted_data[i]

            prev_price = prev_info['price_usd']
            curr_price = curr_info['price_usd']

            # Calculate percentage change
            if prev_price > 0:
                pct_change = abs(curr_price - prev_price) / prev_price

                if pct_change > change_threshold:
                    # Calculate severity based on magnitude
                    if pct_change > 0.50:
                        severity = AnomalySeverity.CRITICAL
                    elif pct_change > 0.30:
                        severity = AnomalySeverity.HIGH
                    elif pct_change > 0.20:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    time_gap = curr_timestamp - prev_timestamp

                    anomalies.append(PriceAnomaly(
                        timestamp=curr_timestamp,
                        price=curr_price,
                        expected_range=(prev_price * 0.9, prev_price * 1.1),
                        z_score=0.0,  # Not applicable
                        severity=severity,
                        detection_method="sudden_change",
                        suggested_action=self._suggest_action(severity),
                        context={
                            'previous_price': prev_price,
                            'change_percentage': pct_change * 100,
                            'time_gap_seconds': time_gap
                        }
                    ))

        return anomalies

    def detect_rule_violations(self) -> List[PriceAnomaly]:
        """
        Detect violations of validation rules

        Returns:
            List of rule violation anomalies
        """
        anomalies = []

        for timestamp, price_info in self.price_data.items():
            price = price_info['price_usd']

            for rule in self.validation_rules:
                violation = None

                # Check price bounds
                if rule.min_price is not None and price < rule.min_price:
                    violation = f"Price ${price:.2f} below minimum ${rule.min_price}"
                    severity = AnomalySeverity.CRITICAL

                elif rule.max_price is not None and price > rule.max_price:
                    violation = f"Price ${price:.2f} above maximum ${rule.max_price}"
                    severity = AnomalySeverity.CRITICAL

                if violation:
                    anomalies.append(PriceAnomaly(
                        timestamp=timestamp,
                        price=price,
                        expected_range=(
                            rule.min_price or 0,
                            rule.max_price or float('inf')
                        ),
                        z_score=0.0,
                        severity=severity,
                        detection_method="rule_violation",
                        suggested_action=CorrectionAction.MANUAL_REVIEW,
                        context={
                            'rule_name': rule.name,
                            'violation': violation
                        }
                    ))

        return anomalies

    def correct_anomalies(
        self,
        anomalies: List[PriceAnomaly],
        auto_correct: bool = False
    ) -> Dict[int, Dict]:
        """
        Correct detected anomalies

        Args:
            anomalies: List of detected anomalies
            auto_correct: If True, automatically apply corrections

        Returns:
            Corrected price data
        """
        corrected_data = self.price_data.copy()
        corrections_applied = 0

        for anomaly in anomalies:
            if not auto_correct and anomaly.severity == AnomalySeverity.CRITICAL:
                logger.warning(
                    f"Critical anomaly at {anomaly.timestamp} requires manual review"
                )
                continue

            if anomaly.suggested_action == CorrectionAction.INTERPOLATE:
                corrected_price = self._interpolate_price(anomaly.timestamp)
                if corrected_price:
                    corrected_data[anomaly.timestamp]['price_usd'] = corrected_price
                    corrected_data[anomaly.timestamp]['corrected'] = True
                    corrected_data[anomaly.timestamp]['original_price'] = anomaly.price
                    corrections_applied += 1

            elif anomaly.suggested_action == CorrectionAction.REMOVE:
                if auto_correct:
                    del corrected_data[anomaly.timestamp]
                    corrections_applied += 1

        logger.info(f"Applied {corrections_applied} corrections out of {len(anomalies)} anomalies")
        return corrected_data

    def _interpolate_price(self, timestamp: int) -> Optional[float]:
        """
        Interpolate price for given timestamp

        Args:
            timestamp: Unix timestamp

        Returns:
            Interpolated price or None
        """
        sorted_data = sorted(self.price_data.items())

        # Find surrounding prices
        before_price = None
        after_price = None

        for i, (ts, price_info) in enumerate(sorted_data):
            if ts < timestamp:
                before_price = (ts, price_info['price_usd'])
            elif ts > timestamp:
                after_price = (ts, price_info['price_usd'])
                break

        # Linear interpolation
        if before_price and after_price:
            before_ts, before_val = before_price
            after_ts, after_val = after_price

            # Calculate interpolated value
            time_ratio = (timestamp - before_ts) / (after_ts - before_ts)
            interpolated = before_val + (after_val - before_val) * time_ratio

            logger.debug(
                f"Interpolated price ${interpolated:.2f} for timestamp {timestamp}"
            )
            return interpolated

        return None

    def generate_quality_report(self, anomalies: List[PriceAnomaly]) -> Dict:
        """
        Generate comprehensive quality report

        Args:
            anomalies: List of detected anomalies

        Returns:
            Quality report dict
        """
        total_points = len(self.price_data)
        anomaly_count = len(anomalies)
        anomaly_rate = anomaly_count / total_points if total_points > 0 else 0

        # Count by severity
        severity_counts = {
            AnomalySeverity.CRITICAL: 0,
            AnomalySeverity.HIGH: 0,
            AnomalySeverity.MEDIUM: 0,
            AnomalySeverity.LOW: 0
        }
        for anomaly in anomalies:
            severity_counts[anomaly.severity] += 1

        # Calculate quality score (0-1)
        quality_score = 1.0 - (
            severity_counts[AnomalySeverity.CRITICAL] * 0.1 +
            severity_counts[AnomalySeverity.HIGH] * 0.05 +
            severity_counts[AnomalySeverity.MEDIUM] * 0.02 +
            severity_counts[AnomalySeverity.LOW] * 0.01
        ) / total_points

        quality_score = max(0.0, min(1.0, quality_score))

        return {
            'total_data_points': total_points,
            'anomalies_detected': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'quality_score': quality_score,
            'quality_grade': self._assign_quality_grade(quality_score),
            'severity_breakdown': {
                'critical': severity_counts[AnomalySeverity.CRITICAL],
                'high': severity_counts[AnomalySeverity.HIGH],
                'medium': severity_counts[AnomalySeverity.MEDIUM],
                'low': severity_counts[AnomalySeverity.LOW]
            },
            'needs_manual_review': severity_counts[AnomalySeverity.CRITICAL] > 0
        }

    def monitor_data_integrity(self) -> Dict:
        """
        Monitor overall data integrity

        Returns:
            Data integrity metrics
        """
        sorted_data = sorted(self.price_data.items())

        if len(sorted_data) < 2:
            return {
                'has_gaps': False,
                'gap_count': 0,
                'coverage_percentage': 0.0,
                'completeness_score': 0.0
            }

        # Check for gaps (missing hourly data)
        expected_hours = (sorted_data[-1][0] - sorted_data[0][0]) // 3600
        actual_hours = len(sorted_data)
        gap_count = expected_hours - actual_hours

        coverage_percentage = (actual_hours / expected_hours * 100) if expected_hours > 0 else 0

        # Check for duplicates
        duplicate_count = len(self.price_data) - len(set(self.price_data.keys()))

        # Calculate completeness score
        completeness_score = min(1.0, actual_hours / expected_hours) if expected_hours > 0 else 0

        return {
            'has_gaps': gap_count > 0,
            'gap_count': gap_count,
            'expected_data_points': expected_hours,
            'actual_data_points': actual_hours,
            'coverage_percentage': coverage_percentage,
            'has_duplicates': duplicate_count > 0,
            'duplicate_count': duplicate_count,
            'completeness_score': completeness_score,
            'time_range': {
                'start_timestamp': sorted_data[0][0],
                'end_timestamp': sorted_data[-1][0],
                'duration_hours': expected_hours
            }
        }

    def _calculate_severity_from_zscore(self, z_score: float) -> AnomalySeverity:
        """Calculate severity based on z-score"""
        if z_score > 5:
            return AnomalySeverity.CRITICAL
        elif z_score > 4:
            return AnomalySeverity.HIGH
        elif z_score > 3:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _suggest_action(self, severity: AnomalySeverity) -> CorrectionAction:
        """Suggest correction action based on severity"""
        if severity == AnomalySeverity.CRITICAL:
            return CorrectionAction.MANUAL_REVIEW
        elif severity == AnomalySeverity.HIGH:
            return CorrectionAction.USE_FALLBACK
        elif severity == AnomalySeverity.MEDIUM:
            return CorrectionAction.INTERPOLATE
        else:
            return CorrectionAction.ACCEPT

    def _assign_quality_grade(self, score: float) -> str:
        """Assign quality grade based on score"""
        if score >= 0.95:
            return 'A'
        elif score >= 0.90:
            return 'B'
        elif score >= 0.80:
            return 'C'
        elif score >= 0.70:
            return 'D'
        else:
            return 'F'

    def _deduplicate_anomalies(
        self,
        anomalies: List[PriceAnomaly]
    ) -> List[PriceAnomaly]:
        """
        Remove duplicate anomalies for same timestamp

        Keep the one with highest severity
        """
        timestamp_anomalies: Dict[int, PriceAnomaly] = {}

        severity_order = {
            AnomalySeverity.CRITICAL: 4,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 1
        }

        for anomaly in anomalies:
            ts = anomaly.timestamp
            if ts not in timestamp_anomalies:
                timestamp_anomalies[ts] = anomaly
            else:
                # Keep higher severity
                if severity_order[anomaly.severity] > severity_order[timestamp_anomalies[ts].severity]:
                    timestamp_anomalies[ts] = anomaly

        return list(timestamp_anomalies.values())