"""Automated quality monitoring and alerting system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from asyncpg import Pool

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class QualityMonitor:
    """Automated quality monitoring and alerting framework."""

    def __init__(self, db_pool: Pool, thresholds: Optional[Dict[str, float]] = None):
        """Initialize quality monitor.

        Args:
            db_pool: Async database connection pool
            thresholds: Custom quality thresholds
        """
        self.db = db_pool
        self.logger = logger.bind(component="quality_monitor")

        # Default quality thresholds
        self.thresholds = thresholds or {
            "completeness_min": 0.95,
            "accuracy_min": 0.95,
            "consistency_min": 0.90,
            "timeliness_max_hours": 24,
            "integrity_min": 0.98,
        }

    async def monitor_quality_metrics(
        self, validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Monitor quality metrics and generate alerts.

        Args:
            validation_results: Complete validation results

        Returns:
            Monitoring report with alerts
        """
        log = self.logger
        log.info("monitoring_quality_metrics")

        alerts = []

        # Check each dimension against thresholds
        completeness_score = validation_results.get("completeness", {}).get(
            "overall_completeness", 1.0
        )
        if completeness_score < self.thresholds["completeness_min"]:
            alerts.append(
                self._create_alert(
                    AlertSeverity.WARNING,
                    "completeness_below_threshold",
                    f"Completeness score {completeness_score:.2f} below threshold "
                    f"{self.thresholds['completeness_min']:.2f}",
                    {"score": completeness_score},
                )
            )

        accuracy_score = validation_results.get("accuracy", {}).get(
            "accuracy_rate", 1.0
        )
        if accuracy_score < self.thresholds["accuracy_min"]:
            alerts.append(
                self._create_alert(
                    AlertSeverity.WARNING,
                    "accuracy_below_threshold",
                    f"Accuracy score {accuracy_score:.2f} below threshold "
                    f"{self.thresholds['accuracy_min']:.2f}",
                    {"score": accuracy_score},
                )
            )

        # Check for critical issues
        integrity_score = validation_results.get("integrity", {}).get(
            "overall_integrity_score", 1.0
        )
        if integrity_score < self.thresholds["integrity_min"]:
            alerts.append(
                self._create_alert(
                    AlertSeverity.CRITICAL,
                    "integrity_violation",
                    f"Database integrity score {integrity_score:.2f} critically low",
                    {"score": integrity_score},
                )
            )

        # Classify alerts by severity
        alert_summary = {
            "total_alerts": len(alerts),
            "critical_count": sum(
                1 for a in alerts if a["severity"] == AlertSeverity.CRITICAL.value
            ),
            "warning_count": sum(
                1 for a in alerts if a["severity"] == AlertSeverity.WARNING.value
            ),
            "info_count": sum(
                1 for a in alerts if a["severity"] == AlertSeverity.INFO.value
            ),
        }

        log.info("quality_monitoring_completed", **alert_summary)

        return {
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "alert_summary": alert_summary,
            "thresholds_used": self.thresholds,
            "requires_attention": alert_summary["critical_count"] > 0,
        }

    def _create_alert(
        self,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create an alert object.

        Args:
            severity: Alert severity level
            alert_type: Type of alert
            message: Alert message
            metadata: Additional metadata

        Returns:
            Alert dict
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "severity": severity.value,
            "alert_type": alert_type,
            "message": message,
            "metadata": metadata,
            "remediation": self._get_remediation_steps(alert_type),
        }

    def _get_remediation_steps(self, alert_type: str) -> List[str]:
        """Get remediation steps for an alert type.

        Args:
            alert_type: Type of alert

        Returns:
            List of remediation steps
        """
        remediation_map = {
            "completeness_below_threshold": [
                "Review data collection processes",
                "Check for gaps in data coverage",
                "Verify all data sources are active",
            ],
            "accuracy_below_threshold": [
                "Review price validation results",
                "Cross-check with external sources",
                "Verify calculation formulas",
            ],
            "integrity_violation": [
                "Run database integrity checks",
                "Review foreign key relationships",
                "Check for orphaned records",
            ],
        }

        return remediation_map.get(
            alert_type, ["Review validation results", "Contact data team"]
        )

    async def track_quality_history(
        self, quality_score: float, dimension_scores: Dict[str, float]
    ) -> None:
        """Track quality score history in database.

        Args:
            quality_score: Composite quality score
            dimension_scores: Individual dimension scores
        """
        log = self.logger
        log.info("tracking_quality_history", quality_score=quality_score)

        query = """
            INSERT INTO quality_metrics_history (
                timestamp,
                composite_score,
                completeness_score,
                accuracy_score,
                consistency_score,
                timeliness_score,
                validity_score
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.now(),
                    quality_score,
                    dimension_scores.get("completeness", 0),
                    dimension_scores.get("accuracy", 0),
                    dimension_scores.get("consistency", 0),
                    dimension_scores.get("timeliness", 0),
                    dimension_scores.get("validity", 0),
                )
            log.info("quality_history_tracked")
        except Exception as e:
            log.warning("failed_to_track_history", error=str(e))

    def classify_error(
        self, error_type: str, error_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify and assess error severity.

        Args:
            error_type: Type of error
            error_details: Error details

        Returns:
            Error classification
        """
        # Error classification logic
        severity_map = {
            "missing_data": AlertSeverity.WARNING,
            "validation_failure": AlertSeverity.WARNING,
            "integrity_violation": AlertSeverity.CRITICAL,
            "constraint_violation": AlertSeverity.CRITICAL,
            "duplicate_record": AlertSeverity.WARNING,
            "orphaned_record": AlertSeverity.WARNING,
        }

        severity = severity_map.get(error_type, AlertSeverity.INFO)

        return {
            "error_type": error_type,
            "severity": severity.value,
            "error_details": error_details,
            "timestamp": datetime.now().isoformat(),
            "requires_immediate_action": severity == AlertSeverity.CRITICAL,
        }