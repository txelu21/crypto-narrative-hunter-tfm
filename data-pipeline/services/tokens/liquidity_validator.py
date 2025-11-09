import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
import json
from pathlib import Path

from data_collection.common.logging_setup import get_logger
from data_collection.common.db import DatabaseManager
from .liquidity_analyzer import PoolData

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    test_name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class AnomalyDetection:
    metric_name: str
    threshold_type: str  # 'upper', 'lower', 'both'
    threshold_value: float
    detected_values: List[Any]
    severity: str  # 'low', 'medium', 'high'


class LiquidityValidator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

        # Validation thresholds
        self.thresholds = {
            "min_coverage_rate": 70.0,  # % of tokens with liquidity
            "min_pricing_coverage": 60.0,  # % of tokens with ETH pricing
            "max_tier_1_percentage": 15.0,  # Max % of tokens in Tier 1
            "min_pools_per_token": 1.5,  # Avg pools per token
            "tvl_outlier_multiplier": 10.0,  # TVL outlier detection
            "volume_consistency_threshold": 0.1,  # Volume vs TVL ratio
        }

        # External validation sources
        self.external_apis = {
            "coingecko": "https://api.coingecko.com/api/v3",
            "defipulse": "https://api.defipulse.com/api/v1"
        }

    def validate_liquidity_coverage(self) -> ValidationResult:
        """Validate that sufficient percentage of tokens have discovered liquidity"""
        with self.db_manager.get_connection() as conn:
            # Get coverage statistics
            result = conn.execute("""
                SELECT
                    COUNT(*) as total_tokens,
                    COUNT(CASE WHEN liquidity_tier != 'Untiered' THEN 1 END) as tokens_with_liquidity,
                    COUNT(CASE WHEN liquidity_tier = 'Tier 1' THEN 1 END) as tier_1_count,
                    COUNT(CASE WHEN liquidity_tier = 'Tier 2' THEN 1 END) as tier_2_count,
                    COUNT(CASE WHEN liquidity_tier = 'Tier 3' THEN 1 END) as tier_3_count
                FROM tokens
                WHERE liquidity_tier IS NOT NULL
            """).fetchone()

            total = result[0]
            with_liquidity = result[1]
            coverage_rate = (with_liquidity / total * 100) if total > 0 else 0

            passed = coverage_rate >= self.thresholds["min_coverage_rate"]
            score = min(100, coverage_rate)

            recommendations = []
            if not passed:
                recommendations.extend([
                    f"Coverage rate {coverage_rate:.1f}% below threshold {self.thresholds['min_coverage_rate']}%",
                    "Consider expanding token filtering to include more DEXs",
                    "Check if Dune query date ranges are appropriate",
                    "Verify token addresses are correctly formatted"
                ])

            return ValidationResult(
                test_name="liquidity_coverage",
                passed=passed,
                score=score,
                details={
                    "total_tokens": total,
                    "tokens_with_liquidity": with_liquidity,
                    "coverage_rate": coverage_rate,
                    "tier_distribution": {
                        "tier_1": result[2],
                        "tier_2": result[3],
                        "tier_3": result[4]
                    }
                },
                recommendations=recommendations
            )

    def validate_pricing_coverage(self) -> ValidationResult:
        """Validate ETH pricing coverage across tokens"""
        with self.db_manager.get_connection() as conn:
            result = conn.execute("""
                SELECT
                    COUNT(DISTINCT t.token_address) as total_tokens,
                    COUNT(DISTINCT CASE WHEN tp.price_eth IS NOT NULL THEN t.token_address END) as tokens_with_eth_price
                FROM tokens t
                LEFT JOIN token_pools tp ON t.token_address = tp.token_address
                WHERE t.liquidity_tier IS NOT NULL
                  AND t.liquidity_tier != 'Untiered'
            """).fetchone()

            total = result[0]
            with_pricing = result[1]
            pricing_coverage = (with_pricing / total * 100) if total > 0 else 0

            passed = pricing_coverage >= self.thresholds["min_pricing_coverage"]
            score = min(100, pricing_coverage)

            recommendations = []
            if not passed:
                recommendations.extend([
                    f"Pricing coverage {pricing_coverage:.1f}% below threshold {self.thresholds['min_pricing_coverage']}%",
                    "Increase fallback routing through stablecoin pairs",
                    "Expand WETH pair discovery in Dune queries",
                    "Consider cross-DEX price aggregation"
                ])

            return ValidationResult(
                test_name="pricing_coverage",
                passed=passed,
                score=score,
                details={
                    "total_tokens": total,
                    "tokens_with_eth_price": with_pricing,
                    "pricing_coverage": pricing_coverage
                },
                recommendations=recommendations
            )

    def validate_tier_distribution(self) -> ValidationResult:
        """Validate liquidity tier distribution is reasonable"""
        with self.db_manager.get_connection() as conn:
            results = conn.execute("""
                SELECT liquidity_tier, COUNT(*) as count
                FROM tokens
                WHERE liquidity_tier IS NOT NULL
                GROUP BY liquidity_tier
            """).fetchall()

            tier_counts = {row[0]: row[1] for row in results}
            total = sum(tier_counts.values())

            if total == 0:
                return ValidationResult(
                    test_name="tier_distribution",
                    passed=False,
                    score=0,
                    details={"error": "No tokens with tier assignments"},
                    recommendations=["Run liquidity analysis first"]
                )

            # Calculate percentages
            tier_percentages = {tier: (count / total * 100) for tier, count in tier_counts.items()}
            tier_1_pct = tier_percentages.get("Tier 1", 0)

            # Validate tier 1 isn't too high (would indicate poor filtering)
            tier_1_valid = tier_1_pct <= self.thresholds["max_tier_1_percentage"]

            # Expect majority in Tier 2/3, minority in Tier 1/Untiered
            distribution_score = 100
            if tier_1_pct > 20:
                distribution_score -= 30  # Too many high-liquidity tokens
            if tier_percentages.get("Untiered", 0) > 50:
                distribution_score -= 40  # Too many tokens without liquidity

            passed = tier_1_valid and distribution_score >= 60
            score = max(0, distribution_score)

            recommendations = []
            if not tier_1_valid:
                recommendations.append(f"Tier 1 percentage {tier_1_pct:.1f}% too high - review TVL thresholds")
            if tier_percentages.get("Untiered", 0) > 30:
                recommendations.append("High untiered percentage - expand DEX coverage")

            return ValidationResult(
                test_name="tier_distribution",
                passed=passed,
                score=score,
                details={
                    "tier_counts": tier_counts,
                    "tier_percentages": tier_percentages,
                    "total_tokens": total
                },
                recommendations=recommendations
            )

    def detect_tvl_anomalies(self) -> List[AnomalyDetection]:
        """Detect TVL outliers that may indicate data quality issues"""
        anomalies = []

        with self.db_manager.get_connection() as conn:
            # Get TVL statistics
            results = conn.execute("""
                SELECT
                    token_address,
                    pool_address,
                    dex_name,
                    tvl_usd,
                    volume_24h_usd
                FROM token_pools
                WHERE tvl_usd > 0
                ORDER BY tvl_usd DESC
            """).fetchall()

            if len(results) < 10:
                return anomalies

            tvl_values = [float(row[3]) for row in results]
            volume_values = [float(row[4]) for row in results if row[4] is not None]

            # Statistical outlier detection for TVL
            q75, q25 = np.percentile(tvl_values, [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + (self.thresholds["tvl_outlier_multiplier"] * iqr)

            tvl_outliers = [row for row in results if float(row[3]) > upper_bound]

            if tvl_outliers:
                anomalies.append(AnomalyDetection(
                    metric_name="tvl_outliers",
                    threshold_type="upper",
                    threshold_value=upper_bound,
                    detected_values=tvl_outliers[:5],  # Top 5 outliers
                    severity="medium" if len(tvl_outliers) < 5 else "high"
                ))

            # Volume consistency check (volume should correlate with TVL)
            if volume_values:
                for row in results:
                    tvl = float(row[3])
                    volume = float(row[4]) if row[4] else 0
                    if tvl > 1000000 and volume > 0:  # Only check significant pools
                        volume_tvl_ratio = volume / tvl
                        if volume_tvl_ratio > 5.0:  # Volume > 5x TVL in 24h is suspicious
                            anomalies.append(AnomalyDetection(
                                metric_name="volume_tvl_inconsistency",
                                threshold_type="upper",
                                threshold_value=5.0,
                                detected_values=[{
                                    "token": row[0],
                                    "pool": row[1],
                                    "dex": row[2],
                                    "ratio": volume_tvl_ratio
                                }],
                                severity="high"
                            ))

        return anomalies

    def cross_validate_with_external_sources(
        self,
        sample_size: int = 10
    ) -> ValidationResult:
        """Cross-validate pricing and TVL data with external sources"""
        try:
            with self.db_manager.get_connection() as conn:
                # Get sample of high-liquidity tokens for validation
                results = conn.execute("""
                    SELECT DISTINCT
                        t.token_address,
                        t.symbol,
                        tp.price_eth,
                        tp.tvl_usd
                    FROM tokens t
                    JOIN token_pools tp ON t.token_address = tp.token_address
                    WHERE t.liquidity_tier IN ('Tier 1', 'Tier 2')
                      AND tp.price_eth IS NOT NULL
                      AND tp.tvl_usd > 100000
                    ORDER BY tp.tvl_usd DESC
                    LIMIT %s
                """, (sample_size,)).fetchall()

            if not results:
                return ValidationResult(
                    test_name="external_validation",
                    passed=False,
                    score=0,
                    details={"error": "No suitable tokens for validation"},
                    recommendations=["Ensure tokens have sufficient liquidity for validation"]
                )

            # Validate against CoinGecko (free tier)
            validated_count = 0
            price_deviations = []

            for row in results:
                symbol = row[1].lower()
                our_price_eth = float(row[2]) if row[2] else None

                if not our_price_eth:
                    continue

                try:
                    # Get ETH price in USD
                    eth_response = requests.get(
                        f"{self.external_apis['coingecko']}/simple/price?ids=ethereum&vs_currencies=usd",
                        timeout=10
                    )
                    eth_price_usd = eth_response.json()["ethereum"]["usd"]

                    # Get token price in USD
                    token_response = requests.get(
                        f"{self.external_apis['coingecko']}/simple/price?ids={symbol}&vs_currencies=usd",
                        timeout=10
                    )

                    if symbol in token_response.json():
                        external_price_usd = token_response.json()[symbol]["usd"]
                        external_price_eth = external_price_usd / eth_price_usd

                        # Calculate deviation
                        deviation = abs(our_price_eth - external_price_eth) / external_price_eth * 100
                        price_deviations.append({
                            "symbol": row[1],
                            "our_price_eth": our_price_eth,
                            "external_price_eth": external_price_eth,
                            "deviation_pct": deviation
                        })

                        validated_count += 1

                except Exception as e:
                    logger.debug(f"Failed to validate {symbol}: {e}")
                    continue

            # Calculate validation score
            if validated_count == 0:
                score = 0
                passed = False
            else:
                avg_deviation = np.mean([d["deviation_pct"] for d in price_deviations])
                # Score based on accuracy (lower deviation = higher score)
                score = max(0, 100 - avg_deviation)
                passed = avg_deviation < 20  # Accept <20% average deviation

            recommendations = []
            if not passed:
                recommendations.extend([
                    f"High price deviation detected: {avg_deviation:.1f}% average",
                    "Review Dune query price calculations",
                    "Consider using time-weighted average prices",
                    "Validate against multiple price sources"
                ])

            return ValidationResult(
                test_name="external_validation",
                passed=passed,
                score=score,
                details={
                    "validated_tokens": validated_count,
                    "price_deviations": price_deviations,
                    "avg_deviation_pct": np.mean([d["deviation_pct"] for d in price_deviations]) if price_deviations else 0
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"External validation failed: {e}")
            return ValidationResult(
                test_name="external_validation",
                passed=False,
                score=0,
                details={"error": str(e)},
                recommendations=["Check network connectivity and API keys"]
            )

    def generate_quality_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive quality validation report"""
        logger.info("Generating liquidity analysis quality report")

        # Run all validations
        validations = [
            self.validate_liquidity_coverage(),
            self.validate_pricing_coverage(),
            self.validate_tier_distribution(),
            self.cross_validate_with_external_sources()
        ]

        # Detect anomalies
        anomalies = self.detect_tvl_anomalies()

        # Calculate overall score
        validation_scores = [v.score for v in validations if v.score is not None]
        overall_score = np.mean(validation_scores) if validation_scores else 0

        # Count critical issues
        critical_issues = sum(1 for v in validations if not v.passed)
        high_severity_anomalies = sum(1 for a in anomalies if a.severity == "high")

        # Determine overall status
        if overall_score >= 80 and critical_issues == 0:
            status = "excellent"
        elif overall_score >= 60 and critical_issues <= 1:
            status = "good"
        elif overall_score >= 40:
            status = "acceptable"
        else:
            status = "poor"

        # Compile report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "overall_score": round(overall_score, 2),
                "status": status,
                "critical_issues": critical_issues,
                "high_severity_anomalies": high_severity_anomalies
            },
            "validation_results": [
                {
                    "test_name": v.test_name,
                    "passed": v.passed,
                    "score": v.score,
                    "details": v.details,
                    "recommendations": v.recommendations
                }
                for v in validations
            ],
            "anomaly_detection": [
                {
                    "metric_name": a.metric_name,
                    "threshold_type": a.threshold_type,
                    "threshold_value": a.threshold_value,
                    "detected_count": len(a.detected_values),
                    "severity": a.severity,
                    "sample_values": a.detected_values[:3] if len(a.detected_values) > 3 else a.detected_values
                }
                for a in anomalies
            ],
            "summary_recommendations": self._generate_summary_recommendations(validations, anomalies)
        }

        # Save report
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Quality report saved to {output_path}")

        return report

    def _generate_summary_recommendations(
        self,
        validations: List[ValidationResult],
        anomalies: List[AnomalyDetection]
    ) -> List[str]:
        """Generate prioritized summary recommendations"""
        recommendations = []

        # High priority: Failed validations
        failed_validations = [v for v in validations if not v.passed]
        if failed_validations:
            recommendations.append("CRITICAL: Address failed validation tests immediately")

        # High severity anomalies
        high_severity = [a for a in anomalies if a.severity == "high"]
        if high_severity:
            recommendations.append("HIGH: Investigate high-severity data anomalies")

        # Coverage improvements
        coverage_result = next((v for v in validations if v.test_name == "liquidity_coverage"), None)
        if coverage_result and coverage_result.score < 80:
            recommendations.append("MEDIUM: Improve liquidity coverage by expanding DEX sources")

        # Pricing improvements
        pricing_result = next((v for v in validations if v.test_name == "pricing_coverage"), None)
        if pricing_result and pricing_result.score < 70:
            recommendations.append("MEDIUM: Enhance ETH pricing coverage with fallback routes")

        # Data quality
        if len(anomalies) > 5:
            recommendations.append("LOW: Review data quality processes to reduce anomalies")

        return recommendations

    def export_pool_data_for_review(self, output_path: str, limit: int = 1000):
        """Export pool data to CSV for manual review"""
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT
                    t.symbol,
                    tp.token_address,
                    tp.pool_address,
                    tp.dex_name,
                    tp.pair_token,
                    tp.tvl_usd,
                    tp.volume_24h_usd,
                    tp.price_eth,
                    t.liquidity_tier,
                    tp.last_updated
                FROM token_pools tp
                JOIN tokens t ON tp.token_address = t.token_address
                ORDER BY tp.tvl_usd DESC
                LIMIT %s
            """

            df = pd.read_sql_query(query, conn, params=(limit,))
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} pool records to {output_path}")

    def export_tier_assignments(self, output_path: str):
        """Export tier assignments summary to CSV"""
        with self.db_manager.get_connection() as conn:
            query = """
                SELECT
                    t.symbol,
                    t.token_address,
                    t.liquidity_tier,
                    COUNT(tp.pool_address) as pool_count,
                    MAX(tp.tvl_usd) as max_tvl_usd,
                    MAX(tp.price_eth) as price_eth,
                    AVG(tp.volume_24h_usd) as avg_volume_24h
                FROM tokens t
                LEFT JOIN token_pools tp ON t.token_address = tp.token_address
                WHERE t.liquidity_tier IS NOT NULL
                GROUP BY t.symbol, t.token_address, t.liquidity_tier
                ORDER BY
                    CASE t.liquidity_tier
                        WHEN 'Tier 1' THEN 1
                        WHEN 'Tier 2' THEN 2
                        WHEN 'Tier 3' THEN 3
                        WHEN 'Untiered' THEN 4
                    END,
                    MAX(tp.tvl_usd) DESC NULLS LAST
            """

            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} tier assignments to {output_path}")