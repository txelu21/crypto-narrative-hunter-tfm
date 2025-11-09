"""Statistical validation and outlier detection.

This module implements multi-dimensional outlier detection and statistical
validation for all numeric fields in the dataset.
"""

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from asyncpg import Pool
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = structlog.get_logger()


class StatisticalValidator:
    """Statistical validation and outlier detection framework."""

    def __init__(self, db_pool: Pool, contamination: float = 0.1):
        """Initialize statistical validator.

        Args:
            db_pool: Async database connection pool
            contamination: Expected proportion of outliers (default 10%)
        """
        self.db = db_pool
        self.contamination = contamination
        self.logger = logger.bind(component="statistical_validator")

        # Initialize anomaly detection models
        self.anomaly_detectors = {
            "isolation_forest": IsolationForest(
                contamination=contamination, random_state=42
            ),
            "local_outlier_factor": LocalOutlierFactor(
                contamination=contamination, novelty=False
            ),
            "one_class_svm": OneClassSVM(nu=contamination),
        }

    async def detect_multidimensional_outliers(
        self,
        dataset_name: str,
        features: List[str],
        min_consensus: int = 2,
    ) -> Dict[str, Any]:
        """Detect outliers using multiple statistical methods.

        Args:
            dataset_name: Name of dataset (e.g., 'transactions', 'wallets')
            features: List of feature columns to analyze
            min_consensus: Minimum number of methods that must agree (default 2)

        Returns:
            Outlier detection results with consensus outliers
        """
        log = self.logger.bind(
            dataset_name=dataset_name, feature_count=len(features)
        )
        log.info("detecting_multidimensional_outliers")

        # Fetch dataset
        dataset = await self._fetch_dataset(dataset_name, features)

        if len(dataset) == 0:
            log.warning("empty_dataset")
            return {
                "dataset_name": dataset_name,
                "total_records": 0,
                "outliers_detected": 0,
                "outlier_rate": 0,
                "consensus_outliers": [],
            }

        # Prepare feature matrix
        X = dataset[features].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        outlier_scores = {}

        # Apply multiple detection methods
        for method_name, detector in self.anomaly_detectors.items():
            try:
                if method_name == "local_outlier_factor":
                    outlier_labels = detector.fit_predict(X_scaled)
                    outlier_scores[method_name] = outlier_labels == -1
                else:
                    detector.fit(X_scaled)
                    outlier_labels = detector.predict(X_scaled)
                    outlier_scores[method_name] = outlier_labels == -1
            except Exception as e:
                log.warning(
                    "detector_failed", method=method_name, error=str(e)
                )
                continue

        # Consensus outlier detection
        consensus_outliers = []
        for i in range(len(dataset)):
            outlier_votes = sum(
                int(scores[i]) for scores in outlier_scores.values()
            )

            # Require minimum consensus
            if outlier_votes >= min_consensus:
                methods_flagged = [
                    method
                    for method, scores in outlier_scores.items()
                    if scores[i]
                ]

                consensus_outliers.append(
                    {
                        "index": i,
                        "data_point": dataset.iloc[i].to_dict(),
                        "outlier_votes": outlier_votes,
                        "methods_flagged": methods_flagged,
                        "consensus_strength": outlier_votes / len(outlier_scores),
                    }
                )

        outlier_rate = len(consensus_outliers) / len(dataset) if dataset else 0

        log.info(
            "outlier_detection_completed",
            total_records=len(dataset),
            outliers_detected=len(consensus_outliers),
            outlier_rate=outlier_rate,
        )

        return {
            "dataset_name": dataset_name,
            "total_records": len(dataset),
            "features_analyzed": features,
            "detection_methods_used": list(outlier_scores.keys()),
            "outliers_detected": len(consensus_outliers),
            "outlier_rate": outlier_rate,
            "consensus_outliers": consensus_outliers[:100],  # Return first 100
        }

    async def _fetch_dataset(
        self, dataset_name: str, features: List[str]
    ) -> pd.DataFrame:
        """Fetch dataset from database for analysis.

        Args:
            dataset_name: Name of the table
            features: List of columns to fetch

        Returns:
            DataFrame with requested features
        """
        # Map dataset names to tables and queries
        table_queries = {
            "transactions": f"""
                SELECT {", ".join(features)}
                FROM transactions
                WHERE transaction_status = 'success'
                LIMIT 10000
            """,
            "wallets": f"""
                SELECT {", ".join(features)}
                FROM wallets
                LIMIT 10000
            """,
            "wallet_balances": f"""
                SELECT {", ".join(features)}
                FROM wallet_balances
                LIMIT 10000
            """,
        }

        query = table_queries.get(dataset_name)
        if not query:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query)
            return pd.DataFrame([dict(row) for row in rows])

    async def validate_trading_patterns(
        self, wallet_address: str
    ) -> Dict[str, Any]:
        """Validate trading patterns for statistical reasonableness.

        Args:
            wallet_address: Wallet address to validate

        Returns:
            Validation results for trading patterns
        """
        log = self.logger.bind(wallet_address=wallet_address)
        log.info("validating_trading_patterns")

        # Fetch wallet transactions
        transactions = await self._fetch_wallet_transactions(wallet_address)

        if len(transactions) == 0:
            log.warning("no_transactions_found")
            return {
                "wallet_address": wallet_address,
                "transaction_count": 0,
                "validation_status": "insufficient_data",
            }

        # Calculate trading pattern metrics
        df = pd.DataFrame(transactions)

        # Time-based metrics
        df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
        trading_period_days = (
            df["block_timestamp"].max() - df["block_timestamp"].min()
        ).days + 1

        pattern_metrics = {
            "trade_frequency": len(df) / max(trading_period_days, 1),
            "avg_trade_size_eth": float(
                df["eth_value_total_usd"].mean() if "eth_value_total_usd" in df else 0
            ),
            "trade_size_variance": float(
                df["eth_value_total_usd"].var() if "eth_value_total_usd" in df else 0
            ),
            "trade_size_std": float(
                df["eth_value_total_usd"].std() if "eth_value_total_usd" in df else 0
            ),
            "unique_tokens_traded": (
                len(set(df["token_in_address"]) | set(df["token_out_address"]))
                if "token_in_address" in df
                else 0
            ),
            "unique_tokens_ratio": (
                len(set(df["token_in_address"]) | set(df["token_out_address"]))
                / len(df)
                if len(df) > 0 and "token_in_address" in df
                else 0
            ),
        }

        # Gas efficiency
        if "gas_used" in df and "eth_value_total_usd" in df:
            df["gas_cost_eth"] = (
                df["gas_used"] * df["gas_price_gwei"] / 1e9
            )
            valid_trades = df[df["eth_value_total_usd"] > 0]
            if len(valid_trades) > 0:
                pattern_metrics["avg_gas_efficiency"] = float(
                    (valid_trades["eth_value_total_usd"] / valid_trades["gas_cost_eth"]).mean()
                )
            else:
                pattern_metrics["avg_gas_efficiency"] = 0
        else:
            pattern_metrics["avg_gas_efficiency"] = 0

        # Define reasonable ranges for each metric
        validation_ranges = {
            "trade_frequency": (0.01, 100),  # 0.01 to 100 trades per day
            "avg_trade_size_eth": (0.001, 10000),  # $0.001 to $10k ETH value
            "trade_size_variance": (0, 1000000),
            "trade_size_std": (0, 5000),
            "unique_tokens_ratio": (0.001, 1.0),
            "avg_gas_efficiency": (0.1, 100000),  # At least some efficiency
        }

        validation_results = {}
        for metric, value in pattern_metrics.items():
            if metric in validation_ranges:
                min_val, max_val = validation_ranges[metric]
                is_valid = min_val <= value <= max_val

                # Calculate z-score if we have enough data
                z_score = self._calculate_z_score_for_metric(metric, value)

                validation_results[metric] = {
                    "value": value,
                    "range": (min_val, max_val),
                    "valid": is_valid,
                    "z_score": z_score,
                }
            else:
                validation_results[metric] = {
                    "value": value,
                    "valid": True,
                }

        # Overall validation status
        all_valid = all(
            r.get("valid", True) for r in validation_results.values()
        )

        log.info(
            "trading_pattern_validation_completed",
            transaction_count=len(df),
            all_valid=all_valid,
        )

        return {
            "wallet_address": wallet_address,
            "transaction_count": len(df),
            "trading_period_days": trading_period_days,
            "pattern_metrics": pattern_metrics,
            "validation_results": validation_results,
            "validation_status": "pass" if all_valid else "fail",
        }

    async def _fetch_wallet_transactions(
        self, wallet_address: str
    ) -> List[Dict[str, Any]]:
        """Fetch all transactions for a wallet."""
        query = """
            SELECT
                tx_hash,
                block_timestamp,
                token_in_address,
                token_out_address,
                amount_in,
                amount_out,
                gas_used,
                gas_price_gwei,
                eth_value_total_usd
            FROM transactions
            WHERE wallet_address = $1
                AND transaction_status = 'success'
            ORDER BY block_timestamp ASC
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, wallet_address)
            return [dict(row) for row in rows]

    def _calculate_z_score_for_metric(
        self, metric_name: str, value: float
    ) -> Optional[float]:
        """Calculate z-score for a metric value.

        This would ideally compare against population statistics.
        For now, returns None as placeholder.
        """
        # TODO: Implement population statistics tracking
        return None

    async def detect_time_series_anomalies(
        self,
        metric_name: str,
        time_series_data: List[Dict[str, Any]],
        window_size: int = 7,
    ) -> Dict[str, Any]:
        """Detect anomalies in time series data.

        Args:
            metric_name: Name of the metric being analyzed
            time_series_data: List of {timestamp, value} dicts
            window_size: Rolling window size for anomaly detection

        Returns:
            Time series anomaly detection results
        """
        log = self.logger.bind(metric_name=metric_name)
        log.info("detecting_time_series_anomalies")

        if len(time_series_data) < window_size:
            log.warning("insufficient_data_points", data_points=len(time_series_data))
            return {
                "metric_name": metric_name,
                "data_points": len(time_series_data),
                "anomalies_detected": 0,
                "validation_status": "insufficient_data",
            }

        df = pd.DataFrame(time_series_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # Calculate rolling statistics
        df["rolling_mean"] = df["value"].rolling(window=window_size).mean()
        df["rolling_std"] = df["value"].rolling(window=window_size).std()

        # Detect anomalies using 3-sigma rule
        df["z_score"] = (
            df["value"] - df["rolling_mean"]
        ) / df["rolling_std"]
        df["is_anomaly"] = np.abs(df["z_score"]) > 3

        anomalies = df[df["is_anomaly"]].to_dict("records")

        log.info(
            "time_series_analysis_completed",
            data_points=len(df),
            anomalies_detected=len(anomalies),
        )

        return {
            "metric_name": metric_name,
            "data_points": len(df),
            "window_size": window_size,
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(df) if len(df) > 0 else 0,
            "anomalies": anomalies[:50],  # Return first 50
            "validation_status": (
                "pass" if len(anomalies) / len(df) < 0.05 else "warning"
            ),
        }

    async def validate_correlation_patterns(
        self, features: List[str], expected_correlations: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Validate correlation patterns between features.

        Args:
            features: List of features to analyze
            expected_correlations: Dict of expected correlation ranges

        Returns:
            Correlation validation results
        """
        log = self.logger.bind(feature_count=len(features))
        log.info("validating_correlations")

        # Fetch transaction data for correlation analysis
        dataset = await self._fetch_dataset("transactions", features)

        if len(dataset) < 30:
            log.warning("insufficient_data_for_correlation", rows=len(dataset))
            return {
                "features": features,
                "validation_status": "insufficient_data",
            }

        # Calculate correlation matrix
        correlation_matrix = dataset[features].corr()

        validation_results = {}
        for feature1, correlations in expected_correlations.items():
            for feature2, expected_corr in correlations.items():
                if feature1 in correlation_matrix.index and feature2 in correlation_matrix.columns:
                    actual_corr = correlation_matrix.loc[feature1, feature2]

                    # Check if correlation is within expected range
                    # Allow ±0.2 tolerance
                    is_valid = abs(actual_corr - expected_corr) <= 0.2

                    validation_results[f"{feature1}_vs_{feature2}"] = {
                        "expected_correlation": expected_corr,
                        "actual_correlation": float(actual_corr),
                        "difference": float(abs(actual_corr - expected_corr)),
                        "valid": is_valid,
                    }

        all_valid = all(r["valid"] for r in validation_results.values())

        log.info(
            "correlation_validation_completed",
            correlations_checked=len(validation_results),
            all_valid=all_valid,
        )

        return {
            "features": features,
            "correlations_validated": len(validation_results),
            "validation_results": validation_results,
            "validation_status": "pass" if all_valid else "fail",
        }

    async def detect_behavioral_anomalies(
        self, wallet_address: str
    ) -> Dict[str, Any]:
        """Detect behavioral anomalies in wallet activities.

        Args:
            wallet_address: Wallet address to analyze

        Returns:
            Behavioral anomaly detection results
        """
        log = self.logger.bind(wallet_address=wallet_address)
        log.info("detecting_behavioral_anomalies")

        transactions = await self._fetch_wallet_transactions(wallet_address)

        if len(transactions) < 10:
            log.warning("insufficient_transaction_history")
            return {
                "wallet_address": wallet_address,
                "validation_status": "insufficient_data",
            }

        df = pd.DataFrame(transactions)
        df["block_timestamp"] = pd.to_datetime(df["block_timestamp"])
        df = df.sort_values("block_timestamp")

        # Calculate time between transactions
        df["time_diff"] = df["block_timestamp"].diff().dt.total_seconds()

        anomalies = []

        # Check for suspicious patterns
        # 1. Sudden change in trading frequency
        recent_freq = len(df[df["block_timestamp"] > df["block_timestamp"].max() - pd.Timedelta(days=7)])
        historical_freq = len(df) / (
            (df["block_timestamp"].max() - df["block_timestamp"].min()).days + 1
        )

        if recent_freq > historical_freq * 5:
            anomalies.append(
                {
                    "type": "sudden_frequency_increase",
                    "recent_weekly_trades": recent_freq,
                    "historical_daily_avg": historical_freq,
                    "ratio": recent_freq / max(historical_freq, 0.1),
                }
            )

        # 2. Unusual trade size patterns
        if "eth_value_total_usd" in df:
            avg_trade = df["eth_value_total_usd"].mean()
            std_trade = df["eth_value_total_usd"].std()
            unusual_trades = df[
                np.abs(df["eth_value_total_usd"] - avg_trade) > 3 * std_trade
            ]

            if len(unusual_trades) > 0:
                anomalies.append(
                    {
                        "type": "unusual_trade_sizes",
                        "count": len(unusual_trades),
                        "avg_trade_size": float(avg_trade),
                        "unusual_sizes": unusual_trades["eth_value_total_usd"]
                        .tolist()[:5],
                    }
                )

        # 3. Burst trading (multiple trades in short time)
        burst_threshold = 300  # 5 minutes in seconds
        if "time_diff" in df:
            bursts = df[df["time_diff"] < burst_threshold]
            if len(bursts) > len(df) * 0.2:  # More than 20% are burst trades
                anomalies.append(
                    {
                        "type": "burst_trading_pattern",
                        "burst_trades": len(bursts),
                        "total_trades": len(df),
                        "burst_ratio": len(bursts) / len(df),
                    }
                )

        log.info(
            "behavioral_analysis_completed",
            transaction_count=len(df),
            anomalies_detected=len(anomalies),
        )

        return {
            "wallet_address": wallet_address,
            "transaction_count": len(df),
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "validation_status": "pass" if len(anomalies) == 0 else "warning",
        }