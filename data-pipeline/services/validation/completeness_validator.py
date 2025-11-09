"""Data completeness and coverage validation.

This module implements comprehensive validation of data completeness,
coverage metrics, and missing data pattern analysis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog
from asyncpg import Pool

logger = structlog.get_logger()


class CompletenessValidator:
    """Data completeness and coverage validation framework."""

    def __init__(self, db_pool: Pool, target_completeness: float = 0.95):
        """Initialize completeness validator.

        Args:
            db_pool: Async database connection pool
            target_completeness: Target completeness threshold (default 95%)
        """
        self.db = db_pool
        self.target_completeness = target_completeness
        self.logger = logger.bind(component="completeness_validator")

    async def validate_dataset_completeness(
        self, dataset_name: str, table_name: str
    ) -> Dict[str, Any]:
        """Comprehensive completeness validation for dataset.

        Args:
            dataset_name: Logical name of dataset
            table_name: Database table name

        Returns:
            Completeness validation report
        """
        log = self.logger.bind(dataset_name=dataset_name, table_name=table_name)
        log.info("validating_dataset_completeness")

        # Get table schema and required fields
        schema = await self._get_table_schema(table_name)
        required_fields = [
            field["name"] for field in schema if not field["nullable"]
        ]

        # Get total record count
        total_records = await self._count_total_records(table_name)

        if total_records == 0:
            log.warning("no_records_found")
            return {
                "dataset_name": dataset_name,
                "table_name": table_name,
                "total_records": 0,
                "validation_status": "no_data",
            }

        completeness_report = {
            "dataset_name": dataset_name,
            "table_name": table_name,
            "total_records": total_records,
            "field_completeness": {},
            "overall_completeness": 0,
            "meets_target": False,
        }

        # Check completeness for each field
        all_fields = [field["name"] for field in schema]
        for field_name in all_fields:
            non_null_count = await self._count_non_null_values(
                table_name, field_name
            )
            field_completeness = (
                non_null_count / total_records if total_records > 0 else 0
            )

            is_required = field_name in required_fields

            completeness_report["field_completeness"][field_name] = {
                "non_null_count": non_null_count,
                "null_count": total_records - non_null_count,
                "total_count": total_records,
                "completeness_rate": field_completeness,
                "required": is_required,
                "meets_target": (
                    field_completeness >= self.target_completeness
                    if is_required
                    else True
                ),
            }

        # Calculate overall completeness (average of required fields)
        required_field_rates = [
            fc["completeness_rate"]
            for field, fc in completeness_report["field_completeness"].items()
            if fc["required"]
        ]

        if required_field_rates:
            completeness_report["overall_completeness"] = sum(
                required_field_rates
            ) / len(required_field_rates)
            completeness_report["meets_target"] = (
                completeness_report["overall_completeness"]
                >= self.target_completeness
            )
        else:
            completeness_report["overall_completeness"] = 1.0
            completeness_report["meets_target"] = True

        log.info(
            "completeness_validation_completed",
            overall_completeness=completeness_report["overall_completeness"],
            meets_target=completeness_report["meets_target"],
        )

        return completeness_report

    async def _get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information.

        Args:
            table_name: Name of the table

        Returns:
            List of field definitions with name and nullable flag
        """
        query = """
            SELECT
                column_name as name,
                data_type,
                is_nullable = 'YES' as nullable
            FROM information_schema.columns
            WHERE table_name = $1
                AND table_schema = 'public'
            ORDER BY ordinal_position
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, table_name)
            return [dict(row) for row in rows]

    async def _count_total_records(self, table_name: str) -> int:
        """Count total records in table."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)
            return row["count"]

    async def _count_non_null_values(
        self, table_name: str, field_name: str
    ) -> int:
        """Count non-null values for a field."""
        query = f"""
            SELECT COUNT({field_name}) as count
            FROM {table_name}
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)
            return row["count"]

    async def identify_missing_data_patterns(
        self, table_name: str, categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identify patterns in missing data.

        Args:
            table_name: Name of the table
            categorical_columns: List of categorical columns to analyze

        Returns:
            Missing data pattern analysis
        """
        log = self.logger.bind(table_name=table_name)
        log.info("identifying_missing_data_patterns")

        # Fetch sample of data for analysis
        query = f"""
            SELECT *
            FROM {table_name}
            LIMIT 10000
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query)

        if len(rows) == 0:
            log.warning("no_data_available")
            return {"table_name": table_name, "patterns": {}}

        df = pd.DataFrame([dict(row) for row in rows])

        missing_patterns = {}

        # Overall missing data statistics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_patterns["overall"] = {
            "total_cells": int(total_cells),
            "missing_cells": int(missing_cells),
            "missing_rate": float(missing_cells / total_cells)
            if total_cells > 0
            else 0,
        }

        # Analyze missing data by time periods (if timestamp column exists)
        if "block_timestamp" in df.columns or "snapshot_date" in df.columns:
            time_col = (
                "block_timestamp"
                if "block_timestamp" in df.columns
                else "snapshot_date"
            )
            df[time_col] = pd.to_datetime(df[time_col])
            df["date"] = df[time_col].dt.date

            missing_by_date = (
                df.groupby("date")
                .apply(lambda x: x.isnull().sum().sum() / x.size)
                .to_dict()
            )

            missing_patterns["temporal"] = {
                str(k): float(v) for k, v in missing_by_date.items()
            }

        # Analyze missing data by categorical variables
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    missing_by_category = (
                        df.groupby(col)
                        .apply(lambda x: x.isnull().sum().sum() / x.size)
                        .to_dict()
                    )
                    missing_patterns[f"by_{col}"] = {
                        str(k): float(v) for k, v in missing_by_category.items()
                    }

        # Identify columns with most missing data
        missing_by_column = df.isnull().sum() / len(df)
        top_missing_columns = (
            missing_by_column[missing_by_column > 0]
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        missing_patterns["top_missing_columns"] = {
            str(k): float(v) for k, v in top_missing_columns.items()
        }

        log.info(
            "missing_pattern_analysis_completed",
            overall_missing_rate=missing_patterns["overall"]["missing_rate"],
        )

        return {
            "table_name": table_name,
            "sample_size": len(df),
            "patterns": missing_patterns,
        }

    async def validate_data_coverage(
        self,
        table_name: str,
        expected_date_range: tuple[datetime, datetime],
        date_column: str = "block_timestamp",
    ) -> Dict[str, Any]:
        """Validate temporal coverage of data.

        Args:
            table_name: Name of the table
            expected_date_range: Expected (start_date, end_date)
            date_column: Name of date/timestamp column

        Returns:
            Coverage validation results
        """
        log = self.logger.bind(table_name=table_name)
        log.info("validating_data_coverage")

        expected_start, expected_end = expected_date_range

        # Get actual date range in data
        query = f"""
            SELECT
                MIN({date_column}) as min_date,
                MAX({date_column}) as max_date,
                COUNT(*) as record_count
            FROM {table_name}
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)

        if row["record_count"] == 0:
            log.warning("no_data_available")
            return {
                "table_name": table_name,
                "validation_status": "no_data",
            }

        actual_start = row["min_date"]
        actual_end = row["max_date"]

        # Check coverage
        has_expected_start = actual_start <= expected_start
        has_expected_end = actual_end >= expected_end

        # Calculate coverage percentage
        expected_duration = (expected_end - expected_start).days
        actual_duration = (actual_end - actual_start).days

        coverage_percentage = min(
            1.0, actual_duration / expected_duration if expected_duration > 0 else 0
        )

        # Identify gaps in daily coverage
        gaps = await self._identify_temporal_gaps(
            table_name, date_column, expected_start, expected_end
        )

        validation_status = "pass"
        if not has_expected_start or not has_expected_end:
            validation_status = "partial_coverage"
        if coverage_percentage < 0.9:
            validation_status = "insufficient_coverage"
        if len(gaps) > 0:
            validation_status = "has_gaps"

        log.info(
            "coverage_validation_completed",
            coverage_percentage=coverage_percentage,
            gap_count=len(gaps),
            validation_status=validation_status,
        )

        return {
            "table_name": table_name,
            "expected_range": (
                expected_start.isoformat(),
                expected_end.isoformat(),
            ),
            "actual_range": (actual_start.isoformat(), actual_end.isoformat()),
            "has_expected_start": has_expected_start,
            "has_expected_end": has_expected_end,
            "coverage_percentage": coverage_percentage,
            "record_count": row["record_count"],
            "gaps_detected": len(gaps),
            "gaps": gaps[:20],  # Return first 20 gaps
            "validation_status": validation_status,
        }

    async def _identify_temporal_gaps(
        self,
        table_name: str,
        date_column: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Identify gaps in temporal coverage.

        Args:
            table_name: Name of the table
            date_column: Name of date column
            start_date: Expected start date
            end_date: Expected end date

        Returns:
            List of identified gaps
        """
        # Get dates with data
        query = f"""
            SELECT DISTINCT DATE({date_column}) as data_date
            FROM {table_name}
            WHERE {date_column} >= $1
                AND {date_column} <= $2
            ORDER BY data_date
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, start_date, end_date)

        dates_with_data = set(row["data_date"] for row in rows)

        # Generate expected date range
        current_date = start_date.date()
        end_date_only = end_date.date()
        expected_dates = []

        while current_date <= end_date_only:
            expected_dates.append(current_date)
            current_date += timedelta(days=1)

        # Find missing dates
        missing_dates = [d for d in expected_dates if d not in dates_with_data]

        # Group consecutive missing dates into gaps
        gaps = []
        if missing_dates:
            gap_start = missing_dates[0]
            gap_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - gap_end).days == 1:
                    gap_end = missing_dates[i]
                else:
                    gaps.append(
                        {
                            "start_date": gap_start.isoformat(),
                            "end_date": gap_end.isoformat(),
                            "duration_days": (gap_end - gap_start).days + 1,
                        }
                    )
                    gap_start = missing_dates[i]
                    gap_end = missing_dates[i]

            # Add final gap
            gaps.append(
                {
                    "start_date": gap_start.isoformat(),
                    "end_date": gap_end.isoformat(),
                    "duration_days": (gap_end - gap_start).days + 1,
                }
            )

        return gaps

    async def validate_data_freshness(
        self,
        table_name: str,
        date_column: str,
        max_staleness_hours: int = 24,
    ) -> Dict[str, Any]:
        """Validate data freshness (how recent is the data).

        Args:
            table_name: Name of the table
            date_column: Name of date/timestamp column
            max_staleness_hours: Maximum acceptable staleness in hours

        Returns:
            Freshness validation results
        """
        log = self.logger.bind(table_name=table_name)
        log.info("validating_data_freshness")

        query = f"""
            SELECT MAX({date_column}) as latest_timestamp
            FROM {table_name}
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)

        if not row["latest_timestamp"]:
            log.warning("no_timestamp_data")
            return {
                "table_name": table_name,
                "validation_status": "no_data",
            }

        latest_timestamp = row["latest_timestamp"]
        current_time = datetime.now()

        staleness = current_time - latest_timestamp
        staleness_hours = staleness.total_seconds() / 3600

        is_fresh = staleness_hours <= max_staleness_hours

        log.info(
            "freshness_validation_completed",
            staleness_hours=staleness_hours,
            is_fresh=is_fresh,
        )

        return {
            "table_name": table_name,
            "latest_timestamp": latest_timestamp.isoformat(),
            "current_time": current_time.isoformat(),
            "staleness_hours": staleness_hours,
            "max_staleness_hours": max_staleness_hours,
            "is_fresh": is_fresh,
            "validation_status": "pass" if is_fresh else "stale",
        }

    async def validate_sample_data(
        self,
        table_name: str,
        sample_size: int = 1000,
        validation_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Validate data quality using statistical sampling.

        Args:
            table_name: Name of the table
            sample_size: Number of records to sample
            validation_rules: List of validation rules to apply

        Returns:
            Sample-based validation results
        """
        log = self.logger.bind(table_name=table_name, sample_size=sample_size)
        log.info("validating_sample_data")

        # Fetch random sample
        query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, sample_size)

        if len(rows) == 0:
            log.warning("no_data_available")
            return {
                "table_name": table_name,
                "validation_status": "no_data",
            }

        df = pd.DataFrame([dict(row) for row in rows])

        validation_results = []

        # Apply default validation rules if none provided
        if not validation_rules:
            validation_rules = [
                {
                    "name": "no_duplicates",
                    "check": "duplicates",
                },
                {
                    "name": "no_nulls_in_key_fields",
                    "check": "nulls",
                    "fields": ["id", "address", "timestamp"],
                },
            ]

        for rule in validation_rules:
            if rule["check"] == "duplicates":
                duplicate_count = df.duplicated().sum()
                validation_results.append(
                    {
                        "rule": rule["name"],
                        "passed": duplicate_count == 0,
                        "duplicate_count": int(duplicate_count),
                        "duplicate_rate": float(duplicate_count / len(df)),
                    }
                )

            elif rule["check"] == "nulls":
                fields = rule.get("fields", [])
                null_counts = {}
                for field in fields:
                    if field in df.columns:
                        null_count = df[field].isnull().sum()
                        null_counts[field] = int(null_count)

                total_nulls = sum(null_counts.values())
                validation_results.append(
                    {
                        "rule": rule["name"],
                        "passed": total_nulls == 0,
                        "null_counts": null_counts,
                        "total_nulls": total_nulls,
                    }
                )

        all_passed = all(r["passed"] for r in validation_results)

        log.info(
            "sample_validation_completed",
            sample_size=len(df),
            rules_applied=len(validation_rules),
            all_passed=all_passed,
        )

        return {
            "table_name": table_name,
            "sample_size": len(df),
            "rules_applied": len(validation_rules),
            "validation_results": validation_results,
            "validation_status": "pass" if all_passed else "fail",
        }