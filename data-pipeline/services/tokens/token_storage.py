"""
Token storage service for database operations.

Implements:
- UPSERT operations using ON CONFLICT for idempotency
- Proper NOT NULL and CHECK constraints for data integrity
- Indexes on token_address (PK) and market_cap_rank for performance
- Batch processing for database writes (100-200 records per batch)
- Transaction management for atomic operations
- Constraint violations and error handling
"""

import time
from typing import List, Dict, Any, Optional
from decimal import Decimal

from data_collection.common.db import get_cursor, execute_with_retry, batch_insert
from data_collection.common.logging_setup import get_logger
from .token_fetcher import TokenMetadata


class TokenStorageError(Exception):
    """Base exception for token storage errors"""
    pass


class TokenStorage:
    """
    Handles database storage operations for token metadata.
    """

    def __init__(self):
        self.logger = get_logger("token_storage")

    def store_tokens(self, tokens: List[TokenMetadata], batch_size: int = 150) -> Dict[str, int]:
        """
        Store token metadata with batch processing and atomic operations.

        Args:
            tokens: List of validated token metadata
            batch_size: Number of records per batch (default: 150)

        Returns:
            Dictionary with statistics: {"inserted": int, "updated": int, "skipped": int}
        """
        if not tokens:
            return {"inserted": 0, "updated": 0, "skipped": 0}

        start_time = time.time()
        stats = {"inserted": 0, "updated": 0, "skipped": 0}

        self.logger.log_operation(
            operation="store_tokens",
            params={"token_count": len(tokens), "batch_size": batch_size},
            status="started",
            message=f"Starting storage of {len(tokens)} tokens"
        )

        try:
            # Process tokens in batches for better performance
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i + batch_size]
                batch_stats = self._store_token_batch(batch)

                # Aggregate statistics
                for key in stats:
                    stats[key] += batch_stats[key]

                self.logger.log_operation(
                    operation="store_batch",
                    params={
                        "batch_number": (i // batch_size) + 1,
                        "batch_size": len(batch),
                        "inserted": batch_stats["inserted"],
                        "updated": batch_stats["updated"]
                    },
                    status="completed",
                    message=f"Batch {(i // batch_size) + 1} processed"
                )

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.log_operation(
                operation="store_tokens",
                params={
                    "total_tokens": len(tokens),
                    "inserted": stats["inserted"],
                    "updated": stats["updated"],
                    "skipped": stats["skipped"]
                },
                status="completed",
                duration_ms=duration_ms,
                message=f"Token storage completed: {stats['inserted']} inserted, {stats['updated']} updated"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="store_tokens",
                status="error",
                error=str(e),
                duration_ms=duration_ms,
                message="Token storage failed"
            )
            raise TokenStorageError(f"Failed to store tokens: {str(e)}") from e

    def _store_token_batch(self, tokens: List[TokenMetadata]) -> Dict[str, int]:
        """Store a batch of tokens with atomic transaction management"""
        batch_stats = {"inserted": 0, "updated": 0, "skipped": 0}

        try:
            with get_cursor() as cur:
                for token in tokens:
                    try:
                        # Convert token to database record
                        db_record = self._token_to_db_record(token)

                        # Perform UPSERT operation
                        result = self._upsert_token(cur, db_record)
                        batch_stats[result] += 1

                    except Exception as e:
                        self.logger.log_operation(
                            operation="store_token_record",
                            params={"token_address": token.token_address[:10] + "..."},
                            status="error",
                            error=str(e),
                            message="Failed to store individual token record"
                        )
                        batch_stats["skipped"] += 1

            return batch_stats

        except Exception as e:
            self.logger.log_operation(
                operation="store_batch",
                status="error",
                error=str(e),
                message="Batch storage transaction failed"
            )
            raise

    def _token_to_db_record(self, token: TokenMetadata) -> Dict[str, Any]:
        """Convert TokenMetadata to database record format"""
        # Map to the existing schema structure
        record = {
            "token_address": token.token_address,
            "symbol": token.symbol,
            "name": token.name,
            "decimals": token.decimals or 18,  # Default to 18 if missing
            "market_cap_rank": token.market_cap_rank,
            "avg_daily_volume_usd": float(token.volume_24h_usd) if token.volume_24h_usd else None,
            "narrative_category": None,  # To be filled in Story 1.4
            "liquidity_tier": None,  # To be filled in Story 1.3
        }

        # Validate record against constraints
        self._validate_record(record)
        return record

    def _validate_record(self, record: Dict[str, Any]) -> None:
        """Validate record against database constraints"""
        # Check required fields
        required_fields = ["token_address", "symbol", "name", "decimals"]
        for field in required_fields:
            if record.get(field) is None:
                raise ValueError(f"Required field '{field}' is missing")

        # Validate token address format
        if not (isinstance(record["token_address"], str) and
                len(record["token_address"]) == 42 and
                record["token_address"].startswith("0x")):
            raise ValueError(f"Invalid token address format: {record['token_address']}")

        # Validate decimals range
        decimals = record["decimals"]
        if not (isinstance(decimals, int) and 0 <= decimals <= 18):
            raise ValueError(f"Decimals must be between 0 and 18, got: {decimals}")

        # Validate market cap rank if present
        if record.get("market_cap_rank") is not None:
            rank = record["market_cap_rank"]
            if not (isinstance(rank, int) and rank > 0):
                raise ValueError(f"Market cap rank must be positive integer, got: {rank}")

        # Validate volume if present
        if record.get("avg_daily_volume_usd") is not None:
            volume = record["avg_daily_volume_usd"]
            if not (isinstance(volume, (int, float)) and volume >= 0):
                raise ValueError(f"Daily volume must be non-negative, got: {volume}")

    def _upsert_token(self, cursor, record: Dict[str, Any]) -> str:
        """
        Perform UPSERT operation for a single token.

        Returns:
            "inserted" or "updated" depending on the operation performed
        """
        # First, try to check if record exists
        check_query = "SELECT token_address, updated_at FROM tokens WHERE token_address = %s"
        cursor.execute(check_query, (record["token_address"],))
        existing = cursor.fetchone()

        # Prepare UPSERT query
        columns = list(record.keys())
        placeholders = ["%s"] * len(columns)
        update_columns = [f"{col} = EXCLUDED.{col}" for col in columns if col != "token_address"]

        upsert_query = f"""
            INSERT INTO tokens ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (token_address)
            DO UPDATE SET
                {', '.join(update_columns)},
                updated_at = NOW()
            RETURNING (xmax = 0) AS inserted
        """

        cursor.execute(upsert_query, list(record.values()))
        result = cursor.fetchone()

        # Return whether this was an insert or update
        return "inserted" if result["inserted"] else "updated"

    def get_token_count(self) -> int:
        """Get total number of tokens in the database"""
        try:
            result = execute_with_retry("SELECT COUNT(*) as count FROM tokens")
            return result[0]["count"] if result else 0
        except Exception as e:
            self.logger.log_operation(
                operation="get_token_count",
                status="error",
                error=str(e)
            )
            return 0

    def get_tokens_by_rank_range(self, min_rank: int = 1, max_rank: int = 500) -> List[Dict[str, Any]]:
        """Get tokens within a specific market cap rank range"""
        query = """
            SELECT token_address, symbol, name, decimals, market_cap_rank,
                   avg_daily_volume_usd, narrative_category, liquidity_tier,
                   created_at, updated_at
            FROM tokens
            WHERE market_cap_rank BETWEEN %s AND %s
            ORDER BY market_cap_rank ASC
        """

        try:
            result = execute_with_retry(query, (min_rank, max_rank))
            return result or []
        except Exception as e:
            self.logger.log_operation(
                operation="get_tokens_by_rank",
                params={"min_rank": min_rank, "max_rank": max_rank},
                status="error",
                error=str(e)
            )
            return []

    def validate_database_constraints(self) -> Dict[str, Any]:
        """
        Validate database integrity and return a summary report.

        Returns:
            Dictionary with validation results
        """
        start_time = time.time()
        validation_report = {
            "total_tokens": 0,
            "constraint_violations": [],
            "statistics": {},
            "validation_passed": True
        }

        try:
            # Get basic statistics
            stats_queries = {
                "total_tokens": "SELECT COUNT(*) as count FROM tokens",
                "tokens_with_rank": "SELECT COUNT(*) as count FROM tokens WHERE market_cap_rank IS NOT NULL",
                "tokens_with_volume": "SELECT COUNT(*) as count FROM tokens WHERE avg_daily_volume_usd IS NOT NULL",
                "unique_symbols": "SELECT COUNT(DISTINCT symbol) as count FROM tokens",
                "duplicate_symbols": """
                    SELECT symbol, COUNT(*) as count
                    FROM tokens
                    GROUP BY symbol
                    HAVING COUNT(*) > 1
                    ORDER BY count DESC
                    LIMIT 10
                """
            }

            for key, query in stats_queries.items():
                if key == "duplicate_symbols":
                    validation_report["statistics"][key] = execute_with_retry(query) or []
                else:
                    result = execute_with_retry(query)
                    validation_report["statistics"][key] = result[0]["count"] if result else 0

            validation_report["total_tokens"] = validation_report["statistics"]["total_tokens"]

            # Check for constraint violations
            constraint_checks = [
                ("invalid_decimals", "SELECT COUNT(*) as count FROM tokens WHERE decimals < 0 OR decimals > 18"),
                ("invalid_ranks", "SELECT COUNT(*) as count FROM tokens WHERE market_cap_rank IS NOT NULL AND market_cap_rank <= 0"),
                ("invalid_volumes", "SELECT COUNT(*) as count FROM tokens WHERE avg_daily_volume_usd IS NOT NULL AND avg_daily_volume_usd < 0"),
                ("invalid_addresses", "SELECT COUNT(*) as count FROM tokens WHERE LENGTH(token_address) != 42 OR NOT token_address LIKE '0x%'"),
                ("missing_required", "SELECT COUNT(*) as count FROM tokens WHERE symbol IS NULL OR name IS NULL OR decimals IS NULL")
            ]

            for check_name, query in constraint_checks:
                result = execute_with_retry(query)
                violation_count = result[0]["count"] if result else 0
                if violation_count > 0:
                    validation_report["constraint_violations"].append({
                        "type": check_name,
                        "count": violation_count
                    })
                    validation_report["validation_passed"] = False

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.log_operation(
                operation="validate_constraints",
                params={
                    "total_tokens": validation_report["total_tokens"],
                    "violations": len(validation_report["constraint_violations"])
                },
                status="completed" if validation_report["validation_passed"] else "warning",
                duration_ms=duration_ms,
                message=f"Database validation completed: {len(validation_report['constraint_violations'])} constraint violations found"
            )

            return validation_report

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="validate_constraints",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            validation_report["validation_passed"] = False
            validation_report["error"] = str(e)
            return validation_report