"""Database integrity and constraint validation.

This module validates referential integrity, constraints, and data consistency
across all database tables.
"""

from typing import Any, Dict, List

import structlog
from asyncpg import Pool

logger = structlog.get_logger()


class DatabaseIntegrityValidator:
    """Database integrity and constraint validation framework."""

    def __init__(self, db_pool: Pool):
        """Initialize database integrity validator.

        Args:
            db_pool: Async database connection pool
        """
        self.db = db_pool
        self.logger = logger.bind(component="database_integrity_validator")

    async def validate_all_integrity(self) -> Dict[str, Any]:
        """Run all integrity validations.

        Returns:
            Complete integrity validation report
        """
        log = self.logger
        log.info("validating_all_database_integrity")

        # Validate referential integrity
        ref_integrity = await self.validate_referential_integrity()

        # Validate data constraints
        constraints = await self.validate_data_constraints()

        # Detect duplicate records
        duplicates = await self.detect_duplicate_records()

        # Detect orphaned records
        orphans = await self.detect_orphaned_records()

        # Calculate overall integrity score
        integrity_metrics = {
            "referential_integrity_score": ref_integrity["overall_score"],
            "constraint_compliance_score": constraints["overall_score"],
            "duplicate_penalty": min(0.2, duplicates["total_duplicates"] * 0.001),
            "orphan_penalty": min(0.2, orphans["total_orphans"] * 0.001),
        }

        overall_score = (
            integrity_metrics["referential_integrity_score"] * 0.4
            + integrity_metrics["constraint_compliance_score"] * 0.4
            - integrity_metrics["duplicate_penalty"]
            - integrity_metrics["orphan_penalty"]
        )
        overall_score = max(0, min(1, overall_score))

        log.info(
            "integrity_validation_completed",
            overall_score=overall_score,
        )

        return {
            "referential_integrity": ref_integrity,
            "constraints": constraints,
            "duplicates": duplicates,
            "orphaned_records": orphans,
            "integrity_metrics": integrity_metrics,
            "overall_integrity_score": overall_score,
            "validation_status": "pass" if overall_score > 0.95 else "fail",
        }

    async def validate_referential_integrity(self) -> Dict[str, Any]:
        """Validate all foreign key relationships.

        Returns:
            Referential integrity validation results
        """
        log = self.logger
        log.info("validating_referential_integrity")

        integrity_results = {}

        # Define foreign key relationships to validate
        fk_relationships = [
            (
                "transactions",
                "wallet_address",
                "wallets",
                "wallet_address",
            ),
            (
                "wallet_balances",
                "wallet_address",
                "wallets",
                "wallet_address",
            ),
            (
                "wallet_balances",
                "token_address",
                "tokens",
                "token_address",
            ),
        ]

        for child_table, child_col, parent_table, parent_col in fk_relationships:
            result = await self._validate_fk_relationship(
                child_table, child_col, parent_table, parent_col
            )
            key = f"{child_table}.{child_col} -> {parent_table}.{parent_col}"
            integrity_results[key] = result

        # Calculate overall score
        all_valid = all(r["is_valid"] for r in integrity_results.values())
        total_orphans = sum(
            r["orphaned_count"] for r in integrity_results.values()
        )

        overall_score = 1.0 if all_valid else max(0, 1 - (total_orphans * 0.001))

        log.info(
            "referential_integrity_completed",
            relationships_checked=len(fk_relationships),
            all_valid=all_valid,
            overall_score=overall_score,
        )

        return {
            "relationships_checked": len(fk_relationships),
            "results": integrity_results,
            "all_valid": all_valid,
            "total_orphaned_records": total_orphans,
            "overall_score": overall_score,
        }

    async def _validate_fk_relationship(
        self,
        child_table: str,
        child_column: str,
        parent_table: str,
        parent_column: str,
    ) -> Dict[str, Any]:
        """Validate a single foreign key relationship."""
        # Check for orphaned records
        query = f"""
            SELECT COUNT(*) as orphaned_count
            FROM {child_table} c
            LEFT JOIN {parent_table} p ON c.{child_column} = p.{parent_column}
            WHERE p.{parent_column} IS NULL
                AND c.{child_column} IS NOT NULL
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)
            orphaned_count = row["orphaned_count"]

        # Get total count for percentage
        total_query = f"SELECT COUNT(*) as total FROM {child_table}"
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(total_query)
            total_count = row["total"]

        integrity_rate = (
            1 - (orphaned_count / total_count) if total_count > 0 else 1.0
        )

        return {
            "child_table": child_table,
            "parent_table": parent_table,
            "orphaned_count": orphaned_count,
            "total_records": total_count,
            "integrity_rate": integrity_rate,
            "is_valid": orphaned_count == 0,
        }

    async def validate_data_constraints(self) -> Dict[str, Any]:
        """Validate all CHECK constraints and data rules.

        Returns:
            Constraint validation results
        """
        log = self.logger
        log.info("validating_data_constraints")

        constraint_results = {}

        # Token constraints
        token_constraints = await self._validate_token_constraints()
        constraint_results["tokens"] = token_constraints

        # Transaction constraints
        transaction_constraints = await self._validate_transaction_constraints()
        constraint_results["transactions"] = transaction_constraints

        # Wallet constraints
        wallet_constraints = await self._validate_wallet_constraints()
        constraint_results["wallets"] = wallet_constraints

        # Balance constraints
        balance_constraints = await self._validate_balance_constraints()
        constraint_results["wallet_balances"] = balance_constraints

        # Calculate overall score
        all_violations = sum(
            sum(c["violations"] for c in table_results.values())
            for table_results in constraint_results.values()
        )

        overall_score = max(0, 1 - (all_violations * 0.0001))

        log.info(
            "constraint_validation_completed",
            total_violations=all_violations,
            overall_score=overall_score,
        )

        return {
            "results": constraint_results,
            "total_violations": all_violations,
            "overall_score": overall_score,
            "validation_status": "pass" if all_violations == 0 else "fail",
        }

    async def _validate_token_constraints(self) -> Dict[str, Any]:
        """Validate token-specific constraints."""
        constraints = {
            "decimals_range": """
                SELECT COUNT(*) as count
                FROM tokens
                WHERE decimals < 0 OR decimals > 18
            """,
            "positive_market_cap_rank": """
                SELECT COUNT(*) as count
                FROM tokens
                WHERE market_cap_rank IS NOT NULL AND market_cap_rank <= 0
            """,
            "valid_token_address": """
                SELECT COUNT(*) as count
                FROM tokens
                WHERE token_address IS NULL
                    OR LENGTH(token_address) != 42
                    OR token_address NOT LIKE '0x%'
            """,
        }

        results = {}
        async with self.db.acquire() as conn:
            for name, query in constraints.items():
                row = await conn.fetchrow(query)
                violations = row["count"]
                results[name] = {
                    "violations": violations,
                    "is_valid": violations == 0,
                }

        return results

    async def _validate_transaction_constraints(self) -> Dict[str, Any]:
        """Validate transaction-specific constraints."""
        constraints = {
            "positive_amounts": """
                SELECT COUNT(*) as count
                FROM transactions
                WHERE (amount_in < 0 OR amount_out < 0)
                    AND transaction_status = 'success'
            """,
            "positive_gas": """
                SELECT COUNT(*) as count
                FROM transactions
                WHERE gas_used < 0 OR gas_price_gwei < 0
            """,
            "valid_tx_hash": """
                SELECT COUNT(*) as count
                FROM transactions
                WHERE tx_hash IS NULL
                    OR LENGTH(tx_hash) != 66
                    OR tx_hash NOT LIKE '0x%'
            """,
            "valid_block_number": """
                SELECT COUNT(*) as count
                FROM transactions
                WHERE block_number <= 0
            """,
            "valid_timestamp": """
                SELECT COUNT(*) as count
                FROM transactions
                WHERE block_timestamp IS NULL
                    OR block_timestamp < '2015-01-01'::timestamp
                    OR block_timestamp > NOW() + INTERVAL '1 day'
            """,
        }

        results = {}
        async with self.db.acquire() as conn:
            for name, query in constraints.items():
                row = await conn.fetchrow(query)
                violations = row["count"]
                results[name] = {
                    "violations": violations,
                    "is_valid": violations == 0,
                }

        return results

    async def _validate_wallet_constraints(self) -> Dict[str, Any]:
        """Validate wallet-specific constraints."""
        constraints = {
            "valid_addresses": """
                SELECT COUNT(*) as count
                FROM wallets
                WHERE wallet_address IS NULL
                    OR LENGTH(wallet_address) != 42
                    OR wallet_address NOT LIKE '0x%'
            """,
            "logical_dates": """
                SELECT COUNT(*) as count
                FROM wallets
                WHERE first_seen_date > last_active_date
            """,
            "positive_transaction_count": """
                SELECT COUNT(*) as count
                FROM wallets
                WHERE total_transactions < 0
            """,
        }

        results = {}
        async with self.db.acquire() as conn:
            for name, query in constraints.items():
                row = await conn.fetchrow(query)
                violations = row["count"]
                results[name] = {
                    "violations": violations,
                    "is_valid": violations == 0,
                }

        return results

    async def _validate_balance_constraints(self) -> Dict[str, Any]:
        """Validate balance-specific constraints."""
        constraints = {
            "positive_balance": """
                SELECT COUNT(*) as count
                FROM wallet_balances
                WHERE balance < 0
            """,
            "valid_snapshot_date": """
                SELECT COUNT(*) as count
                FROM wallet_balances
                WHERE snapshot_date IS NULL
                    OR snapshot_date < '2015-01-01'::date
                    OR snapshot_date > CURRENT_DATE + INTERVAL '1 day'
            """,
        }

        results = {}
        async with self.db.acquire() as conn:
            for name, query in constraints.items():
                row = await conn.fetchrow(query)
                violations = row["count"]
                results[name] = {
                    "violations": violations,
                    "is_valid": violations == 0,
                }

        return results

    async def detect_duplicate_records(self) -> Dict[str, Any]:
        """Detect duplicate records across all tables.

        Returns:
            Duplicate detection results
        """
        log = self.logger
        log.info("detecting_duplicate_records")

        duplicate_results = {}

        # Check transactions for duplicates (by tx_hash)
        transactions_dupes = await self._check_duplicates(
            "transactions", "tx_hash"
        )
        duplicate_results["transactions"] = transactions_dupes

        # Check wallets for duplicates (by wallet_address)
        wallets_dupes = await self._check_duplicates("wallets", "wallet_address")
        duplicate_results["wallets"] = wallets_dupes

        # Check tokens for duplicates (by token_address)
        tokens_dupes = await self._check_duplicates("tokens", "token_address")
        duplicate_results["tokens"] = tokens_dupes

        # Check wallet_balances for duplicates (by composite key)
        balances_dupes = await self._check_balance_duplicates()
        duplicate_results["wallet_balances"] = balances_dupes

        total_duplicates = sum(r["duplicate_count"] for r in duplicate_results.values())

        log.info(
            "duplicate_detection_completed",
            total_duplicates=total_duplicates,
        )

        return {
            "results": duplicate_results,
            "total_duplicates": total_duplicates,
            "validation_status": "pass" if total_duplicates == 0 else "fail",
        }

    async def _check_duplicates(
        self, table_name: str, key_column: str
    ) -> Dict[str, Any]:
        """Check for duplicate records by key column."""
        query = f"""
            SELECT {key_column}, COUNT(*) as count
            FROM {table_name}
            GROUP BY {key_column}
            HAVING COUNT(*) > 1
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query)

        duplicates = [
            {
                "key": row[key_column],
                "count": row["count"],
            }
            for row in rows
        ]

        return {
            "table": table_name,
            "key_column": key_column,
            "duplicate_count": len(duplicates),
            "duplicates": duplicates[:10],  # Return first 10
        }

    async def _check_balance_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate balance records."""
        query = """
            SELECT wallet_address, token_address, snapshot_date, COUNT(*) as count
            FROM wallet_balances
            GROUP BY wallet_address, token_address, snapshot_date
            HAVING COUNT(*) > 1
        """

        async with self.db.acquire() as conn:
            rows = await conn.fetch(query)

        duplicates = [
            {
                "wallet_address": row["wallet_address"],
                "token_address": row["token_address"],
                "snapshot_date": row["snapshot_date"].isoformat(),
                "count": row["count"],
            }
            for row in rows
        ]

        return {
            "table": "wallet_balances",
            "key_columns": ["wallet_address", "token_address", "snapshot_date"],
            "duplicate_count": len(duplicates),
            "duplicates": duplicates[:10],
        }

    async def detect_orphaned_records(self) -> Dict[str, Any]:
        """Detect orphaned records (records without valid foreign key references).

        Returns:
            Orphaned record detection results
        """
        log = self.logger
        log.info("detecting_orphaned_records")

        orphan_results = {}

        # Find transactions with invalid wallet references
        transactions_orphans = await self._find_orphaned_transactions()
        orphan_results["transactions"] = transactions_orphans

        # Find balances with invalid wallet/token references
        balances_orphans = await self._find_orphaned_balances()
        orphan_results["wallet_balances"] = balances_orphans

        total_orphans = sum(r["orphan_count"] for r in orphan_results.values())

        log.info(
            "orphan_detection_completed",
            total_orphans=total_orphans,
        )

        return {
            "results": orphan_results,
            "total_orphans": total_orphans,
            "validation_status": "pass" if total_orphans == 0 else "fail",
        }

    async def _find_orphaned_transactions(self) -> Dict[str, Any]:
        """Find transactions with invalid wallet references."""
        query = """
            SELECT COUNT(*) as count
            FROM transactions t
            LEFT JOIN wallets w ON t.wallet_address = w.wallet_address
            WHERE w.wallet_address IS NULL
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)
            orphan_count = row["count"]

        return {
            "table": "transactions",
            "orphan_count": orphan_count,
        }

    async def _find_orphaned_balances(self) -> Dict[str, Any]:
        """Find balances with invalid wallet or token references."""
        query = """
            SELECT COUNT(*) as wallet_orphans
            FROM wallet_balances wb
            LEFT JOIN wallets w ON wb.wallet_address = w.wallet_address
            WHERE w.wallet_address IS NULL
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query)
            wallet_orphans = row["wallet_orphans"]

        query2 = """
            SELECT COUNT(*) as token_orphans
            FROM wallet_balances wb
            LEFT JOIN tokens t ON wb.token_address = t.token_address
            WHERE t.token_address IS NULL
        """

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(query2)
            token_orphans = row["token_orphans"]

        return {
            "table": "wallet_balances",
            "wallet_orphans": wallet_orphans,
            "token_orphans": token_orphans,
            "orphan_count": wallet_orphans + token_orphans,
        }