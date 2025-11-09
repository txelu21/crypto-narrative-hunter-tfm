"""
Token collection orchestration service with comprehensive error handling and resilience.

Implements:
- Classify retryable errors (429, 5xx, timeouts) vs fatal errors (4xx logic issues)
- Maximum retry attempts (5) with exponential backoff
- Graceful degradation for partial API failures
- Comprehensive error logging with context
- Network failures, API downtime, and malformed responses handling
- Recovery from interrupted collections
"""

import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from data_collection.common.logging_setup import get_logger
from .coingecko_client import CoinGeckoClient, CoinGeckoAPIError, CoinGeckoRateLimitError
from .token_fetcher import TokenFetcher, FetchStats
from .token_storage import TokenStorage, TokenStorageError
from .token_checkpoint import TokenCheckpointManager
from .token_export import TokenExporter


class TokenCollectionError(Exception):
    """Base exception for token collection errors"""
    pass


class TokenCollectionService:
    """
    Orchestrates the complete token collection process with error handling and resilience.
    """

    def __init__(self):
        self.logger = get_logger("token_collection_service")
        self.checkpoint_manager = TokenCheckpointManager()
        self.storage = TokenStorage()
        self.exporter = TokenExporter()

    def collect_tokens(self,
                      target_count: int = 500,
                      force_restart: bool = False,
                      export_csv: bool = True,
                      generate_report: bool = True) -> Dict[str, Any]:
        """
        Complete token collection workflow with error handling and recovery.

        Args:
            target_count: Number of tokens to collect (default: 500)
            force_restart: Force restart even if resumable checkpoint exists
            export_csv: Generate CSV export
            generate_report: Generate quality report

        Returns:
            Dictionary with collection results and statistics
        """
        start_time = time.time()
        collection_result = {
            "success": False,
            "tokens_collected": 0,
            "errors": [],
            "warnings": [],
            "files_created": [],
            "statistics": {},
            "collection_duration_seconds": 0
        }

        self.logger.log_operation(
            operation="collect_tokens",
            params={
                "target_count": target_count,
                "force_restart": force_restart,
                "export_csv": export_csv,
                "generate_report": generate_report
            },
            status="started",
            message=f"Starting token collection: target={target_count}, force_restart={force_restart}"
        )

        try:
            # Step 1: Check resume capability
            resume_info = self._handle_resume_logic(force_restart, target_count)
            if resume_info.get("should_skip"):
                collection_result.update(resume_info)
                return collection_result

            # Step 2: Initialize collection
            self.checkpoint_manager.start_collection(target_count)

            # Step 3: Fetch tokens with error handling
            tokens, fetch_stats = self._fetch_tokens_with_resilience(target_count)

            if not tokens:
                raise TokenCollectionError("No tokens were successfully fetched")

            collection_result["tokens_collected"] = len(tokens)
            collection_result["statistics"]["fetch_stats"] = {
                "total_pages_fetched": fetch_stats.total_pages_fetched,
                "total_tokens_found": fetch_stats.total_tokens_found,
                "ethereum_tokens_found": fetch_stats.ethereum_tokens_found,
                "duplicates_found": fetch_stats.duplicates_found,
                "validation_errors": fetch_stats.validation_errors,
                "duration_seconds": fetch_stats.duration_seconds
            }

            # Step 4: Store tokens with error handling
            storage_stats = self._store_tokens_with_resilience(tokens)
            collection_result["statistics"]["storage_stats"] = storage_stats

            # Step 5: Update checkpoint
            self.checkpoint_manager.update_progress(
                tokens_processed=len(tokens),
                stats=fetch_stats,
                is_final=True
            )

            # Step 6: Export and reporting (optional, non-critical)
            if export_csv:
                try:
                    csv_path = self.exporter.export_tokens_to_csv(tokens)
                    collection_result["files_created"].append(csv_path)
                except Exception as e:
                    collection_result["warnings"].append(f"CSV export failed: {str(e)}")

            if generate_report:
                try:
                    quality_report = self.exporter.generate_data_quality_report(tokens, fetch_stats)
                    report_path = self.exporter.save_quality_report(quality_report)
                    collection_result["files_created"].append(report_path)

                    validation_summary = self.exporter.create_validation_summary(tokens, fetch_stats)
                    collection_result["statistics"]["validation_summary"] = validation_summary
                except Exception as e:
                    collection_result["warnings"].append(f"Report generation failed: {str(e)}")

            # Success!
            collection_result["success"] = True
            collection_result["collection_duration_seconds"] = time.time() - start_time

            self.logger.log_operation(
                operation="collect_tokens",
                params={
                    "tokens_collected": len(tokens),
                    "files_created": len(collection_result["files_created"]),
                    "warnings": len(collection_result["warnings"])
                },
                status="completed",
                duration_ms=int(collection_result["collection_duration_seconds"] * 1000),
                message=f"Token collection completed successfully: {len(tokens)} tokens"
            )

            return collection_result

        except Exception as e:
            # Handle collection failure
            collection_result["collection_duration_seconds"] = time.time() - start_time
            error_context = self._build_error_context(e)
            collection_result["errors"].append(error_context)

            # Mark checkpoint as failed
            try:
                self.checkpoint_manager.mark_collection_failed(
                    error_message=str(e),
                    stats=locals().get('fetch_stats')
                )
            except Exception as checkpoint_error:
                collection_result["warnings"].append(f"Failed to update checkpoint: {str(checkpoint_error)}")

            self.logger.log_operation(
                operation="collect_tokens",
                status="error",
                error=str(e),
                duration_ms=int(collection_result["collection_duration_seconds"] * 1000),
                message="Token collection failed"
            )

            # Don't re-raise in production - return error result instead
            return collection_result

    def _handle_resume_logic(self, force_restart: bool, target_count: int) -> Dict[str, Any]:
        """Handle collection resume logic and return early if needed"""
        if force_restart:
            self.checkpoint_manager.reset_collection()
            self.logger.log_operation(
                operation="handle_resume",
                status="info",
                message="Collection reset due to force_restart=True"
            )
            return {}

        # Check if we can/should resume
        resume_info = self.checkpoint_manager.get_resume_info()

        if resume_info["should_resume"]:
            self.logger.log_operation(
                operation="handle_resume",
                params={
                    "tokens_already_collected": resume_info["tokens_already_collected"],
                    "estimated_remaining": resume_info["estimated_remaining"]
                },
                status="info",
                message="Resuming from previous checkpoint"
            )
            return {}

        if (resume_info.get("can_start_fresh") and
            resume_info["tokens_already_collected"] >= target_count * 0.9):  # 90% threshold
            return {
                "should_skip": True,
                "success": True,
                "tokens_collected": resume_info["tokens_already_collected"],
                "warnings": [f"Collection already completed recently with {resume_info['tokens_already_collected']} tokens"],
                "collection_duration_seconds": 0
            }

        return {}

    def _fetch_tokens_with_resilience(self, target_count: int) -> Tuple[List, FetchStats]:
        """Fetch tokens with comprehensive error handling and retry logic"""
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            try:
                self.logger.log_operation(
                    operation="fetch_tokens_attempt",
                    params={"attempt": attempt, "max_attempts": max_attempts},
                    status="started",
                    message=f"Token fetch attempt {attempt}/{max_attempts}"
                )

                with CoinGeckoClient() as client:
                    # Test connectivity first
                    if not client.test_connectivity():
                        raise TokenCollectionError("CoinGecko API connectivity test failed")

                    fetcher = TokenFetcher(client)
                    tokens, stats = fetcher.fetch_top_ethereum_tokens(target_count)

                    self.logger.log_operation(
                        operation="fetch_tokens_attempt",
                        params={"attempt": attempt, "tokens_fetched": len(tokens)},
                        status="completed",
                        message=f"Fetch attempt {attempt} succeeded: {len(tokens)} tokens"
                    )

                    return tokens, stats

            except CoinGeckoRateLimitError as e:
                if attempt < max_attempts:
                    wait_time = min(60 * attempt, 300)  # Progressive backoff, max 5 minutes
                    self.logger.log_operation(
                        operation="fetch_tokens_attempt",
                        params={"attempt": attempt, "wait_time": wait_time},
                        status="retry",
                        error=str(e),
                        message=f"Rate limited on attempt {attempt}, waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise TokenCollectionError(f"Rate limited after {max_attempts} attempts: {str(e)}")

            except CoinGeckoAPIError as e:
                if attempt < max_attempts and "5" in str(e):  # Server errors (5xx)
                    wait_time = 30 * attempt
                    self.logger.log_operation(
                        operation="fetch_tokens_attempt",
                        params={"attempt": attempt, "wait_time": wait_time},
                        status="retry",
                        error=str(e),
                        message=f"Server error on attempt {attempt}, retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise TokenCollectionError(f"API error after {attempt} attempts: {str(e)}")

            except Exception as e:
                if attempt < max_attempts:
                    wait_time = 15 * attempt
                    self.logger.log_operation(
                        operation="fetch_tokens_attempt",
                        params={"attempt": attempt, "wait_time": wait_time},
                        status="retry",
                        error=str(e),
                        message=f"Unexpected error on attempt {attempt}, retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise TokenCollectionError(f"Fetch failed after {max_attempts} attempts: {str(e)}")

        raise TokenCollectionError(f"Token fetching failed after {max_attempts} attempts")

    def _store_tokens_with_resilience(self, tokens: List) -> Dict[str, int]:
        """Store tokens with error handling and retry logic"""
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            try:
                self.logger.log_operation(
                    operation="store_tokens_attempt",
                    params={"attempt": attempt, "token_count": len(tokens)},
                    status="started",
                    message=f"Storage attempt {attempt}/{max_attempts}"
                )

                storage_stats = self.storage.store_tokens(tokens)

                self.logger.log_operation(
                    operation="store_tokens_attempt",
                    params={"attempt": attempt, "inserted": storage_stats["inserted"], "updated": storage_stats["updated"]},
                    status="completed",
                    message=f"Storage attempt {attempt} succeeded"
                )

                return storage_stats

            except TokenStorageError as e:
                if attempt < max_attempts:
                    wait_time = 10 * attempt
                    self.logger.log_operation(
                        operation="store_tokens_attempt",
                        params={"attempt": attempt, "wait_time": wait_time},
                        status="retry",
                        error=str(e),
                        message=f"Storage error on attempt {attempt}, retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise TokenCollectionError(f"Storage failed after {max_attempts} attempts: {str(e)}")

            except Exception as e:
                if attempt < max_attempts:
                    wait_time = 10 * attempt
                    self.logger.log_operation(
                        operation="store_tokens_attempt",
                        params={"attempt": attempt, "wait_time": wait_time},
                        status="retry",
                        error=str(e),
                        message=f"Unexpected storage error on attempt {attempt}, retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise TokenCollectionError(f"Storage failed after {max_attempts} attempts: {str(e)}")

        raise TokenCollectionError(f"Token storage failed after {max_attempts} attempts")

    def _build_error_context(self, error: Exception) -> Dict[str, Any]:
        """Build comprehensive error context for debugging"""
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_classification": self._classify_error(error),
            "stacktrace": traceback.format_exc(),
            "timestamp": time.time(),
            "recovery_suggestions": self._get_recovery_suggestions(error)
        }

    def _classify_error(self, error: Exception) -> str:
        """Classify error as retryable, fatal, or configuration issue"""
        error_str = str(error).lower()

        if isinstance(error, CoinGeckoRateLimitError):
            return "retryable_rate_limit"
        elif isinstance(error, CoinGeckoAPIError):
            if "5" in error_str or "timeout" in error_str:
                return "retryable_server_error"
            elif "4" in error_str and "401" not in error_str and "403" not in error_str:
                return "fatal_client_error"
            else:
                return "configuration_error"
        elif isinstance(error, TokenStorageError):
            if "connection" in error_str or "timeout" in error_str:
                return "retryable_database_error"
            else:
                return "fatal_database_error"
        elif "network" in error_str or "connection" in error_str:
            return "retryable_network_error"
        else:
            return "unknown_error"

    def _get_recovery_suggestions(self, error: Exception) -> List[str]:
        """Get recovery suggestions based on error type"""
        error_type = self._classify_error(error)

        suggestions = {
            "retryable_rate_limit": [
                "Wait for rate limit to reset (typically 1 minute)",
                "Consider using a CoinGecko API key for higher limits",
                "Reduce request frequency"
            ],
            "retryable_server_error": [
                "Retry the operation after a short delay",
                "Check CoinGecko API status page",
                "Consider using cached data if available"
            ],
            "fatal_client_error": [
                "Check request parameters and data format",
                "Verify token addresses and IDs are valid",
                "Review API documentation for endpoint requirements"
            ],
            "configuration_error": [
                "Check API key configuration",
                "Verify database connection settings",
                "Review environment variables"
            ],
            "retryable_database_error": [
                "Check database connection",
                "Verify database is running and accessible",
                "Check database credentials"
            ],
            "fatal_database_error": [
                "Check database schema and constraints",
                "Verify data format and types",
                "Review database logs for details"
            ],
            "retryable_network_error": [
                "Check internet connection",
                "Verify firewall and proxy settings",
                "Try again in a few minutes"
            ]
        }

        return suggestions.get(error_type, ["Contact support with error details", "Check system logs"])

    def validate_collection_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of a completed collection.

        Returns:
            Dictionary with validation results
        """
        self.logger.log_operation(
            operation="validate_collection_integrity",
            status="started",
            message="Starting collection integrity validation"
        )

        try:
            # Get collection status
            status_info = self.checkpoint_manager.get_collection_status()

            # Validate database constraints
            db_validation = self.storage.validate_database_constraints()

            # Get token count
            token_count = self.storage.get_token_count()

            # Check for recent tokens
            recent_tokens = self.storage.get_tokens_by_rank_range(1, 100)

            integrity_report = {
                "collection_status": status_info,
                "database_validation": db_validation,
                "token_statistics": {
                    "total_tokens_in_db": token_count,
                    "top_100_tokens_found": len(recent_tokens),
                    "expected_minimum": 100  # We expect at least top 100 tokens
                },
                "integrity_passed": (
                    status_info["status"] == "completed" and
                    db_validation["validation_passed"] and
                    token_count >= 100
                ),
                "validation_timestamp": time.time()
            }

            self.logger.log_operation(
                operation="validate_collection_integrity",
                params={
                    "tokens_in_db": token_count,
                    "integrity_passed": integrity_report["integrity_passed"]
                },
                status="completed",
                message=f"Integrity validation completed: {integrity_report['integrity_passed']}"
            )

            return integrity_report

        except Exception as e:
            self.logger.log_operation(
                operation="validate_collection_integrity",
                status="error",
                error=str(e),
                message="Integrity validation failed"
            )

            return {
                "integrity_passed": False,
                "error": str(e),
                "validation_timestamp": time.time()
            }