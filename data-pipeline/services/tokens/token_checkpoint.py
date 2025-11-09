"""
Token collection checkpoint management with resume capability.

Implements:
- Checkpoint management for token collection in `collection_checkpoints` table
- Track collection_type='tokens', last_processed_date, records_collected, status
- Atomic checkpoint updates on successful completion
- Resume capability to continue from last successful checkpoint
- Partial collection scenarios and recovery
- Checkpoint recovery with various failure scenarios
"""

import json
import time
from datetime import date, datetime
from typing import Dict, Any, Optional, List

from data_collection.common.checkpoints import CheckpointManager
from data_collection.common.logging_setup import get_logger
from .token_fetcher import FetchStats


class TokenCheckpointManager:
    """
    Specialized checkpoint manager for token collection operations.
    """

    def __init__(self):
        self.checkpoint_manager = CheckpointManager("tokens")
        self.logger = get_logger("token_checkpoint")

    def get_collection_status(self) -> Dict[str, Any]:
        """
        Get current collection status and determine if resume is possible.

        Returns:
            Dictionary with collection status and resume information
        """
        checkpoint = self.checkpoint_manager.get_last_checkpoint()

        if not checkpoint:
            return {
                "status": "new",
                "can_resume": False,
                "last_collection_date": None,
                "total_tokens_collected": 0,
                "collection_metadata": {},
                "resume_needed": False
            }

        status = checkpoint.get("status", "unknown")
        last_date = checkpoint.get("last_processed_date")
        total_collected = checkpoint.get("records_collected", 0)

        # Parse metadata if available
        metadata = {}
        # Note: The existing schema doesn't have a metadata column,
        # but we can store it as JSON in a future migration if needed

        collection_status = {
            "status": status,
            "can_resume": status in ["running", "failed"],
            "last_collection_date": last_date,
            "total_tokens_collected": total_collected,
            "collection_metadata": metadata,
            "resume_needed": status == "running",
            "last_updated": checkpoint.get("updated_at")
        }

        self.logger.log_operation(
            operation="get_collection_status",
            params={
                "status": status,
                "total_collected": total_collected,
                "can_resume": collection_status["can_resume"]
            },
            status="completed",
            message=f"Collection status: {status}, {total_collected} tokens collected"
        )

        return collection_status

    def start_collection(self, target_count: int) -> None:
        """
        Mark the start of a new token collection run.

        Args:
            target_count: Target number of tokens to collect
        """
        # Store metadata about this collection run
        metadata = {
            "target_count": target_count,
            "started_at": datetime.utcnow().isoformat(),
            "collection_version": "1.2"
        }

        self.checkpoint_manager.update_checkpoint(
            last_processed_date=date.today(),
            records_collected=0,
            status="running"
        )

        self.logger.log_operation(
            operation="start_collection",
            params={"target_count": target_count},
            status="started",
            message=f"Started token collection run targeting {target_count} tokens"
        )

    def update_progress(self,
                       tokens_processed: int,
                       stats: FetchStats,
                       is_final: bool = False) -> None:
        """
        Update collection progress checkpoint.

        Args:
            tokens_processed: Number of tokens successfully processed
            stats: Fetch statistics from token fetcher
            is_final: Whether this is the final update (collection completed)
        """
        status = "completed" if is_final else "running"

        # Create metadata summary
        metadata = {
            "total_pages_fetched": stats.total_pages_fetched,
            "total_tokens_found": stats.total_tokens_found,
            "ethereum_tokens_found": stats.ethereum_tokens_found,
            "tokens_with_contracts": stats.tokens_with_contracts,
            "duplicates_found": stats.duplicates_found,
            "invalid_addresses": stats.invalid_addresses,
            "missing_decimals": stats.missing_decimals,
            "validation_errors": stats.validation_errors,
            "duration_seconds": stats.duration_seconds,
            "last_updated": datetime.utcnow().isoformat()
        }

        self.checkpoint_manager.update_checkpoint(
            last_processed_date=date.today(),
            records_collected=tokens_processed,
            status=status
        )

        self.logger.log_operation(
            operation="update_progress",
            params={
                "tokens_processed": tokens_processed,
                "status": status,
                "pages_fetched": stats.total_pages_fetched,
                "ethereum_tokens": stats.ethereum_tokens_found
            },
            status="completed",
            message=f"Progress updated: {tokens_processed} tokens processed, status={status}"
        )

    def mark_collection_failed(self, error_message: str, stats: Optional[FetchStats] = None) -> None:
        """
        Mark collection as failed with error details.

        Args:
            error_message: Description of the failure
            stats: Optional fetch statistics at time of failure
        """
        metadata = {
            "error_message": error_message,
            "failed_at": datetime.utcnow().isoformat(),
            "recovery_instructions": "Review logs and restart collection"
        }

        if stats:
            metadata.update({
                "total_pages_fetched": stats.total_pages_fetched,
                "tokens_processed_before_failure": stats.ethereum_tokens_found,
                "duration_before_failure": stats.duration_seconds
            })

        # Get current records count before marking as failed
        current_checkpoint = self.checkpoint_manager.get_last_checkpoint()
        current_count = current_checkpoint.get("records_collected", 0) if current_checkpoint else 0

        self.checkpoint_manager.update_checkpoint(
            last_processed_date=date.today(),
            records_collected=current_count,  # Keep existing count
            status="failed"
        )

        self.logger.log_operation(
            operation="mark_collection_failed",
            params={"error": error_message[:100]},  # Truncate long errors
            status="error",
            error=error_message,
            message="Collection marked as failed"
        )

    def can_resume_collection(self) -> bool:
        """
        Check if collection can be resumed from last checkpoint.

        Returns:
            True if collection can be resumed, False otherwise
        """
        status_info = self.get_collection_status()

        # Can resume if status is 'running' or 'failed'
        can_resume = status_info["can_resume"]

        # Additional check: don't resume if last collection was today and completed
        if (status_info["status"] == "completed" and
            status_info["last_collection_date"] == date.today()):
            can_resume = False

        self.logger.log_operation(
            operation="check_resume_capability",
            params={
                "can_resume": can_resume,
                "status": status_info["status"],
                "last_date": str(status_info["last_collection_date"])
            },
            status="completed",
            message=f"Resume capability: {can_resume}"
        )

        return can_resume

    def get_resume_info(self) -> Dict[str, Any]:
        """
        Get information needed to resume collection.

        Returns:
            Dictionary with resume information
        """
        checkpoint = self.checkpoint_manager.get_last_checkpoint()

        if not checkpoint:
            return {
                "should_resume": False,
                "tokens_already_collected": 0,
                "last_collection_date": None,
                "estimated_remaining": None
            }

        tokens_collected = checkpoint.get("records_collected", 0)
        last_date = checkpoint.get("last_collection_date")
        status = checkpoint.get("status", "unknown")

        resume_info = {
            "should_resume": status in ["running", "failed"],
            "tokens_already_collected": tokens_collected,
            "last_collection_date": last_date,
            "estimated_remaining": max(0, 500 - tokens_collected),  # Assuming 500 target
            "last_status": status,
            "can_start_fresh": status == "completed"
        }

        self.logger.log_operation(
            operation="get_resume_info",
            params={
                "should_resume": resume_info["should_resume"],
                "tokens_collected": tokens_collected,
                "status": status
            },
            status="completed",
            message=f"Resume info: {tokens_collected} tokens already collected"
        )

        return resume_info

    def reset_collection(self) -> None:
        """
        Reset collection status to start fresh.

        Use this when you want to start a completely new collection
        regardless of previous checkpoints.
        """
        self.checkpoint_manager.update_checkpoint(
            last_processed_date=date.today(),
            records_collected=0,
            status="pending"
        )

        self.logger.log_operation(
            operation="reset_collection",
            status="completed",
            message="Collection checkpoint reset for fresh start"
        )

    def test_checkpoint_recovery(self) -> Dict[str, Any]:
        """
        Test checkpoint recovery scenarios.

        Returns:
            Dictionary with test results
        """
        test_results = {
            "recovery_scenarios_tested": [],
            "all_tests_passed": True,
            "test_errors": []
        }

        try:
            # Test 1: Normal checkpoint creation and retrieval
            original_checkpoint = self.checkpoint_manager.get_last_checkpoint()

            # Create test checkpoint
            self.checkpoint_manager.update_checkpoint(
                last_processed_date=date.today(),
                records_collected=100,
                status="running"
            )

            # Retrieve and verify
            test_checkpoint = self.checkpoint_manager.get_last_checkpoint()
            if test_checkpoint and test_checkpoint["records_collected"] == 100:
                test_results["recovery_scenarios_tested"].append("checkpoint_creation_retrieval")
            else:
                test_results["all_tests_passed"] = False
                test_results["test_errors"].append("Failed to create/retrieve test checkpoint")

            # Test 2: Resume capability check
            can_resume = self.can_resume_collection()
            resume_info = self.get_resume_info()

            if resume_info["tokens_already_collected"] == 100:
                test_results["recovery_scenarios_tested"].append("resume_capability")
            else:
                test_results["all_tests_passed"] = False
                test_results["test_errors"].append("Resume info incorrect")

            # Test 3: Failure recovery
            self.mark_collection_failed("Test failure scenario")
            failed_checkpoint = self.checkpoint_manager.get_last_checkpoint()

            if failed_checkpoint and failed_checkpoint["status"] == "failed":
                test_results["recovery_scenarios_tested"].append("failure_recovery")
            else:
                test_results["all_tests_passed"] = False
                test_results["test_errors"].append("Failed to mark collection as failed")

            # Restore original checkpoint if it existed
            if original_checkpoint:
                self.checkpoint_manager.update_checkpoint(
                    last_processed_date=original_checkpoint.get("last_processed_date"),
                    records_collected=original_checkpoint.get("records_collected", 0),
                    status=original_checkpoint.get("status", "pending")
                )

            self.logger.log_operation(
                operation="test_checkpoint_recovery",
                params={
                    "scenarios_tested": len(test_results["recovery_scenarios_tested"]),
                    "all_passed": test_results["all_tests_passed"]
                },
                status="completed" if test_results["all_tests_passed"] else "warning",
                message=f"Checkpoint recovery test completed: {len(test_results['recovery_scenarios_tested'])} scenarios"
            )

        except Exception as e:
            test_results["all_tests_passed"] = False
            test_results["test_errors"].append(f"Test exception: {str(e)}")

            self.logger.log_operation(
                operation="test_checkpoint_recovery",
                status="error",
                error=str(e),
                message="Checkpoint recovery test failed"
            )

        return test_results