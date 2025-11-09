"""
Manual review service for narrative classification.

Implements:
- CSV export for manual review with token metadata and suggested categories
- CSV import functionality for reviewed classifications
- Validation for manual review inputs and data integrity
- Audit trail logging for all manual classification decisions
- Batch processing for reviewed classifications
"""

import csv
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from data_collection.common.db import get_cursor, execute_with_retry
from data_collection.common.logging_setup import get_logger
from .narrative_classifier import NarrativeCategory


@dataclass
class ReviewExportStats:
    """Statistics for manual review export operation"""
    total_tokens_exported: int = 0
    pending_review_count: int = 0
    low_confidence_count: int = 0
    unclassified_count: int = 0
    file_path: str = ""
    duration_seconds: float = 0


@dataclass
class ReviewImportStats:
    """Statistics for manual review import operation"""
    total_rows_processed: int = 0
    successful_imports: int = 0
    validation_errors: int = 0
    database_updates: int = 0
    audit_entries_created: int = 0
    duration_seconds: float = 0
    error_details: List[str] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []


class ManualReviewService:
    """
    Service for managing manual review workflow of token classifications.
    """

    def __init__(self, review_data_dir: str = "data/manual_reviews"):
        self.logger = get_logger("manual_review_service")
        self.review_data_dir = Path(review_data_dir)
        self.review_data_dir.mkdir(parents=True, exist_ok=True)

        # Valid categories for validation
        self.valid_categories = {cat.value for cat in NarrativeCategory}

    def export_tokens_for_review(self,
                                include_pending: bool = True,
                                include_low_confidence: bool = True,
                                include_unclassified: bool = True,
                                confidence_threshold: float = 50.0,
                                max_tokens: Optional[int] = None) -> ReviewExportStats:
        """
        Export tokens requiring manual review to CSV format.

        Args:
            include_pending: Include tokens with manual_review_status = 'pending'
            include_low_confidence: Include tokens with confidence below threshold
            include_unclassified: Include tokens without narrative_category
            confidence_threshold: Threshold for low confidence classification
            max_tokens: Maximum number of tokens to export (None for all)

        Returns:
            ReviewExportStats with export results
        """
        start_time = time.time()
        stats = ReviewExportStats()

        self.logger.log_operation(
            operation="export_tokens_for_review",
            params={
                "include_pending": include_pending,
                "include_low_confidence": include_low_confidence,
                "include_unclassified": include_unclassified,
                "confidence_threshold": confidence_threshold,
                "max_tokens": max_tokens
            },
            status="started",
            message="Starting export of tokens for manual review"
        )

        try:
            # Build query conditions
            conditions = []
            params = []

            if include_pending:
                conditions.append("manual_review_status = %s")
                params.append('pending')

            if include_low_confidence:
                conditions.append("(classification_confidence < %s AND classification_confidence IS NOT NULL)")
                params.append(confidence_threshold)

            if include_unclassified:
                conditions.append("narrative_category IS NULL")

            if not conditions:
                raise ValueError("At least one inclusion criteria must be specified")

            # Construct query
            where_clause = " OR ".join(f"({condition})" for condition in conditions)
            limit_clause = f"LIMIT {max_tokens}" if max_tokens else ""

            query = f"""
                SELECT
                    token_address,
                    symbol,
                    name,
                    decimals,
                    market_cap_rank,
                    avg_daily_volume_usd,
                    narrative_category as suggested_category,
                    classification_confidence,
                    manual_review_status,
                    created_at,
                    updated_at
                FROM tokens
                WHERE {where_clause}
                ORDER BY
                    CASE WHEN market_cap_rank IS NOT NULL THEN market_cap_rank ELSE 999999 END ASC,
                    classification_confidence ASC NULLS LAST
                {limit_clause}
            """

            # Execute query
            tokens = execute_with_retry(query, tuple(params))

            if not tokens:
                stats.duration_seconds = time.time() - start_time
                self.logger.log_operation(
                    operation="export_tokens_for_review",
                    status="completed",
                    message="No tokens found for manual review"
                )
                return stats

            # Generate CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tokens_for_review_{timestamp}.csv"
            file_path = self.review_data_dir / filename

            stats.total_tokens_exported = len(tokens)
            stats.file_path = str(file_path)

            # Write CSV file
            self._write_review_csv(file_path, tokens, stats)

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="export_tokens_for_review",
                params={
                    "tokens_exported": stats.total_tokens_exported,
                    "file_path": stats.file_path
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Exported {stats.total_tokens_exported} tokens to {filename}"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="export_tokens_for_review",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def _write_review_csv(self, file_path: Path, tokens: List[Dict[str, Any]], stats: ReviewExportStats) -> None:
        """Write tokens to CSV file with review format"""
        fieldnames = [
            'token_address',
            'symbol',
            'name',
            'market_cap_rank',
            'avg_daily_volume_usd',
            'suggested_category',
            'confidence',
            'manual_category',
            'review_notes',
            'reviewer',
            'review_date'
        ]

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for token in tokens:
                # Count different types for statistics
                if token.get('manual_review_status') == 'pending':
                    stats.pending_review_count += 1
                if token.get('classification_confidence') and token['classification_confidence'] < 50.0:
                    stats.low_confidence_count += 1
                if not token.get('narrative_category'):
                    stats.unclassified_count += 1

                # Write token row
                row = {
                    'token_address': token['token_address'],
                    'symbol': token['symbol'],
                    'name': token['name'],
                    'market_cap_rank': token.get('market_cap_rank', ''),
                    'avg_daily_volume_usd': token.get('avg_daily_volume_usd', ''),
                    'suggested_category': token.get('suggested_category', ''),
                    'confidence': f"{token.get('classification_confidence', 0):.1f}%" if token.get('classification_confidence') else '',
                    'manual_category': '',  # To be filled by reviewer
                    'review_notes': '',     # To be filled by reviewer
                    'reviewer': '',         # To be filled by reviewer
                    'review_date': ''       # To be filled by reviewer
                }
                writer.writerow(row)

    def import_reviewed_classifications(self, csv_file_path: str, reviewer: str) -> ReviewImportStats:
        """
        Import reviewed classifications from CSV file.

        Args:
            csv_file_path: Path to CSV file with reviewed classifications
            reviewer: Username of the reviewer

        Returns:
            ReviewImportStats with import results
        """
        start_time = time.time()
        stats = ReviewImportStats()

        self.logger.log_operation(
            operation="import_reviewed_classifications",
            params={
                "csv_file": os.path.basename(csv_file_path),
                "reviewer": reviewer
            },
            status="started",
            message=f"Starting import of reviewed classifications from {os.path.basename(csv_file_path)}"
        )

        try:
            # Validate file exists
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

            # Read and validate CSV
            reviewed_tokens = self._read_and_validate_review_csv(csv_file_path, stats)

            if not reviewed_tokens:
                self.logger.log_operation(
                    operation="import_reviewed_classifications",
                    status="completed",
                    message="No valid reviewed tokens found in CSV"
                )
                return stats

            # Process reviewed tokens
            self._process_reviewed_tokens(reviewed_tokens, reviewer, stats)

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="import_reviewed_classifications",
                params={
                    "total_processed": stats.total_rows_processed,
                    "successful_imports": stats.successful_imports,
                    "validation_errors": stats.validation_errors,
                    "database_updates": stats.database_updates
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Import completed: {stats.successful_imports}/{stats.total_rows_processed} tokens processed successfully"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="import_reviewed_classifications",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def _read_and_validate_review_csv(self, csv_file_path: str, stats: ReviewImportStats) -> List[Dict[str, Any]]:
        """Read and validate CSV file with reviewed classifications"""
        reviewed_tokens = []

        required_fields = ['token_address', 'manual_category']
        optional_fields = ['review_notes', 'reviewer', 'review_date']

        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # Validate CSV headers
            if not all(field in reader.fieldnames for field in required_fields):
                missing_fields = [field for field in required_fields if field not in reader.fieldnames]
                raise ValueError(f"CSV missing required fields: {missing_fields}")

            for row_num, row in enumerate(reader, start=2):  # Start at 2 to account for header
                stats.total_rows_processed += 1

                try:
                    # Skip rows without manual classification
                    manual_category = row.get('manual_category', '').strip()
                    if not manual_category:
                        continue

                    # Validate token address
                    token_address = row.get('token_address', '').strip()
                    if not token_address or len(token_address) != 42 or not token_address.startswith('0x'):
                        stats.validation_errors += 1
                        stats.error_details.append(f"Row {row_num}: Invalid token address format: {token_address}")
                        continue

                    # Validate narrative category
                    if manual_category not in self.valid_categories:
                        stats.validation_errors += 1
                        stats.error_details.append(f"Row {row_num}: Invalid category '{manual_category}'. Valid categories: {', '.join(self.valid_categories)}")
                        continue

                    # Prepare reviewed token data
                    reviewed_token = {
                        'token_address': token_address,
                        'manual_category': manual_category,
                        'review_notes': row.get('review_notes', '').strip(),
                        'reviewer': row.get('reviewer', '').strip(),
                        'review_date': row.get('review_date', '').strip()
                    }

                    reviewed_tokens.append(reviewed_token)

                except Exception as e:
                    stats.validation_errors += 1
                    stats.error_details.append(f"Row {row_num}: Validation error: {str(e)}")

        return reviewed_tokens

    def _process_reviewed_tokens(self, reviewed_tokens: List[Dict[str, Any]], default_reviewer: str, stats: ReviewImportStats) -> None:
        """Process reviewed tokens and update database"""
        try:
            with get_cursor() as cur:
                for token_data in reviewed_tokens:
                    try:
                        # Get current token data
                        current_token = self._get_current_token_data(cur, token_data['token_address'])
                        if not current_token:
                            stats.validation_errors += 1
                            stats.error_details.append(f"Token not found in database: {token_data['token_address']}")
                            continue

                        # Prepare update data
                        reviewer = token_data.get('reviewer') or default_reviewer
                        review_notes = token_data.get('review_notes', '')

                        # Update token classification
                        self._update_reviewed_classification(
                            cur,
                            token_data['token_address'],
                            token_data['manual_category'],
                            reviewer,
                            review_notes
                        )

                        stats.successful_imports += 1
                        stats.database_updates += 1

                        # Create audit entry
                        self._create_manual_review_audit(
                            cur,
                            current_token,
                            token_data['manual_category'],
                            reviewer,
                            review_notes
                        )

                        stats.audit_entries_created += 1

                    except Exception as e:
                        stats.validation_errors += 1
                        stats.error_details.append(f"Error processing token {token_data['token_address']}: {str(e)}")

        except Exception as e:
            self.logger.log_operation(
                operation="process_reviewed_tokens",
                status="error",
                error=str(e)
            )
            raise

    def _get_current_token_data(self, cursor, token_address: str) -> Optional[Dict[str, Any]]:
        """Get current token data from database"""
        query = """
            SELECT token_address, symbol, name, narrative_category,
                   classification_confidence, manual_review_status
            FROM tokens
            WHERE token_address = %s
        """

        cursor.execute(query, (token_address,))
        result = cursor.fetchone()
        return dict(result) if result else None

    def _update_reviewed_classification(self, cursor, token_address: str, manual_category: str, reviewer: str, review_notes: str) -> None:
        """Update token with reviewed classification"""
        update_query = """
            UPDATE tokens
            SET narrative_category = %s,
                classification_confidence = 100.0,
                manual_review_status = 'manual_reviewed',
                reviewer = %s,
                review_date = NOW(),
                updated_at = NOW()
            WHERE token_address = %s
        """

        cursor.execute(update_query, (manual_category, reviewer, token_address))

    def _create_manual_review_audit(self, cursor, current_token: Dict[str, Any], new_category: str, reviewer: str, review_notes: str) -> None:
        """Create audit entry for manual review"""
        audit_query = """
            INSERT INTO classification_audit (
                token_address, old_category, new_category,
                old_confidence, new_confidence, reviewer,
                review_reason, automated_suggestion, automated_confidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(audit_query, (
            current_token['token_address'],
            current_token.get('narrative_category'),
            new_category,
            current_token.get('classification_confidence'),
            100.0,  # Manual review gets 100% confidence
            reviewer,
            f"Manual review: {review_notes}" if review_notes else "Manual review",
            current_token.get('narrative_category'),
            current_token.get('classification_confidence')
        ))

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics about manual review progress"""
        try:
            stats_query = """
                SELECT
                    COUNT(*) as total_tokens,
                    COUNT(CASE WHEN manual_review_status = 'pending' THEN 1 END) as pending_review,
                    COUNT(CASE WHEN manual_review_status LIKE 'manual%' OR manual_review_status LIKE 'reviewed%' THEN 1 END) as manually_reviewed,
                    COUNT(CASE WHEN classification_confidence < 50 AND narrative_category IS NOT NULL THEN 1 END) as low_confidence,
                    COUNT(CASE WHEN narrative_category IS NULL THEN 1 END) as unclassified,
                    COUNT(DISTINCT reviewer) as unique_reviewers
                FROM tokens
            """

            stats_result = execute_with_retry(stats_query)
            stats = stats_result[0] if stats_result else {}

            # Get reviewer activity
            reviewer_query = """
                SELECT reviewer, COUNT(*) as tokens_reviewed
                FROM tokens
                WHERE reviewer IS NOT NULL
                GROUP BY reviewer
                ORDER BY tokens_reviewed DESC
            """

            reviewer_result = execute_with_retry(reviewer_query)
            reviewer_activity = {row['reviewer']: row['tokens_reviewed'] for row in reviewer_result or []}

            # Get recent audit activity
            audit_query = """
                SELECT DATE(created_at) as review_date, COUNT(*) as reviews_count
                FROM classification_audit
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY review_date DESC
                LIMIT 10
            """

            audit_result = execute_with_retry(audit_query)
            recent_activity = [
                {"date": str(row['review_date']), "count": row['reviews_count']}
                for row in audit_result or []
            ]

            return {
                "statistics": stats,
                "reviewer_activity": reviewer_activity,
                "recent_activity": recent_activity
            }

        except Exception as e:
            self.logger.log_operation(
                operation="get_review_statistics",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}

    def list_review_files(self) -> List[Dict[str, Any]]:
        """List available review CSV files in the review data directory"""
        try:
            review_files = []

            for file_path in self.review_data_dir.glob("*.csv"):
                if file_path.is_file():
                    stat_info = file_path.stat()
                    review_files.append({
                        "filename": file_path.name,
                        "full_path": str(file_path),
                        "size_bytes": stat_info.st_size,
                        "created_at": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })

            return sorted(review_files, key=lambda x: x['modified_at'], reverse=True)

        except Exception as e:
            self.logger.log_operation(
                operation="list_review_files",
                status="error",
                error=str(e)
            )
            return []