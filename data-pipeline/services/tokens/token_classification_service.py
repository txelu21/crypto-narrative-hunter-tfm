"""
Token classification service for narrative categorization.

Implements:
- Batch token classification with the narrative classifier
- Database updates for narrative categories and confidence scores
- Progress tracking for large token datasets
- Error handling and recovery for failed classifications
- Statistics and reporting for classification results
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from data_collection.common.db import get_cursor, execute_with_retry
from data_collection.common.logging_setup import get_logger
from .narrative_classifier import NarrativeClassifier, ClassificationResult, NarrativeCategory


@dataclass
class ClassificationStats:
    """Statistics for token classification operation"""
    total_tokens: int = 0
    classified_tokens: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    manual_review_required: int = 0
    classification_errors: int = 0
    database_updates: int = 0
    duration_seconds: float = 0
    category_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.category_distribution is None:
            self.category_distribution = {}


class TokenClassificationService:
    """
    Service for classifying tokens and updating narrative categories in the database.
    """

    def __init__(self):
        self.classifier = NarrativeClassifier()
        self.logger = get_logger("token_classification_service")

    def classify_all_tokens(self, batch_size: int = 100) -> ClassificationStats:
        """
        Classify all tokens in the database and update narrative categories.

        Args:
            batch_size: Number of tokens to process per batch

        Returns:
            ClassificationStats with operation results
        """
        start_time = time.time()
        stats = ClassificationStats()

        self.logger.log_operation(
            operation="classify_all_tokens",
            params={"batch_size": batch_size},
            status="started",
            message="Starting classification of all tokens"
        )

        try:
            # Get all tokens that need classification
            tokens = self._get_tokens_for_classification()
            stats.total_tokens = len(tokens)

            self.logger.log_operation(
                operation="fetch_tokens",
                params={"token_count": stats.total_tokens},
                status="completed",
                message=f"Found {stats.total_tokens} tokens for classification"
            )

            # Process tokens in batches
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i + batch_size]
                batch_stats = self._classify_token_batch(batch)
                self._merge_stats(stats, batch_stats)

                self.logger.log_operation(
                    operation="classify_batch",
                    params={
                        "batch_number": (i // batch_size) + 1,
                        "batch_size": len(batch),
                        "classified": batch_stats.classified_tokens,
                        "errors": batch_stats.classification_errors
                    },
                    status="completed",
                    message=f"Processed batch {(i // batch_size) + 1}"
                )

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="classify_all_tokens",
                params={
                    "total_tokens": stats.total_tokens,
                    "classified": stats.classified_tokens,
                    "high_confidence": stats.high_confidence,
                    "manual_review": stats.manual_review_required
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Classification completed: {stats.classified_tokens}/{stats.total_tokens} tokens classified"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="classify_all_tokens",
                status="error",
                error=str(e),
                duration_ms=duration_ms,
                message="Token classification failed"
            )
            raise

    def classify_specific_tokens(self, token_addresses: List[str]) -> ClassificationStats:
        """
        Classify specific tokens by their addresses.

        Args:
            token_addresses: List of token contract addresses

        Returns:
            ClassificationStats with operation results
        """
        start_time = time.time()
        stats = ClassificationStats()

        try:
            # Get specific tokens from database
            tokens = self._get_tokens_by_addresses(token_addresses)
            stats.total_tokens = len(tokens)

            # Classify and update
            batch_stats = self._classify_token_batch(tokens)
            self._merge_stats(stats, batch_stats)

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="classify_specific_tokens",
                params={
                    "requested_tokens": len(token_addresses),
                    "found_tokens": stats.total_tokens,
                    "classified": stats.classified_tokens
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Classified {stats.classified_tokens} specific tokens"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="classify_specific_tokens",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def _get_tokens_for_classification(self) -> List[Dict[str, Any]]:
        """Get all tokens that need classification from the database"""
        query = """
            SELECT token_address, symbol, name, decimals,
                   market_cap_rank, narrative_category
            FROM tokens
            WHERE narrative_category IS NULL
               OR narrative_category = ''
            ORDER BY market_cap_rank ASC NULLS LAST
        """

        try:
            result = execute_with_retry(query)
            return result or []
        except Exception as e:
            self.logger.log_operation(
                operation="get_tokens_for_classification",
                status="error",
                error=str(e)
            )
            return []

    def _get_tokens_by_addresses(self, addresses: List[str]) -> List[Dict[str, Any]]:
        """Get specific tokens by their addresses"""
        if not addresses:
            return []

        placeholders = ', '.join(['%s'] * len(addresses))
        query = f"""
            SELECT token_address, symbol, name, decimals,
                   market_cap_rank, narrative_category
            FROM tokens
            WHERE token_address IN ({placeholders})
            ORDER BY market_cap_rank ASC NULLS LAST
        """

        try:
            result = execute_with_retry(query, tuple(addresses))
            return result or []
        except Exception as e:
            self.logger.log_operation(
                operation="get_tokens_by_addresses",
                status="error",
                error=str(e)
            )
            return []

    def _classify_token_batch(self, tokens: List[Dict[str, Any]]) -> ClassificationStats:
        """Classify a batch of tokens and update the database"""
        batch_stats = ClassificationStats()
        batch_stats.total_tokens = len(tokens)
        batch_stats.category_distribution = {cat.value: 0 for cat in NarrativeCategory}

        try:
            with get_cursor() as cur:
                for token in tokens:
                    try:
                        # Perform classification
                        result = self.classifier.classify_token(
                            token['token_address'],
                            token['symbol'],
                            token['name']
                        )

                        # Update statistics
                        batch_stats.classified_tokens += 1
                        batch_stats.category_distribution[result.category.value] += 1

                        # Categorize by confidence level
                        if result.confidence >= 80.0:
                            batch_stats.high_confidence += 1
                        elif result.confidence >= 50.0:
                            batch_stats.medium_confidence += 1
                        else:
                            batch_stats.low_confidence += 1

                        if self.classifier.is_manual_review_required(result):
                            batch_stats.manual_review_required += 1

                        # Update database
                        self._update_token_classification(cur, token['token_address'], result)
                        batch_stats.database_updates += 1

                    except Exception as e:
                        batch_stats.classification_errors += 1
                        self.logger.log_operation(
                            operation="classify_single_token",
                            params={"token_address": token['token_address'][:10] + "..."},
                            status="error",
                            error=str(e),
                            message="Failed to classify individual token"
                        )

            return batch_stats

        except Exception as e:
            self.logger.log_operation(
                operation="classify_token_batch",
                status="error",
                error=str(e),
                message="Batch classification failed"
            )
            raise

    def _update_token_classification(self, cursor, token_address: str, result: ClassificationResult) -> None:
        """Update token classification in the database"""
        # Determine manual review status (boolean)
        requires_review = self.classifier.is_manual_review_required(result)

        update_query = """
            UPDATE tokens
            SET narrative_category = %s,
                classification_confidence = %s,
                requires_manual_review = %s,
                updated_at = NOW()
            WHERE token_address = %s
        """

        cursor.execute(update_query, (
            result.category.value,
            result.confidence,
            requires_review,
            token_address
        ))

    def _merge_stats(self, main_stats: ClassificationStats, batch_stats: ClassificationStats) -> None:
        """Merge batch statistics into main statistics"""
        main_stats.classified_tokens += batch_stats.classified_tokens
        main_stats.high_confidence += batch_stats.high_confidence
        main_stats.medium_confidence += batch_stats.medium_confidence
        main_stats.low_confidence += batch_stats.low_confidence
        main_stats.manual_review_required += batch_stats.manual_review_required
        main_stats.classification_errors += batch_stats.classification_errors
        main_stats.database_updates += batch_stats.database_updates

        # Merge category distributions
        if main_stats.category_distribution is None:
            main_stats.category_distribution = {}

        for category, count in batch_stats.category_distribution.items():
            main_stats.category_distribution[category] = main_stats.category_distribution.get(category, 0) + count

    def get_classification_summary(self) -> Dict[str, Any]:
        """Get summary of current classification state in the database"""
        try:
            # Get basic classification statistics
            stats_query = """
                SELECT
                    COUNT(*) as total_tokens,
                    COUNT(narrative_category) as classified_tokens,
                    COUNT(CASE WHEN classification_confidence >= 80 THEN 1 END) as high_confidence,
                    COUNT(CASE WHEN classification_confidence BETWEEN 50 AND 79.99 THEN 1 END) as medium_confidence,
                    COUNT(CASE WHEN classification_confidence < 50 THEN 1 END) as low_confidence,
                    COUNT(CASE WHEN requires_manual_review = true THEN 1 END) as manual_review_required
                FROM tokens
            """

            stats_result = execute_with_retry(stats_query)
            stats = stats_result[0] if stats_result else {}

            # Get category distribution
            distribution_query = """
                SELECT narrative_category, COUNT(*) as count
                FROM tokens
                WHERE narrative_category IS NOT NULL
                GROUP BY narrative_category
                ORDER BY count DESC
            """

            distribution_result = execute_with_retry(distribution_query)
            distribution = {row['narrative_category']: row['count'] for row in distribution_result or []}

            return {
                "statistics": stats,
                "category_distribution": distribution,
                "completeness_rate": (stats.get('classified_tokens', 0) / stats.get('total_tokens', 1)) * 100
            }

        except Exception as e:
            self.logger.log_operation(
                operation="get_classification_summary",
                status="error",
                error=str(e)
            )
            return {"error": str(e)}

    def reclassify_low_confidence_tokens(self, confidence_threshold: float = 50.0) -> ClassificationStats:
        """
        Reclassify tokens with confidence below threshold.

        Args:
            confidence_threshold: Minimum confidence threshold for reclassification

        Returns:
            ClassificationStats with reclassification results
        """
        start_time = time.time()

        # Get tokens below confidence threshold
        query = """
            SELECT token_address, symbol, name, decimals,
                   market_cap_rank, narrative_category, classification_confidence
            FROM tokens
            WHERE classification_confidence < %s
               OR classification_confidence IS NULL
            ORDER BY market_cap_rank ASC NULLS LAST
        """

        try:
            tokens = execute_with_retry(query, (confidence_threshold,))

            if not tokens:
                return ClassificationStats(duration_seconds=time.time() - start_time)

            stats = ClassificationStats()
            stats.total_tokens = len(tokens)

            # Reclassify tokens
            batch_stats = self._classify_token_batch(tokens)
            self._merge_stats(stats, batch_stats)

            stats.duration_seconds = time.time() - start_time

            self.logger.log_operation(
                operation="reclassify_low_confidence",
                params={
                    "threshold": confidence_threshold,
                    "tokens_reclassified": stats.classified_tokens
                },
                status="completed",
                duration_ms=int(stats.duration_seconds * 1000),
                message=f"Reclassified {stats.classified_tokens} low-confidence tokens"
            )

            return stats

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="reclassify_low_confidence",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise