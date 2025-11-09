"""
Query Error Handler
Handles query failures, timeouts, and implements retry logic with fallback strategies
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

from services.tokens.dune_client import DuneClientError, DuneQueryError, DuneRateLimitError
from common.logging_setup import get_logger

logger = get_logger(__name__)


class ErrorType(Enum):
    QUERY_TIMEOUT = "query_timeout"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    QUERY_SYNTAX_ERROR = "query_syntax_error"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    AUTHENTICATION_FAILED = "authentication_failed"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    RESULT_DOWNLOAD_FAILED = "result_download_failed"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorContext:
    error_type: ErrorType
    original_exception: Exception
    query_type: str
    parameters: Dict[str, Any]
    attempt_number: int
    timestamp: datetime
    execution_time_seconds: float = 0
    additional_info: Dict[str, Any] = None


class QueryErrorClassifier:
    """Classifies errors into retryable and fatal categories"""

    RETRYABLE_ERRORS = {
        ErrorType.QUERY_TIMEOUT,
        ErrorType.RATE_LIMIT_EXCEEDED,
        ErrorType.INTERNAL_SERVER_ERROR,
        ErrorType.RESULT_DOWNLOAD_FAILED,
        ErrorType.NETWORK_ERROR
    }

    FATAL_ERRORS = {
        ErrorType.QUERY_SYNTAX_ERROR,
        ErrorType.INSUFFICIENT_CREDITS,
        ErrorType.AUTHENTICATION_FAILED
    }

    @classmethod
    def classify_error(cls, exception: Exception) -> ErrorType:
        """Classify exception into error type"""
        error_message = str(exception).lower()

        if isinstance(exception, DuneRateLimitError):
            return ErrorType.RATE_LIMIT_EXCEEDED

        if isinstance(exception, DuneQueryError):
            if "timeout" in error_message or "execution timeout" in error_message:
                return ErrorType.QUERY_TIMEOUT
            elif "syntax" in error_message or "sql error" in error_message:
                return ErrorType.QUERY_SYNTAX_ERROR
            elif "credits" in error_message or "credit limit" in error_message:
                return ErrorType.INSUFFICIENT_CREDITS
            elif "authentication" in error_message or "unauthorized" in error_message:
                return ErrorType.AUTHENTICATION_FAILED
            elif "download" in error_message or "result retrieval" in error_message:
                return ErrorType.RESULT_DOWNLOAD_FAILED
            else:
                return ErrorType.UNKNOWN_ERROR

        if isinstance(exception, DuneClientError):
            if "network" in error_message or "connection" in error_message:
                return ErrorType.NETWORK_ERROR
            elif "server error" in error_message or "500" in error_message:
                return ErrorType.INTERNAL_SERVER_ERROR
            else:
                return ErrorType.UNKNOWN_ERROR

        return ErrorType.UNKNOWN_ERROR

    @classmethod
    def is_retryable(cls, error_type: ErrorType) -> bool:
        """Check if error type is retryable"""
        return error_type in cls.RETRYABLE_ERRORS

    @classmethod
    def get_retry_delay(cls, error_type: ErrorType, attempt_number: int) -> float:
        """Get retry delay in seconds based on error type and attempt"""
        base_delays = {
            ErrorType.QUERY_TIMEOUT: 60,  # Wait longer for timeouts
            ErrorType.RATE_LIMIT_EXCEEDED: 120,  # Wait for rate limit reset
            ErrorType.INTERNAL_SERVER_ERROR: 30,
            ErrorType.RESULT_DOWNLOAD_FAILED: 10,
            ErrorType.NETWORK_ERROR: 5
        }

        base_delay = base_delays.get(error_type, 30)

        # Exponential backoff with jitter
        delay = base_delay * (2 ** (attempt_number - 1))
        max_delay = 600  # Maximum 10 minutes

        return min(delay, max_delay)


class FallbackStrategy:
    """Implements fallback strategies for query failures"""

    @staticmethod
    async def cache_fallback(cache_key: str, query_manager) -> Optional[Any]:
        """Try to use cached results as fallback"""
        try:
            cached_result = query_manager.load_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached fallback for {cache_key}")
                return cached_result
        except Exception as e:
            logger.warning(f"Cache fallback failed: {e}")
        return None

    @staticmethod
    async def parameter_adjustment_fallback(
        original_params: Dict[str, Any],
        error_context: ErrorContext
    ) -> Optional[Dict[str, Any]]:
        """Adjust parameters to potentially avoid errors"""

        if error_context.error_type == ErrorType.QUERY_TIMEOUT:
            # Reduce time window for timeout errors
            adjusted_params = original_params.copy()

            # Try reducing the date range
            start_date = datetime.strptime(adjusted_params.get("start_date", "2024-09-20"), "%Y-%m-%d")
            end_date = datetime.strptime(adjusted_params.get("end_date", "2024-09-27"), "%Y-%m-%d")

            # Reduce to half the original window
            date_diff = (end_date - start_date) / 2
            new_end_date = start_date + date_diff

            adjusted_params["end_date"] = new_end_date.strftime("%Y-%m-%d")

            # Increase minimum thresholds to reduce result set
            adjusted_params["min_volume_usd"] = int(adjusted_params.get("min_volume_usd", 10000) * 1.5)
            adjusted_params["min_trade_count"] = int(adjusted_params.get("min_trade_count", 10) * 1.2)

            logger.info(f"Adjusted parameters for timeout: {adjusted_params}")
            return adjusted_params

        elif error_context.error_type == ErrorType.INSUFFICIENT_CREDITS:
            # Use preview parameters to conserve credits
            return {
                "start_date": "2024-09-25",
                "end_date": "2024-09-27",
                "min_volume_usd": 50000,  # Higher threshold
                "min_trade_count": 20
            }

        return None

    @staticmethod
    async def chunked_execution_fallback(
        original_params: Dict[str, Any],
        query_executor: Callable
    ) -> Optional[Any]:
        """Execute query in smaller chunks and combine results"""
        try:
            start_date = datetime.strptime(original_params.get("start_date", "2024-09-20"), "%Y-%m-%d")
            end_date = datetime.strptime(original_params.get("end_date", "2024-09-27"), "%Y-%m-%d")

            # Split into weekly chunks
            chunk_results = []
            current_date = start_date

            while current_date < end_date:
                chunk_end = min(current_date + timedelta(days=7), end_date)

                chunk_params = original_params.copy()
                chunk_params["start_date"] = current_date.strftime("%Y-%m-%d")
                chunk_params["end_date"] = chunk_end.strftime("%Y-%m-%d")

                logger.info(f"Executing chunk: {chunk_params['start_date']} to {chunk_params['end_date']}")

                chunk_result = await query_executor(chunk_params)
                if chunk_result and chunk_result.result_df is not None:
                    chunk_results.append(chunk_result.result_df)

                current_date = chunk_end + timedelta(days=1)

                # Pause between chunks
                await asyncio.sleep(10)

            # Combine results if we have any
            if chunk_results:
                import pandas as pd
                combined_df = pd.concat(chunk_results, ignore_index=True)

                # Remove duplicates if any
                if 'trader_address' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['trader_address'])

                logger.info(f"Chunked execution successful: {len(combined_df)} total results")
                return combined_df

        except Exception as e:
            logger.error(f"Chunked execution fallback failed: {e}")

        return None


class QueryErrorHandler:
    """Main error handler for query execution"""

    def __init__(self, max_retries: int = 3, enable_fallbacks: bool = True):
        self.max_retries = max_retries
        self.enable_fallbacks = enable_fallbacks
        self.error_history: List[ErrorContext] = []

    async def execute_with_retry(
        self,
        query_function: Callable,
        query_type: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute query with retry logic and fallback strategies"""

        last_error = None
        attempt = 1

        while attempt <= self.max_retries:
            try:
                start_time = time.time()

                logger.info(f"Attempting {query_type} execution (attempt {attempt}/{self.max_retries})")

                result = await query_function(parameters=parameters, **kwargs)

                execution_time = time.time() - start_time
                logger.info(f"Query {query_type} succeeded on attempt {attempt} in {execution_time:.1f}s")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                error_type = QueryErrorClassifier.classify_error(e)

                # Record error context
                error_context = ErrorContext(
                    error_type=error_type,
                    original_exception=e,
                    query_type=query_type,
                    parameters=parameters,
                    attempt_number=attempt,
                    timestamp=datetime.now(),
                    execution_time_seconds=execution_time
                )

                self.error_history.append(error_context)
                last_error = error_context

                logger.warning(f"Query {query_type} failed on attempt {attempt}: {error_type.value} - {e}")

                # Check if error is retryable
                if not QueryErrorClassifier.is_retryable(error_type):
                    logger.error(f"Fatal error in {query_type}: {error_type.value}")
                    break

                # Don't retry on last attempt
                if attempt >= self.max_retries:
                    break

                # Wait before retry
                retry_delay = QueryErrorClassifier.get_retry_delay(error_type, attempt)
                logger.info(f"Waiting {retry_delay}s before retry {attempt + 1}")
                await asyncio.sleep(retry_delay)

                # Try parameter adjustment for next attempt
                if error_type in [ErrorType.QUERY_TIMEOUT, ErrorType.INSUFFICIENT_CREDITS]:
                    adjusted_params = await FallbackStrategy.parameter_adjustment_fallback(
                        parameters, error_context
                    )
                    if adjusted_params:
                        parameters = adjusted_params
                        logger.info("Using adjusted parameters for retry")

                attempt += 1

        # If all retries failed, try fallback strategies
        if self.enable_fallbacks and last_error:
            logger.info(f"All retries failed for {query_type}, trying fallback strategies")

            # Try cache fallback if available
            if hasattr(kwargs.get('query_manager'), 'load_from_cache'):
                cache_key = f"{query_type}_{hash(str(sorted(parameters.items())))}"
                cached_result = await FallbackStrategy.cache_fallback(
                    cache_key, kwargs.get('query_manager')
                )
                if cached_result:
                    return cached_result

            # Try chunked execution for timeout errors
            if last_error.error_type == ErrorType.QUERY_TIMEOUT:
                chunked_result = await FallbackStrategy.chunked_execution_fallback(
                    parameters, query_function
                )
                if chunked_result:
                    return chunked_result

        # All attempts and fallbacks failed
        if last_error:
            logger.error(f"Query {query_type} failed after all retry attempts and fallbacks")
            raise last_error.original_exception
        else:
            raise Exception(f"Query {query_type} failed for unknown reasons")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error history"""
        if not self.error_history:
            return {"total_errors": 0}

        error_counts = {}
        for error in self.error_history:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        recent_errors = [
            {
                "query_type": error.query_type,
                "error_type": error.error_type.value,
                "timestamp": error.timestamp.isoformat(),
                "attempt": error.attempt_number
            }
            for error in self.error_history[-10:]  # Last 10 errors
        ]

        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "recent_errors": recent_errors,
            "most_common_error": max(error_counts, key=error_counts.get) if error_counts else None
        }

    def reset_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        logger.info("Error history cleared")