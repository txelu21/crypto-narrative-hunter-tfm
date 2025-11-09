"""
CoinGecko API client with authentication, rate limiting, and disk caching.

Implements:
- Exponential backoff with jitter for rate limit handling (429 responses)
- Disk caching using JSONL format keyed by URL hash
- Proper HTTP timeouts and TLS verification
- Request/response logging with sanitized headers
- API connectivity and response format validation
"""

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from data_collection.common.config import settings
from data_collection.common.logging_setup import get_logger


class CoinGeckoAPIError(Exception):
    """Base exception for CoinGecko API errors"""
    pass


class CoinGeckoRateLimitError(CoinGeckoAPIError):
    """Exception for rate limit errors (429)"""
    pass


class CoinGeckoClient:
    """
    CoinGecko API client with caching, rate limiting, and authentication.

    Features:
    - Exponential backoff with jitter for 429 responses
    - Disk caching for deterministic re-runs
    - Structured logging for all operations
    - Request/response validation
    """

    def __init__(self, cache_dir: str = "cache/coingecko"):
        self.logger = get_logger("coingecko_client")
        self.base_url = settings.coingecko_base_url
        self.api_key = settings.coingecko_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HTTP client configuration
        self.client = httpx.Client(
            timeout=httpx.Timeout(settings.http_timeout),
            verify=settings.tls_verify,
            headers=self._get_headers()
        )

        self.logger.log_operation(
            operation="client_init",
            params={"cache_dir": str(self.cache_dir)},
            status="completed",
            message="CoinGecko client initialized"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with authentication and user agent"""
        headers = {
            "User-Agent": "crypto-narrative-hunter/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        return headers

    def _get_cache_key(self, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from URL and parameters"""
        cache_input = url
        if params:
            # Sort params for consistent hashing
            sorted_params = urlencode(sorted(params.items()))
            cache_input = f"{url}?{sorted_params}"

        return hashlib.md5(cache_input.encode()).hexdigest()[:16]

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key"""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if it exists and is valid"""
        cache_file = self._get_cache_file(cache_key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if cache entry has timestamp and is not expired (24 hours)
            cache_timestamp = cached_data.get("cached_at", 0)
            if time.time() - cache_timestamp > 24 * 3600:  # 24 hours
                self.logger.log_operation(
                    operation="cache_expired",
                    params={"cache_key": cache_key},
                    status="info",
                    message="Cache entry expired, will fetch fresh data"
                )
                return None

            self.logger.log_operation(
                operation="cache_hit",
                params={"cache_key": cache_key},
                status="completed",
                message="Loaded response from cache"
            )

            return cached_data.get("response")

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.log_operation(
                operation="cache_load_error",
                params={"cache_key": cache_key},
                status="error",
                error=str(e),
                message="Failed to load from cache"
            )
            return None

    def _save_to_cache(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """Save response to cache with timestamp"""
        cache_file = self._get_cache_file(cache_key)

        try:
            cache_entry = {
                "cached_at": time.time(),
                "response": response_data
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2)

            self.logger.log_operation(
                operation="cache_save",
                params={"cache_key": cache_key},
                status="completed",
                message="Saved response to cache"
            )

        except Exception as e:
            self.logger.log_operation(
                operation="cache_save_error",
                params={"cache_key": cache_key},
                status="error",
                error=str(e),
                message="Failed to save to cache"
            )

    def _sanitize_headers_for_logging(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers for logging by removing sensitive information"""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in ["authorization", "x-cg-demo-api-key", "x-api-key"]:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized

    @retry(
        retry=retry_if_exception_type(CoinGeckoRateLimitError),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and rate limiting.

        Raises:
            CoinGeckoRateLimitError: For 429 responses (retryable)
            CoinGeckoAPIError: For other API errors (not retryable)
        """
        start_time = time.time()
        url = f"{self.base_url}{endpoint}"

        # Check cache first
        cache_key = self._get_cache_key(url, params)
        cached_response = self._load_from_cache(cache_key)
        if cached_response is not None:
            return cached_response

        # Add jitter to reduce thundering herd
        jitter_delay = random.uniform(0.1, 0.5)
        time.sleep(jitter_delay)

        try:
            self.logger.log_operation(
                operation="api_request",
                params={"endpoint": endpoint, "params_count": len(params or {})},
                status="started",
                message=f"Making request to {endpoint}"
            )

            response = self.client.get(url, params=params)
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request details (sanitized)
            self.logger.log_operation(
                operation="http_request",
                params={
                    "url": url,
                    "status_code": response.status_code,
                    "headers": self._sanitize_headers_for_logging(dict(response.headers))
                },
                status="completed" if response.status_code == 200 else "error",
                duration_ms=duration_ms
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                self.logger.log_operation(
                    operation="rate_limit_hit",
                    params={"retry_after": retry_after},
                    status="error",
                    duration_ms=duration_ms,
                    error="Rate limit exceeded",
                    message=f"Rate limited, will retry after {retry_after}s"
                )
                # Sleep for the retry-after period before raising exception for retry
                time.sleep(int(retry_after))
                raise CoinGeckoRateLimitError(f"Rate limited, retry after {retry_after}s")

            # Handle client errors (4xx) - these are not retryable
            if 400 <= response.status_code < 500:
                error_msg = f"Client error {response.status_code}: {response.text}"
                self.logger.log_operation(
                    operation="client_error",
                    params={"status_code": response.status_code},
                    status="error",
                    duration_ms=duration_ms,
                    error=error_msg
                )
                raise CoinGeckoAPIError(error_msg)

            # Handle server errors (5xx) - these could be retryable but we'll fail fast
            if response.status_code >= 500:
                error_msg = f"Server error {response.status_code}: {response.text}"
                self.logger.log_operation(
                    operation="server_error",
                    params={"status_code": response.status_code},
                    status="error",
                    duration_ms=duration_ms,
                    error=error_msg
                )
                raise CoinGeckoAPIError(error_msg)

            # Parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON response: {str(e)}"
                self.logger.log_operation(
                    operation="json_parse_error",
                    status="error",
                    duration_ms=duration_ms,
                    error=error_msg
                )
                raise CoinGeckoAPIError(error_msg)

            # Cache successful response
            self._save_to_cache(cache_key, response_data)

            self.logger.log_operation(
                operation="api_request",
                params={"endpoint": endpoint},
                status="completed",
                duration_ms=duration_ms,
                message=f"Successfully fetched data from {endpoint}"
            )

            return response_data

        except httpx.TimeoutException as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Request timeout: {str(e)}"
            self.logger.log_operation(
                operation="request_timeout",
                status="error",
                duration_ms=duration_ms,
                error=error_msg
            )
            raise CoinGeckoAPIError(error_msg)

        except httpx.RequestError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Request error: {str(e)}"
            self.logger.log_operation(
                operation="request_error",
                status="error",
                duration_ms=duration_ms,
                error=error_msg
            )
            raise CoinGeckoAPIError(error_msg)

    def test_connectivity(self) -> bool:
        """Test API connectivity and authentication"""
        try:
            self.logger.log_operation(
                operation="connectivity_test",
                status="started",
                message="Testing CoinGecko API connectivity"
            )

            # Simple ping endpoint
            response = self._make_request("/ping")

            if isinstance(response, dict) and "gecko_says" in response:
                self.logger.log_operation(
                    operation="connectivity_test",
                    status="completed",
                    message="API connectivity test successful"
                )
                return True
            else:
                self.logger.log_operation(
                    operation="connectivity_test",
                    status="error",
                    error="Unexpected response format",
                    message="API connectivity test failed"
                )
                return False

        except Exception as e:
            self.logger.log_operation(
                operation="connectivity_test",
                status="error",
                error=str(e),
                message="API connectivity test failed"
            )
            return False

    def get_coins_markets(self,
                          vs_currency: str = "usd",
                          category: str = "ethereum-ecosystem",
                          order: str = "market_cap_desc",
                          per_page: int = 100,
                          page: int = 1,
                          sparkline: bool = False) -> List[Dict[str, Any]]:
        """
        Get coins market data filtered by category.

        Args:
            vs_currency: Target currency for price data (default: "usd")
            category: Filter by category (default: "ethereum-ecosystem")
            order: Sort order (default: "market_cap_desc")
            per_page: Number of results per page (default: 100, max: 250)
            page: Page number (default: 1)
            sparkline: Include sparkline data (default: False)

        Returns:
            List of coin market data dictionaries
        """
        params = {
            "vs_currency": vs_currency,
            "category": category,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": sparkline
        }

        self.logger.log_operation(
            operation="get_markets",
            params={"page": page, "per_page": per_page, "category": category},
            status="started",
            message=f"Fetching markets page {page}"
        )

        response = self._make_request("/coins/markets", params)

        if not isinstance(response, list):
            error_msg = f"Expected list response, got {type(response)}"
            self.logger.log_operation(
                operation="get_markets",
                status="error",
                error=error_msg
            )
            raise CoinGeckoAPIError(error_msg)

        self.logger.log_operation(
            operation="get_markets",
            params={"page": page, "per_page": per_page},
            status="completed",
            message=f"Fetched {len(response)} tokens from page {page}"
        )

        return response

    def get_coin_by_contract_address(self, contract_address: str) -> Dict[str, Any]:
        """
        Get coin information by contract address for ERC-20 validation.

        Args:
            contract_address: Ethereum contract address

        Returns:
            Coin information dictionary
        """
        # Ethereum platform ID in CoinGecko
        platform_id = "ethereum"
        endpoint = f"/coins/{platform_id}/contract/{contract_address}"

        self.logger.log_operation(
            operation="get_coin_by_contract",
            params={"contract_address": contract_address[:10] + "..."},  # Truncate for privacy
            status="started"
        )

        response = self._make_request(endpoint)

        self.logger.log_operation(
            operation="get_coin_by_contract",
            params={"contract_address": contract_address[:10] + "..."},
            status="completed",
            message="Successfully fetched coin by contract address"
        )

        return response

    def get_coin_details_by_id(self, coin_id: str) -> Dict[str, Any]:
        """
        Get detailed coin information by CoinGecko ID.

        Args:
            coin_id: CoinGecko coin ID

        Returns:
            Detailed coin information dictionary
        """
        endpoint = f"/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }

        self.logger.log_operation(
            operation="get_coin_details",
            params={"coin_id": coin_id},
            status="started"
        )

        response = self._make_request(endpoint, params)

        self.logger.log_operation(
            operation="get_coin_details",
            params={"coin_id": coin_id},
            status="completed",
            message="Successfully fetched coin details by ID"
        )

        return response

    def close(self) -> None:
        """Close the HTTP client"""
        self.client.close()
        self.logger.log_operation(
            operation="client_close",
            status="completed",
            message="CoinGecko client closed"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()