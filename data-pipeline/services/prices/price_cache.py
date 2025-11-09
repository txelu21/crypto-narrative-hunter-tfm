"""
Price Caching and Performance Optimization
Implements intelligent caching, batch processing, and RPC optimization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class PriceCache:
    """
    Multi-tier caching system for price data

    Features:
    - In-memory cache with TTL
    - Preloading for common queries
    - Batch request optimization
    - Cache statistics and monitoring
    """

    def __init__(
        self,
        cache_ttl: int = 3600,
        max_memory_items: int = 10000,
        redis_client=None
    ):
        """
        Initialize price cache

        Args:
            cache_ttl: Time-to-live for cache entries (seconds)
            max_memory_items: Maximum items in memory cache
            redis_client: Optional Redis client for distributed caching
        """
        self.cache_ttl = cache_ttl
        self.max_memory_items = max_memory_items
        self.redis = redis_client

        # In-memory cache
        self.memory_cache: Dict[int, Dict] = {}
        self.cache_timestamps: Dict[int, int] = {}

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'redis_hits': 0,
            'database_queries': 0,
            'evictions': 0
        }

        logger.info(
            f"Price cache initialized: TTL={cache_ttl}s, "
            f"max_items={max_memory_items}, redis={'enabled' if redis_client else 'disabled'}"
        )

    async def get(self, timestamp: int) -> Optional[Dict]:
        """
        Get price from cache

        Args:
            timestamp: Unix timestamp

        Returns:
            Price data or None
        """
        # Round to hour for consistency
        hour_timestamp = int(timestamp // 3600) * 3600

        # Check memory cache
        if hour_timestamp in self.memory_cache:
            # Check if expired
            if self._is_expired(hour_timestamp):
                self._evict(hour_timestamp)
            else:
                self.stats['hits'] += 1
                return self.memory_cache[hour_timestamp]

        # Check Redis cache
        if self.redis:
            redis_data = await self._get_from_redis(hour_timestamp)
            if redis_data:
                self.stats['redis_hits'] += 1
                # Store in memory cache
                self._store_in_memory(hour_timestamp, redis_data)
                return redis_data

        self.stats['misses'] += 1
        return None

    async def set(self, timestamp: int, price_data: Dict):
        """
        Store price in cache

        Args:
            timestamp: Unix timestamp
            price_data: Price data dict
        """
        hour_timestamp = int(timestamp // 3600) * 3600

        # Store in memory
        self._store_in_memory(hour_timestamp, price_data)

        # Store in Redis
        if self.redis:
            await self._store_in_redis(hour_timestamp, price_data)

    async def get_batch(self, timestamps: List[int]) -> Dict[int, Dict]:
        """
        Get multiple prices in batch

        Args:
            timestamps: List of unix timestamps

        Returns:
            Dict of {timestamp -> price_data}
        """
        results = {}
        missing_timestamps = []

        for timestamp in timestamps:
            cached = await self.get(timestamp)
            if cached:
                results[timestamp] = cached
            else:
                missing_timestamps.append(timestamp)

        return results, missing_timestamps

    async def set_batch(self, price_data: Dict[int, Dict]):
        """
        Store multiple prices in batch

        Args:
            price_data: Dict of {timestamp -> price_data}
        """
        for timestamp, data in price_data.items():
            await self.set(timestamp, data)

        logger.debug(f"Cached {len(price_data)} prices in batch")

    async def preload_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        fetcher_func
    ) -> int:
        """
        Preload price range into cache

        Args:
            start_timestamp: Start unix timestamp
            end_timestamp: End unix timestamp
            fetcher_func: Async function to fetch missing prices

        Returns:
            Number of prices loaded
        """
        # Generate hourly timestamps
        current = int(start_timestamp // 3600) * 3600
        end = int(end_timestamp // 3600) * 3600
        timestamps = []

        while current <= end:
            timestamps.append(current)
            current += 3600

        # Check which are missing
        cached_data, missing = await self.get_batch(timestamps)

        if missing:
            logger.info(f"Preloading {len(missing)} missing prices")

            # Fetch missing prices
            fetched_data = await fetcher_func(missing)

            # Cache fetched data
            await self.set_batch(fetched_data)

            return len(fetched_data)

        return 0

    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Memory cache cleared")

    def get_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Cache statistics dict
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'redis_hits': self.stats['redis_hits'],
            'database_queries': self.stats['database_queries'],
            'evictions': self.stats['evictions'],
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_max': self.max_memory_items
        }

    def _store_in_memory(self, timestamp: int, price_data: Dict):
        """Store price in memory cache"""
        # Check if we need to evict
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_oldest()

        self.memory_cache[timestamp] = price_data
        self.cache_timestamps[timestamp] = int(datetime.now().timestamp())

    def _is_expired(self, timestamp: int) -> bool:
        """Check if cache entry is expired"""
        if timestamp not in self.cache_timestamps:
            return True

        cached_at = self.cache_timestamps[timestamp]
        age = int(datetime.now().timestamp()) - cached_at
        return age > self.cache_ttl

    def _evict(self, timestamp: int):
        """Evict entry from cache"""
        if timestamp in self.memory_cache:
            del self.memory_cache[timestamp]
            del self.cache_timestamps[timestamp]
            self.stats['evictions'] += 1

    def _evict_oldest(self):
        """Evict oldest cache entry"""
        if not self.cache_timestamps:
            return

        oldest_timestamp = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
        self._evict(oldest_timestamp)

    async def _get_from_redis(self, timestamp: int) -> Optional[Dict]:
        """Get price from Redis"""
        try:
            key = f"eth_price:{timestamp}"
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")

        return None

    async def _store_in_redis(self, timestamp: int, price_data: Dict):
        """Store price in Redis"""
        try:
            key = f"eth_price:{timestamp}"
            await self.redis.setex(
                key,
                self.cache_ttl,
                json.dumps(price_data)
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")


class BatchRPCOptimizer:
    """
    Optimize RPC calls through batching and parallel processing

    Features:
    - Batch request aggregation
    - Parallel processing
    - Rate limiting
    - Cost monitoring
    """

    def __init__(
        self,
        max_batch_size: int = 100,
        max_parallel: int = 5,
        rate_limit_per_second: int = 10
    ):
        """
        Initialize batch optimizer

        Args:
            max_batch_size: Maximum requests per batch
            max_parallel: Maximum parallel batches
            rate_limit_per_second: Rate limit for requests
        """
        self.max_batch_size = max_batch_size
        self.max_parallel = max_parallel
        self.rate_limit = rate_limit_per_second

        self.stats = {
            'total_requests': 0,
            'batches_executed': 0,
            'rpc_calls_saved': 0
        }

        logger.info(
            f"Batch optimizer initialized: batch_size={max_batch_size}, "
            f"parallel={max_parallel}, rate_limit={rate_limit_per_second}/s"
        )

    async def execute_batch(
        self,
        requests: List[Tuple],
        executor_func
    ) -> List:
        """
        Execute requests in optimized batches

        Args:
            requests: List of request tuples
            executor_func: Async function to execute single request

        Returns:
            List of results
        """
        total_requests = len(requests)
        self.stats['total_requests'] += total_requests

        # Split into batches
        batches = [
            requests[i:i + self.max_batch_size]
            for i in range(0, len(requests), self.max_batch_size)
        ]

        logger.info(
            f"Executing {total_requests} requests in {len(batches)} batches"
        )

        all_results = []

        # Process batches with parallelism and rate limiting
        for i in range(0, len(batches), self.max_parallel):
            parallel_batches = batches[i:i + self.max_parallel]

            # Execute batches in parallel
            batch_results = await asyncio.gather(*[
                self._execute_single_batch(batch, executor_func)
                for batch in parallel_batches
            ])

            for results in batch_results:
                all_results.extend(results)

            self.stats['batches_executed'] += len(parallel_batches)

            # Rate limiting delay
            if i + self.max_parallel < len(batches):
                delay = 1.0 / self.rate_limit * self.max_parallel
                await asyncio.sleep(delay)

        # Calculate RPC calls saved
        calls_saved = total_requests - self.stats['batches_executed']
        self.stats['rpc_calls_saved'] += max(0, calls_saved)

        return all_results

    async def _execute_single_batch(
        self,
        batch: List[Tuple],
        executor_func
    ) -> List:
        """Execute single batch of requests"""
        results = []

        for request in batch:
            try:
                result = await executor_func(*request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch request error: {e}")
                results.append(None)

        return results

    def get_stats(self) -> Dict:
        """Get optimization statistics"""
        return {
            'total_requests': self.stats['total_requests'],
            'batches_executed': self.stats['batches_executed'],
            'rpc_calls_saved': self.stats['rpc_calls_saved'],
            'efficiency': (
                self.stats['rpc_calls_saved'] / self.stats['total_requests']
                if self.stats['total_requests'] > 0 else 0
            )
        }


class StreamingPriceProcessor:
    """
    Memory-efficient streaming processor for large datasets

    Features:
    - Streaming data processing
    - Memory-efficient iteration
    - Chunked processing
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Initialize streaming processor

        Args:
            chunk_size: Size of processing chunks
        """
        self.chunk_size = chunk_size
        logger.info(f"Streaming processor initialized with chunk_size={chunk_size}")

    async def process_stream(
        self,
        data_source,
        processor_func
    ):
        """
        Process data in streaming fashion

        Args:
            data_source: Iterator or generator of data
            processor_func: Async function to process each chunk

        Yields:
            Processed results
        """
        chunk = []

        async for item in data_source:
            chunk.append(item)

            if len(chunk) >= self.chunk_size:
                # Process chunk
                results = await processor_func(chunk)
                for result in results:
                    yield result

                # Clear chunk
                chunk = []

        # Process remaining items
        if chunk:
            results = await processor_func(chunk)
            for result in results:
                yield result


class RpcCostMonitor:
    """
    Monitor and track RPC usage costs

    Features:
    - Call counting
    - Cost estimation
    - Budget tracking
    """

    def __init__(self, cost_per_call: float = 0.0001):
        """
        Initialize cost monitor

        Args:
            cost_per_call: Estimated cost per RPC call
        """
        self.cost_per_call = cost_per_call
        self.call_counts = {
            'eth_call': 0,
            'eth_getBlock': 0,
            'eth_getBlockByNumber': 0,
            'total': 0
        }

    def record_call(self, method: str):
        """Record an RPC call"""
        if method in self.call_counts:
            self.call_counts[method] += 1
        self.call_counts['total'] += 1

    def get_cost_estimate(self) -> Dict:
        """Get cost estimates"""
        total_cost = self.call_counts['total'] * self.cost_per_call

        return {
            'total_calls': self.call_counts['total'],
            'estimated_cost_usd': total_cost,
            'call_breakdown': {
                k: v for k, v in self.call_counts.items() if k != 'total'
            },
            'cost_per_call': self.cost_per_call
        }

    def reset(self):
        """Reset counters"""
        for key in self.call_counts:
            self.call_counts[key] = 0