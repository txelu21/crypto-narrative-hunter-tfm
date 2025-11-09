"""Block timing utilities for consistent snapshot scheduling.

This module provides functionality for resolving block numbers at specific
timestamps, ensuring consistent daily snapshot timing across all wallets.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict
from decimal import Decimal

import aiohttp
import structlog
import pytz
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = structlog.get_logger()

# Average Ethereum block time (seconds)
AVERAGE_BLOCK_TIME = 12

# Block timestamp cache to avoid repeated lookups
_block_cache: Dict[int, int] = {}


class BlockTimingError(Exception):
    """Raised when block timing operations fail."""
    pass


class BlockTimingClient:
    """Client for block number resolution and snapshot scheduling."""

    def __init__(
        self,
        api_key: str,
        rate_limit_semaphore: Optional[asyncio.Semaphore] = None,
        timeout: int = 30,
    ):
        """Initialize block timing client.

        Args:
            api_key: Alchemy API key
            rate_limit_semaphore: Optional semaphore for rate limiting
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = f"https://eth-mainnet.g.alchemy.com/v2/{self.api_key}"
        self.rate_limit = rate_limit_semaphore or asyncio.Semaphore(15)
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        self.logger = logger.bind(component="block_timing")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _rpc_call(
        self,
        method: str,
        params: list,
    ) -> dict:
        """Make a JSON-RPC call to Alchemy.

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            RPC response result

        Raises:
            BlockTimingError: On RPC errors
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        async with self.rate_limit:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.base_url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if "error" in data:
                        error_msg = data["error"].get("message", "Unknown error")
                        self.logger.error("rpc_error", method=method, error=error_msg)
                        raise BlockTimingError(f"RPC error: {error_msg}")

                    return data.get("result")

    async def get_block(self, block_number: int) -> dict:
        """Get block information by number.

        Args:
            block_number: Block number to fetch

        Returns:
            Block information including timestamp

        Raises:
            BlockTimingError: If block fetch fails
        """
        # Check cache first
        if block_number in _block_cache:
            return {"number": block_number, "timestamp": _block_cache[block_number]}

        result = await self._rpc_call(
            "eth_getBlockByNumber",
            [hex(block_number), False],
        )

        if result is None:
            raise BlockTimingError(f"Block {block_number} not found")

        timestamp = int(result["timestamp"], 16)
        _block_cache[block_number] = timestamp

        return {
            "number": block_number,
            "timestamp": timestamp,
            "hash": result.get("hash"),
        }

    async def get_latest_block(self) -> dict:
        """Get latest block information.

        Returns:
            Latest block information including number and timestamp
        """
        result = await self._rpc_call(
            "eth_getBlockByNumber",
            ["latest", False],
        )

        if result is None:
            raise BlockTimingError("Failed to fetch latest block")

        block_number = int(result["number"], 16)
        timestamp = int(result["timestamp"], 16)

        # Cache it
        _block_cache[block_number] = timestamp

        return {
            "number": block_number,
            "timestamp": timestamp,
            "hash": result.get("hash"),
        }

    async def get_block_by_timestamp(
        self,
        target_timestamp: int,
        tolerance_seconds: int = 60,
    ) -> dict:
        """Find block number closest to target timestamp using binary search.

        Args:
            target_timestamp: Unix timestamp to find block for
            tolerance_seconds: Maximum acceptable time difference (default: 60s)

        Returns:
            Block information for closest block to target timestamp

        Raises:
            BlockTimingError: If block resolution fails
        """
        log = self.logger.bind(
            target_timestamp=target_timestamp,
            target_datetime=datetime.fromtimestamp(target_timestamp, tz=timezone.utc).isoformat(),
        )

        log.info("resolving_block_by_timestamp")

        # Get latest block for upper bound
        latest = await self.get_latest_block()
        latest_block = latest["number"]
        latest_timestamp = latest["timestamp"]

        # Check if target is in the future
        if target_timestamp > latest_timestamp:
            raise BlockTimingError(
                f"Target timestamp {target_timestamp} is in the future "
                f"(latest: {latest_timestamp})"
            )

        # Estimate starting search range based on average block time
        time_diff = latest_timestamp - target_timestamp
        blocks_back = int(time_diff / AVERAGE_BLOCK_TIME)

        # Set search bounds with safety margin
        upper_block = latest_block
        lower_block = max(0, latest_block - blocks_back - 1000)

        log.info(
            "starting_binary_search",
            lower_block=lower_block,
            upper_block=upper_block,
            estimated_blocks_back=blocks_back,
        )

        # Binary search for closest block
        closest_block = None
        closest_diff = float("inf")

        iteration = 0
        max_iterations = 50  # Safety limit

        while lower_block <= upper_block and iteration < max_iterations:
            iteration += 1
            mid_block = (lower_block + upper_block) // 2

            try:
                block = await self.get_block(mid_block)
                block_timestamp = block["timestamp"]
                diff = abs(block_timestamp - target_timestamp)

                # Track closest match
                if diff < closest_diff:
                    closest_diff = diff
                    closest_block = block

                # Binary search logic
                if block_timestamp < target_timestamp:
                    lower_block = mid_block + 1
                elif block_timestamp > target_timestamp:
                    upper_block = mid_block - 1
                else:
                    # Exact match!
                    log.info(
                        "exact_block_match",
                        block_number=mid_block,
                        timestamp=block_timestamp,
                    )
                    return block

            except Exception as e:
                log.warning(
                    "block_fetch_failed_during_search",
                    block=mid_block,
                    error=str(e),
                )
                # Try to continue search
                upper_block = mid_block - 1

        if closest_block is None:
            raise BlockTimingError("Failed to find any block in search range")

        # Check if closest block is within tolerance
        if closest_diff > tolerance_seconds:
            log.warning(
                "block_tolerance_exceeded",
                closest_block=closest_block["number"],
                time_diff_seconds=closest_diff,
                tolerance=tolerance_seconds,
            )

        log.info(
            "block_resolved",
            block_number=closest_block["number"],
            block_timestamp=closest_block["timestamp"],
            time_diff_seconds=closest_diff,
            iterations=iteration,
        )

        return closest_block

    async def get_end_of_day_block(
        self,
        target_date: datetime,
        timezone_str: str = "UTC",
        hour: int = 23,
        minute: int = 59,
        second: int = 59,
    ) -> dict:
        """Get block number for end of day snapshot.

        Args:
            target_date: Date to get end-of-day block for
            timezone_str: Timezone name (default: "UTC")
            hour: Hour for snapshot (default: 23)
            minute: Minute for snapshot (default: 59)
            second: Second for snapshot (default: 59)

        Returns:
            Block information for end of day

        Raises:
            BlockTimingError: If block resolution fails
        """
        tz = pytz.timezone(timezone_str)

        # Create target datetime
        if target_date.tzinfo is None:
            target_datetime = tz.localize(target_date.replace(
                hour=hour,
                minute=minute,
                second=second,
                microsecond=0,
            ))
        else:
            target_datetime = target_date.astimezone(tz).replace(
                hour=hour,
                minute=minute,
                second=second,
                microsecond=0,
            )

        target_timestamp = int(target_datetime.timestamp())

        self.logger.info(
            "getting_end_of_day_block",
            date=target_date.date().isoformat(),
            timezone=timezone_str,
            target_datetime=target_datetime.isoformat(),
        )

        return await self.get_block_by_timestamp(target_timestamp)

    async def get_start_of_day_block(
        self,
        target_date: datetime,
        timezone_str: str = "UTC",
    ) -> dict:
        """Get block number for start of day snapshot.

        Args:
            target_date: Date to get start-of-day block for
            timezone_str: Timezone name (default: "UTC")

        Returns:
            Block information for start of day
        """
        return await self.get_end_of_day_block(
            target_date,
            timezone_str=timezone_str,
            hour=0,
            minute=0,
            second=0,
        )

    async def get_daily_blocks(
        self,
        start_date: datetime,
        end_date: datetime,
        timezone_str: str = "UTC",
        snapshot_time: str = "end_of_day",
    ) -> Dict[str, dict]:
        """Get block numbers for daily snapshots over a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timezone_str: Timezone name (default: "UTC")
            snapshot_time: Either "end_of_day" or "start_of_day"

        Returns:
            Dictionary mapping date strings (YYYY-MM-DD) to block information

        Raises:
            BlockTimingError: If any block resolution fails
        """
        log = self.logger.bind(
            start_date=start_date.date().isoformat(),
            end_date=end_date.date().isoformat(),
            timezone=timezone_str,
            snapshot_time=snapshot_time,
        )

        log.info("fetching_daily_blocks")

        daily_blocks = {}
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.date().isoformat()

            try:
                if snapshot_time == "end_of_day":
                    block = await self.get_end_of_day_block(current_date, timezone_str)
                else:
                    block = await self.get_start_of_day_block(current_date, timezone_str)

                daily_blocks[date_str] = block

                log.info(
                    "daily_block_resolved",
                    date=date_str,
                    block_number=block["number"],
                )

            except Exception as e:
                log.error(
                    "daily_block_failed",
                    date=date_str,
                    error=str(e),
                )
                raise

            current_date += timedelta(days=1)

        log.info(
            "daily_blocks_complete",
            num_days=len(daily_blocks),
        )

        return daily_blocks

    def clear_cache(self):
        """Clear the block timestamp cache."""
        _block_cache.clear()
        self.logger.info("block_cache_cleared")

    def get_cache_size(self) -> int:
        """Get the number of cached block timestamps."""
        return len(_block_cache)