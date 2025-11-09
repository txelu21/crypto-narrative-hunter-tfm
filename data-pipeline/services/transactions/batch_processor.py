"""
Batch processing and sharding module for transaction extraction.

Implements efficient parallel processing with:
- Wallet-based sharding
- Block range optimization
- Concurrent processing with rate limiting
- Dynamic batch sizing
- Progress tracking
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes transactions in batches with sharding and concurrency control."""

    def __init__(
        self,
        alchemy_concurrency: int = 15,
        dune_concurrency: int = 2,
        wallet_shard_size: int = 100,
        memory_limit_mb: int = 1024
    ):
        """
        Initialize batch processor.

        Args:
            alchemy_concurrency: Max concurrent Alchemy requests
            dune_concurrency: Max concurrent Dune requests
            wallet_shard_size: Number of wallets per shard
            memory_limit_mb: Memory limit in MB for batch processing
        """
        self.alchemy_semaphore = asyncio.Semaphore(alchemy_concurrency)
        self.dune_semaphore = asyncio.Semaphore(dune_concurrency)
        self.wallet_shard_size = wallet_shard_size
        self.memory_limit_mb = memory_limit_mb

    async def process_wallets(
        self,
        wallet_list: List[str],
        process_func: Callable,
        description: str = "Processing wallets"
    ) -> List[Any]:
        """
        Process wallets in shards with progress tracking.

        Args:
            wallet_list: List of wallet addresses
            process_func: Async function to process each wallet
            description: Description for progress bar

        Returns:
            List of results
        """
        # Create wallet shards
        wallet_shards = [
            wallet_list[i:i + self.wallet_shard_size]
            for i in range(0, len(wallet_list), self.wallet_shard_size)
        ]

        logger.info(f"Processing {len(wallet_list)} wallets in {len(wallet_shards)} shards")

        all_results = []

        # Process shards sequentially to control memory usage
        for shard_idx, shard in enumerate(wallet_shards):
            logger.info(f"Processing shard {shard_idx + 1}/{len(wallet_shards)}")

            shard_results = await self._process_shard(shard, process_func, description)
            all_results.extend(shard_results)

        return all_results

    async def _process_shard(
        self,
        shard: List[str],
        process_func: Callable,
        description: str
    ) -> List[Any]:
        """
        Process a single shard with concurrency control.

        Args:
            shard: List of wallet addresses in shard
            process_func: Async function to process each wallet
            description: Description for progress bar

        Returns:
            List of results
        """
        tasks = []

        for wallet in shard:
            task = self._process_with_semaphore(wallet, process_func)
            tasks.append(task)

        # Execute tasks with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=description):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing wallet: {e}")
                results.append(None)

        return results

    async def _process_with_semaphore(
        self,
        wallet: str,
        process_func: Callable
    ) -> Any:
        """
        Process wallet with appropriate semaphore.

        Args:
            wallet: Wallet address
            process_func: Async function to process wallet

        Returns:
            Processing result
        """
        # Use Alchemy semaphore by default
        async with self.alchemy_semaphore:
            return await process_func(wallet)

    def optimize_block_range(
        self,
        start_block: int,
        end_block: int,
        estimated_tx_density: float = 1.0
    ) -> List[tuple[int, int]]:
        """
        Optimize block ranges for efficient querying.

        Args:
            start_block: Starting block number
            end_block: Ending block number
            estimated_tx_density: Estimated transactions per block

        Returns:
            List of (from_block, to_block) tuples
        """
        # Adaptive window sizing based on transaction density
        if estimated_tx_density < 0.1:
            # Low activity - use large windows
            window_size = 10000
        elif estimated_tx_density < 1.0:
            # Medium activity - use medium windows
            window_size = 2000
        else:
            # High activity - use small windows
            window_size = 500

        block_ranges = []
        current_block = start_block

        while current_block < end_block:
            to_block = min(current_block + window_size, end_block)
            block_ranges.append((current_block, to_block))
            current_block = to_block + 1

        return block_ranges

    async def process_block_ranges(
        self,
        wallet_address: str,
        block_ranges: List[tuple[int, int]],
        fetch_func: Callable
    ) -> List[Any]:
        """
        Process multiple block ranges for a wallet.

        Args:
            wallet_address: Wallet address
            block_ranges: List of (from_block, to_block) tuples
            fetch_func: Async function to fetch data for a block range

        Returns:
            Combined results from all block ranges
        """
        tasks = []

        for from_block, to_block in block_ranges:
            task = fetch_func(wallet_address, from_block, to_block)
            tasks.append(task)

        # Execute all block range queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        combined_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error fetching block range: {result}")
                continue

            if isinstance(result, list):
                combined_results.extend(result)
            else:
                combined_results.append(result)

        return combined_results

    def estimate_memory_usage(self, transaction_count: int) -> int:
        """
        Estimate memory usage for transaction count.

        Args:
            transaction_count: Number of transactions

        Returns:
            Estimated memory in MB
        """
        # Rough estimate: 1KB per transaction
        bytes_per_tx = 1024
        total_bytes = transaction_count * bytes_per_tx
        return total_bytes // (1024 * 1024)

    def should_checkpoint(
        self,
        processed_count: int,
        total_count: int,
        checkpoint_frequency: int = 1000
    ) -> bool:
        """
        Determine if checkpoint should be saved.

        Args:
            processed_count: Number of processed items
            total_count: Total number of items
            checkpoint_frequency: Checkpoint every N items

        Returns:
            True if checkpoint should be saved
        """
        return processed_count > 0 and processed_count % checkpoint_frequency == 0