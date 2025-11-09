"""
Block Number Mapping Service

Maps dates to Ethereum block numbers for historical balance queries.
Supports caching to avoid repeated API calls.
"""

from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional
from pathlib import Path
import time

class BlockMapper:
    """
    Service to map dates to Ethereum block numbers.

    Uses average block time (~12 seconds) for estimation, with optional
    API validation for precise block numbers.
    """

    # Ethereum average block time in seconds (post-merge)
    AVG_BLOCK_TIME = 12.0

    # Reference point (known block and timestamp)
    # You'll need to update these with actual values for your study period
    REFERENCE_BLOCK = 20000000  # Example: update with actual block
    REFERENCE_TIMESTAMP = datetime(2024, 5, 1)  # Example: update with actual date

    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize BlockMapper with optional cache file.

        Args:
            cache_file: Path to JSON cache file for storing block mappings
        """
        self.cache_file = cache_file or "data/cache/block_mappings.json"
        self.cache: Dict[str, int] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached block mappings from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"✓ Loaded {len(self.cache)} cached block mappings")
            except Exception as e:
                print(f"⚠ Failed to load block cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save block mappings to cache file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            print(f"✓ Saved {len(self.cache)} block mappings to cache")
        except Exception as e:
            print(f"⚠ Failed to save block cache: {e}")

    def estimate_block_from_date(self, target_date: datetime) -> int:
        """
        Estimate block number from date using average block time.

        Args:
            target_date: Target date to find block for

        Returns:
            Estimated block number
        """
        time_diff = (target_date - self.REFERENCE_TIMESTAMP).total_seconds()
        blocks_diff = int(time_diff / self.AVG_BLOCK_TIME)
        estimated_block = self.REFERENCE_BLOCK + blocks_diff
        return estimated_block

    def date_to_block(
        self,
        date: datetime,
        use_cache: bool = True,
        use_api: bool = False,
        alchemy_client = None
    ) -> int:
        """
        Convert date to block number.

        Args:
            date: Target date
            use_cache: Check cache first
            use_api: Use Alchemy API for precise block (requires alchemy_client)
            alchemy_client: Alchemy Web3 client instance

        Returns:
            Block number closest to the target date
        """
        date_str = date.strftime("%Y-%m-%d")

        # Check cache first
        if use_cache and date_str in self.cache:
            return self.cache[date_str]

        # Use API if available
        if use_api and alchemy_client:
            block = self._get_block_from_api(date, alchemy_client)
        else:
            # Use estimation
            block = self.estimate_block_from_date(date)

        # Cache the result
        self.cache[date_str] = block
        self._save_cache()

        return block

    def _get_block_from_api(self, date: datetime, alchemy_client) -> int:
        """
        Get precise block number from Alchemy API using binary search.

        Args:
            date: Target date
            alchemy_client: Alchemy Web3 client

        Returns:
            Block number closest to target timestamp
        """
        target_timestamp = int(date.timestamp())

        # Start with estimated range
        estimated_block = self.estimate_block_from_date(date)
        search_range = 7200  # ~1 day of blocks

        min_block = max(0, estimated_block - search_range)
        max_block = estimated_block + search_range

        # Binary search for closest block
        best_block = estimated_block
        best_diff = float('inf')

        iterations = 0
        max_iterations = 20

        while min_block <= max_block and iterations < max_iterations:
            mid_block = (min_block + max_block) // 2

            try:
                block_data = alchemy_client.eth.get_block(mid_block)
                block_timestamp = block_data['timestamp']

                diff = abs(block_timestamp - target_timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_block = mid_block

                # Converged (within 1 minute)
                if diff < 60:
                    break

                if block_timestamp < target_timestamp:
                    min_block = mid_block + 1
                else:
                    max_block = mid_block - 1

                iterations += 1
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"⚠ API error at block {mid_block}: {e}")
                # Fall back to estimation
                return estimated_block

        return best_block

    def generate_daily_mapping(
        self,
        start_date: datetime,
        end_date: datetime,
        use_api: bool = False,
        alchemy_client = None
    ) -> Dict[str, int]:
        """
        Generate block mappings for all days in a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            use_api: Use Alchemy API for precise blocks
            alchemy_client: Alchemy client instance

        Returns:
            Dictionary mapping date strings to block numbers
        """
        mappings = {}
        current_date = start_date

        print(f"\n📅 Generating block mappings from {start_date.date()} to {end_date.date()}")

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            block = self.date_to_block(
                current_date,
                use_cache=True,
                use_api=use_api,
                alchemy_client=alchemy_client
            )
            mappings[date_str] = block
            print(f"  {date_str} → Block {block:,}")

            current_date += timedelta(days=1)

        print(f"\n✓ Generated {len(mappings)} block mappings")
        return mappings

    def block_to_date(self, block_number: int) -> datetime:
        """
        Estimate date from block number.

        Args:
            block_number: Block number

        Returns:
            Estimated datetime
        """
        blocks_diff = block_number - self.REFERENCE_BLOCK
        time_diff = blocks_diff * self.AVG_BLOCK_TIME
        estimated_date = self.REFERENCE_TIMESTAMP + timedelta(seconds=time_diff)
        return estimated_date

    def get_study_period_blocks(self) -> Dict[str, int]:
        """
        Get block mappings for the study period (Sept 3 - Oct 3, 2025).

        Returns:
            Dictionary of date -> block mappings
        """
        # Note: These dates are in the future (Oct 2025 from current date perspective)
        # You'll need to update reference block/timestamp or use API when available
        start_date = datetime(2025, 9, 3)
        end_date = datetime(2025, 10, 3)

        return self.generate_daily_mapping(start_date, end_date)


def main():
    """Test block mapper functionality."""
    mapper = BlockMapper()

    # Test estimation
    test_date = datetime(2025, 9, 3)
    block = mapper.estimate_block_from_date(test_date)
    print(f"\nEstimated block for {test_date.date()}: {block:,}")

    # Generate study period mappings
    print("\n" + "="*60)
    print("STUDY PERIOD BLOCK MAPPINGS (Sept 3 - Oct 3, 2025)")
    print("="*60)

    mappings = mapper.get_study_period_blocks()

    print(f"\nFirst day: {list(mappings.keys())[0]} → Block {list(mappings.values())[0]:,}")
    print(f"Last day:  {list(mappings.keys())[-1]} → Block {list(mappings.values())[-1]:,}")
    print(f"\nTotal blocks span: {list(mappings.values())[-1] - list(mappings.values())[0]:,} blocks")

    return mappings


if __name__ == "__main__":
    main()
