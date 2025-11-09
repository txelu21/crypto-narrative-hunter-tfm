"""
Chainlink Price Feed Integration
Implements on-chain ETH/USD price data collection via Chainlink aggregators
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import Web3Exception

logger = logging.getLogger(__name__)


class ChainlinkPriceClient:
    """
    Client for fetching ETH/USD prices from Chainlink on-chain aggregators

    Features:
    - Hourly price collection with block correlation
    - Historical price backfill capability
    - Price caching for optimization
    - Health monitoring and validation
    """

    # Chainlink ETH/USD aggregator address on Ethereum mainnet
    ETH_USD_AGGREGATOR = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"

    # Chainlink aggregator ABI (minimal interface)
    AGGREGATOR_ABI = [
        {
            "inputs": [],
            "name": "latestRoundData",
            "outputs": [
                {"name": "roundId", "type": "uint80"},
                {"name": "answer", "type": "int256"},
                {"name": "startedAt", "type": "uint256"},
                {"name": "updatedAt", "type": "uint256"},
                {"name": "answeredInRound", "type": "uint80"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]

    def __init__(self, web3_provider: str, aggregator_address: Optional[str] = None):
        """
        Initialize Chainlink client

        Args:
            web3_provider: Web3 provider URL (e.g., Alchemy, Infura)
            aggregator_address: Optional custom aggregator address
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))

        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")

        # Initialize contract
        aggregator_addr = aggregator_address or self.ETH_USD_AGGREGATOR
        self.eth_usd_aggregator: Contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(aggregator_addr),
            abi=self.AGGREGATOR_ABI
        )

        # Get decimals for price conversion
        try:
            self.decimals = self.eth_usd_aggregator.functions.decimals().call()
        except Exception as e:
            logger.warning(f"Could not fetch decimals, defaulting to 8: {e}")
            self.decimals = 8

        # Price cache: {timestamp -> price_data}
        self.price_cache: Dict[int, Dict] = {}

        logger.info(f"Chainlink client initialized with aggregator {aggregator_addr}")

    async def get_eth_price_at_timestamp(self, timestamp: int) -> Optional[Dict]:
        """
        Get ETH/USD price for specific timestamp with caching

        Args:
            timestamp: Unix timestamp

        Returns:
            Dict with price data or None if not available
        """
        # Round to nearest hour for consistency
        hour_timestamp = int(timestamp // 3600) * 3600

        # Check cache first
        if hour_timestamp in self.price_cache:
            logger.debug(f"Cache hit for timestamp {hour_timestamp}")
            return self.price_cache[hour_timestamp]

        # Find block closest to target timestamp
        try:
            target_block = await self.find_block_by_timestamp(hour_timestamp)

            if target_block is None:
                logger.warning(f"Could not find block for timestamp {hour_timestamp}")
                return None

            # Get price from Chainlink aggregator
            price_data = await self._fetch_price_at_block(target_block, hour_timestamp)

            if price_data:
                # Cache the result
                self.price_cache[hour_timestamp] = price_data
                logger.info(f"Fetched price ${price_data['price_usd']:.2f} for timestamp {hour_timestamp}")

            return price_data

        except Exception as e:
            logger.error(f"Error fetching price for timestamp {hour_timestamp}: {e}")
            return None

    async def _fetch_price_at_block(self, block_number: int, timestamp: int) -> Optional[Dict]:
        """
        Fetch price from Chainlink at specific block

        Args:
            block_number: Ethereum block number
            timestamp: Unix timestamp for the price

        Returns:
            Dict with price data or None
        """
        try:
            result = self.eth_usd_aggregator.functions.latestRoundData().call(
                block_identifier=block_number
            )

            # Extract price from Chainlink response
            # result = (roundId, answer, startedAt, updatedAt, answeredInRound)
            round_id, answer, started_at, updated_at, answered_in_round = result

            # Convert to USD (Chainlink ETH/USD typically has 8 decimals)
            price_usd = float(answer) / (10 ** self.decimals)

            # Validate price is reasonable
            if price_usd <= 0 or price_usd > 1000000:
                logger.warning(f"Suspicious price detected: ${price_usd} at block {block_number}")
                return None

            return {
                'price_usd': price_usd,
                'timestamp': timestamp,
                'block_number': block_number,
                'source': 'chainlink',
                'round_id': round_id,
                'updated_at': updated_at,
                'confidence_score': 1.0  # Chainlink is primary source
            }

        except Web3Exception as e:
            logger.error(f"Web3 error fetching price at block {block_number}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching price at block {block_number}: {e}")
            return None

    async def find_block_by_timestamp(self, target_timestamp: int) -> Optional[int]:
        """
        Find Ethereum block number closest to target timestamp using binary search

        Args:
            target_timestamp: Unix timestamp

        Returns:
            Block number or None if not found
        """
        try:
            # Get latest block
            latest_block = self.w3.eth.get_block('latest')
            latest_block_number = latest_block['number']
            latest_timestamp = latest_block['timestamp']

            # Check if target is in the future
            if target_timestamp > latest_timestamp:
                logger.warning(f"Target timestamp {target_timestamp} is in the future")
                return latest_block_number

            # Binary search for the block
            left = 0
            right = latest_block_number
            closest_block = None
            min_diff = float('inf')

            # Average block time ~12 seconds
            avg_block_time = 12

            while right - left > 1:
                # Estimate block number
                time_diff = target_timestamp - self.w3.eth.get_block(left)['timestamp']
                estimated_blocks = time_diff // avg_block_time
                mid = min(left + estimated_blocks, right)
                mid = max(mid, left + 1)

                mid_block = self.w3.eth.get_block(mid)
                mid_timestamp = mid_block['timestamp']

                diff = abs(mid_timestamp - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_block = mid

                if mid_timestamp < target_timestamp:
                    left = mid
                elif mid_timestamp > target_timestamp:
                    right = mid
                else:
                    return mid

            # Check final candidates
            for block_num in [left, right]:
                block = self.w3.eth.get_block(block_num)
                diff = abs(block['timestamp'] - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_block = block_num

            logger.debug(f"Found block {closest_block} for timestamp {target_timestamp} (diff: {min_diff}s)")
            return closest_block

        except Exception as e:
            logger.error(f"Error finding block for timestamp {target_timestamp}: {e}")
            return None

    async def backfill_historical_prices(
        self,
        start_timestamp: int,
        end_timestamp: int,
        interval_seconds: int = 3600
    ) -> List[Dict]:
        """
        Backfill historical prices for date range

        Args:
            start_timestamp: Start unix timestamp
            end_timestamp: End unix timestamp
            interval_seconds: Interval between price points (default: 1 hour)

        Returns:
            List of price data dicts
        """
        logger.info(f"Starting backfill from {start_timestamp} to {end_timestamp}")

        current_timestamp = start_timestamp
        price_data = []
        failed_timestamps = []

        while current_timestamp <= end_timestamp:
            try:
                price_info = await self.get_eth_price_at_timestamp(current_timestamp)

                if price_info:
                    price_data.append(price_info)
                else:
                    failed_timestamps.append(current_timestamp)

                # Progress to next interval
                current_timestamp += interval_seconds

                # Rate limiting - small delay to avoid overwhelming RPC
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to get price for {current_timestamp}: {e}")
                failed_timestamps.append(current_timestamp)
                current_timestamp += interval_seconds

        logger.info(
            f"Backfill complete. Collected {len(price_data)} prices, "
            f"failed {len(failed_timestamps)} timestamps"
        )

        if failed_timestamps:
            logger.warning(f"Failed timestamps: {failed_timestamps[:10]}...")

        return price_data

    async def get_price_feed_health(self) -> Dict:
        """
        Check health status of Chainlink price feed

        Returns:
            Dict with health metrics
        """
        try:
            result = self.eth_usd_aggregator.functions.latestRoundData().call()
            round_id, answer, started_at, updated_at, answered_in_round = result

            current_time = int(datetime.now().timestamp())
            staleness = current_time - updated_at

            # Price feed is considered healthy if updated within last hour
            is_healthy = staleness < 3600

            price_usd = float(answer) / (10 ** self.decimals)

            return {
                'is_healthy': is_healthy,
                'latest_price': price_usd,
                'last_updated': updated_at,
                'staleness_seconds': staleness,
                'round_id': round_id,
                'aggregator_address': self.eth_usd_aggregator.address
            }

        except Exception as e:
            logger.error(f"Error checking price feed health: {e}")
            return {
                'is_healthy': False,
                'error': str(e)
            }

    def clear_cache(self):
        """Clear the price cache"""
        self.price_cache.clear()
        logger.info("Price cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached price entries"""
        return len(self.price_cache)