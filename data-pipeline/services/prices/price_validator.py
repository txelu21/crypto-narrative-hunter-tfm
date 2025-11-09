"""
Multi-Source Price Validation and Fallback System
Cross-validates prices from Chainlink, CoinGecko, and Uniswap
"""

import logging
import statistics
from typing import Dict, List, Optional
from services.prices.chainlink_client import ChainlinkPriceClient
from services.prices.coingecko_client import CoinGeckoClient
from services.prices.uniswap_client import UniswapV3Client

logger = logging.getLogger(__name__)


class PriceValidator:
    """
    Multi-source price validation and consensus engine

    Features:
    - Cross-validation across Chainlink, CoinGecko, Uniswap
    - Automatic fallback on source failures
    - Anomaly detection and consensus scoring
    - Price interpolation for missing data
    """

    def __init__(
        self,
        chainlink_client: ChainlinkPriceClient,
        coingecko_client: CoinGeckoClient,
        uniswap_client: UniswapV3Client
    ):
        """
        Initialize price validator with all sources

        Args:
            chainlink_client: Chainlink price feed client
            coingecko_client: CoinGecko API client
            uniswap_client: Uniswap V3 pool client
        """
        self.chainlink = chainlink_client
        self.coingecko = coingecko_client
        self.uniswap = uniswap_client

        logger.info("Price validator initialized with 3 sources")

    async def validate_price_with_sources(
        self,
        timestamp: int,
        tolerance: float = 0.05
    ) -> Dict:
        """
        Cross-validate price across multiple sources

        Args:
            timestamp: Unix timestamp
            tolerance: Maximum acceptable deviation from median (default: 5%)

        Returns:
            Dict with validation results and consensus price
        """
        logger.info(f"Validating price for timestamp {timestamp}")

        # Fetch from all sources
        chainlink_price = await self._get_chainlink_price(timestamp)
        coingecko_price = await self._get_coingecko_price(timestamp)
        uniswap_price = await self._get_uniswap_price(timestamp)

        # Collect available prices
        prices = {}
        if chainlink_price:
            prices['chainlink'] = chainlink_price
        if coingecko_price:
            prices['coingecko'] = coingecko_price
        if uniswap_price:
            prices['uniswap'] = uniswap_price

        if not prices:
            logger.error(f"No prices available for timestamp {timestamp}")
            return {
                'timestamp': timestamp,
                'prices': {},
                'median_price': None,
                'consensus_price': None,
                'valid_sources': [],
                'validation_passed': False,
                'error': 'No price sources available'
            }

        # Calculate median
        median_price = statistics.median(prices.values())

        # Calculate deviations
        deviations = {
            source: abs(price - median_price) / median_price
            for source, price in prices.items()
        }

        # Identify valid sources (within tolerance)
        valid_sources = [
            source for source, deviation in deviations.items()
            if deviation <= tolerance
        ]

        # Determine consensus
        validation_passed = len(valid_sources) >= 2
        consensus_price = median_price if validation_passed else None

        # If only one source, use it with lower confidence
        if len(prices) == 1 and not consensus_price:
            consensus_price = list(prices.values())[0]
            validation_passed = True
            logger.warning(f"Using single source price: ${consensus_price:.2f}")

        result = {
            'timestamp': timestamp,
            'prices': prices,
            'median_price': median_price,
            'deviations': deviations,
            'valid_sources': valid_sources,
            'consensus_price': consensus_price,
            'validation_passed': validation_passed,
            'num_sources': len(prices)
        }

        if validation_passed:
            logger.info(
                f"Price validation passed: ${consensus_price:.2f} "
                f"({len(valid_sources)}/{len(prices)} sources agreed)"
            )
        else:
            logger.warning(
                f"Price validation failed: max deviation {max(deviations.values()):.2%} "
                f"exceeds tolerance {tolerance:.2%}"
            )

        return result

    async def get_price_with_fallback(self, timestamp: int) -> Optional[Dict]:
        """
        Get price with automatic fallback hierarchy

        Fallback order:
        1. Chainlink (primary)
        2. CoinGecko (secondary)
        3. Uniswap (tertiary)

        Args:
            timestamp: Unix timestamp

        Returns:
            Price data dict or None
        """
        # Try Chainlink first (primary source)
        chainlink_price = await self._get_chainlink_price(timestamp)
        if chainlink_price:
            return {
                'price_usd': chainlink_price,
                'timestamp': timestamp,
                'source': 'chainlink',
                'confidence_score': 1.0
            }

        logger.warning("Chainlink unavailable, falling back to CoinGecko")

        # Try CoinGecko (secondary source)
        coingecko_price = await self._get_coingecko_price(timestamp)
        if coingecko_price:
            return {
                'price_usd': coingecko_price,
                'timestamp': timestamp,
                'source': 'coingecko',
                'confidence_score': 0.9
            }

        logger.warning("CoinGecko unavailable, falling back to Uniswap")

        # Try Uniswap (tertiary source)
        uniswap_price = await self._get_uniswap_price(timestamp)
        if uniswap_price:
            return {
                'price_usd': uniswap_price,
                'timestamp': timestamp,
                'source': 'uniswap_v3',
                'confidence_score': 0.85
            }

        logger.error(f"All price sources failed for timestamp {timestamp}")
        return None

    async def interpolate_missing_price(
        self,
        timestamp: int,
        price_data: Dict[int, Dict]
    ) -> Optional[float]:
        """
        Interpolate price for missing timestamp using nearby prices

        Args:
            timestamp: Target timestamp
            price_data: Dict of {timestamp -> price_data}

        Returns:
            Interpolated price or None
        """
        # Find closest timestamps before and after
        sorted_timestamps = sorted(price_data.keys())

        before = None
        after = None

        for ts in sorted_timestamps:
            if ts < timestamp:
                before = ts
            elif ts > timestamp and after is None:
                after = ts
                break

        # Need both before and after for interpolation
        if before is None or after is None:
            logger.warning(f"Cannot interpolate: missing before/after prices for {timestamp}")
            return None

        # Linear interpolation
        before_price = price_data[before]['price_usd']
        after_price = price_data[after]['price_usd']

        time_span = after - before
        time_offset = timestamp - before

        interpolated_price = before_price + (after_price - before_price) * (time_offset / time_span)

        logger.info(
            f"Interpolated price ${interpolated_price:.2f} for timestamp {timestamp} "
            f"(between ${before_price:.2f} and ${after_price:.2f})"
        )

        return interpolated_price

    async def detect_price_anomalies(
        self,
        price_data: Dict[int, Dict],
        std_threshold: float = 3.0
    ) -> List[Dict]:
        """
        Detect price anomalies using statistical methods

        Args:
            price_data: Dict of {timestamp -> price_data}
            std_threshold: Number of standard deviations for anomaly threshold

        Returns:
            List of detected anomalies
        """
        if len(price_data) < 3:
            logger.warning("Insufficient data for anomaly detection")
            return []

        anomalies = []
        prices = [p['price_usd'] for p in price_data.values()]
        mean_price = statistics.mean(prices)
        std_price = statistics.stdev(prices)

        for timestamp, data in price_data.items():
            price = data['price_usd']

            if std_price > 0:
                z_score = abs(price - mean_price) / std_price

                if z_score > std_threshold:
                    anomalies.append({
                        'timestamp': timestamp,
                        'price': price,
                        'mean_price': mean_price,
                        'z_score': z_score,
                        'severity': 'high' if z_score > 5 else 'medium',
                        'source': data.get('source', 'unknown')
                    })

        if anomalies:
            logger.warning(f"Detected {len(anomalies)} price anomalies")

        return anomalies

    async def _get_chainlink_price(self, timestamp: int) -> Optional[float]:
        """Get price from Chainlink"""
        try:
            price_data = await self.chainlink.get_eth_price_at_timestamp(timestamp)
            return price_data['price_usd'] if price_data else None
        except Exception as e:
            logger.error(f"Error fetching Chainlink price: {e}")
            return None

    async def _get_coingecko_price(self, timestamp: int) -> Optional[float]:
        """Get price from CoinGecko"""
        try:
            return await self.coingecko.get_historical_price('ethereum', timestamp)
        except Exception as e:
            logger.error(f"Error fetching CoinGecko price: {e}")
            return None

    async def _get_uniswap_price(self, timestamp: int) -> Optional[float]:
        """Get price from Uniswap"""
        try:
            return await self.uniswap.get_eth_usdc_price(timestamp=timestamp)
        except Exception as e:
            logger.error(f"Error fetching Uniswap price: {e}")
            return None