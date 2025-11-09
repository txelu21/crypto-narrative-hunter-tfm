"""
CoinGecko API Client for Historical Price Data
Provides secondary price source validation and fallback
"""

import logging
import aiohttp
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CoinGeckoClient:
    """
    Client for fetching ETH/USD prices from CoinGecko API

    Features:
    - Historical price data via CoinGecko API
    - Secondary source validation
    - Fallback for Chainlink failures
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CoinGecko client

        Args:
            api_key: Optional CoinGecko API key for higher rate limits
        """
        self.api_key = api_key
        self.headers = {}

        if api_key:
            self.headers['x-cg-pro-api-key'] = api_key

        logger.info("CoinGecko client initialized")

    async def get_historical_price(
        self,
        coin_id: str,
        timestamp: int,
        vs_currency: str = 'usd'
    ) -> Optional[float]:
        """
        Get historical price for a coin at specific timestamp

        Args:
            coin_id: CoinGecko coin ID (e.g., 'ethereum')
            timestamp: Unix timestamp
            vs_currency: Quote currency (default: 'usd')

        Returns:
            Price or None if not available
        """
        try:
            # Convert timestamp to date string (CoinGecko format: dd-mm-yyyy)
            dt = datetime.fromtimestamp(timestamp)
            date_str = dt.strftime('%d-%m-%Y')

            url = f"{self.BASE_URL}/coins/{coin_id}/history"
            params = {
                'date': date_str,
                'localization': 'false'
            }

            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Extract price from market_data
                        if 'market_data' in data and 'current_price' in data['market_data']:
                            price = data['market_data']['current_price'].get(vs_currency)

                            if price and price > 0:
                                logger.info(f"CoinGecko price for {coin_id} on {date_str}: ${price:.2f}")
                                return float(price)

                    elif response.status == 429:
                        logger.warning("CoinGecko rate limit exceeded")
                    else:
                        logger.warning(f"CoinGecko API error: {response.status}")

                    return None

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching CoinGecko price: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko price: {e}")
            return None

    async def get_market_chart_range(
        self,
        coin_id: str,
        from_timestamp: int,
        to_timestamp: int,
        vs_currency: str = 'usd'
    ) -> Optional[list]:
        """
        Get price chart data for a time range

        Args:
            coin_id: CoinGecko coin ID
            from_timestamp: Start unix timestamp
            to_timestamp: End unix timestamp
            vs_currency: Quote currency

        Returns:
            List of [timestamp, price] pairs or None
        """
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart/range"
            params = {
                'vs_currency': vs_currency,
                'from': from_timestamp,
                'to': to_timestamp
            }

            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'prices' in data:
                            logger.info(
                                f"Fetched {len(data['prices'])} price points from CoinGecko"
                            )
                            return data['prices']

                    elif response.status == 429:
                        logger.warning("CoinGecko rate limit exceeded")
                    else:
                        logger.warning(f"CoinGecko API error: {response.status}")

                    return None

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching CoinGecko chart: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko chart: {e}")
            return None

    async def get_current_price(
        self,
        coin_id: str,
        vs_currency: str = 'usd'
    ) -> Optional[float]:
        """
        Get current price for a coin

        Args:
            coin_id: CoinGecko coin ID
            vs_currency: Quote currency

        Returns:
            Current price or None
        """
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': vs_currency
            }

            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()

                        if coin_id in data and vs_currency in data[coin_id]:
                            price = data[coin_id][vs_currency]
                            logger.info(f"Current CoinGecko price for {coin_id}: ${price:.2f}")
                            return float(price)

                    return None

        except Exception as e:
            logger.error(f"Error fetching current CoinGecko price: {e}")
            return None