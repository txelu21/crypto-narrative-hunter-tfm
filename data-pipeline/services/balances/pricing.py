"""Token pricing and ETH value calculation.

This module handles token price fetching, ETH value calculation, and
price validation with fallback mechanisms.
"""

from typing import Dict, Optional, Tuple
from decimal import Decimal
from datetime import date
import psycopg
from psycopg.rows import dict_row

from data_collection.common.logging_setup import get_logger

logger = get_logger(__name__)


class PricingService:
    """Service for token pricing and ETH value calculation."""

    def __init__(self, conn: psycopg.Connection):
        """Initialize pricing service.

        Args:
            conn: Database connection
        """
        self.conn = conn
        self.token_decimals_cache: Dict[str, int] = {}

    def get_token_decimals(self, token_address: str) -> int:
        """Get token decimals, with fallback to standard decimals.

        Args:
            token_address: Token contract address

        Returns:
            Token decimals (default: 18)
        """
        if token_address == 'ETH':
            return 18

        # Check cache
        if token_address in self.token_decimals_cache:
            return self.token_decimals_cache[token_address]

        # Query database (assuming token metadata exists)
        query = """
        SELECT decimals FROM tokens
        WHERE address = %s
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (token_address,))
                result = cur.fetchone()

                if result:
                    decimals = result[0]
                    self.token_decimals_cache[token_address] = decimals
                    return decimals
        except Exception as e:
            logger.warning(f"Could not fetch decimals for {token_address}: {e}")

        # Default to 18 decimals
        logger.debug(f"Using default decimals (18) for {token_address}")
        self.token_decimals_cache[token_address] = 18
        return 18

    def get_token_price_eth(
        self,
        token_address: str,
        snapshot_date: date,
    ) -> Optional[Decimal]:
        """Get token price in ETH for a specific date.

        This queries the liquidity/pricing data from Story 1.3.

        Args:
            token_address: Token contract address
            snapshot_date: Date to get price for

        Returns:
            Token price in ETH or None if not available
        """
        if token_address == 'ETH':
            return Decimal('1.0')

        # Query liquidity pools for price
        # This assumes liquidity pool data exists from Story 1.3
        query = """
        SELECT price_eth
        FROM token_liquidity_daily
        WHERE token_address = %s
        AND date = %s
        ORDER BY total_liquidity_eth DESC
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (token_address, snapshot_date))
                result = cur.fetchone()

                if result and result[0] is not None:
                    return Decimal(str(result[0]))
        except Exception as e:
            logger.warning(
                f"Could not fetch price for {token_address} on {snapshot_date}: {e}"
            )

        return None

    def get_token_prices_batch(
        self,
        token_addresses: list[str],
        snapshot_date: date,
    ) -> Dict[str, Optional[Decimal]]:
        """Get prices for multiple tokens efficiently.

        Args:
            token_addresses: List of token addresses
            snapshot_date: Date to get prices for

        Returns:
            Dictionary mapping token addresses to prices in ETH
        """
        if not token_addresses:
            return {}

        prices = {}

        # ETH is always 1.0
        if 'ETH' in token_addresses:
            prices['ETH'] = Decimal('1.0')
            token_addresses = [t for t in token_addresses if t != 'ETH']

        if not token_addresses:
            return prices

        # Batch query for all tokens
        query = """
        SELECT DISTINCT ON (token_address)
            token_address, price_eth
        FROM token_liquidity_daily
        WHERE token_address = ANY(%s)
        AND date = %s
        ORDER BY token_address, total_liquidity_eth DESC
        """

        try:
            with self.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (token_addresses, snapshot_date))
                results = cur.fetchall()

                for row in results:
                    if row['price_eth'] is not None:
                        prices[row['token_address']] = Decimal(str(row['price_eth']))

        except Exception as e:
            logger.error(f"Batch price fetch failed: {e}")

        # Fill in None for tokens without prices
        for token in token_addresses:
            if token not in prices:
                prices[token] = None

        logger.info(
            f"Fetched prices for {len([p for p in prices.values() if p is not None])}/{len(token_addresses)} tokens"
        )

        return prices

    def calculate_eth_value(
        self,
        token_address: str,
        balance: int,
        price_eth: Optional[Decimal] = None,
    ) -> Optional[Decimal]:
        """Calculate ETH-denominated value for a token balance.

        Args:
            token_address: Token contract address
            balance: Token balance (raw amount)
            price_eth: Token price in ETH (will fetch if not provided)

        Returns:
            ETH value or None if price unavailable
        """
        if balance == 0:
            return Decimal('0')

        if token_address == 'ETH':
            # Convert wei to ETH
            return Decimal(balance) / Decimal(10 ** 18)

        if price_eth is None:
            logger.debug(f"No price provided for {token_address}, cannot calculate value")
            return None

        # Get token decimals
        decimals = self.get_token_decimals(token_address)

        # Convert to decimal amount
        decimal_balance = Decimal(balance) / Decimal(10 ** decimals)

        # Calculate ETH value
        eth_value = decimal_balance * price_eth

        return eth_value

    def calculate_eth_values_batch(
        self,
        balances: Dict[str, int],
        prices: Dict[str, Optional[Decimal]],
    ) -> Dict[str, Optional[Decimal]]:
        """Calculate ETH values for multiple balances.

        Args:
            balances: Dictionary mapping token addresses to balances
            prices: Dictionary mapping token addresses to prices in ETH

        Returns:
            Dictionary mapping token addresses to ETH values
        """
        eth_values = {}

        for token_address, balance in balances.items():
            price = prices.get(token_address)
            eth_value = self.calculate_eth_value(token_address, balance, price)
            eth_values[token_address] = eth_value

        return eth_values

    def validate_price(
        self,
        token_address: str,
        price_eth: Decimal,
        max_change_ratio: Decimal = Decimal('10.0'),
    ) -> Tuple[bool, Optional[str]]:
        """Validate token price against historical data.

        Args:
            token_address: Token contract address
            price_eth: Price to validate
            max_change_ratio: Maximum acceptable price change ratio

        Returns:
            Tuple of (is_valid, reason)
        """
        if price_eth <= 0:
            return False, "Price must be positive"

        # Get recent historical price
        query = """
        SELECT price_eth
        FROM token_liquidity_daily
        WHERE token_address = %s
        AND price_eth IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (token_address,))
                result = cur.fetchone()

                if result and result[0] is not None:
                    historical_price = Decimal(str(result[0]))

                    # Check for outliers
                    price_ratio = price_eth / historical_price

                    if price_ratio > max_change_ratio or price_ratio < (1 / max_change_ratio):
                        return (
                            False,
                            f"Price change ratio {price_ratio:.2f}x exceeds threshold {max_change_ratio}x",
                        )

        except Exception as e:
            logger.warning(f"Price validation failed for {token_address}: {e}")
            # Don't block on validation errors
            return True, None

        return True, None

    def get_fallback_price(
        self,
        token_address: str,
        snapshot_date: date,
    ) -> Optional[Decimal]:
        """Get fallback price from alternative sources.

        Priority:
        1. Most recent historical price (up to 7 days old)
        2. External API (CoinGecko, etc.) - not implemented yet
        3. None

        Args:
            token_address: Token contract address
            snapshot_date: Target date

        Returns:
            Fallback price or None
        """
        # Try recent historical prices
        query = """
        SELECT price_eth
        FROM token_liquidity_daily
        WHERE token_address = %s
        AND date <= %s
        AND date >= %s - INTERVAL '7 days'
        AND price_eth IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (token_address, snapshot_date, snapshot_date))
                result = cur.fetchone()

                if result and result[0] is not None:
                    logger.info(
                        f"Using historical fallback price for {token_address}"
                    )
                    return Decimal(str(result[0]))

        except Exception as e:
            logger.error(f"Fallback price fetch failed for {token_address}: {e}")

        logger.warning(f"No fallback price available for {token_address}")
        return None

    def enrich_balances_with_prices(
        self,
        balances: Dict[str, int],
        snapshot_date: date,
        use_fallback: bool = True,
    ) -> Dict[str, Dict]:
        """Enrich balance data with prices and ETH values.

        Args:
            balances: Dictionary mapping token addresses to balances
            snapshot_date: Snapshot date
            use_fallback: Whether to use fallback prices

        Returns:
            Dictionary with enriched balance data including prices and values
        """
        if not balances:
            return {}

        # Get prices for all tokens
        token_addresses = list(balances.keys())
        prices = self.get_token_prices_batch(token_addresses, snapshot_date)

        # Try fallback for missing prices
        if use_fallback:
            for token in token_addresses:
                if prices.get(token) is None:
                    fallback_price = self.get_fallback_price(token, snapshot_date)
                    if fallback_price:
                        prices[token] = fallback_price

        # Calculate ETH values
        eth_values = self.calculate_eth_values_batch(balances, prices)

        # Build enriched data
        enriched = {}
        for token, balance in balances.items():
            enriched[token] = {
                'balance': balance,
                'price_eth': prices.get(token),
                'eth_value': eth_values.get(token),
                'decimals': self.get_token_decimals(token),
            }

        # Log pricing stats
        total_tokens = len(balances)
        priced_tokens = sum(1 for p in prices.values() if p is not None)
        valued_tokens = sum(1 for v in eth_values.values() if v is not None)

        logger.info(
            f"Pricing summary: {priced_tokens}/{total_tokens} tokens priced, "
            f"{valued_tokens}/{total_tokens} valued"
        )

        return enriched