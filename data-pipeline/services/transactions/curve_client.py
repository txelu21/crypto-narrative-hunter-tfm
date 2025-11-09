"""Curve Finance API client for stablecoin and crypto pool transaction data.

This module provides an async client for interacting with Curve Finance pools
to extract swap transactions, virtual prices, and pool registry data.
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = structlog.get_logger()


# Curve API endpoints
CURVE_API_BASE = "https://api.curve.fi/api"
CURVE_POOLS_ENDPOINT = f"{CURVE_API_BASE}/getPools/ethereum/main"
CURVE_FACTORY_POOLS_ENDPOINT = f"{CURVE_API_BASE}/getPools/ethereum/factory"


class CurveAPIError(Exception):
    """Raised when Curve API request fails."""
    pass


class CurveClient:
    """Async client for Curve Finance API."""

    def __init__(
        self,
        rate_limit_semaphore: Optional[asyncio.Semaphore] = None,
        timeout: int = 30,
    ):
        """Initialize Curve API client.

        Args:
            rate_limit_semaphore: Optional semaphore for rate limiting
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit_semaphore or asyncio.Semaphore(10)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.logger = logger.bind(component="curve_client")

        # Cache for pool data
        self._pool_cache: Dict[str, Dict[str, Any]] = {}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request to Curve API.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            API response JSON

        Raises:
            CurveAPIError: If request fails
        """
        async with self.rate_limit:
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()

                        if not data.get("success", True):
                            error_msg = data.get("error", "Unknown error")
                            raise CurveAPIError(f"Curve API error: {error_msg}")

                        return data

            except aiohttp.ClientError as e:
                self.logger.error("curve_api_error", error=str(e), url=url)
                raise CurveAPIError(f"Curve API request failed: {e}") from e

    async def get_all_pools(
        self,
        include_factory: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get all Curve pools on Ethereum mainnet.

        Args:
            include_factory: Whether to include factory pools

        Returns:
            List of pool data

        Raises:
            CurveAPIError: If request fails
        """
        log = self.logger.bind(operation="get_all_pools")

        log.info("fetching_curve_pools", include_factory=include_factory)

        # Fetch main pools
        main_pools_response = await self._get(CURVE_POOLS_ENDPOINT)
        main_pools = main_pools_response.get("data", {}).get("poolData", [])

        if include_factory:
            # Fetch factory pools
            factory_pools_response = await self._get(CURVE_FACTORY_POOLS_ENDPOINT)
            factory_pools = factory_pools_response.get("data", {}).get("poolData", [])

            # Mark factory pools
            for pool in factory_pools:
                pool["is_factory"] = True

            all_pools = main_pools + factory_pools
        else:
            all_pools = main_pools

        log.info(
            "fetched_curve_pools",
            main_count=len(main_pools),
            factory_count=len(factory_pools) if include_factory else 0,
            total=len(all_pools),
        )

        # Update cache
        for pool in all_pools:
            pool_address = pool.get("address", "").lower()
            if pool_address:
                self._pool_cache[pool_address] = pool

        return all_pools

    async def get_pool_info(
        self,
        pool_address: str,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific pool.

        Args:
            pool_address: Pool contract address
            use_cache: Whether to use cached data

        Returns:
            Pool information or None if not found

        Raises:
            CurveAPIError: If request fails
        """
        pool_address = pool_address.lower()

        # Check cache first
        if use_cache and pool_address in self._pool_cache:
            self.logger.debug("pool_info_from_cache", pool_address=pool_address)
            return self._pool_cache[pool_address]

        # Fetch all pools if cache miss
        await self.get_all_pools()

        return self._pool_cache.get(pool_address)

    async def get_pool_coins(
        self,
        pool_address: str,
    ) -> List[Dict[str, Any]]:
        """Get coin information for a pool.

        Args:
            pool_address: Pool contract address

        Returns:
            List of coin/token data

        Raises:
            CurveAPIError: If pool not found
        """
        pool_info = await self.get_pool_info(pool_address)

        if not pool_info:
            raise CurveAPIError(f"Pool not found: {pool_address}")

        coins = pool_info.get("coins", [])

        self.logger.info(
            "fetched_pool_coins",
            pool_address=pool_address,
            coin_count=len(coins),
        )

        return coins

    async def get_pool_virtual_price(
        self,
        pool_address: str,
    ) -> Optional[float]:
        """Get virtual price for a pool.

        The virtual price represents the price to sell a pool LP token
        back to the underlying assets.

        Args:
            pool_address: Pool contract address

        Returns:
            Virtual price or None if not available
        """
        pool_info = await self.get_pool_info(pool_address)

        if not pool_info:
            return None

        virtual_price = pool_info.get("virtualPrice")

        self.logger.debug(
            "fetched_virtual_price",
            pool_address=pool_address,
            virtual_price=virtual_price,
        )

        return float(virtual_price) if virtual_price else None

    async def get_pool_type(
        self,
        pool_address: str,
    ) -> Optional[str]:
        """Get pool type (stable, crypto, etc.).

        Args:
            pool_address: Pool contract address

        Returns:
            Pool type or None if not found
        """
        pool_info = await self.get_pool_info(pool_address)

        if not pool_info:
            return None

        # Determine pool type based on pool data
        pool_type = pool_info.get("assetType", "").lower()

        # Additional classification logic
        if not pool_type:
            coins = pool_info.get("coins", [])
            coin_symbols = [c.get("symbol", "").upper() for c in coins]

            # Check if it's a stablecoin pool
            stablecoins = {"USDC", "USDT", "DAI", "FRAX", "TUSD", "USDD", "GUSD"}
            if all(symbol in stablecoins for symbol in coin_symbols if symbol):
                pool_type = "stable"
            else:
                pool_type = "crypto"

        self.logger.debug(
            "determined_pool_type",
            pool_address=pool_address,
            pool_type=pool_type,
        )

        return pool_type

    async def get_pool_tvl(
        self,
        pool_address: str,
    ) -> Optional[float]:
        """Get total value locked (TVL) for a pool.

        Args:
            pool_address: Pool contract address

        Returns:
            TVL in USD or None if not available
        """
        pool_info = await self.get_pool_info(pool_address)

        if not pool_info:
            return None

        tvl = pool_info.get("usdTotal")

        self.logger.debug(
            "fetched_pool_tvl",
            pool_address=pool_address,
            tvl_usd=tvl,
        )

        return float(tvl) if tvl else None

    async def get_pool_volume(
        self,
        pool_address: str,
    ) -> Optional[float]:
        """Get 24h trading volume for a pool.

        Args:
            pool_address: Pool contract address

        Returns:
            Volume in USD or None if not available
        """
        pool_info = await self.get_pool_info(pool_address)

        if not pool_info:
            return None

        # Try different volume fields
        volume = pool_info.get("volumeUSD") or pool_info.get("volume")

        self.logger.debug(
            "fetched_pool_volume",
            pool_address=pool_address,
            volume_usd=volume,
        )

        return float(volume) if volume else None

    def get_coin_index(
        self,
        pool_address: str,
        token_address: str,
    ) -> Optional[int]:
        """Get coin index within a pool.

        Args:
            pool_address: Pool contract address
            token_address: Token contract address

        Returns:
            Coin index (0, 1, 2, etc.) or None if not found
        """
        pool_address = pool_address.lower()
        token_address = token_address.lower()

        pool_info = self._pool_cache.get(pool_address)
        if not pool_info:
            return None

        coins = pool_info.get("coins", [])

        for idx, coin in enumerate(coins):
            if coin.get("address", "").lower() == token_address:
                self.logger.debug(
                    "found_coin_index",
                    pool_address=pool_address,
                    token_address=token_address,
                    index=idx,
                )
                return idx

        return None

    async def is_curve_pool(
        self,
        pool_address: str,
    ) -> bool:
        """Check if an address is a known Curve pool.

        Args:
            pool_address: Address to check

        Returns:
            True if it's a Curve pool
        """
        pool_info = await self.get_pool_info(pool_address)
        return pool_info is not None

    async def get_pools_by_token(
        self,
        token_address: str,
    ) -> List[Dict[str, Any]]:
        """Get all pools containing a specific token.

        Args:
            token_address: Token contract address

        Returns:
            List of pools containing the token
        """
        token_address = token_address.lower()

        # Ensure we have all pools in cache
        await self.get_all_pools()

        matching_pools = []

        for pool_address, pool_info in self._pool_cache.items():
            coins = pool_info.get("coins", [])
            coin_addresses = [c.get("address", "").lower() for c in coins]

            if token_address in coin_addresses:
                matching_pools.append(pool_info)

        self.logger.info(
            "found_pools_by_token",
            token_address=token_address,
            pool_count=len(matching_pools),
        )

        return matching_pools

    async def extract_swap_transactions(
        self,
        pool_address: str,
        wallet_address: str,
        from_block: int,
        to_block: int,
        web3_client: Any,
    ) -> List[Dict[str, Any]]:
        """Extract swap transactions from Curve pool for a wallet.

        Args:
            pool_address: Curve pool contract address
            wallet_address: Wallet address to filter swaps for
            from_block: Starting block number
            to_block: Ending block number
            web3_client: Web3 client for on-chain queries

        Returns:
            List of swap transaction data

        Raises:
            CurveAPIError: If extraction fails
        """
        log = self.logger.bind(
            operation="extract_swap_transactions",
            pool_address=pool_address,
            wallet_address=wallet_address,
            from_block=from_block,
            to_block=to_block,
        )

        log.info("extracting_curve_swaps")

        # Get pool info to determine pool type and coins
        pool_info = await self.get_pool_info(pool_address)
        if not pool_info:
            raise CurveAPIError(f"Pool not found: {pool_address}")

        pool_type = await self.get_pool_type(pool_address)
        coins = pool_info.get("coins", [])

        # Curve swap event signatures
        # TokenExchange: exchange(i, j, dx, dy)
        EXCHANGE_EVENT_SIGNATURE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"
        # TokenExchangeUnderlying: exchange_underlying(i, j, dx, dy)
        EXCHANGE_UNDERLYING_EVENT_SIGNATURE = "0xd013ca23e77a65003c2c659c5442c00c805371b7fc1ebd4c206c41d1536bd90b"

        # Query swap events
        swap_events = []

        # Get TokenExchange events
        exchange_logs = await web3_client.eth_get_logs({
            "address": pool_address,
            "topics": [
                EXCHANGE_EVENT_SIGNATURE,
                None,  # buyer (indexed)
            ],
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
        })

        # Get TokenExchangeUnderlying events (for meta pools)
        exchange_underlying_logs = await web3_client.eth_get_logs({
            "address": pool_address,
            "topics": [
                EXCHANGE_UNDERLYING_EVENT_SIGNATURE,
                None,  # buyer (indexed)
            ],
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
        })

        # Combine and filter events for the wallet
        all_logs = exchange_logs + exchange_underlying_logs

        for event_log in all_logs:
            # Extract buyer address from topics (second topic)
            if len(event_log.get("topics", [])) >= 2:
                # Topic is 32 bytes (64 hex chars), address is last 20 bytes (40 hex chars)
                buyer_topic = event_log["topics"][1]
                if buyer_topic.startswith("0x"):
                    buyer_topic = buyer_topic[2:]  # Remove 0x prefix if present
                buyer = "0x" + buyer_topic[-40:]  # Last 20 bytes

                if buyer.lower() != wallet_address.lower():
                    continue

                # Decode event data
                swap_data = self._decode_curve_swap_event(
                    event_log,
                    pool_info,
                    pool_type,
                )

                if swap_data:
                    swap_events.append(swap_data)

        log.info(
            "extracted_curve_swaps",
            swap_count=len(swap_events),
        )

        return swap_events

    def _decode_curve_swap_event(
        self,
        event_log: Dict[str, Any],
        pool_info: Dict[str, Any],
        pool_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Decode Curve swap event log.

        Args:
            event_log: Raw event log
            pool_info: Pool information
            pool_type: Type of pool (stable/crypto)

        Returns:
            Decoded swap data or None if decoding fails
        """
        try:
            # Extract event data
            data = event_log.get("data", "0x")
            topics = event_log.get("topics", [])

            if len(topics) < 2:
                return None

            # Buyer address from second topic
            buyer_topic = topics[1]
            if buyer_topic.startswith("0x"):
                buyer_topic = buyer_topic[2:]
            buyer = "0x" + buyer_topic[-40:]

            # Decode non-indexed parameters from data
            # Format: sold_id (int128), tokens_sold (uint256), bought_id (int128), tokens_bought (uint256)
            data_bytes = bytes.fromhex(data[2:])  # Remove 0x prefix

            if len(data_bytes) < 128:  # 4 * 32 bytes
                return None

            # Parse data fields (each is 32 bytes)
            sold_id = int.from_bytes(data_bytes[0:32], "big")
            tokens_sold = int.from_bytes(data_bytes[32:64], "big")
            bought_id = int.from_bytes(data_bytes[64:96], "big")
            tokens_bought = int.from_bytes(data_bytes[96:128], "big")

            # Get coin information
            coins = pool_info.get("coins", [])

            if sold_id >= len(coins) or bought_id >= len(coins):
                return None

            token_in = coins[sold_id]
            token_out = coins[bought_id]

            return {
                "tx_hash": event_log.get("transactionHash"),
                "block_number": int(event_log.get("blockNumber", "0x0"), 16),
                "log_index": int(event_log.get("logIndex", "0x0"), 16),
                "pool_address": event_log.get("address"),
                "pool_type": pool_type,
                "buyer": buyer,
                "token_in_address": token_in.get("address"),
                "token_in_symbol": token_in.get("symbol"),
                "token_in_decimals": token_in.get("decimals", 18),
                "amount_in": tokens_sold,
                "token_out_address": token_out.get("address"),
                "token_out_symbol": token_out.get("symbol"),
                "token_out_decimals": token_out.get("decimals", 18),
                "amount_out": tokens_bought,
                "coin_index_in": sold_id,
                "coin_index_out": bought_id,
            }

        except Exception as e:
            self.logger.error(
                "failed_to_decode_curve_swap",
                error=str(e),
                tx_hash=event_log.get("transactionHash"),
            )
            return None

    async def calculate_swap_price_eth(
        self,
        swap_data: Dict[str, Any],
        web3_client: Any,
        block_number: int,
    ) -> Optional[float]:
        """Calculate ETH value of a Curve swap using virtual price.

        Args:
            swap_data: Decoded swap transaction data
            web3_client: Web3 client for on-chain queries
            block_number: Block number for historical price

        Returns:
            ETH value of the swap or None if calculation fails
        """
        try:
            pool_address = swap_data["pool_address"]
            pool_type = swap_data["pool_type"]

            # Get virtual price at the block
            virtual_price = await self._get_virtual_price_at_block(
                pool_address,
                block_number,
                web3_client,
            )

            if not virtual_price:
                return None

            # For stablecoin pools, use virtual price directly
            if pool_type == "stable":
                # Virtual price is in 1e18 format
                # Convert amount to USD first, then to ETH
                amount_in = swap_data["amount_in"]
                decimals_in = swap_data["token_in_decimals"]

                # Normalize amount
                normalized_amount = amount_in / (10 ** decimals_in)

                # Calculate USD value using virtual price
                usd_value = normalized_amount * (virtual_price / 1e18)

                # Convert USD to ETH (requires ETH/USD price)
                # This would need to be fetched from another source
                # For now, return None to indicate pricing needs external data
                return None

            else:
                # For crypto pools, more complex pricing needed
                # Would need to fetch pool reserves and calculate
                return None

        except Exception as e:
            self.logger.error(
                "failed_to_calculate_swap_price",
                error=str(e),
                swap_data=swap_data,
            )
            return None

    async def _get_virtual_price_at_block(
        self,
        pool_address: str,
        block_number: int,
        web3_client: Any,
    ) -> Optional[int]:
        """Get virtual price of a pool at a specific block.

        Args:
            pool_address: Pool contract address
            block_number: Block number for historical query
            web3_client: Web3 client for on-chain queries

        Returns:
            Virtual price in 1e18 format or None
        """
        try:
            # ABI for get_virtual_price function
            virtual_price_abi = {
                "name": "get_virtual_price",
                "type": "function",
                "inputs": [],
                "outputs": [{"type": "uint256"}],
            }

            # Encode function call
            encoded = web3_client.encode_function_call(virtual_price_abi, [])

            # Call at specific block
            result = await web3_client.eth_call(
                {
                    "to": pool_address,
                    "data": encoded,
                },
                block_number,
            )

            if result:
                return int(result, 16)

            return None

        except Exception as e:
            self.logger.error(
                "failed_to_get_virtual_price",
                error=str(e),
                pool_address=pool_address,
                block_number=block_number,
            )
            return None

    async def process_stable_pool_swap(
        self,
        swap_data: Dict[str, Any],
        virtual_price: int,
        eth_usd_price: float,
    ) -> Dict[str, Any]:
        """Process swap from a stablecoin pool with specialized handling.

        Args:
            swap_data: Decoded swap transaction data
            virtual_price: Virtual price of the pool
            eth_usd_price: Current ETH/USD price

        Returns:
            Processed swap data with ETH valuations
        """
        try:
            # For stable pools, tokens are typically pegged to USD
            amount_in = swap_data["amount_in"]
            amount_out = swap_data["amount_out"]
            decimals_in = swap_data["token_in_decimals"]
            decimals_out = swap_data["token_out_decimals"]

            # Normalize amounts
            normalized_in = amount_in / (10 ** decimals_in)
            normalized_out = amount_out / (10 ** decimals_out)

            # Calculate USD values using virtual price
            # Virtual price represents the price of LP token in terms of underlying
            vp_normalized = virtual_price / 1e18

            # For stablecoins, assume 1:1 USD peg with slight deviation
            usd_value_in = normalized_in * vp_normalized
            usd_value_out = normalized_out * vp_normalized

            # Convert to ETH
            eth_value_in = usd_value_in / eth_usd_price if eth_usd_price else None
            eth_value_out = usd_value_out / eth_usd_price if eth_usd_price else None

            # Calculate slippage
            expected_out = normalized_in  # For stables, expect 1:1
            actual_out = normalized_out
            slippage = abs(expected_out - actual_out) / expected_out if expected_out > 0 else 0

            return {
                **swap_data,
                "usd_value_in": usd_value_in,
                "usd_value_out": usd_value_out,
                "eth_value_in": eth_value_in,
                "eth_value_out": eth_value_out,
                "slippage_percentage": slippage * 100,
                "price_impact": self._calculate_price_impact(normalized_in, normalized_out),
                "processing_type": "stable_pool",
            }

        except Exception as e:
            self.logger.error(
                "failed_to_process_stable_swap",
                error=str(e),
                swap_data=swap_data,
            )
            return swap_data

    async def process_crypto_pool_swap(
        self,
        swap_data: Dict[str, Any],
        pool_address: str,
        block_number: int,
        web3_client: Any,
    ) -> Dict[str, Any]:
        """Process swap from a crypto pool with specialized handling.

        Args:
            swap_data: Decoded swap transaction data
            pool_address: Pool contract address
            block_number: Block number of the swap
            web3_client: Web3 client for on-chain queries

        Returns:
            Processed swap data with ETH valuations
        """
        try:
            # For crypto pools, need to fetch pool balances/reserves
            balances = await self._get_pool_balances_at_block(
                pool_address,
                block_number,
                web3_client,
            )

            if not balances:
                return swap_data

            coin_index_in = swap_data["coin_index_in"]
            coin_index_out = swap_data["coin_index_out"]

            # Get pool state before swap for price calculation
            balance_in = balances[coin_index_in] if coin_index_in < len(balances) else 0
            balance_out = balances[coin_index_out] if coin_index_out < len(balances) else 0

            amount_in = swap_data["amount_in"]
            amount_out = swap_data["amount_out"]
            decimals_in = swap_data["token_in_decimals"]
            decimals_out = swap_data["token_out_decimals"]

            # Normalize amounts
            normalized_in = amount_in / (10 ** decimals_in)
            normalized_out = amount_out / (10 ** decimals_out)

            # Calculate price based on pool reserves
            if balance_in > 0 and balance_out > 0:
                # Price = (reserve_out / reserve_in)
                spot_price_before = (balance_out / (10 ** decimals_out)) / (balance_in / (10 ** decimals_in))

                # Effective price from the swap
                effective_price = normalized_out / normalized_in if normalized_in > 0 else 0

                # Price impact
                price_impact = abs(spot_price_before - effective_price) / spot_price_before if spot_price_before > 0 else 0
            else:
                effective_price = 0
                price_impact = 0

            # ETH value calculation requires token-to-ETH pricing
            # This would need integration with DEX price oracles
            eth_value_in = await self._get_token_eth_value(
                swap_data["token_in_address"],
                normalized_in,
                block_number,
                web3_client,
            )

            eth_value_out = await self._get_token_eth_value(
                swap_data["token_out_address"],
                normalized_out,
                block_number,
                web3_client,
            )

            return {
                **swap_data,
                "eth_value_in": eth_value_in,
                "eth_value_out": eth_value_out,
                "effective_price": effective_price,
                "price_impact_percentage": price_impact * 100,
                "pool_balance_in": balance_in,
                "pool_balance_out": balance_out,
                "processing_type": "crypto_pool",
            }

        except Exception as e:
            self.logger.error(
                "failed_to_process_crypto_swap",
                error=str(e),
                swap_data=swap_data,
            )
            return swap_data

    def _calculate_price_impact(
        self,
        amount_in: float,
        amount_out: float,
    ) -> float:
        """Calculate price impact for a swap.

        Args:
            amount_in: Normalized input amount
            amount_out: Normalized output amount

        Returns:
            Price impact as a percentage
        """
        if amount_in == 0:
            return 0

        # For stables, ideal rate is 1:1
        ideal_out = amount_in
        actual_out = amount_out

        impact = (ideal_out - actual_out) / ideal_out if ideal_out > 0 else 0
        return abs(impact) * 100

    async def _get_pool_balances_at_block(
        self,
        pool_address: str,
        block_number: int,
        web3_client: Any,
    ) -> Optional[List[int]]:
        """Get pool token balances at a specific block.

        Args:
            pool_address: Pool contract address
            block_number: Block number for historical query
            web3_client: Web3 client for on-chain queries

        Returns:
            List of token balances or None
        """
        try:
            pool_info = await self.get_pool_info(pool_address)
            if not pool_info:
                return None

            coins = pool_info.get("coins", [])
            balances = []

            for i in range(len(coins)):
                # ABI for balances function
                balances_abi = {
                    "name": "balances",
                    "type": "function",
                    "inputs": [{"type": "uint256"}],
                    "outputs": [{"type": "uint256"}],
                }

                # Encode function call
                encoded = web3_client.encode_function_call(balances_abi, [i])

                # Call at specific block
                result = await web3_client.eth_call(
                    {
                        "to": pool_address,
                        "data": encoded,
                    },
                    block_number,
                )

                if result:
                    balance = int(result, 16)
                    balances.append(balance)
                else:
                    balances.append(0)

            return balances

        except Exception as e:
            self.logger.error(
                "failed_to_get_pool_balances",
                error=str(e),
                pool_address=pool_address,
                block_number=block_number,
            )
            return None

    async def _get_token_eth_value(
        self,
        token_address: str,
        amount: float,
        block_number: int,
        web3_client: Any,
    ) -> Optional[float]:
        """Get ETH value of a token amount at a specific block.

        This would need integration with price oracles or DEX pools.

        Args:
            token_address: Token contract address
            amount: Token amount (normalized)
            block_number: Block number for historical price
            web3_client: Web3 client for on-chain queries

        Returns:
            ETH value or None
        """
        # Placeholder - would need actual price oracle integration
        # This could query Uniswap pools, Chainlink oracles, etc.
        return None