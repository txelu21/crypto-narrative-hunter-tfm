"""
Uniswap V3 Price Client
Provides decentralized ETH/USDC price data from Uniswap pools
"""

import logging
from typing import Dict, Optional
from web3 import Web3
from web3.contract import Contract

logger = logging.getLogger(__name__)


class UniswapV3Client:
    """
    Client for fetching ETH/USDC prices from Uniswap V3 pools

    Features:
    - Decentralized tertiary price source
    - On-chain USDC/ETH pool price queries
    - Historical price at specific blocks
    """

    # Uniswap V3 ETH/USDC 0.05% pool on Ethereum mainnet
    ETH_USDC_POOL_005 = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

    # Uniswap V3 Pool ABI (minimal interface for slot0)
    POOL_ABI = [
        {
            "inputs": [],
            "name": "slot0",
            "outputs": [
                {"name": "sqrtPriceX96", "type": "uint160"},
                {"name": "tick", "type": "int24"},
                {"name": "observationIndex", "type": "uint16"},
                {"name": "observationCardinality", "type": "uint16"},
                {"name": "observationCardinalityNext", "type": "uint16"},
                {"name": "feeProtocol", "type": "uint8"},
                {"name": "unlocked", "type": "bool"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "token0",
            "outputs": [{"name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "token1",
            "outputs": [{"name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]

    def __init__(self, web3_provider: str, pool_address: Optional[str] = None):
        """
        Initialize Uniswap V3 client

        Args:
            web3_provider: Web3 provider URL
            pool_address: Optional custom pool address (default: ETH/USDC 0.05%)
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))

        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node")

        # Initialize pool contract
        pool_addr = pool_address or self.ETH_USDC_POOL_005
        self.pool: Contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(pool_addr),
            abi=self.POOL_ABI
        )

        # Determine token order (USDC is token0, WETH is token1 for 0x88e6...)
        self.token0 = self.pool.functions.token0().call()
        self.token1 = self.pool.functions.token1().call()

        logger.info(f"Uniswap V3 client initialized with pool {pool_addr}")
        logger.info(f"Token0: {self.token0}, Token1: {self.token1}")

    async def get_eth_usdc_price(
        self,
        timestamp: Optional[int] = None,
        block_number: Optional[int] = None
    ) -> Optional[float]:
        """
        Get ETH/USDC price from Uniswap pool

        Args:
            timestamp: Optional timestamp (will find corresponding block)
            block_number: Optional specific block number

        Returns:
            ETH price in USD or None
        """
        try:
            # Use latest block if none specified
            if block_number is None and timestamp is None:
                block_identifier = 'latest'
            elif block_number is not None:
                block_identifier = block_number
            else:
                # Would need to find block by timestamp - for now use latest
                logger.warning("Timestamp-to-block lookup not implemented, using latest")
                block_identifier = 'latest'

            # Get pool state
            slot0 = self.pool.functions.slot0().call(block_identifier=block_identifier)
            sqrt_price_x96 = slot0[0]

            # Convert sqrtPriceX96 to price
            # price = (sqrtPriceX96 / 2^96)^2
            price_ratio = (sqrt_price_x96 / (2 ** 96)) ** 2

            # For ETH/USDC pool where USDC is token0:
            # price_ratio = USDC per ETH
            # We need to invert to get ETH price in USDC
            # Also adjust for decimals (USDC: 6, ETH: 18)
            decimals_adjustment = 10 ** (18 - 6)
            eth_price = price_ratio * decimals_adjustment

            if eth_price <= 0 or eth_price > 1000000:
                logger.warning(f"Suspicious Uniswap price detected: ${eth_price}")
                return None

            logger.info(f"Uniswap ETH/USDC price: ${eth_price:.2f} at block {block_identifier}")
            return eth_price

        except Exception as e:
            logger.error(f"Error fetching Uniswap price: {e}")
            return None

    async def get_price_at_block(self, block_number: int) -> Optional[Dict]:
        """
        Get ETH price at specific block with metadata

        Args:
            block_number: Ethereum block number

        Returns:
            Dict with price data or None
        """
        price = await self.get_eth_usdc_price(block_number=block_number)

        if price:
            # Get block timestamp
            try:
                block = self.w3.eth.get_block(block_number)
                return {
                    'price_usd': price,
                    'timestamp': block['timestamp'],
                    'block_number': block_number,
                    'source': 'uniswap_v3',
                    'pool': self.pool.address,
                    'confidence_score': 0.85  # Lower than Chainlink
                }
            except Exception as e:
                logger.error(f"Error getting block info: {e}")
                return None

        return None