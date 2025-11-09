"""
Transaction decoding and normalization module.

Handles decoding of swap transactions from multiple DEX protocols including:
- Uniswap V2 router calls and direct pool interactions
- Uniswap V3 router calls with multi-hop support
- Curve pool swaps with virtual price calculations

Includes price normalization to ETH using pool reserves at transaction time.
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from eth_typing import HexStr
from web3 import Web3
from web3.types import TxReceipt, LogReceipt

from .models import DecodedSwap, PriceImpact

logger = logging.getLogger(__name__)

# Event signatures
UNISWAP_V2_SWAP_SIGNATURE = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
UNISWAP_V3_SWAP_SIGNATURE = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
CURVE_EXCHANGE_SIGNATURE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"

# Known contract addresses
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
UNISWAP_V3_ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
UNISWAP_V3_ROUTER2 = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"


class SwapDecoder:
    """Decodes swap transactions from multiple DEX protocols."""

    def __init__(self, web3: Optional[Web3] = None):
        """
        Initialize decoder with Web3 instance.

        Args:
            web3: Web3 instance for on-chain calls. If None, creates default instance.
        """
        self.web3 = web3 or Web3()
        self._pool_cache: Dict[str, Dict[str, Any]] = {}

    def decode_transaction(
        self,
        tx_hash: str,
        receipt: TxReceipt,
        logs: List[LogReceipt],
        wallet_address: str
    ) -> List[DecodedSwap]:
        """
        Decode swap transaction from receipt and logs.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt
            logs: Transaction logs
            wallet_address: Wallet address that initiated the swap

        Returns:
            List of decoded swap events
        """
        dex_name = self._identify_dex(receipt.get("to", ""))

        if dex_name == "uniswap_v2":
            return self._decode_uniswap_v2_swaps(tx_hash, receipt, logs, wallet_address)
        elif dex_name == "uniswap_v3":
            return self._decode_uniswap_v3_swaps(tx_hash, receipt, logs, wallet_address)
        elif dex_name == "curve":
            return self._decode_curve_swaps(tx_hash, receipt, logs, wallet_address)
        else:
            # Try to decode from logs if router not recognized
            return self._decode_from_logs(tx_hash, receipt, logs, wallet_address)

    def _identify_dex(self, contract_address: str) -> Optional[str]:
        """
        Identify DEX from contract address.

        Args:
            contract_address: Contract address

        Returns:
            DEX name or None if not recognized
        """
        contract_lower = contract_address.lower() if contract_address else ""

        if contract_lower == UNISWAP_V2_ROUTER.lower():
            return "uniswap_v2"
        elif contract_lower in [UNISWAP_V3_ROUTER.lower(), UNISWAP_V3_ROUTER2.lower()]:
            return "uniswap_v3"
        elif self._is_curve_pool(contract_address):
            return "curve"

        return None

    def _is_curve_pool(self, address: str) -> bool:
        """Check if address is a Curve pool."""
        # TODO: Implement proper Curve pool registry check
        # For now, check if address has Curve-like events
        return False

    def _decode_uniswap_v2_swaps(
        self,
        tx_hash: str,
        receipt: TxReceipt,
        logs: List[LogReceipt],
        wallet_address: str
    ) -> List[DecodedSwap]:
        """
        Decode Uniswap V2 swap events.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt
            logs: Transaction logs
            wallet_address: Wallet address

        Returns:
            List of decoded swaps
        """
        decoded_swaps = []

        # Filter swap events
        swap_logs = [
            log for log in logs
            if log.get("topics") and (
                (hasattr(log["topics"][0], "hex") and log["topics"][0].hex() == UNISWAP_V2_SWAP_SIGNATURE) or
                (isinstance(log["topics"][0], bytes) and "0x" + log["topics"][0].hex() == UNISWAP_V2_SWAP_SIGNATURE)
            )
        ]

        for log in swap_logs:
            try:
                pool_address = log["address"]

                # Decode event data
                # Swap event: Swap(address indexed sender, uint amount0In, uint amount1In, uint amount0Out, uint amount1Out, address indexed to)
                data = log["data"]
                if isinstance(data, str):
                    data = bytes.fromhex(data if not data.startswith("0x") else data[2:])

                amount0_in = int.from_bytes(data[0:32], byteorder="big")
                amount1_in = int.from_bytes(data[32:64], byteorder="big")
                amount0_out = int.from_bytes(data[64:96], byteorder="big")
                amount1_out = int.from_bytes(data[96:128], byteorder="big")

                # Determine swap direction
                token_in, amount_in, token_out, amount_out = self._parse_v2_swap_direction(
                    pool_address, amount0_in, amount1_in, amount0_out, amount1_out, receipt["blockNumber"]
                )

                # Calculate ETH value
                eth_value_in = self._normalize_to_eth(
                    token_in, amount_in, pool_address, receipt["blockNumber"]
                )
                eth_value_out = self._normalize_to_eth(
                    token_out, amount_out, pool_address, receipt["blockNumber"]
                )

                # Calculate slippage
                slippage = self._calculate_slippage(eth_value_in, eth_value_out)

                decoded_swap = DecodedSwap(
                    tx_hash=tx_hash,
                    block_number=receipt["blockNumber"],
                    wallet_address=wallet_address,
                    dex_name="uniswap_v2",
                    pool_address=pool_address,
                    token_in=token_in,
                    amount_in=str(amount_in),
                    token_out=token_out,
                    amount_out=str(amount_out),
                    eth_value_in=str(eth_value_in),
                    eth_value_out=str(eth_value_out),
                    gas_used=receipt["gasUsed"],
                    gas_price_gwei=str(Decimal(receipt["effectiveGasPrice"]) / Decimal(1e9)),
                    transaction_status="success" if receipt["status"] == 1 else "failed",
                    slippage_percentage=str(slippage)
                )

                decoded_swaps.append(decoded_swap)

            except Exception as e:
                logger.error(f"Error decoding Uniswap V2 swap in tx {tx_hash}: {e}")

        return decoded_swaps

    def _decode_uniswap_v3_swaps(
        self,
        tx_hash: str,
        receipt: TxReceipt,
        logs: List[LogReceipt],
        wallet_address: str
    ) -> List[DecodedSwap]:
        """
        Decode Uniswap V3 swap events.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt
            logs: Transaction logs
            wallet_address: Wallet address

        Returns:
            List of decoded swaps
        """
        decoded_swaps = []

        # Filter swap events
        swap_logs = [
            log for log in logs
            if log.get("topics") and (
                (hasattr(log["topics"][0], "hex") and log["topics"][0].hex() == UNISWAP_V3_SWAP_SIGNATURE) or
                (isinstance(log["topics"][0], bytes) and "0x" + log["topics"][0].hex() == UNISWAP_V3_SWAP_SIGNATURE)
            )
        ]

        for log in swap_logs:
            try:
                pool_address = log["address"]

                # Decode event data
                # Swap event: Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
                data = log["data"]
                if isinstance(data, str):
                    data = bytes.fromhex(data if not data.startswith("0x") else data[2:])

                # Parse amounts (signed integers)
                amount0 = int.from_bytes(data[0:32], byteorder="big", signed=True)
                amount1 = int.from_bytes(data[32:64], byteorder="big", signed=True)

                # Determine swap direction from signs
                if amount0 > 0 and amount1 < 0:
                    # Swapped token1 for token0
                    token_in_index = 1
                    amount_in = abs(amount1)
                    token_out_index = 0
                    amount_out = abs(amount0)
                elif amount0 < 0 and amount1 > 0:
                    # Swapped token0 for token1
                    token_in_index = 0
                    amount_in = abs(amount0)
                    token_out_index = 1
                    amount_out = abs(amount1)
                else:
                    logger.warning(f"Unexpected V3 swap amounts in tx {tx_hash}: {amount0}, {amount1}")
                    continue

                # Get pool tokens
                token_in, token_out = self._get_v3_pool_tokens(pool_address, token_in_index, token_out_index)

                # Calculate ETH value
                eth_value_in = self._normalize_to_eth(
                    token_in, amount_in, pool_address, receipt["blockNumber"]
                )
                eth_value_out = self._normalize_to_eth(
                    token_out, amount_out, pool_address, receipt["blockNumber"]
                )

                # Calculate slippage
                slippage = self._calculate_slippage(eth_value_in, eth_value_out)

                decoded_swap = DecodedSwap(
                    tx_hash=tx_hash,
                    block_number=receipt["blockNumber"],
                    wallet_address=wallet_address,
                    dex_name="uniswap_v3",
                    pool_address=pool_address,
                    token_in=token_in,
                    amount_in=str(amount_in),
                    token_out=token_out,
                    amount_out=str(amount_out),
                    eth_value_in=str(eth_value_in),
                    eth_value_out=str(eth_value_out),
                    gas_used=receipt["gasUsed"],
                    gas_price_gwei=str(Decimal(receipt["effectiveGasPrice"]) / Decimal(1e9)),
                    transaction_status="success" if receipt["status"] == 1 else "failed",
                    slippage_percentage=str(slippage)
                )

                decoded_swaps.append(decoded_swap)

            except Exception as e:
                logger.error(f"Error decoding Uniswap V3 swap in tx {tx_hash}: {e}")

        return decoded_swaps

    def _decode_curve_swaps(
        self,
        tx_hash: str,
        receipt: TxReceipt,
        logs: List[LogReceipt],
        wallet_address: str
    ) -> List[DecodedSwap]:
        """
        Decode Curve swap events.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt
            logs: Transaction logs
            wallet_address: Wallet address

        Returns:
            List of decoded swaps
        """
        decoded_swaps = []

        # Filter TokenExchange events
        exchange_logs = [
            log for log in logs
            if log.get("topics") and (
                (hasattr(log["topics"][0], "hex") and log["topics"][0].hex() == CURVE_EXCHANGE_SIGNATURE) or
                (isinstance(log["topics"][0], bytes) and "0x" + log["topics"][0].hex() == CURVE_EXCHANGE_SIGNATURE)
            )
        ]

        for log in exchange_logs:
            try:
                pool_address = log["address"]

                # Decode event data
                # TokenExchange(address indexed buyer, int128 sold_id, uint256 tokens_sold, int128 bought_id, uint256 tokens_bought)
                topics = log["topics"]
                buyer_topic = topics[1] if isinstance(topics[1], bytes) else bytes.fromhex(topics[1][2:])
                buyer = "0x" + buyer_topic.hex()[24:]

                data = log["data"]
                if isinstance(data, str):
                    data = bytes.fromhex(data if not data.startswith("0x") else data[2:])

                sold_id = int.from_bytes(data[0:32], byteorder="big", signed=True)
                tokens_sold = int.from_bytes(data[32:64], byteorder="big")
                bought_id = int.from_bytes(data[64:96], byteorder="big", signed=True)
                tokens_bought = int.from_bytes(data[96:128], byteorder="big")

                # Get token addresses from pool
                token_in = self._get_curve_coin(pool_address, sold_id)
                token_out = self._get_curve_coin(pool_address, bought_id)

                # Calculate ETH value using Curve virtual price
                eth_value_in = self._normalize_curve_to_eth(
                    token_in, tokens_sold, pool_address, receipt["blockNumber"]
                )
                eth_value_out = self._normalize_curve_to_eth(
                    token_out, tokens_bought, pool_address, receipt["blockNumber"]
                )

                # Calculate slippage
                slippage = self._calculate_slippage(eth_value_in, eth_value_out)

                decoded_swap = DecodedSwap(
                    tx_hash=tx_hash,
                    block_number=receipt["blockNumber"],
                    wallet_address=wallet_address,
                    dex_name="curve",
                    pool_address=pool_address,
                    token_in=token_in,
                    amount_in=str(tokens_sold),
                    token_out=token_out,
                    amount_out=str(tokens_bought),
                    eth_value_in=str(eth_value_in),
                    eth_value_out=str(eth_value_out),
                    gas_used=receipt["gasUsed"],
                    gas_price_gwei=str(Decimal(receipt["effectiveGasPrice"]) / Decimal(1e9)),
                    transaction_status="success" if receipt["status"] == 1 else "failed",
                    slippage_percentage=str(slippage)
                )

                decoded_swaps.append(decoded_swap)

            except Exception as e:
                logger.error(f"Error decoding Curve swap in tx {tx_hash}: {e}")

        return decoded_swaps

    def _decode_from_logs(
        self,
        tx_hash: str,
        receipt: TxReceipt,
        logs: List[LogReceipt],
        wallet_address: str
    ) -> List[DecodedSwap]:
        """
        Attempt to decode swaps from logs when router is not recognized.

        Args:
            tx_hash: Transaction hash
            receipt: Transaction receipt
            logs: Transaction logs
            wallet_address: Wallet address

        Returns:
            List of decoded swaps
        """
        # Try each decoding method
        decoded_swaps = []

        # Try Uniswap V2
        v2_swaps = self._decode_uniswap_v2_swaps(tx_hash, receipt, logs, wallet_address)
        decoded_swaps.extend(v2_swaps)

        # Try Uniswap V3
        v3_swaps = self._decode_uniswap_v3_swaps(tx_hash, receipt, logs, wallet_address)
        decoded_swaps.extend(v3_swaps)

        # Try Curve
        curve_swaps = self._decode_curve_swaps(tx_hash, receipt, logs, wallet_address)
        decoded_swaps.extend(curve_swaps)

        return decoded_swaps

    def _parse_v2_swap_direction(
        self,
        pool_address: str,
        amount0_in: int,
        amount1_in: int,
        amount0_out: int,
        amount1_out: int,
        block_number: int
    ) -> Tuple[str, int, str, int]:
        """
        Parse Uniswap V2 swap direction and get token addresses.

        Args:
            pool_address: Pool address
            amount0_in: Amount of token0 in
            amount1_in: Amount of token1 in
            amount0_out: Amount of token0 out
            amount1_out: Amount of token1 out
            block_number: Block number

        Returns:
            Tuple of (token_in, amount_in, token_out, amount_out)
        """
        pool_info = self._get_v2_pool_info(pool_address, block_number)

        if amount0_in > 0:
            # Swapped token0 for token1
            return pool_info["token0"], amount0_in, pool_info["token1"], amount1_out
        else:
            # Swapped token1 for token0
            return pool_info["token1"], amount1_in, pool_info["token0"], amount0_out

    def _get_v2_pool_info(self, pool_address: str, block_number: int) -> Dict[str, Any]:
        """
        Get Uniswap V2 pool information.

        Args:
            pool_address: Pool address
            block_number: Block number

        Returns:
            Pool info dict with token0, token1, reserve0, reserve1
        """
        # Check cache
        cache_key = f"{pool_address}_{block_number}"
        if cache_key in self._pool_cache:
            return self._pool_cache[cache_key]

        # TODO: Implement actual pool info fetching via Web3
        # For now, return placeholder
        pool_info = {
            "token0": "0x0000000000000000000000000000000000000000",
            "token1": WETH_ADDRESS,
            "reserve0": 0,
            "reserve1": 0
        }

        self._pool_cache[cache_key] = pool_info
        return pool_info

    def _get_v3_pool_tokens(
        self,
        pool_address: str,
        token_in_index: int,
        token_out_index: int
    ) -> Tuple[str, str]:
        """
        Get Uniswap V3 pool token addresses.

        Args:
            pool_address: Pool address
            token_in_index: Index of token in (0 or 1)
            token_out_index: Index of token out (0 or 1)

        Returns:
            Tuple of (token_in_address, token_out_address)
        """
        # TODO: Implement actual pool token fetching via Web3
        # For now, return placeholder
        if token_in_index == 0:
            return "0x0000000000000000000000000000000000000000", WETH_ADDRESS
        else:
            return WETH_ADDRESS, "0x0000000000000000000000000000000000000000"

    def _get_curve_coin(self, pool_address: str, coin_index: int) -> str:
        """
        Get Curve pool coin address by index.

        Args:
            pool_address: Pool address
            coin_index: Coin index

        Returns:
            Coin address
        """
        # TODO: Implement actual Curve coin fetching via Web3
        return "0x0000000000000000000000000000000000000000"

    def _normalize_to_eth(
        self,
        token_address: str,
        amount: int,
        pool_address: str,
        block_number: int
    ) -> Decimal:
        """
        Normalize token amount to ETH using pool price.

        Args:
            token_address: Token address
            amount: Token amount (in wei)
            pool_address: Pool address for pricing
            block_number: Block number for historical price

        Returns:
            ETH value as Decimal
        """
        # If already WETH, return as-is
        if token_address.lower() == WETH_ADDRESS.lower():
            return Decimal(amount) / Decimal(1e18)

        # TODO: Implement actual price calculation using pool reserves
        # For now, return placeholder conversion
        return Decimal(amount) / Decimal(1e18)

    def _normalize_curve_to_eth(
        self,
        token_address: str,
        amount: int,
        pool_address: str,
        block_number: int
    ) -> Decimal:
        """
        Normalize Curve token amount to ETH using virtual price.

        Args:
            token_address: Token address
            amount: Token amount (in wei)
            pool_address: Pool address
            block_number: Block number

        Returns:
            ETH value as Decimal
        """
        # If already WETH, return as-is
        if token_address.lower() == WETH_ADDRESS.lower():
            return Decimal(amount) / Decimal(1e18)

        # TODO: Implement actual Curve virtual price calculation
        return Decimal(amount) / Decimal(1e18)

    def _calculate_slippage(self, eth_value_in: Decimal, eth_value_out: Decimal) -> Decimal:
        """
        Calculate slippage percentage.

        Args:
            eth_value_in: ETH value in
            eth_value_out: ETH value out

        Returns:
            Slippage percentage (negative means loss)
        """
        if eth_value_in == 0:
            return Decimal(0)

        slippage = ((eth_value_out - eth_value_in) / eth_value_in) * Decimal(100)
        return slippage.quantize(Decimal("0.01"))