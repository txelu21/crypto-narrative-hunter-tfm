"""Alchemy JSON-RPC client for on-chain transaction data extraction.

This module provides a high-performance async client for interacting with the Alchemy API
to extract transaction logs, receipts, and other on-chain data with optimal compute unit usage.
"""

import asyncio
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import aiohttp
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = structlog.get_logger()


# Event signatures for DEX swaps
UNISWAP_V2_SWAP_SIGNATURE = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
UNISWAP_V3_SWAP_SIGNATURE = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
CURVE_TOKEN_EXCHANGE_SIGNATURE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"


class AlchemyComputeUnitError(Exception):
    """Raised when compute unit budget is exceeded."""
    pass


class AlchemyRPCError(Exception):
    """Raised when Alchemy returns an RPC error response."""
    pass


class AlchemyClient:
    """Async client for Alchemy JSON-RPC API with compute unit tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        compute_unit_budget: int = 500_000,
        rate_limit_semaphore: Optional[asyncio.Semaphore] = None,
        timeout: int = 30,
    ):
        """Initialize Alchemy client.

        Args:
            api_key: Alchemy API key (defaults to ALCHEMY_API_KEY env var)
            compute_unit_budget: Maximum compute units to consume
            rate_limit_semaphore: Optional semaphore for rate limiting
            timeout: Request timeout in seconds

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("ALCHEMY_API_KEY")
        if not self.api_key:
            raise ValueError("ALCHEMY_API_KEY environment variable is required")

        self.base_url = f"https://eth-mainnet.g.alchemy.com/v2/{self.api_key}"
        self.compute_unit_budget = compute_unit_budget
        self.compute_units_used = 0
        self.rate_limit = rate_limit_semaphore or asyncio.Semaphore(15)
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        self.logger = logger.bind(component="alchemy_client")

    def _check_compute_budget(self, estimated_cu: int = 0) -> None:
        """Check if operation would exceed compute unit budget.

        Args:
            estimated_cu: Estimated compute units for operation

        Raises:
            AlchemyComputeUnitError: If budget would be exceeded
        """
        if self.compute_units_used + estimated_cu > self.compute_unit_budget:
            raise AlchemyComputeUnitError(
                f"Compute unit budget exceeded: {self.compute_units_used}/{self.compute_unit_budget}"
            )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _rpc_call(
        self,
        method: str,
        params: List[Any],
        estimated_cu: int = 100,
    ) -> Any:
        """Make a JSON-RPC call to Alchemy.

        Args:
            method: RPC method name (e.g., "eth_getLogs")
            params: Method parameters
            estimated_cu: Estimated compute units for this call

        Returns:
            RPC response result

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
            aiohttp.ClientError: On HTTP errors
        """
        self._check_compute_budget(estimated_cu)

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
                        raise AlchemyRPCError(f"RPC error: {error_msg}")

                    # Update compute unit usage (estimated)
                    self.compute_units_used += estimated_cu

                    return data.get("result")

    async def get_swap_logs(
        self,
        wallet_address: str,
        from_block: int,
        to_block: int,
    ) -> List[Dict[str, Any]]:
        """Extract swap logs for a wallet within a block range.

        This method queries for Uniswap V2, V3, and Curve swap events where
        the wallet is a participant.

        Args:
            wallet_address: Ethereum address to query
            from_block: Starting block number
            to_block: Ending block number

        Returns:
            List of log entries

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
        """
        log = self.logger.bind(
            wallet_address=wallet_address,
            from_block=from_block,
            to_block=to_block,
            operation="get_swap_logs",
        )

        log.info("fetching_swap_logs")

        # Estimate compute units based on block range
        block_range = to_block - from_block
        estimated_cu = min(block_range, 2000) * 3  # 3 queries

        # Normalize address for topic filtering
        wallet_topic = self._address_to_topic(wallet_address)

        # Query Uniswap V2 swaps
        v2_logs = await self._rpc_call(
            "eth_getLogs",
            [
                {
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                    "topics": [UNISWAP_V2_SWAP_SIGNATURE, None, wallet_topic],
                }
            ],
            estimated_cu=estimated_cu // 3,
        )

        # Query Uniswap V3 swaps
        v3_logs = await self._rpc_call(
            "eth_getLogs",
            [
                {
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                    "topics": [UNISWAP_V3_SWAP_SIGNATURE, None, wallet_topic],
                }
            ],
            estimated_cu=estimated_cu // 3,
        )

        # Query Curve token exchanges
        curve_logs = await self._rpc_call(
            "eth_getLogs",
            [
                {
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                    "topics": [CURVE_TOKEN_EXCHANGE_SIGNATURE, wallet_topic],
                }
            ],
            estimated_cu=estimated_cu // 3,
        )

        all_logs = v2_logs + v3_logs + curve_logs

        log.info(
            "fetched_swap_logs",
            count=len(all_logs),
            v2_count=len(v2_logs),
            v3_count=len(v3_logs),
            curve_count=len(curve_logs),
        )

        return all_logs

    async def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction receipt for status and gas usage.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction receipt or None if not found

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
        """
        log = self.logger.bind(tx_hash=tx_hash, operation="get_transaction_receipt")

        try:
            receipt = await self._rpc_call(
                "eth_getTransactionReceipt",
                [tx_hash],
                estimated_cu=10,
            )

            if receipt:
                log.debug("fetched_receipt", status=receipt.get("status"))
            else:
                log.warning("receipt_not_found")

            return receipt

        except Exception as e:
            log.error("failed_to_fetch_receipt", error=str(e))
            raise

    async def get_transaction(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get full transaction details.

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction details or None if not found

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
        """
        log = self.logger.bind(tx_hash=tx_hash, operation="get_transaction")

        try:
            tx = await self._rpc_call(
                "eth_getTransactionByHash",
                [tx_hash],
                estimated_cu=15,
            )

            if tx:
                log.debug("fetched_transaction", block_number=tx.get("blockNumber"))
            else:
                log.warning("transaction_not_found")

            return tx

        except Exception as e:
            log.error("failed_to_fetch_transaction", error=str(e))
            raise

    async def get_block_by_number(
        self,
        block_number: int,
        include_transactions: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get block information by number.

        Args:
            block_number: Block number
            include_transactions: Whether to include full transaction objects

        Returns:
            Block data or None if not found

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
        """
        estimated_cu = 50 if include_transactions else 16

        try:
            block = await self._rpc_call(
                "eth_getBlockByNumber",
                [hex(block_number), include_transactions],
                estimated_cu=estimated_cu,
            )

            return block

        except Exception as e:
            self.logger.error(
                "failed_to_fetch_block",
                block_number=block_number,
                error=str(e),
            )
            raise

    async def get_current_block_number(self) -> int:
        """Get the current block number.

        Returns:
            Current block number

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
        """
        result = await self._rpc_call(
            "eth_blockNumber",
            [],
            estimated_cu=10,
        )

        return int(result, 16)

    async def get_logs_batch(
        self,
        log_filters: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Batch multiple eth_getLogs calls.

        Args:
            log_filters: List of log filter parameters

        Returns:
            List of log results (one per filter)

        Raises:
            AlchemyComputeUnitError: If compute budget exceeded
        """
        tasks = []
        for filter_params in log_filters:
            # Estimate CU based on block range
            from_block = int(filter_params.get("fromBlock", "0x0"), 16)
            to_block = int(filter_params.get("toBlock", "0x0"), 16)
            estimated_cu = min(to_block - from_block, 2000)

            task = self._rpc_call(
                "eth_getLogs",
                [filter_params],
                estimated_cu=estimated_cu,
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def _address_to_topic(self, address: str) -> str:
        """Convert Ethereum address to 32-byte topic format.

        Args:
            address: Ethereum address (with or without 0x prefix)

        Returns:
            32-byte hex string suitable for topic filtering
        """
        # Remove 0x prefix if present
        clean_address = address.lower().replace("0x", "")

        # Pad to 32 bytes (64 hex chars)
        padded = clean_address.zfill(64)

        return f"0x{padded}"

    def get_compute_unit_usage(self) -> Tuple[int, int, float]:
        """Get current compute unit usage statistics.

        Returns:
            Tuple of (used, budget, usage_percentage)
        """
        usage_pct = (self.compute_units_used / self.compute_unit_budget) * 100
        return self.compute_units_used, self.compute_unit_budget, usage_pct

    def reset_compute_units(self) -> None:
        """Reset compute unit counter to zero."""
        self.compute_units_used = 0
        self.logger.info("compute_units_reset")