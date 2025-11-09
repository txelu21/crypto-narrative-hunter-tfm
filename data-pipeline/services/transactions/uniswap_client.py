"""Uniswap V2/V3 subgraph GraphQL client for swap data enrichment.

This module provides async GraphQL clients for querying Uniswap subgraphs
to extract historical swap data, pricing information, and pool metadata.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = structlog.get_logger()


# Uniswap subgraph endpoints
UNISWAP_V2_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
UNISWAP_V3_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"


class UniswapSubgraphError(Exception):
    """Raised when subgraph query fails."""
    pass


class UniswapClient:
    """Async GraphQL client for Uniswap V2/V3 subgraphs."""

    def __init__(
        self,
        rate_limit_semaphore: Optional[asyncio.Semaphore] = None,
        timeout: int = 30,
    ):
        """Initialize Uniswap subgraph client.

        Args:
            rate_limit_semaphore: Optional semaphore for rate limiting
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit_semaphore or asyncio.Semaphore(10)
        self.timeout = timeout

        # Initialize transports
        self.v2_transport = AIOHTTPTransport(url=UNISWAP_V2_SUBGRAPH)
        self.v3_transport = AIOHTTPTransport(url=UNISWAP_V3_SUBGRAPH)

        self.logger = logger.bind(component="uniswap_client")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _query_v2(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query against Uniswap V2 subgraph.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result

        Raises:
            UniswapSubgraphError: If query fails
        """
        async with self.rate_limit:
            try:
                async with Client(
                    transport=self.v2_transport,
                    fetch_schema_from_transport=False,
                ) as session:
                    result = await session.execute(
                        gql(query),
                        variable_values=variables or {},
                    )
                    return result

            except Exception as e:
                self.logger.error("v2_query_failed", error=str(e), query=query[:100])
                raise UniswapSubgraphError(f"Uniswap V2 query failed: {e}") from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _query_v3(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query against Uniswap V3 subgraph.

        Args:
            query: GraphQL query string
            variables: Query variables

        Returns:
            Query result

        Raises:
            UniswapSubgraphError: If query fails
        """
        async with self.rate_limit:
            try:
                async with Client(
                    transport=self.v3_transport,
                    fetch_schema_from_transport=False,
                ) as session:
                    result = await session.execute(
                        gql(query),
                        variable_values=variables or {},
                    )
                    return result

            except Exception as e:
                self.logger.error("v3_query_failed", error=str(e), query=query[:100])
                raise UniswapSubgraphError(f"Uniswap V3 query failed: {e}") from e

    async def get_v2_swaps(
        self,
        wallet_address: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        first: int = 1000,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get Uniswap V2 swaps for a wallet address.

        Args:
            wallet_address: Ethereum address to query
            start_timestamp: Start timestamp (Unix)
            end_timestamp: End timestamp (Unix)
            first: Number of results to fetch
            skip: Number of results to skip (for pagination)

        Returns:
            List of swap records

        Raises:
            UniswapSubgraphError: If query fails
        """
        log = self.logger.bind(
            wallet_address=wallet_address,
            version="v2",
            operation="get_swaps",
        )

        query = """
        query GetSwaps($wallet: Bytes!, $startTime: Int, $endTime: Int, $first: Int!, $skip: Int!) {
            swaps(
                where: {
                    to: $wallet
                    timestamp_gte: $startTime
                    timestamp_lte: $endTime
                }
                first: $first
                skip: $skip
                orderBy: timestamp
                orderDirection: desc
            ) {
                id
                transaction {
                    id
                    blockNumber
                    timestamp
                }
                pair {
                    id
                    token0 {
                        id
                        symbol
                        decimals
                    }
                    token1 {
                        id
                        symbol
                        decimals
                    }
                    reserve0
                    reserve1
                }
                sender
                to
                amount0In
                amount1In
                amount0Out
                amount1Out
                amountUSD
            }
        }
        """

        variables = {
            "wallet": wallet_address.lower(),
            "startTime": start_timestamp,
            "endTime": end_timestamp,
            "first": first,
            "skip": skip,
        }

        log.info("querying_v2_swaps", first=first, skip=skip)

        result = await self._query_v2(query, variables)
        swaps = result.get("swaps", [])

        log.info("fetched_v2_swaps", count=len(swaps))

        return swaps

    async def get_v3_swaps(
        self,
        wallet_address: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        first: int = 1000,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get Uniswap V3 swaps for a wallet address.

        Args:
            wallet_address: Ethereum address to query
            start_timestamp: Start timestamp (Unix)
            end_timestamp: End timestamp (Unix)
            first: Number of results to fetch
            skip: Number of results to skip (for pagination)

        Returns:
            List of swap records

        Raises:
            UniswapSubgraphError: If query fails
        """
        log = self.logger.bind(
            wallet_address=wallet_address,
            version="v3",
            operation="get_swaps",
        )

        query = """
        query GetSwaps($wallet: Bytes!, $startTime: BigInt, $endTime: BigInt, $first: Int!, $skip: Int!) {
            swaps(
                where: {
                    recipient: $wallet
                    timestamp_gte: $startTime
                    timestamp_lte: $endTime
                }
                first: $first
                skip: $skip
                orderBy: timestamp
                orderDirection: desc
            ) {
                id
                transaction {
                    id
                    blockNumber
                    timestamp
                }
                pool {
                    id
                    token0 {
                        id
                        symbol
                        decimals
                    }
                    token1 {
                        id
                        symbol
                        decimals
                    }
                    feeTier
                    liquidity
                    sqrtPrice
                    tick
                }
                sender
                recipient
                amount0
                amount1
                amountUSD
                sqrtPriceX96
                tick
            }
        }
        """

        variables = {
            "wallet": wallet_address.lower(),
            "startTime": str(start_timestamp) if start_timestamp else None,
            "endTime": str(end_timestamp) if end_timestamp else None,
            "first": first,
            "skip": skip,
        }

        log.info("querying_v3_swaps", first=first, skip=skip)

        result = await self._query_v3(query, variables)
        swaps = result.get("swaps", [])

        log.info("fetched_v3_swaps", count=len(swaps))

        return swaps

    async def get_all_swaps_paginated(
        self,
        wallet_address: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        page_size: int = 1000,
        max_results: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all swaps for a wallet with automatic pagination.

        Args:
            wallet_address: Ethereum address to query
            start_timestamp: Start timestamp (Unix)
            end_timestamp: End timestamp (Unix)
            page_size: Number of results per page
            max_results: Maximum total results to fetch (None for all)

        Returns:
            Dict with 'v2' and 'v3' swap lists

        Raises:
            UniswapSubgraphError: If queries fail
        """
        log = self.logger.bind(
            wallet_address=wallet_address,
            operation="get_all_swaps_paginated",
        )

        log.info("starting_paginated_fetch")

        # Fetch V2 and V3 swaps concurrently
        v2_task = self._fetch_all_v2_swaps(
            wallet_address,
            start_timestamp,
            end_timestamp,
            page_size,
            max_results,
        )
        v3_task = self._fetch_all_v3_swaps(
            wallet_address,
            start_timestamp,
            end_timestamp,
            page_size,
            max_results,
        )

        v2_swaps, v3_swaps = await asyncio.gather(v2_task, v3_task)

        log.info(
            "completed_paginated_fetch",
            v2_count=len(v2_swaps),
            v3_count=len(v3_swaps),
            total=len(v2_swaps) + len(v3_swaps),
        )

        return {
            "v2": v2_swaps,
            "v3": v3_swaps,
        }

    async def _fetch_all_v2_swaps(
        self,
        wallet_address: str,
        start_timestamp: Optional[int],
        end_timestamp: Optional[int],
        page_size: int,
        max_results: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Fetch all V2 swaps with pagination."""
        all_swaps = []
        skip = 0

        while True:
            swaps = await self.get_v2_swaps(
                wallet_address,
                start_timestamp,
                end_timestamp,
                first=page_size,
                skip=skip,
            )

            if not swaps:
                break

            all_swaps.extend(swaps)
            skip += page_size

            if max_results and len(all_swaps) >= max_results:
                all_swaps = all_swaps[:max_results]
                break

            # Subgraph max skip limit is 5000
            if skip >= 5000:
                self.logger.warning("hit_subgraph_skip_limit", version="v2")
                break

        return all_swaps

    async def _fetch_all_v3_swaps(
        self,
        wallet_address: str,
        start_timestamp: Optional[int],
        end_timestamp: Optional[int],
        page_size: int,
        max_results: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Fetch all V3 swaps with pagination."""
        all_swaps = []
        skip = 0

        while True:
            swaps = await self.get_v3_swaps(
                wallet_address,
                start_timestamp,
                end_timestamp,
                first=page_size,
                skip=skip,
            )

            if not swaps:
                break

            all_swaps.extend(swaps)
            skip += page_size

            if max_results and len(all_swaps) >= max_results:
                all_swaps = all_swaps[:max_results]
                break

            # Subgraph max skip limit is 5000
            if skip >= 5000:
                self.logger.warning("hit_subgraph_skip_limit", version="v3")
                break

        return all_swaps

    async def get_pool_metadata(
        self,
        pool_address: str,
        version: str = "v2",
    ) -> Optional[Dict[str, Any]]:
        """Get pool metadata from subgraph.

        Args:
            pool_address: Pool/pair contract address
            version: "v2" or "v3"

        Returns:
            Pool metadata or None if not found

        Raises:
            UniswapSubgraphError: If query fails
        """
        log = self.logger.bind(
            pool_address=pool_address,
            version=version,
            operation="get_pool_metadata",
        )

        if version == "v2":
            query = """
            query GetPair($id: ID!) {
                pair(id: $id) {
                    id
                    token0 {
                        id
                        symbol
                        name
                        decimals
                    }
                    token1 {
                        id
                        symbol
                        name
                        decimals
                    }
                    reserve0
                    reserve1
                    reserveUSD
                    volumeUSD
                    txCount
                }
            }
            """

            result = await self._query_v2(query, {"id": pool_address.lower()})
            pool_data = result.get("pair")

        else:  # v3
            query = """
            query GetPool($id: ID!) {
                pool(id: $id) {
                    id
                    token0 {
                        id
                        symbol
                        name
                        decimals
                    }
                    token1 {
                        id
                        symbol
                        name
                        decimals
                    }
                    feeTier
                    liquidity
                    sqrtPrice
                    tick
                    volumeUSD
                    txCount
                }
            }
            """

            result = await self._query_v3(query, {"id": pool_address.lower()})
            pool_data = result.get("pool")

        if pool_data:
            log.info("fetched_pool_metadata", success=True)
        else:
            log.warning("pool_not_found")

        return pool_data

    async def close(self) -> None:
        """Close transport connections."""
        await self.v2_transport.close()
        await self.v3_transport.close()