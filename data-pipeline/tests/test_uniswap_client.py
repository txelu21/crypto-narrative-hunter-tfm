"""Tests for Uniswap subgraph GraphQL client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.transactions.uniswap_client import (
    UniswapClient,
    UniswapSubgraphError,
)


@pytest.fixture
def uniswap_client():
    """Create Uniswap client for testing."""
    return UniswapClient()


@pytest.mark.asyncio
async def test_initialization(uniswap_client):
    """Test client initialization."""
    assert uniswap_client.rate_limit._value == 10
    assert uniswap_client.timeout == 30
    assert uniswap_client.v2_transport is not None
    assert uniswap_client.v3_transport is not None


@pytest.mark.asyncio
async def test_get_v2_swaps_success(uniswap_client):
    """Test successful V2 swaps query."""
    mock_swaps = [
        {
            "id": "0xswap1",
            "transaction": {
                "id": "0xtx1",
                "blockNumber": "18000000",
                "timestamp": "1700000000",
            },
            "pair": {
                "id": "0xpair1",
                "token0": {
                    "id": "0xtoken0",
                    "symbol": "WETH",
                    "decimals": "18",
                },
                "token1": {
                    "id": "0xtoken1",
                    "symbol": "USDC",
                    "decimals": "6",
                },
                "reserve0": "1000000000000000000",
                "reserve1": "2000000000",
            },
            "amount0In": "1000000000000000000",
            "amount1In": "0",
            "amount0Out": "0",
            "amount1Out": "2000000000",
            "amountUSD": "2000.0",
        }
    ]

    with patch.object(uniswap_client, "_query_v2", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = {"swaps": mock_swaps}

        result = await uniswap_client.get_v2_swaps(
            wallet_address="0x1234567890123456789012345678901234567890",
            first=1000,
        )

        assert len(result) == 1
        assert result[0]["id"] == "0xswap1"
        assert result[0]["pair"]["token0"]["symbol"] == "WETH"


@pytest.mark.asyncio
async def test_get_v3_swaps_success(uniswap_client):
    """Test successful V3 swaps query."""
    mock_swaps = [
        {
            "id": "0xswap1",
            "transaction": {
                "id": "0xtx1",
                "blockNumber": "18000000",
                "timestamp": "1700000000",
            },
            "pool": {
                "id": "0xpool1",
                "token0": {
                    "id": "0xtoken0",
                    "symbol": "WETH",
                    "decimals": "18",
                },
                "token1": {
                    "id": "0xtoken1",
                    "symbol": "USDC",
                    "decimals": "6",
                },
                "feeTier": "3000",
                "liquidity": "10000000000",
                "sqrtPrice": "792281625142643375935439503360",
            },
            "amount0": "-1000000000000000000",
            "amount1": "2000000000",
            "amountUSD": "2000.0",
            "sqrtPriceX96": "792281625142643375935439503360",
            "tick": "85176",
        }
    ]

    with patch.object(uniswap_client, "_query_v3", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = {"swaps": mock_swaps}

        result = await uniswap_client.get_v3_swaps(
            wallet_address="0x1234567890123456789012345678901234567890",
            first=1000,
        )

        assert len(result) == 1
        assert result[0]["id"] == "0xswap1"
        assert result[0]["pool"]["feeTier"] == "3000"


@pytest.mark.asyncio
async def test_get_all_swaps_paginated(uniswap_client):
    """Test paginated fetching of all swaps."""
    mock_v2_swaps = [{"id": f"0xv2_{i}"} for i in range(5)]
    mock_v3_swaps = [{"id": f"0xv3_{i}"} for i in range(3)]

    with patch.object(
        uniswap_client, "_fetch_all_v2_swaps", new_callable=AsyncMock
    ) as mock_v2:
        with patch.object(
            uniswap_client, "_fetch_all_v3_swaps", new_callable=AsyncMock
        ) as mock_v3:
            mock_v2.return_value = mock_v2_swaps
            mock_v3.return_value = mock_v3_swaps

            result = await uniswap_client.get_all_swaps_paginated(
                wallet_address="0x1234567890123456789012345678901234567890"
            )

            assert len(result["v2"]) == 5
            assert len(result["v3"]) == 3
            assert result["v2"][0]["id"] == "0xv2_0"
            assert result["v3"][0]["id"] == "0xv3_0"


@pytest.mark.asyncio
async def test_get_pool_metadata_v2(uniswap_client):
    """Test fetching V2 pool metadata."""
    mock_pool = {
        "id": "0xpool1",
        "token0": {
            "id": "0xtoken0",
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "decimals": "18",
        },
        "token1": {
            "id": "0xtoken1",
            "symbol": "USDC",
            "name": "USD Coin",
            "decimals": "6",
        },
        "reserve0": "1000000000000000000",
        "reserve1": "2000000000",
        "reserveUSD": "4000000.0",
        "volumeUSD": "100000000.0",
        "txCount": "50000",
    }

    with patch.object(uniswap_client, "_query_v2", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = {"pair": mock_pool}

        result = await uniswap_client.get_pool_metadata(
            pool_address="0xpool1",
            version="v2",
        )

        assert result["id"] == "0xpool1"
        assert result["token0"]["symbol"] == "WETH"
        assert result["reserveUSD"] == "4000000.0"


@pytest.mark.asyncio
async def test_get_pool_metadata_v3(uniswap_client):
    """Test fetching V3 pool metadata."""
    mock_pool = {
        "id": "0xpool1",
        "token0": {
            "id": "0xtoken0",
            "symbol": "WETH",
            "name": "Wrapped Ether",
            "decimals": "18",
        },
        "token1": {
            "id": "0xtoken1",
            "symbol": "USDC",
            "name": "USD Coin",
            "decimals": "6",
        },
        "feeTier": "3000",
        "liquidity": "10000000000",
        "sqrtPrice": "792281625142643375935439503360",
        "tick": "85176",
        "volumeUSD": "100000000.0",
        "txCount": "50000",
    }

    with patch.object(uniswap_client, "_query_v3", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = {"pool": mock_pool}

        result = await uniswap_client.get_pool_metadata(
            pool_address="0xpool1",
            version="v3",
        )

        assert result["id"] == "0xpool1"
        assert result["token0"]["symbol"] == "WETH"
        assert result["feeTier"] == "3000"


@pytest.mark.asyncio
async def test_query_error_handling(uniswap_client):
    """Test error handling for failed queries."""
    with patch.object(uniswap_client, "_query_v2", new_callable=AsyncMock) as mock_query:
        # Side effect will be raised after retry attempts
        mock_query.side_effect = UniswapSubgraphError("Uniswap V2 query failed: Network error")

        with pytest.raises(UniswapSubgraphError, match="Uniswap V2 query failed"):
            await uniswap_client.get_v2_swaps(
                wallet_address="0x1234567890123456789012345678901234567890"
            )