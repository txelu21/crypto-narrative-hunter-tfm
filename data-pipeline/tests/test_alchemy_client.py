"""Tests for Alchemy JSON-RPC client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import ClientError

from services.transactions.alchemy_client import (
    AlchemyClient,
    AlchemyComputeUnitError,
    AlchemyRPCError,
    UNISWAP_V2_SWAP_SIGNATURE,
)


@pytest.fixture
def alchemy_client():
    """Create Alchemy client with test API key."""
    with patch.dict("os.environ", {"ALCHEMY_API_KEY": "test_key"}):
        return AlchemyClient(compute_unit_budget=10000)


@pytest.mark.asyncio
async def test_initialization_with_api_key():
    """Test client initialization with explicit API key."""
    client = AlchemyClient(api_key="explicit_key")
    assert client.api_key == "explicit_key"
    assert client.compute_unit_budget == 500_000
    assert client.compute_units_used == 0


@pytest.mark.asyncio
async def test_initialization_without_api_key():
    """Test client initialization fails without API key."""
    with patch.dict("os.environ", clear=True):
        with pytest.raises(ValueError, match="ALCHEMY_API_KEY"):
            AlchemyClient()


@pytest.mark.asyncio
async def test_compute_budget_check(alchemy_client):
    """Test compute unit budget checking."""
    alchemy_client.compute_units_used = 9500

    # Should raise on budget exceeded
    with pytest.raises(AlchemyComputeUnitError):
        alchemy_client._check_compute_budget(estimated_cu=600)

    # Should pass if under budget
    alchemy_client._check_compute_budget(estimated_cu=400)


@pytest.mark.asyncio
async def test_address_to_topic(alchemy_client):
    """Test address to 32-byte topic conversion."""
    address = "0x1234567890123456789012345678901234567890"
    expected = "0x0000000000000000000000001234567890123456789012345678901234567890"

    result = alchemy_client._address_to_topic(address)
    assert result == expected
    assert len(result) == 66  # 0x + 64 hex chars


@pytest.mark.asyncio
async def test_get_current_block_number(alchemy_client):
    """Test fetching current block number."""
    mock_response = {"jsonrpc": "2.0", "id": 1, "result": "0x10d4f"}

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value.__aenter__.return_value = mock_resp

        block_number = await alchemy_client.get_current_block_number()

        assert block_number == 68943
        assert alchemy_client.compute_units_used == 10


@pytest.mark.asyncio
async def test_get_swap_logs(alchemy_client):
    """Test fetching swap logs for a wallet."""
    wallet = "0x1234567890123456789012345678901234567890"
    from_block = 18000000
    to_block = 18001000

    mock_logs = [
        {
            "address": "0xPoolAddress",
            "topics": [UNISWAP_V2_SWAP_SIGNATURE, "0x...", "0x..."],
            "data": "0x...",
            "blockNumber": "0x1122500",
            "transactionHash": "0xabc123",
            "transactionIndex": "0x1",
            "blockHash": "0xdef456",
            "logIndex": "0x2",
            "removed": False,
        }
    ]

    mock_response = {"jsonrpc": "2.0", "id": 1, "result": mock_logs}

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value.__aenter__.return_value = mock_resp

        logs = await alchemy_client.get_swap_logs(wallet, from_block, to_block)

        # Should return 3x the mock (V2, V3, Curve queries)
        assert len(logs) == 3
        assert all(log in mock_logs for log in logs)


@pytest.mark.asyncio
async def test_rpc_call_with_error_response(alchemy_client):
    """Test RPC call handles error responses."""
    error_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32602, "message": "Invalid params"},
    }

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=error_response)
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value.__aenter__.return_value = mock_resp

        with pytest.raises(AlchemyRPCError, match="Invalid params"):
            await alchemy_client._rpc_call("eth_test", [], estimated_cu=10)


@pytest.mark.asyncio
async def test_get_compute_unit_usage(alchemy_client):
    """Test compute unit usage tracking."""
    alchemy_client.compute_units_used = 5000

    used, budget, percentage = alchemy_client.get_compute_unit_usage()

    assert used == 5000
    assert budget == 10000
    assert percentage == 50.0


@pytest.mark.asyncio
async def test_reset_compute_units(alchemy_client):
    """Test resetting compute unit counter."""
    alchemy_client.compute_units_used = 5000

    alchemy_client.reset_compute_units()

    assert alchemy_client.compute_units_used == 0