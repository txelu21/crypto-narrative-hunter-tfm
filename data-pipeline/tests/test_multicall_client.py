"""Tests for MulticallClient."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from services.balances.multicall_client import (
    MulticallClient,
    MulticallError,
    BALANCE_OF_SIGNATURE,
)


@pytest.fixture
def multicall_client():
    """Create a MulticallClient instance for testing."""
    with patch.dict("os.environ", {"ALCHEMY_API_KEY": "test_key"}):
        return MulticallClient(batch_size=10)


@pytest.fixture
def mock_rpc_response():
    """Mock RPC response data."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000"  # 1 ETH in wei
    }


class TestMulticallClient:
    """Test suite for MulticallClient."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = MulticallClient(api_key="explicit_key")
        assert client.api_key == "explicit_key"
        assert client.batch_size == 100  # Default value

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ALCHEMY_API_KEY"):
                MulticallClient()

    def test_init_with_custom_batch_size(self):
        """Test initialization with custom batch size."""
        with patch.dict("os.environ", {"ALCHEMY_API_KEY": "test_key"}):
            client = MulticallClient(batch_size=50)
            assert client.batch_size == 50

    def test_encode_balance_call(self, multicall_client):
        """Test encoding of balanceOf call data."""
        wallet = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
        calldata = multicall_client._encode_balance_call(wallet)

        assert calldata.startswith(BALANCE_OF_SIGNATURE)
        assert len(calldata) == 74  # 10 chars for signature + 64 for padded address

    @pytest.mark.asyncio
    async def test_get_eth_balance_success(self, multicall_client, mock_rpc_response):
        """Test successful ETH balance retrieval."""
        multicall_client._rpc_call = AsyncMock(return_value=mock_rpc_response["result"])

        balance = await multicall_client._get_eth_balance(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
        )

        assert balance == 1000000000000000000  # 1 ETH in wei
        multicall_client._rpc_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_eth_balance_with_block_number(self, multicall_client):
        """Test ETH balance retrieval with specific block number."""
        multicall_client._rpc_call = AsyncMock(return_value="0x64")

        block_number = 12345678
        balance = await multicall_client._get_eth_balance(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            block_number=block_number,
        )

        # Verify block number was passed as hex
        call_args = multicall_client._rpc_call.call_args
        assert call_args[0][1][1] == hex(block_number)

    @pytest.mark.asyncio
    async def test_get_token_balance_success(self, multicall_client, mock_rpc_response):
        """Test successful token balance retrieval."""
        multicall_client._rpc_call = AsyncMock(return_value=mock_rpc_response["result"])

        balance = await multicall_client._get_token_balance(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        )

        assert balance == 1000000000000000000

    @pytest.mark.asyncio
    async def test_get_token_balance_zero(self, multicall_client):
        """Test token balance retrieval for zero balance."""
        multicall_client._rpc_call = AsyncMock(return_value="0x")

        balance = await multicall_client._get_token_balance(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        )

        assert balance == 0

    @pytest.mark.asyncio
    async def test_get_token_balance_failure(self, multicall_client):
        """Test token balance retrieval when call fails."""
        multicall_client._rpc_call = AsyncMock(side_effect=Exception("RPC error"))

        balance = await multicall_client._get_token_balance(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "0xInvalidToken",
        )

        assert balance is None

    @pytest.mark.asyncio
    async def test_get_wallet_balances_success(self, multicall_client):
        """Test successful wallet balance retrieval for multiple tokens."""
        # Mock ETH balance
        eth_balance = 1000000000000000000  # 1 ETH
        multicall_client._get_eth_balance = AsyncMock(return_value=eth_balance)

        # Mock token balances
        token_balances = [
            500000000,  # USDC (6 decimals)
            2000000000000000000,  # DAI (18 decimals)
            0,  # Zero balance
        ]

        multicall_client._get_token_balance = AsyncMock(
            side_effect=token_balances
        )

        token_addresses = [
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0xZeroBalance",
        ]

        balances = await multicall_client.get_wallet_balances(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            token_addresses,
        )

        # Should include ETH and non-zero token balances
        assert "ETH" in balances
        assert balances["ETH"] == eth_balance
        assert len(balances) == 3  # ETH + 2 non-zero tokens

    @pytest.mark.asyncio
    async def test_get_wallet_balances_skip_zero(self, multicall_client):
        """Test that zero balances are excluded when skip_zero_balances=True."""
        multicall_client._get_eth_balance = AsyncMock(return_value=0)
        multicall_client._get_token_balance = AsyncMock(return_value=0)

        token_addresses = ["0xToken1", "0xToken2"]

        balances = await multicall_client.get_wallet_balances(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            token_addresses,
            skip_zero_balances=True,
        )

        # All balances are zero and should be excluded
        assert len(balances) == 0

    @pytest.mark.asyncio
    async def test_get_wallet_balances_include_zero(self, multicall_client):
        """Test that zero balances are included when skip_zero_balances=False."""
        multicall_client._get_eth_balance = AsyncMock(return_value=0)
        multicall_client._get_token_balance = AsyncMock(return_value=0)

        token_addresses = ["0xToken1"]

        balances = await multicall_client.get_wallet_balances(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            token_addresses,
            skip_zero_balances=False,
        )

        # Zero balances should be included
        assert "ETH" in balances
        assert balances["ETH"] == 0
        assert len(balances) == 2  # ETH + 1 token

    @pytest.mark.asyncio
    async def test_get_multi_wallet_balances(self, multicall_client):
        """Test batch wallet balance retrieval."""
        # Mock get_wallet_balances to return test data
        async def mock_get_balances(wallet, tokens, block, skip_zero):
            return {
                "ETH": 1000000000000000000,
                tokens[0]: 500000000,
            }

        multicall_client.get_wallet_balances = AsyncMock(
            side_effect=mock_get_balances
        )

        wallet_addresses = ["0xWallet1", "0xWallet2", "0xWallet3"]
        token_addresses = ["0xToken1"]

        results = await multicall_client.get_multi_wallet_balances(
            wallet_addresses,
            token_addresses,
        )

        assert len(results) == 3
        assert all(wallet in results for wallet in wallet_addresses)
        assert multicall_client.get_wallet_balances.call_count == 3

    @pytest.mark.asyncio
    async def test_get_multi_wallet_balances_with_progress(self, multicall_client):
        """Test multi-wallet balance retrieval with progress callback."""
        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        multicall_client.get_wallet_balances = AsyncMock(
            return_value={"ETH": 1000000000000000000}
        )

        wallet_addresses = ["0xWallet1", "0xWallet2"]
        token_addresses = ["0xToken1"]

        await multicall_client.get_multi_wallet_balances(
            wallet_addresses,
            token_addresses,
            progress_callback=progress_callback,
        )

        assert progress_calls == [(1, 2), (2, 2)]

    @pytest.mark.asyncio
    async def test_get_multi_wallet_balances_handles_errors(self, multicall_client):
        """Test that multi-wallet query handles individual wallet errors gracefully."""
        async def mock_get_balances(wallet, tokens, block, skip_zero):
            if wallet == "0xFailWallet":
                raise Exception("RPC error")
            return {"ETH": 1000000000000000000}

        multicall_client.get_wallet_balances = AsyncMock(
            side_effect=mock_get_balances
        )

        wallet_addresses = ["0xWallet1", "0xFailWallet", "0xWallet2"]
        token_addresses = ["0xToken1"]

        results = await multicall_client.get_multi_wallet_balances(
            wallet_addresses,
            token_addresses,
        )

        # All wallets should be in results, but failed wallet has empty dict
        assert len(results) == 3
        assert results["0xFailWallet"] == {}
        assert results["0xWallet1"]["ETH"] == 1000000000000000000

    @pytest.mark.asyncio
    async def test_get_balance_changes(self, multicall_client):
        """Test balance change detection between blocks."""
        # Mock balances at different blocks
        old_balances = {
            "ETH": 1000000000000000000,
            "0xToken1": 500000000,
            "0xToken2": 1000000000,
        }

        new_balances = {
            "ETH": 2000000000000000000,  # Changed
            "0xToken1": 500000000,  # Unchanged
            "0xToken3": 300000000,  # New token
            # Token2 removed (balance went to 0)
        }

        # Mock to return old balances on first call, new balances on second call
        multicall_client.get_wallet_balances = AsyncMock()
        multicall_client.get_wallet_balances.side_effect = [old_balances, new_balances]

        changes = await multicall_client.get_balance_changes(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            ["0xToken1", "0xToken2", "0xToken3"],
            from_block=100,
            to_block=200,
        )

        # Should detect ETH change, Token2 removed, and Token3 added
        assert "ETH" in changes
        assert changes["ETH"] == (1000000000000000000, 2000000000000000000)

        assert "0xToken2" in changes
        assert changes["0xToken2"] == (1000000000, 0)

        assert "0xToken3" in changes
        assert changes["0xToken3"] == (0, 300000000)

        # Token1 should not be in changes (unchanged)
        assert "0xToken1" not in changes

    @pytest.mark.asyncio
    async def test_rpc_call_error_handling(self, multicall_client):
        """Test RPC call error handling."""
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "insufficient funds"}
        }

        with patch("aiohttp.ClientSession") as mock_session:
            # Create proper async context manager mocks
            mock_response = MagicMock()
            mock_response.json = AsyncMock(return_value=error_response)
            mock_response.raise_for_status = MagicMock()
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_post)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            with pytest.raises(MulticallError, match="insufficient funds"):
                await multicall_client._rpc_call("eth_call", [])

    @pytest.mark.asyncio
    async def test_batch_processing(self, multicall_client):
        """Test that token queries are batched correctly."""
        # Set small batch size for testing
        multicall_client.batch_size = 2

        multicall_client._get_eth_balance = AsyncMock(return_value=1000000000000000000)
        multicall_client._get_token_balance = AsyncMock(return_value=100000000)

        # 5 tokens with batch size 2 = 3 batches
        token_addresses = [f"0xToken{i}" for i in range(5)]

        balances = await multicall_client.get_wallet_balances(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            token_addresses,
        )

        # All tokens should be queried
        assert multicall_client._get_token_balance.call_count == 5
        assert len(balances) == 6  # ETH + 5 tokens