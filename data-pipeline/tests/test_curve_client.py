"""Tests for Curve Finance API client."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aioresponses import aioresponses

from services.transactions.curve_client import CurveAPIError, CurveClient


@pytest.fixture
def curve_client():
    """Create a CurveClient instance for testing."""
    return CurveClient()


@pytest.fixture
def mock_pool_data():
    """Sample pool data for testing."""
    return {
        "success": True,
        "data": {
            "poolData": [
                {
                    "address": "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc",
                    "assetType": "stable",
                    "virtualPrice": "1001234567890123456",
                    "usdTotal": "1234567890.12",
                    "volumeUSD": "987654.32",
                    "coins": [
                        {
                            "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                            "symbol": "USDC",
                            "decimals": 6,
                        },
                        {
                            "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                            "symbol": "USDT",
                            "decimals": 6,
                        },
                        {
                            "address": "0x6b175474e89094c44da98b954eedeac495271d0f",
                            "symbol": "DAI",
                            "decimals": 18,
                        },
                    ],
                },
                {
                    "address": "0xd632f22692fac7611d2aa1c0d552930d43caed3b",
                    "assetType": "crypto",
                    "virtualPrice": "1234567890123456789",
                    "usdTotal": "9876543210.98",
                    "volumeUSD": "1234567.89",
                    "coins": [
                        {
                            "address": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                            "symbol": "WETH",
                            "decimals": 18,
                        },
                        {
                            "address": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
                            "symbol": "WBTC",
                            "decimals": 8,
                        },
                    ],
                },
            ],
        },
    }


@pytest.fixture
def mock_factory_pool_data():
    """Sample factory pool data for testing."""
    return {
        "success": True,
        "data": {
            "poolData": [
                {
                    "address": "0x1234567890abcdef1234567890abcdef12345678",
                    "assetType": "stable",
                    "virtualPrice": "999999999999999999",
                    "usdTotal": "5000000.00",
                    "coins": [
                        {
                            "address": "0x853d955acef822db058eb8505911ed77f175b99e",
                            "symbol": "FRAX",
                            "decimals": 18,
                        },
                        {
                            "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                            "symbol": "USDC",
                            "decimals": 6,
                        },
                    ],
                },
            ],
        },
    }


class TestCurveClient:
    """Test suite for CurveClient."""

    @pytest.mark.asyncio
    async def test_get_all_pools(self, curve_client, mock_pool_data, mock_factory_pool_data):
        """Test fetching all Curve pools."""
        with aioresponses() as m:
            # Mock main pools response
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )

            # Mock factory pools response
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload=mock_factory_pool_data,
            )

            pools = await curve_client.get_all_pools(include_factory=True)

            assert len(pools) == 3  # 2 main + 1 factory
            assert pools[0]["address"] == "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"
            assert pools[2]["is_factory"] is True

    @pytest.mark.asyncio
    async def test_get_all_pools_main_only(self, curve_client, mock_pool_data):
        """Test fetching only main Curve pools."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )

            pools = await curve_client.get_all_pools(include_factory=False)

            assert len(pools) == 2
            assert all("is_factory" not in pool for pool in pools)

    @pytest.mark.asyncio
    async def test_get_pool_info(self, curve_client, mock_pool_data):
        """Test getting specific pool information."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            pool_info = await curve_client.get_pool_info(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"
            )

            assert pool_info is not None
            assert pool_info["assetType"] == "stable"
            assert len(pool_info["coins"]) == 3

    @pytest.mark.asyncio
    async def test_get_pool_coins(self, curve_client, mock_pool_data):
        """Test getting coin information for a pool."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            coins = await curve_client.get_pool_coins(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"
            )

            assert len(coins) == 3
            assert coins[0]["symbol"] == "USDC"
            assert coins[1]["symbol"] == "USDT"
            assert coins[2]["symbol"] == "DAI"

    @pytest.mark.asyncio
    async def test_get_pool_virtual_price(self, curve_client, mock_pool_data):
        """Test getting virtual price for a pool."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            virtual_price = await curve_client.get_pool_virtual_price(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"
            )

            assert virtual_price == float(1001234567890123456)

    @pytest.mark.asyncio
    async def test_get_pool_type_stable(self, curve_client, mock_pool_data):
        """Test determining pool type for stablecoin pool."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            pool_type = await curve_client.get_pool_type(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"
            )

            assert pool_type == "stable"

    @pytest.mark.asyncio
    async def test_get_pool_type_crypto(self, curve_client, mock_pool_data):
        """Test determining pool type for crypto pool."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            pool_type = await curve_client.get_pool_type(
                "0xd632f22692fac7611d2aa1c0d552930d43caed3b"
            )

            assert pool_type == "crypto"

    @pytest.mark.asyncio
    async def test_get_coin_index(self, curve_client, mock_pool_data):
        """Test getting coin index within a pool."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            # Load pools to populate cache
            await curve_client.get_all_pools()

            # USDC index
            index = curve_client.get_coin_index(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc",
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            )
            assert index == 0

            # DAI index
            index = curve_client.get_coin_index(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc",
                "0x6b175474e89094c44da98b954eedeac495271d0f",
            )
            assert index == 2

    @pytest.mark.asyncio
    async def test_get_pools_by_token(self, curve_client, mock_pool_data):
        """Test finding pools containing a specific token."""
        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            # Find pools containing USDC
            pools = await curve_client.get_pools_by_token(
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
            )

            assert len(pools) == 1
            assert pools[0]["address"] == "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"

    @pytest.mark.asyncio
    async def test_extract_swap_transactions(self, curve_client, mock_pool_data):
        """Test extracting swap transactions from Curve pool."""
        mock_web3 = AsyncMock()

        # Mock swap event logs
        mock_logs = [
            {
                "transactionHash": "0xabc123",
                "blockNumber": "0x1234567",
                "logIndex": "0x0",
                "address": "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc",
                "topics": [
                    "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140",
                    "0x000000000000000000000000abcdef1234567890abcdef1234567890abcdef12",
                ],
                "data": "0x" + "0" * 62 + "00" +  # sold_id = 0
                       "0" * 48 + "000000000005f5e100" +  # tokens_sold = 100000000
                       "0" * 62 + "01" +  # bought_id = 1
                       "0" * 48 + "0000000000062d5b68",  # tokens_bought = 103700328
            }
        ]

        # Mock should return logs for first call, empty for second
        mock_web3.eth_get_logs = AsyncMock(side_effect=[mock_logs, []])

        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            swaps = await curve_client.extract_swap_transactions(
                pool_address="0x3175df0976dfa876431c2e9ee6bc45b65d3473cc",
                wallet_address="0xabcdef1234567890abcdef1234567890abcdef12",
                from_block=1000000,
                to_block=1100000,
                web3_client=mock_web3,
            )

            assert len(swaps) == 1
            assert swaps[0]["tx_hash"] == "0xabc123"
            assert swaps[0]["token_in_symbol"] == "USDC"
            assert swaps[0]["token_out_symbol"] == "USDT"
            assert swaps[0]["amount_in"] == 100000000
            assert swaps[0]["amount_out"] == 103700328

    def test_decode_curve_swap_event(self, curve_client, mock_pool_data):
        """Test decoding Curve swap event log."""
        event_log = {
            "transactionHash": "0xabc123",
            "blockNumber": "0x1234567",
            "logIndex": "0x0",
            "address": "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc",
            "topics": [
                "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140",
                "0x000000000000000000000000abcdef1234567890abcdef1234567890abcdef12",
            ],
            "data": ("0x" +
                   "0" * 64 +  # sold_id = 0 (32 bytes)
                   "0" * 48 + "000000000005f5e100" +  # tokens_sold = 100000000 (32 bytes)
                   "0" * 62 + "02" +  # bought_id = 2 (32 bytes)
                   "0" * 48 + "0de0b6b3a7640000"),  # tokens_bought = 1e18 (32 bytes)
        }

        pool_info = mock_pool_data["data"]["poolData"][0]

        decoded = curve_client._decode_curve_swap_event(
            event_log,
            pool_info,
            "stable",
        )

        assert decoded is not None
        assert decoded["tx_hash"] == "0xabc123"
        assert decoded["buyer"] == "0xabcdef1234567890abcdef1234567890abcdef12"
        assert decoded["coin_index_in"] == 0
        assert decoded["coin_index_out"] == 2
        assert decoded["amount_in"] == 100000000
        assert decoded["amount_out"] == 1000000000000000000
        assert decoded["token_in_symbol"] == "USDC"
        assert decoded["token_out_symbol"] == "DAI"

    @pytest.mark.asyncio
    async def test_process_stable_pool_swap(self, curve_client):
        """Test processing a stablecoin pool swap."""
        swap_data = {
            "amount_in": 100000000,  # 100 USDC (6 decimals)
            "amount_out": 99800000,  # 99.8 USDT (6 decimals)
            "token_in_decimals": 6,
            "token_out_decimals": 6,
        }

        virtual_price = 1001234567890123456  # ~1.001
        eth_usd_price = 2000.0

        processed = await curve_client.process_stable_pool_swap(
            swap_data,
            virtual_price,
            eth_usd_price,
        )

        assert processed["processing_type"] == "stable_pool"
        assert processed["slippage_percentage"] == pytest.approx(0.2, rel=0.01)
        assert processed["eth_value_in"] is not None
        assert processed["eth_value_out"] is not None

    @pytest.mark.asyncio
    async def test_process_crypto_pool_swap(self, curve_client, mock_pool_data):
        """Test processing a crypto pool swap."""
        swap_data = {
            "pool_address": "0xd632f22692fac7611d2aa1c0d552930d43caed3b",
            "amount_in": 1000000000000000000,  # 1 WETH
            "amount_out": 6000000,  # 0.06 WBTC (8 decimals)
            "token_in_decimals": 18,
            "token_out_decimals": 8,
            "token_in_address": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            "token_out_address": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
            "coin_index_in": 0,
            "coin_index_out": 1,
        }

        mock_web3 = AsyncMock()

        # Mock pool balances
        mock_web3.encode_function_call = MagicMock(return_value="0x1234")
        mock_web3.eth_call = AsyncMock(side_effect=[
            "0x0000000000000000000000000000000000000000000000000de0b6b3a7640000",  # 1000 WETH
            "0x00000000000000000000000000000000000000000000000000000000000003e8",  # 1000 WBTC (scaled)
        ])

        with aioresponses() as m:
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            processed = await curve_client.process_crypto_pool_swap(
                swap_data,
                "0xd632f22692fac7611d2aa1c0d552930d43caed3b",
                19000000,
                mock_web3,
            )

            assert processed["processing_type"] == "crypto_pool"
            assert "effective_price" in processed
            assert "price_impact_percentage" in processed

    def test_calculate_price_impact(self, curve_client):
        """Test price impact calculation."""
        # Perfect 1:1 swap
        impact = curve_client._calculate_price_impact(100.0, 100.0)
        assert impact == 0.0

        # 0.2% slippage
        impact = curve_client._calculate_price_impact(100.0, 99.8)
        assert impact == pytest.approx(0.2, rel=0.01)

        # 1% slippage
        impact = curve_client._calculate_price_impact(100.0, 99.0)
        assert impact == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_api_error_handling(self, curve_client):
        """Test error handling for API failures."""
        with aioresponses() as m:
            # Mock error response
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload={"success": False, "error": "API rate limit exceeded"},
            )

            with pytest.raises(CurveAPIError) as exc_info:
                await curve_client.get_all_pools()

            assert "API rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, curve_client):
        """Test retry logic on timeout errors."""
        with aioresponses() as m:
            # First attempt: timeout
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                exception=asyncio.TimeoutError(),
            )

            # Second attempt: success
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload={"success": True, "data": {"poolData": []}},
            )

            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            # Should succeed after retry
            pools = await curve_client.get_all_pools()
            assert pools == []

    @pytest.mark.asyncio
    async def test_is_curve_pool(self, curve_client, mock_pool_data):
        """Test checking if an address is a Curve pool."""
        with aioresponses() as m:
            # Mock both calls for valid pool check
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            # Valid pool
            is_pool = await curve_client.is_curve_pool(
                "0x3175df0976dfa876431c2e9ee6bc45b65d3473cc"
            )
            assert is_pool is True

            # Mock both calls for invalid pool check
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/main",
                payload=mock_pool_data,
            )
            m.get(
                "https://api.curve.fi/api/getPools/ethereum/factory",
                payload={"success": True, "data": {"poolData": []}},
            )

            # Invalid pool
            is_pool = await curve_client.is_curve_pool(
                "0x0000000000000000000000000000000000000000"
            )
            assert is_pool is False