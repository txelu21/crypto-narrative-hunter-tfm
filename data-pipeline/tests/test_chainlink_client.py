"""
Tests for Chainlink Price Feed Integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from web3 import Web3
from services.prices.chainlink_client import ChainlinkPriceClient


@pytest.fixture
def mock_web3():
    """Mock Web3 instance"""
    with patch('services.prices.chainlink_client.Web3') as mock:
        web3_instance = Mock()
        web3_instance.is_connected.return_value = True
        web3_instance.eth = Mock()

        # Mock contract
        contract_instance = Mock()
        contract_instance.functions.decimals().call.return_value = 8
        contract_instance.functions.latestRoundData().call.return_value = (
            123456,  # roundId
            200000000000,  # answer (2000 USD with 8 decimals)
            1234567890,  # startedAt
            1234567890,  # updatedAt
            123456  # answeredInRound
        )
        contract_instance.address = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"

        web3_instance.eth.contract.return_value = contract_instance
        web3_instance.to_checksum_address = Web3.to_checksum_address

        mock.return_value = web3_instance
        mock.HTTPProvider = Mock()
        mock.to_checksum_address = Web3.to_checksum_address

        yield web3_instance


@pytest.mark.asyncio
class TestChainlinkPriceClient:
    """Test suite for ChainlinkPriceClient"""

    async def test_initialization(self, mock_web3):
        """Test client initialization"""
        client = ChainlinkPriceClient("http://localhost:8545")

        assert client.w3 is not None
        assert client.decimals == 8
        assert len(client.price_cache) == 0

    async def test_initialization_connection_failure(self):
        """Test initialization fails gracefully with no connection"""
        with patch('services.prices.chainlink_client.Web3') as mock:
            web3_instance = Mock()
            web3_instance.is_connected.return_value = False
            mock.return_value = web3_instance
            mock.HTTPProvider = Mock()

            with pytest.raises(ConnectionError):
                ChainlinkPriceClient("http://localhost:8545")

    async def test_fetch_price_at_block(self, mock_web3):
        """Test fetching price at specific block"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Mock latestRoundData
        mock_web3.eth.contract().functions.latestRoundData().call.return_value = (
            123456,  # roundId
            250000000000,  # answer (2500 USD with 8 decimals)
            1640000000,  # startedAt
            1640000000,  # updatedAt
            123456  # answeredInRound
        )

        price_data = await client._fetch_price_at_block(15000000, 1640000000)

        assert price_data is not None
        assert price_data['price_usd'] == 2500.0
        assert price_data['block_number'] == 15000000
        assert price_data['source'] == 'chainlink'
        assert price_data['confidence_score'] == 1.0

    async def test_fetch_price_invalid_price(self, mock_web3):
        """Test rejection of invalid prices"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Mock invalid price (negative)
        mock_web3.eth.contract().functions.latestRoundData().call.return_value = (
            123456,
            -100000000,  # negative price
            1640000000,
            1640000000,
            123456
        )

        price_data = await client._fetch_price_at_block(15000000, 1640000000)

        assert price_data is None

    async def test_find_block_by_timestamp(self, mock_web3):
        """Test binary search for block by timestamp"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Mock blocks
        def mock_get_block(block_id):
            if block_id == 'latest':
                return {'number': 18000000, 'timestamp': 1700000000}
            elif isinstance(block_id, int):
                # Simulate 12 second block time
                return {'number': block_id, 'timestamp': 1600000000 + (block_id * 12)}
            return {'number': 0, 'timestamp': 1600000000}

        mock_web3.eth.get_block = mock_get_block

        target_timestamp = 1600001200  # 100 blocks * 12 seconds
        block = await client.find_block_by_timestamp(target_timestamp)

        assert block is not None
        assert isinstance(block, int)

    async def test_get_eth_price_at_timestamp_cached(self, mock_web3):
        """Test that cached prices are returned"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Pre-populate cache
        cached_price = {
            'price_usd': 3000.0,
            'timestamp': 3600,  # Hour 1
            'block_number': 12345,
            'source': 'chainlink'
        }
        client.price_cache[3600] = cached_price

        # Request price for timestamp in cached hour
        price_data = await client.get_eth_price_at_timestamp(3700)

        assert price_data == cached_price
        # Should not have called the blockchain
        assert not mock_web3.eth.get_block.called

    async def test_backfill_historical_prices(self, mock_web3):
        """Test historical price backfill"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Mock find_block_by_timestamp as async
        async def mock_find_block(timestamp):
            return 15000000
        client.find_block_by_timestamp = mock_find_block

        # Mock price fetch
        mock_web3.eth.contract().functions.latestRoundData().call.return_value = (
            123456,
            200000000000,
            1640000000,
            1640000000,
            123456
        )

        start = 1640000000
        end = 1640003600  # 1 hour later
        prices = await client.backfill_historical_prices(start, end, interval_seconds=3600)

        assert len(prices) >= 1
        assert all('price_usd' in p for p in prices)
        assert all(p['source'] == 'chainlink' for p in prices)

    async def test_get_price_feed_health_healthy(self, mock_web3):
        """Test price feed health check when healthy"""
        client = ChainlinkPriceClient("http://localhost:8545")

        current_time = int(datetime.now().timestamp())

        # Mock recent update
        mock_web3.eth.contract().functions.latestRoundData().call.return_value = (
            123456,
            200000000000,
            current_time - 300,  # 5 minutes ago
            current_time - 300,
            123456
        )

        health = await client.get_price_feed_health()

        assert health['is_healthy'] is True
        assert health['latest_price'] == 2000.0
        assert health['staleness_seconds'] < 3600

    async def test_get_price_feed_health_stale(self, mock_web3):
        """Test price feed health check when stale"""
        client = ChainlinkPriceClient("http://localhost:8545")

        current_time = int(datetime.now().timestamp())

        # Mock stale update
        mock_web3.eth.contract().functions.latestRoundData().call.return_value = (
            123456,
            200000000000,
            current_time - 7200,  # 2 hours ago
            current_time - 7200,
            123456
        )

        health = await client.get_price_feed_health()

        assert health['is_healthy'] is False
        assert health['staleness_seconds'] >= 3600

    async def test_cache_management(self, mock_web3):
        """Test cache clearing and size"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Add some cache entries
        client.price_cache[3600] = {'price_usd': 2000}
        client.price_cache[7200] = {'price_usd': 2100}

        assert client.get_cache_size() == 2

        client.clear_cache()

        assert client.get_cache_size() == 0
        assert len(client.price_cache) == 0

    async def test_price_rounding_to_hour(self, mock_web3):
        """Test that timestamps are rounded to nearest hour"""
        client = ChainlinkPriceClient("http://localhost:8545")

        # Mock the fetch as async
        async def mock_find_block(timestamp):
            return 15000000
        client.find_block_by_timestamp = mock_find_block
        mock_web3.eth.contract().functions.latestRoundData().call.return_value = (
            123456, 200000000000, 1640000000, 1640000000, 123456
        )

        # Request price for mid-hour timestamp
        timestamp = 3600 + 1800  # 1.5 hours
        price_data = await client.get_eth_price_at_timestamp(timestamp)

        # Should be cached at rounded hour
        assert 3600 in client.price_cache
        assert 5400 not in client.price_cache  # Not at 1.5 hours