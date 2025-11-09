"""Tests for BlockTimingClient."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch
import pytz

from services.balances.block_timing import (
    BlockTimingClient,
    BlockTimingError,
    AVERAGE_BLOCK_TIME,
)


@pytest.fixture
def block_timing_client():
    """Create a BlockTimingClient instance for testing."""
    return BlockTimingClient(api_key="test_key")


@pytest.fixture
def mock_block_data():
    """Mock block data."""
    return {
        "number": "0xbc614e",  # 12345678
        "hash": "0xabc123",
        "timestamp": "0x61bca856",  # 1639753814 in hex
    }


class TestBlockTimingClient:
    """Test suite for BlockTimingClient."""

    def test_init(self):
        """Test initialization."""
        client = BlockTimingClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert "test_key" in client.base_url

    @pytest.mark.asyncio
    async def test_get_block_success(self, block_timing_client, mock_block_data):
        """Test successful block retrieval."""
        block_timing_client._rpc_call = AsyncMock(return_value=mock_block_data)

        block = await block_timing_client.get_block(12345678)

        assert block["number"] == 12345678
        assert block["timestamp"] == 1639753814
        assert block["hash"] == "0xabc123"

    @pytest.mark.asyncio
    async def test_get_block_caching(self, block_timing_client, mock_block_data):
        """Test that block data is cached."""
        # Clear cache first
        block_timing_client.clear_cache()

        block_timing_client._rpc_call = AsyncMock(return_value=mock_block_data)

        # First call
        block1 = await block_timing_client.get_block(12345678)

        # Second call should use cache
        block2 = await block_timing_client.get_block(12345678)

        assert block1["timestamp"] == block2["timestamp"]
        # RPC should only be called once due to caching
        assert block_timing_client._rpc_call.call_count == 1

    @pytest.mark.asyncio
    async def test_get_block_not_found(self, block_timing_client):
        """Test block not found error."""
        block_timing_client._rpc_call = AsyncMock(return_value=None)

        with pytest.raises(BlockTimingError, match="not found"):
            await block_timing_client.get_block(99999999)

    @pytest.mark.asyncio
    async def test_get_latest_block(self, block_timing_client, mock_block_data):
        """Test latest block retrieval."""
        block_timing_client._rpc_call = AsyncMock(return_value=mock_block_data)

        block = await block_timing_client.get_latest_block()

        assert block["number"] == 12345678
        assert "timestamp" in block
        assert "hash" in block

        # Verify correct RPC method was called
        call_args = block_timing_client._rpc_call.call_args
        assert call_args[0][0] == "eth_getBlockByNumber"
        assert call_args[0][1] == ["latest", False]

    @pytest.mark.asyncio
    async def test_get_latest_block_failure(self, block_timing_client):
        """Test latest block retrieval failure."""
        block_timing_client._rpc_call = AsyncMock(return_value=None)

        with pytest.raises(BlockTimingError, match="Failed to fetch latest block"):
            await block_timing_client.get_latest_block()

    @pytest.mark.asyncio
    async def test_get_block_by_timestamp_exact_match(self, block_timing_client):
        """Test finding block with exact timestamp match."""
        target_timestamp = 1639753814

        # Mock get_latest_block
        latest_block = {
            "number": 12350000,
            "timestamp": 1639800000,
        }
        block_timing_client.get_latest_block = AsyncMock(return_value=latest_block)

        # Mock get_block to return exact match on first try
        target_block = {
            "number": 12345678,
            "timestamp": target_timestamp,
        }
        block_timing_client.get_block = AsyncMock(return_value=target_block)

        block = await block_timing_client.get_block_by_timestamp(target_timestamp)

        assert block["timestamp"] == target_timestamp

    @pytest.mark.asyncio
    async def test_get_block_by_timestamp_close_match(self, block_timing_client):
        """Test finding block with close timestamp match."""
        target_timestamp = 1639753814

        # Mock get_latest_block
        latest_block = {
            "number": 12350000,
            "timestamp": 1639800000,
        }
        block_timing_client.get_latest_block = AsyncMock(return_value=latest_block)

        # Mock get_block to return blocks with varying timestamps
        def mock_get_block(block_num):
            # Simulate blocks getting closer to target
            if block_num <= 12345678:
                return {
                    "number": block_num,
                    "timestamp": target_timestamp - 30,
                }
            else:
                return {
                    "number": block_num,
                    "timestamp": target_timestamp + 30,
                }

        block_timing_client.get_block = AsyncMock(side_effect=mock_get_block)

        block = await block_timing_client.get_block_by_timestamp(target_timestamp)

        # Should return a block close to target
        assert abs(block["timestamp"] - target_timestamp) <= 60

    @pytest.mark.asyncio
    async def test_get_block_by_timestamp_future_error(self, block_timing_client):
        """Test error when target timestamp is in the future."""
        future_timestamp = int(datetime.now(timezone.utc).timestamp()) + 86400

        latest_block = {
            "number": 12345678,
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
        }
        block_timing_client.get_latest_block = AsyncMock(return_value=latest_block)

        with pytest.raises(BlockTimingError, match="in the future"):
            await block_timing_client.get_block_by_timestamp(future_timestamp)

    @pytest.mark.asyncio
    async def test_get_end_of_day_block(self, block_timing_client):
        """Test getting end of day block."""
        target_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        # Mock get_block_by_timestamp
        expected_timestamp = int(
            datetime(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc).timestamp()
        )
        expected_block = {
            "number": 12345678,
            "timestamp": expected_timestamp,
        }
        block_timing_client.get_block_by_timestamp = AsyncMock(
            return_value=expected_block
        )

        block = await block_timing_client.get_end_of_day_block(target_date)

        assert block["number"] == 12345678
        # Verify correct timestamp was requested
        call_args = block_timing_client.get_block_by_timestamp.call_args
        assert call_args[0][0] == expected_timestamp

    @pytest.mark.asyncio
    async def test_get_end_of_day_block_custom_time(self, block_timing_client):
        """Test getting end of day block with custom time."""
        target_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        expected_timestamp = int(
            datetime(2024, 1, 15, 20, 30, 0, tzinfo=timezone.utc).timestamp()
        )
        expected_block = {
            "number": 12345678,
            "timestamp": expected_timestamp,
        }
        block_timing_client.get_block_by_timestamp = AsyncMock(
            return_value=expected_block
        )

        block = await block_timing_client.get_end_of_day_block(
            target_date,
            hour=20,
            minute=30,
            second=0,
        )

        # Verify correct timestamp was requested
        call_args = block_timing_client.get_block_by_timestamp.call_args
        assert call_args[0][0] == expected_timestamp

    @pytest.mark.asyncio
    async def test_get_end_of_day_block_with_timezone(self, block_timing_client):
        """Test getting end of day block with non-UTC timezone."""
        target_date = datetime(2024, 1, 15)
        timezone_str = "America/New_York"

        # Expected timestamp should be in EST/EDT
        tz = pytz.timezone(timezone_str)
        expected_datetime = tz.localize(datetime(2024, 1, 15, 23, 59, 59))
        expected_timestamp = int(expected_datetime.timestamp())

        expected_block = {
            "number": 12345678,
            "timestamp": expected_timestamp,
        }
        block_timing_client.get_block_by_timestamp = AsyncMock(
            return_value=expected_block
        )

        block = await block_timing_client.get_end_of_day_block(
            target_date,
            timezone_str=timezone_str,
        )

        # Verify correct timestamp was requested (should be different from UTC)
        call_args = block_timing_client.get_block_by_timestamp.call_args
        assert call_args[0][0] == expected_timestamp

    @pytest.mark.asyncio
    async def test_get_start_of_day_block(self, block_timing_client):
        """Test getting start of day block."""
        target_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        expected_timestamp = int(
            datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        expected_block = {
            "number": 12345678,
            "timestamp": expected_timestamp,
        }
        block_timing_client.get_block_by_timestamp = AsyncMock(
            return_value=expected_block
        )

        block = await block_timing_client.get_start_of_day_block(target_date)

        assert block["number"] == 12345678
        # Verify correct timestamp was requested
        call_args = block_timing_client.get_block_by_timestamp.call_args
        assert call_args[0][0] == expected_timestamp

    @pytest.mark.asyncio
    async def test_get_daily_blocks(self, block_timing_client):
        """Test getting blocks for multiple days."""
        start_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 17, tzinfo=timezone.utc)

        # Mock get_end_of_day_block to return different blocks for each day
        call_count = [0]

        async def mock_get_eod_block(date, timezone_str="UTC"):
            call_count[0] += 1
            return {
                "number": 12345678 + call_count[0],
                "timestamp": int(date.timestamp()),
            }

        block_timing_client.get_end_of_day_block = AsyncMock(
            side_effect=mock_get_eod_block
        )

        daily_blocks = await block_timing_client.get_daily_blocks(
            start_date,
            end_date,
        )

        # Should have 3 days worth of blocks
        assert len(daily_blocks) == 3
        assert "2024-01-15" in daily_blocks
        assert "2024-01-16" in daily_blocks
        assert "2024-01-17" in daily_blocks

        # Verify all blocks are present
        assert daily_blocks["2024-01-15"]["number"] == 12345679
        assert daily_blocks["2024-01-16"]["number"] == 12345680
        assert daily_blocks["2024-01-17"]["number"] == 12345681

    @pytest.mark.asyncio
    async def test_get_daily_blocks_start_of_day(self, block_timing_client):
        """Test getting start-of-day blocks for multiple days."""
        start_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 16, tzinfo=timezone.utc)

        async def mock_get_sod_block(date, timezone_str="UTC"):
            return {
                "number": 12345678,
                "timestamp": int(date.timestamp()),
            }

        block_timing_client.get_start_of_day_block = AsyncMock(
            side_effect=mock_get_sod_block
        )

        daily_blocks = await block_timing_client.get_daily_blocks(
            start_date,
            end_date,
            snapshot_time="start_of_day",
        )

        assert len(daily_blocks) == 2
        # Verify start_of_day_block was called instead of end_of_day_block
        assert block_timing_client.get_start_of_day_block.call_count == 2

    @pytest.mark.asyncio
    async def test_get_daily_blocks_error_handling(self, block_timing_client):
        """Test error handling in daily blocks retrieval."""
        start_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 16, tzinfo=timezone.utc)

        # Mock to raise error on second day
        call_count = [0]

        async def mock_get_eod_block(date, timezone_str="UTC"):
            call_count[0] += 1
            if call_count[0] == 2:
                raise BlockTimingError("Block resolution failed")
            return {
                "number": 12345678,
                "timestamp": int(date.timestamp()),
            }

        block_timing_client.get_end_of_day_block = AsyncMock(
            side_effect=mock_get_eod_block
        )

        with pytest.raises(BlockTimingError, match="Block resolution failed"):
            await block_timing_client.get_daily_blocks(start_date, end_date)

    def test_clear_cache(self, block_timing_client):
        """Test cache clearing."""
        # Cache is module-level, so we need to import it
        from services.balances.block_timing import _block_cache

        # Add some data to cache
        _block_cache[12345678] = 1639753814
        _block_cache[12345679] = 1639753826

        assert len(_block_cache) == 2

        block_timing_client.clear_cache()

        assert len(_block_cache) == 0

    def test_get_cache_size(self, block_timing_client):
        """Test getting cache size."""
        from services.balances.block_timing import _block_cache

        _block_cache.clear()
        assert block_timing_client.get_cache_size() == 0

        _block_cache[12345678] = 1639753814
        assert block_timing_client.get_cache_size() == 1

        _block_cache[12345679] = 1639753826
        assert block_timing_client.get_cache_size() == 2

        _block_cache.clear()  # Clean up