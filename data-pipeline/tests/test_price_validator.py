"""
Tests for Multi-Source Price Validation
"""

import pytest
from unittest.mock import Mock, AsyncMock
from services.prices.price_validator import PriceValidator


@pytest.fixture
def mock_clients():
    """Mock price source clients"""
    chainlink = Mock()
    coingecko = Mock()
    uniswap = Mock()

    return chainlink, coingecko, uniswap


@pytest.mark.asyncio
class TestPriceValidator:
    """Test suite for PriceValidator"""

    async def test_initialization(self, mock_clients):
        """Test validator initialization"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        assert validator.chainlink == chainlink
        assert validator.coingecko == coingecko
        assert validator.uniswap == uniswap

    async def test_validate_price_all_sources_agree(self, mock_clients):
        """Test validation when all sources agree"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        # Mock all sources returning similar prices
        validator._get_chainlink_price = AsyncMock(return_value=2000.0)
        validator._get_coingecko_price = AsyncMock(return_value=2010.0)
        validator._get_uniswap_price = AsyncMock(return_value=1995.0)

        result = await validator.validate_price_with_sources(1640000000)

        assert result['validation_passed'] is True
        assert result['consensus_price'] == 2000.0  # median
        assert len(result['valid_sources']) == 3
        assert result['num_sources'] == 3

    async def test_validate_price_one_outlier(self, mock_clients):
        """Test validation with one outlier source"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        # Mock one outlier
        validator._get_chainlink_price = AsyncMock(return_value=2000.0)
        validator._get_coingecko_price = AsyncMock(return_value=2010.0)
        validator._get_uniswap_price = AsyncMock(return_value=2500.0)  # outlier

        result = await validator.validate_price_with_sources(1640000000, tolerance=0.05)

        assert result['validation_passed'] is True
        assert result['consensus_price'] == 2010.0  # median
        assert len(result['valid_sources']) == 2  # chainlink and coingecko
        assert 'uniswap' not in result['valid_sources']

    async def test_validate_price_no_consensus(self, mock_clients):
        """Test validation failure when no consensus"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        # Mock widely divergent prices
        validator._get_chainlink_price = AsyncMock(return_value=2000.0)
        validator._get_coingecko_price = AsyncMock(return_value=2500.0)
        validator._get_uniswap_price = AsyncMock(return_value=3000.0)

        result = await validator.validate_price_with_sources(1640000000, tolerance=0.05)

        assert result['validation_passed'] is False
        assert result['consensus_price'] is None

    async def test_validate_price_single_source(self, mock_clients):
        """Test validation with single source available"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        # Only chainlink available
        validator._get_chainlink_price = AsyncMock(return_value=2000.0)
        validator._get_coingecko_price = AsyncMock(return_value=None)
        validator._get_uniswap_price = AsyncMock(return_value=None)

        result = await validator.validate_price_with_sources(1640000000)

        assert result['validation_passed'] is True
        assert result['consensus_price'] == 2000.0
        assert result['num_sources'] == 1

    async def test_validate_price_no_sources(self, mock_clients):
        """Test validation failure when no sources available"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        # No sources available
        validator._get_chainlink_price = AsyncMock(return_value=None)
        validator._get_coingecko_price = AsyncMock(return_value=None)
        validator._get_uniswap_price = AsyncMock(return_value=None)

        result = await validator.validate_price_with_sources(1640000000)

        assert result['validation_passed'] is False
        assert result['consensus_price'] is None
        assert 'error' in result

    async def test_get_price_with_fallback_chainlink(self, mock_clients):
        """Test fallback gets Chainlink first"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        validator._get_chainlink_price = AsyncMock(return_value=2000.0)
        validator._get_coingecko_price = AsyncMock(return_value=2010.0)

        price_data = await validator.get_price_with_fallback(1640000000)

        assert price_data is not None
        assert price_data['price_usd'] == 2000.0
        assert price_data['source'] == 'chainlink'
        assert price_data['confidence_score'] == 1.0

    async def test_get_price_with_fallback_coingecko(self, mock_clients):
        """Test fallback uses CoinGecko when Chainlink fails"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        validator._get_chainlink_price = AsyncMock(return_value=None)
        validator._get_coingecko_price = AsyncMock(return_value=2010.0)

        price_data = await validator.get_price_with_fallback(1640000000)

        assert price_data is not None
        assert price_data['price_usd'] == 2010.0
        assert price_data['source'] == 'coingecko'
        assert price_data['confidence_score'] == 0.9

    async def test_get_price_with_fallback_uniswap(self, mock_clients):
        """Test fallback uses Uniswap when Chainlink and CoinGecko fail"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        validator._get_chainlink_price = AsyncMock(return_value=None)
        validator._get_coingecko_price = AsyncMock(return_value=None)
        validator._get_uniswap_price = AsyncMock(return_value=1995.0)

        price_data = await validator.get_price_with_fallback(1640000000)

        assert price_data is not None
        assert price_data['price_usd'] == 1995.0
        assert price_data['source'] == 'uniswap_v3'
        assert price_data['confidence_score'] == 0.85

    async def test_get_price_with_fallback_all_fail(self, mock_clients):
        """Test fallback returns None when all sources fail"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        validator._get_chainlink_price = AsyncMock(return_value=None)
        validator._get_coingecko_price = AsyncMock(return_value=None)
        validator._get_uniswap_price = AsyncMock(return_value=None)

        price_data = await validator.get_price_with_fallback(1640000000)

        assert price_data is None

    async def test_interpolate_missing_price(self, mock_clients):
        """Test price interpolation"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        price_data = {
            1000: {'price_usd': 2000.0},
            2000: {'price_usd': 2100.0}
        }

        # Interpolate at midpoint
        interpolated = await validator.interpolate_missing_price(1500, price_data)

        assert interpolated is not None
        assert interpolated == 2050.0  # Linear interpolation

    async def test_interpolate_missing_no_before(self, mock_clients):
        """Test interpolation fails without before price"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        price_data = {
            2000: {'price_usd': 2100.0}
        }

        # Try to interpolate before available data
        interpolated = await validator.interpolate_missing_price(1000, price_data)

        assert interpolated is None

    async def test_detect_price_anomalies(self, mock_clients):
        """Test anomaly detection"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        price_data = {
            1000: {'price_usd': 2000.0, 'source': 'chainlink'},
            2000: {'price_usd': 2010.0, 'source': 'chainlink'},
            3000: {'price_usd': 2005.0, 'source': 'chainlink'},
            4000: {'price_usd': 2008.0, 'source': 'chainlink'},
            5000: {'price_usd': 2002.0, 'source': 'chainlink'},
            6000: {'price_usd': 10000.0, 'source': 'coingecko'},  # clear anomaly
        }

        anomalies = await validator.detect_price_anomalies(price_data, std_threshold=2.0)

        assert len(anomalies) >= 1
        assert any(a['timestamp'] == 6000 for a in anomalies)

    async def test_detect_price_anomalies_insufficient_data(self, mock_clients):
        """Test anomaly detection with insufficient data"""
        chainlink, coingecko, uniswap = mock_clients

        validator = PriceValidator(chainlink, coingecko, uniswap)

        price_data = {
            1000: {'price_usd': 2000.0}
        }

        anomalies = await validator.detect_price_anomalies(price_data)

        assert len(anomalies) == 0