"""Tests for statistical validator."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock

from services.validation.statistical_validator import StatisticalValidator


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return AsyncMock()


@pytest.fixture
def statistical_validator(mock_db_pool):
    """Create StatisticalValidator instance."""
    return StatisticalValidator(mock_db_pool)


@pytest.mark.asyncio
async def test_detect_multidimensional_outliers(statistical_validator, mock_db_pool):
    """Test multidimensional outlier detection."""
    # Mock dataset
    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Create sample data with outliers
    data = {
        "amount_in": [100.0] * 95 + [10000.0] * 5,  # 5 outliers
        "amount_out": [200.0] * 95 + [20000.0] * 5,
        "gas_used": [150000] * 100,
    }
    mock_conn.fetch.return_value = [
        {k: v[i] for k, v in data.items()} for i in range(100)
    ]

    result = await statistical_validator.detect_multidimensional_outliers(
        "transactions", ["amount_in", "amount_out", "gas_used"]
    )

    assert result["dataset_name"] == "transactions"
    assert result["total_records"] == 100
    assert "outliers_detected" in result
    assert "outlier_rate" in result


@pytest.mark.asyncio
async def test_validate_trading_patterns_valid(statistical_validator, mock_db_pool):
    """Test trading pattern validation with valid patterns."""
    wallet = "0x1234567890123456789012345678901234567890"

    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Mock valid trading data
    mock_conn.fetch.return_value = [
        {
            "tx_hash": f"0xabc{i}",
            "block_timestamp": datetime(2024, 1, i + 1),
            "token_in_address": f"0xtoken{i % 5}",
            "token_out_address": f"0xtoken{(i + 1) % 5}",
            "amount_in": 100.0 + i,
            "amount_out": 200.0 + i,
            "gas_used": 150000,
            "gas_price_gwei": 50.0,
            "eth_value_total_usd": 1000.0,
        }
        for i in range(20)
    ]

    result = await statistical_validator.validate_trading_patterns(wallet)

    assert result["wallet_address"] == wallet
    assert result["transaction_count"] == 20
    assert "pattern_metrics" in result
    assert "validation_results" in result
    assert result["validation_status"] in ["pass", "fail"]


@pytest.mark.asyncio
async def test_validate_trading_patterns_insufficient_data(
    statistical_validator, mock_db_pool
):
    """Test trading pattern validation with insufficient data."""
    wallet = "0x1234567890123456789012345678901234567890"

    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_conn.fetch.return_value = []

    result = await statistical_validator.validate_trading_patterns(wallet)

    assert result["transaction_count"] == 0
    assert result["validation_status"] == "insufficient_data"


@pytest.mark.asyncio
async def test_detect_time_series_anomalies(statistical_validator):
    """Test time series anomaly detection."""
    # Create time series with anomalies
    time_series_data = [
        {"timestamp": datetime(2024, 1, i), "value": 100.0 + np.random.randn()}
        for i in range(1, 30)
    ]
    # Add anomalies
    time_series_data.append({"timestamp": datetime(2024, 1, 30), "value": 500.0})

    result = await statistical_validator.detect_time_series_anomalies(
        "eth_price", time_series_data, window_size=7
    )

    assert result["metric_name"] == "eth_price"
    assert "anomalies_detected" in result
    assert "anomaly_rate" in result
    assert result["validation_status"] in ["pass", "warning"]


@pytest.mark.asyncio
async def test_validate_correlation_patterns(statistical_validator, mock_db_pool):
    """Test correlation pattern validation."""
    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Mock correlated data
    data = {
        "gas_used": list(range(100, 200)),
        "gas_price_gwei": [x * 0.5 for x in range(100, 200)],
    }
    mock_conn.fetch.return_value = [
        {k: v[i] for k, v in data.items()} for i in range(100)
    ]

    expected_correlations = {"gas_used": {"gas_price_gwei": 0.8}}

    result = await statistical_validator.validate_correlation_patterns(
        ["gas_used", "gas_price_gwei"], expected_correlations
    )

    assert "features" in result
    assert "correlations_validated" in result
    assert result["validation_status"] in ["pass", "fail"]


@pytest.mark.asyncio
async def test_detect_behavioral_anomalies(statistical_validator, mock_db_pool):
    """Test behavioral anomaly detection."""
    wallet = "0x1234567890123456789012345678901234567890"

    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Mock normal trading behavior
    mock_conn.fetch.return_value = [
        {
            "tx_hash": f"0xabc{i}",
            "block_timestamp": datetime(2024, 1, 1) + pd.Timedelta(days=i),
            "token_in_address": f"0xtoken{i % 5}",
            "token_out_address": f"0xtoken{(i + 1) % 5}",
            "amount_in": 100.0,
            "amount_out": 200.0,
            "gas_used": 150000,
            "gas_price_gwei": 50.0,
            "eth_value_total_usd": 1000.0,
        }
        for i in range(20)
    ]

    result = await statistical_validator.detect_behavioral_anomalies(wallet)

    assert result["wallet_address"] == wallet
    assert "anomalies_detected" in result
    assert "anomalies" in result
    assert result["validation_status"] in ["pass", "warning"]