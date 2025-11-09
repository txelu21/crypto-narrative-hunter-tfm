"""Tests for cross-validation framework."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from services.validation.cross_validator import CrossValidator


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    pool = AsyncMock()
    return pool


@pytest.fixture
def cross_validator(mock_db_pool):
    """Create CrossValidator instance."""
    return CrossValidator(mock_db_pool)


@pytest.mark.asyncio
async def test_validate_transaction_balance_consistency_success(cross_validator, mock_db_pool):
    """Test successful transaction-balance consistency validation."""
    wallet = "0x1234567890123456789012345678901234567890"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    # Mock database responses
    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Mock transactions
    mock_conn.fetch.side_effect = [
        [  # transactions
            {
                "tx_hash": "0xabc",
                "block_number": 1000,
                "block_timestamp": datetime(2024, 1, 15),
                "token_in_address": "0xtoken1",
                "token_out_address": "0xtoken2",
                "amount_in": Decimal("100"),
                "amount_out": Decimal("200"),
                "gas_used": 21000,
                "gas_price_gwei": Decimal("50"),
                "eth_value_total_usd": Decimal("100"),
            }
        ],
        [  # balance snapshots
            {
                "snapshot_date": datetime(2024, 1, 1),
                "token_address": "0xtoken1",
                "balance": Decimal("1000"),
                "eth_value": Decimal("10"),
            },
            {
                "snapshot_date": "0xtoken2",
                "balance": Decimal("0"),
                "eth_value": Decimal("0"),
            },
        ],
    ]

    result = await cross_validator.validate_transaction_balance_consistency(
        wallet, start_date, end_date
    )

    assert result["wallet_address"] == wallet
    assert "overall_consistency_score" in result
    assert result["validation_status"] in ["pass", "fail"]


@pytest.mark.asyncio
async def test_validate_transaction_balance_consistency_insufficient_data(
    cross_validator, mock_db_pool
):
    """Test validation with insufficient data."""
    wallet = "0x1234567890123456789012345678901234567890"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Return empty results
    mock_conn.fetch.return_value = []

    result = await cross_validator.validate_transaction_balance_consistency(
        wallet, start_date, end_date
    )

    assert result["validation_status"] == "insufficient_data"
    assert result["snapshot_count"] < 2


@pytest.mark.asyncio
async def test_validate_gas_costs(cross_validator, mock_db_pool):
    """Test gas cost validation."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Mock transactions with valid gas costs
    mock_conn.fetch.return_value = [
        {
            "tx_hash": "0xabc",
            "gas_used": 150000,
            "gas_price_gwei": Decimal("50"),
            "eth_value_total_usd": Decimal("100"),
            "block_timestamp": datetime(2024, 1, 15),
        },
        {
            "tx_hash": "0xdef",
            "gas_used": 200000,
            "gas_price_gwei": Decimal("75"),
            "eth_value_total_usd": Decimal("150"),
            "block_timestamp": datetime(2024, 1, 16),
        },
    ]

    result = await cross_validator.validate_gas_costs(start_date, end_date, sample_size=100)

    assert "transactions_validated" in result
    assert "anomalies_detected" in result
    assert "validation_score" in result
    assert result["validation_status"] in ["pass", "fail"]


@pytest.mark.asyncio
async def test_validate_transaction_completeness(cross_validator, mock_db_pool):
    """Test transaction completeness validation."""
    wallet = "0x1234567890123456789012345678901234567890"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Mock responses
    mock_conn.fetchrow.return_value = {"count": 100}
    mock_conn.fetch.return_value = []  # No gaps

    result = await cross_validator.validate_transaction_completeness(
        wallet, start_date, end_date
    )

    assert result["wallet_address"] == wallet
    assert "transaction_count" in result
    assert "completeness_score" in result
    assert result["validation_status"] in ["pass", "fail"]


def test_calculate_balance_changes_from_transactions(cross_validator):
    """Test balance change calculation from transactions."""
    transactions = [
        {
            "token_in_address": "0xtoken1",
            "amount_in": Decimal("100"),
            "token_out_address": "0xtoken2",
            "amount_out": Decimal("200"),
            "gas_used": 21000,
            "gas_price_gwei": Decimal("50"),
        }
    ]

    changes = cross_validator._calculate_balance_changes_from_transactions(
        transactions
    )

    assert "0xtoken1" in changes
    assert changes["0xtoken1"] == Decimal("-100")
    assert "0xtoken2" in changes
    assert changes["0xtoken2"] == Decimal("200")


def test_compare_balance_changes_matching(cross_validator):
    """Test balance change comparison with matching values."""
    expected = {"0xtoken1": Decimal("100"), "0xtoken2": Decimal("200")}
    actual = {"0xtoken1": Decimal("100"), "0xtoken2": Decimal("200")}

    result = cross_validator._compare_balance_changes(expected, actual, 0.01)

    assert result["score"] == 1.0
    assert result["matching_tokens"] == 2
    assert len(result["discrepancies"]) == 0


def test_compare_balance_changes_with_discrepancy(cross_validator):
    """Test balance change comparison with discrepancies."""
    expected = {"0xtoken1": Decimal("100"), "0xtoken2": Decimal("200")}
    actual = {"0xtoken1": Decimal("90"), "0xtoken2": Decimal("200")}

    result = cross_validator._compare_balance_changes(expected, actual, 0.01)

    assert result["score"] < 1.0
    assert len(result["discrepancies"]) > 0