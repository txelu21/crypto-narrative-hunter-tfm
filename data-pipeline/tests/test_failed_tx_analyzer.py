"""Tests for failed transaction analyzer."""

import pytest
from datetime import datetime
from decimal import Decimal
from services.transactions.failed_tx_analyzer import FailedTransactionAnalyzer, KNOWN_MEV_BOTS


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return FailedTransactionAnalyzer()


def test_analyze_successful_transaction(analyzer):
    """Test that successful transactions return None."""
    tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    receipt = {
        "transactionHash": tx_hash,
        "blockNumber": 15000000,
        "transactionIndex": 50,
        "gasUsed": 150000,
        "effectiveGasPrice": 50000000000,  # 50 gwei
        "status": 1,  # Success
        "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    }

    wallet = "0x1234567890123456789012345678901234567890"
    timestamp = datetime.now()

    result = analyzer.analyze_failed_transaction(tx_hash, receipt, wallet, timestamp)
    assert result is None


def test_analyze_failed_transaction_basic(analyzer):
    """Test basic failed transaction analysis."""
    tx_hash = "0xfailed1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    receipt = {
        "transactionHash": tx_hash,
        "blockNumber": 15000000,
        "transactionIndex": 50,
        "gasUsed": 50000,
        "effectiveGasPrice": 50000000000,  # 50 gwei
        "status": 0,  # Failed
        "to": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
    }

    wallet = "0x1234567890123456789012345678901234567890"
    timestamp = datetime.now()

    result = analyzer.analyze_failed_transaction(tx_hash, receipt, wallet, timestamp)

    assert result is not None
    assert result.tx_hash == tx_hash
    assert result.status == "failed"
    assert result.gas_used == 50000
    assert result.gas_price_gwei == Decimal("50.0")
    assert result.gas_cost_eth == Decimal("0.0025")  # 50000 * 50 gwei


def test_categorize_failure_reason_slippage(analyzer):
    """Test categorization of slippage failures."""
    assert analyzer.categorize_failure_reason("UniswapV2: INSUFFICIENT_OUTPUT_AMOUNT") == "slippage"
    assert analyzer.categorize_failure_reason("Slippage exceeded") == "slippage"


def test_categorize_failure_reason_deadline(analyzer):
    """Test categorization of deadline failures."""
    assert analyzer.categorize_failure_reason("Transaction deadline expired") == "deadline_expired"
    assert analyzer.categorize_failure_reason("EXPIRED") == "deadline_expired"


def test_categorize_failure_reason_balance(analyzer):
    """Test categorization of insufficient balance failures."""
    assert analyzer.categorize_failure_reason("Insufficient balance") == "insufficient_balance"
    assert analyzer.categorize_failure_reason("ERC20: transfer amount exceeds balance") == "insufficient_balance"


def test_categorize_failure_reason_allowance(analyzer):
    """Test categorization of allowance failures."""
    assert analyzer.categorize_failure_reason("ERC20: insufficient allowance") == "insufficient_allowance"
    assert analyzer.categorize_failure_reason("Approval required") == "insufficient_allowance"


def test_categorize_failure_reason_liquidity(analyzer):
    """Test categorization of liquidity failures."""
    assert analyzer.categorize_failure_reason("Insufficient liquidity") == "insufficient_liquidity"


def test_categorize_failure_reason_unknown(analyzer):
    """Test categorization of unknown failures."""
    assert analyzer.categorize_failure_reason(None) == "unknown"
    assert analyzer.categorize_failure_reason("Some random error") == "other"


def test_is_sandwich_attack_detected(analyzer):
    """Test sandwich attack detection."""
    # Mock block transactions
    mev_bot = list(KNOWN_MEV_BOTS)[0]  # Use first known MEV bot

    block_txs = [
        {"from": mev_bot, "to": "0xPool", "gasPrice": 100000000000, "transactionIndex": 49},  # Front-run
        {"from": "0xVictim", "to": "0xPool", "gasPrice": 50000000000, "transactionIndex": 50},  # Victim
        {"from": mev_bot, "to": "0xPool", "gasPrice": 100000000000, "transactionIndex": 51},  # Back-run
    ]

    receipt = {
        "blockNumber": 15000000,
        "transactionIndex": 50,
        "effectiveGasPrice": 50000000000,
        "to": "0xPool"
    }

    assert analyzer._is_sandwich_attack(50, block_txs, receipt) is True


def test_is_sandwich_attack_not_detected(analyzer):
    """Test sandwich attack not detected for normal transaction."""
    block_txs = [
        {"from": "0xUser1", "to": "0xPool", "gasPrice": 50000000000, "transactionIndex": 49},
        {"from": "0xUser2", "to": "0xPool", "gasPrice": 50000000000, "transactionIndex": 50},
        {"from": "0xUser3", "to": "0xPool", "gasPrice": 50000000000, "transactionIndex": 51},
    ]

    receipt = {
        "blockNumber": 15000000,
        "transactionIndex": 50,
        "effectiveGasPrice": 50000000000,
        "to": "0xPool"
    }

    assert analyzer._is_sandwich_attack(50, block_txs, receipt) is False


def test_is_front_run_detected(analyzer):
    """Test front-running detection."""
    block_txs = [
        {"from": "0xBot", "to": "0xDEX", "gasPrice": 100000000000, "transactionIndex": 48},
        {"from": "0xBot", "to": "0xDEX", "gasPrice": 100000000000, "transactionIndex": 49},  # Front-runner
        {"from": "0xVictim", "to": "0xDEX", "gasPrice": 50000000000, "transactionIndex": 50},  # Victim
    ]

    receipt = {
        "blockNumber": 15000000,
        "transactionIndex": 50,
        "effectiveGasPrice": 50000000000,
        "to": "0xDEX"
    }

    assert analyzer._is_front_run(50, block_txs, receipt) is True


def test_is_front_run_not_detected_first_tx(analyzer):
    """Test front-running not detected for first transaction in block."""
    block_txs = [
        {"from": "0xUser", "to": "0xDEX", "gasPrice": 50000000000, "transactionIndex": 0},
    ]

    receipt = {
        "blockNumber": 15000000,
        "transactionIndex": 0,
        "effectiveGasPrice": 50000000000,
        "to": "0xDEX"
    }

    assert analyzer._is_front_run(0, block_txs, receipt) is False


def test_is_back_run_detected(analyzer):
    """Test back-running detection."""
    block_txs = [
        {"from": "0xVictim", "to": "0xDEX", "gasPrice": 50000000000, "transactionIndex": 50},  # Victim
        {"from": "0xBot", "to": "0xDEX", "gasPrice": 100000000000, "transactionIndex": 51},  # Back-runner
    ]

    receipt = {
        "blockNumber": 15000000,
        "transactionIndex": 50,
        "effectiveGasPrice": 50000000000,
        "to": "0xDEX"
    }

    assert analyzer._is_back_run(50, block_txs, receipt) is True


def test_calculate_mev_statistics_empty(analyzer):
    """Test MEV statistics calculation with empty list."""
    stats = analyzer.calculate_mev_statistics([])

    assert stats["total_failed"] == 0
    assert stats["mev_affected_count"] == 0
    assert stats["total_gas_wasted_eth"] == 0.0


def test_calculate_mev_statistics_with_mev(analyzer):
    """Test MEV statistics calculation with MEV-affected transactions."""
    from services.transactions.models import FailedTransaction

    failed_txs = [
        FailedTransaction(
            tx_hash="0x1",
            block_number=15000000,
            timestamp=datetime.now(),
            wallet_address="0x1234567890123456789012345678901234567890",
            status="failed",
            gas_used=50000,
            gas_price_gwei=Decimal("50.0"),
            gas_cost_eth=Decimal("0.0025"),
            mev_type="sandwich",
            mev_damage_eth=Decimal("0.1")
        ),
        FailedTransaction(
            tx_hash="0x2",
            block_number=15000001,
            timestamp=datetime.now(),
            wallet_address="0x1234567890123456789012345678901234567890",
            status="failed",
            gas_used=40000,
            gas_price_gwei=Decimal("60.0"),
            gas_cost_eth=Decimal("0.0024"),
            mev_type="front_run",
            mev_damage_eth=Decimal("0.05")
        ),
        FailedTransaction(
            tx_hash="0x3",
            block_number=15000002,
            timestamp=datetime.now(),
            wallet_address="0x1234567890123456789012345678901234567890",
            status="failed",
            gas_used=30000,
            gas_price_gwei=Decimal("55.0"),
            gas_cost_eth=Decimal("0.00165")
        )
    ]

    stats = analyzer.calculate_mev_statistics(failed_txs)

    assert stats["total_failed"] == 3
    assert stats["mev_affected_count"] == 2
    assert stats["mev_affected_percentage"] == pytest.approx(66.67, rel=0.1)
    assert stats["mev_by_type"]["sandwich"] == 1
    assert stats["mev_by_type"]["front_run"] == 1
    assert stats["total_mev_damage_eth"] == pytest.approx(0.15)
    assert stats["total_gas_wasted_eth"] == pytest.approx(0.00655)