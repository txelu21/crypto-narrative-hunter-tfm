"""
Tests for Transaction Value Calculation
"""

import pytest
from services.prices.transaction_pricer import TransactionPricer, Transaction, EnrichedTransaction


@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    return {
        3600: {'price_usd': 2000.0, 'timestamp': 3600, 'source': 'chainlink'},
        7200: {'price_usd': 2100.0, 'timestamp': 7200, 'source': 'chainlink'},
        10800: {'price_usd': 2050.0, 'timestamp': 10800, 'source': 'coingecko'},
    }


@pytest.fixture
def sample_transactions():
    """Sample transactions for testing"""
    return [
        Transaction(
            tx_hash='0x123',
            timestamp=3600,
            block_number=15000000,
            eth_value_in=0,
            eth_value_out=1.0,
            gas_cost_eth=0.01
        ),
        Transaction(
            tx_hash='0x456',
            timestamp=7200,
            block_number=15000100,
            eth_value_in=2.0,
            eth_value_out=0,
            gas_cost_eth=0.005
        ),
    ]


class TestTransactionPricer:
    """Test suite for TransactionPricer"""

    def test_initialization(self, sample_price_data):
        """Test pricer initialization"""
        pricer = TransactionPricer(sample_price_data)

        assert pricer.price_data == sample_price_data
        assert len(pricer.price_data) == 3

    def test_get_price_exact_match(self, sample_price_data):
        """Test getting price with exact timestamp match"""
        pricer = TransactionPricer(sample_price_data)

        price = pricer.get_price_at_timestamp(3600)

        assert price is not None
        assert price['price_usd'] == 2000.0

    def test_get_price_rounded_to_hour(self, sample_price_data):
        """Test timestamp rounding to nearest hour"""
        pricer = TransactionPricer(sample_price_data)

        # Timestamp in middle of hour should round to hour
        price = pricer.get_price_at_timestamp(3800)

        assert price is not None
        assert price['price_usd'] == 2000.0  # Rounded to 3600

    def test_get_price_within_tolerance(self, sample_price_data):
        """Test getting price within 2-hour tolerance"""
        pricer = TransactionPricer(sample_price_data)

        # 1 hour after 7200
        price = pricer.get_price_at_timestamp(10800 + 3600)

        assert price is not None
        assert price['timestamp'] == 10800

    def test_get_price_no_match(self, sample_price_data):
        """Test returns None when no price within range"""
        pricer = TransactionPricer(sample_price_data)

        # Way outside available range
        price = pricer.get_price_at_timestamp(100000)

        assert price is None

    def test_calculate_transaction_values(self, sample_price_data, sample_transactions):
        """Test calculating values for multiple transactions"""
        pricer = TransactionPricer(sample_price_data)

        enriched = pricer.calculate_transaction_values(sample_transactions)

        assert len(enriched) == 2

        # First transaction (1 ETH out @ $2000)
        tx1 = enriched[0]
        assert tx1.tx_hash == '0x123'
        assert tx1.eth_value_out_usd == 2000.0
        assert tx1.gas_cost_usd == 20.0  # 0.01 ETH * $2000
        assert tx1.net_value_usd == 1980.0  # 2000 - 0 - 20

        # Second transaction (2 ETH in @ $2100)
        tx2 = enriched[1]
        assert tx2.tx_hash == '0x456'
        assert tx2.eth_value_in_usd == 4200.0  # 2 ETH * $2100
        assert tx2.gas_cost_usd == 10.5  # 0.005 ETH * $2100
        assert tx2.net_value_usd == -4210.5  # 0 - 4200 - 10.5

    def test_calculate_single_transaction_value(self, sample_price_data):
        """Test calculating value for single transaction"""
        pricer = TransactionPricer(sample_price_data)

        tx = Transaction(
            tx_hash='0x789',
            timestamp=3600,
            block_number=15000000,
            eth_value_in=1.0,
            eth_value_out=0.5,
            gas_cost_eth=0.01
        )

        enriched = pricer.calculate_single_transaction_value(tx)

        assert enriched is not None
        assert enriched.tx_hash == '0x789'
        assert enriched.eth_value_in_usd == 2000.0
        assert enriched.eth_value_out_usd == 1000.0
        assert enriched.net_value_usd == -1020.0  # 1000 - 2000 - 20

    def test_calculate_transaction_missing_price(self, sample_price_data):
        """Test transaction with missing price returns empty"""
        pricer = TransactionPricer(sample_price_data)

        tx = Transaction(
            tx_hash='0xabc',
            timestamp=100000,  # No price available
            block_number=15000000,
            eth_value_in=1.0,
            eth_value_out=0.5,
            gas_cost_eth=0.01
        )

        enriched = pricer.calculate_transaction_values([tx])

        assert len(enriched) == 0

    def test_reprice_transactions(self, sample_price_data, sample_transactions):
        """Test repricing transactions with updated prices"""
        pricer = TransactionPricer(sample_price_data)

        # Initial pricing
        enriched = pricer.calculate_transaction_values(sample_transactions)
        original_value = enriched[0].eth_value_out_usd

        # New price data with higher prices
        new_prices = {
            3600: {'price_usd': 3000.0, 'timestamp': 3600, 'source': 'chainlink'},
            7200: {'price_usd': 3100.0, 'timestamp': 7200, 'source': 'chainlink'},
        }

        # Reprice
        repriced = pricer.reprice_transactions(enriched, new_prices)

        assert len(repriced) == 2
        assert repriced[0].eth_value_out_usd == 3000.0
        assert repriced[0].eth_value_out_usd != original_value

    def test_get_value_summary(self, sample_price_data, sample_transactions):
        """Test calculating transaction value summary"""
        pricer = TransactionPricer(sample_price_data)

        enriched = pricer.calculate_transaction_values(sample_transactions)
        summary = pricer.get_value_summary(enriched)

        assert summary['total_transactions'] == 2
        assert summary['total_eth_out'] == 1.0
        assert summary['total_eth_in'] == 2.0
        assert summary['total_usd_out'] == 2000.0
        assert summary['total_usd_in'] == 4200.0
        # Net: 2000 - 4200 - (20 + 10.5) = -2230.5
        assert abs(summary['net_value_usd'] - (-2230.5)) < 0.01

    def test_get_value_summary_empty(self, sample_price_data):
        """Test summary with no transactions"""
        pricer = TransactionPricer(sample_price_data)

        summary = pricer.get_value_summary([])

        assert summary['total_transactions'] == 0
        assert summary['total_usd_in'] == 0
        assert summary['net_value_usd'] == 0

    def test_filter_by_value_range(self, sample_price_data, sample_transactions):
        """Test filtering transactions by USD value"""
        pricer = TransactionPricer(sample_price_data)

        enriched = pricer.calculate_transaction_values(sample_transactions)

        # Filter for positive values only
        positive = pricer.filter_by_value_range(enriched, min_usd=0)

        assert len(positive) == 1
        assert positive[0].tx_hash == '0x123'  # Only positive net value

    def test_filter_by_value_range_max(self, sample_price_data, sample_transactions):
        """Test filtering with max value"""
        pricer = TransactionPricer(sample_price_data)

        enriched = pricer.calculate_transaction_values(sample_transactions)

        # Filter for values under $1000
        small = pricer.filter_by_value_range(enriched, max_usd=1000)

        assert len(small) == 1
        assert small[0].tx_hash == '0x456'  # Negative value < 1000

    def test_filter_by_value_range_both(self, sample_price_data):
        """Test filtering with both min and max"""
        pricer = TransactionPricer(sample_price_data)

        txs = [
            Transaction('0x1', 3600, 15000000, 0, 0.5, 0.001),  # ~$980
            Transaction('0x2', 3600, 15000000, 0, 1.5, 0.001),  # ~$2998
            Transaction('0x3', 3600, 15000000, 0, 2.5, 0.001),  # ~$4998
        ]

        enriched = pricer.calculate_transaction_values(txs)

        # Filter for $1000-$3000 range
        mid_range = pricer.filter_by_value_range(enriched, min_usd=1000, max_usd=3000)

        assert len(mid_range) == 1
        assert mid_range[0].tx_hash == '0x2'