"""
Tests for Balance Pricer Module
"""

import pytest
from services.prices.balance_pricer import (
    BalancePricer,
    BalanceSnapshot,
    ValuedBalance,
    PortfolioSnapshot
)


class TestBalancePricer:
    """Test suite for BalancePricer"""

    @pytest.fixture
    def sample_eth_prices(self):
        """Sample ETH price data"""
        return {
            # Hourly prices for testing
            1609459200: {'price_usd': 730.0, 'timestamp': 1609459200, 'source': 'chainlink'},  # 2021-01-01 00:00
            1609462800: {'price_usd': 735.0, 'timestamp': 1609462800, 'source': 'chainlink'},  # 2021-01-01 01:00
            1609466400: {'price_usd': 740.0, 'timestamp': 1609466400, 'source': 'chainlink'},  # 2021-01-01 02:00
            1609470000: {'price_usd': 745.0, 'timestamp': 1609470000, 'source': 'chainlink'},  # 2021-01-01 03:00
        }

    @pytest.fixture
    def sample_token_prices(self):
        """Sample token price data"""
        return {
            '0xTokenA': {
                1609459200: {'price_usd': 10.0, 'timestamp': 1609459200, 'source': 'coingecko'},
                1609462800: {'price_usd': 10.5, 'timestamp': 1609462800, 'source': 'coingecko'},
            },
            '0xTokenB': {
                1609459200: {'price_usd': 0.5, 'timestamp': 1609459200, 'source': 'coingecko'},
                1609462800: {'price_usd': 0.55, 'timestamp': 1609462800, 'source': 'coingecko'},
            }
        }

    @pytest.fixture
    def pricer(self, sample_eth_prices, sample_token_prices):
        """Initialize balance pricer with sample data"""
        return BalancePricer(sample_eth_prices, sample_token_prices)

    @pytest.fixture
    def sample_balance_snapshot(self):
        """Sample balance snapshot"""
        return BalanceSnapshot(
            wallet_address='0xWallet1',
            token_address='0xTokenA',
            token_symbol='TOKA',
            balance=100.0,
            timestamp=1609459200,
            block_number=1000
        )

    @pytest.fixture
    def sample_eth_balance(self):
        """Sample ETH balance snapshot"""
        return BalanceSnapshot(
            wallet_address='0xWallet1',
            token_address='0x0',
            token_symbol='ETH',
            balance=10.0,
            timestamp=1609459200,
            block_number=1000
        )

    def test_price_balance_snapshot(self, pricer, sample_balance_snapshot):
        """Test pricing a single balance snapshot"""
        valued = pricer.price_balance_snapshot(sample_balance_snapshot)

        assert valued is not None
        assert isinstance(valued, ValuedBalance)
        assert valued.token_price_usd == 10.0
        assert valued.balance_value_usd == 1000.0  # 100 * 10
        assert valued.price_source == 'coingecko'
        assert valued.wallet_address == '0xWallet1'
        assert valued.balance == 100.0

    def test_price_eth_balance(self, pricer, sample_eth_balance):
        """Test pricing ETH balance"""
        valued = pricer.price_balance_snapshot(sample_eth_balance)

        assert valued is not None
        assert valued.token_price_usd == 730.0
        assert valued.balance_value_usd == 7300.0  # 10 * 730
        assert valued.price_source == 'chainlink'

    def test_price_balance_with_missing_price(self, pricer):
        """Test pricing balance with missing price data"""
        balance = BalanceSnapshot(
            wallet_address='0xWallet1',
            token_address='0xUnknownToken',
            token_symbol='UNK',
            balance=100.0,
            timestamp=1609459200,
            block_number=1000
        )

        valued = pricer.price_balance_snapshot(balance)
        assert valued is None

    def test_calculate_portfolio_value(self, pricer, sample_eth_prices, sample_token_prices):
        """Test portfolio value calculation"""
        balances = [
            BalanceSnapshot(
                wallet_address='0xWallet1',
                token_address='0xTokenA',
                token_symbol='TOKA',
                balance=100.0,
                timestamp=1609459200,
                block_number=1000
            ),
            BalanceSnapshot(
                wallet_address='0xWallet1',
                token_address='0xTokenB',
                token_symbol='TOKB',
                balance=500.0,
                timestamp=1609459200,
                block_number=1000
            ),
            BalanceSnapshot(
                wallet_address='0xWallet1',
                token_address='0x0',
                token_symbol='ETH',
                balance=5.0,
                timestamp=1609459200,
                block_number=1000
            )
        ]

        portfolio = pricer.calculate_portfolio_value(balances)

        assert portfolio is not None
        assert isinstance(portfolio, PortfolioSnapshot)
        assert portfolio.wallet_address == '0xWallet1'
        assert len(portfolio.balances) == 3
        # Expected: (100 * 10) + (500 * 0.5) + (5 * 730) = 1000 + 250 + 3650 = 4900
        assert portfolio.total_value_usd == 4900.0
        assert portfolio.eth_price == 730.0

    def test_calculate_time_weighted_value(self, pricer, sample_eth_prices):
        """Test time-weighted value calculation"""
        snapshots = [
            PortfolioSnapshot(
                wallet_address='0xWallet1',
                timestamp=1609459200,
                balances=[],
                total_value_usd=1000.0,
                eth_price=730.0
            ),
            PortfolioSnapshot(
                wallet_address='0xWallet1',
                timestamp=1609462800,
                balances=[],
                total_value_usd=1500.0,
                eth_price=735.0
            ),
            PortfolioSnapshot(
                wallet_address='0xWallet1',
                timestamp=1609466400,
                balances=[],
                total_value_usd=2000.0,
                eth_price=740.0
            )
        ]

        result = pricer.calculate_time_weighted_value(snapshots)

        assert result is not None
        assert 'time_weighted_value' in result
        assert 'average_value' in result
        assert result['num_snapshots'] == 3
        assert result['start_value'] == 1000.0
        assert result['end_value'] == 2000.0
        assert result['value_change'] == 1000.0
        assert result['value_change_pct'] == 100.0
        assert result['average_value'] == 1500.0  # (1000 + 1500 + 2000) / 3

    def test_aggregate_values_by_token(self, pricer):
        """Test aggregation by token"""
        valued_balances = [
            ValuedBalance(
                wallet_address='0xWallet1',
                token_address='0xTokenA',
                token_symbol='TOKA',
                balance=100.0,
                timestamp=1609459200,
                block_number=1000,
                token_price_usd=10.0,
                balance_value_usd=1000.0,
                price_timestamp=1609459200,
                price_source='coingecko'
            ),
            ValuedBalance(
                wallet_address='0xWallet2',
                token_address='0xTokenA',
                token_symbol='TOKA',
                balance=50.0,
                timestamp=1609459200,
                block_number=1000,
                token_price_usd=10.0,
                balance_value_usd=500.0,
                price_timestamp=1609459200,
                price_source='coingecko'
            ),
            ValuedBalance(
                wallet_address='0xWallet1',
                token_address='0xTokenB',
                token_symbol='TOKB',
                balance=200.0,
                timestamp=1609459200,
                block_number=1000,
                token_price_usd=0.5,
                balance_value_usd=100.0,
                price_timestamp=1609459200,
                price_source='coingecko'
            )
        ]

        aggregated = pricer.aggregate_values_by_token(valued_balances)

        assert len(aggregated) == 2
        assert 'TOKA' in aggregated
        assert 'TOKB' in aggregated

        toka_data = aggregated['TOKA']
        assert toka_data['total_balance'] == 150.0  # 100 + 50
        assert toka_data['total_value_usd'] == 1500.0  # 1000 + 500
        assert toka_data['num_holders'] == 2
        assert toka_data['avg_price_usd'] == 10.0

    def test_aggregate_values_by_wallet(self, pricer):
        """Test aggregation by wallet"""
        valued_balances = [
            ValuedBalance(
                wallet_address='0xWallet1',
                token_address='0xTokenA',
                token_symbol='TOKA',
                balance=100.0,
                timestamp=1609459200,
                block_number=1000,
                token_price_usd=10.0,
                balance_value_usd=1000.0,
                price_timestamp=1609459200,
                price_source='coingecko'
            ),
            ValuedBalance(
                wallet_address='0xWallet1',
                token_address='0xTokenB',
                token_symbol='TOKB',
                balance=200.0,
                timestamp=1609459200,
                block_number=1000,
                token_price_usd=0.5,
                balance_value_usd=100.0,
                price_timestamp=1609459200,
                price_source='coingecko'
            )
        ]

        aggregated = pricer.aggregate_values_by_wallet(valued_balances)

        assert len(aggregated) == 1
        assert '0xWallet1' in aggregated

        wallet_data = aggregated['0xWallet1']
        assert wallet_data['total_value_usd'] == 1100.0
        assert wallet_data['num_tokens'] == 2
        assert len(wallet_data['tokens']) == 2

    def test_track_portfolio_trend_daily(self, pricer):
        """Test portfolio trend tracking"""
        snapshots = [
            PortfolioSnapshot(
                wallet_address='0xWallet1',
                timestamp=1609459200,  # Day 1, 00:00
                balances=[],
                total_value_usd=1000.0,
                eth_price=730.0
            ),
            PortfolioSnapshot(
                wallet_address='0xWallet1',
                timestamp=1609462800,  # Day 1, 01:00
                balances=[],
                total_value_usd=1200.0,
                eth_price=735.0
            ),
            PortfolioSnapshot(
                wallet_address='0xWallet1',
                timestamp=1609545600,  # Day 2, 00:00
                balances=[],
                total_value_usd=1500.0,
                eth_price=740.0
            )
        ]

        trend = pricer.track_portfolio_trend(snapshots, interval='daily')

        assert len(trend) == 2  # 2 daily intervals
        assert trend[0]['num_snapshots'] == 2  # Day 1 has 2 snapshots
        assert trend[0]['avg_value_usd'] == 1100.0  # (1000 + 1200) / 2
        assert trend[1]['num_snapshots'] == 1  # Day 2 has 1 snapshot
        assert trend[1]['avg_value_usd'] == 1500.0

    def test_reprice_balances(self, pricer, sample_balance_snapshot):
        """Test repricing with updated price data"""
        # New price data with higher prices
        new_prices = {
            1609459200: {'price_usd': 800.0, 'timestamp': 1609459200, 'source': 'chainlink'},
        }

        balances = [sample_balance_snapshot]
        repriced = pricer.reprice_balances(balances, new_prices)

        # Note: reprice_balances currently only handles ETH prices
        # Since our sample is token, this should still use original token prices
        assert len(repriced) == 1
        # Verify original pricer data is restored
        assert pricer.eth_price_data[1609459200]['price_usd'] == 730.0

    def test_get_portfolio_composition(self, pricer):
        """Test portfolio composition breakdown"""
        portfolio = PortfolioSnapshot(
            wallet_address='0xWallet1',
            timestamp=1609459200,
            balances=[
                ValuedBalance(
                    wallet_address='0xWallet1',
                    token_address='0xTokenA',
                    token_symbol='TOKA',
                    balance=100.0,
                    timestamp=1609459200,
                    block_number=1000,
                    token_price_usd=10.0,
                    balance_value_usd=1000.0,
                    price_timestamp=1609459200,
                    price_source='coingecko'
                ),
                ValuedBalance(
                    wallet_address='0xWallet1',
                    token_address='0xTokenB',
                    token_symbol='TOKB',
                    balance=500.0,
                    timestamp=1609459200,
                    block_number=1000,
                    token_price_usd=0.5,
                    balance_value_usd=250.0,
                    price_timestamp=1609459200,
                    price_source='coingecko'
                )
            ],
            total_value_usd=1250.0,
            eth_price=730.0
        )

        composition = pricer.get_portfolio_composition(portfolio)

        assert len(composition) == 2
        # Should be sorted by value (TOKA first)
        assert composition[0]['token_symbol'] == 'TOKA'
        assert composition[0]['value_usd'] == 1000.0
        assert composition[0]['percentage'] == 80.0  # 1000/1250 * 100

        assert composition[1]['token_symbol'] == 'TOKB'
        assert composition[1]['value_usd'] == 250.0
        assert composition[1]['percentage'] == 20.0

    def test_empty_portfolio_composition(self, pricer):
        """Test portfolio composition with zero value"""
        portfolio = PortfolioSnapshot(
            wallet_address='0xWallet1',
            timestamp=1609459200,
            balances=[],
            total_value_usd=0.0,
            eth_price=730.0
        )

        composition = pricer.get_portfolio_composition(portfolio)
        assert composition == []

    def test_empty_portfolio_calculation(self, pricer):
        """Test portfolio calculation with no balances"""
        portfolio = pricer.calculate_portfolio_value([])
        assert portfolio is None

    def test_time_weighted_value_empty(self, pricer):
        """Test time-weighted calculation with no snapshots"""
        result = pricer.calculate_time_weighted_value([])

        assert result['time_weighted_value'] == 0.0
        assert result['average_value'] == 0.0
        assert result['num_snapshots'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])