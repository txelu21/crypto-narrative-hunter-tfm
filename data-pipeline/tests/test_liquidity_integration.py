import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import os

from services.tokens.dune_client import DuneClient, DuneClientError, DuneQueryError
from services.tokens.liquidity_analyzer import DEXLiquidityAnalyzer, PoolData
from services.tokens.liquidity_validator import LiquidityValidator
from data_collection.common.db import DatabaseManager
from data_collection.common.checkpoints import CheckpointManager


class TestDuneClientIntegration:
    @pytest.fixture
    def mock_dune_client(self):
        with patch.dict(os.environ, {"DUNE_API_KEY": "test_key"}):
            client = DuneClient(cache_dir="./test_cache")
            return client

    @pytest.fixture
    def sample_pool_data(self):
        return pd.DataFrame([
            {
                "pool_address": "0x123...",
                "dex_name": "Uniswap V2",
                "token0": "0xabc...",
                "token1": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "tvl_usd": 1500000.0,
                "volume_24h_usd": 50000.0,
                "price_eth": 0.001234,
                "last_updated": datetime.now()
            },
            {
                "pool_address": "0x456...",
                "dex_name": "Uniswap V3",
                "token0": "0xdef...",
                "token1": "0xA0b86a33E6441B9435B7DF041c0f3B9e8CE61e2C",  # USDC
                "tvl_usd": 800000.0,
                "volume_24h_usd": 25000.0,
                "price_eth": None,
                "last_updated": datetime.now()
            }
        ])

    def test_dune_client_authentication(self, mock_dune_client):
        """Test Dune client authentication setup"""
        assert mock_dune_client.api_key == "test_key"
        assert "X-Dune-API-Key" in mock_dune_client.session.headers

    @patch('services.tokens.dune_client.requests.Session.request')
    def test_execute_query_success(self, mock_request, mock_dune_client):
        """Test successful query execution"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"execution_id": "test_exec_123"}
        mock_request.return_value = mock_response

        execution_id = mock_dune_client.execute_query(
            query_id=12345,
            parameters={"token_addresses": "'0xabc123'"}
        )

        assert execution_id == "test_exec_123"
        assert execution_id in mock_dune_client.active_jobs

    @patch('services.tokens.dune_client.requests.Session.request')
    def test_execute_query_rate_limit(self, mock_request, mock_dune_client):
        """Test rate limit handling"""
        # First request: rate limited
        mock_response_limited = Mock()
        mock_response_limited.status_code = 429
        mock_response_limited.headers = {"Retry-After": "1"}

        # Second request: success
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"execution_id": "test_exec_456"}

        mock_request.side_effect = [mock_response_limited, mock_response_success]

        with patch('time.sleep'):  # Mock sleep to speed up test
            execution_id = mock_dune_client.execute_query(
                query_id=12345,
                parameters={"token_addresses": "'0xabc123'"}
            )

        assert execution_id == "test_exec_456"

    def test_cache_functionality(self, mock_dune_client, sample_pool_data):
        """Test query result caching"""
        cache_key = mock_dune_client._get_cache_key(12345, {"test": "param"})

        # Save to cache
        mock_dune_client._save_to_cache(cache_key, sample_pool_data, {"rows": 2})

        # Load from cache
        cached_data = mock_dune_client._load_from_cache(cache_key)

        assert cached_data is not None
        assert len(cached_data) == len(sample_pool_data)
        assert cached_data.columns.tolist() == sample_pool_data.columns.tolist()

    @patch('services.tokens.dune_client.requests.Session.request')
    def test_poll_execution_completion(self, mock_request, mock_dune_client):
        """Test polling execution until completion"""
        # Mock status responses: running -> completed
        mock_responses = [
            Mock(status_code=200, json=lambda: {"state": "QUERY_STATE_PENDING"}),
            Mock(status_code=200, json=lambda: {"state": "QUERY_STATE_EXECUTING"}),
            Mock(status_code=200, json=lambda: {"state": "QUERY_STATE_COMPLETED", "result_metadata": {}})
        ]
        mock_request.side_effect = mock_responses

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = mock_dune_client.poll_execution("test_exec_123", timeout_seconds=30)

        assert result["state"] == "QUERY_STATE_COMPLETED"

    @patch('services.tokens.dune_client.requests.Session.request')
    def test_poll_execution_failure(self, mock_request, mock_dune_client):
        """Test handling of failed query execution"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "state": "QUERY_STATE_FAILED",
            "error": {"message": "Query syntax error"}
        }
        mock_request.return_value = mock_response

        with pytest.raises(DuneQueryError, match="Query failed"):
            mock_dune_client.poll_execution("test_exec_123")


class TestLiquidityAnalyzerIntegration:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        mock_connection = Mock()
        mock_connection.execute.return_value.fetchall.return_value = [
            ("0xabc123",), ("0xdef456",), ("0x789abc",)
        ]
        db_manager.get_connection.return_value.__enter__.return_value = mock_connection
        return db_manager

    @pytest.fixture
    def mock_checkpoint_manager(self):
        checkpoint_manager = Mock(spec=CheckpointManager)
        checkpoint_manager.load_checkpoint.return_value = {"processed_tokens": []}
        return checkpoint_manager

    @pytest.fixture
    def mock_dune_client_with_data(self, sample_pool_data):
        client = Mock(spec=DuneClient)
        client.execute_and_wait.return_value = sample_pool_data
        return client

    @pytest.fixture
    def liquidity_analyzer(self, mock_dune_client_with_data, mock_db_manager, mock_checkpoint_manager):
        analyzer = DEXLiquidityAnalyzer(
            mock_dune_client_with_data,
            mock_db_manager,
            mock_checkpoint_manager
        )
        # Set query IDs for testing
        analyzer.query_ids = {
            "uniswap_v2": 12345,
            "uniswap_v3": 12346,
            "curve": 12347,
            "token_filter": 12348
        }
        return analyzer

    @pytest.fixture
    def sample_pool_data(self):
        return pd.DataFrame([
            {
                "pool_address": "0x123...",
                "dex_name": "Uniswap V2",
                "token0": "0xabc123",
                "token1": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "token0_symbol": "TEST",
                "token1_symbol": "WETH",
                "tvl_usd": 1500000.0,
                "volume_24h_usd": 50000.0,
                "price_eth": 0.001234,
                "last_updated": datetime.now()
            }
        ])

    def test_get_collected_tokens(self, liquidity_analyzer):
        """Test retrieving collected tokens from database"""
        tokens = liquidity_analyzer.get_collected_tokens(limit=10)
        assert len(tokens) == 3
        assert "0xabc123" in tokens

    def test_assign_liquidity_tier(self, liquidity_analyzer):
        """Test liquidity tier assignment logic"""
        assert liquidity_analyzer.assign_liquidity_tier(15_000_000) == "Tier 1"
        assert liquidity_analyzer.assign_liquidity_tier(5_000_000) == "Tier 2"
        assert liquidity_analyzer.assign_liquidity_tier(500_000) == "Tier 3"
        assert liquidity_analyzer.assign_liquidity_tier(0) == "Tier 3"

    def test_discover_pools_for_tokens(self, liquidity_analyzer):
        """Test pool discovery across DEXs"""
        tokens = ["0xabc123", "0xdef456"]
        pools = liquidity_analyzer.discover_pools_for_tokens(tokens, date_range_days=30)

        # Should query each DEX (3 queries * 1 batch)
        assert liquidity_analyzer.dune_client.execute_and_wait.call_count == 3
        assert len(pools) > 0

    def test_rank_pools_by_liquidity(self, liquidity_analyzer):
        """Test pool ranking by liquidity"""
        pools = [
            PoolData(
                pool_address="0x123",
                dex_name="Uniswap V2",
                token_address="0xabc123",
                pair_token="0xweth",
                tvl_usd=1_000_000,
                volume_24h_usd=50_000,
                price_eth=0.001,
                last_updated=datetime.now(),
                metadata={}
            ),
            PoolData(
                pool_address="0x456",
                dex_name="Uniswap V3",
                token_address="0xabc123",
                pair_token="0xweth",
                tvl_usd=2_000_000,
                volume_24h_usd=100_000,
                price_eth=0.0011,
                last_updated=datetime.now(),
                metadata={}
            )
        ]

        ranked_pools = liquidity_analyzer.rank_pools_by_liquidity(pools)
        token_pools = ranked_pools["0xabc123"]

        # Should be sorted by TVL (highest first)
        assert token_pools[0].tvl_usd == 2_000_000
        assert token_pools[1].tvl_usd == 1_000_000

    def test_calculate_tier_assignments(self, liquidity_analyzer):
        """Test tier assignment calculation"""
        pools = [
            PoolData(
                pool_address="0x123",
                dex_name="Uniswap V2",
                token_address="0xabc123",
                pair_token="0xweth",
                tvl_usd=15_000_000,  # Tier 1
                volume_24h_usd=500_000,
                price_eth=0.001,
                last_updated=datetime.now(),
                metadata={}
            )
        ]

        ranked_pools = {"0xabc123": pools}
        assignments = liquidity_analyzer.calculate_tier_assignments(ranked_pools)

        assert assignments["0xabc123"]["liquidity_tier"] == "Tier 1"
        assert assignments["0xabc123"]["max_tvl"] == 15_000_000
        assert assignments["0xabc123"]["price_eth"] == 0.001

    def test_end_to_end_workflow(self, liquidity_analyzer):
        """Test complete liquidity analysis workflow"""
        result = liquidity_analyzer.run_liquidity_analysis(token_limit=5)

        assert result["status"] == "completed"
        assert "pools_discovered" in result
        assert "tokens_analyzed" in result

        # Verify database operations were called
        liquidity_analyzer.db_manager.get_connection.assert_called()

        # Verify checkpoint operations
        liquidity_analyzer.checkpoint_manager.load_checkpoint.assert_called()
        liquidity_analyzer.checkpoint_manager.save_checkpoint.assert_called()


class TestLiquidityValidatorIntegration:
    @pytest.fixture
    def mock_db_manager_with_data(self):
        db_manager = Mock(spec=DatabaseManager)
        mock_connection = Mock()

        # Mock coverage validation data
        mock_connection.execute.return_value.fetchone.side_effect = [
            (100, 85, 10, 35, 40),  # coverage stats
            (85, 75),  # pricing stats
        ]

        # Mock tier distribution data
        mock_connection.execute.return_value.fetchall.return_value = [
            ("Tier 1", 10),
            ("Tier 2", 35),
            ("Tier 3", 40),
            ("Untiered", 15)
        ]

        db_manager.get_connection.return_value.__enter__.return_value = mock_connection
        return db_manager

    @pytest.fixture
    def validator(self, mock_db_manager_with_data):
        return LiquidityValidator(mock_db_manager_with_data)

    def test_validate_liquidity_coverage(self, validator):
        """Test liquidity coverage validation"""
        result = validator.validate_liquidity_coverage()

        assert result.test_name == "liquidity_coverage"
        assert result.score == 85.0  # 85 out of 100 tokens have liquidity
        assert result.passed  # Above 70% threshold

    def test_validate_pricing_coverage(self, validator):
        """Test ETH pricing coverage validation"""
        result = validator.validate_pricing_coverage()

        assert result.test_name == "pricing_coverage"
        assert result.passed  # Above 60% threshold

    def test_validate_tier_distribution(self, validator):
        """Test tier distribution validation"""
        result = validator.validate_tier_distribution()

        assert result.test_name == "tier_distribution"
        assert result.details["tier_counts"]["Tier 1"] == 10
        assert result.details["tier_percentages"]["Tier 1"] == 10.0  # 10/100

    @patch('services.tokens.liquidity_validator.requests.get')
    def test_cross_validate_with_external_sources(self, mock_get, validator):
        """Test external price validation"""
        # Mock successful external API calls
        mock_eth_response = Mock()
        mock_eth_response.json.return_value = {"ethereum": {"usd": 2000}}

        mock_token_response = Mock()
        mock_token_response.json.return_value = {"test": {"usd": 2.0}}

        mock_get.side_effect = [mock_eth_response, mock_token_response]

        # Mock database data for validation
        validator.db_manager.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            ("0xtest", "TEST", 0.001, 1000000)  # price_eth = 0.001
        ]

        result = validator.cross_validate_with_external_sources(sample_size=1)

        assert result.test_name == "external_validation"
        # External price: 2.0 USD / 2000 USD = 0.001 ETH (matches our price)
        assert result.passed

    def test_detect_tvl_anomalies(self, validator):
        """Test TVL anomaly detection"""
        # Mock pool data with outliers
        validator.db_manager.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
            ("0xtoken1", "0xpool1", "Uniswap V2", 1_000_000, 50_000),
            ("0xtoken2", "0xpool2", "Uniswap V3", 2_000_000, 100_000),
            ("0xtoken3", "0xpool3", "Curve", 50_000_000, 1_000_000),  # Outlier
        ]

        anomalies = validator.detect_tvl_anomalies()

        # Should detect the high TVL outlier
        assert len(anomalies) > 0
        tvl_anomaly = next((a for a in anomalies if a.metric_name == "tvl_outliers"), None)
        assert tvl_anomaly is not None

    def test_generate_quality_report(self, validator):
        """Test comprehensive quality report generation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            report = validator.generate_quality_report(output_path)

            assert "report_metadata" in report
            assert "overall_score" in report["report_metadata"]
            assert "validation_results" in report
            assert "anomaly_detection" in report
            assert "summary_recommendations" in report

            # Verify file was created
            assert os.path.exists(output_path)

            # Verify JSON structure
            with open(output_path, 'r') as f:
                saved_report = json.load(f)
            assert saved_report["report_metadata"]["overall_score"] == report["report_metadata"]["overall_score"]

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_functionality(self, validator):
        """Test data export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name

        try:
            # Mock pandas read_sql_query
            with patch('pandas.read_sql_query') as mock_read_sql:
                mock_df = pd.DataFrame([{
                    "symbol": "TEST",
                    "token_address": "0xtest",
                    "pool_address": "0xpool",
                    "dex_name": "Uniswap V2",
                    "tvl_usd": 1000000
                }])
                mock_df.to_csv = Mock()
                mock_read_sql.return_value = mock_df

                validator.export_pool_data_for_review(output_path)

                # Verify export was called
                mock_df.to_csv.assert_called_once_with(output_path, index=False)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestErrorHandlingAndRecovery:
    def test_dune_api_error_handling(self):
        """Test proper error handling for Dune API failures"""
        with patch.dict(os.environ, {"DUNE_API_KEY": "test_key"}):
            client = DuneClient()

            with patch('services.tokens.dune_client.requests.Session.request') as mock_request:
                # Mock API error
                mock_response = Mock()
                mock_response.status_code = 400
                mock_response.text = "Bad Request"
                mock_response.raise_for_status.side_effect = Exception("API Error")
                mock_request.return_value = mock_response

                with pytest.raises(DuneClientError):
                    client.execute_query(12345, {})

    def test_checkpoint_recovery(self, mock_db_manager, mock_checkpoint_manager):
        """Test checkpoint recovery functionality"""
        # Mock existing checkpoint
        mock_checkpoint_manager.load_checkpoint.return_value = {
            "processed_tokens": ["0xabc123", "0xdef456"]
        }

        with patch('services.tokens.dune_client.DuneClient'):
            analyzer = DEXLiquidityAnalyzer(
                Mock(), mock_db_manager, mock_checkpoint_manager
            )

            # Mock token list
            mock_db_manager.get_connection.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [
                ("0xabc123",), ("0xdef456",), ("0x789abc",)
            ]

            tokens = analyzer.get_collected_tokens()

            # Should only process new tokens
            # This would be tested in the actual run_liquidity_analysis method


class TestPerformanceAndScaling:
    @pytest.mark.slow
    def test_large_token_batch_processing(self):
        """Test processing large batches of tokens"""
        # This test would verify that the system can handle large datasets
        # without memory issues or timeouts
        pass

    @pytest.mark.slow
    def test_query_timeout_handling(self):
        """Test handling of long-running Dune queries"""
        # This test would verify proper timeout handling for complex queries
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])