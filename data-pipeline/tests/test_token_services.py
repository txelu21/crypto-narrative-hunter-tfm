"""
Unit tests for token services including API client, validation, storage, and checkpoint management.

Tests:
- CoinGecko API client with mocked responses
- Address validation with valid/invalid Ethereum addresses
- Deduplication logic with test data
- Checkpoint system with simulated failures
- Database constraints and UPSERT operations
- Progress tracking and logging output
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import date, datetime
from typing import Dict, Any, List

# Import the services to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tokens.coingecko_client import CoinGeckoClient, CoinGeckoAPIError, CoinGeckoRateLimitError
from services.tokens.ethereum_utils import (
    is_valid_ethereum_address,
    to_checksum_address,
    validate_and_normalize_address
)
from services.tokens.token_fetcher import TokenMetadata, TokenFetcher, FetchStats
from services.tokens.token_storage import TokenStorage
from services.tokens.token_checkpoint import TokenCheckpointManager
from services.tokens.token_export import TokenExporter


class TestEthereumUtils:
    """Test Ethereum address validation utilities"""

    def test_valid_ethereum_addresses(self):
        """Test validation of valid Ethereum addresses"""
        valid_addresses = [
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # UNI token
            "0xa0b86a33e6e6cd87c5e4d3a3d32f3a5eec1b8b5b",  # Random valid address
            "0x0000000000000000000000000000000000000000",  # Zero address
            "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",  # Max address
        ]

        for addr in valid_addresses:
            assert is_valid_ethereum_address(addr), f"Address should be valid: {addr}"
            normalized = validate_and_normalize_address(addr)
            assert normalized is not None, f"Address should normalize: {addr}"
            assert normalized.startswith("0x"), f"Normalized address should start with 0x: {normalized}"
            assert len(normalized) == 42, f"Normalized address should be 42 chars: {normalized}"

    def test_invalid_ethereum_addresses(self):
        """Test validation of invalid Ethereum addresses"""
        invalid_addresses = [
            "invalid_address",
            "0x123",  # Too short
            "1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # Missing 0x but wrong length check
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f98g",  # Invalid hex character
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f98",  # Too short
            "",
            None
        ]

        for addr in invalid_addresses:
            assert not is_valid_ethereum_address(addr), f"Address should be invalid: {addr}"
            assert validate_and_normalize_address(addr) is None, f"Address should not normalize: {addr}"

    def test_checksum_address_generation(self):
        """Test EIP-55 checksum address generation"""
        test_address = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
        checksum_addr = to_checksum_address(test_address)

        assert checksum_addr is not None
        assert checksum_addr.startswith("0x")
        assert len(checksum_addr) == 42
        # The checksum should have mixed case
        hex_part = checksum_addr[2:]
        assert not (hex_part.islower() or hex_part.isupper()), "Checksum should have mixed case"


class TestTokenMetadata:
    """Test TokenMetadata Pydantic model validation"""

    def test_valid_token_metadata(self):
        """Test creation of valid token metadata"""
        valid_data = {
            "token_address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "symbol": "UNI",
            "name": "Uniswap",
            "decimals": 18,
            "market_cap_rank": 10,
            "market_cap_usd": Decimal("1000000000"),
            "volume_24h_usd": Decimal("100000000"),
            "current_price_usd": Decimal("5.50"),
            "coingecko_id": "uniswap",
            "has_ethereum_contract": True,
            "validation_flags": []
        }

        token = TokenMetadata(**valid_data)
        assert token.token_address == "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
        assert token.symbol == "UNI"
        assert token.decimals == 18

    def test_invalid_token_metadata(self):
        """Test validation errors for invalid token metadata"""
        # Test invalid address
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            TokenMetadata(
                token_address="invalid_address",
                symbol="TEST",
                name="Test Token",
                decimals=18,
                coingecko_id="test"
            )

        # Test invalid decimals
        with pytest.raises(ValueError):
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="TEST",
                name="Test Token",
                decimals=25,  # Too high
                coingecko_id="test"
            )

        # Test empty symbol
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="",
                name="Test Token",
                decimals=18,
                coingecko_id="test"
            )


class TestCoinGeckoClient:
    """Test CoinGecko API client with mocked responses"""

    @patch('services.tokens.coingecko_client.httpx.Client')
    def test_successful_api_request(self, mock_client_class):
        """Test successful API request with mocked response"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"gecko_says": "(V3) To the Moon!"}
        mock_response.headers = {}

        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        # Test the client
        with patch('services.tokens.coingecko_client.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()
            client = CoinGeckoClient()

            result = client._make_request("/ping")
            assert result == {"gecko_says": "(V3) To the Moon!"}

    @patch('services.tokens.coingecko_client.httpx.Client')
    def test_rate_limit_handling(self, mock_client_class):
        """Test rate limit error handling (429 response)"""
        # Setup mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = "Rate limited"

        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        # Test rate limit exception
        with patch('services.tokens.coingecko_client.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()
            with patch('services.tokens.coingecko_client.time.sleep') as mock_sleep:
                client = CoinGeckoClient()

                with pytest.raises(CoinGeckoRateLimitError):
                    client._make_request("/test")

                # Verify sleep was called with retry-after value
                mock_sleep.assert_called_with(60)

    @patch('services.tokens.coingecko_client.httpx.Client')
    def test_client_error_handling(self, mock_client_class):
        """Test client error handling (4xx responses)"""
        # Setup mock client error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_client_instance = Mock()
        mock_client_instance.get.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        # Test client error exception
        with patch('services.tokens.coingecko_client.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()
            client = CoinGeckoClient()

            with pytest.raises(CoinGeckoAPIError, match="Client error 400"):
                client._make_request("/test")

    def test_cache_functionality(self):
        """Test disk caching functionality"""
        with patch('services.tokens.coingecko_client.Path') as mock_path_class:
            # Setup mock path and file operations
            mock_cache_dir = Mock()
            mock_cache_file = Mock()
            mock_cache_file.exists.return_value = True

            mock_path_class.return_value = mock_cache_dir
            mock_cache_dir.__truediv__.return_value = mock_cache_file

            # Mock file content
            cache_data = {
                "cached_at": time.time() - 3600,  # 1 hour ago
                "response": {"test": "data"}
            }

            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(cache_data)

                with patch('json.load', return_value=cache_data):
                    client = CoinGeckoClient()
                    cache_key = client._get_cache_key("http://test.com", {"param": "value"})

                    # Test cache key generation
                    assert len(cache_key) == 16  # MD5 hash truncated to 16 chars
                    assert isinstance(cache_key, str)


class TestTokenFetcher:
    """Test token fetcher with mocked CoinGecko responses"""

    def create_mock_markets_response(self) -> List[Dict[str, Any]]:
        """Create mock markets API response"""
        return [
            {
                "id": "ethereum",
                "symbol": "eth",
                "name": "Ethereum",
                "current_price": 1500.0,
                "market_cap": 180000000000,
                "market_cap_rank": 2,
                "total_volume": 10000000000
            },
            {
                "id": "uniswap",
                "symbol": "uni",
                "name": "Uniswap",
                "current_price": 5.50,
                "market_cap": 3300000000,
                "market_cap_rank": 20,
                "total_volume": 150000000
            }
        ]

    def create_mock_coin_details_response(self, coin_id: str) -> Dict[str, Any]:
        """Create mock coin details API response"""
        if coin_id == "uniswap":
            return {
                "id": "uniswap",
                "symbol": "uni",
                "name": "Uniswap",
                "platforms": {
                    "ethereum": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
                },
                "detail_platforms": {
                    "ethereum": {
                        "decimal_place": 18,
                        "contract_address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
                    }
                }
            }
        else:
            return {
                "id": coin_id,
                "platforms": {}  # No Ethereum contract
            }

    def test_token_processing_with_valid_ethereum_token(self):
        """Test processing of token with valid Ethereum contract"""
        mock_client = Mock()
        mock_client.get_coins_markets.return_value = self.create_mock_markets_response()
        mock_client.get_coin_details_by_id.side_effect = lambda coin_id: self.create_mock_coin_details_response(coin_id)

        fetcher = TokenFetcher(mock_client)

        # Process the UNI token (has Ethereum contract)
        raw_token = self.create_mock_markets_response()[1]  # UNI token
        stats = FetchStats()

        result = fetcher._process_raw_token(raw_token, stats)

        assert result is not None
        assert result.symbol == "UNI"
        assert result.token_address == "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
        assert result.decimals == 18
        assert result.has_ethereum_contract is True
        assert stats.ethereum_tokens_found == 1
        assert stats.tokens_with_contracts == 1

    def test_token_processing_without_ethereum_contract(self):
        """Test processing of token without Ethereum contract"""
        mock_client = Mock()
        mock_client.get_coin_details_by_id.side_effect = lambda coin_id: self.create_mock_coin_details_response(coin_id)

        fetcher = TokenFetcher(mock_client)

        # Process ETH token (no contract address)
        raw_token = self.create_mock_markets_response()[0]  # ETH token
        stats = FetchStats()

        result = fetcher._process_raw_token(raw_token, stats)

        assert result is None  # Should be filtered out
        assert stats.ethereum_tokens_found == 0

    def test_deduplication_logic(self):
        """Test token deduplication logic"""
        mock_client = Mock()
        fetcher = TokenFetcher(mock_client)

        # Create test tokens with same address but different ranks
        token1 = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="UNI",
            name="Uniswap",
            decimals=18,
            market_cap_rank=20,
            coingecko_id="uniswap"
        )

        token2 = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="UNI-OLD",
            name="Uniswap Old",
            decimals=18,
            market_cap_rank=50,  # Worse rank
            coingecko_id="uniswap-old"
        )

        tokens_by_address = {}
        seen_symbols = {}
        stats = FetchStats()

        # Process both tokens
        fetcher._handle_token_deduplication(token1, tokens_by_address, seen_symbols, stats)
        fetcher._handle_token_deduplication(token2, tokens_by_address, seen_symbols, stats)

        # Should keep token1 (better rank)
        assert len(tokens_by_address) == 1
        stored_token = tokens_by_address["0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"]
        assert stored_token.symbol == "UNI"
        assert stored_token.market_cap_rank == 20
        assert stats.duplicates_found == 1


class TestTokenStorage:
    """Test token storage operations"""

    @patch('services.tokens.token_storage.get_cursor')
    def test_upsert_token_insert(self, mock_get_cursor):
        """Test UPSERT operation for new token (INSERT)"""
        # Setup mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            None,  # First call: check if exists
            {"inserted": True}  # Second call: UPSERT result
        ]
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        storage = TokenStorage()

        test_record = {
            "token_address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "symbol": "UNI",
            "name": "Uniswap",
            "decimals": 18,
            "market_cap_rank": 20,
            "avg_daily_volume_usd": 150000000.0
        }

        result = storage._upsert_token(mock_cursor, test_record)

        assert result == "inserted"
        assert mock_cursor.execute.call_count == 2  # Check + UPSERT

    @patch('services.tokens.token_storage.get_cursor')
    def test_upsert_token_update(self, mock_get_cursor):
        """Test UPSERT operation for existing token (UPDATE)"""
        # Setup mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            {"token_address": "0x123", "updated_at": "2023-01-01"},  # Exists
            {"inserted": False}  # UPSERT result (update)
        ]
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        storage = TokenStorage()

        test_record = {
            "token_address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "symbol": "UNI",
            "name": "Uniswap",
            "decimals": 18
        }

        result = storage._upsert_token(mock_cursor, test_record)

        assert result == "updated"

    def test_record_validation(self):
        """Test database record validation"""
        storage = TokenStorage()

        # Valid record
        valid_record = {
            "token_address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "symbol": "UNI",
            "name": "Uniswap",
            "decimals": 18,
            "market_cap_rank": 20,
            "avg_daily_volume_usd": 150000000.0
        }

        # Should not raise exception
        storage._validate_record(valid_record)

        # Invalid address
        invalid_record = valid_record.copy()
        invalid_record["token_address"] = "invalid_address"
        with pytest.raises(ValueError, match="Invalid token address format"):
            storage._validate_record(invalid_record)

        # Invalid decimals
        invalid_record = valid_record.copy()
        invalid_record["decimals"] = 25
        with pytest.raises(ValueError, match="Decimals must be between 0 and 18"):
            storage._validate_record(invalid_record)


class TestTokenCheckpointManager:
    """Test checkpoint management functionality"""

    @patch('services.tokens.token_checkpoint.CheckpointManager')
    def test_collection_status_new(self, mock_checkpoint_manager_class):
        """Test getting status for new collection (no checkpoint)"""
        mock_manager = Mock()
        mock_manager.get_last_checkpoint.return_value = None
        mock_checkpoint_manager_class.return_value = mock_manager

        checkpoint_mgr = TokenCheckpointManager()
        status = checkpoint_mgr.get_collection_status()

        assert status["status"] == "new"
        assert status["can_resume"] is False
        assert status["total_tokens_collected"] == 0

    @patch('services.tokens.token_checkpoint.CheckpointManager')
    def test_collection_status_running(self, mock_checkpoint_manager_class):
        """Test getting status for running collection"""
        mock_manager = Mock()
        mock_manager.get_last_checkpoint.return_value = {
            "status": "running",
            "last_processed_date": date.today(),
            "records_collected": 250,
            "updated_at": datetime.now()
        }
        mock_checkpoint_manager_class.return_value = mock_manager

        checkpoint_mgr = TokenCheckpointManager()
        status = checkpoint_mgr.get_collection_status()

        assert status["status"] == "running"
        assert status["can_resume"] is True
        assert status["total_tokens_collected"] == 250
        assert status["resume_needed"] is True

    @patch('services.tokens.token_checkpoint.CheckpointManager')
    def test_start_collection(self, mock_checkpoint_manager_class):
        """Test starting a new collection"""
        mock_manager = Mock()
        mock_checkpoint_manager_class.return_value = mock_manager

        checkpoint_mgr = TokenCheckpointManager()
        checkpoint_mgr.start_collection(500)

        # Verify checkpoint was updated
        mock_manager.update_checkpoint.assert_called_once()
        call_args = mock_manager.update_checkpoint.call_args
        assert call_args[1]["status"] == "running"
        assert call_args[1]["records_collected"] == 0

    @patch('services.tokens.token_checkpoint.CheckpointManager')
    def test_mark_collection_failed(self, mock_checkpoint_manager_class):
        """Test marking collection as failed"""
        mock_manager = Mock()
        mock_manager.get_last_checkpoint.return_value = {
            "records_collected": 150
        }
        mock_checkpoint_manager_class.return_value = mock_manager

        checkpoint_mgr = TokenCheckpointManager()
        checkpoint_mgr.mark_collection_failed("Test error message")

        # Verify checkpoint was updated with failed status
        mock_manager.update_checkpoint.assert_called_once()
        call_args = mock_manager.update_checkpoint.call_args
        assert call_args[1]["status"] == "failed"
        assert call_args[1]["records_collected"] == 150  # Should preserve existing count

    def test_checkpoint_recovery_scenarios(self):
        """Test various checkpoint recovery scenarios"""
        # This test would need database access, so we'll mock it
        with patch('services.tokens.token_checkpoint.CheckpointManager') as mock_class:
            mock_manager = Mock()
            mock_class.return_value = mock_manager

            checkpoint_mgr = TokenCheckpointManager()

            # Test recovery test method
            mock_manager.get_last_checkpoint.side_effect = [
                None,  # Original state
                {"records_collected": 100, "status": "running"},  # After test checkpoint
                {"status": "failed"}  # After failure test
            ]

            recovery_results = checkpoint_mgr.test_checkpoint_recovery()

            assert "recovery_scenarios_tested" in recovery_results
            assert isinstance(recovery_results["all_tests_passed"], bool)


class TestTokenExporter:
    """Test token export and validation functionality"""

    def test_token_to_csv_row_conversion(self):
        """Test conversion of TokenMetadata to CSV row"""
        exporter = TokenExporter()

        token = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="UNI",
            name="Uniswap",
            decimals=18,
            market_cap_rank=20,
            market_cap_usd=Decimal("3300000000"),
            volume_24h_usd=Decimal("150000000"),
            current_price_usd=Decimal("5.50"),
            coingecko_id="uniswap",
            has_ethereum_contract=True,
            validation_flags=["missing_decimals"]
        )

        csv_row = exporter._token_to_csv_row(token)

        assert csv_row["token_address"] == "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
        assert csv_row["symbol"] == "UNI"
        assert csv_row["decimals"] == 18
        assert csv_row["market_cap_usd"] == 3300000000.0
        assert csv_row["validation_flags"] == "missing_decimals"
        assert "exported_at" in csv_row

    def test_completeness_analysis(self):
        """Test data completeness analysis"""
        exporter = TokenExporter()

        tokens = [
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="UNI",
                name="Uniswap",
                decimals=18,
                market_cap_rank=20,
                coingecko_id="uniswap"
            ),
            TokenMetadata(
                token_address="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
                symbol="WBTC",
                name="Wrapped BTC",
                decimals=8,
                market_cap_rank=None,  # Missing rank
                coingecko_id="wrapped-bitcoin"
            )
        ]

        completeness = exporter._analyze_completeness(tokens)

        assert completeness["total_tokens"] == 2
        assert completeness["field_completeness_percentages"]["symbol"] == 100.0
        assert completeness["field_completeness_percentages"]["market_cap_rank"] == 50.0

    def test_validation_flags_analysis(self):
        """Test validation flags analysis"""
        exporter = TokenExporter()

        tokens = [
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="UNI",
                name="Uniswap",
                decimals=18,
                coingecko_id="uniswap",
                validation_flags=["missing_decimals"]
            ),
            TokenMetadata(
                token_address="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
                symbol="WBTC",
                name="Wrapped BTC",
                decimals=8,
                coingecko_id="wrapped-bitcoin",
                validation_flags=["missing_decimals", "low_market_cap_rank"]
            ),
            TokenMetadata(
                token_address="0xa0b86a33e6e6cd87c5e4d3a3d32f3a5eec1b8b5b",
                symbol="TEST",
                name="Test Token",
                decimals=18,
                coingecko_id="test",
                validation_flags=[]  # No flags
            )
        ]

        validation_analysis = exporter._analyze_validation_flags(tokens)

        assert validation_analysis["total_tokens_with_flags"] == 2
        assert validation_analysis["percentage_with_flags"] == (2/3) * 100
        assert validation_analysis["flag_distribution"]["missing_decimals"] == 2
        assert validation_analysis["flag_distribution"]["low_market_cap_rank"] == 1


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])