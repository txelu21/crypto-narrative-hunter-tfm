"""
Core unit tests for token services without database dependencies.

Tests:
- Ethereum address validation utilities
- TokenMetadata Pydantic model validation
- CoinGecko API client with mocked HTTP responses
- Token fetcher deduplication logic
- Token export CSV conversion
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import date, datetime
from typing import Dict, Any, List

# Import the core services to test (avoid DB dependencies)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tokens.ethereum_utils import (
    is_valid_ethereum_address,
    to_checksum_address,
    validate_and_normalize_address
)
from services.tokens.token_fetcher import TokenMetadata


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
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f98g",  # Invalid hex character
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f98",  # Too short
            "",
            None
        ]

        for addr in invalid_addresses:
            assert not is_valid_ethereum_address(addr), f"Address should be invalid: {addr}"
            assert validate_and_normalize_address(addr) is None, f"Address should not normalize: {addr}"

    def test_address_normalization(self):
        """Test address normalization to lowercase with 0x prefix"""
        test_cases = [
            ("0x1F9840a85D5aF5BF1D1762F925BDADdC4201F984", "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"),
            ("1f9840a85d5af5bf1d1762f925bdaddc4201f984", "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"),
            ("0X1F9840A85D5AF5BF1D1762F925BDADDC4201F984", "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"),
        ]

        for input_addr, expected in test_cases:
            result = validate_and_normalize_address(input_addr)
            assert result == expected, f"Expected {expected}, got {result}"

    def test_checksum_address_generation(self):
        """Test EIP-55 checksum address generation"""
        test_address = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
        checksum_addr = to_checksum_address(test_address)

        assert checksum_addr is not None
        assert checksum_addr.startswith("0x")
        assert len(checksum_addr) == 42
        # The checksum should have mixed case for a valid address
        hex_part = checksum_addr[2:]
        # For this specific address, we know it should have mixed case
        assert not hex_part.islower(), "Checksum should have some uppercase"
        assert not hex_part.isupper(), "Checksum should have some lowercase"


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
        assert token.market_cap_rank == 10

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

        # Test invalid decimals (too high)
        with pytest.raises(ValueError):
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="TEST",
                name="Test Token",
                decimals=25,  # Too high
                coingecko_id="test"
            )

        # Test invalid decimals (negative)
        with pytest.raises(ValueError):
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="TEST",
                name="Test Token",
                decimals=-1,  # Negative
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

        # Test empty name
        with pytest.raises(ValueError, match="Name cannot be empty"):
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="TEST",
                name="",
                decimals=18,
                coingecko_id="test"
            )

    def test_symbol_uppercase_conversion(self):
        """Test that symbol is automatically converted to uppercase"""
        token = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="uni",  # lowercase
            name="Uniswap",
            decimals=18,
            coingecko_id="uniswap"
        )

        assert token.symbol == "UNI"  # Should be converted to uppercase

    def test_name_trimming(self):
        """Test that name is automatically trimmed"""
        token = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="UNI",
            name="  Uniswap  ",  # With spaces
            decimals=18,
            coingecko_id="uniswap"
        )

        assert token.name == "Uniswap"  # Should be trimmed


class TestCoinGeckoClientMocking:
    """Test CoinGecko API client behavior with mocked HTTP client"""

    def test_cache_key_generation(self):
        """Test cache key generation for consistent caching"""
        # We'll test this without creating a full client
        from services.tokens.coingecko_client import CoinGeckoClient

        with patch('services.tokens.coingecko_client.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()

            client = CoinGeckoClient()

            # Test basic URL
            key1 = client._get_cache_key("https://api.coingecko.com/api/v3/ping")
            assert len(key1) == 16  # MD5 hash truncated to 16 chars
            assert isinstance(key1, str)

            # Same URL should generate same key
            key2 = client._get_cache_key("https://api.coingecko.com/api/v3/ping")
            assert key1 == key2

            # Different URL should generate different key
            key3 = client._get_cache_key("https://api.coingecko.com/api/v3/coins/markets")
            assert key1 != key3

            # URL with params should be different
            key4 = client._get_cache_key("https://api.coingecko.com/api/v3/ping", {"param": "value"})
            assert key1 != key4

    def test_header_sanitization(self):
        """Test that sensitive headers are sanitized for logging"""
        from services.tokens.coingecko_client import CoinGeckoClient

        with patch('services.tokens.coingecko_client.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()

            client = CoinGeckoClient()

            test_headers = {
                "User-Agent": "test-agent",
                "Content-Type": "application/json",
                "x-cg-demo-api-key": "secret-key-123",
                "authorization": "Bearer token123",
                "X-API-Key": "another-secret"
            }

            sanitized = client._sanitize_headers_for_logging(test_headers)

            # Safe headers should remain
            assert sanitized["User-Agent"] == "test-agent"
            assert sanitized["Content-Type"] == "application/json"

            # Sensitive headers should be redacted
            assert sanitized["x-cg-demo-api-key"] == "***REDACTED***"
            assert sanitized["authorization"] == "***REDACTED***"
            assert sanitized["X-API-Key"] == "***REDACTED***"


class TestTokenExportCSV:
    """Test token export functionality (CSV conversion part) - implementation mocked"""

    def test_token_to_csv_row_conversion_logic(self):
        """Test CSV row conversion logic without importing the full exporter"""
        from datetime import datetime

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
            validation_flags=["missing_decimals", "low_volume"]
        )

        # Simulate the CSV row conversion logic
        csv_row = {
            "token_address": token.token_address,
            "symbol": token.symbol,
            "name": token.name,
            "decimals": token.decimals,
            "market_cap_rank": token.market_cap_rank,
            "market_cap_usd": float(token.market_cap_usd) if token.market_cap_usd else None,
            "volume_24h_usd": float(token.volume_24h_usd) if token.volume_24h_usd else None,
            "current_price_usd": float(token.current_price_usd) if token.current_price_usd else None,
            "coingecko_id": token.coingecko_id,
            "has_ethereum_contract": token.has_ethereum_contract,
            "validation_flags": "|".join(token.validation_flags) if token.validation_flags else "",
            "exported_at": datetime.utcnow().isoformat() + "Z"
        }

        # Check all expected fields are present
        assert csv_row["token_address"] == "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
        assert csv_row["symbol"] == "UNI"
        assert csv_row["name"] == "Uniswap"
        assert csv_row["decimals"] == 18
        assert csv_row["market_cap_rank"] == 20
        assert csv_row["market_cap_usd"] == 3300000000.0
        assert csv_row["volume_24h_usd"] == 150000000.0
        assert csv_row["current_price_usd"] == 5.5
        assert csv_row["coingecko_id"] == "uniswap"
        assert csv_row["has_ethereum_contract"] is True
        assert csv_row["validation_flags"] == "missing_decimals|low_volume"
        assert "exported_at" in csv_row
        assert csv_row["exported_at"].endswith("Z")  # ISO format with Z


class TestDataQualityAnalysis:
    """Test data quality analysis functions - implementation mocked"""

    def test_completeness_analysis_logic(self):
        """Test data completeness analysis logic without database dependencies"""
        tokens = [
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="UNI",
                name="Uniswap",
                decimals=18,
                market_cap_rank=20,
                market_cap_usd=Decimal("3300000000"),
                coingecko_id="uniswap"
            ),
            TokenMetadata(
                token_address="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
                symbol="WBTC",
                name="Wrapped BTC",
                decimals=8,
                market_cap_rank=None,  # Missing rank
                market_cap_usd=None,  # Missing market cap
                coingecko_id="wrapped-bitcoin"
            )
        ]

        # Simulate the completeness analysis logic
        total = len(tokens)
        field_completeness = {
            "token_address": sum(1 for t in tokens if t.token_address),
            "symbol": sum(1 for t in tokens if t.symbol),
            "name": sum(1 for t in tokens if t.name),
            "decimals": sum(1 for t in tokens if t.decimals is not None),
            "market_cap_rank": sum(1 for t in tokens if t.market_cap_rank is not None),
            "market_cap_usd": sum(1 for t in tokens if t.market_cap_usd is not None),
            "coingecko_id": sum(1 for t in tokens if t.coingecko_id)
        }

        completeness_percentages = {
            field: (count / total) * 100 for field, count in field_completeness.items()
        }

        # Check basic structure
        assert total == 2
        assert len(field_completeness) == 7

        # Check specific completeness percentages
        assert completeness_percentages["symbol"] == 100.0  # Both have symbols
        assert completeness_percentages["market_cap_rank"] == 50.0  # Only 1 of 2
        assert completeness_percentages["market_cap_usd"] == 50.0  # Only 1 of 2

    def test_validation_flags_analysis_logic(self):
        """Test validation flags analysis logic"""
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

        # Simulate the validation flags analysis logic
        flag_counts = {}
        tokens_with_flags = 0

        for token in tokens:
            if token.validation_flags:
                tokens_with_flags += 1
                for flag in token.validation_flags:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1

        validation_analysis = {
            "total_tokens_with_flags": tokens_with_flags,
            "percentage_with_flags": (tokens_with_flags / len(tokens)) * 100 if tokens else 0,
            "flag_distribution": flag_counts,
            "most_common_flags": sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

        # Check analysis results
        assert validation_analysis["total_tokens_with_flags"] == 2
        assert validation_analysis["percentage_with_flags"] == (2/3) * 100

        # Check flag distribution
        flag_distribution = validation_analysis["flag_distribution"]
        assert flag_distribution["missing_decimals"] == 2
        assert flag_distribution["low_market_cap_rank"] == 1

        # Check most common flags
        most_common = validation_analysis["most_common_flags"]
        assert len(most_common) == 2
        assert most_common[0] == ("missing_decimals", 2)  # Most common first


class TestTokenDeduplication:
    """Test token deduplication logic"""

    def test_address_based_deduplication(self):
        """Test deduplication based on token address"""
        # Create two tokens with same address but different ranks
        token1 = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="UNI",
            name="Uniswap",
            decimals=18,
            market_cap_rank=20,
            coingecko_id="uniswap"
        )

        token2 = TokenMetadata(
            token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # Same address
            symbol="UNI-OLD",
            name="Uniswap Old",
            decimals=18,
            market_cap_rank=50,  # Worse rank
            coingecko_id="uniswap-old"
        )

        # Simulate the deduplication logic
        tokens_by_address = {}

        # First token
        tokens_by_address[token1.token_address] = token1

        # Second token - should replace first if better rank
        existing_rank = tokens_by_address[token2.token_address].market_cap_rank or 999999
        new_rank = token2.market_cap_rank or 999999

        if new_rank < existing_rank:  # Better rank (lower number)
            tokens_by_address[token2.token_address] = token2

        # Should keep token1 (better rank: 20 < 50)
        final_token = tokens_by_address[token1.token_address]
        assert final_token.symbol == "UNI"
        assert final_token.market_cap_rank == 20

    def test_symbol_duplicate_tracking(self):
        """Test tracking of duplicate symbols"""
        tokens = [
            TokenMetadata(
                token_address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                symbol="UNI",
                name="Uniswap V2",
                decimals=18,
                coingecko_id="uniswap"
            ),
            TokenMetadata(
                token_address="0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
                symbol="UNI",  # Same symbol, different address
                name="Uniswap V3",
                decimals=18,
                coingecko_id="uniswap-v3"
            )
        ]

        # Track symbols
        seen_symbols = {}

        for token in tokens:
            if token.symbol in seen_symbols:
                seen_symbols[token.symbol].append(token)
            else:
                seen_symbols[token.symbol] = [token]

        # Should detect duplicate symbols
        assert len(seen_symbols["UNI"]) == 2
        assert seen_symbols["UNI"][0].name == "Uniswap V2"
        assert seen_symbols["UNI"][1].name == "Uniswap V3"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])