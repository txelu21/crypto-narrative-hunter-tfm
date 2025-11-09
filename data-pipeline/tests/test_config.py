import pytest
import os
from unittest.mock import patch

from data_collection.common.config import Settings


def test_settings_defaults():
    """Test Settings class with default values"""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()

        # Check defaults
        assert settings.database_url == "postgresql://user:password@localhost:5432/crypto_narratives"
        assert settings.log_level == "INFO"
        assert settings.log_dir == "./logs"
        assert settings.http_timeout == 30
        assert settings.tls_verify is True
        assert settings.max_retries == 5


def test_settings_from_environment():
    """Test Settings class reads from environment variables"""
    env_vars = {
        'DATABASE_URL': 'postgresql://custom:pass@db:5432/mydb',
        'LOG_LEVEL': 'DEBUG',
        'LOG_DIR': '/custom/logs',
        'DUNE_API_KEY': 'dune123',
        'ALCHEMY_API_KEY': 'alchemy456',
        'ETHERSCAN_API_KEY': 'etherscan789',
        'COINGECKO_API_KEY': 'gecko000',
        'HTTP_TIMEOUT': '60',
        'TLS_VERIFY': 'false',
        'MAX_RETRIES': '10'
    }

    with patch.dict(os.environ, env_vars):
        settings = Settings()

        assert settings.database_url == 'postgresql://custom:pass@db:5432/mydb'
        assert settings.log_level == 'DEBUG'
        assert settings.log_dir == '/custom/logs'
        assert settings.dune_api_key == 'dune123'
        assert settings.alchemy_api_key == 'alchemy456'
        assert settings.etherscan_api_key == 'etherscan789'
        assert settings.coingecko_api_key == 'gecko000'
        assert settings.http_timeout == 60
        assert settings.tls_verify is False
        assert settings.max_retries == 10


def test_pinned_base_urls():
    """Test that base URLs are pinned to prevent SSRF"""
    settings = Settings()

    # All base URLs should be HTTPS and pinned
    assert settings.dune_base_url == "https://api.dune.com/api/v1"
    assert settings.alchemy_base_url == "https://eth-mainnet.g.alchemy.com/v2"
    assert settings.etherscan_base_url == "https://api.etherscan.io/api"
    assert settings.coingecko_base_url == "https://api.coingecko.com/api/v3"

    # Verify they all use HTTPS
    assert settings.dune_base_url.startswith("https://")
    assert settings.alchemy_base_url.startswith("https://")
    assert settings.etherscan_base_url.startswith("https://")
    assert settings.coingecko_base_url.startswith("https://")


def test_validate_success():
    """Test successful validation"""
    with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
        settings = Settings()
        assert settings.validate() is True


def test_validate_missing_database_url():
    """Test validation fails when DATABASE_URL is empty"""
    with patch.dict(os.environ, {'DATABASE_URL': ''}, clear=True):
        settings = Settings()
        settings.database_url = ''  # Force empty value

        with pytest.raises(ValueError, match="DATABASE_URL must be set"):
            settings.validate()


def test_tls_verify_parsing():
    """Test TLS_VERIFY parsing from string to boolean"""
    # Test various true values
    for value in ['true', 'True', 'TRUE', '1', 'yes']:
        with patch.dict(os.environ, {'TLS_VERIFY': value}):
            settings = Settings()
            assert settings.tls_verify is True

    # Test various false values
    for value in ['false', 'False', 'FALSE', '0', 'no']:
        with patch.dict(os.environ, {'TLS_VERIFY': value}):
            settings = Settings()
            assert settings.tls_verify is False


def test_api_keys_optional():
    """Test that API keys are optional (empty by default)"""
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()

        # API keys should default to empty strings
        assert settings.dune_api_key == ""
        assert settings.alchemy_api_key == ""
        assert settings.etherscan_api_key == ""
        assert settings.coingecko_api_key == ""

        # Validation should still pass without API keys
        assert settings.validate() is True


def test_security_settings():
    """Test security-related settings are properly configured"""
    settings = Settings()

    # Security settings should have safe defaults
    assert settings.tls_verify is True  # TLS verification enabled by default
    assert settings.http_timeout > 0  # Timeout is set
    assert settings.max_retries > 0  # Retries are limited

    # Base URLs should be pinned (not configurable via env)
    with patch.dict(os.environ, {'DUNE_BASE_URL': 'http://evil.com'}):
        new_settings = Settings()
        # Should still use the pinned URL, not the env var
        assert new_settings.dune_base_url == "https://api.dune.com/api/v1"