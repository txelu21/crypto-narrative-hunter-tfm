import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# Ensure the project root (containing the data_collection package) is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool"""
    with patch('data_collection.common.db._pool') as mock_pool:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_pool.connection.return_value = mock_conn
        yield mock_pool


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary log directory for tests"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def mock_settings(temp_log_dir):
    """Mock settings for tests"""
    with patch('data_collection.common.config.settings') as mock_settings:
        mock_settings.database_url = "postgresql://test:test@localhost:5432/test"
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = str(temp_log_dir)
        mock_settings.http_timeout = 30
        mock_settings.tls_verify = True
        mock_settings.max_retries = 3
        yield mock_settings


@pytest.fixture
def temp_staging_dir(tmp_path):
    """Create temporary staging directory for checkpoint tests"""
    staging_dir = tmp_path / "tmp"
    staging_dir.mkdir()
    return staging_dir