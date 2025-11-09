import pytest
from unittest.mock import MagicMock, patch, call
from psycopg import OperationalError

from data_collection.common import db


def test_init_pool():
    """Test connection pool initialization"""
    with patch('data_collection.common.db.ConnectionPool') as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Reset global pool
        db._pool = None

        # Initialize pool
        db.init_pool()

        # Check pool was created with correct parameters
        mock_pool_class.assert_called_once()
        assert db._pool == mock_pool


def test_get_pool_initializes_if_needed():
    """Test get_pool initializes pool if not already initialized"""
    with patch('data_collection.common.db.init_pool') as mock_init:
        db._pool = None
        pool = db.get_pool()
        mock_init.assert_called_once()


def test_execute_with_retry_success():
    """Test successful query execution with retry logic"""
    with patch('data_collection.common.db.get_pool') as mock_get_pool:
        # Setup mocks
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [{'id': 1}]

        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Execute query
        result = db.execute_with_retry("SELECT * FROM test", fetch=True)

        # Verify
        assert result == [{'id': 1}]
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test", None)
        mock_conn.commit.assert_called_once()


def test_execute_with_retry_retries_on_operational_error():
    """Test that execute_with_retry retries on OperationalError"""
    with patch('data_collection.common.db.get_pool') as mock_get_pool:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # First call raises OperationalError, second succeeds
        mock_cursor.execute.side_effect = [
            OperationalError("Connection lost"),
            None
        ]
        mock_cursor.fetchall.return_value = [{'id': 2}]

        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        # Execute with retry - should succeed on second attempt
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = db.execute_with_retry("SELECT * FROM test", fetch=True)

        assert result == [{'id': 2}]
        assert mock_cursor.execute.call_count == 2


def test_get_cursor_context_manager():
    """Test get_cursor context manager"""
    with patch('data_collection.common.db.get_pool') as mock_get_pool:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1")

        mock_cursor.execute.assert_called_once_with("SELECT 1")
        mock_conn.commit.assert_called_once()


def test_get_cursor_readonly():
    """Test get_cursor in readonly mode"""
    with patch('data_collection.common.db.get_pool') as mock_get_pool:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        with db.get_cursor(readonly=True) as cursor:
            cursor.execute("SELECT 1")

        assert mock_conn.read_only is True
        assert mock_conn.autocommit is True


def test_get_cursor_rollback_on_exception():
    """Test get_cursor rolls back on exception"""
    with patch('data_collection.common.db.get_pool') as mock_get_pool:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        with pytest.raises(RuntimeError):
            with db.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                raise RuntimeError("Test error")

        mock_conn.rollback.assert_called_once()


def test_upsert_record():
    """Test upsert_record helper function"""
    with patch('data_collection.common.db.execute_with_retry') as mock_execute:
        data = {
            'id': 1,
            'name': 'test',
            'value': 100
        }

        db.upsert_record('test_table', data, ['id'])

        # Check the SQL query was built correctly
        call_args = mock_execute.call_args
        query = call_args[0][0]
        assert 'INSERT INTO test_table' in query
        assert 'ON CONFLICT (id)' in query
        assert 'DO UPDATE SET' in query


def test_batch_insert():
    """Test batch_insert helper function"""
    with patch('data_collection.common.db.get_pool') as mock_get_pool:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 2

        mock_pool.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_pool.return_value = mock_pool

        records = [
            {'id': 1, 'name': 'test1'},
            {'id': 2, 'name': 'test2'}
        ]

        total = db.batch_insert('test_table', records)

        assert total == 2
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


def test_test_connection():
    """Test database connection test function"""
    with patch('data_collection.common.db.execute_with_retry') as mock_execute:
        # Test successful connection
        mock_execute.return_value = [{'test': 1}]
        assert db.test_connection() is True

        # Test failed connection
        mock_execute.side_effect = Exception("Connection failed")
        assert db.test_connection() is False


def test_close_pool():
    """Test closing the connection pool"""
    mock_pool = MagicMock()
    db._pool = mock_pool

    db.close_pool()

    mock_pool.close.assert_called_once()
    assert db._pool is None