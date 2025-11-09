import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

from data_collection.common.logging_setup import (
    StructuredJsonFormatter,
    StructuredLogger,
    get_logger,
    setup_logging,
    log_summary
)


def test_structured_json_formatter():
    """Test JSON formatter produces correct structured output"""
    formatter = StructuredJsonFormatter()

    # Create a log record
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=10,
        msg='Test message',
        args=(),
        exc_info=None
    )

    # Add custom fields
    record.component = 'test_component'
    record.operation = 'test_operation'
    record.params_hash = 'abc123'
    record.status = 'completed'
    record.duration_ms = 100
    record.error = ''

    # Format the record
    output = formatter.format(record)
    data = json.loads(output)

    # Check required fields
    assert 'ts' in data
    assert data['component'] == 'test_component'
    assert data['operation'] == 'test_operation'
    assert data['params_hash'] == 'abc123'
    assert data['status'] == 'completed'
    assert data['duration_ms'] == 100
    assert data['level'] == 'INFO'
    assert data['message'] == 'Test message'


def test_structured_json_formatter_with_exception():
    """Test JSON formatter with exception info"""
    formatter = StructuredJsonFormatter()

    # Create a log record with exception
    try:
        raise ValueError("Test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name='test_logger',
        level=logging.ERROR,
        pathname='test.py',
        lineno=10,
        msg='Error occurred',
        args=(),
        exc_info=exc_info
    )

    # Format the record
    output = formatter.format(record)
    data = json.loads(output)

    # Check error fields
    assert data['status'] == 'error'
    assert 'ValueError: Test error' in data['error']


def test_structured_logger():
    """Test StructuredLogger wrapper"""
    mock_logger = MagicMock()
    structured_logger = StructuredLogger(mock_logger)

    # Test log_operation with success
    structured_logger.log_operation(
        operation='test_op',
        params={'key': 'value'},
        status='completed',
        duration_ms=50,
        message='Operation completed'
    )

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    assert 'Operation completed' in call_args[0][0]

    # Test log_operation with error
    mock_logger.reset_mock()
    structured_logger.log_operation(
        operation='test_op',
        status='failed',
        error='Something went wrong'
    )

    mock_logger.error.assert_called_once()


def test_get_logger():
    """Test get_logger returns StructuredLogger instance"""
    logger = get_logger('test_module')

    assert isinstance(logger, StructuredLogger)
    assert hasattr(logger, 'log_operation')


def test_setup_logging(tmp_path):
    """Test logging setup with handlers"""
    # Save and clear existing handlers
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    for handler in previous_handlers:
        root.removeHandler(handler)

    try:
        with patch('data_collection.common.config.settings') as mock_settings:
            mock_settings.log_dir = str(tmp_path / 'logs')
            mock_settings.log_level = 'INFO'

            setup_logging()

            # Check root logger configuration
            assert root.level == logging.INFO

            # Check handlers were added
            handler_types = [type(h).__name__ for h in root.handlers]
            assert 'StreamHandler' in handler_types
            assert 'TimedRotatingFileHandler' in handler_types

            # Check audit logger was created
            audit_logger = logging.getLogger('audit')
            assert audit_logger.level == logging.INFO
            assert audit_logger.propagate is False

    finally:
        # Restore original handlers
        for handler in root.handlers:
            root.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)


def test_log_summary():
    """Test periodic summary logging"""
    with patch('data_collection.common.logging_setup.get_logger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        log_summary(
            component='test_component',
            shard='shard_001',
            records_processed=1000,
            duration_seconds=10.5
        )

        mock_logger.log_operation.assert_called_once_with(
            operation='shard_summary',
            params={'shard': 'shard_001'},
            status='completed',
            duration_ms=10500,
            message='Processed 1000 records in shard shard_001'
        )


def test_structured_logger_params_hash():
    """Test that params are hashed for privacy"""
    mock_logger = MagicMock()
    structured_logger = StructuredLogger(mock_logger)

    # Log with params containing sensitive data
    structured_logger.log_operation(
        operation='api_call',
        params={'api_key': 'secret123', 'endpoint': '/users'},
        status='started'
    )

    # Check that params were hashed, not exposed
    call_args = mock_logger.info.call_args
    extra = call_args[1]['extra']
    assert extra['params_hash'] != ''
    assert len(extra['params_hash']) == 8  # MD5 hash truncated to 8 chars
    assert 'secret123' not in str(extra)


def test_structured_logger_delegation():
    """Test that StructuredLogger delegates unknown methods to underlying logger"""
    mock_logger = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.warning = MagicMock()

    structured_logger = StructuredLogger(mock_logger)

    # These methods should be delegated
    structured_logger.debug("Debug message")
    structured_logger.warning("Warning message")

    mock_logger.debug.assert_called_once_with("Debug message")
    mock_logger.warning.assert_called_once_with("Warning message")


def test_json_formatter_removes_empty_fields():
    """Test that JSON formatter removes empty fields"""
    formatter = StructuredJsonFormatter()

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Test',
        args=(),
        exc_info=None
    )

    # Add fields with empty values
    record.component = 'test'
    record.operation = ''
    record.params_hash = ''
    record.status = 'info'
    record.duration_ms = 0
    record.error = ''

    output = formatter.format(record)
    data = json.loads(output)

    # Empty string fields should be removed
    assert 'operation' not in data or data['operation'] != ''
    assert 'params_hash' not in data or data['params_hash'] != ''
    assert 'error' not in data or data['error'] != ''

    # Non-empty fields should remain
    assert data['component'] == 'test'
    assert data['status'] == 'info'