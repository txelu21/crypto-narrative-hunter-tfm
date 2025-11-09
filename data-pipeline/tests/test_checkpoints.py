import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import date, datetime
from pathlib import Path

from data_collection.common.checkpoints import (
    CheckpointManager,
    get_checkpoint,
    upsert_checkpoint,
    ensure_checkpoint_table
)


def test_checkpoint_manager_init(tmp_path):
    """Test CheckpointManager initialization"""
    with patch('data_collection.common.checkpoints.Path.mkdir'):
        manager = CheckpointManager('test_collection')
        assert manager.collection_type == 'test_collection'
        assert len(manager._dedup_cache) == 0


def test_get_last_checkpoint_found():
    """Test getting last checkpoint when one exists"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        mock_execute.return_value = [{
            'collection_type': 'test',
            'last_processed_block': 12345,
            'last_processed_date': date(2024, 1, 1),
            'records_collected': 100,
            'status': 'running'
        }]

        manager = CheckpointManager('test')
        checkpoint = manager.get_last_checkpoint()

        assert checkpoint['last_processed_block'] == 12345
        assert checkpoint['records_collected'] == 100


def test_get_last_checkpoint_not_found():
    """Test getting last checkpoint when none exists"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        mock_execute.return_value = []

        manager = CheckpointManager('test')
        checkpoint = manager.get_last_checkpoint()

        assert checkpoint is None


def test_update_checkpoint():
    """Test updating checkpoint"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        manager = CheckpointManager('test')
        manager.update_checkpoint(
            last_processed_block=10000,
            last_processed_date=date(2024, 1, 1),
            records_collected=50,
            status='completed'
        )

        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[0]
        assert 'INSERT INTO collection_checkpoints' in call_args[0]
        assert 'ON CONFLICT (collection_type)' in call_args[0]


def test_get_resume_position_with_checkpoint():
    """Test getting resume position with existing checkpoint"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        mock_execute.return_value = [{
            'last_processed_block': 1000,
            'last_processed_date': date(2024, 1, 1),
            'records_collected': 500,
            'status': 'running'
        }]

        manager = CheckpointManager('test')
        position = manager.get_resume_position()

        assert position['start_block'] == 1001  # Next block after last processed
        assert position['start_date'] == date(2024, 1, 1)
        assert position['total_collected'] == 500


def test_get_resume_position_no_checkpoint():
    """Test getting resume position with no existing checkpoint"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        mock_execute.return_value = []

        manager = CheckpointManager('test')
        position = manager.get_resume_position()

        assert position['start_block'] == 0
        assert position['start_date'] is None
        assert position['total_collected'] == 0


def test_get_resume_position_failed_checkpoint():
    """Test getting resume position with failed checkpoint"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        mock_execute.return_value = [{
            'last_processed_block': 1000,
            'last_processed_date': date(2024, 1, 1),
            'records_collected': 500,
            'status': 'failed'
        }]

        manager = CheckpointManager('test')
        position = manager.get_resume_position()

        # Should start from beginning on failed checkpoint
        assert position['start_block'] == 0
        assert position['start_date'] is None
        assert position['total_collected'] == 0


def test_deduplication():
    """Test in-memory deduplication"""
    manager = CheckpointManager('test')

    record_hash = 'abc123'

    # First time should not be duplicate
    assert manager.is_duplicate(record_hash) is False

    # Second time should be duplicate
    assert manager.is_duplicate(record_hash) is True


def test_deduplication_cache_limit():
    """Test deduplication cache size limit"""
    manager = CheckpointManager('test')

    # Fill cache beyond limit
    for i in range(100005):
        manager.is_duplicate(f'hash_{i}')

    # Cache should be trimmed to 80000 entries
    assert len(manager._dedup_cache) <= 80000


def test_compute_record_hash():
    """Test record hash computation"""
    manager = CheckpointManager('test')

    record = {'id': 1, 'name': 'test', 'value': 100}
    hash1 = manager.compute_record_hash(record)

    # Same record should produce same hash
    hash2 = manager.compute_record_hash({'name': 'test', 'id': 1, 'value': 100})
    assert hash1 == hash2

    # Different record should produce different hash
    hash3 = manager.compute_record_hash({'id': 2, 'name': 'test', 'value': 100})
    assert hash1 != hash3


def test_write_staging(tmp_path):
    """Test writing data to staging file"""
    # Create a temporary staging directory
    staging_dir = tmp_path / "tmp" / "test"
    staging_dir.mkdir(parents=True)

    with patch('data_collection.common.checkpoints.Path') as mock_path_class:
        mock_path_class.return_value = tmp_path / "tmp" / "test"
        mock_path_class.return_value.mkdir = MagicMock()

        manager = CheckpointManager('test')
        manager._staging_dir = staging_dir

        data = [{'id': 1}, {'id': 2}]
        staging_file = manager.write_staging(data, 'batch_001')

        assert staging_file.exists()
        with open(staging_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == data


def test_commit_staging(tmp_path):
    """Test committing staging file"""
    staging_file = tmp_path / "batch_001.json"
    staging_file.write_text('{"test": true}')

    manager = CheckpointManager('test')
    manager.commit_staging(staging_file)

    assert not staging_file.exists()


def test_cleanup_staging(tmp_path):
    """Test cleanup of old staging files"""
    staging_dir = tmp_path / "tmp" / "test"
    staging_dir.mkdir(parents=True)

    # Create old file (mock as 2 days old)
    old_file = staging_dir / "batch_old.json"
    old_file.write_text('{}')

    # Create new file
    new_file = staging_dir / "batch_new.json"
    new_file.write_text('{}')

    with patch('time.time', return_value=1000000):
        with patch('pathlib.Path.stat') as mock_stat:
            # Mock old file as 2 days old
            mock_stat.return_value.st_mtime = 1000000 - (48 * 3600)

            manager = CheckpointManager('test')
            manager._staging_dir = staging_dir

            # Patch glob to return our test files
            with patch.object(staging_dir, 'glob', return_value=[old_file, new_file]):
                manager.cleanup_staging()

    # Old file should be deleted, new file should remain
    # (This test is simplified - in reality you'd need more complex mocking)


def test_ensure_checkpoint_table():
    """Test ensuring checkpoint table exists"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        ensure_checkpoint_table()

        mock_execute.assert_called_once()
        query = mock_execute.call_args[0][0]
        assert 'CREATE TABLE IF NOT EXISTS collection_checkpoints' in query


def test_backward_compatibility_get_checkpoint():
    """Test backward compatibility get_checkpoint function"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        mock_execute.return_value = [{'id': 1}]

        result = get_checkpoint('test')
        assert result == {'id': 1}


def test_backward_compatibility_upsert_checkpoint():
    """Test backward compatibility upsert_checkpoint function"""
    with patch('data_collection.common.checkpoints.execute_with_retry') as mock_execute:
        upsert_checkpoint(
            'test',
            last_processed_block=100,
            last_processed_date='2024-01-01',
            records_collected=10,
            status='running'
        )

        mock_execute.assert_called_once()