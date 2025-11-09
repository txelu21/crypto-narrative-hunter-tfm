"""
Tests for comprehensive data exporter functionality.
"""

import pytest
import os
import json
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from services.export.data_exporter import ComprehensiveDataExporter


@pytest.fixture
def mock_db_connection():
    """Create mock database connection"""
    conn = Mock()

    # Mock execute method
    def mock_execute(query):
        result = Mock()
        if 'MIN(DATE' in query and 'transactions' in query:
            result.fetchone.return_value = {
                'min_date': datetime(2024, 1, 1),
                'max_date': datetime(2024, 3, 31),
                'total_count': 1000
            }
        elif 'MIN(DATE' in query and 'wallet_balances' in query:
            result.fetchone.return_value = {
                'min_date': datetime(2024, 1, 1),
                'max_date': datetime(2024, 2, 29),
                'total_count': 500
            }
        return result

    conn.execute = Mock(side_effect=mock_execute)
    return conn


@pytest.fixture
def temp_export_dir():
    """Create temporary directory for exports"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def exporter(mock_db_connection, temp_export_dir):
    """Create data exporter instance"""
    return ComprehensiveDataExporter(mock_db_connection, temp_export_dir)


def test_exporter_initialization(exporter, temp_export_dir):
    """Test that exporter initializes correctly with directory structure"""
    assert exporter.output_path == Path(temp_export_dir)
    assert exporter.export_timestamp is not None

    # Check that directories were created
    formats = ['parquet', 'csv', 'json']
    datasets = ['tokens', 'wallets', 'transactions', 'wallet_balances', 'eth_prices', 'wallet_performance']

    for fmt in formats:
        for dataset in datasets:
            expected_path = Path(temp_export_dir) / fmt / dataset
            assert expected_path.exists(), f"Directory not created: {expected_path}"


def test_export_tokens(exporter, mock_db_connection):
    """Test tokens table export"""
    # Mock pandas read_sql
    mock_df = pd.DataFrame({
        'token_address': ['0x123', '0x456'],
        'symbol': ['UNI', 'AAVE'],
        'name': ['Uniswap', 'Aave'],
        'decimals': [18, 18],
        'narrative_category': ['DeFi', 'DeFi'],
        'market_cap_rank': [10, 15],
        'avg_daily_volume_usd': [1000000.0, 500000.0],
        'liquidity_tier': ['Tier 1', 'Tier 1'],
        'created_at': [datetime.now(), datetime.now()],
        'updated_at': [datetime.now(), datetime.now()]
    })

    with patch('pandas.read_sql', return_value=mock_df):
        result = exporter.export_tokens({})

    assert result['record_count'] == 2
    assert 'parquet' in result['formats']
    assert 'csv' in result['formats']
    assert 'json' in result['formats']

    # Verify files were created
    parquet_path = Path(result['formats']['parquet'])
    csv_path = Path(result['formats']['csv'])
    json_path = Path(result['formats']['json'])

    assert parquet_path.exists()
    assert csv_path.exists()
    assert json_path.exists()


def test_export_wallets(exporter, mock_db_connection):
    """Test wallets table export"""
    mock_df = pd.DataFrame({
        'wallet_address': ['0xabc', '0xdef'],
        'first_seen': [datetime.now(), datetime.now()],
        'last_seen': [datetime.now(), datetime.now()],
        'total_transactions': [100, 200],
        'total_volume_eth': [50.5, 75.2],
        'unique_tokens': [10, 15],
        'avg_profit_per_trade': [0.5, 0.8],
        'total_gas_spent': [1.5, 2.0],
        'performance_score': [0.85, 0.92],
        'risk_score': [0.3, 0.25],
        'sophistication_tier': ['Expert', 'Advanced'],
        'created_at': [datetime.now(), datetime.now()]
    })

    with patch('pandas.read_sql', return_value=mock_df):
        result = exporter.export_wallets({})

    assert result['record_count'] == 2
    assert 'parquet' in result['formats']
    assert 'csv' in result['formats']
    assert 'json' in result['formats']

    # Verify JSON metadata
    json_path = Path(result['formats']['json'])
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    assert metadata['record_count'] == 2
    assert 'schema' in metadata
    assert 'sample_records' in metadata


def test_export_transactions_with_partitioning(exporter, mock_db_connection):
    """Test transactions export with monthly partitioning"""
    # Create mock dataframes for different months
    def mock_read_sql(query, conn, params=None):
        if params and len(params) == 2:
            start_date = params[0]
            # Return data for the month
            return pd.DataFrame({
                'tx_hash': ['0xhash1', '0xhash2'],
                'wallet_address': ['0xwallet1', '0xwallet2'],
                'token_address': ['0xtoken1', '0xtoken2'],
                'transaction_type': ['buy', 'sell'],
                'amount': [100.0, 200.0],
                'amount_usd': [1000.0, 2000.0],
                'timestamp': [start_date + timedelta(days=1), start_date + timedelta(days=2)],
                'block_number': [1000, 1001],
                'gas_used': [21000, 21000],
                'gas_price': [50, 55],
                'dex_protocol': ['Uniswap', 'Curve'],
                'slippage': [0.01, 0.02],
                'created_at': [datetime.now(), datetime.now()]
            })
        return pd.DataFrame()

    with patch('pandas.read_sql', side_effect=mock_read_sql):
        result = exporter.export_transactions({})

    assert result['record_count'] > 0
    assert len(result['partitions']) > 0
    assert 'parquet' in result['formats']
    assert 'csv' in result['formats']

    # Verify partitions were created
    parquet_dir = Path(result['formats']['parquet'])
    parquet_files = list(parquet_dir.glob('*.parquet'))
    assert len(parquet_files) > 0


def test_export_balances_with_partitioning(exporter, mock_db_connection):
    """Test wallet_balances export with monthly partitioning"""
    def mock_read_sql(query, conn, params=None):
        if params and len(params) == 2:
            start_date = params[0]
            return pd.DataFrame({
                'wallet_address': ['0xwallet1', '0xwallet2'],
                'token_address': ['0xtoken1', '0xtoken2'],
                'snapshot_date': [start_date + timedelta(days=1), start_date + timedelta(days=2)],
                'balance': [1000.0, 2000.0],
                'balance_usd': [10000.0, 20000.0],
                'price_at_snapshot': [10.0, 10.0],
                'created_at': [datetime.now(), datetime.now()]
            })
        return pd.DataFrame()

    with patch('pandas.read_sql', side_effect=mock_read_sql):
        result = exporter.export_balances({})

    assert result['record_count'] > 0
    assert len(result['partitions']) > 0
    assert 'parquet' in result['formats']
    assert 'csv' in result['formats']


def test_export_prices(exporter, mock_db_connection):
    """Test eth_prices table export"""
    mock_df = pd.DataFrame({
        'price_date': [datetime.now().date(), (datetime.now() - timedelta(days=1)).date()],
        'eth_price_usd': [2500.0, 2450.0],
        'source': ['Chainlink', 'Chainlink'],
        'created_at': [datetime.now(), datetime.now()]
    })

    with patch('pandas.read_sql', return_value=mock_df):
        result = exporter.export_prices({})

    assert result['record_count'] == 2
    assert 'parquet' in result['formats']
    assert 'csv' in result['formats']
    assert 'json' in result['formats']


def test_export_performance(exporter, mock_db_connection):
    """Test wallet_performance table export"""
    mock_df = pd.DataFrame({
        'wallet_address': ['0xwallet1', '0xwallet2'],
        'calculation_date': [datetime.now().date(), datetime.now().date()],
        'total_return': [0.25, 0.35],
        'sharpe_ratio': [1.5, 1.8],
        'win_rate': [0.65, 0.70],
        'max_drawdown': [-0.15, -0.12],
        'volatility': [0.25, 0.22],
        'total_trades': [100, 150],
        'avg_position_size': [0.1, 0.08],
        'portfolio_concentration': [0.3, 0.25],
        'created_at': [datetime.now(), datetime.now()]
    })

    with patch('pandas.read_sql', return_value=mock_df):
        result = exporter.export_performance({})

    assert result['record_count'] == 2
    assert 'parquet' in result['formats']
    assert 'csv' in result['formats']


def test_export_all_datasets(exporter, mock_db_connection):
    """Test comprehensive export of all datasets"""
    # Mock all export methods
    mock_result = {
        'record_count': 10,
        'formats': {
            'parquet': '/tmp/test.parquet',
            'csv': '/tmp/test.csv',
            'json': '/tmp/test.json'
        },
        'partitions': []
    }

    with patch.object(exporter, 'export_tokens', return_value=mock_result), \
         patch.object(exporter, 'export_wallets', return_value=mock_result), \
         patch.object(exporter, 'export_transactions', return_value=mock_result), \
         patch.object(exporter, 'export_balances', return_value=mock_result), \
         patch.object(exporter, 'export_prices', return_value=mock_result), \
         patch.object(exporter, 'export_performance', return_value=mock_result):

        manifest = exporter.export_all_datasets()

    assert manifest['export_version'] == '1.0'
    assert 'export_timestamp' in manifest
    assert len(manifest['datasets']) == 6
    assert manifest['total_records'] == 60  # 6 datasets * 10 records each


def test_calculate_file_checksum(exporter, temp_export_dir):
    """Test file checksum calculation"""
    # Create a test file
    test_file = Path(temp_export_dir) / 'test.txt'
    test_content = b'test content for checksum'
    test_file.write_bytes(test_content)

    checksum = exporter._calculate_file_checksum(str(test_file))

    assert isinstance(checksum, str)
    assert len(checksum) == 64  # SHA256 hex digest length


def test_validate_export_success(exporter):
    """Test export validation with successful export"""
    manifest = {
        'datasets': {
            'tokens': {
                'export_status': 'success',
                'record_count': 100,
                'formats': {
                    'parquet': str(exporter.output_path / 'parquet' / 'tokens' / 'tokens.parquet'),
                    'csv': str(exporter.output_path / 'csv' / 'tokens' / 'tokens.csv')
                }
            }
        }
    }

    # Create the files
    for fmt, path in manifest['datasets']['tokens']['formats'].items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    validation_results = exporter.validate_export(manifest)

    assert validation_results['overall_status'] == 'passed'
    assert 'tokens' in validation_results['dataset_validations']


def test_validate_export_failure(exporter):
    """Test export validation with missing files"""
    manifest = {
        'datasets': {
            'tokens': {
                'export_status': 'success',
                'record_count': 100,
                'formats': {
                    'parquet': str(exporter.output_path / 'parquet' / 'tokens' / 'missing.parquet')
                }
            }
        }
    }

    validation_results = exporter.validate_export(manifest)

    assert validation_results['overall_status'] == 'failed'
    assert len(validation_results['issues']) > 0


def test_export_with_empty_dataset(exporter, mock_db_connection):
    """Test export handling when dataset is empty"""
    empty_df = pd.DataFrame()

    with patch('pandas.read_sql', return_value=empty_df):
        result = exporter.export_tokens({})

    assert result['record_count'] == 0
    assert 'formats' in result


def test_export_manifest_creation(exporter, temp_export_dir):
    """Test that export manifest is created correctly"""
    with patch.object(exporter, 'export_tokens') as mock_tokens, \
         patch.object(exporter, 'export_wallets') as mock_wallets, \
         patch.object(exporter, 'export_transactions') as mock_transactions, \
         patch.object(exporter, 'export_balances') as mock_balances, \
         patch.object(exporter, 'export_prices') as mock_prices, \
         patch.object(exporter, 'export_performance') as mock_performance:

        # Configure mocks to return empty results
        mock_result = {'record_count': 0, 'formats': {}, 'partitions': []}
        mock_tokens.return_value = mock_result
        mock_wallets.return_value = mock_result
        mock_transactions.return_value = mock_result
        mock_balances.return_value = mock_result
        mock_prices.return_value = mock_result
        mock_performance.return_value = mock_result

        manifest = exporter.export_all_datasets()

    # Verify manifest file was created
    manifest_path = Path(temp_export_dir) / 'export_manifest.json'
    assert manifest_path.exists()

    # Verify manifest content
    with open(manifest_path, 'r') as f:
        saved_manifest = json.load(f)

    assert saved_manifest['export_version'] == '1.0'
    assert saved_manifest['project'] == 'Crypto Narrative Hunter'