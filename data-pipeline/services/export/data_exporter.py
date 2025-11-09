"""
Comprehensive data exporter for multi-format exports with partitioning and validation.

Provides functionality to export all datasets (tokens, wallets, transactions, balances, prices)
in multiple formats (Parquet, CSV, JSON) with optional monthly partitioning for time-series data.
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ComprehensiveDataExporter:
    """Comprehensive data export system for all Crypto Narrative Hunter datasets"""

    def __init__(self, db_connection, output_path: str):
        """
        Initialize the comprehensive data exporter.

        Args:
            db_connection: Database connection object
            output_path: Base directory for exports
        """
        self.db = db_connection
        self.output_path = Path(output_path)
        self.export_timestamp = datetime.utcnow().isoformat()
        self.logger = logging.getLogger(__name__)

        # Create base export directories
        self._create_export_directories()

    def _create_export_directories(self):
        """Create directory structure for exports"""
        formats = ['parquet', 'csv', 'json']
        datasets = ['tokens', 'wallets', 'transactions', 'wallet_balances', 'eth_prices', 'wallet_performance']

        for fmt in formats:
            for dataset in datasets:
                dir_path = self.output_path / fmt / dataset
                dir_path.mkdir(parents=True, exist_ok=True)

    def export_all_datasets(self, export_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export all datasets in multiple formats.

        Args:
            export_config: Optional configuration dict for customizing exports

        Returns:
            Export manifest with metadata and file information
        """
        export_config = export_config or {}

        export_manifest = {
            'export_timestamp': self.export_timestamp,
            'export_version': '1.0',
            'project': 'Crypto Narrative Hunter',
            'datasets': {},
            'total_records': 0,
            'file_sizes': {},
            'checksums': {}
        }

        # Define datasets to export
        datasets = [
            ('tokens', self.export_tokens),
            ('wallets', self.export_wallets),
            ('transactions', self.export_transactions),
            ('wallet_balances', self.export_balances),
            ('eth_prices', self.export_prices),
            ('wallet_performance', self.export_performance)
        ]

        for dataset_name, export_function in datasets:
            self.logger.info(f"Exporting {dataset_name}...")

            try:
                export_result = export_function(export_config.get(dataset_name, {}))

                export_manifest['datasets'][dataset_name] = {
                    'record_count': export_result['record_count'],
                    'formats': export_result['formats'],
                    'partitions': export_result.get('partitions', []),
                    'export_status': 'success'
                }

                export_manifest['total_records'] += export_result['record_count']

                # Calculate file sizes and checksums
                for fmt, file_path in export_result['formats'].items():
                    if isinstance(file_path, str) and os.path.exists(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        checksum = self._calculate_file_checksum(file_path)

                        export_manifest['file_sizes'][f"{dataset_name}_{fmt}"] = f"{size_mb:.2f} MB"
                        export_manifest['checksums'][f"{dataset_name}_{fmt}"] = checksum
                    elif isinstance(file_path, str):
                        # Directory path - calculate total size
                        total_size = sum(f.stat().st_size for f in Path(file_path).rglob('*') if f.is_file())
                        size_mb = total_size / (1024 * 1024)
                        export_manifest['file_sizes'][f"{dataset_name}_{fmt}"] = f"{size_mb:.2f} MB"

            except Exception as e:
                self.logger.error(f"Failed to export {dataset_name}: {e}")
                export_manifest['datasets'][dataset_name] = {
                    'export_status': 'failed',
                    'error': str(e)
                }

        # Save export manifest
        manifest_path = self.output_path / 'export_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(export_manifest, f, indent=2)

        self.logger.info(f"Export completed. Manifest saved to {manifest_path}")
        return export_manifest

    def export_tokens(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export tokens table in all formats"""
        self.logger.info("Exporting tokens dataset...")

        query = """
            SELECT token_address, symbol, name, decimals, narrative_category,
                   market_cap_rank, avg_daily_volume_usd, liquidity_tier,
                   created_at, updated_at
            FROM tokens
            ORDER BY market_cap_rank NULLS LAST
        """

        df = pd.read_sql(query, self.db)
        record_count = len(df)

        # Parquet export
        parquet_path = self.output_path / 'parquet' / 'tokens' / 'tokens.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)

        # CSV export
        csv_path = self.output_path / 'csv' / 'tokens' / 'tokens.csv'
        df.to_csv(csv_path, index=False)

        # JSON export (metadata and sample)
        json_path = self.output_path / 'json' / 'tokens' / 'tokens_metadata.json'
        metadata = {
            'record_count': record_count,
            'export_timestamp': self.export_timestamp,
            'schema': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_records': df.head(10).to_dict(orient='records')
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Exported {record_count} tokens")

        return {
            'record_count': record_count,
            'formats': {
                'parquet': str(parquet_path),
                'csv': str(csv_path),
                'json': str(json_path)
            }
        }

    def export_wallets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export wallets table in all formats"""
        self.logger.info("Exporting wallets dataset...")

        query = """
            SELECT wallet_address, first_seen, last_seen, total_transactions,
                   total_volume_eth, unique_tokens, avg_profit_per_trade,
                   total_gas_spent, performance_score, risk_score,
                   sophistication_tier, created_at
            FROM wallets
            ORDER BY performance_score DESC NULLS LAST
        """

        df = pd.read_sql(query, self.db)
        record_count = len(df)

        # Parquet export
        parquet_path = self.output_path / 'parquet' / 'wallets' / 'wallets.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)

        # CSV export
        csv_path = self.output_path / 'csv' / 'wallets' / 'wallets.csv'
        df.to_csv(csv_path, index=False)

        # JSON export (metadata and sample)
        json_path = self.output_path / 'json' / 'wallets' / 'wallets_metadata.json'
        metadata = {
            'record_count': record_count,
            'export_timestamp': self.export_timestamp,
            'schema': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_records': df.head(10).to_dict(orient='records')
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Exported {record_count} wallets")

        return {
            'record_count': record_count,
            'formats': {
                'parquet': str(parquet_path),
                'csv': str(csv_path),
                'json': str(json_path)
            }
        }

    def export_transactions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export transactions with monthly partitioning"""
        self.logger.info("Exporting transactions dataset with monthly partitioning...")

        # Get date range for partitioning
        date_query = """
            SELECT MIN(DATE(timestamp)) as min_date,
                   MAX(DATE(timestamp)) as max_date,
                   COUNT(*) as total_count
            FROM transactions
        """
        result = self.db.execute(date_query).fetchone()

        if result and result['total_count'] > 0:
            start_date = result['min_date']
            end_date = result['max_date']
            total_records = result['total_count']
        else:
            self.logger.warning("No transactions found to export")
            return {
                'record_count': 0,
                'formats': {},
                'partitions': []
            }

        export_result = {
            'record_count': 0,
            'formats': {},
            'partitions': []
        }

        # Export by month for optimal partitioning
        current_date = start_date.replace(day=1) if hasattr(start_date, 'replace') else datetime.strptime(str(start_date), '%Y-%m-%d').replace(day=1)
        end_date_dt = end_date if hasattr(end_date, 'replace') else datetime.strptime(str(end_date), '%Y-%m-%d')

        while current_date <= end_date_dt:
            next_month = (current_date + timedelta(days=32)).replace(day=1)

            # Query transactions for this month
            month_query = """
                SELECT tx_hash, wallet_address, token_address, transaction_type,
                       amount, amount_usd, timestamp, block_number, gas_used,
                       gas_price, dex_protocol, slippage, created_at
                FROM transactions
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """

            df = pd.read_sql(month_query, self.db, params=[current_date, next_month])

            if len(df) > 0:
                month_str = current_date.strftime('%Y_%m')

                # Parquet export (primary format)
                parquet_dir = self.output_path / 'parquet' / 'transactions'
                parquet_path = parquet_dir / f'transactions_{month_str}.parquet'
                df.to_parquet(parquet_path, compression='snappy', index=False)

                # CSV export (for readability)
                csv_dir = self.output_path / 'csv' / 'transactions'
                csv_path = csv_dir / f'transactions_{month_str}.csv'
                df.to_csv(csv_path, index=False)

                export_result['record_count'] += len(df)
                export_result['partitions'].append(month_str)

                self.logger.info(f"Exported {len(df)} transactions for {month_str}")

            current_date = next_month

        # Set format paths (use directory as reference)
        export_result['formats'] = {
            'parquet': str(self.output_path / 'parquet' / 'transactions'),
            'csv': str(self.output_path / 'csv' / 'transactions')
        }

        # Create JSON metadata
        json_path = self.output_path / 'json' / 'transactions' / 'transactions_metadata.json'
        metadata = {
            'record_count': export_result['record_count'],
            'export_timestamp': self.export_timestamp,
            'partitions': export_result['partitions'],
            'date_range': {
                'start': str(start_date),
                'end': str(end_date)
            }
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        export_result['formats']['json'] = str(json_path)

        self.logger.info(f"Exported {export_result['record_count']} transactions across {len(export_result['partitions'])} partitions")

        return export_result

    def export_balances(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export wallet_balances with monthly partitioning"""
        self.logger.info("Exporting wallet_balances dataset with monthly partitioning...")

        # Get date range for partitioning
        date_query = """
            SELECT MIN(DATE(snapshot_date)) as min_date,
                   MAX(DATE(snapshot_date)) as max_date,
                   COUNT(*) as total_count
            FROM wallet_balances
        """
        result = self.db.execute(date_query).fetchone()

        if result and result['total_count'] > 0:
            start_date = result['min_date']
            end_date = result['max_date']
            total_records = result['total_count']
        else:
            self.logger.warning("No wallet balances found to export")
            return {
                'record_count': 0,
                'formats': {},
                'partitions': []
            }

        export_result = {
            'record_count': 0,
            'formats': {},
            'partitions': []
        }

        # Export by month
        current_date = start_date.replace(day=1) if hasattr(start_date, 'replace') else datetime.strptime(str(start_date), '%Y-%m-%d').replace(day=1)
        end_date_dt = end_date if hasattr(end_date, 'replace') else datetime.strptime(str(end_date), '%Y-%m-%d')

        while current_date <= end_date_dt:
            next_month = (current_date + timedelta(days=32)).replace(day=1)

            # Query balances for this month
            month_query = """
                SELECT wallet_address, token_address, snapshot_date,
                       balance, balance_usd, price_at_snapshot, created_at
                FROM wallet_balances
                WHERE snapshot_date >= ? AND snapshot_date < ?
                ORDER BY snapshot_date
            """

            df = pd.read_sql(month_query, self.db, params=[current_date, next_month])

            if len(df) > 0:
                month_str = current_date.strftime('%Y_%m')

                # Parquet export
                parquet_dir = self.output_path / 'parquet' / 'wallet_balances'
                parquet_path = parquet_dir / f'wallet_balances_{month_str}.parquet'
                df.to_parquet(parquet_path, compression='snappy', index=False)

                # CSV export
                csv_dir = self.output_path / 'csv' / 'wallet_balances'
                csv_path = csv_dir / f'wallet_balances_{month_str}.csv'
                df.to_csv(csv_path, index=False)

                export_result['record_count'] += len(df)
                export_result['partitions'].append(month_str)

                self.logger.info(f"Exported {len(df)} balance records for {month_str}")

            current_date = next_month

        # Set format paths
        export_result['formats'] = {
            'parquet': str(self.output_path / 'parquet' / 'wallet_balances'),
            'csv': str(self.output_path / 'csv' / 'wallet_balances')
        }

        # Create JSON metadata
        json_path = self.output_path / 'json' / 'wallet_balances' / 'wallet_balances_metadata.json'
        metadata = {
            'record_count': export_result['record_count'],
            'export_timestamp': self.export_timestamp,
            'partitions': export_result['partitions'],
            'date_range': {
                'start': str(start_date),
                'end': str(end_date)
            }
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        export_result['formats']['json'] = str(json_path)

        self.logger.info(f"Exported {export_result['record_count']} balance records across {len(export_result['partitions'])} partitions")

        return export_result

    def export_prices(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export eth_prices table in all formats"""
        self.logger.info("Exporting eth_prices dataset...")

        query = """
            SELECT price_date, eth_price_usd, source, created_at
            FROM eth_prices
            ORDER BY price_date
        """

        df = pd.read_sql(query, self.db)
        record_count = len(df)

        # Parquet export
        parquet_path = self.output_path / 'parquet' / 'eth_prices' / 'eth_prices.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)

        # CSV export
        csv_path = self.output_path / 'csv' / 'eth_prices' / 'eth_prices.csv'
        df.to_csv(csv_path, index=False)

        # JSON export (metadata and sample)
        json_path = self.output_path / 'json' / 'eth_prices' / 'eth_prices_metadata.json'
        metadata = {
            'record_count': record_count,
            'export_timestamp': self.export_timestamp,
            'schema': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_records': df.head(10).to_dict(orient='records')
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Exported {record_count} price records")

        return {
            'record_count': record_count,
            'formats': {
                'parquet': str(parquet_path),
                'csv': str(csv_path),
                'json': str(json_path)
            }
        }

    def export_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export wallet_performance table in all formats"""
        self.logger.info("Exporting wallet_performance dataset...")

        query = """
            SELECT wallet_address, calculation_date, total_return, sharpe_ratio,
                   win_rate, max_drawdown, volatility, total_trades,
                   avg_position_size, portfolio_concentration, created_at
            FROM wallet_performance
            ORDER BY calculation_date, performance_score DESC NULLS LAST
        """

        df = pd.read_sql(query, self.db)
        record_count = len(df)

        # Parquet export
        parquet_path = self.output_path / 'parquet' / 'wallet_performance' / 'wallet_performance.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)

        # CSV export
        csv_path = self.output_path / 'csv' / 'wallet_performance' / 'wallet_performance.csv'
        df.to_csv(csv_path, index=False)

        # JSON export (metadata and sample)
        json_path = self.output_path / 'json' / 'wallet_performance' / 'wallet_performance_metadata.json'
        metadata = {
            'record_count': record_count,
            'export_timestamp': self.export_timestamp,
            'schema': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_records': df.head(10).to_dict(orient='records')
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Exported {record_count} performance records")

        return {
            'record_count': record_count,
            'formats': {
                'parquet': str(parquet_path),
                'csv': str(csv_path),
                'json': str(json_path)
            }
        }

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum for a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def validate_export(self, export_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate export integrity and completeness.

        Args:
            export_manifest: Export manifest from export_all_datasets

        Returns:
            Validation results dictionary
        """
        self.logger.info("Validating export integrity...")

        validation_results = {
            'validation_timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'passed',
            'dataset_validations': {},
            'issues': []
        }

        for dataset_name, dataset_info in export_manifest.get('datasets', {}).items():
            if dataset_info.get('export_status') != 'success':
                validation_results['issues'].append(f"{dataset_name}: Export failed")
                validation_results['overall_status'] = 'failed'
                continue

            dataset_validation = {
                'status': 'passed',
                'checks': []
            }

            # Verify files exist
            for fmt, path in dataset_info.get('formats', {}).items():
                if isinstance(path, str):
                    path_obj = Path(path)
                    if path_obj.exists():
                        dataset_validation['checks'].append(f"{fmt}: File exists")
                    else:
                        dataset_validation['checks'].append(f"{fmt}: File missing")
                        dataset_validation['status'] = 'failed'
                        validation_results['issues'].append(f"{dataset_name}/{fmt}: File not found")

            # Verify record counts
            record_count = dataset_info.get('record_count', 0)
            if record_count > 0:
                dataset_validation['checks'].append(f"Record count: {record_count}")
            else:
                dataset_validation['checks'].append("Warning: Zero records")

            validation_results['dataset_validations'][dataset_name] = dataset_validation

        if len(validation_results['issues']) > 0:
            validation_results['overall_status'] = 'failed'

        self.logger.info(f"Export validation completed: {validation_results['overall_status']}")

        return validation_results