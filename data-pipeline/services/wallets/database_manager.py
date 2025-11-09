"""
Database management module for wallet performance metrics storage.

This module provides functionality to store and retrieve wallet performance
metrics with proper indexing, data validation, and batch processing capabilities.
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_values
from contextlib import contextmanager
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal
import json
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database operations for wallet performance metrics.

    Provides methods for creating tables, inserting/updating performance
    metrics, querying data, and maintaining data integrity with proper
    indexing and validation.
    """

    def __init__(
        self,
        db_type: str = "sqlite",
        connection_params: Optional[Dict[str, Any]] = None,
        db_path: Optional[str] = None
    ):
        """
        Initialize the DatabaseManager.

        Args:
            db_type: Database type ("sqlite" or "postgresql")
            connection_params: Database connection parameters for PostgreSQL
            db_path: Path to SQLite database file
        """
        self.db_type = db_type
        self.connection_params = connection_params or {}
        self.db_path = db_path or "wallet_performance.db"

        # Initialize database schema
        self._create_tables()

    @contextmanager
    def get_connection(self):
        """
        Get database connection with proper cleanup.

        Yields:
            Database connection object
        """
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
        elif self.db_type == "postgresql":
            conn = psycopg2.connect(**self.connection_params)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self):
        """Create necessary database tables with proper schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Wallet performance metrics table
            cursor.execute(self._get_wallet_performance_schema())

            # Wallet performance history table for time-series data
            cursor.execute(self._get_performance_history_schema())

            # Portfolio snapshots table
            cursor.execute(self._get_portfolio_snapshots_schema())

            # Performance benchmarks table
            cursor.execute(self._get_benchmarks_schema())

            # Create indexes
            self._create_indexes(cursor)

            conn.commit()

    def _get_wallet_performance_schema(self) -> str:
        """Get the wallet performance table schema."""
        if self.db_type == "sqlite":
            return """
            CREATE TABLE IF NOT EXISTS wallet_performance (
                wallet_address TEXT NOT NULL,
                calculation_date DATE NOT NULL,
                time_period TEXT NOT NULL,

                -- Basic performance metrics
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_return_per_trade REAL DEFAULT 0,
                total_return REAL DEFAULT 0,
                annualized_return REAL DEFAULT 0,

                -- Risk metrics
                volatility REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                sortino_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                var_95 REAL DEFAULT 0,
                calmar_ratio REAL DEFAULT 0,

                -- Efficiency metrics
                total_gas_cost_usd REAL DEFAULT 0,
                volume_per_gas REAL DEFAULT 0,
                net_return_after_costs REAL DEFAULT 0,

                -- Diversification metrics
                unique_tokens_traded INTEGER DEFAULT 0,
                hhi_concentration REAL DEFAULT 0,
                max_position_size REAL DEFAULT 0,
                effective_tokens REAL DEFAULT 0,
                diversification_score REAL DEFAULT 0,

                -- Time-series metrics
                consistency_score REAL DEFAULT 0,
                positive_days_pct REAL DEFAULT 0,
                time_weighted_return REAL DEFAULT 0,

                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (wallet_address, calculation_date, time_period)
            )
            """
        else:  # PostgreSQL
            return """
            CREATE TABLE IF NOT EXISTS wallet_performance (
                wallet_address VARCHAR(42) NOT NULL,
                calculation_date DATE NOT NULL,
                time_period VARCHAR(10) NOT NULL,

                -- Basic performance metrics
                total_trades INTEGER DEFAULT 0,
                win_rate DECIMAL(5,2) DEFAULT 0,
                avg_return_per_trade DECIMAL(10,6) DEFAULT 0,
                total_return DECIMAL(10,6) DEFAULT 0,
                annualized_return DECIMAL(10,6) DEFAULT 0,

                -- Risk metrics
                volatility DECIMAL(10,6) DEFAULT 0,
                sharpe_ratio DECIMAL(10,6) DEFAULT 0,
                sortino_ratio DECIMAL(10,6) DEFAULT 0,
                max_drawdown DECIMAL(10,6) DEFAULT 0,
                var_95 DECIMAL(10,6) DEFAULT 0,
                calmar_ratio DECIMAL(10,6) DEFAULT 0,

                -- Efficiency metrics
                total_gas_cost_usd DECIMAL(15,2) DEFAULT 0,
                volume_per_gas DECIMAL(15,2) DEFAULT 0,
                net_return_after_costs DECIMAL(10,6) DEFAULT 0,

                -- Diversification metrics
                unique_tokens_traded INTEGER DEFAULT 0,
                hhi_concentration DECIMAL(8,6) DEFAULT 0,
                max_position_size DECIMAL(5,2) DEFAULT 0,
                effective_tokens DECIMAL(8,2) DEFAULT 0,
                diversification_score DECIMAL(5,2) DEFAULT 0,

                -- Time-series metrics
                consistency_score DECIMAL(8,4) DEFAULT 0,
                positive_days_pct DECIMAL(5,2) DEFAULT 0,
                time_weighted_return DECIMAL(10,6) DEFAULT 0,

                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (wallet_address, calculation_date, time_period)
            )
            """

    def _get_performance_history_schema(self) -> str:
        """Get the performance history table schema for time-series data."""
        if self.db_type == "sqlite":
            return """
            CREATE TABLE IF NOT EXISTS performance_history (
                wallet_address TEXT NOT NULL,
                date DATE NOT NULL,
                portfolio_value REAL DEFAULT 0,
                daily_return REAL DEFAULT 0,
                cumulative_return REAL DEFAULT 0,
                rolling_7d_return REAL DEFAULT 0,
                rolling_30d_return REAL DEFAULT 0,
                rolling_volatility REAL DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (wallet_address, date)
            )
            """
        else:  # PostgreSQL
            return """
            CREATE TABLE IF NOT EXISTS performance_history (
                wallet_address VARCHAR(42) NOT NULL,
                date DATE NOT NULL,
                portfolio_value DECIMAL(20,6) DEFAULT 0,
                daily_return DECIMAL(10,6) DEFAULT 0,
                cumulative_return DECIMAL(10,6) DEFAULT 0,
                rolling_7d_return DECIMAL(10,6) DEFAULT 0,
                rolling_30d_return DECIMAL(10,6) DEFAULT 0,
                rolling_volatility DECIMAL(10,6) DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (wallet_address, date)
            )
            """

    def _get_portfolio_snapshots_schema(self) -> str:
        """Get the portfolio snapshots table schema."""
        if self.db_type == "sqlite":
            return """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                wallet_address TEXT NOT NULL,
                snapshot_date DATE NOT NULL,
                token_symbol TEXT NOT NULL,
                amount REAL DEFAULT 0,
                value_usd REAL DEFAULT 0,
                weight REAL DEFAULT 0,
                narrative_category TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (wallet_address, snapshot_date, token_symbol)
            )
            """
        else:  # PostgreSQL
            return """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                wallet_address VARCHAR(42) NOT NULL,
                snapshot_date DATE NOT NULL,
                token_symbol VARCHAR(20) NOT NULL,
                amount DECIMAL(20,8) DEFAULT 0,
                value_usd DECIMAL(15,2) DEFAULT 0,
                weight DECIMAL(5,4) DEFAULT 0,
                narrative_category VARCHAR(50),

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (wallet_address, snapshot_date, token_symbol)
            )
            """

    def _get_benchmarks_schema(self) -> str:
        """Get the benchmarks table schema."""
        if self.db_type == "sqlite":
            return """
            CREATE TABLE IF NOT EXISTS benchmarks (
                benchmark_name TEXT NOT NULL,
                date DATE NOT NULL,
                value REAL DEFAULT 0,
                daily_return REAL DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (benchmark_name, date)
            )
            """
        else:  # PostgreSQL
            return """
            CREATE TABLE IF NOT EXISTS benchmarks (
                benchmark_name VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                value DECIMAL(15,6) DEFAULT 0,
                daily_return DECIMAL(10,6) DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (benchmark_name, date)
            )
            """

    def _create_indexes(self, cursor):
        """Create database indexes for performance optimization."""
        indexes = [
            # Wallet performance indexes
            "CREATE INDEX IF NOT EXISTS idx_wallet_performance_return ON wallet_performance(total_return DESC)",
            "CREATE INDEX IF NOT EXISTS idx_wallet_performance_sharpe ON wallet_performance(sharpe_ratio DESC)",
            "CREATE INDEX IF NOT EXISTS idx_wallet_performance_date ON wallet_performance(calculation_date)",
            "CREATE INDEX IF NOT EXISTS idx_wallet_performance_period ON wallet_performance(time_period)",

            # Performance history indexes
            "CREATE INDEX IF NOT EXISTS idx_performance_history_date ON performance_history(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_performance_history_wallet ON performance_history(wallet_address)",

            # Portfolio snapshots indexes
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_token ON portfolio_snapshots(token_symbol)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_narrative ON portfolio_snapshots(narrative_category)",

            # Benchmarks indexes
            "CREATE INDEX IF NOT EXISTS idx_benchmarks_date ON benchmarks(date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON benchmarks(benchmark_name)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

    def upsert_wallet_performance(
        self,
        wallet_address: str,
        performance_data: Dict[str, Any],
        time_period: str = "all_time"
    ) -> bool:
        """
        Insert or update wallet performance metrics.

        Args:
            wallet_address: Wallet address
            performance_data: Dictionary containing performance metrics
            time_period: Time period for the metrics

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Validate data
                validated_data = self._validate_performance_data(performance_data)

                if self.db_type == "sqlite":
                    sql = """
                    INSERT OR REPLACE INTO wallet_performance (
                        wallet_address, calculation_date, time_period,
                        total_trades, win_rate, avg_return_per_trade, total_return,
                        annualized_return, volatility, sharpe_ratio, sortino_ratio,
                        max_drawdown, var_95, calmar_ratio, total_gas_cost_usd,
                        volume_per_gas, net_return_after_costs, unique_tokens_traded,
                        hhi_concentration, max_position_size, effective_tokens,
                        diversification_score, consistency_score, positive_days_pct,
                        time_weighted_return, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                else:  # PostgreSQL
                    sql = """
                    INSERT INTO wallet_performance (
                        wallet_address, calculation_date, time_period,
                        total_trades, win_rate, avg_return_per_trade, total_return,
                        annualized_return, volatility, sharpe_ratio, sortino_ratio,
                        max_drawdown, var_95, calmar_ratio, total_gas_cost_usd,
                        volume_per_gas, net_return_after_costs, unique_tokens_traded,
                        hhi_concentration, max_position_size, effective_tokens,
                        diversification_score, consistency_score, positive_days_pct,
                        time_weighted_return, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (wallet_address, calculation_date, time_period)
                    DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        win_rate = EXCLUDED.win_rate,
                        avg_return_per_trade = EXCLUDED.avg_return_per_trade,
                        total_return = EXCLUDED.total_return,
                        annualized_return = EXCLUDED.annualized_return,
                        volatility = EXCLUDED.volatility,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        sortino_ratio = EXCLUDED.sortino_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        var_95 = EXCLUDED.var_95,
                        calmar_ratio = EXCLUDED.calmar_ratio,
                        total_gas_cost_usd = EXCLUDED.total_gas_cost_usd,
                        volume_per_gas = EXCLUDED.volume_per_gas,
                        net_return_after_costs = EXCLUDED.net_return_after_costs,
                        unique_tokens_traded = EXCLUDED.unique_tokens_traded,
                        hhi_concentration = EXCLUDED.hhi_concentration,
                        max_position_size = EXCLUDED.max_position_size,
                        effective_tokens = EXCLUDED.effective_tokens,
                        diversification_score = EXCLUDED.diversification_score,
                        consistency_score = EXCLUDED.consistency_score,
                        positive_days_pct = EXCLUDED.positive_days_pct,
                        time_weighted_return = EXCLUDED.time_weighted_return,
                        updated_at = EXCLUDED.updated_at
                    """

                values = (
                    wallet_address,
                    validated_data.get('calculation_date', date.today()),
                    time_period,
                    validated_data.get('total_trades', 0),
                    validated_data.get('win_rate', 0),
                    validated_data.get('avg_return_per_trade', 0),
                    validated_data.get('total_return', 0),
                    validated_data.get('annualized_return', 0),
                    validated_data.get('volatility', 0),
                    validated_data.get('sharpe_ratio', 0),
                    validated_data.get('sortino_ratio', 0),
                    validated_data.get('max_drawdown', 0),
                    validated_data.get('var_95', 0),
                    validated_data.get('calmar_ratio', 0),
                    validated_data.get('total_gas_cost_usd', 0),
                    validated_data.get('volume_per_gas', 0),
                    validated_data.get('net_return_after_costs', 0),
                    validated_data.get('unique_tokens_traded', 0),
                    validated_data.get('hhi_concentration', 0),
                    validated_data.get('max_position_size', 0),
                    validated_data.get('effective_tokens', 0),
                    validated_data.get('diversification_score', 0),
                    validated_data.get('consistency_score', 0),
                    validated_data.get('positive_days_pct', 0),
                    validated_data.get('time_weighted_return', 0),
                    datetime.now()
                )

                cursor.execute(sql, values)
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error upserting wallet performance for {wallet_address}: {e}")
            return False

    def batch_upsert_performance_history(
        self,
        wallet_address: str,
        history_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Batch insert performance history data for efficiency.

        Args:
            wallet_address: Wallet address
            history_data: List of daily performance data

        Returns:
            True if successful, False otherwise
        """
        if not history_data:
            return True

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if self.db_type == "sqlite":
                    sql = """
                    INSERT OR REPLACE INTO performance_history (
                        wallet_address, date, portfolio_value, daily_return,
                        cumulative_return, rolling_7d_return, rolling_30d_return,
                        rolling_volatility
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """

                    values = [
                        (
                            wallet_address,
                            record['date'],
                            record.get('portfolio_value', 0),
                            record.get('daily_return', 0),
                            record.get('cumulative_return', 0),
                            record.get('rolling_7d_return', 0),
                            record.get('rolling_30d_return', 0),
                            record.get('rolling_volatility', 0)
                        )
                        for record in history_data
                    ]

                    cursor.executemany(sql, values)

                else:  # PostgreSQL
                    sql = """
                    INSERT INTO performance_history (
                        wallet_address, date, portfolio_value, daily_return,
                        cumulative_return, rolling_7d_return, rolling_30d_return,
                        rolling_volatility
                    ) VALUES %s
                    ON CONFLICT (wallet_address, date) DO UPDATE SET
                        portfolio_value = EXCLUDED.portfolio_value,
                        daily_return = EXCLUDED.daily_return,
                        cumulative_return = EXCLUDED.cumulative_return,
                        rolling_7d_return = EXCLUDED.rolling_7d_return,
                        rolling_30d_return = EXCLUDED.rolling_30d_return,
                        rolling_volatility = EXCLUDED.rolling_volatility
                    """

                    values = [
                        (
                            wallet_address,
                            record['date'],
                            record.get('portfolio_value', 0),
                            record.get('daily_return', 0),
                            record.get('cumulative_return', 0),
                            record.get('rolling_7d_return', 0),
                            record.get('rolling_30d_return', 0),
                            record.get('rolling_volatility', 0)
                        )
                        for record in history_data
                    ]

                    execute_values(cursor, sql, values)

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error batch upserting performance history for {wallet_address}: {e}")
            return False

    def store_portfolio_snapshot(
        self,
        wallet_address: str,
        snapshot_date: date,
        portfolio_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Store portfolio snapshot for point-in-time analysis.

        Args:
            wallet_address: Wallet address
            snapshot_date: Date of the snapshot
            portfolio_data: List of token holdings

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # First, delete existing snapshot for this date
                delete_sql = f"DELETE FROM portfolio_snapshots WHERE wallet_address = {'?' if self.db_type == 'sqlite' else '%s'} AND snapshot_date = {'?' if self.db_type == 'sqlite' else '%s'}"
                cursor.execute(delete_sql, (wallet_address, snapshot_date))

                # Insert new snapshot data
                if self.db_type == "sqlite":
                    sql = """
                    INSERT INTO portfolio_snapshots (
                        wallet_address, snapshot_date, token_symbol, amount,
                        value_usd, weight, narrative_category
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """

                    values = [
                        (
                            wallet_address,
                            snapshot_date,
                            holding['token_symbol'],
                            holding.get('amount', 0),
                            holding.get('value_usd', 0),
                            holding.get('weight', 0),
                            holding.get('narrative_category')
                        )
                        for holding in portfolio_data
                    ]

                    cursor.executemany(sql, values)

                else:  # PostgreSQL
                    sql = """
                    INSERT INTO portfolio_snapshots (
                        wallet_address, snapshot_date, token_symbol, amount,
                        value_usd, weight, narrative_category
                    ) VALUES %s
                    """

                    values = [
                        (
                            wallet_address,
                            snapshot_date,
                            holding['token_symbol'],
                            holding.get('amount', 0),
                            holding.get('value_usd', 0),
                            holding.get('weight', 0),
                            holding.get('narrative_category')
                        )
                        for holding in portfolio_data
                    ]

                    execute_values(cursor, sql, values)

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing portfolio snapshot for {wallet_address}: {e}")
            return False

    def get_wallet_performance(
        self,
        wallet_address: str,
        time_period: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve wallet performance metrics.

        Args:
            wallet_address: Wallet address
            time_period: Specific time period filter
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Performance metrics dictionary or None
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                where_conditions = ["wallet_address = ?"] if self.db_type == "sqlite" else ["wallet_address = %s"]
                params = [wallet_address]

                if time_period:
                    where_conditions.append("time_period = ?") if self.db_type == "sqlite" else where_conditions.append("time_period = %s")
                    params.append(time_period)

                if start_date:
                    where_conditions.append("calculation_date >= ?") if self.db_type == "sqlite" else where_conditions.append("calculation_date >= %s")
                    params.append(start_date)

                if end_date:
                    where_conditions.append("calculation_date <= ?") if self.db_type == "sqlite" else where_conditions.append("calculation_date <= %s")
                    params.append(end_date)

                sql = f"""
                SELECT * FROM wallet_performance
                WHERE {' AND '.join(where_conditions)}
                ORDER BY calculation_date DESC
                LIMIT 1
                """

                cursor.execute(sql, params)
                result = cursor.fetchone()

                if result:
                    return dict(result) if self.db_type == "sqlite" else dict(zip([desc[0] for desc in cursor.description], result))

                return None

        except Exception as e:
            logger.error(f"Error retrieving wallet performance for {wallet_address}: {e}")
            return None

    def get_top_performers(
        self,
        metric: str = "total_return",
        time_period: str = "all_time",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get top performing wallets by specified metric.

        Args:
            metric: Performance metric to rank by
            time_period: Time period filter
            limit: Maximum number of results

        Returns:
            List of top performer records
        """
        # Validate metric name to prevent SQL injection
        allowed_metrics = [
            'total_return', 'sharpe_ratio', 'win_rate', 'diversification_score',
            'volume_per_gas', 'consistency_score'
        ]

        if metric not in allowed_metrics:
            raise ValueError(f"Invalid metric: {metric}")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                param_placeholder = "?" if self.db_type == "sqlite" else "%s"

                sql = f"""
                SELECT * FROM wallet_performance
                WHERE time_period = {param_placeholder}
                ORDER BY {metric} DESC
                LIMIT {limit}
                """

                cursor.execute(sql, (time_period,))
                results = cursor.fetchall()

                if self.db_type == "sqlite":
                    return [dict(row) for row in results]
                else:
                    return [dict(zip([desc[0] for desc in cursor.description], row)) for row in results]

        except Exception as e:
            logger.error(f"Error retrieving top performers: {e}")
            return []

    def _validate_performance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize performance data.

        Args:
            data: Raw performance data

        Returns:
            Validated data dictionary
        """
        validated = {}

        # Numeric fields with validation
        numeric_fields = {
            'total_trades': (int, 0, 1000000),
            'win_rate': (float, 0, 1),
            'avg_return_per_trade': (float, -10, 10),
            'total_return': (float, -1, 100),
            'annualized_return': (float, -1, 10),
            'volatility': (float, 0, 10),
            'sharpe_ratio': (float, -10, 10),
            'sortino_ratio': (float, -10, 10),
            'max_drawdown': (float, -1, 0),
            'var_95': (float, -1, 0),
            'calmar_ratio': (float, -10, 10),
            'total_gas_cost_usd': (float, 0, 1000000),
            'volume_per_gas': (float, 0, 1000000),
            'net_return_after_costs': (float, -1, 100),
            'unique_tokens_traded': (int, 0, 1000),
            'hhi_concentration': (float, 0, 1),
            'max_position_size': (float, 0, 100),
            'effective_tokens': (float, 0, 1000),
            'diversification_score': (float, 0, 100),
            'consistency_score': (float, -10, 10),
            'positive_days_pct': (float, 0, 100),
            'time_weighted_return': (float, -1, 100)
        }

        for field, (field_type, min_val, max_val) in numeric_fields.items():
            if field in data:
                try:
                    value = field_type(data[field])

                    # Handle NaN and infinity
                    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        value = 0.0

                    # Clamp to valid range
                    value = max(min_val, min(max_val, value))
                    validated[field] = value

                except (ValueError, TypeError):
                    validated[field] = 0 if field_type == int else 0.0

        # Date fields
        if 'calculation_date' in data:
            if isinstance(data['calculation_date'], (date, datetime)):
                validated['calculation_date'] = data['calculation_date']
            else:
                validated['calculation_date'] = date.today()

        return validated

    def export_performance_data(
        self,
        output_format: str = "csv",
        time_period: Optional[str] = None,
        wallet_addresses: Optional[List[str]] = None
    ) -> str:
        """
        Export performance data in specified format.

        Args:
            output_format: Export format ("csv", "json", "parquet")
            time_period: Optional time period filter
            wallet_addresses: Optional list of specific wallets

        Returns:
            File path of exported data
        """
        try:
            with self.get_connection() as conn:
                where_conditions = []
                params = []

                if time_period:
                    where_conditions.append("time_period = ?") if self.db_type == "sqlite" else where_conditions.append("time_period = %s")
                    params.append(time_period)

                if wallet_addresses:
                    placeholders = ",".join(["?" if self.db_type == "sqlite" else "%s"] * len(wallet_addresses))
                    where_conditions.append(f"wallet_address IN ({placeholders})")
                    params.extend(wallet_addresses)

                where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

                query = f"SELECT * FROM wallet_performance {where_clause} ORDER BY total_return DESC"

                df = pd.read_sql_query(query, conn, params=params)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"wallet_performance_{timestamp}.{output_format}"

                # Export in specified format
                if output_format == "csv":
                    df.to_csv(filename, index=False)
                elif output_format == "json":
                    df.to_json(filename, orient="records", indent=2)
                elif output_format == "parquet":
                    df.to_parquet(filename, index=False)
                else:
                    raise ValueError(f"Unsupported export format: {output_format}")

                logger.info(f"Exported {len(df)} records to {filename}")
                return filename

        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            raise