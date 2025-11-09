"""
Price Database Management and Time-Series Optimization
Handles database operations for price data with optimized queries
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)


class PriceDatabase:
    """
    Database manager for price data with time-series optimizations

    Features:
    - Optimized schema for time-series queries
    - Efficient bulk operations
    - Query optimization
    - Data retention policies
    - Compression and partitioning
    """

    # SQL schema definitions
    CREATE_TABLES_SQL = """
    -- Main price data table
    CREATE TABLE IF NOT EXISTS eth_prices (
        timestamp TIMESTAMP PRIMARY KEY,
        price_usd DECIMAL(15,8) NOT NULL,
        source VARCHAR(20) NOT NULL,
        block_number BIGINT,
        confidence_score DECIMAL(4,3) DEFAULT 1.0,
        validation_status VARCHAR(20) DEFAULT 'pending',
        corrected BOOLEAN DEFAULT FALSE,
        original_price DECIMAL(15,8),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );

    -- Indexes for time-series queries
    CREATE INDEX IF NOT EXISTS idx_eth_prices_timestamp ON eth_prices(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_eth_prices_source ON eth_prices(source);
    CREATE INDEX IF NOT EXISTS idx_eth_prices_block ON eth_prices(block_number);
    CREATE INDEX IF NOT EXISTS idx_eth_prices_validation ON eth_prices(validation_status);
    CREATE INDEX IF NOT EXISTS idx_eth_prices_created ON eth_prices(created_at);

    -- Price validation results table
    CREATE TABLE IF NOT EXISTS price_validations (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        chainlink_price DECIMAL(15,8),
        coingecko_price DECIMAL(15,8),
        uniswap_price DECIMAL(15,8),
        median_price DECIMAL(15,8),
        max_deviation DECIMAL(6,4),
        consensus_reached BOOLEAN,
        validation_timestamp TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_eth_prices FOREIGN KEY (timestamp) REFERENCES eth_prices(timestamp)
    );

    CREATE INDEX IF NOT EXISTS idx_price_validations_timestamp ON price_validations(timestamp);
    CREATE INDEX IF NOT EXISTS idx_price_validations_consensus ON price_validations(consensus_reached);

    -- Price anomalies table
    CREATE TABLE IF NOT EXISTS price_anomalies (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        price DECIMAL(15,8) NOT NULL,
        z_score DECIMAL(6,3),
        severity VARCHAR(20) NOT NULL,
        detection_method VARCHAR(50),
        suggested_action VARCHAR(50),
        corrected BOOLEAN DEFAULT FALSE,
        detected_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_eth_prices_anomaly FOREIGN KEY (timestamp) REFERENCES eth_prices(timestamp)
    );

    CREATE INDEX IF NOT EXISTS idx_price_anomalies_timestamp ON price_anomalies(timestamp);
    CREATE INDEX IF NOT EXISTS idx_price_anomalies_severity ON price_anomalies(severity);
    CREATE INDEX IF NOT EXISTS idx_price_anomalies_corrected ON price_anomalies(corrected);

    -- Token price data (for non-ETH tokens)
    CREATE TABLE IF NOT EXISTS token_prices (
        id SERIAL PRIMARY KEY,
        token_address VARCHAR(42) NOT NULL,
        token_symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        price_usd DECIMAL(18,8) NOT NULL,
        source VARCHAR(20) NOT NULL,
        volume_24h DECIMAL(24,2),
        market_cap DECIMAL(24,2),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(token_address, timestamp)
    );

    CREATE INDEX IF NOT EXISTS idx_token_prices_address_time ON token_prices(token_address, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_token_prices_symbol ON token_prices(token_symbol);
    """

    def __init__(self, connection_pool: asyncpg.Pool):
        """
        Initialize price database

        Args:
            connection_pool: AsyncPG connection pool
        """
        self.pool = connection_pool
        logger.info("Price database initialized")

    async def initialize_schema(self):
        """Create database tables and indexes"""
        async with self.pool.acquire() as conn:
            await conn.execute(self.CREATE_TABLES_SQL)
        logger.info("Database schema initialized")

    async def store_price(
        self,
        timestamp: int,
        price_usd: float,
        source: str,
        block_number: Optional[int] = None,
        confidence_score: float = 1.0
    ) -> bool:
        """
        Store single price data point

        Args:
            timestamp: Unix timestamp
            price_usd: Price in USD
            source: Data source
            block_number: Optional block number
            confidence_score: Price confidence score

        Returns:
            Success boolean
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO eth_prices
                        (timestamp, price_usd, source, block_number, confidence_score)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (timestamp) DO UPDATE
                    SET price_usd = $2, source = $3, block_number = $4,
                        confidence_score = $5, updated_at = NOW()
                    """,
                    datetime.fromtimestamp(timestamp),
                    price_usd,
                    source,
                    block_number,
                    confidence_score
                )
            return True

        except Exception as e:
            logger.error(f"Error storing price: {e}")
            return False

    async def bulk_insert_prices(
        self,
        price_data: List[Dict]
    ) -> int:
        """
        Bulk insert price data with optimization

        Args:
            price_data: List of price data dicts

        Returns:
            Number of rows inserted
        """
        if not price_data:
            return 0

        try:
            async with self.pool.acquire() as conn:
                # Prepare data for COPY
                records = [
                    (
                        datetime.fromtimestamp(p['timestamp']),
                        p['price_usd'],
                        p.get('source', 'unknown'),
                        p.get('block_number'),
                        p.get('confidence_score', 1.0)
                    )
                    for p in price_data
                ]

                # Use COPY for efficient bulk insert
                result = await conn.copy_records_to_table(
                    'eth_prices',
                    records=records,
                    columns=['timestamp', 'price_usd', 'source', 'block_number', 'confidence_score']
                )

                logger.info(f"Bulk inserted {len(price_data)} prices")
                return len(price_data)

        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            return 0

    async def get_price_at_timestamp(
        self,
        timestamp: int,
        tolerance_seconds: int = 7200
    ) -> Optional[Dict]:
        """
        Get price at specific timestamp with tolerance

        Args:
            timestamp: Target unix timestamp
            tolerance_seconds: Time tolerance (default: 2 hours)

        Returns:
            Price data dict or None
        """
        try:
            async with self.pool.acquire() as conn:
                dt = datetime.fromtimestamp(timestamp)
                tolerance_dt = datetime.fromtimestamp(timestamp + tolerance_seconds)

                row = await conn.fetchrow(
                    """
                    SELECT
                        EXTRACT(EPOCH FROM timestamp)::INTEGER as timestamp,
                        price_usd,
                        source,
                        block_number,
                        confidence_score,
                        validation_status
                    FROM eth_prices
                    WHERE timestamp BETWEEN $1 AND $2
                    ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - $1)))
                    LIMIT 1
                    """,
                    dt,
                    tolerance_dt
                )

                if row:
                    return dict(row)

        except Exception as e:
            logger.error(f"Error fetching price: {e}")

        return None

    async def get_price_range(
        self,
        start_timestamp: int,
        end_timestamp: int
    ) -> List[Dict]:
        """
        Get price data for time range

        Args:
            start_timestamp: Start unix timestamp
            end_timestamp: End unix timestamp

        Returns:
            List of price data dicts
        """
        try:
            async with self.pool.acquire() as conn:
                start_dt = datetime.fromtimestamp(start_timestamp)
                end_dt = datetime.fromtimestamp(end_timestamp)

                rows = await conn.fetch(
                    """
                    SELECT
                        EXTRACT(EPOCH FROM timestamp)::INTEGER as timestamp,
                        price_usd,
                        source,
                        block_number,
                        confidence_score,
                        validation_status,
                        corrected,
                        original_price
                    FROM eth_prices
                    WHERE timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp ASC
                    """,
                    start_dt,
                    end_dt
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error fetching price range: {e}")
            return []

    async def store_validation_result(
        self,
        timestamp: int,
        chainlink_price: Optional[float],
        coingecko_price: Optional[float],
        uniswap_price: Optional[float],
        median_price: float,
        max_deviation: float,
        consensus_reached: bool
    ) -> bool:
        """
        Store price validation result

        Args:
            timestamp: Unix timestamp
            chainlink_price: Chainlink price
            coingecko_price: CoinGecko price
            uniswap_price: Uniswap price
            median_price: Median price
            max_deviation: Maximum deviation
            consensus_reached: Whether consensus was reached

        Returns:
            Success boolean
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO price_validations
                        (timestamp, chainlink_price, coingecko_price, uniswap_price,
                         median_price, max_deviation, consensus_reached)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    datetime.fromtimestamp(timestamp),
                    chainlink_price,
                    coingecko_price,
                    uniswap_price,
                    median_price,
                    max_deviation,
                    consensus_reached
                )
            return True

        except Exception as e:
            logger.error(f"Error storing validation result: {e}")
            return False

    async def store_anomaly(
        self,
        timestamp: int,
        price: float,
        z_score: float,
        severity: str,
        detection_method: str,
        suggested_action: str
    ) -> bool:
        """
        Store detected price anomaly

        Args:
            timestamp: Unix timestamp
            price: Anomalous price
            z_score: Z-score
            severity: Severity level
            detection_method: Detection method used
            suggested_action: Suggested correction action

        Returns:
            Success boolean
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO price_anomalies
                        (timestamp, price, z_score, severity, detection_method, suggested_action)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    datetime.fromtimestamp(timestamp),
                    price,
                    z_score,
                    severity,
                    detection_method,
                    suggested_action
                )
            return True

        except Exception as e:
            logger.error(f"Error storing anomaly: {e}")
            return False

    async def get_statistics(
        self,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> Dict:
        """
        Get price data statistics

        Args:
            start_timestamp: Optional start timestamp
            end_timestamp: Optional end timestamp

        Returns:
            Statistics dict
        """
        try:
            async with self.pool.acquire() as conn:
                where_clause = ""
                params = []

                if start_timestamp and end_timestamp:
                    where_clause = "WHERE timestamp BETWEEN $1 AND $2"
                    params = [
                        datetime.fromtimestamp(start_timestamp),
                        datetime.fromtimestamp(end_timestamp)
                    ]

                row = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) as total_count,
                        MIN(price_usd) as min_price,
                        MAX(price_usd) as max_price,
                        AVG(price_usd) as avg_price,
                        STDDEV(price_usd) as std_price,
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest
                    FROM eth_prices
                    {where_clause}
                    """,
                    *params
                )

                return dict(row) if row else {}

        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}

    async def apply_retention_policy(
        self,
        retention_days: int = 365
    ) -> int:
        """
        Apply data retention policy

        Args:
            retention_days: Number of days to retain

        Returns:
            Number of rows deleted
        """
        try:
            async with self.pool.acquire() as conn:
                cutoff_date = datetime.now() - timedelta(days=retention_days)

                result = await conn.execute(
                    """
                    DELETE FROM eth_prices
                    WHERE timestamp < $1
                    """,
                    cutoff_date
                )

                # Extract row count from result
                deleted_count = int(result.split()[-1])
                logger.info(f"Deleted {deleted_count} old price records")
                return deleted_count

        except Exception as e:
            logger.error(f"Error applying retention policy: {e}")
            return 0

    async def optimize_tables(self):
        """Optimize database tables (vacuum, analyze)"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("VACUUM ANALYZE eth_prices")
                await conn.execute("VACUUM ANALYZE price_validations")
                await conn.execute("VACUUM ANALYZE price_anomalies")

            logger.info("Database tables optimized")

        except Exception as e:
            logger.error(f"Error optimizing tables: {e}")

    async def get_coverage_report(
        self,
        start_timestamp: int,
        end_timestamp: int
    ) -> Dict:
        """
        Generate data coverage report

        Args:
            start_timestamp: Start unix timestamp
            end_timestamp: End unix timestamp

        Returns:
            Coverage report dict
        """
        try:
            async with self.pool.acquire() as conn:
                # Get actual data points
                actual_count = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM eth_prices
                    WHERE timestamp BETWEEN $1 AND $2
                    """,
                    datetime.fromtimestamp(start_timestamp),
                    datetime.fromtimestamp(end_timestamp)
                )

                # Calculate expected hourly data points
                expected_count = (end_timestamp - start_timestamp) // 3600

                # Get gaps
                gaps = await conn.fetch(
                    """
                    SELECT
                        EXTRACT(EPOCH FROM gap_start)::INTEGER as gap_start,
                        EXTRACT(EPOCH FROM gap_end)::INTEGER as gap_end,
                        gap_hours
                    FROM (
                        SELECT
                            timestamp + INTERVAL '1 hour' as gap_start,
                            LEAD(timestamp) OVER (ORDER BY timestamp) as gap_end,
                            EXTRACT(EPOCH FROM (LEAD(timestamp) OVER (ORDER BY timestamp) - timestamp)) / 3600 as gap_hours
                        FROM eth_prices
                        WHERE timestamp BETWEEN $1 AND $2
                    ) gaps
                    WHERE gap_hours > 1
                    ORDER BY gap_hours DESC
                    LIMIT 10
                    """,
                    datetime.fromtimestamp(start_timestamp),
                    datetime.fromtimestamp(end_timestamp)
                )

                coverage_pct = (actual_count / expected_count * 100) if expected_count > 0 else 0

                return {
                    'expected_data_points': expected_count,
                    'actual_data_points': actual_count,
                    'missing_data_points': expected_count - actual_count,
                    'coverage_percentage': coverage_pct,
                    'largest_gaps': [dict(g) for g in gaps]
                }

        except Exception as e:
            logger.error(f"Error generating coverage report: {e}")
            return {}


# Import at module level for SQL operations
from datetime import timedelta