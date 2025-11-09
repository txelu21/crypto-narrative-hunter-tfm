"""Database schema for wallet balance snapshots.

This module defines the optimized table structure for storing daily balance snapshots
with proper partitioning, indexing, and compression.
"""

from typing import Optional, List
import psycopg
from data_collection.common.logging_setup import get_logger

logger = get_logger(__name__)


# SQL for creating the main partitioned table
CREATE_WALLET_BALANCES_TABLE = """
CREATE TABLE IF NOT EXISTS wallet_balances (
    id BIGSERIAL,
    wallet_address VARCHAR(42) NOT NULL,
    token_address VARCHAR(42) NOT NULL,
    snapshot_date DATE NOT NULL,
    block_number BIGINT NOT NULL,
    balance NUMERIC(78,0) NOT NULL,  -- Raw token amount (supports up to uint256)
    eth_value NUMERIC(36,18),        -- ETH-denominated value
    price_eth NUMERIC(36,18),        -- Token price in ETH at snapshot
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (wallet_address, token_address, snapshot_date, block_number)
) PARTITION BY RANGE (snapshot_date);
"""

# Indexes for query performance
CREATE_WALLET_BALANCES_INDEXES = """
-- Index for wallet + date queries (most common)
CREATE INDEX IF NOT EXISTS idx_wallet_balances_wallet_date
    ON wallet_balances(wallet_address, snapshot_date);

-- Index for token + date queries (portfolio analysis)
CREATE INDEX IF NOT EXISTS idx_wallet_balances_token_date
    ON wallet_balances(token_address, snapshot_date);

-- Index for finding top holders by eth_value
CREATE INDEX IF NOT EXISTS idx_wallet_balances_eth_value
    ON wallet_balances(eth_value DESC NULLS LAST);

-- Index for block number queries (historical lookups)
CREATE INDEX IF NOT EXISTS idx_wallet_balances_block
    ON wallet_balances(block_number);
"""

# Position tracking table for incremental updates
CREATE_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS wallet_positions (
    id BIGSERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) NOT NULL,
    token_address VARCHAR(42) NOT NULL,
    first_seen_date DATE NOT NULL,
    last_seen_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'active', 'closed'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(wallet_address, token_address)
);

CREATE INDEX IF NOT EXISTS idx_positions_wallet
    ON wallet_positions(wallet_address);

CREATE INDEX IF NOT EXISTS idx_positions_status
    ON wallet_positions(status);
"""

# Snapshot metadata table for tracking collection progress
CREATE_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS balance_snapshots_meta (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,
    block_number BIGINT NOT NULL,
    block_timestamp BIGINT NOT NULL,
    wallets_processed INTEGER DEFAULT 0,
    tokens_tracked INTEGER DEFAULT 0,
    total_balances INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL,  -- 'pending', 'processing', 'completed', 'failed'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date
    ON balance_snapshots_meta(snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_snapshots_status
    ON balance_snapshots_meta(status);
"""


def create_monthly_partition(
    conn: psycopg.Connection,
    year: int,
    month: int,
) -> None:
    """Create a monthly partition for wallet_balances table.

    Args:
        conn: Database connection
        year: Year for partition
        month: Month for partition (1-12)
    """
    # Calculate date range
    from datetime import date

    start_date = date(year, month, 1)

    # Calculate next month
    if month == 12:
        end_date = date(year + 1, 1, 1)
    else:
        end_date = date(year, month + 1, 1)

    partition_name = f"wallet_balances_{year}_{month:02d}"

    create_partition_sql = f"""
    CREATE TABLE IF NOT EXISTS {partition_name}
    PARTITION OF wallet_balances
    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
    """

    with conn.cursor() as cur:
        cur.execute(create_partition_sql)
        conn.commit()

    logger.info(
        f"Created partition {partition_name} for date range {start_date} to {end_date}"
    )


def create_partitions_for_date_range(
    conn: psycopg.Connection,
    start_date: str,
    end_date: str,
) -> None:
    """Create all necessary partitions for a date range.

    Args:
        conn: Database connection
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start
    partitions_created = []

    while current <= end:
        try:
            create_monthly_partition(conn, current.year, current.month)
            partitions_created.append(f"{current.year}-{current.month:02d}")
        except Exception as e:
            logger.error(f"Failed to create partition for {current.year}-{current.month:02d}: {e}")
            raise

        current += relativedelta(months=1)

    logger.info(f"Created {len(partitions_created)} partitions: {partitions_created}")


def init_schema(conn: psycopg.Connection) -> None:
    """Initialize complete database schema for balance snapshots.

    Args:
        conn: Database connection
    """
    logger.info("Initializing balance snapshots database schema")

    with conn.cursor() as cur:
        # Create main tables
        logger.info("Creating wallet_balances table")
        cur.execute(CREATE_WALLET_BALANCES_TABLE)

        logger.info("Creating indexes for wallet_balances")
        cur.execute(CREATE_WALLET_BALANCES_INDEXES)

        logger.info("Creating wallet_positions table")
        cur.execute(CREATE_POSITIONS_TABLE)

        logger.info("Creating balance_snapshots_meta table")
        cur.execute(CREATE_SNAPSHOTS_TABLE)

        conn.commit()

    logger.info("Schema initialization completed successfully")


def ensure_partition_exists(
    conn: psycopg.Connection,
    snapshot_date: str,
) -> None:
    """Ensure partition exists for a given snapshot date.

    Args:
        conn: Database connection
        snapshot_date: Snapshot date (YYYY-MM-DD)
    """
    from datetime import datetime

    dt = datetime.strptime(snapshot_date, "%Y-%m-%d")

    # Check if partition exists
    partition_name = f"wallet_balances_{dt.year}_{dt.month:02d}"

    check_sql = """
    SELECT COUNT(*) FROM pg_tables
    WHERE tablename = %s
    """

    with conn.cursor() as cur:
        cur.execute(check_sql, (partition_name,))
        exists = cur.fetchone()[0] > 0

        if not exists:
            logger.info(f"Creating missing partition for {snapshot_date}")
            create_monthly_partition(conn, dt.year, dt.month)
        else:
            logger.debug(f"Partition {partition_name} already exists")


def drop_old_partitions(
    conn: psycopg.Connection,
    retention_months: int = 12,
) -> List[str]:
    """Drop partitions older than retention period.

    Args:
        conn: Database connection
        retention_months: Number of months to retain (default: 12)

    Returns:
        List of dropped partition names
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    cutoff_date = datetime.now() - relativedelta(months=retention_months)

    # Find old partitions
    find_partitions_sql = """
    SELECT tablename FROM pg_tables
    WHERE schemaname = 'public'
    AND tablename LIKE 'wallet_balances_%'
    """

    dropped_partitions = []

    with conn.cursor() as cur:
        cur.execute(find_partitions_sql)
        partitions = cur.fetchall()

        for (partition_name,) in partitions:
            try:
                # Extract year and month from partition name
                parts = partition_name.split('_')
                if len(parts) == 4:  # wallet_balances_YYYY_MM
                    year = int(parts[2])
                    month = int(parts[3])

                    partition_date = datetime(year, month, 1)

                    if partition_date < cutoff_date:
                        logger.info(f"Dropping old partition {partition_name}")
                        cur.execute(f"DROP TABLE IF EXISTS {partition_name}")
                        dropped_partitions.append(partition_name)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse partition name {partition_name}: {e}")

        conn.commit()

    logger.info(f"Dropped {len(dropped_partitions)} old partitions")
    return dropped_partitions


def get_table_stats(conn: psycopg.Connection) -> dict:
    """Get statistics about balance snapshot tables.

    Args:
        conn: Database connection

    Returns:
        Dictionary with table statistics
    """
    stats = {}

    with conn.cursor() as cur:
        # Count total balances
        cur.execute("SELECT COUNT(*) FROM wallet_balances")
        stats['total_balances'] = cur.fetchone()[0]

        # Count active positions
        cur.execute("SELECT COUNT(*) FROM wallet_positions WHERE status = 'active'")
        stats['active_positions'] = cur.fetchone()[0]

        # Count closed positions
        cur.execute("SELECT COUNT(*) FROM wallet_positions WHERE status = 'closed'")
        stats['closed_positions'] = cur.fetchone()[0]

        # Count snapshots
        cur.execute("SELECT COUNT(*) FROM balance_snapshots_meta")
        stats['total_snapshots'] = cur.fetchone()[0]

        # Count completed snapshots
        cur.execute("SELECT COUNT(*) FROM balance_snapshots_meta WHERE status = 'completed'")
        stats['completed_snapshots'] = cur.fetchone()[0]

        # Get date range
        cur.execute("SELECT MIN(snapshot_date), MAX(snapshot_date) FROM wallet_balances")
        min_date, max_date = cur.fetchone()
        stats['date_range'] = {
            'min': str(min_date) if min_date else None,
            'max': str(max_date) if max_date else None,
        }

        # Get unique wallets
        cur.execute("SELECT COUNT(DISTINCT wallet_address) FROM wallet_balances")
        stats['unique_wallets'] = cur.fetchone()[0]

        # Get unique tokens
        cur.execute("SELECT COUNT(DISTINCT token_address) FROM wallet_balances")
        stats['unique_tokens'] = cur.fetchone()[0]

    return stats