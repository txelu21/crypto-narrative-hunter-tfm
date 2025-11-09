import contextlib
import time
from typing import Iterator, Optional, Dict, Any, List, Tuple
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
import psycopg
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)

# Global connection pool
_pool: Optional[ConnectionPool] = None


def init_pool() -> None:
    """Initialize the connection pool with retry logic"""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=2,
            max_size=20,
            timeout=30,
            max_idle=300,  # 5 minutes
            max_lifetime=3600,  # 1 hour
            check=ConnectionPool.check_connection,
        )
        logger.info("Database connection pool initialized")


def get_pool() -> ConnectionPool:
    """Get the connection pool, initializing if needed"""
    if _pool is None:
        init_pool()
    return _pool


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((psycopg.OperationalError, psycopg.InterfaceError))
)
def execute_with_retry(query: str, params: Optional[Tuple] = None, fetch: bool = True) -> Optional[List[Dict]]:
    """Execute a query with exponential backoff retry logic"""
    pool = get_pool()
    start_time = time.time()

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            result = cur.fetchall() if fetch else None
            conn.commit()

    duration_ms = int((time.time() - start_time) * 1000)
    logger.log_operation(
        operation="db_query",
        params={"query_type": query.split()[0].upper()},
        status="completed",
        duration_ms=duration_ms
    )

    return result


@contextlib.contextmanager
def get_cursor(readonly: bool = False) -> Iterator[psycopg.Cursor]:
    """Context manager for database cursor with connection pooling"""
    pool = get_pool()
    conn = None

    try:
        with pool.connection() as conn:
            if readonly:
                conn.read_only = True
                conn.autocommit = True

            with conn.cursor(row_factory=dict_row) as cur:
                yield cur

            if not readonly:
                conn.commit()

    except psycopg.Error as e:
        if conn and not readonly:
            conn.rollback()
        logger.log_operation(
            operation="db_cursor",
            status="failed",
            error=str(e)
        )
        raise
    except Exception as e:
        if conn and not readonly:
            conn.rollback()
        logger.log_operation(
            operation="db_cursor",
            status="failed",
            error=str(e)
        )
        raise


# Helper functions for common database operations
def upsert_record(table: str, data: Dict[str, Any], conflict_columns: List[str]) -> None:
    """Perform UPSERT operation with ON CONFLICT handling"""
    columns = list(data.keys())
    placeholders = [f"%({col})s" for col in columns]
    conflict_cols = ", ".join(conflict_columns)
    update_cols = ", ".join([
        f"{col} = EXCLUDED.{col}"
        for col in columns
        if col not in conflict_columns
    ])

    query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT ({conflict_cols})
        DO UPDATE SET {update_cols}, updated_at = NOW()
    """

    execute_with_retry(query, tuple(data.values()), fetch=False)


def batch_insert(table: str, records: List[Dict[str, Any]], batch_size: int = 1000) -> int:
    """Insert multiple records in batches"""
    if not records:
        return 0

    total_inserted = 0
    pool = get_pool()

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        columns = list(batch[0].keys())

        # Build the VALUES clause with placeholders
        values_template = "(" + ", ".join(["%s"] * len(columns)) + ")"
        values_clause = ", ".join([values_template] * len(batch))

        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES {values_clause}
            ON CONFLICT DO NOTHING
        """

        # Flatten the values
        values = []
        for record in batch:
            values.extend([record[col] for col in columns])

        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                total_inserted += cur.rowcount
                conn.commit()

    logger.log_operation(
        operation="batch_insert",
        params={"table": table, "records": len(records)},
        status="completed",
        message=f"Inserted {total_inserted} records into {table}"
    )

    return total_inserted


def test_connection() -> bool:
    """Test database connection and return True if successful"""
    try:
        result = execute_with_retry("SELECT 1 as test", fetch=True)
        return result is not None and len(result) > 0
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def close_pool() -> None:
    """Close the connection pool gracefully"""
    global _pool
    if _pool:
        _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


class DatabaseManager:
    """Database manager class for consistent connection handling"""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        init_pool()

    def get_connection(self):
        """Get a database connection from the pool"""
        return get_pool().connection()

    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch: bool = True):
        """Execute a query with retry logic"""
        return execute_with_retry(query, params, fetch)

    def test_connection(self) -> bool:
        """Test database connectivity"""
        return test_connection()

    def close(self):
        """Close the connection pool"""
        close_pool()
