import asyncio
import json
import os
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import requests
from dataclasses import dataclass
from data_collection.common.logging_setup import get_logger
from data_collection.common.config import get_config

logger = get_logger(__name__)


@dataclass
class QueryJob:
    execution_id: str
    query_id: int
    parameters: Dict[str, Any]
    submitted_at: datetime
    status: str = "submitted"
    result_id: Optional[str] = None
    error_message: Optional[str] = None


class DuneClientError(Exception):
    pass


class DuneQueryError(DuneClientError):
    pass


class DuneRateLimitError(DuneClientError):
    pass


class DuneClient:
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        self.api_key = api_key or os.getenv("DUNE_API_KEY")
        if not self.api_key:
            raise DuneClientError("DUNE_API_KEY not found in environment")

        self.base_url = "https://api.dune.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json"
        })

        # Setup caching
        self.cache_dir = Path(cache_dir or get_config().get("dune_cache_dir", "./cache/dune"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

        # Track active jobs
        self.active_jobs: Dict[str, QueryJob] = {}

    def _rate_limit(self):
        """Implement basic rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated request with rate limiting and error handling"""
        self._rate_limit()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                time.sleep(retry_after)
                return self._make_request(method, endpoint, **kwargs)

            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise DuneRateLimitError(f"Rate limit exceeded: {e}")
            elif response.status_code >= 400:
                error_msg = f"Dune API error {response.status_code}: {response.text}"
                raise DuneQueryError(error_msg)
            else:
                raise DuneClientError(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            raise DuneClientError(f"Request failed: {e}")

    def _get_cache_key(self, query_id: int, parameters: Dict[str, Any]) -> str:
        """Generate cache key from query ID and parameters"""
        param_str = json.dumps(parameters, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"query_{query_id}_{param_hash}"

    def _get_cache_path(self, cache_key: str, format: str = "parquet") -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.{format}"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file is valid (exists and not too old)"""
        if not cache_path.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, metadata: Dict[str, Any]):
        """Save results to cache in both parquet and JSON formats"""
        try:
            # Save data as parquet
            parquet_path = self._get_cache_path(cache_key, "parquet")
            data.to_parquet(parquet_path, index=False)

            # Save metadata as JSON
            json_path = self._get_cache_path(cache_key, "json")
            with open(json_path, 'w') as f:
                json.dump({
                    "metadata": metadata,
                    "cached_at": datetime.now().isoformat(),
                    "record_count": len(data)
                }, f, indent=2)

            logger.info(f"Cached {len(data)} records to {parquet_path}")

        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load results from cache"""
        try:
            parquet_path = self._get_cache_path(cache_key, "parquet")
            if self._is_cache_valid(parquet_path):
                data = pd.read_parquet(parquet_path)
                logger.info(f"Loaded {len(data)} records from cache: {parquet_path}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        return None

    def execute_query(
        self,
        query_id: int,
        parameters: Dict[str, Any],
        use_cache: bool = True,
        cache_max_age_hours: int = 24
    ) -> str:
        """Execute a parameterized query and return execution ID"""

        cache_key = self._get_cache_key(query_id, parameters)

        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                # Return special cache execution ID
                return f"cache_{cache_key}"

        # Submit query
        endpoint = f"query/{query_id}/execute"
        payload = {"query_parameters": parameters}

        logger.info(f"Executing Dune query {query_id} with parameters: {parameters}")

        response = self._make_request("POST", endpoint, json=payload)
        result = response.json()

        execution_id = result.get("execution_id")
        if not execution_id:
            raise DuneQueryError(f"No execution_id in response: {result}")

        # Track job
        job = QueryJob(
            execution_id=execution_id,
            query_id=query_id,
            parameters=parameters,
            submitted_at=datetime.now()
        )
        self.active_jobs[execution_id] = job

        logger.info(f"Query submitted with execution_id: {execution_id}")
        return execution_id

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of query execution"""

        # Handle cached results
        if execution_id.startswith("cache_"):
            cache_key = execution_id[6:]  # Remove "cache_" prefix
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return {
                    "execution_id": execution_id,
                    "query_id": "cached",
                    "state": "QUERY_STATE_COMPLETED",
                    "submitted_at": "cached",
                    "result_metadata": {
                        "row_count": len(cached_data),
                        "result_set_bytes": cached_data.memory_usage(deep=True).sum(),
                        "total_result_set_bytes": cached_data.memory_usage(deep=True).sum()
                    }
                }
            else:
                return {
                    "execution_id": execution_id,
                    "state": "QUERY_STATE_FAILED",
                    "error": {"type": "CacheError", "message": "Cache miss"}
                }

        endpoint = f"execution/{execution_id}/status"
        response = self._make_request("GET", endpoint)
        return response.json()

    def poll_execution(
        self,
        execution_id: str,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 5
    ) -> Dict[str, Any]:
        """Poll execution until completion with exponential backoff"""

        start_time = time.time()
        current_interval = poll_interval_seconds

        while time.time() - start_time < timeout_seconds:
            status = self.get_execution_status(execution_id)
            state = status.get("state")

            logger.debug(f"Execution {execution_id} status: {state}")

            if state == "QUERY_STATE_COMPLETED":
                logger.info(f"Query execution completed: {execution_id}")
                return status
            elif state == "QUERY_STATE_FAILED":
                error_info = status.get("error", {})
                error_msg = f"Query failed: {error_info.get('message', 'Unknown error')}"
                raise DuneQueryError(error_msg)
            elif state in ["QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"]:
                raise DuneQueryError(f"Query {state.lower()}: {execution_id}")

            # Exponential backoff (max 30 seconds)
            time.sleep(current_interval)
            current_interval = min(current_interval * 1.5, 30)

        raise DuneQueryError(f"Query execution timeout after {timeout_seconds} seconds")

    def get_execution_result(self, execution_id: str) -> pd.DataFrame:
        """Get execution results as DataFrame"""

        # Handle cached results
        if execution_id.startswith("cache_"):
            cache_key = execution_id[6:]
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            else:
                raise DuneQueryError(f"Cache miss for execution_id: {execution_id}")

        endpoint = f"execution/{execution_id}/results"
        response = self._make_request("GET", endpoint)
        result = response.json()

        # Convert to DataFrame
        if "result" not in result or "rows" not in result["result"]:
            raise DuneQueryError(f"Invalid result format: {result}")

        rows = result["result"]["rows"]
        if not rows:
            logger.warning(f"No data returned for execution {execution_id}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Cache the results if from a real query
        if execution_id in self.active_jobs:
            job = self.active_jobs[execution_id]
            cache_key = self._get_cache_key(job.query_id, job.parameters)
            metadata = result.get("result", {}).get("metadata", {})
            self._save_to_cache(cache_key, df, metadata)

        logger.info(f"Retrieved {len(df)} records from execution {execution_id}")
        return df

    def execute_and_wait(
        self,
        query_id: int,
        parameters: Dict[str, Any],
        timeout_seconds: int = 600,
        use_cache: bool = True,
        cache_max_age_hours: int = 24
    ) -> pd.DataFrame:
        """Execute query and wait for results"""

        execution_id = self.execute_query(
            query_id=query_id,
            parameters=parameters,
            use_cache=use_cache,
            cache_max_age_hours=cache_max_age_hours
        )

        # Poll until completion
        self.poll_execution(execution_id, timeout_seconds=timeout_seconds)

        # Get results
        return self.get_execution_result(execution_id)

    def cleanup_cache(self, max_age_days: int = 7):
        """Remove old cache files"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)

        removed_count = 0
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache files")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_directory": str(self.cache_dir),
            "cached_queries": len(cache_files),
            "total_size_mb": total_size / 1024 / 1024,
            "oldest_cache": min(
                (datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files),
                default=None
            ),
            "newest_cache": max(
                (datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files),
                default=None
            )
        }