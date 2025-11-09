"""
Smart Wallet Query Manager
Manages execution, caching, and optimization of smart wallet identification queries
"""
import asyncio
import json
import time
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.tokens.dune_client import DuneClient, DuneClientError, QueryJob
from data_collection.common.logging_setup import get_logger
from data_collection.common.config import get_config

logger = get_logger(__name__)


class QueryType(Enum):
    UNISWAP_VOLUME = "uniswap_volume"
    CURVE_VOLUME = "curve_volume"
    COMBINED_DEX = "combined_dex"
    BOT_DETECTION = "bot_detection"
    MEV_DETECTION = "mev_detection"
    PERFORMANCE_METRICS = "performance_metrics"


@dataclass
class QueryParameters:
    start_date: str
    end_date: str
    min_volume_usd: int
    min_trade_count: int
    analysis_window: str = ""
    purpose: str = "production"

    def to_dict(self, query_type: Optional['QueryType'] = None) -> Dict[str, Any]:
        """Return only the parameters that specific Dune query accepts"""
        # Some queries only accept date parameters
        date_only_queries = [QueryType.BOT_DETECTION, QueryType.COMBINED_DEX]

        if query_type and query_type in date_only_queries:
            return {
                "start_date": self.start_date,
                "end_date": self.end_date
            }

        # Other queries accept all parameters
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "min_volume_usd": self.min_volume_usd,
            "min_trade_count": self.min_trade_count
        }

    def get_hash(self) -> str:
        """Generate a hash for caching purposes"""
        param_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()


@dataclass
class QueryExecution:
    query_type: QueryType
    parameters: QueryParameters
    execution_id: str
    submitted_at: datetime
    status: str = "submitted"
    result_df: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    credit_cost: Optional[int] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None


class SmartWalletQueryManager:
    def __init__(self, dune_client: Optional[DuneClient] = None, cache_dir: Optional[str] = None):
        self.dune_client = dune_client or DuneClient()
        self.cache_dir = Path(cache_dir or get_config().get("smart_wallet_cache_dir", "./cache/smart_wallets"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load query IDs from config file
        self.query_mappings = self._load_query_ids()

        # Parameter sets for different scenarios
        self.parameter_presets = self._load_parameter_presets()

        # Track active executions
        self.active_executions: Dict[str, QueryExecution] = {}

    def _load_query_ids(self) -> Dict[QueryType, int]:
        """Load Dune query IDs from config file"""
        try:
            import yaml
            config_file = Path(__file__).parent.parent / "config" / "dune_query_ids.yaml"

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            wallet_discovery = config.get("wallet_discovery", {})
            wallet_analysis = config.get("wallet_analysis", {})

            return {
                QueryType.UNISWAP_VOLUME: wallet_discovery.get("uniswap_volume"),
                QueryType.CURVE_VOLUME: wallet_discovery.get("curve_volume"),
                QueryType.COMBINED_DEX: wallet_discovery.get("combined_dex"),
                QueryType.BOT_DETECTION: wallet_discovery.get("bot_detection"),
                QueryType.MEV_DETECTION: wallet_discovery.get("mev_detection"),
                QueryType.PERFORMANCE_METRICS: wallet_analysis.get("performance_metrics")
            }
        except Exception as e:
            logger.warning(f"Could not load query IDs from config: {e}. Using placeholders.")
            # Fallback to placeholders if config not found
            return {
                QueryType.UNISWAP_VOLUME: 5878470,
                QueryType.CURVE_VOLUME: 5878483,
                QueryType.COMBINED_DEX: 5878544,
                QueryType.BOT_DETECTION: 5878555,
                QueryType.MEV_DETECTION: 5878561,
                QueryType.PERFORMANCE_METRICS: 5878574
            }

    def _load_parameter_presets(self) -> Dict[str, QueryParameters]:
        """Load predefined parameter sets"""
        try:
            params_file = self.cache_dir.parent / "sql" / "dune_queries" / "query_parameters.json"
            if params_file.exists():
                with open(params_file, 'r') as f:
                    config = json.load(f)

                presets = {}
                for name, params in config["smart_wallet_queries"]["parameter_sets"].items():
                    presets[name] = QueryParameters(**params)

                return presets
        except Exception as e:
            logger.warning(f"Could not load parameter presets: {e}")

        # Default presets
        return {
            "30_day_analysis": QueryParameters(
                start_date="2024-08-27",
                end_date="2024-09-27",
                min_volume_usd=10000,
                min_trade_count=10,
                analysis_window="30 days"
            ),
            "90_day_analysis": QueryParameters(
                start_date="2024-06-27",
                end_date="2024-09-27",
                min_volume_usd=25000,
                min_trade_count=15,
                analysis_window="90 days"
            ),
            "preview_7day": QueryParameters(
                start_date="2024-09-20",
                end_date="2024-09-27",
                min_volume_usd=5000,
                min_trade_count=5,
                analysis_window="7 days",
                purpose="testing"
            )
        }

    def get_cache_key(self, query_type: QueryType, parameters: QueryParameters) -> str:
        """Generate cache key for query and parameters"""
        return f"{query_type.value}_{parameters.get_hash()}"

    def get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.parquet"

    def get_metadata_path(self, cache_key: str) -> Path:
        """Get metadata file path"""
        return self.cache_dir / f"{cache_key}_metadata.json"

    def is_cache_valid(self, cache_key: str, max_age_hours: int = 24) -> bool:
        """Check if cached result is valid"""
        cache_path = self.get_cache_path(cache_key)
        if not cache_path.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)

    def save_to_cache(self, cache_key: str, execution: QueryExecution):
        """Save execution results to cache"""
        try:
            if execution.result_df is not None:
                # Save data
                cache_path = self.get_cache_path(cache_key)
                execution.result_df.to_parquet(cache_path, index=False)

                # Save metadata
                metadata_path = self.get_metadata_path(cache_key)
                metadata = {
                    "query_type": execution.query_type.value,
                    "parameters": execution.parameters.to_dict(),
                    "execution_id": execution.execution_id,
                    "submitted_at": execution.submitted_at.isoformat(),
                    "execution_time_seconds": execution.execution_time_seconds,
                    "credit_cost": execution.credit_cost,
                    "record_count": len(execution.result_df),
                    "cached_at": datetime.now().isoformat(),
                    "metadata": execution.metadata
                }

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"Cached {len(execution.result_df)} records for {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def load_from_cache(self, cache_key: str) -> Optional[QueryExecution]:
        """Load execution results from cache"""
        try:
            cache_path = self.get_cache_path(cache_key)
            metadata_path = self.get_metadata_path(cache_key)

            if not self.is_cache_valid(cache_key):
                return None

            # Load data
            df = pd.read_parquet(cache_path)

            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            # Create execution object
            execution = QueryExecution(
                query_type=QueryType(metadata.get("query_type")),
                parameters=QueryParameters(**metadata.get("parameters", {})),
                execution_id=f"cache_{cache_key}",
                submitted_at=datetime.fromisoformat(metadata.get("submitted_at", datetime.now().isoformat())),
                status="completed",
                result_df=df,
                metadata=metadata.get("metadata"),
                credit_cost=metadata.get("credit_cost"),
                execution_time_seconds=metadata.get("execution_time_seconds")
            )

            logger.info(f"Loaded {len(df)} records from cache for {cache_key}")
            return execution

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    async def execute_query(
        self,
        query_type: QueryType,
        parameters: Optional[QueryParameters] = None,
        use_cache: bool = True,
        cache_max_age_hours: int = 24
    ) -> QueryExecution:
        """Execute a smart wallet query with caching"""

        # Use default parameters if not provided
        if parameters is None:
            parameters = self.parameter_presets.get("30_day_analysis")
            if parameters is None:
                raise ValueError("No parameters provided and no default available")

        cache_key = self.get_cache_key(query_type, parameters)

        # Check cache first
        if use_cache:
            cached_execution = self.load_from_cache(cache_key)
            if cached_execution is not None:
                return cached_execution

        # Get Dune query ID
        dune_query_id = self.query_mappings.get(query_type)
        if dune_query_id is None:
            raise ValueError(f"No Dune query mapping for {query_type}")

        # Execute query
        start_time = time.time()

        try:
            query_params = parameters.to_dict(query_type)
            logger.info(f"Executing {query_type.value} query with parameters: {query_params}")

            execution_id = self.dune_client.execute_query(
                query_id=dune_query_id,
                parameters=query_params,
                use_cache=False  # We handle caching at this level
            )

            execution = QueryExecution(
                query_type=query_type,
                parameters=parameters,
                execution_id=execution_id,
                submitted_at=datetime.now(),
                status="executing"
            )

            self.active_executions[execution_id] = execution

            # Poll for completion
            status = self.dune_client.poll_execution(execution_id, timeout_seconds=600)

            # Get results
            result_df = self.dune_client.get_execution_result(execution_id)

            execution_time = time.time() - start_time

            # Update execution object
            execution.status = "completed"
            execution.result_df = result_df
            execution.execution_time_seconds = execution_time
            execution.metadata = status.get("result_metadata", {})

            # Estimate credit cost (would be actual from Dune API in real implementation)
            execution.credit_cost = self._estimate_credit_cost(query_type, execution_time, len(result_df))

            # Cache results
            if use_cache:
                self.save_to_cache(cache_key, execution)

            logger.info(f"Query {query_type.value} completed in {execution_time:.1f}s with {len(result_df)} results")

            return execution

        except Exception as e:
            execution_time = time.time() - start_time

            execution = QueryExecution(
                query_type=query_type,
                parameters=parameters,
                execution_id="failed",
                submitted_at=datetime.now(),
                status="failed",
                execution_time_seconds=execution_time,
                error_message=str(e)
            )

            logger.error(f"Query {query_type.value} failed after {execution_time:.1f}s: {e}")
            raise

    def _estimate_credit_cost(self, query_type: QueryType, execution_time: float, result_count: int) -> int:
        """Estimate credit cost based on query type and execution characteristics"""
        base_cost = {
            QueryType.UNISWAP_VOLUME: 150,
            QueryType.CURVE_VOLUME: 100,
            QueryType.COMBINED_DEX: 250,
            QueryType.BOT_DETECTION: 200,
            QueryType.MEV_DETECTION: 300,
            QueryType.PERFORMANCE_METRICS: 180
        }

        cost = base_cost.get(query_type, 100)

        # Adjust for execution time and result size
        if execution_time > 120:
            cost = int(cost * 1.5)
        if result_count > 15000:
            cost = int(cost * 1.3)

        return cost

    async def run_preview_test(self, query_type: QueryType) -> Dict[str, Any]:
        """Run a preview test with 7-day window"""
        preview_params = self.parameter_presets["preview_7day"]

        execution = await self.execute_query(query_type, preview_params, use_cache=True)

        return {
            "query_type": query_type.value,
            "status": execution.status,
            "result_count": len(execution.result_df) if execution.result_df is not None else 0,
            "execution_time": execution.execution_time_seconds,
            "estimated_credit_cost": execution.credit_cost,
            "parameters": execution.parameters.to_dict()
        }

    async def run_full_analysis(self, analysis_type: str = "90_day_analysis") -> Dict[str, QueryExecution]:
        """Run complete smart wallet analysis pipeline"""
        if analysis_type not in self.parameter_presets:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        parameters = self.parameter_presets[analysis_type]
        results = {}

        # Execute core queries in sequence (could be parallelized)
        query_sequence = [
            QueryType.UNISWAP_VOLUME,
            QueryType.CURVE_VOLUME,
            QueryType.BOT_DETECTION,
            QueryType.PERFORMANCE_METRICS,
            QueryType.COMBINED_DEX
        ]

        for query_type in query_sequence:
            try:
                logger.info(f"Executing {query_type.value} for {analysis_type}")
                execution = await self.execute_query(query_type, parameters)
                results[query_type.value] = execution

                # Brief pause between queries to respect rate limits
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to execute {query_type.value}: {e}")
                results[query_type.value] = None

        return results

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_directory": str(self.cache_dir),
            "cached_queries": len(cache_files),
            "total_size_mb": total_size / 1024 / 1024,
            "query_types": list(set(f.stem.split('_')[0] for f in cache_files)),
            "oldest_cache": min(
                (datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files),
                default=None
            ),
            "newest_cache": max(
                (datetime.fromtimestamp(f.stat().st_mtime) for f in cache_files),
                default=None
            )
        }

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

        return removed_count


async def main():
    """Main entry point for smart wallet discovery"""
    print("🔍 Starting Smart Wallet Discovery Pipeline")
    print("=" * 60)

    try:
        # Initialize manager
        manager = SmartWalletQueryManager()

        # Show cache statistics
        cache_stats = manager.get_cache_statistics()
        print(f"\n📁 Cache Status:")
        print(f"   Cached queries: {cache_stats['cached_queries']}")
        print(f"   Total size: {cache_stats['total_size_mb']:.2f} MB")

        # Run full 90-day analysis
        print(f"\n🚀 Starting 90-day wallet analysis...")
        print(f"   This will execute 5 Dune queries and may take 2-4 hours")
        print(f"   Queries will be cached to avoid re-running")

        results = await manager.run_full_analysis("90_day_analysis")

        # Display results
        print(f"\n✅ Analysis Complete!")
        print(f"=" * 60)

        total_wallets = 0
        for query_type, execution in results.items():
            if execution and execution.status == "completed":
                wallet_count = len(execution.result_df) if execution.result_df is not None else 0
                total_wallets += wallet_count
                print(f"   {query_type}: {wallet_count} wallets")
                print(f"      Execution time: {execution.execution_time_seconds:.1f}s")
                print(f"      Estimated credits: {execution.credit_cost}")
            else:
                print(f"   {query_type}: FAILED")

        print(f"\n📊 Total unique wallets discovered: {total_wallets}")
        print(f"\n💾 Results cached in: {cache_stats['cache_directory']}")

        # Update cache stats
        final_stats = manager.get_cache_statistics()
        print(f"\n📈 Final cache size: {final_stats['total_size_mb']:.2f} MB")

        return 0

    except Exception as e:
        logger.exception("Wallet discovery failed")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)