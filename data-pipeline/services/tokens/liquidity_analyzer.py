from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json
import math

from data_collection.common.logging_setup import get_logger
from data_collection.common.db import DatabaseManager
from data_collection.common.checkpoints import CheckpointManager
from .dune_client import DuneClient, DuneClientError

logger = get_logger(__name__)


def clean_for_json(obj: Any) -> Any:
    """Clean object to be JSON-serializable by converting NaN/Inf to None"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif pd.isna(obj):
        return None
    else:
        return obj


@dataclass
class LiquidityTier:
    name: str
    min_tvl: float
    max_tvl: Optional[float] = None

    def matches(self, tvl: float) -> bool:
        if self.max_tvl is None:
            return tvl >= self.min_tvl
        return self.min_tvl <= tvl < self.max_tvl


@dataclass
class PoolData:
    pool_address: str
    dex_name: str
    token_address: str
    pair_token: str
    tvl_usd: float
    volume_24h_usd: float
    price_eth: Optional[float]
    last_updated: datetime
    metadata: Dict[str, Any]


class DEXLiquidityAnalyzer:
    def __init__(
        self,
        dune_client: DuneClient,
        db_manager: DatabaseManager,
        checkpoint_manager: CheckpointManager
    ):
        self.dune_client = dune_client
        self.db_manager = db_manager
        self.checkpoint_manager = checkpoint_manager

        # Liquidity tiers as defined in story requirements
        self.liquidity_tiers = [
            LiquidityTier("Tier 1", 10_000_000),  # > $10M
            LiquidityTier("Tier 2", 1_000_000, 10_000_000),  # $1M - $10M
            LiquidityTier("Tier 3", 0, 1_000_000),  # < $1M
        ]

        # Dune query IDs (configured 2025-09-30)
        self.query_ids = {
            "uniswap_v2": 5878381,
            "uniswap_v3": 5878390,
            "curve": 5878406,
            "token_filter": 5878444
        }

    def assign_liquidity_tier(self, tvl_usd: float) -> str:
        """Assign liquidity tier based on TVL"""
        for tier in self.liquidity_tiers:
            if tier.matches(tvl_usd):
                return tier.name
        return "Untiered"

    def get_collected_tokens(self, limit: Optional[int] = None) -> List[str]:
        """Get list of collected token addresses from database"""
        query = """
            SELECT token_address
            FROM tokens
            ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.db_manager.get_connection() as conn:
            result = conn.execute(query).fetchall()
            return [row[0] for row in result]

    def filter_tokens_for_analysis(
        self,
        token_addresses: List[str],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Filter tokens to prioritize those with existing liquidity"""
        filtered_tokens = []

        # Process in batches to avoid query parameter limits
        for i in range(0, len(token_addresses), batch_size):
            batch = token_addresses[i:i + batch_size]
            token_params = ",".join(f"'{addr}'" for addr in batch)

            try:
                # Use token filtering helper query
                if self.query_ids["token_filter"]:
                    df = self.dune_client.execute_and_wait(
                        query_id=self.query_ids["token_filter"],
                        parameters={"token_addresses": token_params}
                    )

                    # Convert to list of dicts for easier processing
                    batch_results = df.to_dict('records')
                    filtered_tokens.extend(batch_results)
                else:
                    # Fallback: include all tokens with basic priority
                    for addr in batch:
                        filtered_tokens.append({
                            "token_address": addr,
                            "analysis_priority": "Medium Priority",
                            "liquidity_score": 3
                        })

            except Exception as e:
                logger.warning(f"Failed to filter token batch: {e}")
                # Include tokens without filtering on error
                for addr in batch:
                    filtered_tokens.append({
                        "token_address": addr,
                        "analysis_priority": "Low Priority",
                        "liquidity_score": 1
                    })

        # Sort by priority and liquidity score
        priority_order = {"High Priority": 3, "Medium Priority": 2, "Low Priority": 1}
        filtered_tokens.sort(
            key=lambda x: (
                priority_order.get(x.get("analysis_priority", "Low Priority"), 0),
                x.get("liquidity_score", 0)
            ),
            reverse=True
        )

        logger.info(f"Filtered {len(token_addresses)} tokens to {len(filtered_tokens)} for analysis")
        return filtered_tokens

    def discover_pools_for_tokens(
        self,
        token_addresses: List[str],
        date_range_days: int = 30
    ) -> List[PoolData]:
        """Discover pools across all DEXs for given tokens"""
        all_pools = []

        # Date range for analysis
        # Note: Using 2024 dates as Dune blockchain data is currently available for 2024
        end_date = datetime(2024, 12, 31)
        start_date = end_date - timedelta(days=date_range_days)

        date_params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }

        # Convert all token addresses to lowercase for case-insensitive matching
        token_addresses_lower = [addr.lower() for addr in token_addresses]
        logger.info(f"Discovering pools for {len(token_addresses)} tokens")

        # Query each DEX once and filter for all our tokens
        for dex_name, query_id in self.query_ids.items():
            if not query_id or dex_name == "token_filter":
                continue

            try:
                logger.info(f"Querying {dex_name} pools...")

                # Dune queries only accept date parameters, return all pools for date range
                df = self.dune_client.execute_and_wait(
                    query_id=query_id,
                    parameters=date_params,
                    timeout_seconds=900  # 15 minutes for complex queries
                )

                # Filter results to only include our tokens
                if not df.empty:
                    # Handle different column names for different DEXs
                    if 'token0' in df.columns and 'token1' in df.columns:
                        # Uniswap V2/V3 format
                        mask = (df['token0'].str.lower().isin(token_addresses_lower)) | \
                               (df['token1'].str.lower().isin(token_addresses_lower))
                    elif 'token' in df.columns:
                        # Curve format (single token column)
                        mask = df['token'].str.lower().isin(token_addresses_lower)
                    else:
                        logger.warning(f"Unknown schema for {dex_name}, skipping filtering")
                        mask = pd.Series([True] * len(df))

                    df_filtered = df[mask]

                    # Convert to PoolData objects
                    dex_pools = self._process_dex_results(df_filtered, dex_name)
                    all_pools.extend(dex_pools)

                    logger.info(f"Found {len(dex_pools)} pools in {dex_name} (filtered from {len(df)} total)")
                else:
                    logger.warning(f"No pools returned from {dex_name}")

            except Exception as e:
                logger.error(f"Failed to query {dex_name}: {e}")
                continue

        logger.info(f"Discovered {len(all_pools)} pools total")
        return all_pools

    def _process_dex_results(self, df: pd.DataFrame, dex_name: str) -> List[PoolData]:
        """Process DEX query results into PoolData objects"""
        pools = []

        for _, row in df.iterrows():
            try:
                # Handle different column names across DEXs
                pool_address = row.get("pool_address") or row.get("pair_address")
                token_address = (
                    row.get("target_token") or
                    row.get("token0") if row.get("token0") != "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2" else row.get("token1")
                )

                # Skip if we don't have required fields
                if not pool_address or not token_address:
                    logger.debug(f"Skipping pool with missing required fields: pool={pool_address}, token={token_address}")
                    continue

                # Determine pair token (prefer ETH, then major stablecoins)
                pair_token = self._determine_pair_token(row)

                pool = PoolData(
                    pool_address=pool_address,
                    dex_name=dex_name,
                    token_address=token_address,
                    pair_token=pair_token,
                    tvl_usd=float(row.get("tvl_usd", 0)),
                    volume_24h_usd=float(row.get("volume_24h_usd", 0)),
                    price_eth=row.get("price_eth"),
                    last_updated=pd.to_datetime(row.get("last_updated", datetime.now())),
                    metadata=row.to_dict()
                )

                pools.append(pool)

            except Exception as e:
                logger.warning(f"Failed to process pool row: {e}")
                continue

        return pools

    def _determine_pair_token(self, row: Dict[str, Any]) -> str:
        """Determine the pair token, preferring ETH then major stablecoins"""
        weth_address = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        usdc_address = "0xA0b86a33E6441B9435B7DF041c0f3B9e8CE61e2C"
        usdt_address = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

        # Check for direct ETH pairs
        token0 = row.get("token0", "").lower()
        token1 = row.get("token1", "").lower()

        if weth_address.lower() in [token0, token1]:
            return weth_address
        elif usdc_address.lower() in [token0, token1]:
            return usdc_address
        elif usdt_address.lower() in [token0, token1]:
            return usdt_address
        else:
            # Return the non-target token
            target = row.get("target_token", "").lower()
            return token1 if token0 == target else token0

    def rank_pools_by_liquidity(self, pools: List[PoolData]) -> Dict[str, List[PoolData]]:
        """Rank pools by token and select best liquidity sources"""
        token_pools = {}

        for pool in pools:
            token_addr = pool.token_address.lower()
            if token_addr not in token_pools:
                token_pools[token_addr] = []
            token_pools[token_addr].append(pool)

        # Sort pools by TVL and volume for each token
        for token_addr in token_pools:
            token_pools[token_addr].sort(
                key=lambda p: (p.tvl_usd, p.volume_24h_usd),
                reverse=True
            )

        return token_pools

    def calculate_tier_assignments(self, token_pools: Dict[str, List[PoolData]]) -> Dict[str, Dict[str, Any]]:
        """Calculate liquidity tier assignments for tokens"""
        tier_assignments = {}

        for token_addr, pools in token_pools.items():
            if not pools:
                tier_assignments[token_addr] = {
                    "liquidity_tier": "Untiered",
                    "max_tvl": 0,
                    "best_pool": None,
                    "total_pools": 0,
                    "price_eth": None
                }
                continue

            # Use highest TVL pool for tier assignment
            best_pool = pools[0]
            max_tvl = best_pool.tvl_usd
            tier = self.assign_liquidity_tier(max_tvl)

            # Get ETH price from best direct ETH pair
            eth_price = None
            for pool in pools:
                if pool.price_eth is not None:
                    eth_price = pool.price_eth
                    break

            tier_assignments[token_addr] = {
                "liquidity_tier": tier,
                "max_tvl": max_tvl,
                "best_pool": best_pool,
                "total_pools": len(pools),
                "price_eth": eth_price,
                "all_pools": pools[:5]  # Keep top 5 pools
            }

        return tier_assignments

    def store_pool_data(self, pools: List[PoolData]):
        """Store pool data in database"""
        with self.db_manager.get_connection() as conn:
            try:
                # Create pools table if not exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS token_pools (
                        id SERIAL PRIMARY KEY,
                        token_address VARCHAR(42) NOT NULL,
                        pool_address VARCHAR(42) NOT NULL,
                        dex_name VARCHAR(20) NOT NULL,
                        pair_token VARCHAR(42) NOT NULL,
                        tvl_usd DECIMAL(20,2),
                        volume_24h_usd DECIMAL(20,2),
                        price_eth DECIMAL(36,18),
                        last_updated TIMESTAMP,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(token_address, pool_address)
                    )
                """)

                # Insert/update pool data
                for pool in pools:
                    # Clean numeric fields that might contain NaN
                    price_eth = clean_for_json(pool.price_eth)
                    tvl_usd = clean_for_json(pool.tvl_usd)
                    volume_24h_usd = clean_for_json(pool.volume_24h_usd)

                    conn.execute("""
                        INSERT INTO token_pools (
                            token_address, pool_address, dex_name, pair_token,
                            tvl_usd, volume_24h_usd, price_eth, last_updated, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (token_address, pool_address)
                        DO UPDATE SET
                            tvl_usd = EXCLUDED.tvl_usd,
                            volume_24h_usd = EXCLUDED.volume_24h_usd,
                            price_eth = EXCLUDED.price_eth,
                            last_updated = EXCLUDED.last_updated,
                            metadata = EXCLUDED.metadata
                    """, (
                        pool.token_address,
                        pool.pool_address,
                        pool.dex_name,
                        pool.pair_token,
                        tvl_usd,
                        volume_24h_usd,
                        price_eth,
                        pool.last_updated,
                        json.dumps(clean_for_json(pool.metadata))
                    ))

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to store pool data: {e}")
                raise

    def update_token_tiers(self, tier_assignments: Dict[str, Dict[str, Any]]):
        """Update liquidity tiers in tokens table"""
        with self.db_manager.get_connection() as conn:
            try:
                # Check if column exists and what type it is
                result = conn.execute("""
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_name = 'tokens' AND column_name = 'liquidity_tier'
                """).fetchone()

                if result:
                    # Column exists - check if it's the right type
                    if result[0] != 'character varying':
                        # Wrong type - drop and recreate
                        logger.info("Dropping existing liquidity_tier column (wrong type)")
                        conn.execute("ALTER TABLE tokens DROP COLUMN IF EXISTS liquidity_tier")
                        conn.execute("""
                            ALTER TABLE tokens
                            ADD COLUMN liquidity_tier VARCHAR(10)
                            CHECK (liquidity_tier IN ('Tier 1', 'Tier 2', 'Tier 3', 'Untiered'))
                        """)
                else:
                    # Column doesn't exist - create it
                    conn.execute("""
                        ALTER TABLE tokens
                        ADD COLUMN liquidity_tier VARCHAR(10)
                        CHECK (liquidity_tier IN ('Tier 1', 'Tier 2', 'Tier 3', 'Untiered'))
                    """)

                # Create index if not exists
                try:
                    conn.execute("CREATE INDEX idx_tokens_liquidity_tier ON tokens(liquidity_tier)")
                except Exception as e:
                    # Index might already exist - rollback this specific statement and continue
                    conn.rollback()
                    logger.debug(f"Could not create index (may already exist): {e}")

                # Update tiers
                for token_addr, assignment in tier_assignments.items():
                    conn.execute("""
                        UPDATE tokens
                        SET liquidity_tier = %s
                        WHERE token_address = %s
                    """, (assignment["liquidity_tier"], token_addr))

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to update token tiers: {e}")
                raise

    def generate_analysis_report(
        self,
        tier_assignments: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive liquidity analysis report"""

        # Calculate distribution statistics
        tier_counts = {"Tier 1": 0, "Tier 2": 0, "Tier 3": 0, "Untiered": 0}
        total_pools = 0
        tokens_with_eth_price = 0

        for assignment in tier_assignments.values():
            tier_counts[assignment["liquidity_tier"]] += 1
            total_pools += assignment["total_pools"]
            if assignment["price_eth"] is not None:
                tokens_with_eth_price += 1

        report = {
            "analysis_summary": {
                "total_tokens_analyzed": len(tier_assignments),
                "total_pools_discovered": total_pools,
                "tokens_with_eth_pricing": tokens_with_eth_price,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "tier_distribution": tier_counts,
            "tier_percentages": {
                tier: (count / len(tier_assignments)) * 100 if len(tier_assignments) > 0 else 0
                for tier, count in tier_counts.items()
            },
            "quality_metrics": {
                "coverage_rate": (len(tier_assignments) - tier_counts.get("Untiered", 0)) / len(tier_assignments) * 100 if len(tier_assignments) > 0 else 0,
                "pricing_coverage": tokens_with_eth_price / len(tier_assignments) * 100 if len(tier_assignments) > 0 else 0,
                "avg_pools_per_token": total_pools / len(tier_assignments) if len(tier_assignments) > 0 else 0
            }
        }

        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Analysis report saved to {output_path}")

        return report

    def run_liquidity_analysis(
        self,
        token_limit: Optional[int] = None,
        date_range_days: int = 30
    ) -> Dict[str, Any]:
        """Run complete liquidity analysis workflow"""

        checkpoint_key = "liquidity_analysis"

        try:
            # Load checkpoint
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_key)
            processed_tokens = checkpoint.get("processed_tokens", []) if checkpoint else []

            # Get tokens to analyze
            all_tokens = self.get_collected_tokens(limit=token_limit)
            tokens_to_process = [t for t in all_tokens if t not in processed_tokens]

            logger.info(f"Processing {len(tokens_to_process)} tokens (skipping {len(processed_tokens)} already processed)")

            if not tokens_to_process:
                logger.info("No tokens to process")
                return {"status": "completed", "message": "No new tokens to process"}

            # Filter tokens by liquidity potential
            filtered_tokens = self.filter_tokens_for_analysis(tokens_to_process)
            priority_tokens = [t["token_address"] for t in filtered_tokens if t["analysis_priority"] != "Skip - No Liquidity"]

            logger.info(f"Analyzing {len(priority_tokens)} high-priority tokens")

            # Discover pools
            all_pools = self.discover_pools_for_tokens(priority_tokens, date_range_days)

            # Store pool data
            self.store_pool_data(all_pools)

            # Rank and assign tiers
            token_pools = self.rank_pools_by_liquidity(all_pools)
            tier_assignments = self.calculate_tier_assignments(token_pools)

            # Update database
            self.update_token_tiers(tier_assignments)

            # Generate report
            report = self.generate_analysis_report(tier_assignments)

            # Update checkpoint
            processed_tokens.extend(priority_tokens)
            self.checkpoint_manager.save_checkpoint(
                checkpoint_key,
                {
                    "processed_tokens": processed_tokens,
                    "last_run": datetime.now().isoformat(),
                    "pools_discovered": len(all_pools),
                    "tier_assignments": len(tier_assignments)
                }
            )

            logger.info("Liquidity analysis completed successfully")
            return {
                "status": "completed",
                "report": report,
                "pools_discovered": len(all_pools),
                "tokens_analyzed": len(tier_assignments)
            }

        except Exception as e:
            logger.error(f"Liquidity analysis failed: {e}")
            raise