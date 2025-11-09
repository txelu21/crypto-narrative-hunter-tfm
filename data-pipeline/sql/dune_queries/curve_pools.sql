-- FIXED: Curve Pool Discovery Query
-- Query ID: 5878406
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date)
-- Dune uses Trino/Presto, not PostgreSQL

-- Curve has multiple pool implementations, focusing on main ones
-- Simplified to discover active Curve pools with basic metrics

WITH curve_pool_events AS (
    -- Get Curve trading activity from dex.trades (curated table)
    SELECT
        project_contract_address as pool_address,
        taker as liquidity_provider,
        block_time as evt_block_time,
        tx_hash as evt_tx_hash
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND project = 'curve'
      AND block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
),

active_pools AS (
    -- Identify pools with recent activity
    SELECT
        pool_address,
        COUNT(DISTINCT liquidity_provider) as unique_providers,
        COUNT(DISTINCT evt_tx_hash) as liquidity_events,
        MAX(evt_block_time) as last_activity
    FROM curve_pool_events
    GROUP BY pool_address
    HAVING COUNT(DISTINCT evt_tx_hash) >= 5  -- Minimum activity threshold
),

-- Get swap activity for volume estimation from dex.trades
combined_swaps AS (
    SELECT
        project_contract_address as pool_address,
        COUNT(*) as total_swaps_7d,
        COUNT(DISTINCT taker) as total_unique_traders,
        SUM(amount_usd) as volume_7d_usd
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND project = 'curve'
      AND block_time >= NOW() - INTERVAL '7' day
    GROUP BY project_contract_address
)

-- Final result
SELECT
    ap.pool_address,
    'Curve' as dex_name,
    ap.unique_providers,
    ap.liquidity_events,
    ap.last_activity,

    -- Swap metrics
    COALESCE(cs.total_swaps_7d, 0) as swaps_last_7d,
    COALESCE(cs.total_unique_traders, 0) as unique_traders_7d,

    -- Activity score (simple heuristic)
    (COALESCE(cs.total_swaps_7d, 0) * 0.7 + ap.liquidity_events * 0.3) as activity_score,

    -- Pool type indicator
    CASE
        WHEN cs.total_swaps_7d > 100 THEN 'high_activity'
        WHEN cs.total_swaps_7d > 20 THEN 'medium_activity'
        ELSE 'low_activity'
    END as activity_level

FROM active_pools ap
LEFT JOIN combined_swaps cs ON cs.pool_address = ap.pool_address

-- Filter for active pools
WHERE ap.liquidity_events >= 5
  OR COALESCE(cs.total_swaps_7d, 0) > 0

ORDER BY activity_score DESC
LIMIT 1000;