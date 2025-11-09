-- FIXED: Smart Wallet Combined DEX Query
-- Query ID: 5878544
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date)
-- Dune uses Trino/Presto, not PostgreSQL

-- Identifies high-volume traders across multiple DEXs
-- Simplified to focus on Uniswap V2/V3 initially

WITH uniswap_v2_traders AS (
    -- Get Uniswap V2 swap activity from dex.trades
    SELECT
        t.taker as trader_address,
        COUNT(DISTINCT t.tx_hash) as trade_count,
        COUNT(DISTINCT t.project_contract_address) as unique_pools,
        COUNT(DISTINCT DATE(t.block_time)) as active_days,
        SUM(t.amount_usd) as total_volume_usd
    FROM dex.trades t
    WHERE t.blockchain = 'ethereum'
      AND t.project = 'uniswap'
      AND t.version = '2'
      AND t.block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND t.block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND t.taker != 0x0000000000000000000000000000000000000000
    GROUP BY t.taker
    HAVING COUNT(DISTINCT t.tx_hash) >= 10  -- Minimum 10 trades
),

uniswap_v3_traders AS (
    -- Get Uniswap V3 swap activity from dex.trades
    SELECT
        t.taker as trader_address,
        COUNT(DISTINCT t.tx_hash) as trade_count,
        COUNT(DISTINCT t.project_contract_address) as unique_pools,
        COUNT(DISTINCT DATE(t.block_time)) as active_days,
        SUM(t.amount_usd) as total_volume_usd
    FROM dex.trades t
    WHERE t.blockchain = 'ethereum'
      AND t.project = 'uniswap'
      AND t.version = '3'
      AND t.block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND t.block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND t.taker != 0x0000000000000000000000000000000000000000
    GROUP BY t.taker
    HAVING COUNT(DISTINCT t.tx_hash) >= 10
),

combined_traders AS (
    -- Combine V2 and V3 traders with aggregated metrics
    SELECT
        trader_address,
        SUM(trade_count) as total_trades,
        SUM(unique_pools) as total_unique_pools,
        MAX(active_days) as active_days,
        SUM(total_volume_usd) as total_volume_usd,
        COUNT(DISTINCT CASE WHEN trade_count > 0 THEN 'v2' END) as v2_active,
        COUNT(DISTINCT CASE WHEN trade_count > 0 THEN 'v3' END) as v3_active
    FROM (
        SELECT trader_address, trade_count, unique_pools, active_days, total_volume_usd FROM uniswap_v2_traders
        UNION ALL
        SELECT trader_address, trade_count, unique_pools, active_days, total_volume_usd FROM uniswap_v3_traders
    ) all_trades
    GROUP BY trader_address
),

-- Calculate trading patterns (no expensive transaction join needed!)
trader_volumes AS (
    SELECT
        trader_address,
        total_trades,
        total_unique_pools,
        active_days,
        total_volume_usd,

        -- Trading patterns
        total_trades / NULLIF(active_days, 0) as avg_trades_per_day,
        total_unique_pools / NULLIF(total_trades, 0) as pool_diversity_ratio,
        total_volume_usd / NULLIF(total_trades, 0) as avg_trade_size_usd

    FROM combined_traders
),

-- Bot detection: Flag suspicious patterns
bot_scores AS (
    SELECT
        trader_address,

        -- Bot indicators
        CASE
            WHEN avg_trades_per_day > 100 THEN 5  -- Very high frequency
            WHEN avg_trades_per_day > 50 THEN 3   -- High frequency
            WHEN avg_trades_per_day > 20 THEN 1   -- Moderate
            ELSE 0
        END +
        CASE
            WHEN pool_diversity_ratio < 0.1 THEN 2  -- Very low diversity
            WHEN pool_diversity_ratio < 0.3 THEN 1  -- Low diversity
            ELSE 0
        END as bot_risk_score,

        -- Classification
        CASE
            WHEN avg_trades_per_day > 100 OR pool_diversity_ratio < 0.1 THEN 'likely_bot'
            WHEN avg_trades_per_day > 50 OR pool_diversity_ratio < 0.3 THEN 'possible_bot'
            ELSE 'likely_human'
        END as wallet_type

    FROM trader_volumes
)

-- Final result
SELECT
    tv.trader_address as wallet_address,
    tv.total_trades,
    tv.total_unique_pools as unique_pools_traded,
    tv.active_days,
    tv.total_volume_usd,
    tv.avg_trades_per_day,
    tv.pool_diversity_ratio,
    tv.avg_trade_size_usd,

    -- Bot detection
    bs.bot_risk_score,
    bs.wallet_type,

    -- Multi-DEX activity indicator
    CASE
        WHEN tv.total_trades >= 100 THEN 'high_volume'
        WHEN tv.total_trades >= 50 THEN 'medium_volume'
        ELSE 'low_volume'
    END as volume_tier,

    'combined_dex' as source

FROM trader_volumes tv
INNER JOIN bot_scores bs ON bs.trader_address = tv.trader_address

-- Filter criteria
WHERE tv.total_trades >= 10
  AND tv.active_days >= 7
  AND tv.total_unique_pools >= 3
  AND bs.wallet_type != 'likely_bot'  -- Exclude obvious bots
  AND tv.total_volume_usd > 1000  -- Minimum $1k volume

ORDER BY tv.total_volume_usd DESC
LIMIT 10000;