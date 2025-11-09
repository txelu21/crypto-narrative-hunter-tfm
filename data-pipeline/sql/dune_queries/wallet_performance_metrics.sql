-- FIXED: Wallet Performance and Efficiency Metrics Calculation
-- Query ID: 5878574
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date), min_volume_usd (number), min_trade_count (number)
-- Calculates comprehensive trading performance metrics for smart wallet assessment
-- Dune uses Trino/Presto, not PostgreSQL

WITH
-- Base trading data from multiple DEXs (simplified for Trino)
all_trades AS (
    -- All DEX trades from dex.trades (Uniswap V2, V3, Curve, etc.)
    -- Gas metrics removed to improve query performance
    SELECT
        t.taker as trader_address,
        t.block_time as trade_time,
        t.tx_hash,
        CONCAT(t.project, ' ', COALESCE(t.version, 'v1')) as dex_name,
        t.project_contract_address as pool_address,
        t.token_bought_address as token0,
        t.token_sold_address as token1,
        t.amount_usd as trade_value_usd,
        t.token_bought_address as token_in,
        t.token_sold_address as token_out
    FROM dex.trades t
    WHERE t.blockchain = 'ethereum'
      AND t.project IN ('uniswap', 'curve', 'sushiswap')
      AND t.block_time >= COALESCE(TRY_CAST('{{start_date}}' AS DATE), DATE('2023-01-01'))
      AND t.block_time <= COALESCE(TRY_CAST('{{end_date}}' AS DATE), DATE('2024-01-01')) + INTERVAL '1' day
      AND t.taker IS NOT NULL
      AND t.amount_usd > 0
),

-- Daily trading patterns for consistency analysis
daily_trading_patterns AS (
    SELECT
        trader_address,
        DATE(trade_time) as trade_date,
        COUNT(*) as daily_trade_count,
        SUM(trade_value_usd) as daily_volume_usd,
        COUNT(DISTINCT dex_name) as dexs_used_daily,
        COUNT(DISTINCT pool_address) as pools_used_daily,
        MIN(trade_time) as first_trade_daily,
        MAX(trade_time) as last_trade_daily
    FROM all_trades
    WHERE trade_value_usd > 0
    GROUP BY trader_address, DATE(trade_time)
),

-- Time-based activity analysis
temporal_activity AS (
    SELECT
        trader_address,
        COUNT(DISTINCT trade_date) as active_days,
        COUNT(DISTINCT EXTRACT(HOUR FROM first_trade_daily)) as active_hours,
        COUNT(DISTINCT EXTRACT(DOW FROM trade_date)) as active_weekdays,
        STDDEV(daily_trade_count) as trade_count_volatility,
        STDDEV(daily_volume_usd) as volume_volatility,
        AVG(CAST(TO_UNIXTIME(last_trade_daily) - TO_UNIXTIME(first_trade_daily) AS DOUBLE) / 3600) as avg_daily_trading_duration_hours
    FROM daily_trading_patterns
    GROUP BY trader_address
),

-- Token and pool diversity metrics
diversity_metrics AS (
    SELECT
        trader_address,
        COUNT(DISTINCT token_in) as unique_tokens_in,
        COUNT(DISTINCT token_out) as unique_tokens_out,
        COUNT(DISTINCT token_in) + COUNT(DISTINCT token_out) as total_unique_tokens,
        COUNT(DISTINCT pool_address) as unique_pools,
        COUNT(DISTINCT dex_name) as unique_dexs
    FROM all_trades
    GROUP BY trader_address
),

-- Volume efficiency and trade sizing
volume_efficiency AS (
    SELECT
        trader_address,
        COUNT(*) as total_trades,
        SUM(trade_value_usd) as total_volume,
        AVG(trade_value_usd) as avg_trade_size,
        APPROX_PERCENTILE(trade_value_usd, 0.5) as median_trade_size,
        STDDEV(trade_value_usd) / NULLIF(AVG(trade_value_usd), 0) as trade_size_consistency,
        COUNT(CASE WHEN trade_value_usd >= 10000 THEN 1 END) as large_trades_10k,
        COUNT(CASE WHEN trade_value_usd >= 100000 THEN 1 END) as large_trades_100k,
        COUNT(CASE WHEN trade_value_usd < 100 THEN 1 END) as small_trades_sub100,
        SUM(CASE WHEN trade_value_usd >= 10000 THEN trade_value_usd ELSE 0 END) / NULLIF(SUM(trade_value_usd), 0) as large_trade_volume_share,
        MIN(trade_time) as first_trade_time,
        MAX(trade_time) as last_trade_time
    FROM all_trades
    WHERE trade_value_usd > 0
    GROUP BY trader_address
)

-- Comprehensive performance metrics output
SELECT
    ve.trader_address,

    -- Basic volume and activity metrics
    ve.total_volume as total_volume_usd,
    ve.total_trades,
    ta.active_days,
    ve.avg_trade_size as avg_trade_size_usd,
    ve.median_trade_size as median_trade_size_usd,

    -- Volume efficiency metrics
    ve.total_volume / NULLIF(ve.total_trades, 0) as volume_per_trade,

    -- Token diversity metrics
    dm.total_unique_tokens,
    dm.unique_pools,
    dm.unique_dexs,
    dm.total_unique_tokens / NULLIF(ta.active_days, 0) as token_diversity_rate,

    -- Activity pattern metrics
    ve.first_trade_time,
    ve.last_trade_time,

    -- Consistency and sophistication indicators
    ve.trade_size_consistency,
    ta.volume_volatility / NULLIF(ve.avg_trade_size, 0) as volume_consistency,
    CAST(ta.active_hours AS DOUBLE) / 24.0 as time_diversification_score,
    CAST(ta.active_weekdays AS DOUBLE) / 7.0 as weekly_consistency_score,

    -- Advanced efficiency metrics
    ve.large_trade_volume_share,

    -- Composite quality score (gas metrics removed for performance)
    LEAST(1.0, (
        -- Volume component (40%, increased from 30%)
        LEAST(0.4, ve.total_volume / COALESCE(TRY_CAST('{{min_volume_usd}}' AS DOUBLE), 25000.0) * 0.4) +

        -- Activity component (25%, increased from 20%)
        LEAST(0.25, CAST(ve.total_trades AS DOUBLE) / COALESCE(TRY_CAST('{{min_trade_count}}' AS DOUBLE), 10.0) * 0.25) +

        -- Diversity component (20%, same)
        LEAST(0.2, CAST(dm.total_unique_tokens AS DOUBLE) / 10.0 * 0.2) +

        -- Consistency component (15%, increased from 10%)
        CASE
            WHEN ve.trade_size_consistency <= 2 AND ta.active_days >= 5 THEN 0.15
            WHEN ve.trade_size_consistency <= 3 AND ta.active_days >= 3 THEN 0.10
            WHEN ta.active_days >= 2 THEN 0.05
            ELSE 0
        END
    )) as performance_quality_score,

    -- Analysis metadata
    COALESCE(TRY_CAST('{{start_date}}' AS DATE), DATE('2023-01-01')) as analysis_start,
    COALESCE(TRY_CAST('{{end_date}}' AS DATE), DATE('2024-01-01')) as analysis_end

FROM volume_efficiency ve
INNER JOIN temporal_activity ta ON ta.trader_address = ve.trader_address
INNER JOIN diversity_metrics dm ON dm.trader_address = ve.trader_address

WHERE
    -- Minimum thresholds for meaningful analysis
    ve.total_volume >= COALESCE(TRY_CAST('{{min_volume_usd}}' AS DOUBLE), 25000.0)
    AND ve.total_trades >= COALESCE(TRY_CAST('{{min_trade_count}}' AS INTEGER), 10)
    AND ta.active_days >= 2
    AND dm.total_unique_tokens >= 2

ORDER BY performance_quality_score DESC, ve.total_volume DESC
LIMIT 10000;