-- FIXED: Bot Detection Patterns Query
-- Query ID: 5878555
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date)
-- Identifies bot-like trading patterns

WITH trader_patterns AS (
    -- Get trading patterns from Uniswap V2/V3
    SELECT
        tx."from" as trader_address,
        DATE(s.evt_block_time) as trade_date,
        DATE_TRUNC('hour', s.evt_block_time) as trade_hour,
        COUNT(*) as trades_in_hour,
        COUNT(DISTINCT s.contract_address) as unique_pools_hour,
        AVG(tx.gas_price) as avg_gas_price,
        STDDEV(tx.gas_price) as stddev_gas_price
    FROM uniswap_v2_ethereum.pair_evt_swap s
    INNER JOIN ethereum.transactions tx
        ON tx.hash = s.evt_tx_hash
        AND tx.block_time = s.evt_block_time
    WHERE s.evt_block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND s.evt_block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND tx."from" != 0x0000000000000000000000000000000000000000
    GROUP BY tx."from", DATE(s.evt_block_time), DATE_TRUNC('hour', s.evt_block_time)
),

trader_stats AS (
    -- Aggregate trader statistics
    SELECT
        trader_address,
        COUNT(DISTINCT trade_date) as active_days,
        COUNT(DISTINCT trade_hour) as active_hours,
        SUM(trades_in_hour) as total_trades,
        MAX(trades_in_hour) as max_trades_per_hour,
        AVG(trades_in_hour) as avg_trades_per_hour,

        -- Gas price consistency (bots often use similar gas)
        AVG(avg_gas_price) as overall_avg_gas,
        AVG(stddev_gas_price) as gas_variability,

        -- Pool diversity
        AVG(unique_pools_hour) as avg_pools_per_hour,

        -- Time pattern (bots trade 24/7 more consistently)
        COUNT(DISTINCT EXTRACT(HOUR FROM trade_hour)) as unique_hours_of_day

    FROM trader_patterns
    GROUP BY trader_address
),

bot_scores AS (
    -- Calculate bot likelihood scores
    SELECT
        trader_address,
        active_days,
        total_trades,
        max_trades_per_hour,
        avg_trades_per_hour,
        avg_pools_per_hour,
        gas_variability,
        unique_hours_of_day,

        -- Bot indicators (higher = more bot-like)
        CASE
            WHEN max_trades_per_hour >= 50 THEN 10
            WHEN max_trades_per_hour >= 20 THEN 7
            WHEN max_trades_per_hour >= 10 THEN 4
            ELSE 0
        END +
        CASE
            WHEN unique_hours_of_day >= 20 THEN 8  -- Trading across all hours
            WHEN unique_hours_of_day >= 15 THEN 5
            ELSE 0
        END +
        CASE
            WHEN avg_pools_per_hour < 1.5 THEN 6  -- Very focused trading
            WHEN avg_pools_per_hour < 2.5 THEN 3
            ELSE 0
        END +
        CASE
            WHEN gas_variability < 1000000000 THEN 5  -- Very consistent gas
            ELSE 0
        END +
        CASE
            WHEN total_trades / NULLIF(active_days, 0) > 100 THEN 7  -- >100 trades/day
            WHEN total_trades / NULLIF(active_days, 0) > 50 THEN 4
            ELSE 0
        END as bot_risk_score,

        -- Pattern classification
        CASE
            WHEN max_trades_per_hour >= 50 THEN 'high_frequency_bot'
            WHEN max_trades_per_hour >= 20 AND unique_hours_of_day >= 18 THEN 'likely_bot'
            WHEN avg_pools_per_hour < 1.5 AND total_trades / NULLIF(active_days, 0) > 50 THEN 'mev_bot'
            WHEN total_trades / NULLIF(active_days, 0) > 100 THEN 'possible_bot'
            ELSE 'likely_human'
        END as pattern_classification

    FROM trader_stats
)

-- Final result with bot detection
SELECT
    trader_address,
    active_days,
    total_trades,
    max_trades_per_hour,
    avg_trades_per_hour,
    unique_hours_of_day,
    bot_risk_score,
    pattern_classification,

    -- Human-readable indicators
    CASE WHEN max_trades_per_hour >= 50 THEN true ELSE false END as flag_high_frequency,
    CASE WHEN unique_hours_of_day >= 20 THEN true ELSE false END as flag_24_7_trading,
    CASE WHEN avg_pools_per_hour < 1.5 THEN true ELSE false END as flag_focused_trading,
    CASE WHEN total_trades / NULLIF(active_days, 0) > 100 THEN true ELSE false END as flag_high_daily_volume

FROM bot_scores

-- Return all for analysis
ORDER BY bot_risk_score DESC, total_trades DESC
LIMIT 10000;