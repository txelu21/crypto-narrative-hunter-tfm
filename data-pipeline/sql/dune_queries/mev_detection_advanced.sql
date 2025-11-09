-- FIXED: Advanced MEV Bot Detection
-- Query ID: 5878561
-- Status: Fixed for Dune/Trino compatibility  - OPTIMIZED to prevent timeout
-- Parameters: start_date (date), end_date (date), block_range_limit (number)
-- Specialized patterns for identifying MEV bots using transaction timing and gas patterns
-- Dune uses Trino/Presto, not PostgreSQL
-- OPTIMIZATION: Added stricter filtering and sampling to prevent timeout

WITH
-- Sample blocks to reduce data volume (every 10th block)
sampled_blocks AS (
    SELECT DISTINCT block_number
    FROM ethereum.transactions
    WHERE block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND MOD(block_number, 10) = 0  -- Sample every 10th block
    LIMIT 10000  -- Limit total blocks analyzed
),

-- Block statistics for gas price analysis (only sampled blocks)
block_stats AS (
    SELECT
        t.block_number,
        COUNT(*) as total_block_txs,
        APPROX_PERCENTILE(CAST(t.gas_price AS DOUBLE), 0.5) as median_gas_price
    FROM ethereum.transactions t
    INNER JOIN sampled_blocks sb ON sb.block_number = t.block_number
    WHERE t.block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND t.block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
    GROUP BY t.block_number
),

-- Transaction timing analysis for MEV detection (only sampled blocks)
transaction_timing AS (
    SELECT
        t."from" as trader_address,
        t.block_number,
        t.index as transaction_index,
        t.gas_price,
        t.gas_used,
        t.value,
        t.block_time,
        bs.total_block_txs,
        bs.median_gas_price,
        -- Calculate position in block
        CAST(t.index AS DOUBLE) / NULLIF(CAST(bs.total_block_txs AS DOUBLE), 0) as position_in_block,
        -- Gas price relative to block median
        CAST(t.gas_price AS DOUBLE) / NULLIF(bs.median_gas_price, 0) as gas_price_ratio
    FROM ethereum.transactions t
    INNER JOIN sampled_blocks sb ON sb.block_number = t.block_number
    INNER JOIN block_stats bs ON bs.block_number = t.block_number
    WHERE t.block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND t.block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND t.gas_price > 0  -- Filter out zero gas price txs
),

-- High gas traders (candidates for MEV)
high_gas_traders AS (
    SELECT DISTINCT trader_address
    FROM transaction_timing
    WHERE gas_price_ratio >= 1.5
    LIMIT 5000  -- Limit to top candidates
),

-- Front-running pattern detection (optimized with pre-filter)
front_running_patterns AS (
    SELECT
        t1.trader_address as potential_frontrunner,
        COUNT(*) as frontrun_count,
        AVG(t1.gas_price_ratio) as avg_gas_premium,
        AVG(t1.position_in_block) as avg_block_position
    FROM transaction_timing t1
    INNER JOIN high_gas_traders hgt ON hgt.trader_address = t1.trader_address
    INNER JOIN transaction_timing t2
        ON t1.block_number = t2.block_number
        AND t1.transaction_index = t2.transaction_index - 1  -- Immediately before
        AND t1.trader_address != t2.trader_address
        AND t1.gas_price > t2.gas_price * 1.1  -- Significant gas premium
    GROUP BY t1.trader_address
    HAVING COUNT(*) >= 3  -- Reduced threshold for sampled data
),

-- Back-running (MEV extraction) patterns (optimized)
back_running_patterns AS (
    SELECT
        t2.trader_address as potential_backrunner,
        COUNT(*) as backrun_count,
        AVG(t2.gas_price_ratio) as avg_gas_premium,
        COUNT(DISTINCT t1.trader_address) as victims_count
    FROM transaction_timing t1
    INNER JOIN transaction_timing t2
        ON t1.block_number = t2.block_number
        AND t2.transaction_index = t1.transaction_index + 1  -- Immediately after
        AND t1.trader_address != t2.trader_address
        AND t2.gas_price > t1.gas_price * 0.9  -- Similar or higher gas
    INNER JOIN high_gas_traders hgt ON hgt.trader_address = t2.trader_address
    GROUP BY t2.trader_address
    HAVING COUNT(*) >= 3  -- Reduced threshold for sampled data
),

-- Gas war participation (competitive bidding) - optimized
gas_war_participants AS (
    SELECT
        t.trader_address,
        COUNT(*) as high_gas_txs,
        AVG(t.gas_price_ratio) as avg_gas_ratio,
        MAX(CAST(t.gas_price AS DOUBLE) / 1e9) as max_gas_price_gwei
    FROM transaction_timing t
    INNER JOIN high_gas_traders hgt ON hgt.trader_address = t.trader_address
    WHERE t.gas_price_ratio >= 2.0  -- 2x or higher than block median
    GROUP BY t.trader_address
    HAVING COUNT(*) >= 5  -- Reduced threshold
       AND AVG(t.gas_price_ratio) >= 1.5
),

-- Cross-block coordination (sophisticated MEV) - simplified
cross_block_coordination AS (
    SELECT
        trader_address,
        COUNT(DISTINCT block_number) as coordination_patterns
    FROM transaction_timing
    WHERE trader_address IN (SELECT trader_address FROM high_gas_traders)
      AND gas_price_ratio > 1.5
    GROUP BY trader_address
    HAVING COUNT(DISTINCT block_number) >= 10  -- Active across multiple blocks
)

-- Comprehensive MEV bot scoring
SELECT
    COALESCE(
        frp.potential_frontrunner,
        brp.potential_backrunner,
        gwp.trader_address,
        cbc.trader_address
    ) as trader_address,

    -- Pattern indicators
    COALESCE(frp.frontrun_count, 0) as frontrun_instances,
    COALESCE(brp.backrun_count, 0) as backrun_instances,
    COALESCE(gwp.high_gas_txs, 0) as gas_war_instances,
    COALESCE(cbc.coordination_patterns, 0) as coordination_instances,

    -- Performance metrics
    COALESCE(frp.avg_gas_premium, 0) as frontrun_gas_premium,
    COALESCE(brp.avg_gas_premium, 0) as backrun_gas_premium,
    COALESCE(gwp.max_gas_price_gwei, 0) as max_gas_price_gwei,

    -- MEV sophistication score (0-1)
    LEAST(1.0, (
        CASE WHEN frp.frontrun_count > 0 THEN 0.25 ELSE 0 END +
        CASE WHEN brp.backrun_count > 0 THEN 0.25 ELSE 0 END +
        CASE WHEN gwp.high_gas_txs > 0 THEN 0.25 ELSE 0 END +
        CASE WHEN cbc.coordination_patterns > 0 THEN 0.25 ELSE 0 END
    )) as mev_sophistication_score,

    -- Bot classification
    CASE
        WHEN COALESCE(frp.frontrun_count, 0) > 50 AND COALESCE(brp.backrun_count, 0) > 50
            THEN 'Advanced Sandwich Bot'
        WHEN COALESCE(frp.frontrun_count, 0) > 20
            THEN 'Frontrunning Bot'
        WHEN COALESCE(gwp.high_gas_txs, 0) > 50
            THEN 'Gas War Participant'
        WHEN COALESCE(cbc.coordination_patterns, 0) > 50
            THEN 'Sophisticated MEV Bot'
        ELSE 'Basic MEV Activity'
    END as mev_classification,

    -- Risk assessment for wallet filtering
    CASE
        WHEN (
            COALESCE(frp.frontrun_count, 0) +
            COALESCE(brp.backrun_count, 0)
        ) > 100 THEN 'High Risk - Exclude'
        WHEN (
            COALESCE(frp.frontrun_count, 0) +
            COALESCE(brp.backrun_count, 0)
        ) > 25 THEN 'Medium Risk - Review'
        ELSE 'Low Risk - Potential Include'
    END as exclusion_recommendation

FROM front_running_patterns frp
FULL OUTER JOIN back_running_patterns brp ON brp.potential_backrunner = frp.potential_frontrunner
FULL OUTER JOIN gas_war_participants gwp ON gwp.trader_address = COALESCE(frp.potential_frontrunner, brp.potential_backrunner)
FULL OUTER JOIN cross_block_coordination cbc ON cbc.trader_address = COALESCE(frp.potential_frontrunner, brp.potential_backrunner, gwp.trader_address)

WHERE (
    CASE WHEN frp.frontrun_count > 0 THEN 0.25 ELSE 0 END +
    CASE WHEN brp.backrun_count > 0 THEN 0.25 ELSE 0 END +
    CASE WHEN gwp.high_gas_txs > 0 THEN 0.25 ELSE 0 END +
    CASE WHEN cbc.coordination_patterns > 0 THEN 0.25 ELSE 0 END
) >= 0.3  -- Minimum threshold for MEV activity

ORDER BY mev_sophistication_score DESC, frontrun_instances + backrun_instances DESC
LIMIT 10000;