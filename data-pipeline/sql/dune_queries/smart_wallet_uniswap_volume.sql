-- FIXED: Smart Wallet Identification: Uniswap V2/V3 Volume Analysis
-- Query ID: 5878470
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date), min_volume_usd (number), min_trade_count (number)
-- Dune uses Trino/Presto, not PostgreSQL

WITH
-- Uniswap V2 trades from dex.trades (much faster than raw events!)
uniswap_v2_trades AS (
    SELECT
        t.taker as trader_address,
        t.block_time,
        t.tx_hash,
        t.token_bought_address as token0,
        t.token_sold_address as token1,
        -- Trade value in USD (already calculated in dex.trades)
        t.amount_usd as trade_value_usd,
        -- Token traded for diversity
        t.token_bought_address as token_traded,
        -- Gas data
        tx.gas_used,
        tx.gas_price,
        (CAST(tx.gas_used AS DOUBLE) * CAST(tx.gas_price AS DOUBLE) / 1e18) * COALESCE(eth_price.price, 0) as gas_cost_usd
    FROM dex.trades t
    LEFT JOIN ethereum.transactions tx
        ON tx.hash = t.tx_hash
        AND tx.block_time = t.block_time
    LEFT JOIN prices.usd eth_price ON eth_price.contract_address = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
        AND eth_price.minute = DATE_TRUNC('minute', t.block_time)
        AND eth_price.blockchain = 'ethereum'
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
      AND t.taker IS NOT NULL
      -- Exclude zero-value trades
      AND t.amount_usd > 0
),

-- Uniswap V3 trades
uniswap_v3_trades AS (
    SELECT
        t.taker as trader_address,
        t.block_time,
        t.tx_hash,
        t.token_bought_address as token0,
        t.token_sold_address as token1,
        -- Trade value in USD (already calculated in dex.trades)
        t.amount_usd as trade_value_usd,
        -- Token traded for diversity
        t.token_bought_address as token_traded,
        -- Gas data
        tx.gas_used,
        tx.gas_price,
        (CAST(tx.gas_used AS DOUBLE) * CAST(tx.gas_price AS DOUBLE) / 1e18) * COALESCE(eth_price.price, 0) as gas_cost_usd
    FROM dex.trades t
    LEFT JOIN ethereum.transactions tx
        ON tx.hash = t.tx_hash
        AND tx.block_time = t.block_time
    LEFT JOIN prices.usd eth_price ON eth_price.contract_address = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
        AND eth_price.minute = DATE_TRUNC('minute', t.block_time)
        AND eth_price.blockchain = 'ethereum'
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
      AND t.taker IS NOT NULL
      -- Exclude zero-value trades
      AND t.amount_usd > 0
),

-- Combined Uniswap trades
all_uniswap_trades AS (
    SELECT * FROM uniswap_v2_trades
    UNION ALL
    SELECT * FROM uniswap_v3_trades
),

-- Known contract addresses to exclude (routers, aggregators, etc.)
excluded_contracts AS (
    SELECT address FROM (
        VALUES
        -- Uniswap routers
        (0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D),  -- V2 Router
        (0xE592427A0AEce92De3Edee1F18E0157C05861564),  -- V3 Router
        (0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45),  -- V3 Router 2

        -- 1inch routers
        (0x1111111254fb6c44bAC0beD2854e76F90643097d),  -- 1inch V4
        (0x11111112542D85B3EF69AE05771c2dCCff4fAa26),  -- 1inch V3

        -- Other major aggregators
        (0xDEF1C0ded9bec7F1a1670819833240f027b25EfF),  -- 0x Protocol
        (0x6131B5fae19EA4f9D964eAc0408E4408b66337b5),  -- Kyber Router

        -- Known MEV bots
        (0x0000000000000000000000000000000000000000)   -- Null address
    ) as t(address)
),

-- Daily trading statistics for bot detection
daily_stats AS (
    SELECT
        trader_address,
        DATE(block_time) as trade_date,
        COUNT(*) as daily_trade_count,
        SUM(trade_value_usd) as daily_volume_usd,
        AVG(gas_cost_usd / NULLIF(trade_value_usd, 0)) as avg_gas_efficiency
    FROM all_uniswap_trades
    WHERE trade_value_usd > 0
    GROUP BY trader_address, DATE(block_time)
),

-- Bot detection heuristics
potential_bots AS (
    SELECT DISTINCT trader_address
    FROM daily_stats
    WHERE daily_trade_count > 100  -- High frequency trading threshold
       OR avg_gas_efficiency > 0.05  -- Suspiciously high gas usage relative to trade value
),

-- Wallet aggregation with performance metrics (OPTIMIZED - no cartesian join!)
wallet_metrics AS (
    SELECT
        t.trader_address,
        COUNT(*) as trade_count,
        SUM(t.trade_value_usd) as total_volume_usd,
        AVG(t.trade_value_usd) as avg_trade_size_usd,
        COUNT(DISTINCT t.token_traded) as unique_tokens_traded,
        COUNT(DISTINCT DATE(t.block_time)) as active_days,
        MIN(t.block_time) as first_trade_time,
        MAX(t.block_time) as last_trade_time,

        -- Volume efficiency metrics
        SUM(t.trade_value_usd) / NULLIF(COUNT(*), 0) as volume_per_trade,
        SUM(t.trade_value_usd) / NULLIF(SUM(t.gas_cost_usd), 0) as volume_per_gas_dollar,

        -- Gas efficiency
        AVG(t.gas_cost_usd / NULLIF(t.trade_value_usd, 0)) as avg_gas_efficiency_ratio,

        -- Token diversity rate
        COUNT(DISTINCT t.token_traded) / NULLIF(COUNT(DISTINCT DATE(t.block_time)), 0) as token_diversity_rate

    FROM all_uniswap_trades t
    WHERE t.trade_value_usd > 1  -- Minimum $1 trade size
    GROUP BY t.trader_address
),

-- Calculate volume consistency from daily_stats (separate to avoid cartesian join)
volume_consistency AS (
    SELECT
        trader_address,
        STDDEV(daily_volume_usd) / NULLIF(AVG(daily_volume_usd), 0) as volume_consistency_ratio
    FROM daily_stats
    GROUP BY trader_address
)

-- Final results with filtering
SELECT
    wm.trader_address,
    'Uniswap' as dex_platform,
    wm.trade_count,
    wm.total_volume_usd,
    wm.avg_trade_size_usd,
    wm.unique_tokens_traded,
    wm.active_days,
    wm.first_trade_time,
    wm.last_trade_time,

    -- Performance metrics
    wm.volume_per_trade,
    wm.volume_per_gas_dollar,
    COALESCE(vc.volume_consistency_ratio, 0) as volume_consistency_ratio,
    wm.avg_gas_efficiency_ratio,
    wm.token_diversity_rate,

    -- Quality score (composite metric)
    (
        CASE WHEN wm.total_volume_usd >= COALESCE(TRY_CAST('{{min_volume_usd}}' AS DOUBLE), 25000.0) THEN 0.3 ELSE 0 END +
        CASE WHEN wm.trade_count >= COALESCE(TRY_CAST('{{min_trade_count}}' AS INTEGER), 10) THEN 0.2 ELSE 0 END +
        CASE WHEN wm.unique_tokens_traded >= 3 THEN 0.2 ELSE 0 END +
        CASE WHEN COALESCE(vc.volume_consistency_ratio, 0) <= 2 THEN 0.15 ELSE 0 END +  -- Not too erratic
        CASE WHEN wm.avg_gas_efficiency_ratio <= 0.02 THEN 0.15 ELSE 0 END -- Reasonable gas usage
    ) as quality_score,

    -- Analysis period
    COALESCE(TRY_CAST('{{start_date}}' AS DATE), DATE('2023-01-01')) as analysis_start,
    COALESCE(TRY_CAST('{{end_date}}' AS DATE), DATE('2024-01-01')) as analysis_end

FROM wallet_metrics wm
LEFT JOIN volume_consistency vc ON vc.trader_address = wm.trader_address
WHERE
    -- Volume and activity thresholds
    wm.total_volume_usd >= COALESCE(TRY_CAST('{{min_volume_usd}}' AS DOUBLE), 25000.0)
    AND wm.trade_count >= COALESCE(TRY_CAST('{{min_trade_count}}' AS INTEGER), 10)
    AND wm.unique_tokens_traded >= 3

    -- Exclude contracts and known addresses
    AND wm.trader_address NOT IN (SELECT address FROM excluded_contracts)

    -- Exclude potential bots
    AND wm.trader_address NOT IN (SELECT trader_address FROM potential_bots)

    -- Reasonable activity period (not just one day of trading)
    AND wm.active_days >= 2

ORDER BY wm.total_volume_usd DESC
LIMIT 15000;