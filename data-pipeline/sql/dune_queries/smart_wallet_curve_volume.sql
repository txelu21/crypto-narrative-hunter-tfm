-- FIXED: Smart Wallet Identification: Curve Trading Volume Analysis
-- Query ID: 5878483
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date), min_volume_usd (number), min_trade_count (number)
-- Dune uses Trino/Presto, not PostgreSQL

WITH
-- Curve trades from dex.trades (curated table)
curve_exchanges AS (
    SELECT
        t.taker as trader_address,
        t.block_time,
        t.tx_hash,
        t.project_contract_address as pool_address,
        t.token_sold_address as tokens_sold,
        t.token_bought_address as tokens_bought,
        t.amount_usd as trade_value_usd,
        -- Get transaction details for gas calculation
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
      AND t.project = 'curve'
      AND t.block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND t.block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND t.taker IS NOT NULL
      AND t.amount_usd > 0
),

-- Curve trades already have USD values from dex.trades
curve_trades_with_values AS (
    SELECT
        e.trader_address,
        e.block_time,
        e.tx_hash,
        e.pool_address,
        e.tokens_sold,
        e.tokens_bought,
        e.gas_cost_usd,
        e.trade_value_usd

    FROM curve_exchanges e
),

-- Exclude known contract addresses
excluded_contracts AS (
    SELECT address FROM (
        VALUES
        -- Curve routers and zaps
        (0x8301AE4fc9c624d1D396cbDAa1ed877821D7C511),  -- Curve.fi Router
        (0xA79828DF1850E8a3A3064576f380D90aECDD3359),  -- Curve.fi Zap

        -- Aggregators that use Curve
        (0x1111111254fb6c44bAC0beD2854e76F90643097d),  -- 1inch V4
        (0xDEF1C0ded9bec7F1a1670819833240f027b25EfF),  -- 0x Protocol

        -- Curve gauge contracts (staking)
        (0x7ca5b0a2910B33e9759DC7dDB0413949071D7575),  -- 3pool gauge
        (0xA90996896660DEcC6E997655E065b23788857849),  -- susd gauge

        -- Null address
        (0x0000000000000000000000000000000000000000)
    ) as t(address)
),

-- Daily trading statistics for bot detection
daily_curve_stats AS (
    SELECT
        trader_address,
        DATE(block_time) as trade_date,
        COUNT(*) as daily_trade_count,
        SUM(trade_value_usd) as daily_volume_usd,
        COUNT(DISTINCT pool_address) as pools_used_daily
    FROM curve_trades_with_values
    WHERE trade_value_usd > 0
    GROUP BY trader_address, DATE(block_time)
),

-- Bot detection for Curve
curve_potential_bots AS (
    SELECT DISTINCT trader_address
    FROM daily_curve_stats
    WHERE daily_trade_count > 50  -- High frequency threshold for Curve
       OR pools_used_daily > 10   -- Suspiciously many pools in one day
),

-- Wallet aggregation with Curve-specific metrics
curve_wallet_metrics AS (
    SELECT
        t.trader_address,
        COUNT(*) as trade_count,
        SUM(t.trade_value_usd) as total_volume_usd,
        AVG(t.trade_value_usd) as avg_trade_size_usd,
        COUNT(DISTINCT t.pool_address) as unique_pools_used,
        COUNT(DISTINCT DATE(t.block_time)) as active_days,
        MIN(t.block_time) as first_trade_time,
        MAX(t.block_time) as last_trade_time,

        -- Efficiency metrics
        SUM(t.trade_value_usd) / NULLIF(COUNT(*), 0) as volume_per_trade,
        SUM(t.trade_value_usd) / NULLIF(SUM(t.gas_cost_usd), 0) as volume_per_gas_dollar,

        -- Activity consistency
        STDDEV(ds.daily_volume_usd) / NULLIF(AVG(ds.daily_volume_usd), 0) as volume_consistency_ratio,

        -- Pool diversity
        COUNT(DISTINCT t.pool_address) / NULLIF(COUNT(DISTINCT DATE(t.block_time)), 0) as pool_diversity_rate

    FROM curve_trades_with_values t
    LEFT JOIN daily_curve_stats ds ON ds.trader_address = t.trader_address
    WHERE t.trade_value_usd > 1  -- Minimum $1 trade size
    GROUP BY t.trader_address
)

-- Final results with filtering
SELECT
    wm.trader_address,
    'Curve' as dex_platform,
    wm.trade_count,
    wm.total_volume_usd,
    wm.avg_trade_size_usd,
    wm.unique_pools_used,
    wm.active_days,
    wm.first_trade_time,
    wm.last_trade_time,

    -- Performance metrics
    wm.volume_per_trade,
    wm.volume_per_gas_dollar,
    wm.volume_consistency_ratio,
    wm.pool_diversity_rate,

    -- Quality score specific to Curve trading
    (
        CASE WHEN wm.total_volume_usd >= COALESCE(TRY_CAST('{{min_volume_usd}}' AS DOUBLE), 25000.0) THEN 0.3 ELSE 0 END +
        CASE WHEN wm.trade_count >= COALESCE(TRY_CAST('{{min_trade_count}}' AS INTEGER), 10) THEN 0.2 ELSE 0 END +
        CASE WHEN wm.unique_pools_used >= 2 THEN 0.2 ELSE 0 END +
        CASE WHEN wm.volume_consistency_ratio <= 2 THEN 0.15 ELSE 0 END +
        CASE WHEN wm.pool_diversity_rate <= 1 THEN 0.15 ELSE 0 END  -- Not using too many pools per day
    ) as quality_score,

    -- Analysis period
    COALESCE(TRY_CAST('{{start_date}}' AS DATE), DATE('2023-01-01')) as analysis_start,
    COALESCE(TRY_CAST('{{end_date}}' AS DATE), DATE('2024-01-01')) as analysis_end

FROM curve_wallet_metrics wm
WHERE
    -- Volume and activity thresholds
    wm.total_volume_usd >= COALESCE(TRY_CAST('{{min_volume_usd}}' AS DOUBLE), 25000.0)
    AND wm.trade_count >= COALESCE(TRY_CAST('{{min_trade_count}}' AS INTEGER), 10)
    AND wm.unique_pools_used >= 1

    -- Exclude contracts and known addresses
    AND wm.trader_address NOT IN (SELECT address FROM excluded_contracts)

    -- Exclude potential bots
    AND wm.trader_address NOT IN (SELECT trader_address FROM curve_potential_bots)

    -- Reasonable activity period
    AND wm.active_days >= 2

ORDER BY wm.total_volume_usd DESC
LIMIT 15000;