-- FIXED: Uniswap V3 Pool Discovery Query
-- Query ID: 5878390
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date)
-- Dune uses Trino/Presto, not PostgreSQL

-- Uniswap V3 uses concentrated liquidity, so TVL calculation is different
-- This version focuses on getting basic pool info with liquidity

WITH v3_pools AS (
    -- Get all Uniswap V3 pools
    SELECT DISTINCT
        pool as pool_address,
        token0,
        token1,
        fee,
        evt_block_time as created_at
    FROM uniswap_v3_ethereum.factory_evt_poolcreated
    WHERE evt_block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
    LIMIT 50000
),

-- Use dex.trades for Uniswap V3 activity (curated table with proper schema)
pool_state AS (
    SELECT
        t.project_contract_address as pool_address,
        t.block_time as evt_block_time,
        ROW_NUMBER() OVER (PARTITION BY t.project_contract_address ORDER BY t.block_time DESC) as rn
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
),

latest_pool_state AS (
    SELECT
        pool_address,
        evt_block_time
    FROM pool_state
    WHERE rn = 1
),

-- Get recent swap activity for volume calculation
pool_swaps AS (
    SELECT
        t.project_contract_address as pool_address,
        COUNT(*) as swap_count_24h,
        SUM(t.amount_usd) as volume_24h_usd
    FROM dex.trades t
    WHERE t.blockchain = 'ethereum'
      AND t.project = 'uniswap'
      AND t.version = '3'
      AND t.block_time >= NOW() - INTERVAL '24' hour
    GROUP BY t.project_contract_address
),

-- Get token prices
token_prices_raw AS (
    SELECT
        contract_address,
        price as price_usd,
        minute as price_time,
        ROW_NUMBER() OVER (PARTITION BY contract_address ORDER BY minute DESC) as price_rn
    FROM prices.usd
    WHERE blockchain = 'ethereum'
      AND minute >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND minute <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
),

token_prices AS (
    SELECT
        contract_address,
        price_usd,
        price_time
    FROM token_prices_raw
    WHERE price_rn = 1
)

-- Final result
SELECT
    vp.pool_address,
    'Uniswap V3' as dex_name,
    vp.token0,
    vp.token1,
    vp.fee / 10000.0 as fee_percent,  -- Convert to percentage

    -- Volume metrics
    COALESCE(psw.swap_count_24h, 0) as swap_count_24h,
    COALESCE(psw.volume_24h_usd, 0) as volume_24h_usd,

    -- Token prices in USD
    COALESCE(tp0.price_usd, 0) as token0_price_usd,
    COALESCE(tp1.price_usd, 0) as token1_price_usd,

    -- Price ratio (token0/token1)
    CASE
        WHEN tp1.price_usd > 0
        THEN tp0.price_usd / tp1.price_usd
        ELSE NULL
    END as token0_price_in_token1,

    -- Price in ETH if one token is WETH
    CASE
        WHEN vp.token1 = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
        THEN tp0.price_usd / NULLIF(tp1.price_usd, 0)
        WHEN vp.token0 = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
        THEN tp1.price_usd / NULLIF(tp0.price_usd, 0)
        ELSE NULL
    END as price_eth,

    ps.evt_block_time as last_updated,
    vp.created_at

FROM v3_pools vp
LEFT JOIN latest_pool_state ps ON ps.pool_address = vp.pool_address
LEFT JOIN pool_swaps psw ON psw.pool_address = vp.pool_address
LEFT JOIN token_prices tp0 ON tp0.contract_address = vp.token0
LEFT JOIN token_prices tp1 ON tp1.contract_address = vp.token1

-- Filter for active pools
WHERE psw.swap_count_24h > 0

ORDER BY psw.swap_count_24h DESC NULLS LAST
LIMIT 1000;
