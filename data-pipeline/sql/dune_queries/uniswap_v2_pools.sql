-- FIXED: Uniswap V2 Pool Discovery Query
-- Query ID: 5878381
-- Status: Optimized to use dex.trades for performance
-- Parameters: start_date (date), end_date (date)

-- This query discovers Uniswap V2 pools with TVL and volume for Ethereum tokens
-- Optimized to start with active pools from dex.trades, then enrich with reserves

WITH active_pools AS (
    -- Use dex.trades to identify pools with recent activity (fast, curated table)
    SELECT
        t.project_contract_address as pair_address,
        COUNT(*) as swap_count_24h,
        SUM(t.amount_usd) as volume_24h_usd
    FROM dex.trades t
    WHERE t.blockchain = 'ethereum'
      AND t.project = 'uniswap'
      AND t.version = '2'
      AND t.block_time >= NOW() - INTERVAL '24' hour
      AND t.amount_usd > 0
    GROUP BY t.project_contract_address
),

pool_tokens AS (
    -- Get token0/token1 only for active pools (not all 50k pairs)
    SELECT DISTINCT
        pair as pair_address,
        token0,
        token1
    FROM uniswap_v2_ethereum.Factory_evt_PairCreated
    WHERE pair IN (SELECT pair_address FROM active_pools)
),

latest_reserves AS (
    -- Get most recent reserve state only for active pools
    SELECT
        contract_address as pair_address,
        reserve0,
        reserve1,
        evt_block_time,
        ROW_NUMBER() OVER (PARTITION BY contract_address ORDER BY evt_block_time DESC) as rn
    FROM uniswap_v2_ethereum.Pair_evt_Sync
    WHERE contract_address IN (SELECT pair_address FROM active_pools)
      AND evt_block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND evt_block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
),

pool_with_reserves AS (
    -- Join active pools with their latest reserves
    SELECT
        pt.pair_address,
        pt.token0,
        pt.token1,
        lr.reserve0,
        lr.reserve1,
        lr.evt_block_time as last_updated
    FROM pool_tokens pt
    INNER JOIN latest_reserves lr ON lr.pair_address = pt.pair_address AND lr.rn = 1
    WHERE lr.reserve0 > 0 AND lr.reserve1 > 0  -- Only pools with liquidity
),

-- Get token prices only for tokens in active pools (optimized scope)
active_tokens AS (
    SELECT DISTINCT token0 as token_address FROM pool_tokens
    UNION
    SELECT DISTINCT token1 as token_address FROM pool_tokens
),

token_prices AS (
    SELECT
        contract_address,
        price as price_usd,
        minute as price_time,
        ROW_NUMBER() OVER (PARTITION BY contract_address ORDER BY minute DESC) as price_rn
    FROM prices.usd
    WHERE blockchain = 'ethereum'
      AND contract_address IN (SELECT token_address FROM active_tokens)
      AND minute >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND minute <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
),

latest_token_prices AS (
    -- Get only the latest price for each token
    SELECT
        contract_address,
        price_usd,
        price_time
    FROM token_prices
    WHERE price_rn = 1
)

-- Final result with TVL calculations
SELECT
    p.pair_address,
    'Uniswap V2' as dex_name,
    p.token0,
    p.token1,
    p.reserve0,
    p.reserve1,

    -- Calculate TVL in USD (simple approximation)
    COALESCE(
        (CAST(p.reserve0 AS DOUBLE) / 1e18) * COALESCE(tp0.price_usd, 0) +
        (CAST(p.reserve1 AS DOUBLE) / 1e18) * COALESCE(tp1.price_usd, 0),
        0
    ) as tvl_usd,

    -- 24h volume from dex.trades (pre-calculated in USD)
    COALESCE(ap.volume_24h_usd, 0) as volume_24h_usd,
    COALESCE(ap.swap_count_24h, 0) as swap_count_24h,

    -- Price information
    CASE
        -- If token1 is WETH (0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2)
        WHEN p.token1 = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
        THEN (CAST(p.reserve1 AS DOUBLE) / CAST(p.reserve0 AS DOUBLE))
        -- If token0 is WETH
        WHEN p.token0 = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
        THEN (CAST(p.reserve0 AS DOUBLE) / CAST(p.reserve1 AS DOUBLE))
        ELSE NULL
    END as price_eth,

    p.last_updated

FROM pool_with_reserves p
INNER JOIN active_pools ap ON ap.pair_address = p.pair_address
LEFT JOIN latest_token_prices tp0 ON tp0.contract_address = p.token0
LEFT JOIN latest_token_prices tp1 ON tp1.contract_address = p.token1

-- Filter for pools with meaningful liquidity
WHERE (
    (CAST(p.reserve0 AS DOUBLE) / 1e18) * COALESCE(tp0.price_usd, 0) +
    (CAST(p.reserve1 AS DOUBLE) / 1e18) * COALESCE(tp1.price_usd, 0)
) > 1000  -- Min $1k TVL

ORDER BY tvl_usd DESC
LIMIT 1000;