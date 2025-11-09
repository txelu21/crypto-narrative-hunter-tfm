-- FIXED: Token Filtering Helper Query
-- Query ID: 5878444
-- Status: Optimized to return only actively traded tokens
-- Purpose: Get basic token information for filtering

-- Use dex.trades to identify legitimate, actively traded tokens
-- This avoids fake/duplicate tokens in tokens.erc20

WITH traded_tokens AS (
    -- Get tokens that are actually traded on DEXs
    SELECT DISTINCT
        token_bought_address as token_address
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_time >= NOW() - INTERVAL '30' day
      AND amount_usd > 1000  -- Only significant trades

    UNION

    SELECT DISTINCT
        token_sold_address as token_address
    FROM dex.trades
    WHERE blockchain = 'ethereum'
      AND block_time >= NOW() - INTERVAL '30' day
      AND amount_usd > 1000
),

token_info AS (
    -- Get token metadata only for actively traded tokens
    SELECT
        t.contract_address,
        t.symbol,
        t.decimals,
        ROW_NUMBER() OVER (PARTITION BY t.symbol ORDER BY t.contract_address) as symbol_rank
    FROM tokens.erc20 t
    INNER JOIN traded_tokens tt ON tt.token_address = t.contract_address
    WHERE t.blockchain = 'ethereum'
      -- Focus on well-known tokens
      AND t.symbol IN ('USDT', 'USDC', 'DAI', 'WETH', 'WBTC', 'LINK', 'UNI', 'AAVE', 'COMP', 'MKR',
                       'SNX', 'CRV', 'BAL', 'YFI', 'SUSHI', 'PEPE', 'SHIB', 'MATIC', 'BNB', 'STETH')
      AND t.decimals > 0
),

latest_prices AS (
    -- Get latest price for each token
    SELECT
        contract_address,
        price as latest_price_usd,
        ROW_NUMBER() OVER (PARTITION BY contract_address ORDER BY minute DESC) as price_rn
    FROM prices.usd
    WHERE blockchain = 'ethereum'
      AND minute >= NOW() - INTERVAL '7' day
      AND contract_address IN (SELECT contract_address FROM token_info WHERE symbol_rank = 1)
)

-- Final result with one token per symbol
SELECT
    ti.contract_address as token_address,
    ti.symbol,
    ti.decimals,
    lp.latest_price_usd,
    'ERC20' as token_type,
    'High Priority' as analysis_priority,
    3 as liquidity_score

FROM token_info ti
LEFT JOIN latest_prices lp ON lp.contract_address = ti.contract_address AND lp.price_rn = 1
WHERE ti.symbol_rank = 1  -- Only one token per symbol (primary/most traded)

ORDER BY ti.symbol
LIMIT 100;