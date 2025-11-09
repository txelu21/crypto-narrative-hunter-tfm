-- Wallet Transaction History (Optimized with dex.trades)
-- Extracts DEX swap transactions for a single wallet address
-- Uses Dune's pre-indexed dex.trades table for fast performance
-- Parameters: wallet_address (without 0x prefix), start_date, end_date

SELECT
    tx_hash,
    block_number,
    block_time AS timestamp,
    taker AS wallet_address,
    project AS dex_name,
    project_contract_address AS pool_address,

    -- Token addresses
    token_bought_address AS token_in,
    token_sold_address AS token_out,

    -- Amounts (in raw token units)
    CAST(token_bought_amount_raw AS DOUBLE) AS amount_in,
    CAST(token_sold_amount_raw AS DOUBLE) AS amount_out,

    -- USD values (if available)
    amount_usd,

    -- Transaction details (from dex.trades, gas data not directly available)
    tx_from,
    tx_to

FROM dex.trades

WHERE blockchain = 'ethereum'
    AND (
        taker = FROM_HEX('{{wallet_address}}')
        OR tx_from = FROM_HEX('{{wallet_address}}')
    )
    AND block_time >= CAST('{{start_date}}' AS TIMESTAMP)
    AND block_time <= CAST('{{end_date}}' AS TIMESTAMP)

ORDER BY block_time DESC
LIMIT 10000;
