-- Wallet Transaction History - BATCH VERSION with Contract Filtering
-- Extracts DEX transactions for MULTIPLE wallets at once
-- Uses dex.trades for performance with pagination support
--
-- Filtering Layers:
-- 1. Contract Exclusion: Removes smart contracts (only EOAs remain)
-- 2. Transaction Limiting: Max 2000 txs per wallet (filters extreme HFT)
--
-- Parameters: start_date, end_date, batch_size (default 10), batch_offset (default 0)

WITH numbered_wallets AS (
    -- Add row numbers to wallet dataset for pagination
    SELECT
        FROM_HEX(address) as wallet_address,
        ROW_NUMBER() OVER (ORDER BY address) as row_num
    FROM dune.tokencluster_team_9042.dataset_wallet_addresses
),
filtered_wallets AS (
    -- LAYER 1: Exclude contracts (only keep EOAs)
    -- This focuses on genuine retail EOA traders
    SELECT
        nw.wallet_address,
        nw.row_num
    FROM numbered_wallets nw
    -- Exclude smart contracts (only keep EOAs)
    -- Use creation_traces which has ALL contracts, not just decoded ones
    LEFT JOIN ethereum.creation_traces ct
        ON nw.wallet_address = ct.address
    WHERE ct.address IS NULL  -- Not a contract
),
target_wallets AS (
    -- Select wallet batch using row number filtering
    SELECT wallet_address
    FROM filtered_wallets
    WHERE row_num > {{batch_offset}}
      AND row_num <= {{batch_offset}} + {{batch_size}}
),
ranked_transactions AS (
    -- LAYER 2: Rank transactions per wallet to limit high-frequency traders
    -- Caps at 2000 most recent transactions per wallet (filters extreme HFT/MEV)
    SELECT
        tx_hash,
        block_number,
        block_time AS timestamp,
        taker AS wallet_address,
        project AS dex_name,
        project_contract_address AS pool_address,
        token_bought_address AS token_in,
        token_sold_address AS token_out,
        CAST(token_bought_amount_raw AS DOUBLE) AS amount_in,
        CAST(token_sold_amount_raw AS DOUBLE) AS amount_out,
        amount_usd,
        ROW_NUMBER() OVER (PARTITION BY taker ORDER BY block_time DESC) as tx_rank
    FROM dex.trades
    WHERE blockchain = 'ethereum'
        AND block_time >= CAST('{{start_date}}' AS TIMESTAMP)
        AND block_time <= CAST('{{end_date}}' AS TIMESTAMP)
        AND taker IN (SELECT wallet_address FROM target_wallets)
)

SELECT
    tx_hash,
    block_number,
    timestamp,
    wallet_address,
    dex_name,
    pool_address,
    token_in,
    token_out,
    amount_in,
    amount_out,
    amount_usd
FROM ranked_transactions
WHERE tx_rank <= 2000  -- Limit to 2000 most recent transactions per wallet (~67 tx/day avg)
ORDER BY timestamp DESC;
