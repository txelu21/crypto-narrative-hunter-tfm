-- Wallet Token Balances Table Schema
-- Stores daily balance snapshots for smart money wallets
-- Enables accumulation/distribution pattern analysis

-- Drop existing table if needed (for clean migration)
-- DROP TABLE IF EXISTS wallet_token_balances CASCADE;

CREATE TABLE IF NOT EXISTS wallet_token_balances (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) NOT NULL,
    token_address VARCHAR(42) NOT NULL,
    snapshot_date DATE NOT NULL,
    block_number BIGINT NOT NULL,

    -- Balance data
    balance_raw NUMERIC(78,0), -- Raw balance in smallest unit (wei equivalent)
    balance_formatted NUMERIC(30,18), -- Human-readable balance with decimals

    -- Token metadata (denormalized for performance)
    token_symbol VARCHAR(20),
    token_name VARCHAR(100),
    decimals INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Unique constraint: one balance per wallet-token-date combination
    CONSTRAINT unique_wallet_token_snapshot UNIQUE(wallet_address, token_address, snapshot_date),

    -- Foreign keys
    FOREIGN KEY (wallet_address) REFERENCES wallets(wallet_address) ON DELETE CASCADE,
    FOREIGN KEY (token_address) REFERENCES tokens(token_address) ON DELETE CASCADE
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_balances_wallet ON wallet_token_balances(wallet_address);
CREATE INDEX IF NOT EXISTS idx_balances_token ON wallet_token_balances(token_address);
CREATE INDEX IF NOT EXISTS idx_balances_date ON wallet_token_balances(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_balances_block ON wallet_token_balances(block_number);
CREATE INDEX IF NOT EXISTS idx_balances_wallet_date ON wallet_token_balances(wallet_address, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_balances_wallet_token ON wallet_token_balances(wallet_address, token_address);

-- Composite index for time-series queries
CREATE INDEX IF NOT EXISTS idx_balances_wallet_token_date ON wallet_token_balances(wallet_address, token_address, snapshot_date);

-- Index for non-zero balances only (most queries filter out zero balances)
CREATE INDEX IF NOT EXISTS idx_balances_nonzero ON wallet_token_balances(wallet_address, token_address)
    WHERE balance_raw > 0;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_balance_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_balance_modtime
    BEFORE UPDATE ON wallet_token_balances
    FOR EACH ROW
    EXECUTE FUNCTION update_balance_timestamp();

-- Comments for documentation
COMMENT ON TABLE wallet_token_balances IS 'Daily balance snapshots for wallet token holdings - enables accumulation/distribution analysis';
COMMENT ON COLUMN wallet_token_balances.balance_raw IS 'Raw balance in smallest unit (e.g., wei for ETH, base units for ERC20)';
COMMENT ON COLUMN wallet_token_balances.balance_formatted IS 'Human-readable balance with decimal adjustment';
COMMENT ON COLUMN wallet_token_balances.snapshot_date IS 'Date of balance snapshot (daily granularity)';
COMMENT ON COLUMN wallet_token_balances.block_number IS 'Ethereum block number at which balance was captured';

-- Create materialized view for daily portfolio summaries
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_portfolio_summary AS
SELECT
    wallet_address,
    snapshot_date,
    COUNT(DISTINCT token_address) as num_tokens_held,
    SUM(CASE WHEN balance_raw > 0 THEN 1 ELSE 0 END) as num_nonzero_positions,
    -- Calculate HHI (Herfindahl-Hirschman Index) for concentration
    -- Note: This is a simplified version, full HHI needs value-weighted calculation
    COUNT(DISTINCT token_address) as portfolio_diversity
FROM wallet_token_balances
WHERE balance_raw > 0
GROUP BY wallet_address, snapshot_date;

CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_summary_wallet_date
    ON daily_portfolio_summary(wallet_address, snapshot_date);

COMMENT ON MATERIALIZED VIEW daily_portfolio_summary IS 'Aggregated daily portfolio metrics per wallet - refresh after data collection';

-- Helper view: Latest balances per wallet
CREATE OR REPLACE VIEW latest_wallet_balances AS
SELECT DISTINCT ON (wallet_address, token_address)
    wallet_address,
    token_address,
    snapshot_date,
    block_number,
    balance_raw,
    balance_formatted,
    token_symbol
FROM wallet_token_balances
WHERE balance_raw > 0
ORDER BY wallet_address, token_address, snapshot_date DESC;

COMMENT ON VIEW latest_wallet_balances IS 'Most recent non-zero balance for each wallet-token pair';
