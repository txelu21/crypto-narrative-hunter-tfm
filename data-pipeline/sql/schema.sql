-- Idempotent schema creation for data collection pipeline

CREATE TABLE IF NOT EXISTS tokens (
    token_address VARCHAR(42) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    decimals INT NOT NULL CHECK (decimals >= 0 AND decimals <= 18),
    narrative_category VARCHAR(50),
    market_cap_rank INT CHECK (market_cap_rank > 0),
    avg_daily_volume_usd DECIMAL(20,2) CHECK (avg_daily_volume_usd >= 0),
    liquidity_tier INT CHECK (liquidity_tier IN (1, 2, 3)),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wallets (
    wallet_address VARCHAR(42) PRIMARY KEY,
    first_seen_date DATE NOT NULL,
    last_active_date DATE NOT NULL,
    total_trades_30d INT NOT NULL DEFAULT 0 CHECK (total_trades_30d >= 0),
    avg_daily_volume_eth DECIMAL(20,8) CHECK (avg_daily_volume_eth >= 0),
    unique_tokens_traded INT NOT NULL DEFAULT 0 CHECK (unique_tokens_traded >= 0),
    is_smart_money BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_hash VARCHAR(66) PRIMARY KEY,
    block_number BIGINT NOT NULL CHECK (block_number > 0),
    timestamp TIMESTAMP NOT NULL,
    wallet_address VARCHAR(42) NOT NULL,
    dex_name VARCHAR(20) NOT NULL,
    pool_address VARCHAR(42) NOT NULL,
    token_in VARCHAR(42) NOT NULL,
    amount_in DECIMAL(36,18) NOT NULL CHECK (amount_in >= 0),
    token_out VARCHAR(42) NOT NULL,
    amount_out DECIMAL(36,18) NOT NULL CHECK (amount_out >= 0),
    gas_used BIGINT NOT NULL CHECK (gas_used > 0),
    gas_price_gwei DECIMAL(10,2) NOT NULL CHECK (gas_price_gwei >= 0),
    eth_value_in DECIMAL(20,8) CHECK (eth_value_in >= 0),
    eth_value_out DECIMAL(20,8) CHECK (eth_value_out >= 0),
    CONSTRAINT fk_tx_wallet FOREIGN KEY (wallet_address) REFERENCES wallets(wallet_address)
);

CREATE TABLE IF NOT EXISTS wallet_balances (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) NOT NULL,
    token_address VARCHAR(42) NOT NULL,
    snapshot_date DATE NOT NULL,
    balance DECIMAL(36,18) NOT NULL CHECK (balance >= 0),
    eth_value DECIMAL(20,8) CHECK (eth_value >= 0),
    CONSTRAINT fk_bal_wallet FOREIGN KEY (wallet_address) REFERENCES wallets(wallet_address),
    CONSTRAINT fk_bal_token FOREIGN KEY (token_address) REFERENCES tokens(token_address),
    CONSTRAINT uq_wallet_token_date UNIQUE(wallet_address, token_address, snapshot_date)
);

CREATE TABLE IF NOT EXISTS eth_prices (
    timestamp TIMESTAMP PRIMARY KEY,
    price_usd DECIMAL(10,2) NOT NULL CHECK (price_usd > 0),
    source VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_checkpoints (
    id SERIAL PRIMARY KEY,
    collection_type VARCHAR(50) NOT NULL UNIQUE,
    last_processed_block BIGINT CHECK (last_processed_block > 0),
    last_processed_date DATE,
    records_collected INT DEFAULT 0 CHECK (records_collected >= 0),
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_tokens_market_cap_rank ON tokens(market_cap_rank);
CREATE INDEX IF NOT EXISTS idx_tx_wallet ON transactions(wallet_address);
CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_number);
CREATE INDEX IF NOT EXISTS idx_bal_wallet_date ON wallet_balances(wallet_address, snapshot_date);
