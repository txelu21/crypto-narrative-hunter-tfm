-- Database schema updates for DEX liquidity analysis
-- Updates tokens table and creates token_pools table

-- Update tokens table to support liquidity tiers
ALTER TABLE tokens
DROP CONSTRAINT IF EXISTS tokens_liquidity_tier_check;

ALTER TABLE tokens
DROP COLUMN IF EXISTS liquidity_tier;

ALTER TABLE tokens
ADD COLUMN liquidity_tier VARCHAR(10)
CHECK (liquidity_tier IN ('Tier 1', 'Tier 2', 'Tier 3', 'Untiered'));

-- Create index for liquidity tier queries
DROP INDEX IF EXISTS idx_tokens_liquidity_tier;
CREATE INDEX idx_tokens_liquidity_tier ON tokens(liquidity_tier);

-- Create token_pools table for tracking discovered liquidity sources
CREATE TABLE IF NOT EXISTS token_pools (
    id SERIAL PRIMARY KEY,
    token_address VARCHAR(42) NOT NULL,
    pool_address VARCHAR(42) NOT NULL,
    dex_name VARCHAR(20) NOT NULL,
    pair_token VARCHAR(42) NOT NULL,
    tvl_usd DECIMAL(20,2) CHECK (tvl_usd >= 0),
    volume_24h_usd DECIMAL(20,2) CHECK (volume_24h_usd >= 0),
    price_eth DECIMAL(36,18) CHECK (price_eth >= 0),
    last_updated TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT fk_pool_token FOREIGN KEY (token_address) REFERENCES tokens(token_address),
    CONSTRAINT uq_token_pool UNIQUE(token_address, pool_address)
);

-- Indexes for efficient pool queries
CREATE INDEX IF NOT EXISTS idx_token_pools_token ON token_pools(token_address);
CREATE INDEX IF NOT EXISTS idx_token_pools_dex ON token_pools(dex_name);
CREATE INDEX IF NOT EXISTS idx_token_pools_tvl ON token_pools(tvl_usd DESC);
CREATE INDEX IF NOT EXISTS idx_token_pools_volume ON token_pools(volume_24h_usd DESC);
CREATE INDEX IF NOT EXISTS idx_token_pools_updated ON token_pools(last_updated);

-- Create pool rankings view for easy access to best pools per token
CREATE OR REPLACE VIEW token_pool_rankings AS
SELECT
    tp.*,
    ROW_NUMBER() OVER (
        PARTITION BY tp.token_address
        ORDER BY tp.tvl_usd DESC, tp.volume_24h_usd DESC
    ) as pool_rank
FROM token_pools tp
WHERE tp.tvl_usd > 0;

-- Create tier summary view
CREATE OR REPLACE VIEW liquidity_tier_summary AS
SELECT
    liquidity_tier,
    COUNT(*) as token_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM tokens
WHERE liquidity_tier IS NOT NULL
GROUP BY liquidity_tier
ORDER BY
    CASE liquidity_tier
        WHEN 'Tier 1' THEN 1
        WHEN 'Tier 2' THEN 2
        WHEN 'Tier 3' THEN 3
        WHEN 'Untiered' THEN 4
    END;

-- Update collection_checkpoints to support liquidity analysis
INSERT INTO collection_checkpoints (collection_type, status)
VALUES ('liquidity_analysis', 'pending')
ON CONFLICT (collection_type) DO NOTHING;

-- Add metadata column to collection_checkpoints if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'collection_checkpoints'
        AND column_name = 'metadata'
    ) THEN
        ALTER TABLE collection_checkpoints ADD COLUMN metadata JSONB;
    END IF;
END $$;