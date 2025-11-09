-- SQL Migration: Enhance Token Metrics Schema
-- Date: 2025-10-08
-- Purpose: Add columns for token holder count, market metrics, and derived calculations
-- Reference: docs/FEATURE_ENGINEERING_SPEC.md Section 4.2

-- Add new columns to tokens table
-- Using IF NOT EXISTS pattern for idempotency

DO $$
BEGIN
    -- Holder count (from Etherscan or CoinGecko)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'holder_count'
    ) THEN
        ALTER TABLE tokens ADD COLUMN holder_count INTEGER;
        COMMENT ON COLUMN tokens.holder_count IS 'Number of unique token holders (Etherscan/CoinGecko)';
    END IF;

    -- Circulating supply
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'circulating_supply'
    ) THEN
        ALTER TABLE tokens ADD COLUMN circulating_supply NUMERIC(30,2);
        COMMENT ON COLUMN tokens.circulating_supply IS 'Tokens currently in circulation (CoinGecko)';
    END IF;

    -- Current market cap
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'current_market_cap'
    ) THEN
        ALTER TABLE tokens ADD COLUMN current_market_cap NUMERIC(30,2);
        COMMENT ON COLUMN tokens.current_market_cap IS 'Market cap based on circulating supply (USD)';
    END IF;

    -- Current price
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'current_price_usd'
    ) THEN
        ALTER TABLE tokens ADD COLUMN current_price_usd NUMERIC(18,8);
        COMMENT ON COLUMN tokens.current_price_usd IS 'Current token price in USD';
    END IF;

    -- Fully Diluted Valuation (FDV)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'fdv'
    ) THEN
        ALTER TABLE tokens ADD COLUMN fdv NUMERIC(30,2);
        COMMENT ON COLUMN tokens.fdv IS 'Fully Diluted Valuation = total_supply × current_price_usd';
    END IF;

    -- Volume/Market Cap ratio
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'volume_mc_ratio'
    ) THEN
        ALTER TABLE tokens ADD COLUMN volume_mc_ratio NUMERIC(10,4);
        COMMENT ON COLUMN tokens.volume_mc_ratio IS 'Volume/Market Cap ratio (liquidity indicator)';
    END IF;

    -- 24h price change %
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'price_change_24h_pct'
    ) THEN
        ALTER TABLE tokens ADD COLUMN price_change_24h_pct NUMERIC(10,4);
        COMMENT ON COLUMN tokens.price_change_24h_pct IS '24-hour price change percentage';
    END IF;

    -- 24h trading volume
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'volume_24h'
    ) THEN
        ALTER TABLE tokens ADD COLUMN volume_24h NUMERIC(30,2);
        COMMENT ON COLUMN tokens.volume_24h IS '24-hour trading volume (USD) from CoinGecko';
    END IF;

    -- Metrics update timestamp
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tokens' AND column_name = 'metrics_updated_at'
    ) THEN
        ALTER TABLE tokens ADD COLUMN metrics_updated_at TIMESTAMP;
        COMMENT ON COLUMN tokens.metrics_updated_at IS 'Timestamp of last metrics update';
    END IF;

END $$;

-- Create index on holder_count for filtering
CREATE INDEX IF NOT EXISTS idx_tokens_holder_count ON tokens(holder_count);

-- Create index on fdv for sorting/filtering
CREATE INDEX IF NOT EXISTS idx_tokens_fdv ON tokens(fdv);

-- Create index on volume_mc_ratio for analysis
CREATE INDEX IF NOT EXISTS idx_tokens_volume_mc_ratio ON tokens(volume_mc_ratio);

-- Verify migration
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'tokens'
    AND column_name IN (
        'holder_count',
        'circulating_supply',
        'current_market_cap',
        'current_price_usd',
        'fdv',
        'volume_mc_ratio',
        'price_change_24h_pct',
        'volume_24h',
        'metrics_updated_at'
    )
ORDER BY ordinal_position;

-- Show sample of existing tokens (before enhancement)
SELECT
    symbol,
    name,
    narrative_category,
    current_price_usd,
    current_market_cap,
    fdv,
    holder_count
FROM tokens
LIMIT 5;
