-- Wallet Performance Metrics Table Schema
-- Stores comprehensive trading performance metrics for smart money wallets

CREATE TABLE IF NOT EXISTS wallet_performance (
    wallet_address VARCHAR(42) NOT NULL,
    calculation_date TIMESTAMP NOT NULL DEFAULT NOW(),
    time_period VARCHAR(20) NOT NULL DEFAULT 'all_time',

    -- Basic Performance Metrics
    total_trades INTEGER NOT NULL DEFAULT 0,
    win_rate NUMERIC(5,4) CHECK (win_rate >= 0 AND win_rate <= 1),
    avg_return_per_trade NUMERIC(10,6),
    total_return NUMERIC(10,6),
    annualized_return NUMERIC(10,6),

    -- Risk-Adjusted Metrics
    volatility NUMERIC(10,6),
    sharpe_ratio NUMERIC(10,6),
    sortino_ratio NUMERIC(10,6),
    max_drawdown NUMERIC(10,6),
    var_95 NUMERIC(10,6),
    calmar_ratio NUMERIC(10,6),

    -- Efficiency Metrics
    total_gas_cost_usd NUMERIC(20,8),
    volume_per_gas NUMERIC(20,8),
    net_return_after_costs NUMERIC(10,6),

    -- Diversification Metrics
    unique_tokens_traded INTEGER NOT NULL DEFAULT 0,
    hhi_concentration NUMERIC(10,6),
    max_position_size NUMERIC(10,6),

    -- Additional Metrics
    avg_holding_period_days NUMERIC(10,2),
    profit_factor NUMERIC(10,6),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Primary key on wallet + time_period for multiple period analysis
    PRIMARY KEY (wallet_address, time_period),

    -- Foreign key to wallets table
    FOREIGN KEY (wallet_address) REFERENCES wallets(wallet_address) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_wallet_perf_address ON wallet_performance(wallet_address);
CREATE INDEX IF NOT EXISTS idx_wallet_perf_sharpe ON wallet_performance(sharpe_ratio DESC) WHERE sharpe_ratio IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wallet_perf_return ON wallet_performance(total_return DESC);
CREATE INDEX IF NOT EXISTS idx_wallet_perf_win_rate ON wallet_performance(win_rate DESC);
CREATE INDEX IF NOT EXISTS idx_wallet_perf_calc_date ON wallet_performance(calculation_date DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_wallet_performance_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_wallet_performance_modtime
    BEFORE UPDATE ON wallet_performance
    FOR EACH ROW
    EXECUTE FUNCTION update_wallet_performance_timestamp();

-- Comments for documentation
COMMENT ON TABLE wallet_performance IS 'Comprehensive trading performance metrics for smart money wallets';
COMMENT ON COLUMN wallet_performance.win_rate IS 'Percentage of profitable trades (0-1)';
COMMENT ON COLUMN wallet_performance.sharpe_ratio IS 'Risk-adjusted return metric (annualized)';
COMMENT ON COLUMN wallet_performance.max_drawdown IS 'Maximum peak-to-trough decline';
COMMENT ON COLUMN wallet_performance.hhi_concentration IS 'Herfindahl-Hirschman Index for portfolio concentration';
