# Feature Engineering Specification
**Crypto Narrative Hunter - Master Thesis**

**Version:** 1.0
**Date:** October 8, 2025
**Status:** Active - Technical Reference

---

## Purpose

This document defines all variables (raw and derived) used in wallet clustering and behavioral analysis. It directly addresses **Tutor Question 3: "What variables will you use?"** by providing comprehensive variable definitions, calculation formulas, justifications, and expected ranges.

---

## Overview

**Feature Categories:**
1. **Raw Variables** - Directly from collected data (CSV exports, database tables)
2. **Derived Variables** - Calculated from raw data (Epic 4.1)
3. **Clustering Features** - Subset of variables used for wallet clustering (Epic 4.3)
4. **Token-Level Variables** - For Tier 3 analysis and context (Phase 2 enhancement)

**Total Variable Count:**
- Raw: 42 variables
- Derived: 28 variables
- Clustering: 18 features (normalized)

---

## 1. Raw Variables (Collected Data)

### 1.1 Transaction-Level Variables
**Source:** `transactions.csv` (34,034 records)

| Variable | Type | Description | Example | Used In |
|----------|------|-------------|---------|---------|
| `tx_hash` | VARCHAR(66) | Unique transaction identifier | 0xbd2361c9a... | Linkage |
| `block_number` | INTEGER | Ethereum block number | 23500676 | Temporal analysis |
| `timestamp` | TIMESTAMP | Transaction execution time | 2025-10-03 23:56:11 | Time series |
| `wallet_address` | VARCHAR(42) | Trading wallet address | 0x22c45fb5... | Aggregation key |
| `dex_name` | VARCHAR(20) | DEX protocol (uniswap, curve, balancer) | uniswap | DEX preference |
| `pool_address` | VARCHAR(42) | Liquidity pool address | 0x00000000... | Pool linkage |
| `token_in` | VARCHAR(42) | Token address sold | 0x000000... | Portfolio flow |
| `amount_in` | NUMERIC | Token amount sold (raw) | 4.239e+17 | Trade sizing |
| `token_out` | VARCHAR(42) | Token address bought | 0xa0b86991... | Portfolio flow |
| `amount_out` | NUMERIC | Token amount bought (raw) | 1916005400.0 | Trade sizing |
| `gas_used` | INTEGER | Gas consumed | 339232 | Gas efficiency |
| `gas_price_gwei` | NUMERIC(10,2) | Gas price in Gwei | 0.21 | Gas cost |
| `transaction_status` | VARCHAR(10) | success/failed | success | Win rate calc |
| `eth_value_in` | NUMERIC(18,8) | ETH value sold | (calculated) | USD valuation |
| `eth_value_out` | NUMERIC(18,8) | ETH value bought | (calculated) | USD valuation |
| `usd_value_in` | NUMERIC(18,2) | USD value sold | (calculated) | Performance |
| `usd_value_out` | NUMERIC(18,2) | USD value bought | (calculated) | Performance |

**Note:** `eth_value_*` and `usd_value_*` to be calculated in Epic 4.1 using token prices and ETH/USD rates.

---

### 1.2 Token-Level Variables
**Source:** `tokens.csv` (500 records)

| Variable | Type | Description | Example | Used In |
|----------|------|-------------|---------|---------|
| `token_address` | VARCHAR(42) | Ethereum contract address | 0xdac17f95... | Linkage |
| `symbol` | VARCHAR(20) | Token ticker | USDT | Display |
| `name` | VARCHAR(100) | Full token name | Tether | Display |
| `decimals` | INTEGER | Decimal places (0-18) | 6 | Amount conversion |
| `narrative_category` | VARCHAR(50) | AI, DeFi, Gaming, Meme, Infrastructure, Stablecoin, Other | Stablecoin | Clustering |
| `market_cap_rank` | INTEGER | CoinGecko ranking | 3 | Tier 3 analysis |
| `avg_daily_volume_usd` | NUMERIC(20,2) | 24h trading volume (USD) | 50697516882.0 | Liquidity tier |
| `liquidity_tier` | VARCHAR(10) | Tier 1/2/3 (based on TVL) | Tier 2 | Filtering |
| `classification_confidence` | INTEGER | 0-100 confidence score | 95 | Manual review flag |
| `requires_manual_review` | BOOLEAN | Low confidence flag | True | Data quality |

**Phase 2 Enhancement (to be added):**
| Variable | Type | Description | Calculation | Used In |
|----------|------|-------------|-------------|---------|
| `holder_count` | INTEGER | Number of unique holders | Etherscan API | Token popularity |
| `circulating_supply` | NUMERIC(20,2) | Tokens in circulation | CoinGecko API | Market cap calc |
| `current_market_cap` | NUMERIC(20,2) | Market cap (USD) | CoinGecko API | Tier 3 analysis |
| `current_price_usd` | NUMERIC(18,8) | Current token price | CoinGecko API | Valuation |
| `fdv` | NUMERIC(20,2) | Fully Diluted Valuation | total_supply × current_price_usd | Tier 3 analysis |
| `volume_mc_ratio` | NUMERIC(10,4) | Volume/Market Cap ratio | avg_daily_volume_usd / current_market_cap | Liquidity metric |

---

### 1.3 Wallet-Level Variables
**Source:** `wallets.csv` (25,161 records)

| Variable | Type | Description | Example | Used In |
|----------|------|-------------|---------|---------|
| `wallet_address` | VARCHAR(42) | Wallet address | 0x5de4ef48... | Primary key |
| `first_seen_date` | DATE | First DEX transaction | 2024-06-27 | Wallet age |
| `last_active_date` | DATE | Most recent transaction | 2024-09-26 | Activity recency |
| `total_trades_30d` | INTEGER | Trade count (30 days) | 78660 | Activity level |
| `avg_daily_volume_eth` | NUMERIC(18,8) | Average daily volume (ETH) | 1001.61 | Smart money filter |
| `unique_tokens_traded` | INTEGER | Distinct tokens | 0 (to be updated) | Diversity |
| `is_smart_money` | BOOLEAN | Smart money flag | True | Filtering |

---

### 1.4 Balance Snapshot Variables
**Source:** `wallet_token_balances.csv` (70,290 snapshots)

| Variable | Type | Description | Example | Used In |
|----------|------|-------------|---------|---------|
| `wallet_address` | VARCHAR(42) | Wallet address | 0x0000000... | Linkage |
| `token_address` | VARCHAR(42) | Token held | 0x004394... | Portfolio composition |
| `snapshot_date` | DATE | Snapshot date | 2025-09-03 | Time series |
| `block_number` | INTEGER | Block at snapshot | 23528000 | Precision |
| `balance_raw` | NUMERIC | Raw balance (wei) | 0 | Storage |
| `balance_formatted` | NUMERIC(18,8) | Human-readable balance | 0.0 | Analysis |
| `token_symbol` | VARCHAR(20) | Token ticker | DEEPSEEK R1 | Display |
| `decimals` | INTEGER | Decimal places | 8 | Conversion |

**Snapshot Frequency:** Daily (Sept 3 - Oct 3, 2025 = 30 days)
**Coverage:** 2,343 wallets × 30 days = 70,290 snapshots

---

### 1.5 Price Data Variables
**Source:** `eth_prices.csv` (729 records)

| Variable | Type | Description | Example | Used In |
|----------|------|-------------|---------|---------|
| `timestamp` | TIMESTAMP | Price timestamp (hourly) | 2025-09-03 00:02:39 | Valuation |
| `price_usd` | NUMERIC(10,2) | ETH/USD price | 2400.50 | USD conversion |
| `source` | VARCHAR(50) | Data source | coingecko | Data lineage |

**Coverage:** Sept 3 - Oct 3, 2025 (hourly granularity)

---

## 2. Derived Variables (Epic 4.1)

### 2.1 Performance Metrics

#### Win Rate
**Formula:**
```python
win_rate = (successful_profitable_trades / total_successful_trades) * 100
```

**Calculation Logic:**
```python
# For each wallet
profitable_trades = 0
total_trades = 0

for tx in transactions:
    if tx.status == 'success':
        total_trades += 1

        # Get token prices at entry and exit
        entry_price = get_token_price(tx.token_out, tx.timestamp)
        exit_price = get_token_price(tx.token_out, current_time)

        if exit_price > entry_price:
            profitable_trades += 1

win_rate = (profitable_trades / total_trades) * 100
```

**Expected Range:** 0% - 100%
**Typical Values:** 40% - 70% for smart money
**Use Case:** Cluster characterization, H3 early adoption hypothesis

---

#### Total Return (ROI %)
**Formula:**
```python
ROI_pct = ((ending_portfolio_value - starting_portfolio_value) / starting_portfolio_value) * 100
```

**Calculation Logic:**
```python
# Starting portfolio value (Sept 3, 2025)
starting_value = sum(balance_snapshot[date='2025-09-03'].balance_formatted * token.price_usd)

# Ending portfolio value (Oct 3, 2025)
ending_value = sum(balance_snapshot[date='2025-10-03'].balance_formatted * token.price_usd)

# Subtract gas costs
total_gas_cost = sum(tx.gas_used * tx.gas_price_gwei * eth_price / 1e9)

ROI_pct = ((ending_value - starting_value - total_gas_cost) / starting_value) * 100
```

**Expected Range:** -100% to +∞ (typically -50% to +500% in 1 month)
**Typical Values:** -10% to +50% for smart money (1-month window)
**Use Case:** Cluster performance comparison, H3, H4, H5

---

#### Sharpe Ratio (Risk-Adjusted Return)
**Formula:**
```python
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
```

**Calculation Logic:**
```python
# Daily returns from balance snapshots
daily_returns = []
for i in range(1, 30):
    portfolio_value_t = get_portfolio_value(date=day_i)
    portfolio_value_t_1 = get_portfolio_value(date=day_i-1)
    daily_return = (portfolio_value_t - portfolio_value_t_1) / portfolio_value_t_1
    daily_returns.append(daily_return)

# Annualized return
avg_daily_return = mean(daily_returns)
annualized_return = (1 + avg_daily_return) ** 365 - 1

# Annualized volatility
daily_volatility = stdev(daily_returns)
annualized_volatility = daily_volatility * sqrt(365)

# Risk-free rate (assume 4% annual)
risk_free_rate = 0.04

# Sharpe ratio
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
```

**Expected Range:** -2.0 to +3.0 (typically -1.0 to +2.0)
**Typical Values:** 0.5 - 1.5 for smart money
**Interpretation:** >1.0 = good risk-adjusted performance, >2.0 = excellent
**Use Case:** Primary performance metric for cluster comparison, H4

---

#### Maximum Drawdown
**Formula:**
```python
max_drawdown = min((portfolio_value_t - peak_value) / peak_value)
```

**Calculation Logic:**
```python
peak_value = 0
max_drawdown = 0

for day in snapshot_dates:
    portfolio_value = get_portfolio_value(day)

    # Update peak
    if portfolio_value > peak_value:
        peak_value = portfolio_value

    # Calculate drawdown from peak
    drawdown = (portfolio_value - peak_value) / peak_value

    if drawdown < max_drawdown:
        max_drawdown = drawdown

max_drawdown_pct = max_drawdown * 100
```

**Expected Range:** -100% to 0%
**Typical Values:** -10% to -40% for smart money (1-month window)
**Use Case:** Risk profiling, cluster characterization

---

#### Volatility (Portfolio Std Dev)
**Formula:**
```python
volatility_annualized = stdev(daily_returns) * sqrt(365)
```

**Calculation Logic:** (Calculated as part of Sharpe ratio)

**Expected Range:** 0% to +∞ (typically 30% - 150% annualized for crypto)
**Typical Values:** 50% - 100% for smart money
**Use Case:** Risk profiling, cluster separation

---

### 2.2 Trading Behavior Metrics

#### Trade Frequency (Trades per Day)
**Formula:**
```python
trade_frequency = total_trades / active_days
```

**Calculation Logic:**
```python
total_trades = count(transactions WHERE wallet = wallet_address)
active_days = count(DISTINCT DATE(timestamp) WHERE wallet = wallet_address)
trade_frequency = total_trades / active_days
```

**Expected Range:** 0.1 - 100+ trades/day
**Typical Values:** 1 - 20 trades/day for smart money
**Use Case:** Archetype separation (high-frequency traders vs buy-and-hold)

---

#### Average Holding Period (Days)
**Formula:**
```python
avg_holding_period = mean(exit_date - entry_date for each position)
```

**Calculation Logic:**
```python
holding_periods = []

for token in unique_tokens:
    positions = get_positions(wallet, token)

    for position in positions:
        entry_date = position.buy_tx.timestamp
        exit_date = position.sell_tx.timestamp if sold else current_date
        holding_period = (exit_date - entry_date).days
        holding_periods.append(holding_period)

avg_holding_period = mean(holding_periods)
```

**Expected Range:** 0.1 - 30+ days (within our 1-month window)
**Typical Values:** 2 - 10 days for smart money
**Use Case:** Archetype separation (day traders vs swing traders)

---

#### Trade Size Distribution (Median Trade Size USD)
**Formula:**
```python
median_trade_size_usd = median(usd_value_out for all transactions)
```

**Calculation Logic:**
```python
trade_sizes_usd = [tx.usd_value_out for tx in transactions WHERE wallet = wallet_address]
median_trade_size = median(trade_sizes_usd)
```

**Expected Range:** $100 - $1M+
**Typical Values:** $5K - $50K for smart money
**Use Case:** Wallet size classification, risk profiling

---

### 2.3 Portfolio Composition Metrics

#### Narrative Exposure (% per Narrative)
**Formula:**
```python
narrative_exposure_pct = (portfolio_value_in_narrative / total_portfolio_value) * 100
```

**Calculation Logic:**
```python
# For each snapshot date
for date in snapshot_dates:
    portfolio = get_balances(wallet, date)

    for narrative in ['AI', 'DeFi', 'Gaming', 'Meme', 'Infrastructure', 'Stablecoin', 'Other']:
        narrative_value = sum(
            balance.balance_formatted * token.current_price_usd
            for balance in portfolio
            if token.narrative_category == narrative
        )

        total_value = sum(balance.balance_formatted * token.current_price_usd for balance in portfolio)

        exposure_pct = (narrative_value / total_value) * 100

        # Store time-series
        narrative_exposures[narrative][date] = exposure_pct

# Average exposure over 30 days
avg_ai_exposure = mean(narrative_exposures['AI'].values())
avg_defi_exposure = mean(narrative_exposures['DeFi'].values())
# ... etc
```

**Expected Range:** 0% - 100% per narrative
**Typical Values:** 10% - 40% in dominant narrative, <10% in others
**Use Case:** PRIMARY clustering feature, H2 hypothesis testing

---

#### Portfolio Concentration (HHI - Herfindahl-Hirschman Index)
**Formula:**
```python
HHI = sum((token_allocation_pct / 100)^2 for each token)
```

**Calculation Logic:**
```python
# For each snapshot date
portfolio = get_balances(wallet, date)
total_value = sum(balance.balance_formatted * token.current_price_usd for balance in portfolio)

allocations_squared = []
for balance in portfolio:
    token_value = balance.balance_formatted * token.current_price_usd
    allocation_pct = (token_value / total_value) * 100
    allocations_squared.append((allocation_pct / 100) ** 2)

HHI = sum(allocations_squared)

# Average HHI over 30 days
avg_HHI = mean(HHI_values)
```

**Expected Range:** 0 (perfect diversification) to 1.0 (single token)
**Interpretation:**
- HHI < 0.15: Highly diversified
- HHI 0.15-0.25: Moderately diversified
- HHI 0.25-0.50: Concentrated
- HHI > 0.50: Highly concentrated

**Use Case:** Portfolio strategy classification, H4 hypothesis

---

#### Narrative Diversity (Gini Coefficient)
**Formula:**
```python
gini = (sum(abs(xi - xj) for all pairs)) / (2 * n * sum(xi))
```

**Calculation Logic:**
```python
# Narrative allocations (7 categories)
narrative_allocations = [
    avg_ai_exposure,
    avg_defi_exposure,
    avg_gaming_exposure,
    avg_meme_exposure,
    avg_infrastructure_exposure,
    avg_stablecoin_exposure,
    avg_other_exposure
]

# Remove zeros
narrative_allocations = [x for x in narrative_allocations if x > 0]

# Calculate Gini
n = len(narrative_allocations)
sum_differences = sum(abs(narrative_allocations[i] - narrative_allocations[j])
                      for i in range(n) for j in range(i+1, n))
sum_allocations = sum(narrative_allocations)

gini = sum_differences / (2 * n * sum_allocations)
```

**Expected Range:** 0 (perfect equality) to 1.0 (perfect inequality)
**Typical Values:** 0.3 - 0.7 for smart money
**Use Case:** Diversity metric, archetype classification

---

### 2.4 Accumulation Pattern Metrics

#### Accumulation Bias Score
**Formula:**
```python
accumulation_bias = (accumulation_events - distribution_events) / total_balance_changes
```

**Calculation Logic:**
```python
# For each token position
for token in unique_tokens:
    balances = get_daily_balances(wallet, token)

    accumulation_events = 0
    distribution_events = 0

    for i in range(1, len(balances)):
        balance_change = balances[i] - balances[i-1]

        if balance_change > 0:
            accumulation_events += 1
        elif balance_change < 0:
            distribution_events += 1

    total_changes = accumulation_events + distribution_events

    if total_changes > 0:
        accumulation_bias = (accumulation_events - distribution_events) / total_changes

# Average across all tokens
avg_accumulation_bias = mean(accumulation_bias_per_token)
```

**Expected Range:** -1.0 (always selling) to +1.0 (always buying)
**Typical Values:** -0.3 to +0.3 for smart money
**Interpretation:**
- > +0.5: Strong accumulation (diamond hands)
- -0.5 to +0.5: Balanced trading
- < -0.5: Strong distribution (profit-taking)

**Use Case:** Behavioral classification, H5 hypothesis

---

#### Conviction Score (Holding Through Volatility)
**Formula:**
```python
conviction_score = tokens_held_through_drawdown / total_tokens_with_drawdown
```

**Calculation Logic:**
```python
tokens_with_drawdown = 0
tokens_held = 0

for token in unique_tokens:
    # Check if token experienced >10% drawdown
    price_peak = max(token.price_history)
    price_trough = min(token.price_history)
    drawdown = (price_trough - price_peak) / price_peak

    if drawdown < -0.10:  # >10% drawdown
        tokens_with_drawdown += 1

        # Check if wallet held through drawdown
        balance_at_peak = get_balance(wallet, token, date_of_peak)
        balance_at_trough = get_balance(wallet, token, date_of_trough)

        if balance_at_trough >= balance_at_peak * 0.95:  # Held >95% of position
            tokens_held += 1

conviction_score = tokens_held / tokens_with_drawdown if tokens_with_drawdown > 0 else 0
```

**Expected Range:** 0 (no conviction, panic sells) to 1.0 (diamond hands)
**Typical Values:** 0.4 - 0.8 for smart money
**Use Case:** Risk tolerance classification, H5 hypothesis

---

#### Add-on-Dips Behavior (Buy the Dip Score)
**Formula:**
```python
add_on_dips_score = accumulation_during_dips / total_dip_events
```

**Calculation Logic:**
```python
dip_events = 0
add_on_events = 0

for token in unique_tokens:
    # Identify dip events (>10% price drop)
    for i in range(1, len(token.price_history)):
        price_change_pct = (token.price_history[i] - token.price_history[i-1]) / token.price_history[i-1]

        if price_change_pct < -0.10:  # >10% drop
            dip_events += 1

            # Check if wallet accumulated during dip
            balance_before = get_balance(wallet, token, date=i-1)
            balance_after = get_balance(wallet, token, date=i)

            if balance_after > balance_before * 1.05:  # Increased position by >5%
                add_on_events += 1

add_on_dips_score = add_on_events / dip_events if dip_events > 0 else 0
```

**Expected Range:** 0 (never buys dips) to 1.0 (always buys dips)
**Typical Values:** 0.2 - 0.6 for smart money
**Use Case:** Contrarian behavior detection, H5 hypothesis

---

### 2.5 Gas Efficiency Metrics

#### Gas Cost per Trade (USD)
**Formula:**
```python
avg_gas_cost = mean(gas_used * gas_price_gwei * eth_price / 1e9)
```

**Calculation Logic:**
```python
gas_costs_usd = []

for tx in transactions:
    eth_price_at_tx = get_eth_price(tx.timestamp)
    gas_cost_eth = tx.gas_used * tx.gas_price_gwei / 1e9
    gas_cost_usd = gas_cost_eth * eth_price_at_tx
    gas_costs_usd.append(gas_cost_usd)

avg_gas_cost = mean(gas_costs_usd)
```

**Expected Range:** $1 - $500 per trade
**Typical Values:** $5 - $50 for smart money
**Use Case:** Gas efficiency profiling, net ROI calculation

---

#### Gas Efficiency Ratio
**Formula:**
```python
gas_efficiency = total_gas_cost_usd / total_volume_traded_usd
```

**Calculation Logic:**
```python
total_gas = sum(tx.gas_used * tx.gas_price_gwei * eth_price / 1e9 for tx in transactions)
total_volume = sum(tx.usd_value_out for tx in transactions)

gas_efficiency_ratio = total_gas / total_volume
```

**Expected Range:** 0% - 10%+ (lower is better)
**Typical Values:** 0.1% - 2% for smart money
**Use Case:** Net performance calculation, H7 exploratory hypothesis

---

## 3. Clustering Feature Set

### 3.1 Selected Features for Wallet Clustering (18 features)

**Performance Dimension (5 features):**
1. `roi_pct` - Total return percentage
2. `sharpe_ratio` - Risk-adjusted return
3. `max_drawdown_pct` - Maximum loss from peak
4. `volatility_annualized` - Portfolio volatility
5. `win_rate_pct` - Percentage of profitable trades

**Trading Behavior Dimension (5 features):**
6. `trade_frequency` - Trades per day
7. `avg_holding_period_days` - Average holding period
8. `median_trade_size_usd` - Typical trade size
9. `dex_diversity` - Number of DEX protocols used
10. `unique_tokens_traded` - Number of unique tokens

**Portfolio Composition Dimension (7 features):**
11. `avg_ai_exposure_pct` - AI narrative allocation
12. `avg_defi_exposure_pct` - DeFi narrative allocation
13. `avg_gaming_exposure_pct` - Gaming narrative allocation
14. `avg_meme_exposure_pct` - Meme narrative allocation
15. `portfolio_concentration_hhi` - HHI concentration index
16. `narrative_diversity_gini` - Gini diversity coefficient
17. `avg_stablecoin_exposure_pct` - Stablecoin allocation

**Accumulation Behavior Dimension (1 feature):**
18. `accumulation_bias_score` - Net accumulation vs distribution

**Note:** Conviction score and add-on-dips excluded from clustering due to data sparsity (limited dip events in 1-month window).

---

### 3.2 Feature Normalization Strategy

**Standardization (Z-score):**
Most features will be standardized to mean=0, std=1:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
```

**Rationale:**
- HDBSCAN and K-Means sensitive to scale
- Features have different units (%, days, USD, ratios)
- Standardization preserves outliers (important for identifying extreme behaviors)

**Alternative Considered:**
- Min-Max scaling: Not used (sensitive to outliers, loses distributional info)
- Robust scaling: Could be used for features with heavy tails (e.g., trade_frequency)

---

### 3.3 Feature Importance Justification

| Feature | Justification | Expected Cluster Separation |
|---------|---------------|----------------------------|
| `roi_pct` | Direct performance measure | High (distinguishes winners from losers) |
| `sharpe_ratio` | Risk-adjusted performance | High (separates risk-takers from conservative) |
| `trade_frequency` | Activity level | Very High (day traders vs buy-and-hold) |
| `avg_holding_period_days` | Investment horizon | Very High (complements trade frequency) |
| `avg_ai_exposure_pct` | Narrative preference | High (identifies AI-focused wallets) |
| `avg_defi_exposure_pct` | Narrative preference | High (identifies DeFi farmers) |
| `avg_meme_exposure_pct` | Narrative preference | High (identifies meme speculators) |
| `portfolio_concentration_hhi` | Diversification strategy | Medium (concentrated bets vs diversified) |
| `accumulation_bias_score` | Trading style | Medium (accumulators vs distributors) |

---

## 4. Token-Level Variables (Tier 3 Analysis)

### 4.1 Current Token Metrics (Existing)

- `market_cap_rank` - CoinGecko ranking
- `avg_daily_volume_usd` - 24h trading volume
- `liquidity_tier` - Tier 1/2/3 classification

### 4.2 Enhanced Token Metrics (Phase 2)

**From Etherscan API:**
- `holder_count` - Number of unique token holders

**From CoinGecko API:**
- `current_price_usd` - Current token price
- `current_market_cap` - Market cap (circulating supply)
- `circulating_supply` - Tokens in circulation
- `total_supply` - Maximum tokens (if capped)

**Calculated:**
- `fdv` = `total_supply` × `current_price_usd`
- `volume_mc_ratio` = `avg_daily_volume_usd` / `current_market_cap`
- `daily_price_change_pct` - Daily % price change

**Use Cases:**
- Narrative performance analysis (Tier 3)
- Token liquidity validation
- Market context for wallet behavior
- Supporting visualizations in dashboard

**Note:** Token-level metrics NOT used for wallet clustering (focus is on wallet behavior, not token fundamentals).

---

## 5. Feature Engineering Pipeline (Epic 4.1)

### Step 1: Transaction-Level Calculations
```python
# Calculate USD values for each transaction
for tx in transactions:
    # Get token prices at transaction time
    token_in_price = get_token_price(tx.token_in, tx.timestamp)
    token_out_price = get_token_price(tx.token_out, tx.timestamp)
    eth_price = get_eth_price(tx.timestamp)

    # Convert amounts to USD
    tx.usd_value_in = (tx.amount_in / 10^token_in_decimals) * token_in_price
    tx.usd_value_out = (tx.amount_out / 10^token_out_decimals) * token_out_price

    # Calculate ETH values
    tx.eth_value_in = tx.usd_value_in / eth_price
    tx.eth_value_out = tx.usd_value_out / eth_price
```

### Step 2: Wallet-Level Aggregations
```python
for wallet in wallets:
    # Performance metrics
    wallet.roi_pct = calculate_roi(wallet)
    wallet.sharpe_ratio = calculate_sharpe(wallet)
    wallet.max_drawdown_pct = calculate_max_drawdown(wallet)
    wallet.win_rate_pct = calculate_win_rate(wallet)

    # Trading behavior
    wallet.trade_frequency = calculate_trade_frequency(wallet)
    wallet.avg_holding_period = calculate_holding_period(wallet)
    wallet.median_trade_size_usd = calculate_median_trade_size(wallet)

    # Portfolio composition
    wallet.narrative_exposures = calculate_narrative_exposure(wallet)
    wallet.portfolio_hhi = calculate_hhi(wallet)
    wallet.narrative_gini = calculate_gini(wallet)

    # Accumulation patterns
    wallet.accumulation_bias = calculate_accumulation_bias(wallet)
    wallet.conviction_score = calculate_conviction(wallet)
    wallet.add_on_dips_score = calculate_add_on_dips(wallet)
```

### Step 3: Feature Matrix Construction
```python
import pandas as pd

feature_matrix = pd.DataFrame({
    'wallet_address': wallet_addresses,
    'roi_pct': [w.roi_pct for w in wallets],
    'sharpe_ratio': [w.sharpe_ratio for w in wallets],
    # ... all 18 clustering features
})

# Handle missing values
feature_matrix.fillna(feature_matrix.median(), inplace=True)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(feature_matrix.drop('wallet_address', axis=1))

# Save for clustering
feature_matrix.to_parquet('outputs/wallet_features_clustering.parquet')
```

---

## 6. Data Quality Considerations

### 6.1 Missing Data Handling

**Expected Missing Values:**
- `conviction_score`: Wallets without >10% drawdown events (no dips to hold through)
- `add_on_dips_score`: Wallets without dip accumulation opportunities
- `win_rate_pct`: Wallets with no closed positions (all still holding)

**Imputation Strategy:**
- Median imputation for performance metrics
- Zero imputation for behavioral scores (absence of behavior = 0)
- Document imputation in limitations section

### 6.2 Outlier Treatment

**Detection:**
- IQR method: Values >Q3 + 1.5×IQR or <Q1 - 1.5×IQR flagged as outliers

**Treatment:**
- **Do NOT remove outliers** (extreme behaviors are informative)
- Use robust scaling for heavy-tailed features if needed
- Document outliers in cluster analysis (e.g., "Cluster 3 contains 5 extreme high-frequency traders")

---

## 7. Variable Summary Statistics (To Be Calculated)

**Output:** `outputs/documentation/FEATURE_SUMMARY_STATISTICS.csv`

| Feature | Mean | Std Dev | Min | Q1 | Median | Q3 | Max | Missing % |
|---------|------|---------|-----|----|----|----|----|-----------|
| roi_pct | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0% |
| sharpe_ratio | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**To be generated in Epic 4.1 after feature calculation.**

---

## 8. Alignment with Tutor Questions

| Tutor Question | Document Section |
|----------------|------------------|
| **3. What variables will you use?** | Sections 1-4 (comprehensive variable inventory) |
| **3a. Justify relevance** | Section 3.3 (Feature Importance Justification) |
| **3b. Include derived variables** | Section 2 (all derived variables with formulas) |
| **3c. Examples (CMC-style)** | Section 4.2 (token metrics matching tutor's examples) |

---

## 9. Next Steps

1. **Epic 4.1 Implementation:** Code feature calculation pipeline
2. **Validation:** Generate summary statistics, check distributions
3. **Documentation:** Update with actual statistics after calculation
4. **Epic 4.3:** Use feature matrix for clustering

---

## References

### Internal Documents
- `RESEARCH_HYPOTHESES.md` - Hypotheses using these features
- `MODEL_EVALUATION_FRAMEWORK.md` - Metrics for evaluating feature quality
- `DATA_DICTIONARY.md` - Raw data schema
- `MVP_STRATEGY.md` - Strategic context

### Statistical Methods
- HHI: U.S. Department of Justice (1982) - Merger Guidelines
- Gini: Gini (1912) - Variability and Mutability
- Sharpe: Sharpe (1966) - Mutual Fund Performance

---

## Document Version Control

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Oct 8, 2025 | Initial feature specification | Dev Agent |

---

**Status:** ACTIVE - Reference for Epic 4.1 implementation
**Next Review:** After Epic 4.1 (update with calculated statistics)
