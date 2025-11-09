# Daily Balance Snapshot Collection Guide

**Version:** 1.0
**Date:** October 5, 2025
**Status:** Ready for Production

---

## Overview

This guide explains how to collect and analyze historical wallet balance snapshots using the Alchemy API. Daily balance snapshots enable:

- **Accumulation/distribution pattern analysis**
- **Portfolio composition tracking over time**
- **Transaction data validation**
- **Enhanced wallet behavioral clustering**
- **Smart money conviction signals**

---

## Cost Analysis

### API Pricing (Alchemy Free Tier)

- **Free tier limit:** 30,000,000 Compute Units (CUs) per month
- **API method:** `alchemy_getTokenBalances`
- **Cost per call:** 20 CUs
- **Rate limit:** 25 requests/second (we use 10 for safety)

### Collection Estimates

**Tier 1 Wallets (2,343 wallets) - 30 Daily Snapshots:**
- Total calls: **70,290**
- Compute units: **1,405,800 CUs**
- **Cost: $0** (uses 4.7% of free tier)
- Runtime: ~2 hours at 10 req/sec

**All Wallets (25,161 wallets) - 30 Daily Snapshots:**
- Total calls: **754,830**
- Compute units: **15,096,600 CUs**
- **Cost: $0** (uses 50.3% of free tier)
- Runtime: ~21 hours at 10 req/sec

---

## Setup

### 1. Get Alchemy API Key

1. Go to [alchemy.com](https://www.alchemy.com/)
2. Sign up for a free account
3. Create a new app:
   - **Chain:** Ethereum
   - **Network:** Mainnet
   - **Plan:** Free
4. Copy your API key

### 2. Configure Environment

Add your API key to `.env`:

```bash
ALCHEMY_API_KEY=your_api_key_here
```

### 3. Create Database Schema

Run the schema migration:

```bash
psql -U user -d crypto_narratives -f sql/wallet_balances_schema.sql
```

This creates:
- `wallet_token_balances` table
- Indexes for performance
- `daily_portfolio_summary` materialized view
- `latest_wallet_balances` view

---

## Usage

### Basic Collection (Tier 1 Wallets)

Collect daily snapshots for all wallets with transaction data:

```bash
python cli_collect_daily_balances.py \
  --start-date 2025-09-03 \
  --end-date 2025-10-03 \
  --rate-limit 10
```

**Parameters:**
- `--start-date`: First snapshot date (YYYY-MM-DD)
- `--end-date`: Last snapshot date (YYYY-MM-DD)
- `--rate-limit`: Requests per second (default: 10, max: 25)

### Custom Wallet List

Collect for specific wallets:

```bash
# Create wallet list file
echo "0x1234..." > my_wallets.txt
echo "0x5678..." >> my_wallets.txt

python cli_collect_daily_balances.py \
  --start-date 2025-09-03 \
  --end-date 2025-10-03 \
  --wallet-file my_wallets.txt
```

### Resume Interrupted Collection

The script automatically creates checkpoints. If interrupted, simply re-run:

```bash
# Will resume from last checkpoint
python cli_collect_daily_balances.py \
  --start-date 2025-09-03 \
  --end-date 2025-10-03
```

Checkpoints are stored in `data/checkpoints/balances/`

---

## Data Structure

### wallet_token_balances Table

```sql
CREATE TABLE wallet_token_balances (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) NOT NULL,
    token_address VARCHAR(42) NOT NULL,
    snapshot_date DATE NOT NULL,
    block_number BIGINT NOT NULL,
    balance_raw NUMERIC(78,0),        -- Raw balance (wei-equivalent)
    balance_formatted NUMERIC(30,18), -- Human-readable balance
    token_symbol VARCHAR(20),
    token_name VARCHAR(100),
    decimals INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(wallet_address, token_address, snapshot_date)
);
```

### Example Query: Daily Portfolio

```sql
SELECT
    snapshot_date,
    token_symbol,
    balance_formatted
FROM wallet_token_balances
WHERE wallet_address = '0x...'
  AND balance_formatted > 0
ORDER BY snapshot_date, token_symbol;
```

---

## Analysis

### 1. Accumulation Metrics

Calculate accumulation/distribution patterns:

```python
from services.balances.balance_analyzer import BalanceAnalyzer

analyzer = BalanceAnalyzer(db_pool)

metrics = await analyzer.calculate_accumulation_metrics(
    wallet_address="0x...",
    token_address="0x...",
    start_date=datetime(2025, 9, 3),
    end_date=datetime(2025, 10, 3)
)

print(f"Accumulation rate: {metrics.accumulation_rate} tokens/day")
print(f"Consistency score: {metrics.consistency_score}")
print(f"Held through volatility: {metrics.held_through_volatility}")
```

### 2. Portfolio Composition

Track narrative allocation over time:

```python
# Get narrative allocation on specific date
allocation = await analyzer.get_narrative_allocation(
    wallet_address="0x...",
    snapshot_date=datetime(2025, 9, 15)
)

# Output: {'DeFi': 0.4, 'AI': 0.35, 'Gaming': 0.25}
```

### 3. Wallet Archetype Classification

```python
archetype = await analyzer.identify_wallet_archetype(
    wallet_address="0x...",
    start_date=datetime(2025, 9, 3),
    end_date=datetime(2025, 10, 3)
)

# Archetypes: diamond_hands, rotator, concentrated, diversified, etc.
```

### 4. Comprehensive Wallet Report

```python
report = await analyzer.generate_wallet_balance_report(
    wallet_address="0x...",
    start_date=datetime(2025, 9, 3),
    end_date=datetime(2025, 10, 3)
)

print(report['archetype'])  # e.g., "diamond_hands"
print(report['portfolio_summary'])  # Accumulation bias, token counts
print(report['token_metrics'])  # Top 10 tokens by % change
```

---

## Feature Engineering for Clustering

Balance snapshots enable powerful time-series features:

### Temporal Features

```python
# Daily portfolio volatility
daily_hhi = [
    await analyzer.calculate_hhi_concentration(wallet, date)
    for date in date_range
]
hhi_std = np.std(daily_hhi)

# Narrative rotation frequency
narrative_changes = count_narrative_switches(wallet, date_range)
```

### Behavioral Features

```python
# Accumulation bias score
accumulation_bias = num_accumulating_tokens / total_tokens

# Conviction score (holding through drawdowns)
conviction_score = sum(
    1 for token in tokens
    if held_through_volatility(wallet, token)
) / total_tokens
```

### Risk Features

```python
# Max concentration over period
max_hhi = max(daily_hhi)

# Position sizing volatility
position_sizes = get_daily_position_sizes(wallet)
position_volatility = np.std(position_sizes)
```

---

## Validation

### Cross-Check with Transactions

Validate balance snapshots against transaction data:

```sql
-- Compare final balance (from snapshots) vs derived (from transactions)
WITH transaction_derived AS (
    SELECT
        wallet_address,
        token_address,
        SUM(amount_in) - SUM(amount_out) as derived_balance
    FROM transactions
    GROUP BY wallet_address, token_address
),
snapshot_actual AS (
    SELECT
        wallet_address,
        token_address,
        balance_formatted as actual_balance
    FROM wallet_token_balances
    WHERE snapshot_date = '2025-10-03'
)
SELECT
    t.wallet_address,
    t.token_address,
    t.derived_balance,
    s.actual_balance,
    ABS(t.derived_balance - s.actual_balance) as discrepancy
FROM transaction_derived t
JOIN snapshot_actual s USING (wallet_address, token_address)
WHERE ABS(t.derived_balance - s.actual_balance) > 0.01
ORDER BY discrepancy DESC;
```

---

## Performance Tips

### 1. Rate Limiting

- Use `--rate-limit 10` for safety (free tier supports 25/sec)
- Monitor Alchemy dashboard for usage

### 2. Batch Processing

- Process by date (complete all wallets for day 1, then day 2, etc.)
- Checkpoints saved per date for easy resume

### 3. Database Optimization

```sql
-- Refresh materialized view after collection
REFRESH MATERIALIZED VIEW daily_portfolio_summary;

-- Analyze tables for query optimization
ANALYZE wallet_token_balances;
```

### 4. Storage Management

- 70K snapshots ≈ 100MB storage
- Regularly vacuum database:

```sql
VACUUM ANALYZE wallet_token_balances;
```

---

## Troubleshooting

### Issue: API Rate Limit Exceeded

**Error:** `429 Too Many Requests`

**Solution:** Reduce rate limit:
```bash
python cli_collect_daily_balances.py --rate-limit 5
```

### Issue: Missing Block Numbers

**Error:** `Block not found`

**Solution:** Update reference block in `block_mapper.py`:
```python
REFERENCE_BLOCK = <current_block>
REFERENCE_TIMESTAMP = datetime.now()
```

Or use API-based block lookup:
```python
mapper.generate_daily_mapping(
    start_date,
    end_date,
    use_api=True,
    alchemy_client=client
)
```

### Issue: Database Connection Timeout

**Solution:** Increase pool size:
```python
db_pool = await asyncpg.create_pool(
    db_url,
    min_size=2,
    max_size=20  # Increase from 10
)
```

---

## Integration with Thesis

### Epic 4: Feature Engineering

Add balance-derived features to `wallet_performance` calculations:

```python
# In calculate_wallet_features()
balance_features = {
    'accumulation_bias': ...,
    'avg_portfolio_hhi': ...,
    'narrative_rotation_freq': ...,
    'conviction_score': ...,
}
```

### Epic 5: Clustering

Use time-series features for enhanced clustering:

```python
from sklearn.cluster import HDBSCAN

# Include balance-derived features
features = pd.DataFrame({
    'win_rate': ...,
    'sharpe_ratio': ...,
    'accumulation_bias': ...,  # New
    'conviction_score': ...,   # New
    'avg_hhi': ...             # New
})

clusterer = HDBSCAN(min_cluster_size=50)
clusters = clusterer.fit_predict(features)
```

---

## Next Steps

1. ✅ **Run schema migration** (`sql/wallet_balances_schema.sql`)
2. ✅ **Get Alchemy API key** and add to `.env`
3. ✅ **Test with small sample** (10 wallets, 3 days)
4. ✅ **Run full collection** (2,343 wallets, 30 days)
5. ✅ **Refresh materialized views**
6. ✅ **Run validation queries**
7. ✅ **Generate balance features** for clustering
8. ✅ **Update thesis methodology** section

---

## References

- **Alchemy Docs:** https://docs.alchemy.com/reference/alchemy-gettokenbalances
- **Database Schema:** `sql/wallet_balances_schema.sql`
- **Analysis Module:** `services/balances/balance_analyzer.py`
- **MVP Strategy:** `docs/data-collection-phase/MVP_STRATEGY.md`

---

**Questions?** Check the Balance Analyzer module docstrings or review example queries in this guide.
