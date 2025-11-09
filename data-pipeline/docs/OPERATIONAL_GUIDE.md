# Data Collection Ecosystem - Operational Guide
**Crypto Narrative Hunter - TFM Project**

Version: 1.0
Last Updated: 2025-09-29
Status: Production Ready

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Summary](#architecture-summary)
3. [Prerequisites Setup](#prerequisites-setup)
4. [Complete Execution Workflow](#complete-execution-workflow)
5. [Automated Pipeline Script](#automated-pipeline-script)
6. [Monitoring & Checkpoints](#monitoring--checkpoints)
7. [API Rate Limits & Considerations](#api-rate-limits--considerations)
8. [Data Quality Targets](#data-quality-targets)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Expected Timeline](#expected-timeline)
11. [Final Deliverables](#final-deliverables)

---

## System Overview

The Crypto Narrative Hunter data collection ecosystem is a **comprehensive 3-epic, 13-story pipeline** that extracts, processes, validates, and exports Ethereum blockchain data for smart money behavior analysis and narrative trend identification.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EPIC 1: TOKEN UNIVERSE                        │
├─────────────────────────────────────────────────────────────────┤
│ Story 1.1: Infrastructure Setup                                 │
│ Story 1.2: Token Metadata Collection (CoinGecko)                │
│ Story 1.3: DEX Liquidity Analysis (Dune + Uniswap/Curve)        │
│ Story 1.4: Narrative Categorization & Validation                │
│ ➜ Output: 500 categorized tokens with liquidity tiers           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               EPIC 2: SMART WALLET IDENTIFICATION                │
├─────────────────────────────────────────────────────────────────┤
│ Story 2.1: Smart Wallet Query Development (Dune SQL)            │
│ Story 2.2: Wallet Performance Metrics Extraction                │
│ Story 2.3: Wallet Filtering & Bot Detection                     │
│ Story 2.4: Final Wallet Cohort Selection                        │
│ ➜ Output: 8,000-12,000 validated smart money wallets            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│          EPIC 3: TRANSACTION & BALANCE DATA COLLECTION           │
├─────────────────────────────────────────────────────────────────┤
│ Story 3.1: Transaction History Extraction (Alchemy RPC)         │
│ Story 3.2: Daily Balance Snapshots Collection                   │
│ Story 3.3: ETH Price History & Value Calculations (Chainlink)   │
│ Story 3.4: Data Validation & Quality Assurance                  │
│ Story 3.5: Final Data Export & Documentation                    │
│ ➜ Output: Quality-certified datasets ready for analysis         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Summary

### Technology Stack
- **Language:** Python 3.9+
- **Package Manager:** uv (modern Python package management)
- **Database:** PostgreSQL 14+ with connection pooling
- **Data Formats:** Parquet (snappy), CSV, JSON
- **Key Libraries:** pandas, pyarrow, web3, aiohttp, structlog, pydantic

### Data Sources
- **CoinGecko API:** Token metadata, market data
- **Dune Analytics:** Blockchain analytics, wallet identification
- **Alchemy RPC:** Ethereum on-chain data (transactions, balances)
- **Uniswap Subgraphs:** DEX swap data and pricing
- **Curve API:** Stablecoin pool data
- **Chainlink:** ETH/USD price feeds
- **Etherscan:** Fallback verification

### Database Schema
- **tokens** - 500 ERC-20 tokens with categorization
- **wallets** - 8-12K smart money wallet addresses
- **transactions** - Complete swap history (monthly partitioned)
- **wallet_balances** - Daily portfolio snapshots
- **eth_prices** - Hourly ETH/USD price data
- **wallet_performance** - Calculated performance metrics
- **collection_checkpoints** - Pipeline progress tracking

---

## Prerequisites Setup

### 1. System Requirements

```bash
# Required software
- Python 3.9 or higher
- PostgreSQL 14+ (running locally or via Docker)
- uv package manager v0.4+
- Git (for version control)

# Recommended hardware
- 8GB+ RAM
- 50GB+ available disk space
- Stable internet connection (API-intensive operations)
```

### 2. Install uv Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 3. Clone and Navigate to Project

```bash
cd /path/to/BMAD_TFM/data-collection
```

### 4. Create Virtual Environment

```bash
# Create isolated Python environment
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### 5. Install Project Dependencies

```bash
# Sync all dependencies from pyproject.toml
uv pip sync pyproject.toml

# Verify installation
python --version
pip list
```

### 6. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**

```env
# Blockchain Data
ALCHEMY_API_KEY=your_alchemy_key_here
ETHERSCAN_API_KEY=your_etherscan_key_here

# Market Data
COINGECKO_API_KEY=your_coingecko_key_here

# Analytics
DUNE_API_KEY=your_dune_key_here

# Database Connection
DATABASE_URL=postgresql://user:password@localhost:5432/crypto_hunter

# Optional: Logging
LOG_LEVEL=INFO
```

**Where to Get API Keys:**
- **Alchemy:** https://alchemy.com (free tier: 300M compute units/month)
- **Etherscan:** https://etherscan.io/apis (free tier: 5 calls/second)
- **CoinGecko:** https://www.coingecko.com/en/api (free tier: 10-50 calls/min)
- **Dune Analytics:** https://dune.com/settings/api (requires credits)

### 7. Database Initialization

```bash
# Start PostgreSQL (if using Docker)
docker-compose up -d postgres

# Initialize database schema
uv run data-collection init-db

# Create checkpoint table
uv run data-collection ensure-checkpoints

# Verify connectivity
uv run data-collection health
```

**Expected Output:**
```
✓ Database schema initialized successfully
✓ Checkpoint table created
✓ Database connection healthy
```

---

## Complete Execution Workflow

### Phase 1: Token Metadata Collection

#### Story 1.2 - Collect Top 500 Tokens from CoinGecko

**Purpose:** Establish the token universe with complete metadata

**Command:**
```bash
python services/tokens/token_collection_service.py
```

**What it does:**
1. Fetches top 500 Ethereum tokens by market cap from CoinGecko
2. Validates Ethereum contract addresses with checksum verification
3. Deduplicates tokens by contract address (handles multiple symbols)
4. Extracts: symbol, name, decimals, market_cap_rank, daily_volume
5. Stores in `tokens` table
6. Exports to `outputs/csv/tokens_metadata.csv` for review

**Expected Output:**
- ~497-500 tokens stored in database
- CSV export for manual validation
- Structured logs in `logs/collection_YYYYMMDD.log`

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type tokens
```

**Time Estimate:** 30-60 minutes (depending on API rate limits)

---

#### Story 1.3 - Analyze DEX Liquidity via Dune Analytics

**Purpose:** Assign liquidity tiers based on DEX TVL analysis

**Command:**
```bash
python cli_liquidity.py
```

**What it does:**
1. Executes Dune SQL queries for Uniswap V2/V3 and Curve pools
2. Calculates Total Value Locked (TVL) for each token
3. Assigns liquidity tiers:
   - **Tier 1:** TVL > $10M (highest liquidity)
   - **Tier 2:** TVL $1M-$10M (moderate liquidity)
   - **Tier 3:** TVL < $1M (low liquidity)
   - **Untiered:** No significant liquidity found
4. Updates `tokens` table with `liquidity_tier` field
5. Generates quality report with pool coverage analysis

**Expected Output:**
- All tokens assigned liquidity tiers
- Pool metadata stored for audit trail
- Validation report showing tier distribution

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type liquidity_analysis
```

**Time Estimate:** 1-2 hours (Dune query execution + processing)

**Important Notes:**
- Uses Dune Analytics credits (monitor usage)
- Results are cached to avoid re-spending credits
- Preview runs available with 7-day windows for testing

---

#### Story 1.4 - Narrative Categorization & Validation

**Purpose:** Categorize tokens by narrative themes with manual review

**Command:**
```bash
python narrative_classification_orchestrator.py
```

**Interactive Workflow:**

**Step 1: Automated Classification**
```
Running rule-based classifier...
✓ 450 tokens auto-classified (confidence >80%)
✓ 47 tokens flagged for manual review
```

**Step 2: Manual Review**
```
Exported for review: outputs/csv/tokens_for_review.csv

Review the CSV and update the 'manual_category' column:
- DeFi (decentralized finance, DEXs, lending)
- Gaming (GameFi, NFTs, metaverse)
- AI (artificial intelligence, machine learning)
- Infrastructure (L2, bridges, oracles)
- Meme (community tokens, meme coins)
- Stablecoin (USD-pegged, algorithmic)
- Other (multi-category or uncategorized)

Save and import reviewed file when ready.
```

**Step 3: Import Reviewed Classifications**
```bash
# Import reviewed classifications
python narrative_classification_orchestrator.py --import outputs/csv/tokens_reviewed.csv
```

**Step 4: Final Export**
```
✓ All 497 tokens categorized
✓ Quality report generated
✓ Final dataset: outputs/csv/tokens_final.csv
```

**Expected Output:**
- Complete narrative categorization for all tokens
- Audit trail of classification decisions
- Data quality report with distribution metrics

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type narrative_classification
```

**Time Estimate:** 2-3 hours (including manual review)

---

### Phase 2: Smart Wallet Identification

#### Story 2.1 - Execute Dune Queries for Wallet Discovery

**Purpose:** Identify high-volume, high-performance wallets trading target tokens

**Command:**
```bash
python services/smart_wallet_query_manager.py
```

**What it does:**
1. Executes parameterized Dune SQL queries with volume filters:
   - Minimum 30-day volume: $10,000 USD
   - Minimum 90-day volume: $25,000 USD
   - Minimum trade count: 10 trades
   - Minimum unique tokens: 3 different contracts
2. Applies bot detection heuristics:
   - Excludes high-frequency traders (>100 trades/day)
   - Filters out MEV bots using gas price patterns
   - Removes known router/aggregator contracts
   - Detects sandwich attack patterns
3. Caches query results to conserve Dune credits
4. Stores candidate wallets in `wallets` table

**Expected Output:**
- 8,000-12,000 candidate wallet addresses
- Bot detection summary report
- Cached Dune query results in `cache/dune_queries/`

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type wallet_identification
```

**Time Estimate:** 2-4 hours (Dune query complexity dependent)

**Important Notes:**
- Use preview runs (7-day windows) to validate queries before full execution
- Monitor Dune credit usage carefully
- Results are deterministic once cached

---

#### Story 2.2 - Calculate Wallet Performance Metrics

**Purpose:** Quantify trading effectiveness and risk-adjusted returns

**Command:**
```bash
python src/services/wallets/performance_calculator.py
```

**What it does:**
1. **Trading Performance:**
   - Win rate (% profitable trades)
   - Average return per trade (FIFO accounting)
   - Total return and annualized return
2. **Risk-Adjusted Metrics:**
   - Sharpe ratio (volatility-adjusted return)
   - Sortino ratio (downside risk focus)
   - Maximum drawdown from peak
   - Value at Risk (VaR) at 95% confidence
3. **Gas Efficiency:**
   - Trading volume per gas dollar spent
   - Gas optimization patterns
   - Net returns after transaction costs
4. **Diversification:**
   - Unique tokens traded
   - Portfolio concentration (HHI index)
   - Sector allocation by narrative

**Expected Output:**
- Performance metrics stored in `wallet_performance` table
- Wallet rankings by Sharpe ratio, return, efficiency
- Top performer identification report

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type performance_calculation
```

**Time Estimate:** 3-5 hours (computational complexity)

---

### Phase 3: Transaction & Balance Data Collection

#### Story 3.1 - Extract Complete Transaction History

**Purpose:** Collect all DEX swap transactions for the wallet cohort

**Command:**
```bash
python services/transactions/batch_processor.py
```

**What it does:**
1. **Multi-Source Extraction:**
   - Alchemy RPC: `eth_getLogs` for swap events
   - Uniswap Subgraphs: GraphQL for enriched swap data
   - Curve API: Stablecoin and crypto pool transactions
2. **Transaction Decoding:**
   - Router call decoding (multi-hop swaps)
   - Direct pool interaction handling
   - ABI-based function signature matching
3. **Price Normalization:**
   - Converts all amounts to ETH using pool reserves
   - Calculates slippage and MEV impact
   - Handles Uniswap V2/V3 and Curve pricing
4. **Failed Transaction Processing:**
   - Identifies failed transactions
   - Attributes gas costs
   - Detects MEV sandwich attacks
5. **Batch Processing:**
   - Wallet-based sharding (100 wallets/batch)
   - Concurrent processing with rate limiting
   - Progress tracking with ETA estimation

**Configuration:**
- Batch size: 100 wallets per shard
- Concurrent Alchemy requests: 15 (semaphore-limited)
- Block range: Dynamically optimized based on activity

**Expected Output:**
- Complete transaction history in `transactions` table
- Monthly partitioned for performance
- MEV detection and gas cost analysis

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type transaction_extraction
```

**Time Estimate:** 12-24 hours (10K wallets × historical data)

**Monitoring:**
```bash
# View progress
tail -f logs/collection_YYYYMMDD.log

# Check transaction count
psql $DATABASE_URL -c "SELECT COUNT(*) FROM transactions;"
```

---

#### Story 3.3 - Collect ETH Price History

**Purpose:** Enable USD value calculations for all transactions and balances

**Command:**
```bash
python services/prices/chainlink_client.py
```

**What it does:**
1. **Price Collection:**
   - Fetches hourly ETH/USD prices from Chainlink aggregator
   - Backfills historical prices as needed
   - Correlates prices with block numbers
2. **Multi-Source Validation:**
   - Primary: Chainlink on-chain aggregator
   - Secondary: CoinGecko historical API
   - Tertiary: Uniswap USDC/ETH pool
   - Consensus mechanism with 5% tolerance
3. **Quality Assurance:**
   - Anomaly detection (z-score >3)
   - Sudden change detection (>10% hourly)
   - Gap identification and interpolation
4. **Caching:**
   - Multi-tier caching (memory + Redis + disk)
   - Preloading for batch operations

**Expected Output:**
- Hourly ETH/USD prices in `eth_prices` table
- >99% coverage for analysis period
- Price validation report with accuracy metrics

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type price_collection
```

**Time Estimate:** 1-2 hours (historical backfill)

---

#### Story 3.4 - Data Validation & Quality Assurance

**Purpose:** Ensure data accuracy, completeness, and reliability

**Command:**
```bash
python services/validation/quality_reporter.py
```

**What it does:**
1. **Cross-Validation:**
   - Transaction-to-balance reconciliation
   - Multi-source transaction count verification
   - Etherscan spot-check validation (100 samples)
2. **Statistical Validation:**
   - Outlier detection (IsolationForest, LOF, OneClassSVM)
   - Trading pattern reasonableness checks
   - Behavioral anomaly detection
3. **Completeness Validation:**
   - Field-level completeness (>95% target)
   - Missing data pattern analysis
   - Temporal gap detection
4. **Database Integrity:**
   - Referential integrity verification
   - Constraint validation (CHECK, NOT NULL)
   - Orphaned record detection
5. **Quality Scoring:**
   - Composite score across 5 dimensions:
     - Completeness (25%)
     - Accuracy (25%)
     - Consistency (20%)
     - Timeliness (15%)
     - Validity (15%)

**Expected Output:**
- Comprehensive quality report (JSON + Markdown + HTML)
- Overall quality grade (A+ to F)
- Detailed validation results by dimension
- Recommendations for quality improvement

**Quality Targets:**
- **Completeness:** >95% for critical fields
- **Accuracy:** >99% transaction decoding
- **Consistency:** <5% variance between sources
- **Overall Quality:** Grade A (>90%)

**Checkpoint:**
```bash
uv run data-collection checkpoint-show --type validation
```

**Time Estimate:** 2-3 hours (comprehensive validation)

---

#### Story 3.5 - Final Data Export & Documentation

**Purpose:** Create analysis-ready datasets with complete documentation

**Command:**
```bash
python services/export/data_exporter.py
```

**What it does:**
1. **Multi-Format Exports:**
   - **Parquet:** Snappy compression, optimal for analytics
   - **CSV:** Human-readable, spreadsheet compatible
   - **JSON:** Metadata and schema definitions
2. **Partitioning:**
   - Transactions: Monthly partitions (YYYY_MM)
   - Balances: Monthly partitions
   - Other tables: Single file exports
3. **Documentation Generation:**
   - Complete data dictionary (all fields)
   - Methodology documentation
   - Data lineage and transformation logic
   - Quality certification report
   - User guides and API reference
4. **Backup & Archival:**
   - Compressed backup creation
   - Integrity verification (checksums)
   - Long-term storage preparation

**Output Structure:**
```
outputs/
├── parquet/
│   ├── tokens/
│   │   └── tokens.parquet
│   ├── wallets/
│   │   └── wallets.parquet
│   ├── transactions/
│   │   ├── transactions_2024_01.parquet
│   │   ├── transactions_2024_02.parquet
│   │   └── ... (monthly partitions)
│   ├── wallet_balances/
│   │   └── ... (monthly partitions)
│   ├── eth_prices/
│   │   └── eth_prices.parquet
│   └── wallet_performance/
│       └── wallet_performance.parquet
├── csv/
│   └── (same structure for human review)
├── json/
│   ├── data_dictionary.json
│   ├── export_manifest.json
│   ├── quality_certification.json
│   └── lineage_documentation.json
└── documentation/
    ├── methodology.md
    ├── user_guide.md
    ├── api_reference.md
    └── troubleshooting.md
```

**Expected Output:**
- Complete dataset exports (all formats)
- Comprehensive documentation package
- Quality certification (Grade A expected)
- Handoff manifest for analysis team

**Time Estimate:** 2-3 hours (export + documentation)

---

## Automated Pipeline Script

For complete end-to-end execution, use this bash script:

```bash
#!/bin/bash
# File: run_complete_pipeline.sh
# Purpose: Execute entire data collection pipeline

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify prerequisites
log_info "Verifying prerequisites..."
python --version || { log_error "Python not found"; exit 1; }
uv --version || { log_error "uv not found"; exit 1; }

# Check database connectivity
log_info "Testing database connection..."
uv run data-collection health || { log_error "Database connection failed"; exit 1; }

# PHASE 1: TOKEN COLLECTION
log_info "========================================"
log_info "PHASE 1: TOKEN METADATA COLLECTION"
log_info "========================================"

log_info "Story 1.2: Collecting tokens from CoinGecko..."
python services/tokens/token_collection_service.py
log_info "✓ Token collection complete"

log_info "Story 1.3: Analyzing DEX liquidity..."
python cli_liquidity.py
log_info "✓ Liquidity analysis complete"

log_info "Story 1.4: Narrative categorization..."
python narrative_classification_orchestrator.py
log_warning "Manual review required - edit outputs/csv/tokens_for_review.csv"
read -p "Press Enter when manual review is complete..."
python narrative_classification_orchestrator.py --import outputs/csv/tokens_reviewed.csv
log_info "✓ Narrative categorization complete"

# PHASE 2: WALLET IDENTIFICATION
log_info "========================================"
log_info "PHASE 2: SMART WALLET IDENTIFICATION"
log_info "========================================"

log_info "Story 2.1: Identifying smart money wallets..."
python services/smart_wallet_query_manager.py
log_info "✓ Wallet identification complete"

log_info "Story 2.2: Calculating performance metrics..."
python src/services/wallets/performance_calculator.py
log_info "✓ Performance calculation complete"

# PHASE 3: TRANSACTION & BALANCE DATA
log_info "========================================"
log_info "PHASE 3: TRANSACTION & BALANCE DATA"
log_info "========================================"

log_info "Story 3.1: Extracting transaction history..."
log_warning "This will take 12-24 hours. Progress: logs/collection_YYYYMMDD.log"
python services/transactions/batch_processor.py
log_info "✓ Transaction extraction complete"

log_info "Story 3.3: Collecting ETH price history..."
python services/prices/chainlink_client.py
log_info "✓ Price collection complete"

# PHASE 4: VALIDATION & EXPORT
log_info "========================================"
log_info "PHASE 4: VALIDATION & EXPORT"
log_info "========================================"

log_info "Story 3.4: Running data validation..."
python services/validation/quality_reporter.py
log_info "✓ Validation complete"

log_info "Story 3.5: Exporting final datasets..."
python services/export/data_exporter.py
log_info "✓ Export complete"

# SUMMARY
log_info "========================================"
log_info "PIPELINE EXECUTION COMPLETE"
log_info "========================================"
log_info "Final datasets available in: outputs/"
log_info "Documentation available in: outputs/documentation/"
log_info "Quality report: outputs/json/quality_certification.json"

# Display final statistics
log_info "Final Statistics:"
psql $DATABASE_URL -c "
    SELECT
        (SELECT COUNT(*) FROM tokens) as tokens,
        (SELECT COUNT(*) FROM wallets) as wallets,
        (SELECT COUNT(*) FROM transactions) as transactions,
        (SELECT COUNT(*) FROM wallet_balances) as balances
    ;
"

log_info "✓ Ready for analysis phase!"
```

**Usage:**
```bash
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

---

## Monitoring & Checkpoints

### Checkpoint System

The pipeline uses a checkpoint table to track progress and enable recovery from failures.

**View all checkpoints:**
```bash
psql $DATABASE_URL -c "SELECT * FROM collection_checkpoints ORDER BY updated_at DESC;"
```

**View specific checkpoint:**
```bash
uv run data-collection checkpoint-show --type <TYPE>

# Available types:
# - tokens
# - liquidity_analysis
# - narrative_classification
# - wallet_identification
# - performance_calculation
# - transaction_extraction
# - balance_collection
# - price_collection
# - validation
```

**Update checkpoint manually:**
```bash
uv run data-collection checkpoint-update \
    --type transaction_extraction \
    --status in_progress \
    --records 125000 \
    --block 18500000
```

### Progress Monitoring

**Real-time log monitoring:**
```bash
# Follow logs in real-time
tail -f logs/collection_$(date +%Y%m%d).log

# Filter for errors
tail -f logs/collection_$(date +%Y%m%d).log | grep ERROR

# Filter for specific component
tail -f logs/collection_$(date +%Y%m%d).log | grep "transaction_extraction"
```

**Database monitoring:**
```bash
# Count records in each table
psql $DATABASE_URL -c "
    SELECT
        'tokens' as table_name, COUNT(*) as record_count FROM tokens
    UNION ALL
        SELECT 'wallets', COUNT(*) FROM wallets
    UNION ALL
        SELECT 'transactions', COUNT(*) FROM transactions
    UNION ALL
        SELECT 'wallet_balances', COUNT(*) FROM wallet_balances
    UNION ALL
        SELECT 'eth_prices', COUNT(*) FROM eth_prices;
"

# View transaction processing rate
psql $DATABASE_URL -c "
    SELECT
        DATE(timestamp) as date,
        COUNT(*) as transactions
    FROM transactions
    GROUP BY DATE(timestamp)
    ORDER BY date DESC
    LIMIT 30;
"
```

### Recovery from Failures

The system is designed for automatic recovery:

**1. Automatic Resume:**
```bash
# Simply re-run the failed command
python services/transactions/batch_processor.py

# The system will:
# - Read the last checkpoint
# - Skip already processed data
# - Continue from the last successful point
```

**2. Manual Checkpoint Reset (if needed):**
```bash
# Reset checkpoint to start over
psql $DATABASE_URL -c "
    UPDATE collection_checkpoints
    SET status = 'pending',
        last_processed_block = 0,
        records_collected = 0
    WHERE collection_type = 'transaction_extraction';
"
```

**3. Partial Data Cleanup:**
```bash
# Remove incomplete data for a specific wallet
psql $DATABASE_URL -c "
    DELETE FROM transactions
    WHERE wallet_address = '0x...'
    AND timestamp > '2024-01-01';
"
```

---

## API Rate Limits & Considerations

### CoinGecko API

**Rate Limits:**
- Free tier: 10-50 calls/minute (endpoint dependent)
- Demo/Pro tier: Higher limits with API key

**Optimization Strategies:**
- Use pagination efficiently
- Cache successful responses in `cache/coingecko/`
- Implement exponential backoff on 429 responses
- Monitor daily/monthly limits

**Cost Considerations:**
- Free tier sufficient for one-time collection
- Cached responses enable deterministic re-runs

---

### Dune Analytics

**Rate Limits:**
- Credit-based system (not time-based)
- Query complexity affects credit consumption

**Optimization Strategies:**
- Use preview runs with 7-day windows before full execution
- Cache completed query results aggressively
- Batch multiple tokens per query
- Reuse subquery results across related queries

**Cost Considerations:**
- Monitor credit usage closely
- Typical usage: 500-1000 credits for full pipeline
- Use cached results during development/testing

**Query Optimization:**
```sql
-- Use indexed columns for filtering
WHERE block_time >= '{{start_date}}'
    AND block_time <= '{{end_date}}'
    AND trader_address IN (...)

-- Aggregate efficiently
GROUP BY trader_address
HAVING SUM(amount_usd) >= {{min_volume}}
```

---

### Alchemy RPC

**Rate Limits:**
- Compute unit (CU) based system
- Free tier: 300M CU/month
- Different methods consume different CUs

**Compute Unit Costs:**
- `eth_getLogs`: 75 CU per request
- `eth_getTransactionReceipt`: 15 CU
- `eth_call`: 26 CU
- `eth_getBlockByNumber`: 16 CU

**Optimization Strategies:**
- Use indexed topics for efficient log filtering
- Batch requests where possible
- Prefer `eth_getLogs` over multiple `eth_getTransaction` calls
- Monitor CU usage with tracking

**CU Optimization Example:**
```python
# Efficient: Single eth_getLogs call (75 CU)
logs = await client.eth_get_logs({
    'fromBlock': hex(start_block),
    'toBlock': hex(end_block),
    'topics': [SWAP_EVENT_SIGNATURE, None, hex_to_32_bytes(wallet_address)]
})

# Inefficient: Multiple eth_getTransaction calls (15 CU × N transactions)
for tx_hash in tx_hashes:
    tx = await client.eth_get_transaction(tx_hash)
```

---

### Etherscan API

**Rate Limits:**
- Free tier: 5 requests/second
- Used only as fallback verification

**Optimization Strategies:**
- Use only for spot-checking and verification
- Not primary data source
- Respect rate limits with delays

---

### Uniswap Subgraphs

**Rate Limits:**
- Generally permissive for public access
- Pagination required for large result sets

**Optimization Strategies:**
- Use proper pagination (first, skip parameters)
- Request only required fields
- Cache results for development

---

### Chainlink Price Feeds

**Rate Limits:**
- On-chain calls via Alchemy RPC (26 CU per call)
- No direct rate limits (blockchain-based)

**Optimization Strategies:**
- Cache hourly prices aggressively
- Batch historical backfill
- Use efficient block lookup strategies

---

## Data Quality Targets

### Completeness Targets

| Dataset | Target | Metric |
|---------|--------|--------|
| Tokens | 95% | Valid contract addresses |
| Wallets | 90% | Performance metrics calculated |
| Transactions | 95% | Successful extraction vs Etherscan |
| Balances | 95% | Daily snapshots coverage |
| Prices | 99% | Hourly price availability |

### Accuracy Targets

| Validation Type | Target | Measurement |
|----------------|--------|-------------|
| Transaction Decoding | 99% | Spot-check vs Etherscan |
| Price Accuracy | 98% | <2% deviation from consensus |
| Balance Reconciliation | 94% | Transaction-balance consistency |
| Performance Metrics | 95% | Calculation verification |

### Consistency Targets

| Consistency Check | Target | Tolerance |
|------------------|--------|-----------|
| Cross-source validation | 95% | <5% variance |
| Temporal consistency | 95% | No logical conflicts |
| Referential integrity | 100% | No orphaned records |

### Overall Quality Score

**Composite Score Formula:**
```
Quality Score = (
    Completeness × 0.25 +
    Accuracy × 0.25 +
    Consistency × 0.20 +
    Timeliness × 0.15 +
    Validity × 0.15
)
```

**Quality Grades:**
- **A+ (≥95%):** Production ready, high confidence
- **A (≥90%):** Production ready with minor caveats
- **B+ (≥85%):** Ready with limitations documented
- **B (≥80%):** Ready for exploratory analysis
- **C (≥70%):** Requires quality improvements
- **F (<70%):** Not ready for analysis

**Minimum Acceptable Quality:** Grade A (≥90%)

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Database Connection Failures

**Symptom:**
```
Error: could not connect to database
```

**Solutions:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres
# or
pg_isready -h localhost -p 5432

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Test connection manually
psql $DATABASE_URL -c "SELECT 1;"

# Restart PostgreSQL
docker-compose restart postgres
```

---

#### 2. API Authentication Errors

**Symptom:**
```
Error 401: Unauthorized
Error 403: Forbidden
```

**Solutions:**
```bash
# Verify API keys in .env
cat .env | grep API_KEY

# Test CoinGecko key
curl -H "x-cg-demo-api-key: YOUR_KEY" \
  "https://api.coingecko.com/api/v3/ping"

# Test Alchemy key
curl https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Regenerate keys if needed
```

---

#### 3. Rate Limit Exceeded

**Symptom:**
```
Error 429: Too Many Requests
```

**Solutions:**
```bash
# Check cache for existing data
ls -lh cache/coingecko/
ls -lh cache/dune_queries/

# Increase backoff delays in code
# (Already implemented with exponential backoff + jitter)

# Wait and retry
sleep 60 && python services/tokens/token_collection_service.py

# Consider upgrading API tier if persistent
```

---

#### 4. Transaction Extraction Timeout

**Symptom:**
```
Error: eth_getLogs request timeout
Block range too large
```

**Solutions:**
```bash
# The batch processor automatically handles this with:
# - Adaptive block range sizing
# - Smaller ranges for high-activity periods
# - Retry logic with exponential backoff

# Manual adjustment if needed (in code):
# services/transactions/batch_processor.py
BLOCK_RANGE_SIZE = 1000  # Reduce from default 2000

# Resume from checkpoint
python services/transactions/batch_processor.py
```

---

#### 5. Memory Issues During Processing

**Symptom:**
```
MemoryError: Unable to allocate array
Process killed (OOM)
```

**Solutions:**
```bash
# Reduce batch size
# services/transactions/batch_processor.py
WALLET_BATCH_SIZE = 50  # Reduce from 100

# Use streaming for large datasets
# (Already implemented in export services)

# Increase system memory if available

# Process in smaller chunks manually
python services/transactions/batch_processor.py --start-index 0 --end-index 1000
python services/transactions/batch_processor.py --start-index 1000 --end-index 2000
```

---

#### 6. Data Validation Failures

**Symptom:**
```
Validation failed: Balance reconciliation error
Consistency check failed: Transaction mismatch
```

**Solutions:**
```bash
# Review validation report
cat outputs/json/quality_report.json | jq '.validation_results'

# Check for missing transactions
psql $DATABASE_URL -c "
    SELECT wallet_address, COUNT(*)
    FROM transactions
    GROUP BY wallet_address
    HAVING COUNT(*) < 10;
"

# Re-run extraction for specific wallets
python services/transactions/batch_processor.py --wallet-list failed_wallets.txt

# Check for price data gaps
psql $DATABASE_URL -c "
    SELECT DATE(timestamp), COUNT(*)
    FROM eth_prices
    WHERE timestamp >= '2024-01-01'
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp);
"
```

---

#### 7. Checkpoint Corruption

**Symptom:**
```
Error: Checkpoint data inconsistent
Invalid checkpoint state
```

**Solutions:**
```bash
# View current checkpoint
uv run data-collection checkpoint-show --type transaction_extraction

# Reset checkpoint
psql $DATABASE_URL -c "
    UPDATE collection_checkpoints
    SET status = 'pending',
        last_processed_block = 0
    WHERE collection_type = 'transaction_extraction';
"

# Verify data before reset
psql $DATABASE_URL -c "
    SELECT MIN(block_number), MAX(block_number), COUNT(*)
    FROM transactions;
"
```

---

#### 8. Dune Query Failures

**Symptom:**
```
Error: Query execution timeout
Error: Dune credits exhausted
```

**Solutions:**
```bash
# Use preview run with smaller date range
# Modify query parameters:
start_date = '2024-09-01'  # 7-day window
end_date = '2024-09-07'

# Check query results cache
ls -lh cache/dune_queries/

# Optimize query (add filters, reduce date range)
# See sql/dune_queries/ for query files

# Monitor credit usage via Dune dashboard
# https://dune.com/settings/api
```

---

### Getting Help

**Log Files:**
```bash
# View recent errors
tail -n 100 logs/collection_$(date +%Y%m%d).log | grep ERROR

# Search for specific errors
grep -r "MemoryError" logs/

# View structured logs
cat logs/collection_$(date +%Y%m%d).log | jq '.error'
```

**Debug Mode:**
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with debug output
python services/tokens/token_collection_service.py 2>&1 | tee debug.log
```

**Support Resources:**
- **GitHub Issues:** BMAD_TFM/issues
- **Documentation:** docs/data-collection-phase/
- **Architecture:** docs/data-collection-phase/architecture.md

---

## Expected Timeline

### Phase 1: Token Metadata Collection
- **Story 1.2 (CoinGecko):** 30-60 minutes
- **Story 1.3 (Liquidity):** 1-2 hours
- **Story 1.4 (Categorization):** 2-3 hours (includes manual review)
- **Total:** 4-6 hours

### Phase 2: Smart Wallet Identification
- **Story 2.1 (Queries):** 2-4 hours
- **Story 2.2 (Performance):** 3-5 hours
- **Total:** 5-9 hours

### Phase 3: Transaction & Balance Data
- **Story 3.1 (Transactions):** 12-24 hours ⚠️ *Longest phase*
- **Story 3.2 (Balances):** 6-8 hours
- **Story 3.3 (Prices):** 1-2 hours
- **Story 3.4 (Validation):** 2-3 hours
- **Story 3.5 (Export):** 2-3 hours
- **Total:** 23-40 hours

### Overall Timeline
**Sequential Execution:** 32-55 hours (~2-3 days)
**Parallel Optimization:** 20-35 hours (with concurrent processing)

**Recommended Approach:**
1. Run Phase 1 & 2 during business hours (manual review needed)
2. Start Phase 3.1 (transactions) overnight
3. Complete remaining phases next business day

---

## Final Deliverables

After successful pipeline execution, you will have:

### 1. Database Tables (PostgreSQL)
- ✅ **tokens** (~500 records) - Categorized tokens with liquidity tiers
- ✅ **wallets** (8-12K records) - Smart money wallet cohort
- ✅ **transactions** (500K-2M records) - Complete swap history
- ✅ **wallet_balances** (100K-500K records) - Daily portfolio snapshots
- ✅ **eth_prices** (8,000+ records) - Hourly ETH/USD prices
- ✅ **wallet_performance** (8-12K records) - Calculated metrics

### 2. Export Files

**Parquet (Analysis-Ready):**
```
outputs/parquet/
├── tokens.parquet (~100KB)
├── wallets.parquet (~2MB)
├── transactions/ (monthly partitions, ~500MB total)
├── wallet_balances/ (monthly partitions, ~200MB total)
├── eth_prices.parquet (~1MB)
└── wallet_performance.parquet (~5MB)
```

**CSV (Human-Readable):**
```
outputs/csv/
└── (same structure, ~2GB total)
```

**JSON (Metadata):**
```
outputs/json/
├── data_dictionary.json
├── export_manifest.json
├── quality_certification.json
└── lineage_documentation.json
```

### 3. Documentation

```
outputs/documentation/
├── methodology.md - Complete data collection methodology
├── data_dictionary.md - Field-level documentation
├── quality_report.pdf - Validation results and certification
├── user_guide.md - Analyst quickstart guide
├── api_reference.md - Programmatic access documentation
└── troubleshooting.md - Common issues and solutions
```

### 4. Quality Certification

**Expected Quality Metrics:**
- **Overall Quality Score:** 0.92 (Grade A)
- **Completeness:** 97% average across all datasets
- **Accuracy:** 98% validation pass rate
- **Consistency:** 95% cross-source agreement
- **Certification Status:** ✅ CERTIFIED FOR ANALYSIS

### 5. Analysis-Ready Features

**Token Universe:**
- 500 tokens with complete metadata
- Narrative categorization (7 categories)
- Liquidity tier assignments
- Market cap and volume data

**Smart Money Cohort:**
- 8,000-12,000 validated wallets
- Bot-filtered and Sybil-resistant
- Performance metrics (Sharpe, win rate, etc.)
- Risk-adjusted returns calculated

**Transaction Data:**
- Complete DEX swap history
- ETH-denominated values
- USD valuations at transaction time
- MEV impact analysis
- Gas cost attribution

**Portfolio Tracking:**
- Daily balance snapshots
- Multi-token position tracking
- Historical valuations
- Performance attribution

---

## Usage for Analysis Phase

### Loading Data (Python)

```python
import pandas as pd
import pyarrow.parquet as pq

# Load tokens
tokens_df = pd.read_parquet('outputs/parquet/tokens.parquet')

# Load wallets with performance metrics
wallets_df = pd.read_parquet('outputs/parquet/wallets.parquet')
performance_df = pd.read_parquet('outputs/parquet/wallet_performance.parquet')

# Load transactions (specific month)
tx_202401 = pd.read_parquet('outputs/parquet/transactions/transactions_2024_01.parquet')

# Or load all transactions
import glob
tx_files = glob.glob('outputs/parquet/transactions/*.parquet')
all_transactions = pd.concat([pd.read_parquet(f) for f in tx_files])

# Load balances and prices
balances_df = pd.read_parquet('outputs/parquet/wallet_balances/wallet_balances_2024_01.parquet')
prices_df = pd.read_parquet('outputs/parquet/eth_prices.parquet')
```

### Common Analysis Queries

```python
# Top performing wallets by Sharpe ratio
top_wallets = performance_df.nlargest(100, 'sharpe_ratio')

# Token distribution by narrative
narrative_dist = tokens_df['narrative_category'].value_counts()

# Transaction volume by DEX
dex_volume = all_transactions.groupby('dex_name')['eth_value_in_usd'].sum()

# Wallet activity timeline
wallet_activity = all_transactions.groupby([
    pd.Grouper(key='timestamp', freq='D'),
    'wallet_address'
]).size()
```

---

## Next Steps: Analysis Phase

With the data collection complete, you can now proceed to:

1. **Narrative Trend Analysis**
   - Time-series analysis of narrative popularity
   - Smart money allocation patterns by narrative
   - Leading indicators of narrative emergence

2. **Smart Money Behavior Modeling**
   - Trading pattern recognition
   - Entry/exit timing analysis
   - Portfolio construction strategies

3. **Predictive Modeling**
   - ML models for narrative trend prediction
   - Smart money behavior forecasting
   - Risk assessment models

4. **Visualization & Reporting**
   - Interactive dashboards
   - Portfolio tracking interfaces
   - Research publications

---

## Document Maintenance

**Version Control:**
- Document version tracked in header
- Changes logged in git commit history

**Update Frequency:**
- Review quarterly or after major changes
- Update immediately for critical corrections

**Feedback:**
- Report issues or improvements via GitHub Issues
- Contribute improvements via Pull Requests

---

## Appendix: Quick Reference

### Essential Commands

```bash
# Database operations
uv run data-collection init-db
uv run data-collection health
uv run data-collection checkpoint-show --type <TYPE>

# Phase 1: Token Collection
python services/tokens/token_collection_service.py
python cli_liquidity.py
python narrative_classification_orchestrator.py

# Phase 2: Wallet Identification
python services/smart_wallet_query_manager.py
python src/services/wallets/performance_calculator.py

# Phase 3: Transaction & Balance Data
python services/transactions/batch_processor.py
python services/prices/chainlink_client.py

# Phase 4: Validation & Export
python services/validation/quality_reporter.py
python services/export/data_exporter.py

# Monitoring
tail -f logs/collection_$(date +%Y%m%d).log
psql $DATABASE_URL -c "SELECT COUNT(*) FROM transactions;"
```

### Key File Locations

```
BMAD_TFM/data-collection/
├── .env - API keys and configuration
├── pyproject.toml - Python dependencies
├── services/ - Data collection services
├── tests/ - Test suites
├── sql/ - Database schema and queries
├── cache/ - API response caches
├── logs/ - Structured log files
└── outputs/ - Exported datasets
    ├── parquet/ - Analysis-ready data
    ├── csv/ - Human-readable exports
    ├── json/ - Metadata and schemas
    └── documentation/ - Generated docs
```

---

**End of Operational Guide**

For additional support, consult:
- Architecture documentation: `docs/data-collection-phase/architecture.md`
- User stories: `docs/data-collection-phase/stories/`
- Implementation guide: `docs/data-collection-phase/implementation-guide.md`