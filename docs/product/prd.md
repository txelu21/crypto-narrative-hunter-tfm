# Ethereum Data Collection Pipeline - Product Requirements Document (PRD)

## Goals and Background Context

### Goals
- Deliver complete Ethereum blockchain raw transaction dataset (<1GB) by October 10, 2025 to enable feature engineering
- Collect daily token balance snapshots for top 500 tokens by market cap to track accumulation/distribution patterns over time
- Extract historical balance changes with all values converted to ETH equivalent for standardized comparisons
- Collect comprehensive swap transaction data including token amounts and ETH-equivalent values at transaction time
- Extract wallet address lists with transaction counts and volumes to identify potential smart money candidates
- Gather token swap events with complete input/output token amounts and ETH conversion rates
- Capture wallet portfolio composition changes valued in ETH to identify narrative rotation signals
- Collect gas fees (already in ETH) and transaction metadata for profitability calculations
- Map top 500 ERC-20 tokens to 10 predefined narrative categories for clustering analysis
- Obtain historical ETH/USD prices as the single fiat conversion reference
- Output all final metrics (ROI, PnL, portfolio values) in USD for analysis and visualization
- Provide raw datasets in Parquet format that support downstream ROI, accumulation, and performance calculations
- Document data transformation pipeline: Token Amount → ETH Equivalent → USD Value

### Background Context

This data collection pipeline serves as a critical parallel workstream for the Crypto Narrative Discovery thesis project, addressing the fundamental challenge of acquiring blockchain data at scale. The main project requires raw on-chain transaction data that will enable calculation of USD-denominated performance metrics like ROI, win rates, and portfolio values during the feature engineering phase.

The solution implements a two-stage valuation approach: first converting all token amounts to ETH equivalents using on-chain DEX pair data (readily available), then converting ETH to USD using a single historical price feed. This strategy significantly simplifies data collection while ensuring all final metrics are presented in USD for intuitive analysis and thesis presentation. By focusing on the top 500 tokens by market cap rather than 1,000, the pipeline reduces complexity while still capturing the vast majority of meaningful trading activity and narrative movements.

The pipeline focuses exclusively on Uniswap and Curve DEXs, which together represent ~80% of Ethereum DEX volume, significantly simplifying data collection while maintaining comprehensive coverage. It excludes LP positions and staked tokens for the MVP scope, focusing on simple token holdings that can be tracked through balance changes. This ensures deliverability within the critical 13-day window while providing all necessary data for sophisticated narrative clustering analysis.

### Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-09-27 | 1.0 | Initial PRD creation based on project brief | John (PM Agent) |

## Requirements

### Functional Requirements

- **FR1:** The system shall collect daily balance snapshots for the top 500 ERC-20 tokens by market cap across identified wallet addresses
- **FR2:** The system shall extract swap transactions exclusively from Uniswap V2, Uniswap V3, and Curve pools for tracked wallets
- **FR3:** The system shall retrieve token-to-ETH exchange rates specifically from Uniswap and Curve liquidity pools
- **FR4:** The system shall collect and store historical ETH/USD prices at hourly intervals for the entire analysis period
- **FR5:** The system shall categorize each of the 500 tokens into one of 10 predefined narrative categories with metadata storage
- **FR6:** The system shall identify wallet addresses based on their Uniswap and Curve trading activity meeting tiered volume thresholds
- **FR7:** The system shall calculate 30-day average trading volumes using only Uniswap and Curve transaction data
- **FR8:** The system shall capture complete transaction metadata including gas fees, block numbers, timestamps, and transaction hashes
- **FR9:** The system shall validate token liquidity by checking for active pools on Uniswap V2/V3 or Curve with minimum $50K TVL
- **FR10:** The system shall export all collected data in compressed Parquet format optimized for pandas DataFrame loading
- **FR11:** The system shall generate a data quality report showing completeness percentages and missing data patterns
- **FR12:** The system shall implement incremental data collection to resume after interruptions without re-downloading
- **FR13:** The system shall maintain an audit log of all API calls, data transformations, and filtering decisions
- **FR14:** The system shall identify and decode Uniswap V2/V3 and Curve swap events using their specific contract ABIs
- **FR15:** The system shall handle Uniswap V3's concentrated liquidity price ranges for accurate token pricing
- **FR16:** The system shall process Curve's stableswap and crypto pool types for multi-asset pools

### Non-Functional Requirements

- **NFR1:** The complete dataset must not exceed 1GB in storage when compressed in Parquet format
- **NFR2:** The data pipeline must complete initial collection within 13 days (by October 10, 2025)
- **NFR3:** The system must operate within free-tier API limits (Dune: 1000 credits/month, Alchemy: 100M compute units, Etherscan: 5 calls/second)
- **NFR4:** The system must run on machines with maximum 16GB RAM without memory overflow
- **NFR5:** Data extraction scripts must be idempotent and resumable after failures
- **NFR6:** The system must achieve >95% data completeness for identified wallets' transaction history
- **NFR7:** All data transformations must maintain 18 decimal precision for token amounts
- **NFR8:** The pipeline must efficiently query Uniswap Subgraph and Curve API endpoints
- **NFR9:** API keys and credentials must be stored in environment variables, never in code
- **NFR10:** The system must implement exponential backoff for rate-limited API calls with maximum 5 retry attempts
- **NFR11:** Data collection logs must be human-readable and include timestamp, operation, and result status
- **NFR12:** The system must validate data integrity using transaction hash verification before storage
- **NFR13:** Documentation must include reproducible steps allowing third-party validation of methodology
- **NFR14:** The pipeline must support parallel processing of independent data streams where possible
- **NFR15:** The system must handle Uniswap router contracts (V2 Router, V3 SwapRouter) for transaction routing

## Technical Assumptions

### Repository Structure: Monorepo

The data collection pipeline will be integrated as a module within the existing Crypto Narrative Hunter monorepo at `/data-collection/`, maintaining consistency with the main project structure while allowing independent execution.

### Service Architecture

**Batch Processing Pipeline** - The system will implement a sequential batch processing architecture with three distinct stages:
1. **Token Metadata Collection Service** - Downloads and categorizes top 500 tokens with narrative mapping
2. **Wallet Identification Service** - Queries Uniswap/Curve activity to identify high-value traders
3. **Transaction Extraction Service** - Retrieves historical swaps and balance snapshots

This modular approach allows for independent testing and restart capabilities for each stage, critical for meeting the 13-day deadline.

### Testing Requirements

**Data Validation Testing** - Given the academic nature and time constraints, the focus will be on data integrity validation rather than traditional unit testing:
- Schema validation for all collected data
- Completeness checks (>95% required fields populated)
- Cross-validation between data sources (transaction counts match across APIs)
- Sample data verification against Etherscan for accuracy
- Automated data quality reports after each collection run

### Additional Technical Assumptions and Requests

**Programming Language & Core Libraries:**
- Python 3.9+ for all data collection scripts
- pandas and numpy for data manipulation
- web3.py for Ethereum interaction
- aiohttp for async API calls
- pyarrow for Parquet file handling

**Data Storage Architecture:**
- Local PostgreSQL for intermediate storage during collection
- Final export to Parquet files organized by data type (wallets/, transactions/, tokens/)
- CSV backups for critical datasets

**API Integration Strategy:**
- Primary: Dune Analytics for aggregated DEX data
- Secondary: Alchemy for real-time blockchain queries
- Fallback: Etherscan for validation and missing data
- Uniswap Subgraph for V2/V3 specific queries
- Curve API for pool-specific data

**Performance Optimizations:**
- Async/concurrent API calls where rate limits allow
- Batch processing in 1000-record chunks
- In-memory caching for frequently accessed token metadata
- Connection pooling for database operations

**Development Environment:**
- Jupyter notebooks for exploratory queries and documentation
- Git for version control with `.gitignore` for API keys
- Virtual environment with requirements.txt for dependencies
- Environment variables for all sensitive configuration

**Monitoring & Logging:**
- Structured logging to `logs/collection_YYYYMMDD.log`
- Progress tracking with tqdm for long-running operations
- Slack/email alerts for critical failures (optional if time allows)
- Daily collection summary reports

**Data Pipeline Scheduling:**
- Manual execution for MVP (no cron/orchestration required)
- Checkpoint files to track collection progress
- Ability to resume from last successful checkpoint

**Database Schema Design:**

```sql
-- Token metadata and narrative mapping
tokens (
    token_address VARCHAR(42) PRIMARY KEY,
    symbol VARCHAR(20),
    name VARCHAR(100),
    decimals INT,
    narrative_category VARCHAR(50),
    market_cap_rank INT,
    avg_daily_volume_usd DECIMAL(20,2),
    liquidity_tier INT, -- 1, 2, or 3
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)

-- Wallet identification and metrics
wallets (
    wallet_address VARCHAR(42) PRIMARY KEY,
    first_seen_date DATE,
    last_active_date DATE,
    total_trades_30d INT,
    avg_daily_volume_eth DECIMAL(20,8),
    unique_tokens_traded INT,
    is_smart_money BOOLEAN,
    created_at TIMESTAMP
)

-- Raw swap transactions from Uniswap/Curve
transactions (
    tx_hash VARCHAR(66) PRIMARY KEY,
    block_number BIGINT,
    timestamp TIMESTAMP,
    wallet_address VARCHAR(42),
    dex_name VARCHAR(20), -- 'uniswap_v2', 'uniswap_v3', 'curve'
    pool_address VARCHAR(42),
    token_in VARCHAR(42),
    amount_in DECIMAL(36,18),
    token_out VARCHAR(42),
    amount_out DECIMAL(36,18),
    gas_used BIGINT,
    gas_price_gwei DECIMAL(10,2),
    eth_value_in DECIMAL(20,8),
    eth_value_out DECIMAL(20,8),
    FOREIGN KEY (wallet_address) REFERENCES wallets(wallet_address)
)

-- Daily balance snapshots
wallet_balances (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42),
    token_address VARCHAR(42),
    snapshot_date DATE,
    balance DECIMAL(36,18),
    eth_value DECIMAL(20,8),
    FOREIGN KEY (wallet_address) REFERENCES wallets(wallet_address),
    FOREIGN KEY (token_address) REFERENCES tokens(token_address),
    UNIQUE(wallet_address, token_address, snapshot_date)
)

-- ETH/USD price history
eth_prices (
    timestamp TIMESTAMP PRIMARY KEY,
    price_usd DECIMAL(10,2),
    source VARCHAR(50)
)

-- Collection metadata and progress tracking
collection_checkpoints (
    id SERIAL PRIMARY KEY,
    collection_type VARCHAR(50),
    last_processed_block BIGINT,
    last_processed_date DATE,
    records_collected INT,
    status VARCHAR(20),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

## Epic List

### Epic 1: Foundation & Multi-Source Token Intelligence
Establish the complete project infrastructure and data collection foundation while delivering a comprehensive token dataset with narrative categorization.

### Epic 2: Smart Wallet Discovery via Dune Analytics
Identify and extract comprehensive lists of smart money wallets by analyzing Uniswap and Curve trading patterns.

### Epic 3: Deep On-Chain Data via Alchemy
Extract comprehensive transaction histories and balance snapshots for all identified smart wallets using Alchemy's Web3 API.

## Epic Details

### Epic 1: Foundation & Multi-Source Token Intelligence

**Goal:** Establish the complete project infrastructure and data collection foundation while delivering a comprehensive token dataset with narrative categorization, providing immediate value for exploratory analysis and defining the token universe for subsequent wallet and transaction collection.

#### Story 1.1: Project Infrastructure Setup
**As a** data engineer,
**I want** to establish the project structure, database, and API connections,
**so that** all subsequent data collection has a solid foundation.

**Acceptance Criteria:**
1. Python project structure created under `/data-collection/` with proper package organization
2. PostgreSQL database initialized with complete schema from Technical Assumptions
3. Virtual environment configured with all required dependencies (requirements.txt)
4. Environment variables configured for API keys (Dune, Alchemy, Etherscan, CoinGecko)
5. Logging framework implemented with rotation and structured output
6. Database connection pooling configured and tested
7. Collection checkpoint system implemented and tested with resume capability
8. Git repository initialized with proper .gitignore for sensitive data

#### Story 1.2: Token Metadata Collection from CoinGecko
**As a** data analyst,
**I want** to collect metadata for the top 500 ERC-20 tokens by market cap,
**so that** I have a complete token universe with current market data.

**Acceptance Criteria:**
1. Script connects to CoinGecko API (free tier) and retrieves top 500 ERC-20 tokens
2. Token data includes: address, symbol, name, decimals, market cap, and 24h volume
3. Data validated for completeness (no null addresses or symbols)
4. Results stored in tokens table with proper decimal precision
5. Market cap rankings preserved for filtering and analysis
6. Script handles rate limiting with exponential backoff
7. Duplicate tokens (same address, different symbols) properly handled

#### Story 1.3: DEX Liquidity Analysis via Dune
**As a** data engineer,
**I want** to query Uniswap and Curve pool data from Dune Analytics,
**so that** I can identify liquid trading pairs and validate token tradability.

**Acceptance Criteria:**
1. Dune queries retrieve all Uniswap V2/V3 pools for top 500 tokens
2. Curve pool data extracted for applicable tokens (stablecoins, ETH derivatives)
3. Pool TVL calculated and stored for liquidity tier assignment
4. Token-to-ETH conversion rates captured from largest pools
5. Pools with <$50K TVL marked but included for completeness
6. Query results cached to minimize Dune credit usage
7. Liquidity tier (1, 2, or 3) assigned based on thresholds defined in requirements

#### Story 1.4: Narrative Categorization and Token Validation
**As a** data scientist,
**I want** to categorize all tokens into narrative themes and validate their data quality,
**so that** I can perform narrative-based clustering analysis.

**Acceptance Criteria:**
1. All 500 tokens assigned to one of 10 narrative categories
2. Categorization logic documented with clear rules per narrative
3. Manual review completed for ambiguous tokens (est. 30-50 tokens)
4. Token validation confirms Ethereum mainnet addresses
5. Cross-validation between CoinGecko and Dune data (address matching)
6. Daily volume calculated using 30-day average from both sources
7. Final token list exported to CSV for review with all metadata
8. Narrative distribution report generated showing token count per category

### Epic 2: Smart Wallet Discovery via Dune Analytics

**Goal:** Identify and extract comprehensive lists of smart money wallets by analyzing Uniswap and Curve trading patterns, establishing the wallet universe for detailed transaction and balance collection while optimizing Dune credit usage.

#### Story 2.1: Smart Wallet Query Development
**As a** data analyst,
**I want** to develop and test Dune SQL queries for identifying high-performance wallets,
**so that** I can efficiently extract smart money addresses.

**Acceptance Criteria:**
1. SQL query developed to identify wallets with >$100K monthly volume on Uniswap/Curve
2. Query calculates 30-day trading frequency per wallet
3. Filters exclude known contracts, routers, and aggregators
4. Query tested on 7-day sample to validate performance (should run <5 min)
5. Results include wallet address, trade count, volume in ETH, unique tokens traded
6. Query optimized to minimize Dune compute units used
7. Parameterized for different date ranges and volume thresholds

#### Story 2.2: Wallet Performance Metrics Extraction
**As a** data scientist,
**I want** to extract detailed performance metrics for identified wallets,
**so that** I can prioritize which wallets to analyze deeply.

**Acceptance Criteria:**
1. Query extracts win rate (profitable trades / total trades) per wallet
2. Average trade size in ETH calculated for each wallet
3. Most traded tokens identified (top 5 per wallet)
4. First and last trade dates captured for activity windows
5. Gas efficiency metrics calculated (gas spent vs. volume traded)
6. Results stored in wallets table with proper data types
7. Performance metrics cover full 3-month analysis period

#### Story 2.3: Wallet Filtering and Validation
**As a** data engineer,
**I want** to filter and validate the wallet list based on quality criteria,
**so that** I focus on genuinely sophisticated traders.

**Acceptance Criteria:**
1. Wallets filtered to those trading at least 10 unique tokens
2. Minimum 20 trades in 30-day period enforced
3. Bot detection applied (removing ultra-high frequency traders >1000 trades/day)
4. MEV bot addresses identified and excluded using known lists
5. Final list contains 8,000-12,000 wallets (target: 10,000)
6. Wallet addresses validated as EOAs (not contracts) where possible
7. is_smart_money flag set based on composite criteria
8. Summary report generated showing filtering funnel and reasons for exclusion

#### Story 2.4: Wallet Cohort Analysis and Export
**As a** data analyst,
**I want** to analyze wallet cohorts and export the final smart money list,
**so that** downstream collection can proceed with validated addresses.

**Acceptance Criteria:**
1. Wallets segmented into cohorts (whales >$1M, dolphins $100K-$1M, fish <$100K)
2. Narrative preferences analyzed (which narratives each wallet trades most)
3. Wallet activity patterns documented (time of day, day of week)
4. Cross-wallet correlation analysis for potential Sybil detection
5. Final wallet list exported to CSV with all metrics
6. PostgreSQL wallets table fully populated
7. Checkpoint saved marking Epic 2 completion
8. Documentation created explaining wallet selection methodology

### Epic 3: Deep On-Chain Data via Alchemy

**Goal:** Extract comprehensive transaction histories and balance snapshots for all identified smart wallets using Alchemy's Web3 API, completing the dataset with granular on-chain data needed for ROI calculations and accumulation analysis.

#### Story 3.1: Transaction History Extraction Pipeline
**As a** data engineer,
**I want** to build a robust pipeline for extracting Uniswap/Curve transactions,
**so that** I can collect complete trading histories for all smart wallets.

**Acceptance Criteria:**
1. Script connects to Alchemy API and retrieves swap events for target wallets
2. Uniswap V2/V3 Router events properly decoded with input/output amounts
3. Curve exchange events decoded including multi-asset swaps
4. Transaction data includes: hash, block, timestamp, gas used, token amounts
5. Pagination implemented to handle wallets with >10,000 transactions
6. Rate limiting respected with automatic backoff when limits approached
7. Failed transactions included but marked with status flag
8. Progress tracking shows wallets processed and estimated completion time

#### Story 3.2: Balance Snapshot Collection
**As a** data scientist,
**I want** to collect daily token balance snapshots for all wallets,
**so that** I can track accumulation patterns over time.

**Acceptance Criteria:**
1. Daily snapshots collected for all 500 tokens across all smart wallets
2. Balance queries optimized using multicall contracts for efficiency
3. Zero balances excluded to reduce storage requirements
4. ETH balance included as native token tracking
5. Snapshots cover full 3-month period at daily intervals
6. Balance changes validated against transaction history for sample wallets
7. Collection resumes from last snapshot date if interrupted
8. Storage optimized using incremental updates (only changed balances)

#### Story 3.3: ETH Price History and Value Calculations
**As a** data analyst,
**I want** to collect ETH/USD prices and calculate portfolio values,
**so that** all metrics can be presented in USD terms.

**Acceptance Criteria:**
1. Hourly ETH/USD prices collected for entire 3-month period
2. Price data sourced from reliable oracle (Chainlink via Alchemy)
3. Token amounts converted to ETH using DEX rates at transaction time
4. ETH values converted to USD using hourly price data
5. Portfolio values calculated daily for each wallet in USD
6. Transaction values (in/out) calculated in both ETH and USD
7. Gas costs converted to USD for profitability calculations
8. Price data cached locally to minimize repeated API calls

#### Story 3.4: Data Validation and Quality Assurance
**As a** data engineer,
**I want** to validate all collected data and ensure quality standards,
**so that** the dataset is reliable for analysis.

**Acceptance Criteria:**
1. Transaction counts cross-validated between Dune and Alchemy data
2. Sample of 100 transactions manually verified against Etherscan
3. Balance calculations verified: start balance + changes = end balance
4. Data completeness report shows >95% coverage for core fields
5. Duplicate transactions identified and removed
6. Data anomalies flagged (e.g., impossibly high gas prices)
7. Missing data patterns documented with explanations
8. Quality metrics stored in collection_checkpoints table

#### Story 3.5: Final Data Export and Documentation
**As a** data scientist,
**I want** to export the complete dataset in analysis-ready format,
**so that** machine learning models can begin immediately.

**Acceptance Criteria:**
1. All tables exported to Parquet files with appropriate compression
2. File sizes verified to be under 1GB total
3. Schema documentation generated with field descriptions
4. Sample data loading script provided for pandas/jupyter
5. Data dictionary created with all field definitions and units
6. Methodology document written explaining all filtering/calculations
7. README created with instructions for data access and usage
8. Final checkpoint marked showing successful pipeline completion

## Narrative Categorization Strategy

### 10 Narrative Categories with Token Distribution

1. **DeFi Infrastructure** (~80-100 tokens)
   - DEX tokens (UNI, SUSHI, CURVE)
   - Lending protocols (AAVE, COMP, MKR)
   - Yield optimizers and aggregators

2. **Layer 2 & Scaling** (~30-40 tokens)
   - L2 tokens (ARB, OP, MATIC)
   - Sidechains and scaling solutions

3. **AI & Big Data** (~40-50 tokens)
   - AI-focused projects (FET, OCEAN, RNDR)
   - Data marketplaces and compute

4. **Gaming & Metaverse** (~60-70 tokens)
   - Gaming tokens (SAND, MANA, AXS)
   - Virtual worlds and NFT gaming

5. **Real World Assets (RWA)** (~20-30 tokens)
   - Tokenized real estate, commodities
   - Bridge between TradFi and DeFi

6. **Liquid Staking & Restaking** (~20-25 tokens)
   - LSD tokens (stETH, rETH)
   - Restaking protocols

7. **Privacy & Security** (~15-20 tokens)
   - Privacy coins and protocols
   - Security-focused infrastructure

8. **Memecoins & Community** (~50-60 tokens)
   - High-volume meme tokens (SHIB, PEPE)
   - Community-driven projects

9. **Stablecoins & Payments** (~15-20 tokens)
   - Algorithmic and collateralized stables
   - Payment-focused tokens

10. **Infrastructure & Interoperability** (~40-50 tokens)
    - Oracles (LINK)
    - Cross-chain bridges
    - Core infrastructure

### Tiered Liquidity Framework

**Tier 1 - High Liquidity** (>$1M daily volume minimum)
- DeFi Infrastructure, Layer 2 & Scaling, Stablecoins & Payments

**Tier 2 - Medium Liquidity** (>$500K daily volume minimum)
- Gaming & Metaverse, AI & Big Data, Infrastructure & Interoperability

**Tier 3 - Lower Liquidity** (>$100K daily volume minimum)
- Real World Assets, Privacy & Security, Liquid Staking & Restaking

**Special Case - Memecoins** (>$250K daily volume minimum)
- High volatility requires some liquidity floor but need to capture emerging narratives

## Next Steps

### Data Collection Architect Prompt
Please review this PRD and create a comprehensive technical architecture document for the Ethereum Data Collection Pipeline. Focus on system design, API integration patterns, error handling strategies, and performance optimization techniques that will enable completion within the 13-day timeline while staying within free-tier API limits.

### Implementation Developer Prompt
Using this PRD as your guide, begin implementing Epic 1 starting with Story 1.1 (Project Infrastructure Setup). Ensure all code follows Python best practices, implements proper error handling, and includes comprehensive logging for debugging and monitoring purposes.

## Appendices

### Key Constraints
- **Budget:** $0 (must use free tiers only)
- **Timeline:** 13 days until October 10, 2025 deadline
- **Resources:** 1-2 part-time developers
- **Storage:** <1GB total dataset size
- **API Limits:** Dune (1000 credits/month), Alchemy (100M compute units), Etherscan (5 calls/sec)

### Success Criteria
- Dataset delivered by October 10, 2025
- >95% data completeness for identified wallets
- Total storage under 1GB
- 10,000 smart wallets with complete 3-month history
- All data properly documented and reproducible

---

*Document Version: 1.0*
*Created: September 27, 2025*
*Product Manager: John (PM Agent)*