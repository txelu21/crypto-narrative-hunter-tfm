# Ethereum Data Collection Pipeline – Technical Architecture

Version: 1.0 (2025-09-27)
Scope: Implements the PRD “Ethereum Data Collection Pipeline” with a production‑grade, budget‑aware architecture that can be delivered in 13 days and stays within free-tier limits.

## 1) Objectives, Constraints, and Success Criteria

- Deliver analysis-ready on-chain datasets for the top-500 ERC‑20 tokens and 10k smart wallets, with values normalized to ETH then USD.
- Complete first dataset within 13 days; total Parquet size < 1GB; >95% completeness; reproducible end-to-end.
- Operate on $0 budget: Dune (≤1,000 credits/mo), Alchemy (≤100M compute units), Etherscan (≤5 rps), CoinGecko free tier.
- Run on commodity hardware (≤16 GB RAM); robust to interruptions; resumable via checkpoints.

## 2) High-Level System Design

Architecture style: Modular batch pipeline with three services, each idempotent and restartable:

- Token Metadata Collection Service (CoinGecko + liquidity augmentation via Dune/Uniswap/Curve)
- Wallet Identification Service (Dune SQL for Uniswap/Curve behavior)
- Transaction & Balances Extraction Service (Alchemy/Etherscan + on-chain queries)

Execution pattern: Manual CLI entry points per stage; resumable with checkpoints. Concurrency is bounded per-integration to respect rate limits.

### 2.1 Logical Components

- Ingestion Layer: HTTP clients for CoinGecko, Dune API, Uniswap Subgraphs (V2/V3 GraphQL), Curve API, Alchemy JSON‑RPC, Etherscan REST.
- Processing Layer: Async workers with bounded concurrency, dedupe, decoding, pricing normalization (Token → ETH → USD), and validation.
- Storage Layer: Postgres (intermediate, idempotent upserts), Parquet (final, snappy compression), CSV (select backups).
- Control Layer: Checkpoints, retries with backoff, structured logging, metrics, progress reporting.
- Auditability: Request/response hashes, tx hash verification, rule-driven filtering recorded to an audit log.

### 2.2 Reference Data Flow (text diagram)

1) Token Universe → CoinGecko top‑500 → enrich (Uniswap/Curve pools via Dune/Subgraphs) → tokens table → Parquet export.
2) Smart Wallets → Dune SQL on Uniswap/Curve → quality filters → wallets table → CSV/Parquet export.
3) Tx & Balances → Alchemy (getLogs/call) + Etherscan (fallback) → decode swaps → pricing to ETH (DEX pools) → USD (Chainlink) → Postgres → Parquet.

## 3) Module Designs

### 3.1 Token Metadata Collection Service

Responsibilities
- Fetch top‑500 ERC‑20 tokens from CoinGecko (contract address, symbol, name, decimals, market cap, 24h volume, rank).
- Validate Ethereum mainnet addresses; dedupe symbols; preserve ranks.
- Enrich liquidity: For each token, discover Uniswap V2/V3 and Curve pools; compute TVL tiers; capture token⇄ETH rates from top pools.
- Assign narrative category (rule-based + manual review file).

Inputs/Outputs
- Inputs: CoinGecko pages; Uniswap/Curve pool metadata; narrative rules file.
- Outputs: tokens table; liquidity tier; CSV export for review; Parquet final.

Performance & Limits
- Paginate CoinGecko; sleep/backoff on 429; cache responses to local disk for deterministic re-runs.
- Dune queries cached by query-id+parameters; reuse results for subsequent runs to conserve credits.

Edge Cases
- Duplicate addresses with multiple tickers; missing decimals; non-ETH chains; deprecations; low-liquidity tokens tagged Tier 3.

### 3.2 Wallet Identification Service

Responsibilities
- Execute parameterized Dune SQL to find wallets with Uniswap/Curve volume > thresholds (30‑day and 3‑month windows).
- Exclude contracts/routers/aggregators; detect bots (MEV, ultrahigh frequency) using heuristics + public lists.
- Compute metrics: trade counts, volumes (ETH), unique tokens, first/last activity, gas/volume efficiency.

Inputs/Outputs
- Inputs: Dune SQL endpoints; MEV/bot lists (static files).
- Outputs: wallets table; funnel report; CSV export; checkpoint of selected 8–12k addresses.

Performance & Limits
- Use preview runs (7‑day) to validate and refine SQL; cache results; export to Postgres in batches (1k rows/chunk).

Edge Cases
- Contract accounts misclassified; sybil clusters; time-bounded bursts; missing gas on some aggregates.

### 3.3 Transaction & Balances Extraction Service

Responsibilities
- Pull Uniswap V2/V3 and Curve swap activity per target wallet; decode router calls and events; include failed tx flagged.
- Daily balances for top‑500 tokens + ETH using Multicall at end-of-day blocks.
- Price normalization: token amounts → ETH using pool price at tx time; ETH → USD using hourly Chainlink prices.
- Integrity: de‑dupe by tx_hash; verify against Etherscan when uncertain.

Inputs/Outputs
- Inputs: Wallet list; token list; Alchemy RPC; Uniswap subgraphs; Curve API; Chainlink feeds; Etherscan fallback.
- Outputs: transactions, wallet_balances, eth_prices tables; Parquet exports; quality reports.

Performance & Limits
- Bounded concurrency per wallet (e.g., 5–10), per provider (e.g., 2–4 for Dune, 10–20 for Alchemy depending on CU costs).
- Use block range partitioning and continuation tokens; resume from last processed block/date via checkpoints.

Edge Cases
- Very active wallets (>10k tx); pool renames; token decimal anomalies; chain reorgs near boundaries.

## 4) Data Model and Storage

Use the PRD schema verbatim in Postgres for staging. Final exports use Parquet (snappy) partitioned by table and time where applicable.

- tokens(token_address PK, symbol, name, decimals, narrative_category, market_cap_rank, avg_daily_volume_usd, liquidity_tier, created_at, updated_at)
- wallets(wallet_address PK, first_seen_date, last_active_date, total_trades_30d, avg_daily_volume_eth, unique_tokens_traded, is_smart_money, created_at)
- transactions(tx_hash PK, block_number, timestamp, wallet_address FK, dex_name, pool_address, token_in, amount_in, token_out, amount_out, gas_used, gas_price_gwei, eth_value_in, eth_value_out)
- wallet_balances(id PK, wallet_address FK, token_address FK, snapshot_date, balance, eth_value, UNIQUE(wallet_address, token_address, snapshot_date))
- eth_prices(timestamp PK, price_usd, source)
- collection_checkpoints(id PK, collection_type, last_processed_block, last_processed_date, records_collected, status, created_at, updated_at)

Schema enforcement: Postgres constraints, NOT NULL where applicable, CHECK constraints for tiers and enums, and upsert (ON CONFLICT) for idempotency.

Partitioning: For Parquet, partition by month for transactions and balances to accelerate analysis and keep total size under 1GB.

## 5) API Integration Patterns

### 5.1 CoinGecko
- REST with pagination for markets listing filtered by “category=ethereum-ecosystem” or contract lookup API for ERC‑20s.
- Backoff on 429 with exponential jitter; cache successful pages to disk (JSONL) keyed by URL hash.

### 5.2 Dune Analytics
- Parameterized SQL stored in repo with query IDs; execute via API and poll job status.
- Cache completed result IDs; serialize to parquet/json locally to avoid re-spend of credits.
- Use narrow date windows for development; widen for final once queries are stable.

### 5.3 Uniswap Subgraphs (V2/V3)
- GraphQL queries for pools, token liquidity, and historical snapshots where available.
- For V3: fetch slot0 sqrtPriceX96 and liquidity when computing prices.

### 5.4 Curve API
- REST endpoints for pools, virtual price, and coin indices; prefer ETH‑based pools for conversion to ETH.

### 5.5 Alchemy JSON‑RPC
- Primary on-chain access for logs (eth_getLogs with block windows), calls (eth_call), and receipts (eth_getTransactionReceipt).
- Compute unit awareness: prefer fewer eth_getLogs with reasonable block spans and indexed topics; avoid per‑tx eth_getTransaction when not needed.

### 5.6 Etherscan (Fallback)
- REST for tx list per address (throttled at 5 rps) and event verification when Alchemy/subgraph data is inconclusive.

### 5.7 Chainlink Prices
- Read on-chain aggregator via eth_call at hourly block heights; cache hourly ETH/USD to avoid repeated calls.

## 6) Decoding and Pricing Algorithms

### 6.1 Uniswap V2
- Identify swaps via Pair Swap events and/or Router function signatures.
- Pricing to ETH: prefer direct token↔WETH pools; otherwise route token→USDC→WETH if needed with caution (only for pricing, not path discovery).

### 6.2 Uniswap V3
- Decode via pool events or SwapRouter calls; handle tick math.
- Price formula: For token0→token1 price at time t using the pool’s slot0 sqrtPriceX96,
  
  $price_{1/0} = \frac{(sqrtPriceX96)^2}{2^{192}} \times 10^{decimals_0 - decimals_1}$
  
- Convert amounts using the instantaneous price; prefer time‑weighted if subgraph provides.

### 6.3 Curve Pools
- Use pool reserves and virtual price; for stables, near-par approximations are acceptable when TVL > $50k.
- For crypto pools, compute spot via on-chain getters or pool API; convert to ETH through pool’s ETH coin index.

### 6.4 ETH→USD Conversion
- Use hourly Chainlink ETH/USD price at the transaction hour; for balances, use the day’s nearest hour.

Precision: Keep token amounts in integer base units, convert with Decimal(36, 18) precision; round only for presentation.

## 7) Resiliency, Idempotency, and Checkpointing

- Retries: Exponential backoff with jitter, max 5 attempts (configurable); classify retryable errors (429, 5xx, timeouts) vs fatal (4xx logic issues).
- Idempotency: Upserts by natural keys (tx_hash, wallet+token+date, timestamp primary keys), dedupe in-memory by hash before DB writes.
- Checkpoints: One row per collection type capturing last processed block/date and status. On resume, read checkpoint, continue, and update atomically.
- Partial writes: Stage files to tmp/ then move to final paths on success to avoid partial Parquet corruption.

## 8) Performance and Free‑Tier Optimization

Concurrency model
- Async I/O with aiohttp/web3.py; bounded semaphores per provider (e.g., Alchemy=15, Dune=2, Etherscan=3–5) tuned empirically.

Batching
- Process wallets in shards of 100–250; for each shard, fetch logs in block windows sized by historical density (adaptive window sizing).
- Balance snapshots via Multicall batching (e.g., 100–200 calls/batch), skipping zeros to minimize storage.

Caching
- Local on-disk caches for: CoinGecko pages, Dune results, hourly ETH/USD prices, pool metadata, and ABI decoding maps.

CU and credit budgets (guardrails)
- Track per‑method counts (eth_getLogs, eth_call) and estimate CU cost; stop early if budget exceeded and persist partial results.
- Prefer subgraph data for historical pricing where accurate to reduce on-chain reads.

Memory / Storage
- Stream writes to Postgres and Parquet using chunked DataFrames; avoid whole‑dataset materialization.
- Snappy compression; drop columns not needed for analysis (keep hashes, amounts, addresses, timestamps, gas, pool, dex, values).

## 9) Logging, Monitoring, and Audit

- Structured logging JSON to logs/collection_YYYYMMDD.log with fields: ts, component, operation, params_hash, status, duration_ms, error.
- Progress bars (tqdm) for developer UX; periodic summaries per shard.
- Audit log table/file: for each filtering decision (e.g., bot exclusion), record rule_id and evidence.
- Optional: Email/Slack hook for critical failures (off by default, config-driven).

## 10) Security and Configuration

- Secrets via environment variables (.env for local; never commit). Keys: DUNE_API_KEY, ALCHEMY_API_KEY, ETHERSCAN_API_KEY, COINGECKO_API_KEY (optional).
- Principle of least privilege; sanitize logs (no secrets); redact wallet-private data (not applicable—EOA public addresses only).
- HTTP timeouts and TLS verification enabled; pin base URLs in config to avoid SSRF.

## 11) Deployment and Execution

- Language: Python 3.9+; libraries: pandas, numpy, web3.py, aiohttp, pyarrow, psycopg2/asyncpg, tenacity (backoff), tqdm, pydantic (config/schema), python-dotenv.
- Environment: venv or Docker (optional). Manual CLI subcommands:
  - tokens collect …
  - wallets discover …
  - tx extract …
  - balances snapshot …
  - prices backfill …
  - export parquet …

- Scheduling: Manual runs for MVP; each command writes checkpoint rows and resumes from last success.

## 12) Quality Assurance and Validation

- Schema validation: pydantic models for rows; DB constraints; decimal precision checks.
- Completeness: compute % non-null per critical field; target >95%.
- Cross-validation: transaction counts per wallet vs Dune aggregates; spot-check 100 tx on Etherscan.
- Reconciliation: For sample wallets, start balance + changes − gas = end balance within tolerance.
- Anomaly detection: bounds on gas_price_gwei, amount ranges, timestamps ordering.

Outputs
- Data quality report per stage, written to docs/data-collection-phase/reports/ with metrics and missingness heatmaps.

## 13) Directory Layout (proposed)

- data-collection/
  - config/ (YAML/ENV samples)
  - services/
    - tokens/
    - wallets/
    - transactions/
    - balances/
    - prices/
  - common/ (http clients, backoff, logging, db, schemas, checkpoints)
  - scripts/ (CLI)
  - tests/ (schema/validation)
  - notebooks/ (exploratory)
  - outputs/
    - parquet/
    - csv/
  - logs/

## 14) Delivery Plan (13 days)

- Days 1–2: Repo scaffolding (data-collection/), config, logging, Postgres schema, checkpoints; CoinGecko integration + token ingest.
- Days 3–4: Dune SQL finalized and cached; wallets table populated; filtering + cohorting; export lists.
- Days 5–7: Tx extraction skeleton; Uniswap V2/V3 decoding; Chainlink price backfill; ETH normalization.
- Days 8–9: Curve support; fallback to Etherscan; pagination robustness; resume logic battle‑tested.
- Days 10–11: Daily balances via Multicall; zero‑skip; incremental updates; cross-checks.
- Day 12: Data quality reports; parquet exports; size and completeness validation.
- Day 13: Documentation, data dictionary, README, final checkpoint, handoff.

## 15) Risks and Mitigations

- API budget exhaustion → Aggressive caching; progressive sampling; dry-run previews; early stop guards.
- Rate limits/timeouts → Unified backoff with jitter; dynamic concurrency throttles; resume from checkpoints.
- Decoding complexity (V3 ticks, Curve varieties) → Start with common pools; expand coverage iteratively; log unknown ABIs for later.
- Data volume >1GB → Zero-balance pruning; limit horizon to 3 months; drop non-essential columns; strong compression.
- Schedule risk → Parallelize independent tasks (prices backfill, token ingestion) and keep daily measurable outputs.

## 16) Requirements Coverage

- Functional: FR1–FR16 mapped via Services 3.1–3.3, pricing logic (Section 6), DEX focus, logs and metadata.
- Non-Functional: NFR1 size (Section 8), NFR2 timeline (Section 14), NFR3 limits (Sections 5 & 8), NFR4 memory (Section 8), NFR5 idempotency/checkpoints (Section 7), NFR6 completeness (Section 12), NFR7 precision (Section 6), NFR8 subgraphs/APIs (Section 5), NFR9 secrets (Section 10), NFR10 backoff (Section 7), NFR11 logging (Section 9), NFR12 tx hash verification (Section 3.3/12), NFR13 reproducibility (Sections 9/11/12), NFR14 parallelism (Section 8), NFR15 routers (Section 6).

## 17) Next Steps

- Approve architecture; then implement Epic 1, Story 1.1 scaffolding under data-collection/ with config, logging, DB schema, and checkpoint utilities.
- Create Dune SQL drafts and token ingestion scripts; wire caching; prepare minimal CLI.
- Establish monitoring dashboards for runtime stats (simple CSV summaries to start).
- Coordinate closely with the implementation guide (`implementation-guide.md`) for operational specifics, uv workflows, and future story roadmaps.

---

This document is the living blueprint for the MVP pipeline. All deviations should be logged in docs/data-collection-phase/CHANGELOG.md with rationale and impact on PRD requirements.
