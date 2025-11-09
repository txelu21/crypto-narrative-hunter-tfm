# Smart Wallet Query Documentation

## Overview

This directory contains SQL queries and supporting code for identifying smart money wallets through DEX trading volume analysis. The queries are designed to run on Dune Analytics and identify high-performing traders while excluding bots and automated systems.

## Query Files

### Core Identification Queries

#### `smart_wallet_uniswap_volume.sql`
**Purpose**: Identifies high-volume wallets on Uniswap V2/V3
**Expected Results**: 8,000-12,000 wallets
**Credit Cost**: 150-250 credits

**Parameters**:
- `start_date` (string): Analysis period start date (YYYY-MM-DD format)
- `end_date` (string): Analysis period end date (YYYY-MM-DD format)
- `min_volume_usd` (integer): Minimum total trading volume in USD
- `min_trade_count` (integer): Minimum number of trades required

**Output Schema**:
```
trader_address: string         # Ethereum address of the wallet
dex_platform: string          # Always "Uniswap"
trade_count: integer          # Total number of trades
total_volume_usd: decimal     # Total trading volume in USD
avg_trade_size_usd: decimal   # Average trade size
unique_tokens_traded: integer # Number of different tokens traded
first_trade_time: timestamp   # First trade in analysis period
last_trade_time: timestamp    # Last trade in analysis period
volume_per_trade: decimal     # Volume efficiency metric
volume_per_gas_dollar: decimal # Gas efficiency metric
quality_score: decimal        # Composite quality score (0-1)
```

**Example Usage**:
```sql
-- 30-day high-volume analysis
start_date = '2024-08-27'
end_date = '2024-09-27'
min_volume_usd = 10000
min_trade_count = 10
```

#### `smart_wallet_curve_volume.sql`
**Purpose**: Identifies high-volume wallets on Curve Finance
**Expected Results**: 3,000-6,000 wallets
**Credit Cost**: 100-200 credits

**Additional Output Fields**:
```
stablecoin_volume_usd: decimal    # Volume in stablecoin pools
crypto_volume_usd: decimal        # Volume in crypto pools
stablecoin_volume_ratio: decimal  # Ratio of stablecoin to total volume
unique_pools_used: integer        # Number of different Curve pools used
```

#### `smart_wallet_combined_dex.sql`
**Purpose**: Unified analysis across multiple DEXs with cross-platform metrics
**Expected Results**: 8,000-12,000 unique wallets
**Credit Cost**: 200-400 credits

**Additional Output Fields**:
```
dex_usage_pattern: string         # "Multi-DEX", "Uniswap-Only", "Curve-Only"
trader_profile: string            # "Sophisticated Multi-DEX", "High Volume Specialist", etc.
uniswap_volume_share: decimal     # Percentage of volume on Uniswap
combined_quality_score: decimal   # Enhanced quality score for multi-DEX users
```

### Bot Detection Queries

#### `bot_detection_patterns.sql`
**Purpose**: Identifies automated trading patterns and bots for exclusion
**Parameters**:
- Standard date range parameters
- `min_detection_confidence` (integer): Minimum confidence level (1-5)

**Output Schema**:
```
address: string                   # Address being evaluated
contract_type: string            # Type if known contract
bot_confidence_score: decimal    # Confidence this is a bot (0-1)
bot_classification: string       # Classification category
max_daily_trades: integer        # Maximum trades in a single day
sandwich_count: integer          # Number of sandwich attack patterns
flash_loan_count: integer        # Number of flash loan uses
```

#### `mev_detection_advanced.sql`
**Purpose**: Advanced MEV bot detection using transaction timing and gas patterns
**Parameters**:
- Standard date range parameters
- `block_range_limit` (integer): Maximum block range for pattern analysis

**Output Schema**:
```
trader_address: string            # Address being analyzed
frontrun_instances: integer       # Number of front-running patterns
backrun_instances: integer        # Number of back-running patterns
mev_sophistication_score: decimal # MEV sophistication (0-1)
mev_classification: string        # Type of MEV activity
exclusion_recommendation: string  # "High Risk - Exclude", etc.
```

### Performance Analysis

#### `wallet_performance_metrics.sql`
**Purpose**: Calculates comprehensive performance and efficiency metrics
**Credit Cost**: 180-300 credits

**Output Schema**:
```
trader_address: string            # Wallet address
total_volume_usd: decimal        # Total trading volume
trades_per_active_day: decimal   # Trading frequency metric
volume_per_gas_dollar: decimal   # Gas efficiency
total_unique_tokens: integer     # Token diversity
performance_quality_score: decimal # Composite performance score
trading_period_days: integer     # Length of trading activity
```

### Testing and Validation

#### `query_testing_suite.sql`
**Purpose**: Validates query logic and performance with test data
**Parameters**:
- `test_mode` (string): "preview_7day", "validation", "full_test"
- `preview_start_date` (string): Test period start
- `preview_end_date` (string): Test period end

#### `exclusion_validation.sql`
**Purpose**: Tests effectiveness of bot detection against known addresses

## Parameter Presets

### Standard Analysis Windows

#### 30-Day Analysis (`30_day_analysis`)
```json
{
  "start_date": "2024-08-27",
  "end_date": "2024-09-27",
  "min_volume_usd": 10000,
  "min_trade_count": 10,
  "analysis_window": "30 days"
}
```

#### 90-Day Analysis (`90_day_analysis`)
```json
{
  "start_date": "2024-06-27",
  "end_date": "2024-09-27",
  "min_volume_usd": 25000,
  "min_trade_count": 15,
  "analysis_window": "90 days"
}
```

#### Preview Testing (`preview_7day`)
```json
{
  "start_date": "2024-09-20",
  "end_date": "2024-09-27",
  "min_volume_usd": 5000,
  "min_trade_count": 5,
  "purpose": "testing"
}
```

## Execution Workflow

### 1. Preview Testing
Before running full analysis, always test with 7-day preview:

```python
# Test query logic with small dataset
preview_results = await query_manager.run_preview_test(QueryType.UNISWAP_VOLUME)
```

### 2. Bot Detection
Run bot detection to build exclusion lists:

```python
# Identify bots and contracts to exclude
bot_results = await query_manager.execute_query(
    QueryType.BOT_DETECTION,
    parameters=parameters_90day
)
```

### 3. Core Analysis
Execute main wallet identification queries:

```python
# Run main analysis pipeline
analysis_results = await query_manager.run_full_analysis("90_day_analysis")
```

### 4. Performance Metrics
Calculate detailed performance metrics for qualified wallets:

```python
# Get performance metrics for identified wallets
metrics_results = await query_manager.execute_query(
    QueryType.PERFORMANCE_METRICS,
    parameters=parameters_90day
)
```

## Quality Assurance

### Expected Result Validation

**Volume Distribution**:
- 80% of wallets should have volumes between min_threshold and 10x min_threshold
- Top 5% should represent 30-50% of total volume
- Median wallet should have 15-25 trades

**Bot Exclusion Effectiveness**:
- Should exclude >95% of known router contracts
- Should exclude >90% of known MEV bots
- False positive rate should be <5%

**Performance Consistency**:
- 90% of results should have quality_score > 0.3
- 50% should show activity across multiple days
- Token diversity should average 3-8 unique tokens

### Common Issues and Solutions

#### Query Timeouts
**Symptoms**: Execution time >600 seconds
**Solutions**:
- Reduce date range to smaller windows
- Increase minimum volume thresholds
- Use chunked execution strategy

#### High Credit Usage
**Symptoms**: >400 credits per query
**Solutions**:
- Use preview mode for development
- Cache results aggressively
- Optimize WHERE clauses for early filtering

#### Low Result Counts
**Symptoms**: <1000 wallets returned
**Solutions**:
- Lower minimum thresholds
- Check date range validity
- Verify DEX activity in period

#### High Bot Detection
**Symptoms**: >50% of volume from excluded addresses
**Solutions**:
- Review exclusion criteria
- Adjust bot detection thresholds
- Manual review of high-volume exclusions

## Cache Management

### Cache Structure
```
cache/smart_wallets/
├── uniswap_volume_abc123.parquet
├── uniswap_volume_abc123_metadata.json
├── curve_volume_def456.parquet
└── curve_volume_def456_metadata.json
```

### Cache Policies
- **Default TTL**: 24 hours
- **Preview Queries**: 6 hours
- **Bot Detection**: 48 hours (more stable)
- **Performance Metrics**: 12 hours

### Cache Invalidation
Cache is invalidated when:
- TTL expires
- Parameters change
- Query logic is updated
- Manual cache clear

## Error Handling

### Retryable Errors
- Query timeouts → Retry with shorter time windows
- Rate limits → Exponential backoff
- Network errors → Immediate retry
- Server errors → Wait and retry

### Fatal Errors
- SQL syntax errors → Fix query logic
- Authentication failures → Check API keys
- Credit exhaustion → Wait for credit refresh

### Fallback Strategies
1. **Cache Fallback**: Use cached results if available
2. **Parameter Adjustment**: Reduce scope for timeouts
3. **Chunked Execution**: Split large queries into smaller pieces

## Performance Optimization

### Query Optimization
- Always filter by `evt_block_time` first
- Use indexed columns in WHERE clauses
- Minimize JOIN complexity
- Aggregate data efficiently

### Credit Conservation
- Use narrow time windows for testing
- Cache all successful executions
- Reuse intermediate results
- Monitor credit usage actively

### Execution Time Optimization
- Target <60 seconds for production queries
- Use preview runs for development
- Optimize JOIN order and conditions
- Monitor execution plans

## Integration Points

### Data Pipeline Integration
The query results integrate with:
- Wallet processing pipeline (`wallet_processor.py`)
- Portfolio analysis system
- Risk assessment modules
- Narrative classification system

### API Integration
Results are exposed through:
- RESTful API endpoints
- Real-time WebSocket feeds
- Batch processing jobs
- Analytics dashboards

## Monitoring and Alerting

### Key Metrics to Monitor
- Query success rate (target: >95%)
- Average execution time (target: <60s)
- Credit usage rate
- Cache hit rate (target: >80%)
- Result count consistency

### Alert Conditions
- Query failure rate >10%
- Execution time >300 seconds
- Credit usage >500/day
- Result count deviation >50%
- Cache miss rate >50%