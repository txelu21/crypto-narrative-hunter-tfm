# Dune Analytics SQL Queries

This folder contains SQL queries designed for Dune Analytics (Trino SQL engine). All queries are operational and uploaded to Dune platform.

## Query Inventory

### Liquidity Analysis (3 queries)
| Query | Purpose | Query ID | Status |
|-------|---------|----------|--------|
| `uniswap_v2_pools.sql` | Uniswap V2 pool discovery & TVL | 5874506 | ✅ |
| `uniswap_v3_pools.sql` | Uniswap V3 pool discovery & TVL | 5874518 | ✅ |
| `curve_pools.sql` | Curve pool discovery & TVL | 5874531 | ✅ |

### Wallet Discovery (5 queries)
| Query | Purpose | Query ID | Status |
|-------|---------|----------|--------|
| `smart_wallet_uniswap_volume.sql` | High-volume Uniswap traders | 5875125 | ✅ |
| `smart_wallet_curve_volume.sql` | High-volume Curve traders | 5875138 | ✅ |
| `smart_wallet_combined_dex.sql` | Multi-DEX wallet analysis | 5875152 | ✅ |
| `bot_detection_patterns.sql` | Automated trading detection | 5875168 | ✅ |
| `wallet_performance_metrics.sql` | Trading performance metrics | 5878574 | ✅ |

### Transaction Collection (2 queries)
| Query | Purpose | Query ID | Status |
|-------|---------|----------|--------|
| `wallet_transactions.sql` | Single wallet transaction history | 5896947 | ✅ |
| `wallet_transactions_batch.sql` | Batched wallet transactions | 5897009 | ✅ |

### Validation & Utilities (3 queries)
| Query | Purpose | Query ID | Status |
|-------|---------|----------|--------|
| `query_testing_suite.sql` | Query validation & testing | 5878595 | ✅ |
| `exclusion_validation.sql` | Bot exclusion validation | 5878612 | ✅ |
| `token_filtering_helper.sql` | Token filtering utilities | 5878623 | ✅ |
| `mev_detection_advanced.sql` | Advanced MEV bot detection | 5875182 | ✅ |

## Usage

### Via Python CLI
```bash
# Liquidity analysis
python cli_liquidity.py

# Wallet discovery
python services/smart_wallet_query_manager.py

# Transaction collection
python cli_transactions_dune.py
```

### Via Dune Platform
Access queries directly at: `https://dune.com/queries/{QUERY_ID}`

## SQL Compatibility

All queries are written for **Trino SQL** (Dune's query engine). Key differences from standard PostgreSQL:

- String functions: Use `SUBSTRING()` not `SUBSTR()`
- Type casting: Use `CAST(x AS TYPE)` not `x::TYPE`
- Hex conversion: Use `FROM_HEX()` not `DECODE()`
- Window functions: Proper `OVER()` clause required
- Date functions: Use `EXTRACT()` not `DATE_PART()`

For detailed SQL compatibility guide, see: `../../docs/reference/TRINO_SQL_GUIDE.md`

## Query Parameters

### Standard Parameters
- `start_date` (string): YYYY-MM-DD format
- `end_date` (string): YYYY-MM-DD format
- `min_volume_usd` (integer): Minimum trading volume
- `min_trade_count` (integer): Minimum number of trades

### Batch Parameters
- `batch_size` (integer): Wallets per batch (default: 1000)
- `batch_offset` (integer): Starting position (default: 0)

## Documentation

- **Detailed Query Reference**: `../../docs/reference/DUNE_QUERIES.md`
- **Query Optimization**: `../../docs/reference/QUERY_OPTIMIZATION.md`
- **Operational Guide**: `../../docs/OPERATIONAL_GUIDE.md`

## Maintenance

**Last Updated:** October 4, 2025
**All Queries Status:** ✅ Operational
**Maintained By:** Product Owner

---

For query modification history and SQL compatibility fixes, see: `../../docs/archive/dune-fixes/`
