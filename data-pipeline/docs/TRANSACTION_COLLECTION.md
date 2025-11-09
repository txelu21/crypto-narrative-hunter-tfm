# Transaction Collection Guide

**Last Updated:** October 4, 2025
**Status:** ⚠️ PAUSED - Dune Credits Exhausted at 14% Completion

---

## Current Status

### Progress Summary
- ✅ **Wallet Dataset**: 30,547 smart money wallets identified
- ⚠️ **Transaction Collection**: 14% complete (~4,300 wallets)
- ❌ **Blocker**: Dune Analytics credits exhausted
- 📊 **Estimated Remaining**: ~22,500 transactions needed

### What's Working
- ✅ Dune batch query operational (`wallet_transactions_batch.sql`)
- ✅ CLI orchestrator (`cli_transactions_dune.py`) with checkpointing
- ✅ Database schema ready for transaction ingestion
- ✅ Automatic resume capability from checkpoints

---

## Architecture Overview

### Recommended Approach: Dune Analytics (Current)

**Why Dune:**
- ✅ Proven to work (5/5 data collection phases succeeded with Dune)
- ✅ ~200x faster than Alchemy RPC for large datasets
- ✅ No rate limiting issues
- ✅ Built-in MEV/bot detection via pre-indexed tables

**Query Design:**
- Batched processing: 1,000 wallets per batch
- Uses `dex.trades` table for performance
- ROW_NUMBER() pagination (Trino SQL compatible)
- Automatic checkpointing for safe resumption

---

## Batch Collection Strategy

### Query Parameters
```python
{
    "start_date": "2024-01-01",
    "end_date": "2024-09-30",
    "batch_size": 1000,      # Wallets per batch
    "batch_offset": 0        # Starting position (auto-incremented)
}
```

### Execution Flow
1. **Batch 1**: Wallets 1-1,000 (✅ Complete)
2. **Batch 2**: Wallets 1,001-2,000 (✅ Complete)
3. **Batch 3**: Wallets 2,001-3,000 (✅ Complete)
4. **Batch 4**: Wallets 3,001-4,000 (✅ Complete)
5. **Batch 5**: Wallets 4,001-5,000 (❌ Paused - credits exhausted)
6. **Batches 6-31**: Wallets 5,001-30,547 (⏳ Pending)

### Checkpoint System
```sql
-- Current checkpoint stored in collection_checkpoints table
collection_type: 'transaction_extraction'
status: 'in_progress'
last_processed_batch: 4
records_collected: ~14% of total
```

---

## How to Resume Collection

### Prerequisites
1. **Dune Credits**: Acquire additional credits (~880 credits needed)
   - Option A: Purchase credits at https://dune.com/settings/api
   - Option B: Wait for monthly credit refresh
2. **Database**: Ensure PostgreSQL is running
3. **Environment**: `.env` file with `DUNE_API_KEY`

### Resume Command
```bash
cd BMAD_TFM/data-collection

# CLI automatically resumes from last checkpoint
python cli_transactions_dune.py

# Or specify starting batch manually
python cli_transactions_dune.py --start-batch 5
```

### Expected Timeline
- **Per Batch**: ~2-3 minutes (query + insertion)
- **Remaining Batches**: 27 batches × 3 min = ~81 minutes
- **Total to Completion**: ~1.5 hours

### Cost Estimate
- **Credits per Batch**: ~30-35 credits
- **Total Remaining**: 27 batches × 35 = ~945 credits
- **Approximate Cost**: ~$95 (at Dune's standard pricing)

---

## Query Details

### SQL File: `sql/dune_queries/wallet_transactions_batch.sql`

**Data Extracted:**
- `tx_hash`: Transaction identifier
- `block_time`: Timestamp of transaction
- `block_number`: Block height
- `wallet_address`: Trader wallet
- `token_bought_address`, `token_sold_address`: Token contracts
- `amount_usd`: USD value of trade
- `gas_price`, `gas_used`: Transaction costs
- `project`: DEX platform (Uniswap, Curve, etc.)

**Filters Applied:**
- Date range: 2024-01-01 to 2024-09-30
- Only wallets in validated dataset
- Successful transactions only
- DEX swaps only (no transfers, mints, burns)

---

## Alternative Approach: Alchemy RPC

### Status: ❌ Not Implemented

**Why Not Used:**
The RPC-based collection infrastructure exists but is incomplete:
- ✅ Client wrappers exist (`alchemy_client.py`, `uniswap_client.py`)
- ❌ Integration methods not implemented
- ❌ Would be 200x slower than Dune
- ❌ Subject to rate limiting

**If Required:**
Would need ~20 hours of development + 12-24 hours execution time.

---

## Data Quality Targets

### Completeness
- **Target**: >95% of wallet transactions captured
- **Current**: 14% complete
- **Validation**: Cross-check with Etherscan spot-checks

### Accuracy
- **Target**: >99% transaction decoding accuracy
- **Method**: Dune's pre-indexed `dex.trades` table
- **Verification**: Sample validation against raw logs

### Performance Metrics
- **Query Time**: 30-60 seconds per batch
- **Insertion Time**: 10-20 seconds per batch
- **Total Throughput**: ~300-400 wallets/minute

---

## Troubleshooting

### Issue: Query Timeout
**Symptoms**: Batch execution >300 seconds
**Solution**: Reduce `batch_size` from 1000 to 500

### Issue: Duplicate Transactions
**Symptoms**: Primary key violations on `tx_hash`
**Solution**: CLI includes automatic deduplication, but check checkpoint accuracy

### Issue: Missing Wallets
**Symptoms**: Wallet count mismatch
**Solution**: Verify `dune.tokencluster_team_9042.dataset_wallet_addresses` upload

### Issue: Checkpoint Corruption
**Symptoms**: Resume starts from wrong batch
**Solution**:
```sql
-- Reset checkpoint manually
UPDATE collection_checkpoints
SET last_processed_batch = 4,
    status = 'in_progress'
WHERE collection_type = 'transaction_extraction';
```

---

## Next Steps

### Immediate (User Action Required)
1. ⚠️ **Acquire Dune Credits**: ~945 credits needed
2. ✅ **Verify Dataset**: Wallet addresses uploaded to Dune

### After Credit Acquisition
1. Run: `python cli_transactions_dune.py`
2. Monitor progress: `tail -f logs/collection_*.log`
3. Validate completion: `SELECT COUNT(*) FROM transactions;`

### Post-Collection
1. Run data validation: `python services/validation/quality_reporter.py`
2. Calculate performance metrics: Story 2.2
3. Collect balance snapshots: Story 3.2
4. Final export: Story 3.5

---

## References

- **Main Documentation**: `docs/OPERATIONAL_GUIDE.md`
- **Execution Status**: `docs/EXECUTION_SUMMARY.md`
- **Database Schema**: `outputs/documentation/DATA_DICTIONARY.md`
- **Dune Query**: https://dune.com/queries/5897009

---

**Document Version**: 1.0
**Maintained By**: PO Agent (Sarah)
