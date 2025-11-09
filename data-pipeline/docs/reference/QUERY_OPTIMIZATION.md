# Smart Wallet Query Optimization Guide

## Performance Optimization Strategies

### 1. Index Usage
- **Always filter by `evt_block_time`** first - this column is indexed
- **Use `trader_address` filters** early in WHERE clauses
- **Avoid functions on indexed columns** in WHERE clauses

### 2. Query Structure Optimization

#### Early Filtering
```sql
-- Good: Filter by time first
WHERE s.evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
  AND s.evt_block_time <= CAST('{{end_date}}' AS TIMESTAMP)
  AND s.amount0_in > 0

-- Bad: Complex calculations before time filtering
WHERE (complex_calculation) > threshold
  AND s.evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
```

#### Efficient JOINs
```sql
-- Good: Join on indexed columns
INNER JOIN uniswap_v2.Factory_evt_PairCreated p
    ON p.pair_address = s.pair

-- Good: Use LEFT JOIN for optional data
LEFT JOIN prices.usd price0 ON price0.contract_address = p.token0
    AND price0.minute = date_trunc('minute', s.evt_block_time)
```

### 3. Aggregation Optimization

#### Group By Efficiency
```sql
-- Efficient grouping order
GROUP BY
    trader_address,      -- High cardinality first
    DATE(block_time),    -- Then time-based
    pool_address         -- Then other dimensions
```

#### Selective Aggregation
```sql
-- Use CASE statements for conditional aggregation
SUM(CASE WHEN condition THEN value ELSE 0 END) as conditional_sum
```

### 4. Credit Conservation Techniques

#### Preview Queries
- Use 7-day windows for testing: `start_date='2024-09-20'`
- Limit result sets: `LIMIT 1000` for validation
- Cache successful executions using query parameter hashing

#### Data Scanning Reduction
```sql
-- Minimize scanned rows with targeted filters
WHERE s.evt_block_time >= CAST('{{start_date}}' AS TIMESTAMP)
  AND s.evt_block_time <= CAST('{{end_date}}' AS TIMESTAMP)
  AND (s.amount0_in > 0 OR s.amount1_in > 0)  -- Exclude empty trades early
```

### 5. Memory Management

#### Avoid Large Intermediate Results
```sql
-- Good: Filter before union
(SELECT ... FROM table1 WHERE conditions)
UNION ALL
(SELECT ... FROM table2 WHERE conditions)

-- Bad: Filter after union
SELECT ... FROM (
    SELECT ... FROM table1
    UNION ALL
    SELECT ... FROM table2
) WHERE conditions
```

#### Use CTEs Strategically
- Break complex queries into readable CTEs
- Ensure each CTE filters data appropriately
- Avoid deeply nested CTEs (max 3-4 levels)

### 6. Bot Detection Optimization

#### Efficient Pattern Detection
```sql
-- Use window functions for pattern analysis
ROW_NUMBER() OVER (
    PARTITION BY trader_address, DATE(block_time)
    ORDER BY block_time
) as trade_sequence
```

#### Pre-computed Exclusions
```sql
-- Static exclusion list (computed once)
excluded_contracts AS (
    SELECT DISTINCT contract_address as address
    FROM (VALUES (...)) as contracts(contract_address)
)
```

## Query Execution Monitoring

### Performance Metrics to Track
1. **Execution Time**: Target <60 seconds for production queries
2. **Credit Usage**: Monitor via Dune dashboard
3. **Result Set Size**: 8k-12k wallets for full analysis
4. **Cache Hit Rate**: >80% during development

### Bottleneck Identification
- Large JOINs on non-indexed columns
- Complex calculations in WHERE clauses
- Excessive data scanning in time ranges
- Memory-intensive aggregations

### Query Plan Analysis
```sql
-- Add EXPLAIN for query plan analysis (when supported)
EXPLAIN (ANALYZE, BUFFERS)
SELECT ...
```

## Error Prevention

### Common Query Failures
1. **Timeout Errors**: Reduce time window or add more filters
2. **Memory Errors**: Break into smaller CTEs or reduce JOIN complexity
3. **Syntax Errors**: Validate parameter substitution
4. **Data Type Mismatches**: Ensure proper CAST operations

### Recovery Strategies
1. **Query Chunking**: Split large time ranges into smaller windows
2. **Incremental Processing**: Process data in daily/weekly chunks
3. **Fallback Caching**: Use cached results when fresh execution fails
4. **Parameter Adjustment**: Reduce thresholds for problematic periods

## Best Practices Summary

1. **Start with narrow time windows** (7 days) for testing
2. **Filter early and often** using indexed columns
3. **Monitor execution time and credit usage**
4. **Cache successful results** with parameter hashing
5. **Use preview queries** before full execution
6. **Validate results** against known benchmarks
7. **Document performance characteristics** for each query
8. **Implement graceful error handling** with retries