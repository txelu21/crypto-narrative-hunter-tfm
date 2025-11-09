# Dune Query Trino/Presto SQL Compatibility Guide
**Generated:** 2025-09-30
**Critical:** Dune uses Trino/Presto, NOT PostgreSQL

---

## 🚨 Critical Differences

### 1. DISTINCT ON - NOT SUPPORTED ❌

**PostgreSQL (Wrong):**
```sql
SELECT DISTINCT ON (contract_address)
    contract_address,
    price,
    minute
FROM prices.usd
ORDER BY contract_address, minute DESC
```

**Trino/Presto (Correct):**
```sql
-- Method 1: ROW_NUMBER() with separate CTE
WITH prices_ranked AS (
    SELECT
        contract_address,
        price,
        minute,
        ROW_NUMBER() OVER (PARTITION BY contract_address ORDER BY minute DESC) as rn
    FROM prices.usd
)
SELECT
    contract_address,
    price,
    minute
FROM prices_ranked
WHERE rn = 1

-- Method 2: Inline in single query (less readable but works)
SELECT * FROM (
    SELECT
        contract_address,
        price,
        minute,
        ROW_NUMBER() OVER (PARTITION BY contract_address ORDER BY minute DESC) as rn
    FROM prices.usd
) WHERE rn = 1
```

**Why:** `DISTINCT ON` is PostgreSQL-specific. Trino requires window functions.

---

### 2. Parameter Handling with Type Safety ✅

**Basic (Fragile):**
```sql
WHERE evt_block_time >= CAST('{{start_date}}' AS DATE)
```

**Production (Robust):**
```sql
WHERE evt_block_time >= COALESCE(
    TRY_CAST('{{start_date}}' AS DATE),
    DATE('2023-01-01')  -- Fallback default
)
```

**Why:**
- `TRY_CAST` returns NULL on failure instead of error
- `COALESCE` provides fallback when parameter is missing/invalid
- Prevents query breakage from bad parameters

**Parameter Setup in Dune:**
- Parameter name: `start_date`
- Type: `date`
- Default value: `2023-01-01`

---

### 3. Bytea Address Format - NOT SUPPORTED ❌

**PostgreSQL (Wrong):**
```sql
SELECT '\xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'::bytea
WHERE contract_address = '\xdAC17F958D2ee523a2206206994597C13D831ec7'::bytea
```

**Trino/Presto (Correct):**
```sql
SELECT 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
WHERE contract_address = 0xdAC17F958D2ee523a2206206994597C13D831ec7
```

**Why:** Trino uses VARBINARY type natively, no casting needed.

---

### 4. Table Schema Names 📋

**Dune Format:** `{protocol}_{version}_{blockchain}.{table}`

**Examples:**

| Wrong | Right |
|-------|-------|
| `uniswap_v2.Factory` | `uniswap_v2_ethereum.Factory_evt_PairCreated` |
| `uniswap_v3.Pool` | `uniswap_v3_ethereum.Pool_evt_Swap` |
| `curve.StableSwap` | `curvefi_ethereum.StableSwap_evt_AddLiquidity` |
| `tokens.erc20` | `tokens.erc20_ethereum` |
| `prices.usd` | `prices.usd` (+ `WHERE blockchain = 'ethereum'`) |

**Event Table Naming:**
- Format: `{Contract}_{evt/call}_{EventName}`
- Example: `Factory_evt_PairCreated`, `Pair_evt_Sync`, `Pool_evt_Swap`

---

### 5. Type Conversions 🔄

**Function Names:**

| PostgreSQL | Trino/Presto |
|------------|--------------|
| `POW(x, y)` | `POWER(x, y)` |
| `RANDOM()` | `RAND()` |
| `NOW()` | `NOW()` or `CURRENT_TIMESTAMP` |

**Explicit Casting:**
```sql
-- Always use explicit casts for calculations
CAST(reserve0 AS DOUBLE)
CAST(amount AS BIGINT)
CAST(price AS DECIMAL(18, 6))
```

---

### 6. Date/Time Functions 📅

**Trino Functions:**
```sql
-- Current time
NOW()
CURRENT_TIMESTAMP

-- Date manipulation
DATE('2024-01-01')
TIMESTAMP '2024-01-01 12:00:00'

-- Intervals
NOW() - INTERVAL '7' day
NOW() - INTERVAL '24' hour
DATE '2024-01-01' + INTERVAL '1' day

-- Truncation
DATE_TRUNC('hour', evt_block_time)
DATE_TRUNC('day', evt_block_time)

-- Extraction
EXTRACT(HOUR FROM evt_block_time)
EXTRACT(DAY FROM evt_block_time)
DATE(evt_block_time)
```

---

### 7. Array/List Operations 📦

**PostgreSQL UNNEST (Wrong):**
```sql
SELECT unnest(ARRAY[{{token_addresses}}]) as token_address
```

**Trino UNNEST (Correct):**
```sql
-- If parameter is array type
SELECT address as token_address
FROM UNNEST({{token_addresses}}) as t(address)

-- Or simpler with IN clause
WHERE contract_address IN ({{token_addresses}})
```

---

### 8. String Operations 📝

**Case Sensitivity:**
```sql
-- Trino is case-sensitive for identifiers
-- Use exact column names from schema

-- Column names with reserved words need quotes
SELECT "from", "to", "value"
FROM ethereum.transactions
```

**String Functions:**
```sql
LOWER(address)
UPPER(symbol)
CONCAT(a, b, c)
LENGTH(string)
SUBSTR(string, start, length)
```

---

### 9. NULL Handling 🔍

**Safe Division:**
```sql
-- Always protect against division by zero
value / NULLIF(divisor, 0)

-- Or with COALESCE
COALESCE(value / NULLIF(divisor, 0), 0)
```

**NULL Comparisons:**
```sql
-- Use proper NULL checks
WHERE value IS NOT NULL
WHERE value IS NULL

-- NOT these:
WHERE value != NULL  -- Wrong! Always false
WHERE value <> NULL  -- Wrong! Always false
```

---

### 10. Window Functions 📊

**Correct Usage:**
```sql
-- Row number for deduplication
ROW_NUMBER() OVER (
    PARTITION BY contract_address
    ORDER BY evt_block_time DESC
) as rn

-- Then filter
WHERE rn = 1

-- Ranking
RANK() OVER (ORDER BY tvl DESC) as rank
DENSE_RANK() OVER (ORDER BY volume DESC) as dense_rank

-- Aggregates
SUM(amount) OVER (PARTITION BY trader ORDER BY block_time) as cumulative_total
AVG(price) OVER (ORDER BY timestamp ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) as moving_avg
```

---

## 📋 Complete Query Template

```sql
-- FIXED: Query Name
-- Query ID: XXXXXXX
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date)
-- Dune uses Trino/Presto, not PostgreSQL

WITH data_ranked AS (
    -- Get data with row numbers for deduplication
    SELECT
        contract_address,
        evt_block_time,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY contract_address
            ORDER BY evt_block_time DESC
        ) as rn
    FROM protocol_ethereum.Contract_evt_Event
    WHERE evt_block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND evt_block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND contract_address != 0x0000000000000000000000000000000000000000
),

latest_data AS (
    -- Get only latest record per address
    SELECT
        contract_address,
        evt_block_time,
        amount
    FROM data_ranked
    WHERE rn = 1
),

price_data AS (
    -- Get token prices
    SELECT
        contract_address,
        price as price_usd,
        ROW_NUMBER() OVER (
            PARTITION BY contract_address
            ORDER BY minute DESC
        ) as price_rn
    FROM prices.usd
    WHERE blockchain = 'ethereum'
      AND minute >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
)

-- Final result
SELECT
    ld.contract_address,
    ld.evt_block_time,
    CAST(ld.amount AS DOUBLE) / POWER(10, 18) as amount_normalized,
    pd.price_usd,
    (CAST(ld.amount AS DOUBLE) / POWER(10, 18)) * COALESCE(pd.price_usd, 0) as value_usd
FROM latest_data ld
LEFT JOIN price_data pd
    ON pd.contract_address = ld.contract_address
    AND pd.price_rn = 1
WHERE COALESCE(pd.price_usd, 0) > 0  -- Only records with valid prices
ORDER BY value_usd DESC
LIMIT 1000;
```

---

## ✅ Testing Checklist

Before uploading query to Dune:

- [ ] No `DISTINCT ON` usage - replaced with ROW_NUMBER()
- [ ] All CAST operations wrapped in TRY_CAST + COALESCE
- [ ] No `::bytea` casts - plain 0x addresses
- [ ] Table names use `protocol_blockchain.*` format
- [ ] All parameters have defaults in COALESCE
- [ ] POWER() instead of POW()
- [ ] Explicit CAST to DOUBLE/BIGINT for calculations
- [ ] NULL-safe division with NULLIF()
- [ ] LIMIT clauses for safety
- [ ] Proper window function syntax

---

## 🐛 Common Errors & Fixes

### Error: "Function 'distinct_on' not found"
**Fix:** Replace with ROW_NUMBER() pattern (see #1 above)

### Error: "Cannot cast 'varchar' to 'date'"
**Fix:** Wrap in TRY_CAST + COALESCE with default

### Error: "Table not found: uniswap_v2.Factory"
**Fix:** Change to `uniswap_v2_ethereum.Factory_evt_PairCreated`

### Error: "Column 'contract_address' is ambiguous"
**Fix:** Add table aliases: `t1.contract_address`, `t2.contract_address`

### Error: "Syntax error at or near '::'"
**Fix:** Remove PostgreSQL-style casts (`::type`), use `CAST(x AS type)` or plain addresses

---

## 📚 Resources

**Dune Documentation:**
- https://dune.com/docs/query/
- https://dune.com/docs/data-tables/

**Trino SQL Reference:**
- https://trino.io/docs/current/functions.html
- https://trino.io/docs/current/sql.html

**Key Differences from PostgreSQL:**
- No DISTINCT ON
- Different type system
- Stricter NULL handling
- Case-sensitive identifiers

---

## 🎯 Summary

**Always Remember:**
1. Dune = Trino/Presto, NOT PostgreSQL
2. Use ROW_NUMBER() instead of DISTINCT ON
3. Wrap parameters in TRY_CAST + COALESCE
4. Use correct table schema format
5. Test with small date ranges first

**Apply These Patterns:**
- Every query needs TRY_CAST + COALESCE for parameters
- Every deduplication needs ROW_NUMBER() + WHERE rn = 1
- Every calculation needs explicit CAST to DOUBLE
- Every division needs NULLIF() protection

---

**Ready to create production-quality Dune queries!** 🚀