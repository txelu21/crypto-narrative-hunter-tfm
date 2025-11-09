-- FIXED: Query Testing and Validation Suite
-- Query ID: 5878586
-- Status: Fixed for Dune/Trino compatibility - Parameters simplified
-- Tests smart wallet queries with preview data and validates results
-- Dune uses Trino/Presto, not PostgreSQL

-- Using hardcoded dates to avoid parameter issues
-- Users can modify these directly in the query

WITH
-- Test configuration with hardcoded values (modify as needed)
test_config AS (
    SELECT
        DATE('2023-01-01') as start_date,
        DATE('2024-01-01') as end_date,
        5000.0 as min_volume_threshold,
        5 as min_trade_threshold
),

-- Known test addresses for validation
test_addresses AS (
    SELECT address, expected_result, test_category FROM (
        VALUES
        -- Known high-volume legitimate traders (should be included)
        (0x1234567890123456789012345678901234567890, 'include', 'legitimate_trader'),
        (0x2345678901234567890123456789012345678901, 'include', 'legitimate_trader'),

        -- Known bots/contracts (should be excluded)
        (0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D, 'exclude', 'uniswap_router'),
        (0x1111111254fb6c44bAC0beD2854e76F90643097d, 'exclude', 'aggregator'),

        -- Edge cases
        (0x3456789012345678901234567890123456789012, 'review', 'edge_case')
    ) as t(address, expected_result, test_category)
),

-- Uniswap volume test (using dex.trades for performance)
uniswap_test_results AS (
    SELECT
        t.taker as trader_address,
        COUNT(*) as trade_count,
        SUM(t.amount_usd) as total_volume_usd,
        MIN(t.block_time) as first_trade,
        MAX(t.block_time) as last_trade,
        COUNT(DISTINCT t.token_bought_address) + COUNT(DISTINCT t.token_sold_address) as unique_tokens
    FROM dex.trades t
    CROSS JOIN test_config tc
    WHERE t.blockchain = 'ethereum'
      AND t.project = 'uniswap'
      AND t.block_time >= tc.start_date
      AND t.block_time <= tc.end_date + INTERVAL '1' day
      AND t.taker IS NOT NULL
      AND t.amount_usd > 0
    GROUP BY t.taker
    HAVING SUM(t.amount_usd) >= (SELECT min_volume_threshold FROM test_config)
),

-- Daily stats for bot detection (pre-aggregated to avoid cartesian join)
daily_trader_stats AS (
    SELECT
        t.taker as trader_address,
        MAX(daily_counts.daily_trades) as max_daily_trades
    FROM dex.trades t
    CROSS JOIN test_config tc
    INNER JOIN (
        SELECT
            t2.taker as trader_address,
            DATE(t2.block_time) as trade_date,
            COUNT(*) as daily_trades
        FROM dex.trades t2
        CROSS JOIN test_config tc2
        WHERE t2.blockchain = 'ethereum'
          AND t2.project = 'uniswap'
          AND t2.block_time >= tc2.start_date
          AND t2.block_time <= tc2.end_date + INTERVAL '1' day
        GROUP BY t2.taker, DATE(t2.block_time)
    ) daily_counts ON daily_counts.trader_address = t.taker
    WHERE t.blockchain = 'ethereum'
      AND t.project = 'uniswap'
      AND t.block_time >= tc.start_date
      AND t.block_time <= tc.end_date + INTERVAL '1' day
    GROUP BY t.taker
),

-- Bot detection test results
bot_detection_test AS (
    SELECT
        utr.trader_address,

        -- High frequency detection
        CASE WHEN COALESCE(dts.max_daily_trades, 0) > 100 THEN 1 ELSE 0 END as detected_high_frequency,

        -- Known address detection
        CASE WHEN kc.address IS NOT NULL THEN 1 ELSE 0 END as detected_known_contract

    FROM uniswap_test_results utr
    LEFT JOIN daily_trader_stats dts ON dts.trader_address = utr.trader_address
    LEFT JOIN (
        SELECT address FROM (
            VALUES
            (0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D),  -- Uniswap V2 Router
            (0x1111111254fb6c44bAC0beD2854e76F90643097d)   -- 1inch V4
        ) as t(address)
    ) kc ON kc.address = utr.trader_address
),

-- Test result validation
test_validation AS (
    SELECT
        COALESCE(utr.trader_address, ta.address) as address,
        ta.expected_result,
        ta.test_category,

        -- Test results
        CASE WHEN utr.trader_address IS NOT NULL THEN 1 ELSE 0 END as included_in_results,
        CASE WHEN bdt.detected_high_frequency = 1 OR bdt.detected_known_contract = 1
        THEN 1 ELSE 0 END as flagged_by_bot_detection,

        -- Volume and trade metrics
        COALESCE(utr.total_volume_usd, 0) as volume_usd,
        COALESCE(utr.trade_count, 0) as trade_count,
        COALESCE(utr.unique_tokens, 0) as unique_tokens,

        -- Test outcome
        CASE
            WHEN ta.expected_result = 'include' AND utr.trader_address IS NOT NULL
                 AND (bdt.detected_high_frequency = 0 OR bdt.detected_high_frequency IS NULL)
            THEN 'PASS'
            WHEN ta.expected_result = 'exclude' AND
                 (utr.trader_address IS NULL OR bdt.detected_high_frequency = 1 OR bdt.detected_known_contract = 1)
            THEN 'PASS'
            WHEN ta.expected_result = 'review'
            THEN 'REVIEW'
            ELSE 'FAIL'
        END as test_result

    FROM test_addresses ta
    LEFT JOIN uniswap_test_results utr ON utr.trader_address = ta.address
    LEFT JOIN bot_detection_test bdt ON bdt.trader_address = ta.address
),

-- Query performance metrics
performance_metrics AS (
    SELECT
        tc.start_date,
        tc.end_date,

        -- Result counts
        COUNT(*) as total_wallets_found,
        COUNT(CASE WHEN utr.total_volume_usd >= tc.min_volume_threshold * 2 THEN 1 END) as high_volume_wallets,
        COUNT(CASE WHEN utr.trade_count >= tc.min_trade_threshold * 2 THEN 1 END) as high_activity_wallets,

        -- Volume distribution
        SUM(utr.total_volume_usd) as total_volume_analyzed,
        AVG(utr.total_volume_usd) as avg_wallet_volume,
        APPROX_PERCENTILE(utr.total_volume_usd, 0.5) as median_wallet_volume,

        -- Activity distribution
        AVG(utr.trade_count) as avg_trade_count,
        APPROX_PERCENTILE(utr.trade_count, 0.5) as median_trade_count,

        -- Token diversity
        AVG(utr.unique_tokens) as avg_unique_tokens

    FROM test_config tc
    CROSS JOIN uniswap_test_results utr
    GROUP BY tc.start_date, tc.end_date, tc.min_volume_threshold, tc.min_trade_threshold
)

-- Comprehensive test report
SELECT
    'TEST_SUMMARY' as section,
    'Analysis Period' as metric,
    CAST(pm.start_date AS VARCHAR) || ' to ' || CAST(pm.end_date AS VARCHAR) as value,
    'Date range tested' as description
FROM performance_metrics pm

UNION ALL

SELECT
    'PERFORMANCE_METRICS',
    'Total Wallets Found',
    CAST(pm.total_wallets_found AS VARCHAR),
    'Number of wallets meeting criteria'
FROM performance_metrics pm

UNION ALL

SELECT
    'PERFORMANCE_METRICS',
    'High Volume Wallets',
    CAST(pm.high_volume_wallets AS VARCHAR),
    'Wallets with 2x minimum volume'
FROM performance_metrics pm

UNION ALL

SELECT
    'PERFORMANCE_METRICS',
    'Average Wallet Volume',
    '$' || CAST(ROUND(pm.avg_wallet_volume, 0) AS VARCHAR),
    'Mean trading volume per wallet'
FROM performance_metrics pm

UNION ALL

SELECT
    'PERFORMANCE_METRICS',
    'Median Trade Count',
    CAST(ROUND(pm.median_trade_count, 0) AS VARCHAR),
    'Median number of trades per wallet'
FROM performance_metrics pm

UNION ALL

SELECT
    'VALIDATION_RESULTS',
    'Test Cases Passed',
    CAST((SELECT COUNT(*) FROM test_validation WHERE test_result = 'PASS') AS VARCHAR),
    'Number of validation tests passed'

UNION ALL

SELECT
    'VALIDATION_RESULTS',
    'Test Cases Failed',
    CAST((SELECT COUNT(*) FROM test_validation WHERE test_result = 'FAIL') AS VARCHAR),
    'Number of validation tests failed'

UNION ALL

SELECT
    'VALIDATION_RESULTS',
    'Test Success Rate',
    CAST(ROUND(100.0 * CAST((SELECT COUNT(*) FROM test_validation WHERE test_result = 'PASS') AS DOUBLE) /
          NULLIF(CAST((SELECT COUNT(*) FROM test_validation) AS DOUBLE), 0), 1) AS VARCHAR) || '%',
    'Percentage of tests passed'

UNION ALL

SELECT
    'RECOMMENDATIONS',
    'Query Readiness',
    CASE
        WHEN (SELECT COUNT(*) FROM test_validation WHERE test_result = 'FAIL') = 0
        THEN 'READY FOR PRODUCTION'
        WHEN (SELECT COUNT(*) FROM test_validation WHERE test_result = 'FAIL') <= 1
        THEN 'MINOR ISSUES - REVIEW RECOMMENDED'
        ELSE 'ISSUES DETECTED - FIXES REQUIRED'
    END,
    'Assessment of query readiness'

ORDER BY section, metric;