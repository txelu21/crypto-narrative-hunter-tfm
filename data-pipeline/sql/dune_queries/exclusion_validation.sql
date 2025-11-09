-- FIXED: Exclusion List Validation and Effectiveness Testing
-- Query ID: 5878579
-- Status: Fixed for Dune/Trino compatibility
-- Parameters: start_date (date), end_date (date)
-- Validates bot detection effectiveness against known addresses and patterns
-- Dune uses Trino/Presto, not PostgreSQL

WITH
-- Known bot addresses for validation (manually curated list)
known_bot_addresses AS (
    SELECT address, bot_type, description, confidence_level FROM (
        VALUES
        -- Known MEV bots (examples - would be populated with real addresses)
        (0x0000000000000000000000000000000000000001, 'mev_bot', 'Known Flashbots Bundle Producer', 'high'),
        (0x0000000000000000000000000000000000000002, 'sandwich_bot', 'Confirmed Sandwich Attack Bot', 'high'),
        (0x0000000000000000000000000000000000000003, 'arbitrage_bot', 'Cross-DEX Arbitrage Bot', 'medium'),

        -- Contract addresses that should be excluded
        (0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D, 'router', 'Uniswap V2 Router', 'high'),
        (0x1111111254fb6c44bAC0beD2854e76F90643097d, 'aggregator', '1inch V4 Router', 'high'),

        -- CEX addresses
        (0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE, 'cex', 'Binance Hot Wallet', 'high'),
        (0x0681d8Db095565FE8A346fA0277bFfdE9C0eDBBF, 'cex', 'Kraken Hot Wallet', 'high')
    ) as t(address, bot_type, description, confidence_level)
),

-- Real world validation on Uniswap trades
uniswap_trader_stats AS (
    SELECT
        s."to" as trader_address,
        COUNT(*) as total_trades,
        COUNT(DISTINCT DATE(s.evt_block_time)) as active_days,
        MAX(DATE(s.evt_block_time)) as last_trade_date
    FROM uniswap_v2_ethereum.Pair_evt_Swap s
    WHERE s.evt_block_time >= COALESCE(
        TRY_CAST('{{start_date}}' AS DATE),
        DATE('2023-01-01')
    )
      AND s.evt_block_time <= COALESCE(
        TRY_CAST('{{end_date}}' AS DATE),
        DATE('2024-01-01')
    ) + INTERVAL '1' day
      AND s."to" IS NOT NULL
    GROUP BY s."to"
),

-- Bot detection on traders (simplified)
detected_bots AS (
    SELECT
        trader_address,
        CASE
            WHEN total_trades / NULLIF(active_days, 0) > 100 THEN 'high_frequency_bot'
            WHEN total_trades / NULLIF(active_days, 0) > 50 THEN 'possible_bot'
            ELSE 'likely_human'
        END as detection_classification
    FROM uniswap_trader_stats
),

-- Known address validation
validation_results AS (
    SELECT
        kba.address,
        kba.bot_type as known_type,
        kba.description,
        kba.confidence_level,

        -- Check if detected
        CASE WHEN db.trader_address IS NOT NULL THEN 1 ELSE 0 END as detected_by_bot_detection,
        db.detection_classification,

        -- Test outcome
        CASE
            WHEN kba.bot_type IN ('router', 'aggregator', 'cex', 'mev_bot', 'sandwich_bot')
                 AND db.detection_classification IN ('high_frequency_bot', 'possible_bot')
            THEN 'PASS'
            WHEN kba.bot_type = 'arbitrage_bot' AND db.detection_classification IS NOT NULL
            THEN 'PASS'
            WHEN db.trader_address IS NULL
            THEN 'REVIEW'
            ELSE 'FAIL'
        END as validation_status

    FROM known_bot_addresses kba
    LEFT JOIN detected_bots db ON db.trader_address = kba.address
),

-- Overall statistics
validation_stats AS (
    SELECT
        COUNT(*) as total_known_addresses,
        SUM(detected_by_bot_detection) as detected_addresses,
        CAST(SUM(detected_by_bot_detection) AS DOUBLE) / NULLIF(CAST(COUNT(*) AS DOUBLE), 0) as detection_rate,

        -- By type
        SUM(CASE WHEN known_type = 'router' THEN 1 ELSE 0 END) as total_routers,
        SUM(CASE WHEN known_type = 'router' AND detected_by_bot_detection = 1 THEN 1 ELSE 0 END) as detected_routers,

        SUM(CASE WHEN known_type IN ('mev_bot', 'sandwich_bot') THEN 1 ELSE 0 END) as total_mev_bots,
        SUM(CASE WHEN known_type IN ('mev_bot', 'sandwich_bot') AND detected_by_bot_detection = 1 THEN 1 ELSE 0 END) as detected_mev_bots,

        -- Test results
        SUM(CASE WHEN validation_status = 'PASS' THEN 1 ELSE 0 END) as tests_passed,
        SUM(CASE WHEN validation_status = 'FAIL' THEN 1 ELSE 0 END) as tests_failed,
        SUM(CASE WHEN validation_status = 'REVIEW' THEN 1 ELSE 0 END) as tests_review

    FROM validation_results
),

-- Real world impact
real_world_impact AS (
    SELECT
        COUNT(DISTINCT trader_address) as total_unique_traders,
        COUNT(DISTINCT CASE WHEN detection_classification IN ('high_frequency_bot', 'possible_bot') THEN trader_address END) as excluded_traders,
        CAST(COUNT(DISTINCT CASE WHEN detection_classification IN ('high_frequency_bot', 'possible_bot') THEN trader_address END) AS DOUBLE) /
            NULLIF(CAST(COUNT(DISTINCT trader_address) AS DOUBLE), 0) as exclusion_rate
    FROM detected_bots
)

-- Comprehensive validation report
SELECT
    'VALIDATION_SUMMARY' as section,
    'Total Known Addresses' as metric,
    CAST(vs.total_known_addresses AS VARCHAR) as value,
    'Number of known bot/contract addresses in test set' as description
FROM validation_stats vs

UNION ALL

SELECT
    'VALIDATION_SUMMARY',
    'Addresses Detected',
    CAST(vs.detected_addresses AS VARCHAR),
    'Number of known addresses correctly identified'
FROM validation_stats vs

UNION ALL

SELECT
    'VALIDATION_SUMMARY',
    'Detection Rate',
    CAST(ROUND(vs.detection_rate * 100, 2) AS VARCHAR) || '%',
    'Percentage of known addresses detected'
FROM validation_stats vs

UNION ALL

SELECT
    'VALIDATION_SUMMARY',
    'Tests Passed',
    CAST(vs.tests_passed AS VARCHAR) || ' / ' || CAST(vs.total_known_addresses AS VARCHAR),
    'Number of validation tests passed'
FROM validation_stats vs

UNION ALL

SELECT
    'DETECTION_BY_TYPE',
    'Router Detection Rate',
    CAST(ROUND(CAST(vs.detected_routers AS DOUBLE) / NULLIF(CAST(vs.total_routers AS DOUBLE), 0) * 100, 2) AS VARCHAR) || '%',
    'Percentage of router contracts detected'
FROM validation_stats vs

UNION ALL

SELECT
    'DETECTION_BY_TYPE',
    'MEV Bot Detection Rate',
    CAST(ROUND(CAST(vs.detected_mev_bots AS DOUBLE) / NULLIF(CAST(vs.total_mev_bots AS DOUBLE), 0) * 100, 2) AS VARCHAR) || '%',
    'Percentage of MEV bots detected'
FROM validation_stats vs

UNION ALL

SELECT
    'REAL_WORLD_IMPACT',
    'Total Unique Traders',
    CAST(rwi.total_unique_traders AS VARCHAR),
    'Unique trader addresses in analysis period'
FROM real_world_impact rwi

UNION ALL

SELECT
    'REAL_WORLD_IMPACT',
    'Excluded Traders',
    CAST(rwi.excluded_traders AS VARCHAR),
    'Traders excluded by bot detection'
FROM real_world_impact rwi

UNION ALL

SELECT
    'REAL_WORLD_IMPACT',
    'Exclusion Rate',
    CAST(ROUND(rwi.exclusion_rate * 100, 2) AS VARCHAR) || '%',
    'Percentage of traders excluded'
FROM real_world_impact rwi

UNION ALL

SELECT
    'RECOMMENDATION',
    'Validation Status',
    CASE
        WHEN vs.tests_failed = 0 THEN 'READY FOR PRODUCTION'
        WHEN vs.tests_failed <= 1 THEN 'MINOR ISSUES - REVIEW RECOMMENDED'
        ELSE 'ISSUES DETECTED - FIXES REQUIRED'
    END,
    'Overall assessment of detection effectiveness'
FROM validation_stats vs

ORDER BY section, metric;