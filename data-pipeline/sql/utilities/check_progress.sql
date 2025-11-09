-- Balance Collection Progress Check

SELECT
    '✅ Wallets completed: ' || COUNT(DISTINCT wallet_address) || ' / 2,343 (' ||
    ROUND(COUNT(DISTINCT wallet_address)::numeric * 100 / 2343, 2) || '%)' as progress,
    '📊 Total snapshots:   ' || COUNT(*) as snapshots,
    '🪙 Unique tokens:     ' || COUNT(DISTINCT token_address) as tokens
FROM wallet_token_balances;
