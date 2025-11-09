#!/bin/bash
# Balance Collection Progress Monitor

DB_URL="postgresql://txelusanchez@localhost:5432/crypto_narratives"

echo "======================================"
echo "  BALANCE COLLECTION PROGRESS"
echo "======================================"
echo ""

# Get progress stats
psql "$DB_URL" -t -c "
SELECT
    COUNT(DISTINCT wallet_address) as wallets,
    COUNT(*) as snapshots,
    COUNT(DISTINCT token_address) as unique_tokens
FROM wallet_token_balances;
" | while read wallets snapshots tokens; do
    wallets=$(echo $wallets | tr -d ' ')
    snapshots=$(echo $snapshots | tr -d ' ')
    tokens=$(echo $tokens | tr -d ' ')

    # Calculate percentage
    pct=$(echo "scale=2; $wallets * 100 / 2343" | bc)

    echo "✅ Wallets completed: $wallets / 2,343 ($pct%)"
    echo "📊 Total snapshots:   $snapshots"
    echo "🪙 Unique tokens:     $tokens"
done

echo ""
echo "Latest checkpoint:"
ls -lht data/checkpoints/balances/*.json 2>/dev/null | head -1 | awk '{print $6, $7, $8, $9}'

echo ""
echo "To watch live progress:"
echo "  watch -n 10 ./check_balance_progress.sh"
echo ""
