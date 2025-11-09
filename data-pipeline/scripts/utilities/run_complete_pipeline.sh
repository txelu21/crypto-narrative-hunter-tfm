#!/bin/bash
# File: run_complete_pipeline.sh
# Purpose: Execute entire data collection pipeline
# Based on OPERATIONAL_GUIDE.md Section 4

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_phase() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Change to data-collection directory
cd "$(dirname "$0")"

# Verify prerequisites
log_info "Verifying prerequisites..."
which uv > /dev/null || { log_error "uv not found"; exit 1; }
uv --version || { log_error "uv not working"; exit 1; }
log_info "✓ Using uv package manager"

# Check database connectivity
log_info "Testing database connection..."
psql "${DATABASE_URL:-postgresql://txelusanchez@localhost:5432/crypto_narratives}" -c "SELECT 1;" > /dev/null || { log_error "Database connection failed"; exit 1; }
log_info "✓ Database connection successful"

# PHASE 1: TOKEN METADATA COLLECTION
log_phase "PHASE 1: TOKEN METADATA COLLECTION"

# Check if tokens are already collected
TOKEN_COUNT=$(psql "${DATABASE_URL:-postgresql://txelusanchez@localhost:5432/crypto_narratives}" -t -c "SELECT COUNT(*) FROM tokens;")
if [ "$TOKEN_COUNT" -ge 400 ]; then
    log_info "✓ Token collection already complete ($TOKEN_COUNT tokens found)"
else
    log_info "Story 1.2: Collecting tokens from CoinGecko..."
    uv run services/tokens/token_collection_service.py
    log_info "✓ Token collection complete"
fi

log_info "Story 1.3: Analyzing DEX liquidity..."
uv run cli_liquidity.py analyze
log_info "✓ Liquidity analysis complete"

log_info "Story 1.4: Narrative categorization..."
uv run narrative_classification_orchestrator.py pipeline --all
log_info "✓ Narrative categorization complete"

# PHASE 2: WALLET IDENTIFICATION
log_phase "PHASE 2: SMART WALLET IDENTIFICATION"

log_info "Story 2.1: Identifying smart money wallets..."
uv run services/smart_wallet_query_manager.py
log_info "✓ Wallet identification complete"

log_info "Story 2.2: Calculating performance metrics..."
if [ -f "services/wallets/performance_calculator.py" ]; then
    uv run services/wallets/performance_calculator.py
    log_info "✓ Performance calculation complete"
else
    log_warning "Performance calculator not found, skipping..."
fi

# PHASE 3: TRANSACTION & BALANCE DATA
log_phase "PHASE 3: TRANSACTION & BALANCE DATA"

log_info "Story 3.1: Extracting transaction history..."
log_warning "This may take 12-24 hours. Progress will be logged to logs/collection_*.log"
uv run services/transactions/batch_processor.py
log_info "✓ Transaction extraction complete"

log_info "Story 3.2: Collecting balance snapshots..."
if [ -f "services/balances/balance_extractor.py" ]; then
    uv run services/balances/balance_extractor.py
    log_info "✓ Balance collection complete"
else
    log_warning "Balance extractor not found, skipping..."
fi

log_info "Story 3.3: Collecting ETH price history..."
uv run services/prices/chainlink_client.py
log_info "✓ Price collection complete"

# PHASE 4: VALIDATION & EXPORT
log_phase "PHASE 4: VALIDATION & EXPORT"

log_info "Story 3.4: Running data validation..."
uv run services/validation/quality_reporter.py
log_info "✓ Validation complete"

log_info "Story 3.5: Exporting final datasets..."
uv run services/export/data_exporter.py
log_info "✓ Export complete"

# SUMMARY
log_phase "PIPELINE EXECUTION COMPLETE"
log_info "Final datasets available in: outputs/"
log_info "Documentation available in: outputs/documentation/"
log_info "Quality report: outputs/json/quality_certification.json"

# Display final statistics
log_info "Final Statistics:"
psql "${DATABASE_URL:-postgresql://txelusanchez@localhost:5432/crypto_narratives}" -c "
    SELECT
        (SELECT COUNT(*) FROM tokens) as tokens,
        (SELECT COUNT(*) FROM wallets) as wallets,
        (SELECT COUNT(*) FROM transactions) as transactions,
        (SELECT COUNT(*) FROM wallet_balances) as balances,
        (SELECT COUNT(*) FROM eth_prices) as eth_prices
    ;
"

log_info "✓ Ready for analysis phase!"