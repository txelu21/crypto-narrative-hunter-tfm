#!/usr/bin/env python3
"""
Data Quality Report Generator

Generates a comprehensive quality report for all collected data.
"""

import psycopg
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def generate_quality_report():
    db_url = os.getenv('DATABASE_URL')
    conn = psycopg.connect(db_url)

    report = []
    report.append("=" * 80)
    report.append("DATA QUALITY REPORT - Crypto Narrative Hunter")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    with conn.cursor() as cur:
        # 1. Dataset Completeness
        report.append("## 1. DATASET COMPLETENESS")
        report.append("-" * 40)

        cur.execute("SELECT COUNT(*) FROM tokens")
        token_count = cur.fetchone()[0]
        report.append(f"✅ Tokens: {token_count} records")

        cur.execute("SELECT COUNT(*) FROM wallets")
        wallet_count = cur.fetchone()[0]
        report.append(f"✅ Wallets: {wallet_count} records")

        cur.execute("SELECT COUNT(*) FROM transactions")
        tx_count = cur.fetchone()[0]
        report.append(f"✅ Transactions: {tx_count} records")

        cur.execute("SELECT COUNT(*) FROM eth_prices")
        price_count = cur.fetchone()[0]
        report.append(f"✅ ETH Prices: {price_count} records")

        cur.execute("SELECT COUNT(*) FROM token_pools")
        pool_count = cur.fetchone()[0]
        report.append(f"✅ DEX Pools: {pool_count} records")

        cur.execute("SELECT COUNT(*) FROM wallet_performance")
        perf_count = cur.fetchone()[0]
        report.append(f"✅ Wallet Performance: {perf_count} records")

        report.append("")

        # 2. Data Quality Metrics
        report.append("## 2. DATA QUALITY METRICS")
        report.append("-" * 40)

        # Token quality
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE narrative_category IS NOT NULL) as categorized,
                COUNT(*) as total,
                ROUND(100.0 * COUNT(*) FILTER (WHERE narrative_category IS NOT NULL) / COUNT(*), 1) as pct
            FROM tokens
        """)
        categorized, total, pct = cur.fetchone()
        report.append(f"Tokens with narratives: {categorized}/{total} ({pct}%)")

        # Transaction coverage
        cur.execute("""
            SELECT
                COUNT(DISTINCT wallet_address) as wallets_with_tx,
                (SELECT COUNT(*) FROM wallets) as total_wallets,
                ROUND(100.0 * COUNT(DISTINCT wallet_address) / (SELECT COUNT(*) FROM wallets), 2) as pct
            FROM transactions
        """)
        wallets_with_tx, total_wallets, pct = cur.fetchone()
        report.append(f"Wallets with transactions: {wallets_with_tx}/{total_wallets} ({pct}%)")

        # Price coverage
        cur.execute("""
            SELECT
                MIN(timestamp)::date as earliest,
                MAX(timestamp)::date as latest,
                COUNT(*) as total_points
            FROM eth_prices
        """)
        earliest, latest, total_points = cur.fetchone()
        report.append(f"ETH price coverage: {earliest} to {latest} ({total_points} points)")

        report.append("")

        # 3. Data Validity
        report.append("## 3. DATA VALIDITY")
        report.append("-" * 40)

        # Check for NULL values in critical fields
        cur.execute("""
            SELECT COUNT(*) FROM transactions
            WHERE wallet_address IS NULL OR timestamp IS NULL
        """)
        null_txs = cur.fetchone()[0]
        status = "✅" if null_txs == 0 else "⚠️"
        report.append(f"{status} Transactions with NULL critical fields: {null_txs}")

        # Check for invalid addresses
        cur.execute("""
            SELECT COUNT(*) FROM wallets
            WHERE LENGTH(wallet_address) != 42 OR wallet_address NOT LIKE '0x%'
        """)
        invalid_addresses = cur.fetchone()[0]
        status = "✅" if invalid_addresses == 0 else "⚠️"
        report.append(f"{status} Invalid wallet addresses: {invalid_addresses}")

        # Check for duplicate transactions
        cur.execute("""
            SELECT COUNT(*) - COUNT(DISTINCT tx_hash) as duplicates
            FROM transactions
        """)
        dup_txs = cur.fetchone()[0]
        status = "✅" if dup_txs == 0 else "⚠️"
        report.append(f"{status} Duplicate transactions: {dup_txs}")

        report.append("")

        # 4. Statistical Summary
        report.append("## 4. STATISTICAL SUMMARY")
        report.append("-" * 40)

        cur.execute("""
            SELECT
                AVG(total_trades)::numeric(10,1) as avg_trades,
                MIN(total_trades) as min_trades,
                MAX(total_trades) as max_trades,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_trades)::numeric(10,1) as median_trades
            FROM wallet_performance
        """)
        avg_trades, min_trades, max_trades, median_trades = cur.fetchone()
        report.append(f"Wallet trading activity:")
        report.append(f"  - Average: {avg_trades} trades/wallet")
        report.append(f"  - Median: {median_trades} trades/wallet")
        report.append(f"  - Range: {min_trades} to {max_trades} trades")

        cur.execute("""
            SELECT
                narrative_category,
                COUNT(*) as count
            FROM tokens
            WHERE narrative_category IS NOT NULL
            GROUP BY narrative_category
            ORDER BY count DESC
        """)
        report.append("")
        report.append("Narrative distribution:")
        for row in cur.fetchall():
            report.append(f"  - {row[0]}: {row[1]} tokens")

        report.append("")

        # 5. Data Readiness Assessment
        report.append("## 5. DATA READINESS ASSESSMENT")
        report.append("-" * 40)

        # Calculate composite score
        completeness_score = min(100, (tx_count / 50000) * 100)  # Target 50K transactions
        quality_score = (categorized / total) * 100  # Narrative coverage
        coverage_score = (wallets_with_tx / total_wallets) * 100  # Wallet coverage

        composite_score = (completeness_score * 0.3 + quality_score * 0.3 + coverage_score * 0.4)

        grade = "A" if composite_score >= 90 else "B" if composite_score >= 75 else "C" if composite_score >= 60 else "D"

        report.append(f"Completeness Score: {completeness_score:.1f}/100")
        report.append(f"Quality Score: {quality_score:.1f}/100")
        report.append(f"Coverage Score: {coverage_score:.1f}/100")
        report.append("")
        report.append(f"📊 COMPOSITE QUALITY SCORE: {composite_score:.1f}/100 (Grade: {grade})")

        report.append("")

        # 6. Recommendations
        report.append("## 6. RECOMMENDATIONS")
        report.append("-" * 40)

        if categorized < total:
            uncategorized = total - categorized
            report.append(f"⚠️  Review {uncategorized} uncategorized tokens for narrative classification")

        if wallets_with_tx < total_wallets:
            missing = total_wallets - wallets_with_tx
            pct_missing = (missing / total_wallets) * 100
            if pct_missing > 10:
                report.append(f"⚠️  {missing} wallets ({pct_missing:.1f}%) have no transaction data")

        if tx_count < 50000:
            report.append(f"ℹ️  Transaction dataset at {(tx_count/50000)*100:.1f}% of target (50K transactions)")

        if composite_score >= 75:
            report.append("✅ Dataset is ready for analysis and thesis work")
        else:
            report.append("⚠️  Consider additional data collection to improve quality score")

        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

    conn.close()

    # Print and save report
    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    os.makedirs("outputs/reports", exist_ok=True)
    filename = f"outputs/reports/data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(report_text)

    print(f"\n📄 Report saved to: {filename}")

if __name__ == '__main__':
    generate_quality_report()
