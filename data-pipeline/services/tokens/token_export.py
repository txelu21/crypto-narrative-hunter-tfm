"""
Token export and validation service.

Implements:
- CSV export to `outputs/csv/tokens_metadata.csv` for manual review
- Structured JSON logging with required fields: ts, component, operation, params_hash, status, duration_ms
- Progress visualization using tqdm for developer experience
- Data quality report with completeness metrics
- Validation summary with token count, duplicates found, invalid addresses
- Audit logging for filtering decisions
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from decimal import Decimal

from tqdm import tqdm

from data_collection.common.logging_setup import get_logger
from .token_fetcher import TokenMetadata, FetchStats
from .token_storage import TokenStorage


class TokenExporter:
    """
    Handles token data export and validation reporting.
    """

    def __init__(self):
        self.logger = get_logger("token_exporter")
        self.output_dir = Path("outputs")
        self.csv_dir = self.output_dir / "csv"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def export_tokens_to_csv(self,
                            tokens: List[TokenMetadata],
                            filename: str = "tokens_metadata.csv") -> str:
        """
        Export token metadata to CSV for manual review.

        Args:
            tokens: List of token metadata to export
            filename: CSV filename (default: "tokens_metadata.csv")

        Returns:
            Path to the created CSV file
        """
        start_time = time.time()
        csv_path = self.csv_dir / filename

        self.logger.log_operation(
            operation="export_tokens_csv",
            params={"token_count": len(tokens), "filename": filename},
            status="started",
            message=f"Starting CSV export of {len(tokens)} tokens"
        )

        try:
            # Define CSV headers
            headers = [
                "token_address",
                "symbol",
                "name",
                "decimals",
                "market_cap_rank",
                "market_cap_usd",
                "volume_24h_usd",
                "current_price_usd",
                "coingecko_id",
                "has_ethereum_contract",
                "validation_flags",
                "exported_at"
            ]

            # Write CSV with progress tracking
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                # Export tokens with progress bar
                with tqdm(tokens, desc="Exporting tokens to CSV") as pbar:
                    for token in pbar:
                        row = self._token_to_csv_row(token)
                        writer.writerow(row)

                        pbar.set_postfix({
                            'symbol': token.symbol,
                            'rank': token.market_cap_rank or 'N/A'
                        })

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.log_operation(
                operation="export_tokens_csv",
                params={"token_count": len(tokens), "file_size_bytes": csv_path.stat().st_size},
                status="completed",
                duration_ms=duration_ms,
                message=f"CSV export completed: {csv_path}"
            )

            return str(csv_path)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="export_tokens_csv",
                status="error",
                error=str(e),
                duration_ms=duration_ms,
                message="CSV export failed"
            )
            raise

    def _token_to_csv_row(self, token: TokenMetadata) -> Dict[str, Any]:
        """Convert TokenMetadata to CSV row format"""
        return {
            "token_address": token.token_address,
            "symbol": token.symbol,
            "name": token.name,
            "decimals": token.decimals,
            "market_cap_rank": token.market_cap_rank,
            "market_cap_usd": float(token.market_cap_usd) if token.market_cap_usd else None,
            "volume_24h_usd": float(token.volume_24h_usd) if token.volume_24h_usd else None,
            "current_price_usd": float(token.current_price_usd) if token.current_price_usd else None,
            "coingecko_id": token.coingecko_id,
            "has_ethereum_contract": token.has_ethereum_contract,
            "validation_flags": "|".join(token.validation_flags) if token.validation_flags else "",
            "exported_at": datetime.utcnow().isoformat() + "Z"
        }

    def generate_data_quality_report(self,
                                   tokens: List[TokenMetadata],
                                   stats: FetchStats) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report with completeness metrics.

        Args:
            tokens: List of validated tokens
            stats: Fetch statistics

        Returns:
            Dictionary with data quality metrics
        """
        start_time = time.time()

        self.logger.log_operation(
            operation="generate_quality_report",
            params={"token_count": len(tokens)},
            status="started",
            message="Generating data quality report"
        )

        try:
            # Basic counts
            total_tokens = len(tokens)

            # Completeness analysis
            completeness = self._analyze_completeness(tokens)

            # Validation flags analysis
            validation_analysis = self._analyze_validation_flags(tokens)

            # Market cap and ranking analysis
            ranking_analysis = self._analyze_market_cap_ranking(tokens)

            # Price and volume analysis
            price_volume_analysis = self._analyze_price_volume_data(tokens)

            # Address format validation
            address_validation = self._validate_address_formats(tokens)

            # Summary statistics
            summary_stats = {
                "total_tokens_validated": total_tokens,
                "collection_duration_seconds": stats.duration_seconds,
                "tokens_per_second": total_tokens / stats.duration_seconds if stats.duration_seconds > 0 else 0,
                "data_quality_score": self._calculate_quality_score(completeness, validation_analysis)
            }

            # Compile full report
            quality_report = {
                "report_metadata": {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "report_version": "1.0",
                    "collection_stats": {
                        "total_pages_fetched": stats.total_pages_fetched,
                        "total_tokens_found": stats.total_tokens_found,
                        "ethereum_tokens_found": stats.ethereum_tokens_found,
                        "tokens_with_contracts": stats.tokens_with_contracts,
                        "duplicates_found": stats.duplicates_found,
                        "invalid_addresses": stats.invalid_addresses,
                        "missing_decimals": stats.missing_decimals,
                        "validation_errors": stats.validation_errors
                    }
                },
                "summary_statistics": summary_stats,
                "completeness_metrics": completeness,
                "validation_analysis": validation_analysis,
                "ranking_analysis": ranking_analysis,
                "price_volume_analysis": price_volume_analysis,
                "address_validation": address_validation,
                "recommendations": self._generate_recommendations(completeness, validation_analysis)
            }

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.log_operation(
                operation="generate_quality_report",
                params={
                    "total_tokens": total_tokens,
                    "quality_score": summary_stats["data_quality_score"]
                },
                status="completed",
                duration_ms=duration_ms,
                message=f"Quality report generated: {summary_stats['data_quality_score']:.2f} quality score"
            )

            return quality_report

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="generate_quality_report",
                status="error",
                error=str(e),
                duration_ms=duration_ms,
                message="Quality report generation failed"
            )
            raise

    def _analyze_completeness(self, tokens: List[TokenMetadata]) -> Dict[str, Any]:
        """Analyze data completeness for critical fields"""
        if not tokens:
            return {"error": "No tokens to analyze"}

        total = len(tokens)

        # Count non-null values for each field
        field_completeness = {
            "token_address": sum(1 for t in tokens if t.token_address),
            "symbol": sum(1 for t in tokens if t.symbol),
            "name": sum(1 for t in tokens if t.name),
            "decimals": sum(1 for t in tokens if t.decimals is not None),
            "market_cap_rank": sum(1 for t in tokens if t.market_cap_rank is not None),
            "market_cap_usd": sum(1 for t in tokens if t.market_cap_usd is not None),
            "volume_24h_usd": sum(1 for t in tokens if t.volume_24h_usd is not None),
            "current_price_usd": sum(1 for t in tokens if t.current_price_usd is not None),
            "coingecko_id": sum(1 for t in tokens if t.coingecko_id)
        }

        # Calculate percentages
        completeness_percentages = {
            field: (count / total) * 100 for field, count in field_completeness.items()
        }

        # Critical fields (should be 100%)
        critical_fields = ["token_address", "symbol", "name", "decimals", "coingecko_id"]
        critical_completeness = {
            field: completeness_percentages[field] for field in critical_fields
        }

        return {
            "total_tokens": total,
            "field_completeness_counts": field_completeness,
            "field_completeness_percentages": completeness_percentages,
            "critical_field_completeness": critical_completeness,
            "overall_completeness_score": sum(completeness_percentages.values()) / len(completeness_percentages)
        }

    def _analyze_validation_flags(self, tokens: List[TokenMetadata]) -> Dict[str, Any]:
        """Analyze validation flags and warnings"""
        flag_counts = {}
        tokens_with_flags = 0

        for token in tokens:
            if token.validation_flags:
                tokens_with_flags += 1
                for flag in token.validation_flags:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1

        return {
            "total_tokens_with_flags": tokens_with_flags,
            "percentage_with_flags": (tokens_with_flags / len(tokens)) * 100 if tokens else 0,
            "flag_distribution": flag_counts,
            "most_common_flags": sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def _analyze_market_cap_ranking(self, tokens: List[TokenMetadata]) -> Dict[str, Any]:
        """Analyze market cap ranking distribution"""
        tokens_with_rank = [t for t in tokens if t.market_cap_rank is not None]

        if not tokens_with_rank:
            return {"error": "No tokens with market cap ranking"}

        ranks = [t.market_cap_rank for t in tokens_with_rank]

        return {
            "tokens_with_ranking": len(tokens_with_rank),
            "percentage_with_ranking": (len(tokens_with_rank) / len(tokens)) * 100,
            "rank_range": {"min": min(ranks), "max": max(ranks)},
            "top_10_count": sum(1 for r in ranks if r <= 10),
            "top_100_count": sum(1 for r in ranks if r <= 100),
            "top_500_count": sum(1 for r in ranks if r <= 500),
            "ranking_gaps": self._find_ranking_gaps(ranks)
        }

    def _find_ranking_gaps(self, ranks: List[int]) -> List[Dict[str, int]]:
        """Find gaps in ranking sequence"""
        sorted_ranks = sorted(ranks)
        gaps = []

        for i in range(len(sorted_ranks) - 1):
            gap_size = sorted_ranks[i + 1] - sorted_ranks[i] - 1
            if gap_size > 0:
                gaps.append({
                    "start_rank": sorted_ranks[i] + 1,
                    "end_rank": sorted_ranks[i + 1] - 1,
                    "gap_size": gap_size
                })

        return gaps[:10]  # Return top 10 gaps

    def _analyze_price_volume_data(self, tokens: List[TokenMetadata]) -> Dict[str, Any]:
        """Analyze price and volume data quality"""
        price_data = [t.current_price_usd for t in tokens if t.current_price_usd is not None]
        volume_data = [t.volume_24h_usd for t in tokens if t.volume_24h_usd is not None]

        return {
            "price_data": {
                "tokens_with_price": len(price_data),
                "percentage_with_price": (len(price_data) / len(tokens)) * 100 if tokens else 0,
                "price_range": {
                    "min": float(min(price_data)) if price_data else 0,
                    "max": float(max(price_data)) if price_data else 0
                }
            },
            "volume_data": {
                "tokens_with_volume": len(volume_data),
                "percentage_with_volume": (len(volume_data) / len(tokens)) * 100 if tokens else 0,
                "volume_range": {
                    "min": float(min(volume_data)) if volume_data else 0,
                    "max": float(max(volume_data)) if volume_data else 0
                }
            }
        }

    def _validate_address_formats(self, tokens: List[TokenMetadata]) -> Dict[str, Any]:
        """Validate Ethereum address formats"""
        valid_addresses = 0
        checksum_addresses = 0

        for token in tokens:
            if token.token_address:
                # Basic format check
                if (len(token.token_address) == 42 and
                    token.token_address.startswith('0x') and
                    all(c in '0123456789abcdefABCDEF' for c in token.token_address[2:])):
                    valid_addresses += 1

                    # Check if it has mixed case (checksum)
                    hex_part = token.token_address[2:]
                    if not (hex_part.islower() or hex_part.isupper()):
                        checksum_addresses += 1

        return {
            "total_addresses": len(tokens),
            "valid_format_addresses": valid_addresses,
            "percentage_valid_format": (valid_addresses / len(tokens)) * 100 if tokens else 0,
            "checksum_addresses": checksum_addresses,
            "percentage_checksum": (checksum_addresses / len(tokens)) * 100 if tokens else 0
        }

    def _calculate_quality_score(self, completeness: Dict, validation: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            # Weight different factors
            completeness_score = completeness.get("overall_completeness_score", 0)

            # Penalty for validation flags
            tokens_with_flags_pct = validation.get("percentage_with_flags", 0)
            validation_penalty = min(tokens_with_flags_pct * 0.5, 20)  # Max 20 point penalty

            # Final score
            quality_score = max(0, completeness_score - validation_penalty)
            return round(quality_score, 2)

        except Exception:
            return 0.0

    def _generate_recommendations(self, completeness: Dict, validation: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Completeness recommendations
        critical_completeness = completeness.get("critical_field_completeness", {})
        for field, percentage in critical_completeness.items():
            if percentage < 95:
                recommendations.append(f"Improve {field} completeness (currently {percentage:.1f}%)")

        # Validation flag recommendations
        flag_counts = validation.get("flag_distribution", {})
        for flag, count in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            recommendations.append(f"Address {flag} issues ({count} tokens affected)")

        return recommendations

    def save_quality_report(self,
                           quality_report: Dict[str, Any],
                           filename: str = "tokens_quality_report.json") -> str:
        """
        Save quality report to JSON file.

        Args:
            quality_report: Quality report dictionary
            filename: JSON filename

        Returns:
            Path to the saved report file
        """
        report_path = self.output_dir / filename

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, default=str)

            self.logger.log_operation(
                operation="save_quality_report",
                params={"filename": filename, "file_size": report_path.stat().st_size},
                status="completed",
                message=f"Quality report saved: {report_path}"
            )

            return str(report_path)

        except Exception as e:
            self.logger.log_operation(
                operation="save_quality_report",
                status="error",
                error=str(e),
                message="Failed to save quality report"
            )
            raise

    def create_validation_summary(self,
                                tokens: List[TokenMetadata],
                                stats: FetchStats) -> Dict[str, Any]:
        """
        Create validation summary with key metrics.

        Args:
            tokens: List of validated tokens
            stats: Fetch statistics

        Returns:
            Dictionary with validation summary
        """
        # Count unique addresses
        unique_addresses = len(set(t.token_address for t in tokens))

        # Count tokens by validation flags
        flagged_tokens = sum(1 for t in tokens if t.validation_flags)

        # Count invalid addresses from stats
        invalid_addresses = stats.invalid_addresses

        validation_summary = {
            "collection_summary": {
                "target_tokens": 500,  # From story requirement
                "actual_tokens_collected": len(tokens),
                "collection_success_rate": (len(tokens) / 500) * 100,
                "total_pages_processed": stats.total_pages_fetched,
                "duration_seconds": round(stats.duration_seconds, 2)
            },
            "token_validation": {
                "total_tokens_validated": len(tokens),
                "unique_addresses": unique_addresses,
                "duplicate_addresses_found": len(tokens) - unique_addresses,
                "tokens_with_validation_flags": flagged_tokens,
                "percentage_flagged": (flagged_tokens / len(tokens)) * 100 if tokens else 0
            },
            "data_quality": {
                "tokens_with_ethereum_contracts": sum(1 for t in tokens if t.has_ethereum_contract),
                "tokens_with_market_cap_rank": sum(1 for t in tokens if t.market_cap_rank is not None),
                "tokens_with_volume_data": sum(1 for t in tokens if t.volume_24h_usd is not None),
                "tokens_with_missing_decimals": sum(1 for t in tokens if "missing_decimals" in (t.validation_flags or []))
            },
            "processing_statistics": {
                "total_raw_tokens_found": stats.total_tokens_found,
                "ethereum_tokens_identified": stats.ethereum_tokens_found,
                "tokens_with_contract_details": stats.tokens_with_contracts,
                "duplicates_filtered_out": stats.duplicates_found,
                "invalid_addresses_rejected": stats.invalid_addresses,
                "validation_errors": stats.validation_errors
            }
        }

        self.logger.log_operation(
            operation="create_validation_summary",
            params={
                "tokens_validated": len(tokens),
                "success_rate": validation_summary["collection_summary"]["collection_success_rate"],
                "quality_issues": flagged_tokens
            },
            status="completed",
            message=f"Validation summary created: {len(tokens)} tokens, {flagged_tokens} with flags"
        )

        return validation_summary