"""
Enhanced export service for narrative categorization and token validation.

Implements:
- Clean dataset export functionality for downstream analysis
- Final token CSV with all metadata and classifications
- Parquet export with proper compression and partitioning
- Data dictionary generation for exported datasets
- Summary statistics for wallet analysis input validation
- Checkpoint preparation for transition to wallet identification phase
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_collection.common.db import execute_with_retry
from data_collection.common.logging_setup import get_logger


class EnhancedExportService:
    """
    Enhanced export service for final token dataset with narrative categorization.

    Provides multiple export formats:
    - CSV: Human-readable format for analysis and review
    - Parquet: Compressed format for downstream data processing
    - JSON: Metadata format with data dictionary and validation summary
    """

    def __init__(self, export_dir: str = "outputs/final_exports"):
        self.logger = get_logger("enhanced_export_service")
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different formats
        self.csv_dir = self.export_dir / "csv"
        self.parquet_dir = self.export_dir / "parquet"
        self.metadata_dir = self.export_dir / "metadata"

        for dir_path in [self.csv_dir, self.parquet_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

    def export_final_dataset(self,
                            include_unclassified: bool = False,
                            include_invalid: bool = False,
                            min_confidence: float = 0.0) -> Dict[str, str]:
        """
        Export the final clean token dataset in multiple formats.

        Args:
            include_unclassified: Include tokens without narrative classification
            include_invalid: Include tokens marked as invalid
            min_confidence: Minimum classification confidence threshold

        Returns:
            Dictionary with paths to exported files
        """
        start_time = time.time()

        self.logger.log_operation(
            operation="export_final_dataset",
            params={
                "include_unclassified": include_unclassified,
                "include_invalid": include_invalid,
                "min_confidence": min_confidence
            },
            status="started",
            message="Starting final dataset export"
        )

        try:
            # Get clean token dataset
            tokens_df = self._get_clean_token_dataset(
                include_unclassified, include_invalid, min_confidence
            )

            if tokens_df.empty:
                raise ValueError("No tokens found matching export criteria")

            # Generate timestamp for file naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export to different formats
            export_paths = {}

            # 1. CSV Export
            csv_path = self._export_to_csv(tokens_df, timestamp)
            export_paths['csv'] = csv_path

            # 2. Parquet Export
            parquet_path = self._export_to_parquet(tokens_df, timestamp)
            export_paths['parquet'] = parquet_path

            # 3. Generate and export metadata
            metadata_path = self._export_metadata(tokens_df, timestamp, export_paths)
            export_paths['metadata'] = metadata_path

            # 4. Generate data dictionary
            dictionary_path = self._export_data_dictionary(timestamp)
            export_paths['data_dictionary'] = dictionary_path

            # 5. Export summary statistics
            summary_path = self._export_summary_statistics(tokens_df, timestamp)
            export_paths['summary'] = summary_path

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.log_operation(
                operation="export_final_dataset",
                params={
                    "tokens_exported": len(tokens_df),
                    "formats_exported": len(export_paths)
                },
                status="completed",
                duration_ms=duration_ms,
                message=f"Final dataset exported: {len(tokens_df)} tokens in {len(export_paths)} formats"
            )

            return export_paths

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.log_operation(
                operation="export_final_dataset",
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
            raise

    def _get_clean_token_dataset(self,
                                include_unclassified: bool,
                                include_invalid: bool,
                                min_confidence: float) -> pd.DataFrame:
        """Get clean token dataset from database based on criteria"""

        # Build WHERE clause based on criteria
        where_conditions = []
        params = []

        if not include_invalid:
            where_conditions.append("(validation_status IS NULL OR validation_status != 'invalid')")

        if not include_unclassified:
            where_conditions.append("narrative_category IS NOT NULL")

        if min_confidence > 0:
            where_conditions.append("(classification_confidence IS NULL OR classification_confidence >= %s)")
            params.append(min_confidence)

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        query = f"""
            SELECT
                token_address,
                symbol,
                name,
                decimals,
                market_cap_rank,
                avg_daily_volume_usd,
                narrative_category,
                classification_confidence,
                liquidity_tier,
                manual_review_status,
                reviewer,
                review_date,
                validation_status,
                validation_flags,
                created_at,
                updated_at
            FROM tokens
            WHERE {where_clause}
            ORDER BY
                CASE WHEN market_cap_rank IS NOT NULL THEN market_cap_rank ELSE 999999 END ASC,
                symbol ASC
        """

        try:
            result = execute_with_retry(query, tuple(params))

            if not result:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(result)

            # Clean and format data
            df = self._clean_dataframe(df)

            return df

        except Exception as e:
            self.logger.log_operation(
                operation="get_clean_token_dataset",
                status="error",
                error=str(e)
            )
            raise

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format the DataFrame for export"""
        # Convert decimal fields to float
        numeric_fields = ['avg_daily_volume_usd', 'classification_confidence']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')

        # Convert timestamps to string for better compatibility
        timestamp_fields = ['created_at', 'updated_at', 'review_date']
        for field in timestamp_fields:
            if field in df.columns:
                df[field] = df[field].astype(str)

        # Replace NaN values with None for better JSON serialization
        df = df.where(pd.notnull(df), None)

        return df

    def _export_to_csv(self, df: pd.DataFrame, timestamp: str) -> str:
        """Export DataFrame to CSV format"""
        filename = f"tokens_final_dataset_{timestamp}.csv"
        csv_path = self.csv_dir / filename

        try:
            df.to_csv(csv_path, index=False, encoding='utf-8')

            self.logger.log_operation(
                operation="export_to_csv",
                params={
                    "filename": filename,
                    "rows": len(df),
                    "file_size": csv_path.stat().st_size
                },
                status="completed",
                message=f"CSV export completed: {filename}"
            )

            return str(csv_path)

        except Exception as e:
            self.logger.log_operation(
                operation="export_to_csv",
                status="error",
                error=str(e)
            )
            raise

    def _export_to_parquet(self, df: pd.DataFrame, timestamp: str) -> str:
        """Export DataFrame to Parquet format with compression"""
        filename = f"tokens_final_dataset_{timestamp}.parquet"
        parquet_path = self.parquet_dir / filename

        try:
            # Convert DataFrame to Arrow Table for better control
            table = pa.Table.from_pandas(df)

            # Add metadata to the table
            metadata = {
                b"export_timestamp": timestamp.encode(),
                b"export_version": b"1.4.0",
                b"story_version": b"1.4",
                b"data_source": b"crypto_narrative_hunter_tfm"
            }

            table = table.replace_schema_metadata(metadata)

            # Write with Snappy compression
            pq.write_table(
                table,
                parquet_path,
                compression='snappy',
                use_dictionary=True,
                row_group_size=10000
            )

            self.logger.log_operation(
                operation="export_to_parquet",
                params={
                    "filename": filename,
                    "rows": len(df),
                    "file_size": parquet_path.stat().st_size,
                    "compression": "snappy"
                },
                status="completed",
                message=f"Parquet export completed: {filename}"
            )

            return str(parquet_path)

        except Exception as e:
            self.logger.log_operation(
                operation="export_to_parquet",
                status="error",
                error=str(e)
            )
            raise

    def _export_metadata(self, df: pd.DataFrame, timestamp: str, export_paths: Dict[str, str]) -> str:
        """Export metadata about the dataset"""
        filename = f"export_metadata_{timestamp}.json"
        metadata_path = self.metadata_dir / filename

        try:
            # Calculate dataset statistics
            total_tokens = len(df)
            classified_tokens = len(df[df['narrative_category'].notna()])
            validated_tokens = len(df[df['validation_status'] == 'valid'])

            # Category distribution
            category_dist = df['narrative_category'].value_counts().to_dict()

            # Liquidity tier distribution
            liquidity_dist = df['liquidity_tier'].value_counts().to_dict()

            # Confidence distribution
            confidence_stats = {}
            if 'classification_confidence' in df.columns:
                confidence_col = df['classification_confidence'].dropna()
                if not confidence_col.empty:
                    confidence_stats = {
                        "mean": float(confidence_col.mean()),
                        "median": float(confidence_col.median()),
                        "std": float(confidence_col.std()),
                        "min": float(confidence_col.min()),
                        "max": float(confidence_col.max())
                    }

            # Manual review statistics
            review_stats = {
                "total_reviewed": len(df[df['reviewer'].notna()]),
                "review_statuses": df['manual_review_status'].value_counts().to_dict()
            }

            # Validation statistics
            validation_stats = {
                "validation_statuses": df['validation_status'].value_counts().to_dict(),
                "tokens_with_flags": len(df[df['validation_flags'].notna()])
            }

            metadata = {
                "export_info": {
                    "export_timestamp": timestamp,
                    "export_version": "1.4.0",
                    "story_version": "1.4",
                    "generated_by": "enhanced_export_service"
                },
                "dataset_statistics": {
                    "total_tokens": total_tokens,
                    "classified_tokens": classified_tokens,
                    "validated_tokens": validated_tokens,
                    "classification_rate": (classified_tokens / total_tokens) * 100 if total_tokens > 0 else 0,
                    "validation_rate": (validated_tokens / total_tokens) * 100 if total_tokens > 0 else 0
                },
                "category_distribution": category_dist,
                "liquidity_distribution": liquidity_dist,
                "confidence_statistics": confidence_stats,
                "review_statistics": review_stats,
                "validation_statistics": validation_stats,
                "exported_files": export_paths,
                "data_quality": {
                    "completeness_rates": self._calculate_completeness_rates(df),
                    "data_types": df.dtypes.astype(str).to_dict()
                }
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)

            self.logger.log_operation(
                operation="export_metadata",
                params={"filename": filename},
                status="completed",
                message=f"Metadata export completed: {filename}"
            )

            return str(metadata_path)

        except Exception as e:
            self.logger.log_operation(
                operation="export_metadata",
                status="error",
                error=str(e)
            )
            raise

    def _export_data_dictionary(self, timestamp: str) -> str:
        """Export data dictionary describing all fields"""
        filename = f"data_dictionary_{timestamp}.json"
        dictionary_path = self.metadata_dir / filename

        data_dictionary = {
            "version": "1.4.0",
            "generated_at": timestamp,
            "description": "Data dictionary for crypto token dataset with narrative categorization",
            "fields": {
                "token_address": {
                    "type": "string",
                    "description": "Ethereum contract address for the token",
                    "format": "0x followed by 40 hexadecimal characters",
                    "constraints": "Unique primary key, checksum validated",
                    "example": "0xA0b86a33E6422EaB84D391FB4CBf77d6e4D91aB6"
                },
                "symbol": {
                    "type": "string",
                    "description": "Token symbol/ticker",
                    "constraints": "Max 20 characters, uppercase",
                    "example": "UNI"
                },
                "name": {
                    "type": "string",
                    "description": "Full token name",
                    "constraints": "Max 100 characters",
                    "example": "Uniswap"
                },
                "decimals": {
                    "type": "integer",
                    "description": "Number of decimal places for token amounts",
                    "constraints": "0-18, typically 18 for ERC-20",
                    "example": 18
                },
                "market_cap_rank": {
                    "type": "integer",
                    "description": "CoinGecko market cap ranking",
                    "constraints": "Positive integer, lower is better",
                    "example": 1
                },
                "avg_daily_volume_usd": {
                    "type": "float",
                    "description": "Average daily trading volume in USD",
                    "constraints": "Non-negative",
                    "example": 1000000.50
                },
                "narrative_category": {
                    "type": "string",
                    "description": "Narrative theme classification",
                    "constraints": "One of: DeFi, Gaming, AI, Infrastructure, Meme, Stablecoin, Other",
                    "example": "DeFi"
                },
                "classification_confidence": {
                    "type": "float",
                    "description": "Confidence score for narrative classification",
                    "constraints": "0.0-100.0, higher is more confident",
                    "example": 95.5
                },
                "liquidity_tier": {
                    "type": "integer",
                    "description": "Liquidity tier based on trading volume",
                    "constraints": "1 (high), 2 (medium), 3 (low)",
                    "example": 1
                },
                "manual_review_status": {
                    "type": "string",
                    "description": "Status of manual review process",
                    "constraints": "pending, auto_classified, manual_reviewed, reviewed_approved, reviewed_rejected",
                    "example": "manual_reviewed"
                },
                "reviewer": {
                    "type": "string",
                    "description": "Username of person who reviewed classification",
                    "constraints": "Max 50 characters",
                    "example": "alice"
                },
                "review_date": {
                    "type": "timestamp",
                    "description": "When manual review was completed",
                    "format": "ISO 8601",
                    "example": "2025-09-27T14:30:00Z"
                },
                "validation_status": {
                    "type": "string",
                    "description": "Result of token validation",
                    "constraints": "valid, invalid",
                    "example": "valid"
                },
                "validation_flags": {
                    "type": "string",
                    "description": "Comma-separated validation warnings/issues",
                    "example": "unusual_decimals,low_volume_top_token"
                },
                "created_at": {
                    "type": "timestamp",
                    "description": "When token was first added to database",
                    "format": "ISO 8601",
                    "example": "2025-09-27T10:00:00Z"
                },
                "updated_at": {
                    "type": "timestamp",
                    "description": "When token was last updated",
                    "format": "ISO 8601",
                    "example": "2025-09-27T15:45:00Z"
                }
            },
            "narrative_categories": {
                "DeFi": "Decentralized finance protocols, DEXs, lending, yield farming",
                "Gaming": "GameFi, NFT gaming, virtual worlds, play-to-earn",
                "AI": "Artificial intelligence, machine learning, data processing",
                "Infrastructure": "Layer 2, bridges, oracles, development tools",
                "Meme": "Community-driven tokens, meme coins, social tokens",
                "Stablecoin": "USD-pegged, algorithmic stables, reserve-backed",
                "Other": "Uncategorized or multi-category tokens"
            },
            "liquidity_tiers": {
                "1": "High liquidity: >$1M daily volume",
                "2": "Medium liquidity: $100K-$1M daily volume",
                "3": "Low liquidity: <$100K daily volume"
            }
        }

        try:
            with open(dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(data_dictionary, f, indent=2)

            self.logger.log_operation(
                operation="export_data_dictionary",
                params={"filename": filename},
                status="completed",
                message=f"Data dictionary export completed: {filename}"
            )

            return str(dictionary_path)

        except Exception as e:
            self.logger.log_operation(
                operation="export_data_dictionary",
                status="error",
                error=str(e)
            )
            raise

    def _export_summary_statistics(self, df: pd.DataFrame, timestamp: str) -> str:
        """Export summary statistics for wallet analysis input validation"""
        filename = f"summary_statistics_{timestamp}.json"
        summary_path = self.metadata_dir / filename

        try:
            # Basic token counts by category
            category_counts = df['narrative_category'].value_counts().to_dict()

            # Volume statistics by category
            volume_stats_by_category = {}
            for category in category_counts.keys():
                if category:
                    category_df = df[df['narrative_category'] == category]
                    volume_col = category_df['avg_daily_volume_usd'].dropna()
                    if not volume_col.empty:
                        volume_stats_by_category[category] = {
                            "count": len(volume_col),
                            "mean_volume": float(volume_col.mean()),
                            "median_volume": float(volume_col.median()),
                            "total_volume": float(volume_col.sum())
                        }

            # Market cap rank distribution
            rank_col = df['market_cap_rank'].dropna()
            rank_distribution = {}
            if not rank_col.empty:
                rank_distribution = {
                    "top_10": len(rank_col[rank_col <= 10]),
                    "top_50": len(rank_col[rank_col <= 50]),
                    "top_100": len(rank_col[rank_col <= 100]),
                    "top_500": len(rank_col[rank_col <= 500]),
                    "beyond_500": len(rank_col[rank_col > 500])
                }

            # Liquidity tier analysis
            liquidity_analysis = {}
            if 'liquidity_tier' in df.columns:
                liquidity_counts = df['liquidity_tier'].value_counts().to_dict()
                liquidity_analysis = {
                    "tier_distribution": liquidity_counts,
                    "high_liquidity_tokens": liquidity_counts.get(1, 0),
                    "medium_liquidity_tokens": liquidity_counts.get(2, 0),
                    "low_liquidity_tokens": liquidity_counts.get(3, 0)
                }

            # Data quality summary
            quality_summary = {
                "total_tokens": len(df),
                "classified_tokens": len(df[df['narrative_category'].notna()]),
                "validated_tokens": len(df[df['validation_status'] == 'valid']),
                "manually_reviewed_tokens": len(df[df['reviewer'].notna()]),
                "tokens_with_market_rank": len(df[df['market_cap_rank'].notna()]),
                "tokens_with_volume_data": len(df[df['avg_daily_volume_usd'].notna()])
            }

            summary_statistics = {
                "export_info": {
                    "generated_at": timestamp,
                    "dataset_version": "1.4.0",
                    "purpose": "Input validation for wallet analysis phase"
                },
                "data_quality_summary": quality_summary,
                "narrative_distribution": {
                    "category_counts": category_counts,
                    "volume_stats_by_category": volume_stats_by_category
                },
                "market_analysis": {
                    "rank_distribution": rank_distribution,
                    "liquidity_analysis": liquidity_analysis
                },
                "readiness_indicators": {
                    "classification_completeness": (quality_summary["classified_tokens"] / quality_summary["total_tokens"]) * 100,
                    "validation_completeness": (quality_summary["validated_tokens"] / quality_summary["total_tokens"]) * 100,
                    "manual_review_coverage": (quality_summary["manually_reviewed_tokens"] / quality_summary["total_tokens"]) * 100,
                    "ready_for_wallet_analysis": quality_summary["classified_tokens"] >= quality_summary["total_tokens"] * 0.9
                }
            }

            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_statistics, f, indent=2, default=str)

            self.logger.log_operation(
                operation="export_summary_statistics",
                params={"filename": filename},
                status="completed",
                message=f"Summary statistics export completed: {filename}"
            )

            return str(summary_path)

        except Exception as e:
            self.logger.log_operation(
                operation="export_summary_statistics",
                status="error",
                error=str(e)
            )
            raise

    def _calculate_completeness_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness rates for all fields"""
        completeness = {}
        total_rows = len(df)

        if total_rows == 0:
            return completeness

        for column in df.columns:
            non_null_count = df[column].notna().sum()
            completeness[column] = (non_null_count / total_rows) * 100

        return completeness

    def create_checkpoint_file(self, export_paths: Dict[str, str]) -> str:
        """
        Create checkpoint file for transition to wallet identification phase.

        Args:
            export_paths: Dictionary of exported file paths

        Returns:
            Path to checkpoint file
        """
        checkpoint_filename = f"story_1_4_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        checkpoint_path = self.export_dir / checkpoint_filename

        try:
            # Get basic statistics for checkpoint
            stats_query = """
                SELECT
                    COUNT(*) as total_tokens,
                    COUNT(narrative_category) as classified_tokens,
                    COUNT(CASE WHEN validation_status = 'valid' THEN 1 END) as validated_tokens,
                    COUNT(CASE WHEN manual_review_status LIKE 'manual%' OR manual_review_status LIKE 'reviewed%' THEN 1 END) as manually_reviewed
                FROM tokens
            """

            stats_result = execute_with_retry(stats_query)
            stats = stats_result[0] if stats_result else {}

            checkpoint_data = {
                "story_info": {
                    "story_version": "1.4",
                    "story_name": "Narrative Categorization and Token Validation",
                    "completion_date": datetime.now().isoformat(),
                    "status": "completed"
                },
                "completion_statistics": stats,
                "exported_datasets": export_paths,
                "next_phase": {
                    "story_version": "1.5",
                    "story_name": "Smart Wallet Identification",
                    "input_datasets": [export_paths.get('parquet', ''), export_paths.get('csv', '')],
                    "prerequisites_met": True
                },
                "data_quality_checkpoint": {
                    "classification_rate": (stats.get('classified_tokens', 0) / stats.get('total_tokens', 1)) * 100,
                    "validation_rate": (stats.get('validated_tokens', 0) / stats.get('total_tokens', 1)) * 100,
                    "manual_review_coverage": (stats.get('manually_reviewed', 0) / stats.get('total_tokens', 1)) * 100,
                    "ready_for_next_phase": True
                }
            }

            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            self.logger.log_operation(
                operation="create_checkpoint_file",
                params={"filename": checkpoint_filename},
                status="completed",
                message=f"Checkpoint file created: {checkpoint_filename}"
            )

            return str(checkpoint_path)

        except Exception as e:
            self.logger.log_operation(
                operation="create_checkpoint_file",
                status="error",
                error=str(e)
            )
            raise