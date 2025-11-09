"""
Integration tests for narrative classification system.

Tests the complete token processing pipeline from CoinGecko to final export:
- Complete token processing pipeline from CoinGecko to final export
- Narrative classification accuracy with known test datasets
- Manual review workflow with simulated user interactions
- Data validation and cleaning effectiveness
- Export format compatibility with downstream systems
"""

import pytest
import tempfile
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from services.tokens.narrative_classifier import NarrativeClassifier, NarrativeCategory, ClassificationResult
from services.tokens.token_classification_service import TokenClassificationService
from services.tokens.manual_review_service import ManualReviewService
from services.tokens.token_validation_service import TokenValidationService
from services.tokens.data_quality_service import DataQualityService
from services.tokens.enhanced_export_service import EnhancedExportService
from data_collection.common.db import execute_with_retry


class TestNarrativeClassificationIntegration:
    """Integration tests for the complete narrative classification system"""

    @pytest.fixture
    def test_tokens(self) -> List[Dict[str, Any]]:
        """Sample test tokens with known narrative categories"""
        return [
            {
                "token_address": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
                "symbol": "UNI",
                "name": "Uniswap",
                "decimals": 18,
                "market_cap_rank": 1,
                "expected_category": "DeFi"
            },
            {
                "token_address": "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce",
                "symbol": "SHIB",
                "name": "Shiba Inu",
                "decimals": 18,
                "market_cap_rank": 2,
                "expected_category": "Meme"
            },
            {
                "token_address": "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0",
                "symbol": "MATIC",
                "name": "Polygon",
                "decimals": 18,
                "market_cap_rank": 3,
                "expected_category": "Infrastructure"
            },
            {
                "token_address": "0xa0b86a33e6422eab84d391fb4cbf77d6e4d91ab6",
                "symbol": "SAND",
                "name": "The Sandbox",
                "decimals": 18,
                "market_cap_rank": 4,
                "expected_category": "Gaming"
            },
            {
                "token_address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "symbol": "USDT",
                "name": "Tether USD",
                "decimals": 6,
                "market_cap_rank": 5,
                "expected_category": "Stablecoin"
            }
        ]

    @pytest.fixture
    def classifier(self) -> NarrativeClassifier:
        """Create narrative classifier instance"""
        return NarrativeClassifier()

    @pytest.fixture
    def temp_review_dir(self) -> Path:
        """Create temporary directory for review files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_narrative_classifier_accuracy(self, classifier: NarrativeClassifier, test_tokens: List[Dict[str, Any]]):
        """Test narrative classification accuracy with known test datasets"""
        correct_classifications = 0
        total_classifications = len(test_tokens)

        for token in test_tokens:
            result = classifier.classify_token(
                token["token_address"],
                token["symbol"],
                token["name"]
            )

            # Check if classification matches expected category
            if result.category.value == token["expected_category"]:
                correct_classifications += 1

            # Verify result structure
            assert isinstance(result, ClassificationResult)
            assert isinstance(result.category, NarrativeCategory)
            assert 0 <= result.confidence <= 100
            assert isinstance(result.matched_keywords, list)
            assert isinstance(result.reasoning, str)

        # Calculate accuracy
        accuracy = (correct_classifications / total_classifications) * 100

        # Assert minimum accuracy threshold
        assert accuracy >= 80, f"Classification accuracy too low: {accuracy}%"

        print(f"Classification accuracy: {accuracy}% ({correct_classifications}/{total_classifications})")

    def test_complete_classification_pipeline(self, test_tokens: List[Dict[str, Any]]):
        """Test complete token processing pipeline from classification to database"""
        classification_service = TokenClassificationService()

        # Mock database calls for testing
        with patch('services.tokens.token_classification_service.execute_with_retry') as mock_execute:
            # Mock getting tokens for classification
            mock_execute.return_value = test_tokens

            with patch('services.tokens.token_classification_service.get_cursor'):
                # Run classification
                stats = classification_service.classify_specific_tokens([
                    token["token_address"] for token in test_tokens
                ])

                # Verify statistics
                assert stats.total_tokens_validated >= 0
                assert stats.classified_tokens >= 0
                assert stats.duration_seconds >= 0

    def test_manual_review_workflow(self, temp_review_dir: Path, test_tokens: List[Dict[str, Any]]):
        """Test manual review workflow with simulated user interactions"""
        review_service = ManualReviewService(str(temp_review_dir))

        # Mock database calls
        with patch('services.tokens.manual_review_service.execute_with_retry') as mock_execute:
            # Mock tokens needing review
            mock_execute.return_value = test_tokens

            # Test CSV export for manual review
            export_stats = review_service.export_tokens_for_review()

            assert export_stats.total_tokens_exported == len(test_tokens)
            assert export_stats.file_path
            assert Path(export_stats.file_path).exists()

            # Verify CSV format
            with open(export_stats.file_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == len(test_tokens)

                # Check required headers
                required_headers = ['token_address', 'symbol', 'name', 'manual_category']
                for header in required_headers:
                    assert header in reader.fieldnames

            # Simulate manual review by creating reviewed CSV
            reviewed_csv_path = temp_review_dir / "reviewed_tokens.csv"
            self._create_reviewed_csv(reviewed_csv_path, test_tokens)

            # Test CSV import
            with patch('services.tokens.manual_review_service.get_cursor'):
                import_stats = review_service.import_reviewed_classifications(
                    str(reviewed_csv_path),
                    "test_reviewer"
                )

                assert import_stats.total_rows_processed == len(test_tokens)
                assert import_stats.successful_imports > 0
                assert import_stats.validation_errors == 0

    def test_token_validation_effectiveness(self, test_tokens: List[Dict[str, Any]]):
        """Test data validation and cleaning effectiveness"""
        validation_service = TokenValidationService()

        # Add some invalid tokens to test validation
        invalid_tokens = [
            {
                "token_address": "invalid_address",
                "symbol": "",
                "name": "",
                "decimals": -1
            },
            {
                "token_address": "0x123",  # Too short
                "symbol": "TEST" * 10,  # Too long
                "name": "Test Token",
                "decimals": 25  # Out of range
            }
        ]

        all_tokens = test_tokens + invalid_tokens

        # Mock database calls
        with patch('services.tokens.token_validation_service.execute_with_retry') as mock_execute:
            mock_execute.return_value = all_tokens

            with patch('services.tokens.token_validation_service.get_cursor'):
                stats = validation_service.validate_all_tokens()

                # Verify that validation detected invalid tokens
                assert stats.invalid_tokens > 0
                assert stats.address_validation_errors > 0
                assert stats.validation_errors > 0

    def test_data_quality_reporting(self, test_tokens: List[Dict[str, Any]]):
        """Test data quality reporting and metrics generation"""
        quality_service = DataQualityService()

        # Mock database calls for comprehensive report
        with patch('services.tokens.data_quality_service.execute_with_retry') as mock_execute:
            # Mock different queries with appropriate responses
            def mock_query_response(query, params=None):
                if "COUNT(*)" in query and "total_tokens" in query:
                    return [{"total_tokens": len(test_tokens), "classified_tokens": len(test_tokens), "validated_tokens": len(test_tokens)}]
                elif "narrative_category" in query and "GROUP BY" in query:
                    return [{"narrative_category": "DeFi", "count": 2}, {"narrative_category": "Meme", "count": 1}]
                elif "validation_status" in query:
                    return [{"total_tokens": len(test_tokens), "valid_tokens": len(test_tokens), "invalid_tokens": 0}]
                else:
                    return []

            mock_execute.side_effect = mock_query_response

            # Generate comprehensive report
            metrics = quality_service.generate_comprehensive_report()

            # Verify report structure
            assert metrics.total_tokens > 0
            assert 0 <= metrics.overall_quality_score <= 100
            assert isinstance(metrics.category_distribution, dict)
            assert isinstance(metrics.completeness_rates, dict)

    def test_export_format_compatibility(self, test_tokens: List[Dict[str, Any]], temp_review_dir: Path):
        """Test export format compatibility with downstream systems"""
        export_service = EnhancedExportService(str(temp_review_dir))

        # Mock database calls
        with patch('services.tokens.enhanced_export_service.execute_with_retry') as mock_execute:
            # Convert test tokens to DataFrame format
            df_data = []
            for token in test_tokens:
                df_data.append({
                    **token,
                    "narrative_category": token["expected_category"],
                    "classification_confidence": 95.0,
                    "liquidity_tier": 1,
                    "validation_status": "valid"
                })

            mock_execute.return_value = df_data

            # Test export functionality
            export_paths = export_service.export_final_dataset()

            # Verify all export formats were created
            assert 'csv' in export_paths
            assert 'parquet' in export_paths
            assert 'metadata' in export_paths
            assert 'data_dictionary' in export_paths

            # Verify CSV format
            csv_path = Path(export_paths['csv'])
            assert csv_path.exists()

            df_csv = pd.read_csv(csv_path)
            assert len(df_csv) == len(test_tokens)
            assert 'token_address' in df_csv.columns
            assert 'narrative_category' in df_csv.columns

            # Verify Parquet format
            parquet_path = Path(export_paths['parquet'])
            assert parquet_path.exists()

            df_parquet = pd.read_parquet(parquet_path)
            assert len(df_parquet) == len(test_tokens)

            # Verify metadata format
            metadata_path = Path(export_paths['metadata'])
            assert metadata_path.exists()

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                assert 'export_info' in metadata
                assert 'dataset_statistics' in metadata
                assert 'category_distribution' in metadata

    def test_end_to_end_pipeline_performance(self, test_tokens: List[Dict[str, Any]]):
        """Test end-to-end pipeline performance and error handling"""
        import time

        start_time = time.time()

        # Initialize all services
        classifier = NarrativeClassifier()
        classification_service = TokenClassificationService()
        validation_service = TokenValidationService()

        # Test classification performance
        classification_results = []
        for token in test_tokens:
            result = classifier.classify_token(
                token["token_address"],
                token["symbol"],
                token["name"]
            )
            classification_results.append(result)

        # Verify performance
        end_time = time.time()
        duration = end_time - start_time
        tokens_per_second = len(test_tokens) / duration

        # Performance assertions
        assert tokens_per_second > 10, f"Classification too slow: {tokens_per_second:.2f} tokens/second"
        assert all(isinstance(r, ClassificationResult) for r in classification_results)

        # Test error handling with invalid input
        try:
            classifier.classify_token("", "", "")
            # Should not raise exception, should return valid result
        except Exception as e:
            pytest.fail(f"Classifier should handle invalid input gracefully: {e}")

    def test_edge_cases_and_error_recovery(self, classifier: NarrativeClassifier):
        """Test handling of edge cases and error recovery scenarios"""

        # Test empty/null inputs
        result = classifier.classify_token("", "", "")
        assert isinstance(result, ClassificationResult)
        assert result.category == NarrativeCategory.OTHER

        # Test very long inputs
        long_name = "A" * 1000
        result = classifier.classify_token(
            "0x1234567890123456789012345678901234567890",
            "LONG",
            long_name
        )
        assert isinstance(result, ClassificationResult)

        # Test special characters
        result = classifier.classify_token(
            "0x1234567890123456789012345678901234567890",
            "SP€C!@L",
            "Special Çharactérs Token"
        )
        assert isinstance(result, ClassificationResult)

        # Test conflicting keywords
        result = classifier.classify_token(
            "0x1234567890123456789012345678901234567890",
            "DEFI_GAME",
            "DeFi Gaming Meme Token"
        )
        assert isinstance(result, ClassificationResult)
        # Should pick one category based on priority rules

    def _create_reviewed_csv(self, csv_path: Path, tokens: List[Dict[str, Any]]) -> None:
        """Create a reviewed CSV file for testing import functionality"""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'token_address', 'symbol', 'name', 'suggested_category',
                'confidence', 'manual_category', 'review_notes', 'reviewer'
            ])

            # Write token data with manual categories
            for token in tokens:
                writer.writerow([
                    token['token_address'],
                    token['symbol'],
                    token['name'],
                    token.get('expected_category', ''),
                    '95.0',
                    token['expected_category'],
                    'Manually reviewed for testing',
                    'test_reviewer'
                ])


class TestPerformanceOptimization:
    """Test performance optimization features"""

    def test_batch_processing_performance(self):
        """Test batch processing performance with large datasets"""
        classification_service = TokenClassificationService()

        # Create mock large dataset
        large_token_set = [
            {
                "token_address": f"0x{'a' * 38}{i:04d}",
                "symbol": f"TOKEN{i}",
                "name": f"Test Token {i}",
                "decimals": 18
            }
            for i in range(1000)
        ]

        with patch('services.tokens.token_classification_service.execute_with_retry') as mock_execute:
            mock_execute.return_value = large_token_set

            with patch('services.tokens.token_classification_service.get_cursor'):
                start_time = time.time()
                stats = classification_service.classify_all_tokens(batch_size=100)
                duration = time.time() - start_time

                # Performance assertions
                assert stats.total_tokens_validated == len(large_token_set)
                assert duration < 60, f"Batch processing too slow: {duration:.2f}s for 1000 tokens"

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process large dataset
        classifier = NarrativeClassifier()
        for i in range(1000):
            classifier.classify_token(
                f"0x{'a' * 38}{i:04d}",
                f"TOKEN{i}",
                f"Test Token {i}"
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory should not increase significantly
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage increased too much: {memory_increase / 1024 / 1024:.2f}MB"

    def test_concurrent_classification(self):
        """Test concurrent classification processing"""
        import concurrent.futures
        import threading

        classifier = NarrativeClassifier()
        tokens = [
            ("0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", "UNI", "Uniswap"),
            ("0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", "SHIB", "Shiba Inu"),
            ("0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0", "MATIC", "Polygon"),
        ]

        def classify_token(token_data):
            address, symbol, name = token_data
            return classifier.classify_token(address, symbol, name)

        # Test concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            start_time = time.time()
            results = list(executor.map(classify_token, tokens))
            duration = time.time() - start_time

            # Verify results
            assert len(results) == len(tokens)
            assert all(isinstance(r, ClassificationResult) for r in results)

            # Should be faster than sequential processing
            assert duration < 5, f"Concurrent processing too slow: {duration:.2f}s"


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v"])