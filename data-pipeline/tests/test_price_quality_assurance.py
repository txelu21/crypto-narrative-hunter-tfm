"""
Tests for Price Quality Assurance Module
"""

import pytest
from services.prices.price_quality_assurance import (
    PriceQualityAssurance,
    QualityGrade,
    QualityMetric,
    BenchmarkComparison
)


class TestPriceQualityAssurance:
    """Test suite for Price QA"""

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data with good quality"""
        data = {}
        base_price = 2000.0

        # Create hourly data for 100 hours
        for i in range(100):
            timestamp = 1609459200 + (i * 3600)
            # Add small variations
            price = base_price + (i * 5)  # Gradual increase
            data[timestamp] = {
                'price_usd': price,
                'source': 'chainlink',
                'block_number': 1000 + i,
                'confidence_score': 1.0
            }

        return data

    @pytest.fixture
    def sample_benchmark_data(self):
        """Sample benchmark data"""
        data = {}
        base_price = 2000.0

        for i in range(100):
            timestamp = 1609459200 + (i * 3600)
            # Slightly different but within tolerance
            price = base_price + (i * 5) + 1.0  # Small deviation
            data[timestamp] = price

        return data

    @pytest.fixture
    def qa_system(self, sample_price_data):
        """Initialize QA system"""
        return PriceQualityAssurance(sample_price_data, accuracy_tolerance=0.02)

    def test_initialization(self, sample_price_data):
        """Test QA system initialization"""
        qa = PriceQualityAssurance(sample_price_data)
        assert len(qa.price_data) == 100
        assert qa.accuracy_tolerance == 0.02

    def test_coverage_check_perfect(self, qa_system):
        """Test coverage check with perfect data"""
        start = 1609459200
        end = 1609459200 + (99 * 3600)

        metric = qa_system._check_coverage(start, end)

        assert metric.name == 'Coverage'
        assert metric.score == 1.0
        assert metric.status == 'pass'
        assert metric.details['coverage_percentage'] >= 99.0

    def test_coverage_check_with_gaps(self):
        """Test coverage check with data gaps"""
        # Data with gaps
        data = {}
        for i in [0, 1, 2, 5, 6, 7, 10]:  # Missing 3, 4, 8, 9
            timestamp = 1609459200 + (i * 3600)
            data[timestamp] = {
                'price_usd': 2000.0,
                'source': 'chainlink',
                'confidence_score': 1.0
            }

        qa = PriceQualityAssurance(data)
        start = 1609459200
        end = 1609459200 + (10 * 3600)

        metric = qa._check_coverage(start, end)

        assert metric.details['gap_count'] == 3  # 10 expected - 7 actual
        assert metric.details['coverage_percentage'] < 100.0
        assert metric.status == 'fail'

    def test_accuracy_check_good(self, qa_system, sample_benchmark_data):
        """Test accuracy check with good data"""
        metric = qa_system._check_accuracy(sample_benchmark_data)

        assert metric.name == 'Accuracy'
        assert metric.score > 0.95
        assert metric.status == 'pass'
        assert metric.details['accuracy_percentage'] > 95.0

    def test_accuracy_check_poor(self, sample_price_data):
        """Test accuracy check with poor data"""
        # Create benchmark with large deviations
        benchmark = {}
        for timestamp in sample_price_data:
            # 10% deviation - should fail 2% tolerance
            benchmark[timestamp] = sample_price_data[timestamp]['price_usd'] * 1.10

        qa = PriceQualityAssurance(sample_price_data, accuracy_tolerance=0.02)
        metric = qa._check_accuracy(benchmark)

        assert metric.status == 'fail'
        assert metric.details['accuracy_percentage'] < 98.0

    def test_consistency_check_good(self, qa_system):
        """Test consistency check with gradual changes"""
        metric = qa_system._check_consistency()

        assert metric.name == 'Consistency'
        assert metric.score >= 0.95
        assert metric.status == 'pass'
        assert metric.details['extreme_changes'] == 0

    def test_consistency_check_with_spikes(self):
        """Test consistency check with price spikes"""
        data = {}
        for i in range(50):
            timestamp = 1609459200 + (i * 3600)
            # Add a huge spike at i=25
            price = 2000.0 if i != 25 else 4000.0
            data[timestamp] = {
                'price_usd': price,
                'source': 'chainlink',
                'confidence_score': 1.0
            }

        qa = PriceQualityAssurance(data)
        metric = qa._check_consistency()

        # The spike should be detected
        assert metric.details['extreme_changes'] > 0
        # Check that consistency is low (multiple extreme changes present)
        assert metric.details['consistency_percentage'] < 100.0

    def test_anomaly_rate_check_clean(self, qa_system):
        """Test anomaly rate with clean data"""
        metric = qa_system._check_anomaly_rate()

        assert metric.name == 'Anomaly Rate'
        assert metric.score == 1.0
        assert metric.status == 'pass'
        assert metric.details['anomaly_count'] == 0

    def test_anomaly_rate_check_with_anomalies(self):
        """Test anomaly rate with corrected prices"""
        data = {}
        for i in range(100):
            timestamp = 1609459200 + (i * 3600)
            data[timestamp] = {
                'price_usd': 2000.0,
                'source': 'chainlink',
                'confidence_score': 1.0,
                'corrected': i % 10 == 0  # 10% anomaly rate
            }

        qa = PriceQualityAssurance(data)
        metric = qa._check_anomaly_rate()

        assert metric.details['anomaly_count'] == 10
        assert metric.details['anomaly_rate_pct'] == 10.0
        assert metric.status == 'fail'

    def test_data_integrity_check_good(self, qa_system):
        """Test data integrity with clean data"""
        metric = qa_system._check_data_integrity()

        assert metric.name == 'Data Integrity'
        assert metric.score >= 0.99
        assert metric.status == 'pass'
        assert 'No integrity issues' in metric.details['issues']

    def test_data_integrity_check_with_issues(self):
        """Test data integrity with various issues"""
        data = {}
        for i in range(100):
            timestamp = 1609459200 + (i * 3600)
            data[timestamp] = {
                'price_usd': 0 if i == 50 else 2000.0,  # Invalid price
                'source': '' if i == 60 else 'chainlink',  # Missing source
                'confidence_score': 1.0
            }

        qa = PriceQualityAssurance(data)
        metric = qa._check_data_integrity()

        assert metric.details['total_issues'] > 0
        assert metric.status == 'fail'

    def test_assign_grade(self, qa_system):
        """Test grade assignment"""
        assert qa_system._assign_grade(0.97) == QualityGrade.A
        assert qa_system._assign_grade(0.92) == QualityGrade.B
        assert qa_system._assign_grade(0.85) == QualityGrade.C
        assert qa_system._assign_grade(0.75) == QualityGrade.D
        assert qa_system._assign_grade(0.50) == QualityGrade.F

    def test_full_qa_report(self, qa_system, sample_benchmark_data):
        """Test full QA report generation"""
        start = min(qa_system.price_data.keys())
        end = max(qa_system.price_data.keys())

        report = qa_system.run_full_qa(
            benchmark_data=sample_benchmark_data,
            start_timestamp=start,
            end_timestamp=end
        )

        assert 'overall_score' in report
        assert 'quality_grade' in report
        assert 'metrics' in report
        assert 'recommendations' in report
        assert 'passed' in report
        assert len(report['metrics']) == 5  # 5 quality metrics
        assert report['passed'] is True
        assert report['quality_grade'] in ['A', 'B', 'C', 'D', 'F']

    def test_generate_recommendations_no_issues(self, qa_system):
        """Test recommendation generation with passing metrics"""
        metrics = [
            QualityMetric(
                name='Test',
                score=1.0,
                weight=1.0,
                status='pass',
                details={}
            )
        ]

        recs = qa_system._generate_recommendations(metrics)
        assert len(recs) == 1
        assert 'no action required' in recs[0].lower()

    def test_generate_recommendations_with_issues(self, qa_system):
        """Test recommendation generation with failing metrics"""
        metrics = [
            QualityMetric(
                name='Coverage',
                score=0.5,
                weight=0.25,
                status='fail',
                details={
                    'gap_count': 50,
                    'coverage_percentage': 90.0,
                    'target_coverage': 99.0
                }
            ),
            QualityMetric(
                name='Accuracy',
                score=0.7,
                weight=0.30,
                status='fail',
                details={
                    'avg_deviation_pct': 5.0,
                    'tolerance_threshold': 2.0
                }
            )
        ]

        recs = qa_system._generate_recommendations(metrics)
        assert len(recs) >= 2
        assert any('gap' in r.lower() for r in recs)
        assert any('deviation' in r.lower() for r in recs)

    def test_validate_against_external_benchmark(self, qa_system):
        """Test validation against external benchmark"""
        external_data = {}
        for timestamp, price_info in qa_system.price_data.items():
            # Add small deviation
            external_data[timestamp] = price_info['price_usd'] * 1.01

        result = qa_system.validate_against_external_benchmark(
            'test_source',
            external_data
        )

        assert result['external_source'] == 'test_source'
        assert result['comparisons_count'] == len(qa_system.price_data)
        assert 'avg_deviation_pct' in result
        assert 'accuracy_percentage' in result

    def test_validate_external_no_overlap(self, qa_system):
        """Test external validation with no overlapping data"""
        external_data = {999999: 2000.0}  # No overlap

        result = qa_system.validate_against_external_benchmark(
            'test_source',
            external_data
        )

        assert 'error' in result
        assert 'No overlapping timestamps' in result['error']

    def test_quality_dashboard(self, qa_system):
        """Test quality dashboard generation"""
        dashboard = qa_system.generate_quality_dashboard()

        assert 'summary' in dashboard
        assert 'source_breakdown' in dashboard
        assert 'recent_anomalies' in dashboard
        assert 'quality_targets' in dashboard

        assert dashboard['summary']['total_data_points'] == 100
        assert 'price_range' in dashboard['summary']
        assert 'time_range' in dashboard['summary']

    def test_source_breakdown(self, qa_system):
        """Test source breakdown calculation"""
        breakdown = qa_system._get_source_breakdown()

        assert 'chainlink' in breakdown
        assert breakdown['chainlink'] == 100

    def test_recent_anomalies(self):
        """Test recent anomalies extraction"""
        data = {}
        for i in range(50):
            timestamp = 1609459200 + (i * 3600)
            data[timestamp] = {
                'price_usd': 2000.0,
                'source': 'chainlink',
                'confidence_score': 1.0,
                'corrected': i < 5,  # First 5 are anomalies
                'original_price': 1500.0 if i < 5 else None
            }

        qa = PriceQualityAssurance(data)
        anomalies = qa._get_recent_anomalies(limit=10)

        assert len(anomalies) == 5
        assert all('original_price' in a for a in anomalies)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])