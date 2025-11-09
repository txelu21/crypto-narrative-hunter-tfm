"""Tests for quality scorer."""

import pytest
from services.validation.quality_scorer import QualityScorer


@pytest.fixture
def quality_scorer():
    """Create QualityScorer instance."""
    return QualityScorer()


def test_calculate_composite_quality_score_perfect(quality_scorer):
    """Test composite quality score calculation with perfect scores."""
    validation_results = {
        "completeness": {"overall_completeness": 1.0},
        "accuracy": {"price_accuracy_rate": 1.0},
        "consistency": {"overall_consistency_score": 1.0},
        "timeliness": {"is_fresh": True},
        "validity": {"constraint_compliance_score": 1.0},
    }

    result = quality_scorer.calculate_composite_quality_score(validation_results)

    assert result["composite_score"] >= 0.95
    assert result["quality_grade"] in ["A+", "A"]
    assert len(result["recommendations"]) >= 0


def test_calculate_composite_quality_score_poor(quality_scorer):
    """Test composite quality score calculation with poor scores."""
    validation_results = {
        "completeness": {"overall_completeness": 0.6},
        "accuracy": {"price_accuracy_rate": 0.7},
        "consistency": {"overall_consistency_score": 0.5},
        "timeliness": {"is_fresh": False, "staleness_hours": 48},
        "validity": {"constraint_compliance_score": 0.6},
    }

    result = quality_scorer.calculate_composite_quality_score(validation_results)

    assert result["composite_score"] < 0.80
    assert result["quality_grade"] in ["C", "D", "F"]
    assert len(result["recommendations"]) > 0


def test_assign_quality_grade(quality_scorer):
    """Test quality grade assignment."""
    assert quality_scorer._assign_quality_grade(0.97) == "A+"
    assert quality_scorer._assign_quality_grade(0.92) == "A"
    assert quality_scorer._assign_quality_grade(0.87) == "B+"
    assert quality_scorer._assign_quality_grade(0.82) == "B"
    assert quality_scorer._assign_quality_grade(0.72) == "C"
    assert quality_scorer._assign_quality_grade(0.62) == "D"
    assert quality_scorer._assign_quality_grade(0.52) == "F"


def test_generate_quality_recommendations_all_good(quality_scorer):
    """Test recommendation generation with good scores."""
    dimension_scores = {
        "completeness": 0.95,
        "accuracy": 0.95,
        "consistency": 0.90,
        "timeliness": 0.95,
        "validity": 0.95,
    }

    recommendations = quality_scorer._generate_quality_recommendations(
        dimension_scores
    )

    assert len(recommendations) > 0
    assert any("acceptable thresholds" in r.lower() for r in recommendations)


def test_generate_quality_recommendations_with_issues(quality_scorer):
    """Test recommendation generation with quality issues."""
    dimension_scores = {
        "completeness": 0.70,
        "accuracy": 0.75,
        "consistency": 0.60,
        "timeliness": 0.80,
        "validity": 0.70,
    }

    recommendations = quality_scorer._generate_quality_recommendations(
        dimension_scores
    )

    assert len(recommendations) > 0
    # Should have recommendations for completeness, consistency, and validity
    assert sum(1 for r in recommendations if "completeness" in r.lower()) > 0


def test_calculate_confidence_interval(quality_scorer):
    """Test confidence interval calculation."""
    dimension_scores = {
        "completeness": 0.90,
        "accuracy": 0.95,
        "consistency": 0.85,
        "timeliness": 0.92,
        "validity": 0.88,
    }

    confidence = quality_scorer._calculate_confidence_interval(dimension_scores)

    assert "lower_bound" in confidence
    assert "upper_bound" in confidence
    assert "confidence_level" in confidence
    assert 0 <= confidence["lower_bound"] <= 1
    assert 0 <= confidence["upper_bound"] <= 1
    assert confidence["lower_bound"] <= confidence["upper_bound"]


def test_calculate_quality_trend_improving(quality_scorer):
    """Test quality trend calculation with improving trend."""
    historical_scores = [
        {"timestamp": "2024-01-01", "composite_score": 0.80},
        {"timestamp": "2024-01-02", "composite_score": 0.82},
        {"timestamp": "2024-01-03", "composite_score": 0.85},
        {"timestamp": "2024-01-04", "composite_score": 0.87},
        {"timestamp": "2024-01-05", "composite_score": 0.90},
    ]

    result = quality_scorer.calculate_quality_trend(historical_scores)

    assert result["trend"] == "improving"
    assert result["rate_of_change"] > 0
    assert result["current_score"] == 0.90


def test_calculate_quality_trend_deteriorating(quality_scorer):
    """Test quality trend calculation with deteriorating trend."""
    historical_scores = [
        {"timestamp": "2024-01-01", "composite_score": 0.90},
        {"timestamp": "2024-01-02", "composite_score": 0.87},
        {"timestamp": "2024-01-03", "composite_score": 0.85},
        {"timestamp": "2024-01-04", "composite_score": 0.82},
        {"timestamp": "2024-01-05", "composite_score": 0.80},
    ]

    result = quality_scorer.calculate_quality_trend(historical_scores)

    assert result["trend"] == "deteriorating"
    assert result["rate_of_change"] < 0
    assert result["current_score"] == 0.80


def test_calculate_quality_trend_insufficient_data(quality_scorer):
    """Test quality trend with insufficient data."""
    historical_scores = [{"timestamp": "2024-01-01", "composite_score": 0.90}]

    result = quality_scorer.calculate_quality_trend(historical_scores)

    assert result["trend"] == "insufficient_data"
    assert result["data_points"] == 1