"""
Wallet analysis services

This package contains services for analyzing wallet performance and behavior.
"""

from .performance_calculator import (
    WalletPerformanceCalculator,
    Trade,
    TradeReturn,
    PerformanceMetrics,
    validate_performance_metrics
)

from .risk_analyzer import (
    AdvancedRiskAnalyzer,
    RiskMetrics,
    DrawdownPeriod
)

from .gas_analyzer import (
    GasEfficiencyAnalyzer,
    TransactionGasData,
    TransactionType,
    GasMetrics,
    MEVImpactAnalysis
)

from .wallet_filter import (
    WalletFilter,
    BatchWalletFilter,
    FilterCriteria,
    FilterResult,
    FilterReason,
    WalletData
)

from .sybil_detector import (
    SybilDetector,
    SybilCluster,
    SybilDetectionResult,
    FundingConnection,
    TradingPatternSimilarity
)

from .performance_validator import (
    PerformanceValidator,
    ConsistencyResult,
    TimeWindowMetrics,
    MarketCondition
)

from .activity_analyzer import (
    ActivityAnalyzer,
    ActivityMetrics,
    TradingPattern,
    NarrativeCategory
)

from .manual_review import (
    ManualReviewSystem,
    ReviewCase,
    ReviewStatus,
    ReviewPriority,
    ReviewReason
)

from .cohort_optimizer import (
    CohortOptimizer,
    CohortResult,
    WalletScore,
    QualityWeights,
    CohortTargets,
    OptimizationStrategy
)

from .audit_trail import (
    AuditTrailManager,
    AuditEvent,
    FilteringRun,
    EventType,
    DecisionType
)

from .quality_reporter import (
    QualityReporter,
    QualityMetrics,
    ValidationResult,
    ExportPackage
)

__all__ = [
    # Core performance and analysis
    'WalletPerformanceCalculator',
    'Trade',
    'TradeReturn',
    'PerformanceMetrics',
    'validate_performance_metrics',
    'AdvancedRiskAnalyzer',
    'RiskMetrics',
    'DrawdownPeriod',
    'GasEfficiencyAnalyzer',
    'TransactionGasData',
    'TransactionType',
    'GasMetrics',
    'MEVImpactAnalysis',

    # Filtering system
    'WalletFilter',
    'BatchWalletFilter',
    'FilterCriteria',
    'FilterResult',
    'FilterReason',
    'WalletData',

    # Sybil detection
    'SybilDetector',
    'SybilCluster',
    'SybilDetectionResult',
    'FundingConnection',
    'TradingPatternSimilarity',

    # Performance validation
    'PerformanceValidator',
    'ConsistencyResult',
    'TimeWindowMetrics',
    'MarketCondition',

    # Activity analysis
    'ActivityAnalyzer',
    'ActivityMetrics',
    'TradingPattern',
    'NarrativeCategory',

    # Manual review
    'ManualReviewSystem',
    'ReviewCase',
    'ReviewStatus',
    'ReviewPriority',
    'ReviewReason',

    # Cohort optimization
    'CohortOptimizer',
    'CohortResult',
    'WalletScore',
    'QualityWeights',
    'CohortTargets',
    'OptimizationStrategy',

    # Audit trail
    'AuditTrailManager',
    'AuditEvent',
    'FilteringRun',
    'EventType',
    'DecisionType',

    # Quality reporting
    'QualityReporter',
    'QualityMetrics',
    'ValidationResult',
    'ExportPackage'
]